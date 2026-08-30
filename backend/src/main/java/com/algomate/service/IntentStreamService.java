package com.algomate.service;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicBoolean;

import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.algomate.service.AgentClient.AgentResponse;
import com.algomate.service.ChatService.PreparedIntent;

@Service
public class IntentStreamService {
    private static final long STREAM_TIMEOUT_MS = 420_000L;
    private static final long RETRY_STATUS_POLL_MS = 300L;

    private final ChatService chatService;
    private final AgentClient agentClient;
    private final Executor executor;
    private final ConcurrentHashMap<StreamKey, ActiveStream> activeStreams = new ConcurrentHashMap<>();

    public IntentStreamService(ChatService chatService,
                               AgentClient agentClient,
                               @Qualifier("intentTaskExecutor") Executor executor) {
        this.chatService = chatService;
        this.agentClient = agentClient;
        this.executor = executor;
    }

    public SseEmitter stream(Long userId, Long sessionId, String content) {
        SseEmitter emitter = new SseEmitter(STREAM_TIMEOUT_MS);
        StreamKey key = new StreamKey(userId, sessionId);
        ActiveStream active = new ActiveStream(emitter);
        ActiveStream previous = activeStreams.put(key, active);
        if (previous != null) {
            previous.cancel(true);
        }
        emitter.onCompletion(() -> cancel(key, active, false));
        emitter.onTimeout(() -> cancel(key, active, false));
        emitter.onError(ignored -> cancel(key, active, false));
        executor.execute(() -> execute(key, active, userId, sessionId, content));
        return emitter;
    }

    public void cancel(Long userId, Long sessionId) {
        StreamKey key = new StreamKey(userId, sessionId);
        ActiveStream active = activeStreams.remove(key);
        if (active != null) {
            active.cancel(true);
        }
    }

    private void cancel(StreamKey key, ActiveStream active, boolean completeEmitter) {
        activeStreams.remove(key, active);
        active.cancel(completeEmitter);
    }

    private void execute(StreamKey key,
                         ActiveStream active,
                         Long userId,
                         Long sessionId,
                         String content) {
        SseEmitter emitter = active.emitter();
        try {
            if (!active.isCancelled()) {
                try {
                    send(emitter, "status", Map.of(
                            "phase", "saving",
                            "agent", "系统",
                            "message", "正在保存用户输入",
                            "detail", "建立本轮任务与会话关联"));
                } catch (IOException disconnected) {
                    // Even if the browser stops immediately, persist the submitted user message first.
                    active.cancel(false);
                }
            }
            PreparedIntent prepared = chatService.prepareIntent(userId, sessionId, content);
            if (active.isCancelled()) {
                return;
            }

            send(emitter, "status", Map.of(
                    "phase", "analyzing",
                    "agent", "首脑智能体",
                    "message", "正在启动多智能体任务",
                    "detail", "将依次处理意图、记忆、RAG、解题与验证"));
            CompletableFuture<AgentResponse> resultFuture = agentClient.analyzeIntentAsync(
                    userId,
                    prepared.sessionId(),
                    prepared.prompt(),
                    prepared.history(),
                    prepared.previousContextSnapshot());
            active.attach(resultFuture);
            int lastReportedRetry = 0;
            boolean currentProgressStarted = false;
            long lastReportedGeneration = -1L;
            long lastReportedProgress = 0L;
            while (!resultFuture.isDone()) {
                if (active.isCancelled()) {
                    resultFuture.cancel(true);
                    return;
                }
                try {
                    var progress = agentClient.getProgressStatus(prepared.sessionId());
                    if (progress != null
                            && (currentProgressStarted || "active".equals(progress.state()))) {
                        currentProgressStarted = true;
                        if (progress.generation() != lastReportedGeneration) {
                            lastReportedGeneration = progress.generation();
                            lastReportedProgress = 0L;
                        }
                        if (progress.sequence() > lastReportedProgress) {
                            lastReportedProgress = progress.sequence();
                            send(emitter, "status", progressPayload(progress));
                        }
                    }
                } catch (AgentClient.AgentServiceException ignored) {
                    // Progress visibility must not interrupt the model request itself.
                }
                try {
                    var retryStatus = agentClient.getRetryStatus(prepared.sessionId());
                    if ("retrying".equals(retryStatus.phase())
                            && retryStatus.retryCount() > lastReportedRetry) {
                        lastReportedRetry = retryStatus.retryCount();
                        send(emitter, "status", Map.of(
                                "phase", "reconnecting",
                                "message", "正在重新连接 " + retryStatus.retryCount()
                                        + "/" + retryStatus.maxRetries(),
                                "retryCount", retryStatus.retryCount(),
                                "maxRetries", retryStatus.maxRetries()));
                    }
                } catch (AgentClient.AgentServiceException ignored) {
                    // Retry-state visibility must not interrupt the model request itself.
                }
                Thread.sleep(RETRY_STATUS_POLL_MS);
            }
            AgentResponse result = awaitResult(resultFuture);
            if (active.isCancelled()) {
                return;
            }

            send(emitter, "intent", result);
            if (active.isCancelled()) {
                return;
            }
            var conversation = chatService.completeIntent(
                    userId, sessionId, prepared.userMessageId(), result);
            send(emitter, "complete", conversation);
            emitter.complete();
        } catch (CancellationException exception) {
            active.cancel(true);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            active.cancel(true);
        } catch (Exception exception) {
            if (active.isCancelled()) {
                return;
            }
            try {
                send(emitter, "failed", Map.of(
                        "message", "智能体回答生成失败",
                        "detail", safeMessage(exception)));
                emitter.complete();
            } catch (Exception sendFailure) {
                emitter.completeWithError(sendFailure);
            }
        } finally {
            activeStreams.remove(key, active);
        }
    }

    private AgentResponse awaitResult(CompletableFuture<AgentResponse> future) {
        try {
            return future.join();
        } catch (CompletionException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof RuntimeException runtimeException) {
                throw runtimeException;
            }
            throw exception;
        }
    }

    private void send(SseEmitter emitter, String event, Object data) throws IOException {
        emitter.send(SseEmitter.event().name(event).data(data));
    }

    private Map<String, Object> progressPayload(AgentClient.ProgressStatus progress) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("phase", progress.phase());
        payload.put("message", progress.message());
        payload.put("sequence", progress.sequence());
        payload.put("state", progress.state());
        if (progress.agent() != null && !progress.agent().isBlank()) {
            payload.put("agent", progress.agent());
        }
        if (progress.detail() != null && !progress.detail().isBlank()) {
            payload.put("detail", progress.detail());
        }
        return payload;
    }

    private String safeMessage(Exception exception) {
        String message = exception.getMessage();
        return message == null || message.isBlank() ? "未知错误" : message;
    }

    private record StreamKey(Long userId, Long sessionId) {}

    private static final class ActiveStream {
        private final SseEmitter emitter;
        private final AtomicBoolean cancelled = new AtomicBoolean(false);
        private volatile CompletableFuture<AgentResponse> resultFuture;

        private ActiveStream(SseEmitter emitter) {
            this.emitter = emitter;
        }

        private SseEmitter emitter() {
            return emitter;
        }

        private boolean isCancelled() {
            return cancelled.get();
        }

        private void attach(CompletableFuture<AgentResponse> future) {
            resultFuture = future;
            if (isCancelled()) {
                future.cancel(true);
            }
        }

        private void cancel(boolean completeEmitter) {
            if (!cancelled.compareAndSet(false, true)) {
                return;
            }
            CompletableFuture<AgentResponse> future = resultFuture;
            if (future != null) {
                future.cancel(true);
            }
            if (completeEmitter) {
                emitter.complete();
            }
        }
    }
}
