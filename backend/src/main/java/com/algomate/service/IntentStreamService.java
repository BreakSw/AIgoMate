package com.algomate.service;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;

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

    public IntentStreamService(ChatService chatService,
                               AgentClient agentClient,
                               @Qualifier("intentTaskExecutor") Executor executor) {
        this.chatService = chatService;
        this.agentClient = agentClient;
        this.executor = executor;
    }

    public SseEmitter stream(Long userId, Long sessionId, String content) {
        SseEmitter emitter = new SseEmitter(STREAM_TIMEOUT_MS);
        executor.execute(() -> execute(emitter, userId, sessionId, content));
        return emitter;
    }

    private void execute(SseEmitter emitter, Long userId, Long sessionId, String content) {
        try {
            send(emitter, "status", Map.of("phase", "saving", "message", "正在保存用户输入"));
            PreparedIntent prepared = chatService.prepareIntent(userId, sessionId, content);

            send(emitter, "status", Map.of(
                    "phase", "analyzing",
                    "message", "deepseek-v4-pro 正在改写输入并识别用户意图"));
            CompletableFuture<AgentResponse> resultFuture = agentClient.analyzeIntentAsync(
                    prepared.sessionId(),
                    prepared.prompt(),
                    prepared.history(),
                    prepared.previousContextSnapshot());
            int lastReportedRetry = 0;
            while (!resultFuture.isDone()) {
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

            send(emitter, "intent", result);
            var conversation = chatService.completeIntent(
                    userId, sessionId, prepared.userMessageId(), result);
            send(emitter, "complete", conversation);
            emitter.complete();
        } catch (Exception exception) {
            try {
                send(emitter, "failed", Map.of(
                        "message", "意图识别失败",
                        "detail", safeMessage(exception)));
                emitter.complete();
            } catch (Exception sendFailure) {
                emitter.completeWithError(sendFailure);
            }
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

    private String safeMessage(Exception exception) {
        String message = exception.getMessage();
        return message == null || message.isBlank() ? "未知错误" : message;
    }
}
