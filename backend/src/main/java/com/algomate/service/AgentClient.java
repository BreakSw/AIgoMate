package com.algomate.service;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.algomate.domain.ChatMessage;

@Component
public class AgentClient {
    private static final Logger log = LoggerFactory.getLogger(AgentClient.class);
    private final HttpClient httpClient;
    private final ObjectMapper objectMapper;
    private final URI analyzeIntentUri;
    private final String retryStatusBaseUrl;
    private final String progressStatusBaseUrl;
    private final String ragOverviewBaseUrl;
    private final URI modelConfigUri;
    private final String sessionMemoryBaseUrl;

    public AgentClient(@Value("${agent.service.base-url}") String baseUrl, ObjectMapper objectMapper) {
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(3))
                .build();
        this.objectMapper = objectMapper;
        this.analyzeIntentUri = URI.create(baseUrl + "/api/agent/analyze-intent");
        this.retryStatusBaseUrl = baseUrl + "/api/agent/sessions/";
        this.progressStatusBaseUrl = baseUrl + "/api/agent/sessions/";
        this.ragOverviewBaseUrl = baseUrl + "/api/rag/overview";
        this.modelConfigUri = URI.create(baseUrl + "/api/model-config");
        this.sessionMemoryBaseUrl = baseUrl + "/api/agent/users/";
    }

    public String respond(Long userId,
                          Long sessionId,
                          String prompt,
                          List<ChatMessage> history) {
        try {
            return analyzeIntent(
                    userId, sessionId, prompt, history).content();
        } catch (AgentServiceException exception) {
            log.warn("Agent service is unavailable: {}", exception.getMessage());
            return "意图识别服务暂时不可用，但你的消息已经安全保存。请检查模型配置后重试。";
        }
    }

    public AgentResponse analyzeIntent(Long userId,
                                       Long sessionId,
                                       String prompt,
                                       List<ChatMessage> history) {
        try {
            return analyzeIntentAsync(
                    userId, sessionId, prompt, history, null).join();
        } catch (CompletionException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof AgentServiceException agentServiceException) {
                throw agentServiceException;
            }
            throw new AgentServiceException("Agent request failed", cause);
        }
    }

    public CompletableFuture<AgentResponse> analyzeIntentAsync(
            Long userId,
            Long sessionId,
            String prompt,
            List<ChatMessage> history,
            JsonNode previousContextSnapshot) {
        var context = history.stream()
                .map(message -> new AgentMessage(message.getRole().name().toLowerCase(), message.getContent()))
                .toList();

        try {
            String requestJson = objectMapper.writeValueAsString(
                    new AgentRequest(
                            userId,
                            sessionId,
                            prompt,
                            context,
                            previousContextSnapshot));
            HttpRequest request = HttpRequest.newBuilder(analyzeIntentUri)
                    .timeout(Duration.ofMinutes(7))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestJson, StandardCharsets.UTF_8))
                    .build();
            CompletableFuture<HttpResponse<String>> requestFuture = httpClient.sendAsync(
                    request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            CompletableFuture<AgentResponse> resultFuture = new CompletableFuture<>();
            requestFuture.whenComplete((response, failure) -> {
                if (resultFuture.isCancelled()) {
                    return;
                }
                if (failure != null) {
                    resultFuture.completeExceptionally(new AgentServiceException(
                            "Agent service connection failed", failure));
                    return;
                }
                try {
                    resultFuture.complete(decodeAgentResponse(response));
                } catch (RuntimeException exception) {
                    resultFuture.completeExceptionally(exception);
                }
            });
            resultFuture.whenComplete((ignored, failure) -> {
                if (resultFuture.isCancelled()) {
                    requestFuture.cancel(true);
                }
            });
            return resultFuture;
        } catch (JsonProcessingException exception) {
            return CompletableFuture.failedFuture(
                    new AgentServiceException("Agent payload could not be encoded", exception));
        }
    }

    public RetryStatus getRetryStatus(Long sessionId) {
        HttpRequest request = HttpRequest.newBuilder(
                        URI.create(retryStatusBaseUrl + sessionId + "/retry-status"))
                .timeout(Duration.ofSeconds(2))
                .GET()
                .build();
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new AgentServiceException("Retry status endpoint returned HTTP " + response.statusCode());
            }
            return objectMapper.readValue(response.body(), RetryStatus.class);
        } catch (JsonProcessingException exception) {
            throw new AgentServiceException("Retry status payload could not be decoded", exception);
        } catch (IOException exception) {
            throw new AgentServiceException("Retry status endpoint is unavailable", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AgentServiceException("Retry status request was interrupted", exception);
        }
    }

    public ProgressStatus getProgressStatus(Long sessionId) {
        HttpRequest request = HttpRequest.newBuilder(
                        URI.create(progressStatusBaseUrl + sessionId + "/progress-status"))
                .timeout(Duration.ofSeconds(2))
                .GET()
                .build();
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new AgentServiceException("Progress status endpoint returned HTTP " + response.statusCode());
            }
            return objectMapper.readValue(response.body(), ProgressStatus.class);
        } catch (JsonProcessingException exception) {
            throw new AgentServiceException("Progress status payload could not be decoded", exception);
        } catch (IOException exception) {
            throw new AgentServiceException("Progress status endpoint is unavailable", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AgentServiceException("Progress status request was interrupted", exception);
        }
    }

    public JsonNode getRagOverview(Long userId) {
        HttpRequest request = HttpRequest.newBuilder(
                        URI.create(ragOverviewBaseUrl + "?user_id=" + userId))
                .timeout(Duration.ofSeconds(8))
                .GET()
                .build();
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new AgentServiceException("RAG overview endpoint returned HTTP " + response.statusCode());
            }
            return objectMapper.readTree(response.body());
        } catch (JsonProcessingException exception) {
            throw new AgentServiceException("RAG overview payload could not be decoded", exception);
        } catch (IOException exception) {
            throw new AgentServiceException("RAG overview endpoint is unavailable", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AgentServiceException("RAG overview request was interrupted", exception);
        }
    }

    public JsonNode getModelConfig() {
        return sendModelConfigRequest(
                HttpRequest.newBuilder(modelConfigUri)
                        .timeout(Duration.ofSeconds(5))
                        .GET()
                        .build());
    }

    public JsonNode saveModelConfig(Object payload) {
        try {
            String requestJson = objectMapper.writeValueAsString(payload);
            return sendModelConfigRequest(
                    HttpRequest.newBuilder(modelConfigUri)
                            .timeout(Duration.ofSeconds(5))
                            .header("Content-Type", "application/json")
                            .PUT(HttpRequest.BodyPublishers.ofString(
                                    requestJson, StandardCharsets.UTF_8))
                            .build());
        } catch (JsonProcessingException exception) {
            throw new AgentServiceException(
                    "模型配置无法序列化", exception);
        }
    }

    public void deleteModelConfig() {
        deleteModelConfigAt(modelConfigUri);
    }

    public void deleteModelConfigSection(String section) {
        deleteModelConfigAt(URI.create(modelConfigUri + "/" + section));
    }

    public void clearSessionMemory(Long userId, Long sessionId) {
        URI uri = URI.create(
                sessionMemoryBaseUrl + userId + "/sessions/" + sessionId + "/memory");
        HttpRequest request = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(5))
                .DELETE()
                .build();
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() != 204) {
                throw new AgentServiceException(extractErrorDetail(response));
            }
        } catch (IOException exception) {
            throw new AgentServiceException("会话记忆服务不可用", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AgentServiceException("会话记忆清理请求被中断", exception);
        }
    }

    private void deleteModelConfigAt(URI uri) {
        HttpRequest request = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(5))
                .DELETE()
                .build();
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() != 204) {
                throw new AgentServiceException(extractErrorDetail(response));
            }
        } catch (IOException exception) {
            throw new AgentServiceException("Redis 模型配置服务不可用", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AgentServiceException("模型配置删除请求被中断", exception);
        }
    }

    private JsonNode sendModelConfigRequest(HttpRequest request) {
        try {
            HttpResponse<String> response = httpClient.send(
                    request,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new AgentServiceException(extractErrorDetail(response));
            }
            return objectMapper.readTree(response.body());
        } catch (JsonProcessingException exception) {
            throw new AgentServiceException("模型配置响应无法解析", exception);
        } catch (IOException exception) {
            throw new AgentServiceException("Redis 模型配置服务不可用", exception);
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            throw new AgentServiceException("模型配置请求被中断", exception);
        }
    }

    private AgentResponse decodeAgentResponse(HttpResponse<String> httpResponse) {
        if (httpResponse.statusCode() < 200 || httpResponse.statusCode() >= 300) {
            throw new AgentServiceException(extractErrorDetail(httpResponse));
        }
        try {
            AgentResponse response = objectMapper.readValue(httpResponse.body(), AgentResponse.class);
            if (response != null
                    && response.content() != null
                    && !response.content().isBlank()
                    && response.taskSpec() != null) {
                return response;
            }
        } catch (JsonProcessingException exception) {
            throw new AgentServiceException("Agent response could not be decoded", exception);
        }
        throw new AgentServiceException("Agent service returned an empty response");
    }

    private String extractErrorDetail(HttpResponse<String> response) {
        try {
            JsonNode body = objectMapper.readTree(response.body());
            String detail = body.path("detail").asText();
            if (!detail.isBlank()) {
                return detail;
            }
        } catch (JsonProcessingException ignored) {
            // Fall back to the HTTP status without exposing an arbitrary upstream body.
        }
        return "Agent service returned HTTP " + response.statusCode();
    }

    public record AgentRequest(
            Long userId,
            Long sessionId,
            String message,
            List<AgentMessage> history,
            JsonNode previousContextSnapshot) {}
    public record AgentMessage(String role, String content) {}
    public record RetryStatus(
            String phase,
            @JsonProperty("retry_count") int retryCount,
            @JsonProperty("max_retries") int maxRetries,
            @JsonProperty("retry_delay_seconds") Double retryDelaySeconds) {}
    public record ProgressStatus(
            long generation,
            long sequence,
            String phase,
            String agent,
            String message,
            String detail,
            String state,
            @JsonProperty("updated_at") String updatedAt) {}
    public record AgentResponse(
            String content,
            String intent,
            @JsonProperty("context_messages_used") int contextMessagesUsed,
            @JsonProperty("task_spec") JsonNode taskSpec,
            @JsonProperty("context_snapshot") JsonNode contextSnapshot,
            String model,
            String provider) {}

    public static class AgentServiceException extends RuntimeException {
        public AgentServiceException(String message) {
            super(message);
        }

        public AgentServiceException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
