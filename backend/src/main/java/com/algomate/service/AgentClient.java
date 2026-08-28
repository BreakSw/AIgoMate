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

    public AgentClient(@Value("${agent.service.base-url}") String baseUrl, ObjectMapper objectMapper) {
        this.httpClient = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .connectTimeout(Duration.ofSeconds(3))
                .build();
        this.objectMapper = objectMapper;
        this.analyzeIntentUri = URI.create(baseUrl + "/api/agent/analyze-intent");
        this.retryStatusBaseUrl = baseUrl + "/api/agent/sessions/";
    }

    public String respond(Long sessionId, String prompt, List<ChatMessage> history) {
        try {
            return analyzeIntent(sessionId, prompt, history).content();
        } catch (AgentServiceException exception) {
            log.warn("Agent service is unavailable: {}", exception.getMessage());
            return "意图识别服务暂时不可用，但你的消息已经安全保存。请检查模型配置后重试。";
        }
    }

    public AgentResponse analyzeIntent(Long sessionId, String prompt, List<ChatMessage> history) {
        try {
            return analyzeIntentAsync(sessionId, prompt, history, null).join();
        } catch (CompletionException exception) {
            Throwable cause = exception.getCause();
            if (cause instanceof AgentServiceException agentServiceException) {
                throw agentServiceException;
            }
            throw new AgentServiceException("Agent request failed", cause);
        }
    }

    public CompletableFuture<AgentResponse> analyzeIntentAsync(
            Long sessionId,
            String prompt,
            List<ChatMessage> history,
            JsonNode previousContextSnapshot) {
        var context = history.stream()
                .map(message -> new AgentMessage(message.getRole().name().toLowerCase(), message.getContent()))
                .toList();

        try {
            String requestJson = objectMapper.writeValueAsString(
                    new AgentRequest(sessionId, prompt, context, previousContextSnapshot));
            HttpRequest request = HttpRequest.newBuilder(analyzeIntentUri)
                    .timeout(Duration.ofMinutes(7))
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestJson, StandardCharsets.UTF_8))
                    .build();
            return httpClient.sendAsync(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8))
                    .handle((response, failure) -> {
                        if (failure != null) {
                            throw new CompletionException(new AgentServiceException(
                                    "Agent service connection failed",
                                    failure));
                        }
                        return decodeAgentResponse(response);
                    });
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
