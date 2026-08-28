package com.algomate.api;

import java.time.Instant;
import java.util.List;

import com.algomate.domain.MessageRole;
import com.fasterxml.jackson.databind.JsonNode;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;

public final class ChatDtos {
    private ChatDtos() {}

    public record CreateSessionRequest(@NotNull Long userId, @Size(max = 80) String title) {}

    public record SendMessageRequest(
            @NotNull Long userId,
            @NotBlank @Size(max = 12000) String content) {}

    public record SessionResponse(
            Long id,
            String title,
            String summary,
            long messageCount,
            Instant createdAt,
            Instant updatedAt) {}

    public record MessageResponse(
            Long id,
            MessageRole role,
            String content,
            Instant createdAt,
            JsonNode contextSnapshot) {}

    public record ConversationResponse(
            SessionResponse session,
            List<MessageResponse> messages) {}
}
