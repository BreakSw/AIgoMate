package com.algomate.api;

import java.util.List;
import java.util.Map;

import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import com.algomate.api.ChatDtos.ConversationResponse;
import com.algomate.api.ChatDtos.CreateSessionRequest;
import com.algomate.api.ChatDtos.SendMessageRequest;
import com.algomate.api.ChatDtos.SessionResponse;
import com.algomate.service.ChatService;
import com.algomate.service.IntentStreamService;

import jakarta.validation.Valid;

@RestController
@RequestMapping("/api")
public class ChatController {
    private final ChatService chatService;
    private final IntentStreamService intentStreamService;

    public ChatController(ChatService chatService, IntentStreamService intentStreamService) {
        this.chatService = chatService;
        this.intentStreamService = intentStreamService;
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "service", "algomate-backend");
    }

    @GetMapping("/sessions")
    public List<SessionResponse> listSessions(@RequestParam(defaultValue = "1") Long userId) {
        return chatService.listSessions(userId);
    }

    @PostMapping("/sessions")
    @ResponseStatus(HttpStatus.CREATED)
    public SessionResponse createSession(@Valid @RequestBody CreateSessionRequest request) {
        return chatService.createSession(request.userId(), request.title());
    }

    @GetMapping("/sessions/{sessionId}/messages")
    public ConversationResponse getMessages(@PathVariable Long sessionId,
                                            @RequestParam(defaultValue = "1") Long userId) {
        return chatService.getConversation(userId, sessionId);
    }

    @PostMapping("/sessions/{sessionId}/messages")
    public ConversationResponse sendMessage(@PathVariable Long sessionId,
                                            @Valid @RequestBody SendMessageRequest request) {
        return chatService.sendMessage(request.userId(), sessionId, request.content());
    }

    @PostMapping(
            value = "/sessions/{sessionId}/messages/stream",
            produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamIntent(@PathVariable Long sessionId,
                                   @Valid @RequestBody SendMessageRequest request) {
        return intentStreamService.stream(request.userId(), sessionId, request.content());
    }

    @PostMapping("/sessions/{sessionId}/messages/stream/cancel")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void cancelStream(@PathVariable Long sessionId,
                             @RequestParam(defaultValue = "1") Long userId) {
        intentStreamService.cancel(userId, sessionId);
    }

    @DeleteMapping("/sessions/{sessionId}")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteSession(@PathVariable Long sessionId,
                              @RequestParam(defaultValue = "1") Long userId) {
        chatService.deleteSession(userId, sessionId);
    }
}
