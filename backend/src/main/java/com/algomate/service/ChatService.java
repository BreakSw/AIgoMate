package com.algomate.service;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.web.server.ResponseStatusException;

import com.algomate.api.ChatDtos.ConversationResponse;
import com.algomate.api.ChatDtos.MessageResponse;
import com.algomate.api.ChatDtos.SessionResponse;
import com.algomate.domain.AppUser;
import com.algomate.domain.ChatMessage;
import com.algomate.domain.ChatSession;
import com.algomate.domain.MessageRole;
import com.algomate.domain.IntentAnalysis;
import com.algomate.repository.AppUserRepository;
import com.algomate.repository.ChatMessageRepository;
import com.algomate.repository.ChatSessionRepository;
import com.algomate.repository.IntentAnalysisRepository;
import com.algomate.service.AgentClient.AgentResponse;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class ChatService {
    private final AppUserRepository userRepository;
    private final ChatSessionRepository sessionRepository;
    private final ChatMessageRepository messageRepository;
    private final IntentAnalysisRepository intentAnalysisRepository;
    private final AgentClient agentClient;
    private final ObjectMapper objectMapper;
    private final TransactionTemplate transactionTemplate;

    public ChatService(AppUserRepository userRepository,
                       ChatSessionRepository sessionRepository,
                       ChatMessageRepository messageRepository,
                       IntentAnalysisRepository intentAnalysisRepository,
                       AgentClient agentClient,
                       ObjectMapper objectMapper,
                       TransactionTemplate transactionTemplate) {
        this.userRepository = userRepository;
        this.sessionRepository = sessionRepository;
        this.messageRepository = messageRepository;
        this.intentAnalysisRepository = intentAnalysisRepository;
        this.agentClient = agentClient;
        this.objectMapper = objectMapper;
        this.transactionTemplate = transactionTemplate;
    }

    @Transactional(readOnly = true)
    public List<SessionResponse> listSessions(Long userId) {
        requireUser(userId);
        return sessionRepository.findAllByUserIdOrderByUpdatedAtDesc(userId).stream()
                .map(this::toSessionResponse)
                .toList();
    }

    @Transactional
    public SessionResponse createSession(Long userId, String requestedTitle) {
        AppUser user = requireUser(userId);
        String title = requestedTitle == null || requestedTitle.isBlank() ? "新的算法探索" : requestedTitle.trim();
        return toSessionResponse(sessionRepository.save(new ChatSession(user, title)));
    }

    @Transactional(readOnly = true)
    public ConversationResponse getConversation(Long userId, Long sessionId) {
        ChatSession session = requireSession(userId, sessionId);
        List<MessageResponse> messages = messageRepository.findAllBySessionIdOrderByCreatedAtAsc(sessionId)
                .stream().map(this::toMessageResponse).toList();
        return new ConversationResponse(toSessionResponse(session), messages);
    }

    /**
     * 非流式发送。历史上该方法整体处于一个事务中，而 agent 调用最长可达数分钟，
     * 会长期占用唯一的数据库连接，阻塞其他会话的一切读写。现在拆分为三段：
     * 短事务保存用户消息 -> 无事务调用 agent -> 短事务保存回复。
     */
    public ConversationResponse sendMessage(Long userId, Long sessionId, String content) {
        PreparedIntent prepared = transactionTemplate.execute(
                status -> prepareIntent(userId, sessionId, content));
        String answer = agentClient.respond(userId, sessionId, prepared.prompt(), prepared.history());
        return transactionTemplate.execute(
                status -> completeSimpleAnswer(userId, sessionId, prepared.userMessageId(), answer));
    }

    private ConversationResponse completeSimpleAnswer(Long userId,
                                                      Long sessionId,
                                                      Long userMessageId,
                                                      String answer) {
        ChatSession session = requireSession(userId, sessionId);
        messageRepository.findById(userMessageId)
                .filter(message -> message.getSession().getId().equals(sessionId)
                        && message.getRole() == MessageRole.USER)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "用户消息不存在"));
        messageRepository.save(new ChatMessage(session, MessageRole.ASSISTANT, answer));
        session.touch();
        sessionRepository.save(session);
        return getConversation(userId, sessionId);
    }

    @Transactional
    public PreparedIntent prepareIntent(Long userId, Long sessionId, String content) {
        ChatSession session = requireSession(userId, sessionId);
        String prompt = content.trim();
        long previousCount = messageRepository.countBySessionId(sessionId);
        if (previousCount == 0) {
            session.rename(titleFrom(prompt));
        }
        List<ChatMessage> history = messageRepository.findAllBySessionIdOrderByCreatedAtAsc(sessionId);
        ChatMessage userMessage = messageRepository.save(new ChatMessage(session, MessageRole.USER, prompt));
        session.touch();
        sessionRepository.save(session);
        return new PreparedIntent(
                sessionId,
                userMessage.getId(),
                prompt,
                history,
                latestContextSnapshot(sessionId));
    }

    @Transactional
    public ConversationResponse completeIntent(Long userId,
                                               Long sessionId,
                                               Long userMessageId,
                                               AgentResponse result) {
        ChatSession session = requireSession(userId, sessionId);
        ChatMessage userMessage = messageRepository.findById(userMessageId)
                .filter(message -> message.getSession().getId().equals(sessionId)
                        && message.getRole() == MessageRole.USER)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "用户消息不存在"));
        try {
            String taskSpecJson = objectMapper.writeValueAsString(result.taskSpec());
            String contextSnapshotJson = objectMapper.writeValueAsString(result.contextSnapshot());
            ChatMessage assistantMessage = messageRepository.save(
                    new ChatMessage(session, MessageRole.ASSISTANT, result.content()));
            intentAnalysisRepository.save(new IntentAnalysis(
                    session,
                    userMessage,
                    assistantMessage,
                    result.taskSpec().path("schema_version").asText("1.0"),
                    result.intent(),
                    taskSpecJson,
                    contextSnapshotJson,
                    result.model(),
                    result.provider()));
        } catch (JsonProcessingException exception) {
            throw new IllegalStateException("TaskSpec 无法序列化", exception);
        }
        session.touch();
        sessionRepository.save(session);
        return getConversation(userId, sessionId);
    }

    @Transactional
    public void deleteSession(Long userId, Long sessionId) {
        ChatSession session = requireSession(userId, sessionId);
        intentAnalysisRepository.deleteAllBySessionId(sessionId);
        messageRepository.deleteAllBySessionId(sessionId);
        sessionRepository.delete(session);
    }

    private AppUser requireUser(Long userId) {
        return userRepository.findById(userId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "用户不存在"));
    }

    private ChatSession requireSession(Long userId, Long sessionId) {
        return sessionRepository.findByIdAndUserId(sessionId, userId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "会话不存在"));
    }

    private String titleFrom(String content) {
        String singleLine = content.replaceAll("\\s+", " ");
        return singleLine.length() <= 24 ? singleLine : singleLine.substring(0, 24) + "…";
    }

    private SessionResponse toSessionResponse(ChatSession session) {
        return new SessionResponse(session.getId(), session.getTitle(), session.getSummary(),
                messageRepository.countBySessionId(session.getId()), session.getCreatedAt(), session.getUpdatedAt());
    }

    private MessageResponse toMessageResponse(ChatMessage message) {
        return new MessageResponse(
                message.getId(),
                message.getRole(),
                message.getContent(),
                message.getCreatedAt(),
                contextSnapshotFor(message));
    }

    private com.fasterxml.jackson.databind.JsonNode contextSnapshotFor(ChatMessage message) {
        if (message.getRole() != MessageRole.ASSISTANT) {
            return null;
        }
        return intentAnalysisRepository.findByAssistantMessageId(message.getId())
                .map(IntentAnalysis::getContextSnapshotJson)
                .filter(value -> value != null && !value.isBlank())
                .map(value -> {
                    try {
                        return objectMapper.readTree(value);
                    } catch (JsonProcessingException exception) {
                        throw new IllegalStateException("ContextSnapshot 无法反序列化", exception);
                    }
                })
                .orElse(null);
    }

    private com.fasterxml.jackson.databind.JsonNode latestContextSnapshot(Long sessionId) {
        return intentAnalysisRepository.findFirstBySessionIdOrderByCreatedAtDesc(sessionId)
                .map(IntentAnalysis::getContextSnapshotJson)
                .filter(value -> value != null && !value.isBlank())
                .map(value -> {
                    try {
                        return objectMapper.readTree(value);
                    } catch (JsonProcessingException exception) {
                        throw new IllegalStateException("Previous ContextSnapshot 无法反序列化", exception);
                    }
                })
                .orElse(null);
    }

    public record PreparedIntent(
            Long sessionId,
            Long userMessageId,
            String prompt,
            List<ChatMessage> history,
            com.fasterxml.jackson.databind.JsonNode previousContextSnapshot) {}
}
