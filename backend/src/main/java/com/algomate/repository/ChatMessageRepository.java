package com.algomate.repository;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

import com.algomate.domain.ChatMessage;

public interface ChatMessageRepository extends JpaRepository<ChatMessage, Long> {
    List<ChatMessage> findAllBySessionIdOrderByCreatedAtAsc(Long sessionId);
    long countBySessionId(Long sessionId);
    void deleteAllBySessionId(Long sessionId);
}

