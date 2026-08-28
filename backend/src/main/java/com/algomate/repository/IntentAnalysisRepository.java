package com.algomate.repository;

import java.util.Optional;

import org.springframework.data.jpa.repository.JpaRepository;

import com.algomate.domain.IntentAnalysis;

public interface IntentAnalysisRepository extends JpaRepository<IntentAnalysis, Long> {
    void deleteAllBySessionId(Long sessionId);
    Optional<IntentAnalysis> findByAssistantMessageId(Long assistantMessageId);
    Optional<IntentAnalysis> findFirstBySessionIdOrderByCreatedAtDesc(Long sessionId);
}
