package com.algomate.domain;

import java.time.Instant;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToOne;
import jakarta.persistence.Table;

@Entity
@Table(name = "intent_analysis")
public class IntentAnalysis {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "session_id", nullable = false)
    private ChatSession session;

    @OneToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_message_id", nullable = false, unique = true)
    private ChatMessage userMessage;

    @ManyToOne(fetch = FetchType.LAZY)
    // SQLite cannot add a UNIQUE column to an existing table with ALTER TABLE.
    // DatabaseIndexInitializer adds the equivalent unique index after Hibernate
    // has created this nullable migration column.
    @JoinColumn(name = "assistant_message_id")
    private ChatMessage assistantMessage;

    @Column(name = "schema_version", nullable = false, length = 16)
    private String schemaVersion;

    @Column(name = "primary_intent", nullable = false, length = 64)
    private String primaryIntent;

    @Column(name = "task_spec_json", nullable = false, columnDefinition = "TEXT")
    private String taskSpecJson;

    @Column(name = "context_snapshot_json", columnDefinition = "TEXT")
    private String contextSnapshotJson;

    @Column(nullable = false, length = 120)
    private String model;

    @Column(nullable = false, length = 80)
    private String provider;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    protected IntentAnalysis() {}

    public IntentAnalysis(ChatSession session,
                          ChatMessage userMessage,
                          ChatMessage assistantMessage,
                          String schemaVersion,
                          String primaryIntent,
                          String taskSpecJson,
                          String contextSnapshotJson,
                          String model,
                          String provider) {
        this.session = session;
        this.userMessage = userMessage;
        this.assistantMessage = assistantMessage;
        this.schemaVersion = schemaVersion;
        this.primaryIntent = primaryIntent;
        this.taskSpecJson = taskSpecJson;
        this.contextSnapshotJson = contextSnapshotJson;
        this.model = model;
        this.provider = provider;
        this.createdAt = Instant.now();
    }

    public Long getId() { return id; }
    public ChatSession getSession() { return session; }
    public ChatMessage getUserMessage() { return userMessage; }
    public ChatMessage getAssistantMessage() { return assistantMessage; }
    public String getSchemaVersion() { return schemaVersion; }
    public String getPrimaryIntent() { return primaryIntent; }
    public String getTaskSpecJson() { return taskSpecJson; }
    public String getContextSnapshotJson() { return contextSnapshotJson; }
    public String getModel() { return model; }
    public String getProvider() { return provider; }
    public Instant getCreatedAt() { return createdAt; }
}
