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
import jakarta.persistence.Table;

@Entity
@Table(name = "chat_session")
public class ChatSession {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "user_id", nullable = false)
    private AppUser user;

    @Column(nullable = false)
    private String title;

    @Column(columnDefinition = "TEXT")
    private String summary;

    @Column(name = "created_at", nullable = false)
    private Instant createdAt;

    @Column(name = "updated_at", nullable = false)
    private Instant updatedAt;

    protected ChatSession() {}

    public ChatSession(AppUser user, String title) {
        this.user = user;
        this.title = title;
        this.createdAt = Instant.now();
        this.updatedAt = this.createdAt;
    }

    public void rename(String title) { this.title = title; }
    public void resetConversation() {
        this.title = "新的算法探索";
        this.summary = null;
        touch();
    }
    public void touch() { this.updatedAt = Instant.now(); }

    public Long getId() { return id; }
    public AppUser getUser() { return user; }
    public String getTitle() { return title; }
    public String getSummary() { return summary; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getUpdatedAt() { return updatedAt; }
}
