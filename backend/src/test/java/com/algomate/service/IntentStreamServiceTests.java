package com.algomate.service;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.Test;

import com.algomate.service.AgentClient.AgentResponse;
import com.algomate.service.AgentClient.RetryStatus;
import com.algomate.service.ChatService.PreparedIntent;

class IntentStreamServiceTests {

    @Test
    void cancellationBeforeWorkerStartsStillPersistsUserMessage() {
        ChatService chatService = mock(ChatService.class);
        AgentClient agentClient = mock(AgentClient.class);
        AtomicReference<Runnable> queuedTask = new AtomicReference<>();
        IntentStreamService service = new IntentStreamService(chatService, agentClient, queuedTask::set);

        service.stream(1L, 7L, "学习二分查找");
        service.cancel(1L, 7L);
        queuedTask.get().run();

        verify(chatService).prepareIntent(1L, 7L, "学习二分查找");
        verify(agentClient, never()).analyzeIntentAsync(any(), any(), any(), any(), any());
    }

    @Test
    void cancellationStopsActiveAgentFutureWithoutSavingAssistantReply() throws Exception {
        ChatService chatService = mock(ChatService.class);
        AgentClient agentClient = mock(AgentClient.class);
        ExecutorService executor = Executors.newSingleThreadExecutor();
        CompletableFuture<AgentResponse> agentFuture = new CompletableFuture<>();
        CountDownLatch agentStarted = new CountDownLatch(1);
        PreparedIntent prepared = new PreparedIntent(7L, 11L, "学习二分查找", List.of(), null);

        when(chatService.prepareIntent(1L, 7L, "学习二分查找")).thenReturn(prepared);
        when(agentClient.analyzeIntentAsync(
                eq(1L), eq(7L), eq("学习二分查找"), any(), any()))
                .thenAnswer(ignored -> {
                    agentStarted.countDown();
                    return agentFuture;
                });
        when(agentClient.getRetryStatus(7L)).thenReturn(new RetryStatus("idle", 0, 5, null));

        try {
            IntentStreamService service = new IntentStreamService(chatService, agentClient, executor);
            service.stream(1L, 7L, "学习二分查找");
            assertThat(agentStarted.await(1, TimeUnit.SECONDS)).isTrue();

            service.cancel(1L, 7L);
            service.cancel(1L, 7L);

            assertThat(agentFuture.isCancelled()).isTrue();
        } finally {
            executor.shutdown();
            assertThat(executor.awaitTermination(2, TimeUnit.SECONDS)).isTrue();
        }
        verify(chatService, never()).completeIntent(any(), any(), any(), any());
    }
}
