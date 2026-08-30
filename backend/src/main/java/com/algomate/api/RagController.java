package com.algomate.api;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.algomate.service.AgentClient;
import com.fasterxml.jackson.databind.JsonNode;

@RestController
@RequestMapping("/api/rag")
public class RagController {
    private final AgentClient agentClient;

    public RagController(AgentClient agentClient) {
        this.agentClient = agentClient;
    }

    @GetMapping("/overview")
    public JsonNode overview(@RequestParam(defaultValue = "1") Long userId) {
        return agentClient.getRagOverview(userId);
    }
}
