package com.algomate.api;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

import com.algomate.service.AgentClient;
import com.algomate.service.AgentClient.AgentServiceException;
import com.fasterxml.jackson.databind.JsonNode;

import jakarta.validation.Valid;
import jakarta.validation.constraints.Max;
import jakarta.validation.constraints.Min;
import jakarta.validation.constraints.Size;

@RestController
@RequestMapping("/api/model-config")
public class ModelConfigController {
    private final AgentClient agentClient;

    public ModelConfigController(AgentClient agentClient) {
        this.agentClient = agentClient;
    }

    @GetMapping
    public JsonNode get() {
        try {
            return agentClient.getModelConfig();
        } catch (AgentServiceException exception) {
            throw unavailable(exception);
        }
    }

    @PutMapping
    public JsonNode save(
            @Valid @RequestBody SaveModelConfigRequest request) {
        try {
            return agentClient.saveModelConfig(request);
        } catch (AgentServiceException exception) {
            throw unavailable(exception);
        }
    }

    @DeleteMapping
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void delete() {
        try {
            agentClient.deleteModelConfig();
        } catch (AgentServiceException exception) {
            throw unavailable(exception);
        }
    }

    @DeleteMapping("/model")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteModel() {
        try {
            agentClient.deleteModelConfigSection("model");
        } catch (AgentServiceException exception) {
            throw unavailable(exception);
        }
    }

    @DeleteMapping("/search")
    @ResponseStatus(HttpStatus.NO_CONTENT)
    public void deleteSearch() {
        try {
            agentClient.deleteModelConfigSection("search");
        } catch (AgentServiceException exception) {
            throw unavailable(exception);
        }
    }

    private ResponseStatusException unavailable(AgentServiceException exception) {
        return new ResponseStatusException(
                HttpStatus.BAD_GATEWAY,
                exception.getMessage(),
                exception);
    }

    public record SaveModelConfigRequest(
            @Size(min = 8, max = 512) String apiKey,
            @Size(min = 8, max = 512) String serpapiApiKey,
            @Size(max = 200) String model,
            @Size(max = 500) String baseUrl,
            @Min(300) @Max(31_536_000) int ttlSeconds,
            Boolean updateModel,
            Boolean updateSearch) {}
}
