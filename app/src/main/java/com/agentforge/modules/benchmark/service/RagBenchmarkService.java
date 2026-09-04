package com.agentforge.modules.benchmark.service;

import com.agentforge.common.ai.LlmProviderRegistry;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.EmbedResponse;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.GenerateResponse;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.RewriteResponse;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@ConditionalOnProperty(prefix = "app.benchmark", name = "enabled", havingValue = "true")
public class RagBenchmarkService {

  private final LlmProviderRegistry providerRegistry;
  private final PromptTemplate rewritePromptTemplate;

  public RagBenchmarkService(
      LlmProviderRegistry providerRegistry,
      ResourceLoader resourceLoader
  ) throws IOException {
    this.providerRegistry = providerRegistry;
    this.rewritePromptTemplate = new PromptTemplate(
        resourceLoader.getResource("classpath:prompts/knowledgebase-query-rewrite.st")
            .getContentAsString(StandardCharsets.UTF_8)
    );
  }

  public EmbedResponse embed(List<String> texts) {
    long started = System.nanoTime();
    List<float[]> vectors = providerRegistry.getDefaultEmbeddingModel().embed(texts);
    long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started);
    int dimensions = vectors.isEmpty() ? 0 : vectors.getFirst().length;
    return new EmbedResponse(vectors, dimensions, elapsedMs);
  }

  public RewriteResponse rewrite(String question, String history) {
    Map<String, Object> variables = new HashMap<>();
    variables.put("question", question);
    variables.put("history", history == null ? "" : history);
    String prompt = rewritePromptTemplate.render(variables);

    ChatClient chatClient = providerRegistry.getPlainChatClient();
    long started = System.nanoTime();
    String content = chatClient.prompt().user(prompt).call().content();
    long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started);
    String rewritten = content == null || content.isBlank()
        ? question
        : content.replace('\n', ' ').replace('\r', ' ').trim();
    return new RewriteResponse(rewritten, elapsedMs);
  }

  public GenerateResponse generate(String system, String user) {
    var promptSpec = providerRegistry.getPlainChatClient().prompt();
    if (system != null && !system.isBlank()) {
      promptSpec = promptSpec.system(system);
    }

    long started = System.nanoTime();
    String content = promptSpec.user(user).call().content();
    long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - started);
    return new GenerateResponse(content == null ? "" : content, elapsedMs);
  }
}
