package com.agentforge.modules.benchmark.model;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.Size;

import java.util.List;

public final class RagBenchmarkDTO {

  private RagBenchmarkDTO() {
  }

  public record EmbedRequest(
      @NotEmpty @Size(max = 10) List<@NotBlank @Size(max = 4000) String> texts
  ) {
  }

  public record EmbedResponse(
      List<float[]> vectors,
      int dimensions,
      long elapsedMs
  ) {
  }

  public record RewriteRequest(
      @NotBlank @Size(max = 2000) String question,
      @Size(max = 4000) String history
  ) {
  }

  public record RewriteResponse(
      String rewritten,
      long elapsedMs
  ) {
  }

  public record GenerateRequest(
      @Size(max = 10000) String system,
      @NotBlank @Size(max = 60000) String user
  ) {
  }

  public record GenerateResponse(
      String content,
      long elapsedMs
  ) {
  }
}
