package com.agentforge.modules.benchmark;

import com.agentforge.common.annotation.RateLimit;
import com.agentforge.common.exception.BusinessException;
import com.agentforge.common.exception.ErrorCode;
import com.agentforge.common.result.Result;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.EmbedRequest;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.EmbedResponse;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.GenerateRequest;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.GenerateResponse;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.RewriteRequest;
import com.agentforge.modules.benchmark.model.RagBenchmarkDTO.RewriteResponse;
import com.agentforge.modules.benchmark.service.RagBenchmarkService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

@RestController
@RequestMapping("/api/benchmark")
@ConditionalOnProperty(prefix = "app.benchmark", name = "enabled", havingValue = "true")
public class RagBenchmarkController {

  private final RagBenchmarkService benchmarkService;
  private final String benchmarkToken;

  public RagBenchmarkController(
      RagBenchmarkService benchmarkService,
      @Value("${app.benchmark.token:}") String benchmarkToken
  ) {
    this.benchmarkService = benchmarkService;
    this.benchmarkToken = benchmarkToken;
  }

  @PostMapping("/embeddings")
  @RateLimit(
      dimension = RateLimit.Dimension.GLOBAL,
      count = 30,
      timeUnit = RateLimit.TimeUnit.MINUTES
  )
  public Result<EmbedResponse> embed(
      @RequestHeader(name = "X-Benchmark-Token", required = false) String token,
      @Valid @RequestBody EmbedRequest request
  ) {
    authorize(token);
    return Result.success(benchmarkService.embed(request.texts()));
  }

  @PostMapping("/rewrite")
  @RateLimit(
      dimension = RateLimit.Dimension.GLOBAL,
      count = 30,
      timeUnit = RateLimit.TimeUnit.MINUTES
  )
  public Result<RewriteResponse> rewrite(
      @RequestHeader(name = "X-Benchmark-Token", required = false) String token,
      @Valid @RequestBody RewriteRequest request
  ) {
    authorize(token);
    return Result.success(benchmarkService.rewrite(request.question(), request.history()));
  }

  @PostMapping("/generate")
  @RateLimit(
      dimension = RateLimit.Dimension.GLOBAL,
      count = 120,
      timeUnit = RateLimit.TimeUnit.MINUTES
  )
  public Result<GenerateResponse> generate(
      @RequestHeader(name = "X-Benchmark-Token", required = false) String token,
      @Valid @RequestBody GenerateRequest request
  ) {
    authorize(token);
    return Result.success(benchmarkService.generate(request.system(), request.user()));
  }

  private void authorize(String token) {
    if (benchmarkToken.isBlank() || token == null) {
      throw new BusinessException(ErrorCode.FORBIDDEN, "Benchmark token is not configured");
    }
    byte[] expected = benchmarkToken.getBytes(StandardCharsets.UTF_8);
    byte[] actual = token.getBytes(StandardCharsets.UTF_8);
    if (!MessageDigest.isEqual(expected, actual)) {
      throw new BusinessException(ErrorCode.FORBIDDEN, "Invalid benchmark token");
    }
  }
}
