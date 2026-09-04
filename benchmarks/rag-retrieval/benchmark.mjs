import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const benchmarkDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(benchmarkDir, '..', '..');
const args = new Set(process.argv.slice(2));
const skipRewrite = args.has('--skip-rewrite');
const backendBase = process.env.BENCHMARK_BACKEND_URL?.replace(/\/$/, '');
const backendToken = process.env.BENCHMARK_TOKEN ?? 'local-only-benchmark-token';
const openAiApiKey = process.env.OPENAI_API_KEY;
const apiKey = process.env.OPENAI_COMPATIBLE_API_KEY
  ?? process.env.AI_BAILIAN_API_KEY
  ?? openAiApiKey;
const apiBase = (process.env.OPENAI_COMPATIBLE_BASE_URL
  ?? process.env.DASHSCOPE_BASE_URL
  ?? (openAiApiKey
    ? 'https://api.openai.com/v1'
    : 'https://dashscope.aliyuncs.com/compatible-mode/v1')).replace(/\/$/, '');
let chatModel = process.env.AI_MODEL ?? (openAiApiKey ? '' : 'qwen3.5-flash');
let embeddingModel = process.env.AI_EMBEDDING_MODEL
  ?? (openAiApiKey ? '' : 'text-embedding-v3');
let dimensions = Number(process.env.APP_AI_EMBEDDING_DIMENSIONS ?? 1024);
const maxBatchSize = 10;
const retrievalPolicy = {
  shortQueryLength: 4,
  mediumQueryLength: 12,
  topKShort: 20,
  topKMedium: 12,
  topKLong: 8,
  minScoreShort: 0.18,
  minScoreDefault: 0.28,
};

if (!backendBase && (!apiKey || !embeddingModel || (!skipRewrite && !chatModel))) {
  throw new Error(
    'Set BENCHMARK_BACKEND_URL, or provide a direct API key plus AI_MODEL and '
      + 'AI_EMBEDDING_MODEL. OPENAI_API_KEY and AI_BAILIAN_API_KEY are supported.',
  );
}

const dataset = JSON.parse(await readFile(path.join(benchmarkDir, 'dataset.json'), 'utf8'));
const rewriteTemplate = await readFile(
  path.join(repoRoot, 'app', 'src', 'main', 'resources', 'prompts', 'knowledgebase-query-rewrite.st'),
  'utf8',
);

async function requestJson(url, body, retries = 3) {
  let lastError;
  for (let attempt = 1; attempt <= retries; attempt += 1) {
    const started = performance.now();
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          ...(backendBase
            ? { 'X-Benchmark-Token': backendToken }
            : { Authorization: `Bearer ${apiKey}` }),
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(60_000),
      });
      const elapsedMs = performance.now() - started;
      const text = await response.text();
      if (!response.ok) {
        const error = new Error(`HTTP ${response.status}: ${text.slice(0, 500)}`);
        error.retryable = response.status === 429 || response.status >= 500;
        throw error;
      }
      return { data: JSON.parse(text), elapsedMs };
    } catch (error) {
      lastError = error;
      if (attempt === retries || error.retryable === false) {
        break;
      }
      await new Promise(resolve => setTimeout(resolve, 600 * (2 ** (attempt - 1))));
    }
  }
  throw lastError;
}

async function embedTexts(texts) {
  const vectors = [];
  const latenciesMs = [];
  for (let start = 0; start < texts.length; start += maxBatchSize) {
    const batch = texts.slice(start, start + maxBatchSize);
    const request = backendBase
      ? requestJson(`${backendBase}/api/benchmark/embeddings`, { texts: batch })
      : requestJson(`${apiBase}/embeddings`, {
        model: embeddingModel,
        input: batch,
        dimensions,
        encoding_format: 'float',
      });
    const { data, elapsedMs } = await request;
    const rawVectors = backendBase
      ? unwrapBackend(data).vectors
      : [...data.data].sort((a, b) => a.index - b.index).map(item => item.embedding);
    vectors.push(...rawVectors.map(normalizeVector));
    latenciesMs.push(elapsedMs);
    process.stdout.write(`Embedded ${Math.min(start + batch.length, texts.length)}/${texts.length}\r`);
  }
  process.stdout.write('\n');
  return { vectors, latenciesMs };
}

async function rewriteQuery(query) {
  const prompt = rewriteTemplate
    .replace('{question}', query.question)
    .replace('{history}', query.history ?? '');
  const request = backendBase
    ? requestJson(`${backendBase}/api/benchmark/rewrite`, {
      question: query.question,
      history: query.history ?? '',
    })
    : requestJson(`${apiBase}/chat/completions`, {
      model: chatModel,
      temperature: 0,
      messages: [{ role: 'user', content: prompt }],
    });
  const { data, elapsedMs } = await request;
  const raw = backendBase
    ? unwrapBackend(data).rewritten
    : data.choices?.[0]?.message?.content ?? query.question;
  const rewritten = raw
    .replace(/^```(?:text)?\s*/i, '')
    .replace(/```$/i, '')
    .replace(/[\r\n]+/g, ' ')
    .trim();
  return { text: rewritten || query.question, elapsedMs };
}

function unwrapBackend(response) {
  if (!response?.success) {
    throw new Error(`Benchmark backend failed: ${response?.message ?? 'unknown error'}`);
  }
  return response.data;
}

async function loadBackendProviderInfo() {
  if (!backendBase) return;
  const response = await fetch(`${backendBase}/api/llm-provider/list`, {
    signal: AbortSignal.timeout(10_000),
  });
  const payload = await response.json();
  const providers = unwrapBackend(payload);
  const chatProvider = providers.find(provider => provider.defaultChatProvider);
  const vectorProvider = providers.find(provider => provider.defaultEmbeddingProvider);
  chatModel = chatProvider?.model ?? 'application-default';
  embeddingModel = vectorProvider?.embeddingModel ?? 'application-default';
  dimensions = vectorProvider?.embeddingDimensions ?? dimensions;
}

function normalizeVector(vector) {
  const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
  return vector.map(value => value / (norm || 1));
}

function cosineSimilarity(left, right) {
  let score = 0;
  for (let index = 0; index < left.length; index += 1) {
    score += left[index] * right[index];
  }
  return score;
}

function rankDocuments(queryVector, documentVectors, minScore = Number.NEGATIVE_INFINITY) {
  return dataset.documents
    .map((document, index) => ({
      id: document.id,
      title: document.title,
      score: cosineSimilarity(queryVector, documentVectors[index]),
    }))
    .filter(item => item.score >= minScore)
    .sort((left, right) => right.score - left.score);
}

function resolveSearchParams(question) {
  const compactLength = question.replace(/\s+/g, '').length;
  if (compactLength <= retrievalPolicy.shortQueryLength) {
    return { topK: retrievalPolicy.topKShort, minScore: retrievalPolicy.minScoreShort };
  }
  if (compactLength <= retrievalPolicy.mediumQueryLength) {
    return { topK: retrievalPolicy.topKMedium, minScore: retrievalPolicy.minScoreDefault };
  }
  return { topK: retrievalPolicy.topKLong, minScore: retrievalPolicy.minScoreDefault };
}

function evaluate(rankings, kValues = [1, 3, 5]) {
  const totals = Object.fromEntries(kValues.flatMap(k => [
    [`hit@${k}`, 0],
    [`recall@${k}`, 0],
    [`ndcg@${k}`, 0],
  ]));
  let reciprocalRank = 0;
  let noHit = 0;

  rankings.forEach(({ query, ranking }) => {
    const relevant = new Set(query.relevant);
    if (ranking.length === 0) noHit += 1;
    const firstRelevant = ranking.findIndex(item => relevant.has(item.id));
    if (firstRelevant >= 0 && firstRelevant < 5) reciprocalRank += 1 / (firstRelevant + 1);

    kValues.forEach(k => {
      const top = ranking.slice(0, k);
      const hits = top.filter(item => relevant.has(item.id)).length;
      totals[`hit@${k}`] += hits > 0 ? 1 : 0;
      totals[`recall@${k}`] += hits / relevant.size;
      const dcg = top.reduce(
        (sum, item, index) => sum + (relevant.has(item.id) ? 1 / Math.log2(index + 2) : 0),
        0,
      );
      const idealCount = Math.min(relevant.size, k);
      const idcg = Array.from({ length: idealCount }, (_, index) => 1 / Math.log2(index + 2))
        .reduce((sum, value) => sum + value, 0);
      totals[`ndcg@${k}`] += idcg === 0 ? 0 : dcg / idcg;
    });
  });

  const count = rankings.length;
  return {
    ...Object.fromEntries(Object.entries(totals).map(([key, value]) => [key, value / count])),
    'mrr@5': reciprocalRank / count,
    coverage: (count - noHit) / count,
  };
}

function evaluateByGroup(rankings) {
  return Object.fromEntries([...new Set(rankings.map(item => item.query.group))].map(group => [
    group,
    evaluate(rankings.filter(item => item.query.group === group)),
  ]));
}

function percentile(values, percentileValue) {
  if (values.length === 0) return null;
  const ordered = [...values].sort((a, b) => a - b);
  const index = Math.min(ordered.length - 1, Math.ceil(percentileValue * ordered.length) - 1);
  return ordered[index];
}

function latencySummary(values) {
  return {
    count: values.length,
    meanMs: values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length,
    p50Ms: percentile(values, 0.50),
    p95Ms: percentile(values, 0.95),
  };
}

function formatMetric(value) {
  return value == null ? '-' : value.toFixed(4);
}

function markdownReport(result) {
  const rows = [
    ['Raw query', result.metrics.raw],
    ['Rewritten query', result.metrics.rewritten],
    ['Simulated retrieval policy', result.metrics.simulatedPolicy],
  ];
  const groupRows = Object.keys(result.queryGroups).map(group => (
    `| ${group} | ${result.queryGroups[group]} | ${formatMetric(result.metricsByGroup.raw[group]['hit@1'])} | ${formatMetric(result.metricsByGroup.simulatedPolicy[group]['hit@1'])} |`
  )).join('\n');
  const queryRows = result.queries.map(item => (
    `| ${item.id} | ${item.group} | ${item.question.replaceAll('|', '\\|')} | ${item.rewritten.replaceAll('|', '\\|')} | ${item.rawTop1} | ${item.policyTop1} | ${item.policyCorrect ? 'yes' : 'no'} |`
  )).join('\n');

  return `# RAG Retrieval Benchmark\n\n`
    + `Generated: ${result.generatedAt}\n\n`
    + `Dataset: ${result.dataset} (${result.corpusSize} documents, ${result.queryCount} queries)\n\n`
    + `Models: ${result.embeddingModel} (${result.dimensions} dimensions), ${result.chatModel ?? 'rewrite skipped'}\n\n`
    + `Method: exact cosine ranking in memory. The simulated policy mirrors the configured top-K and score thresholds, but does not exercise pgvector, HNSW, metadata filtering, or end-to-end answer generation.\n\n`
    + `> This is a small synthetic engineering benchmark, not a production accuracy claim. Re-run on a domain-labelled dataset before putting an uplift number on a resume.\n\n`
    + `## Retrieval quality\n\n`
    + `| Variant | Hit@1 | Hit@3 | Recall@5 | MRR@5 | nDCG@5 | Coverage |\n`
    + `|---|---:|---:|---:|---:|---:|---:|\n`
    + rows.map(([name, metrics]) => `| ${name} | ${formatMetric(metrics['hit@1'])} | ${formatMetric(metrics['hit@3'])} | ${formatMetric(metrics['recall@5'])} | ${formatMetric(metrics['mrr@5'])} | ${formatMetric(metrics['ndcg@5'])} | ${formatMetric(metrics.coverage)} |`).join('\n')
    + `\n\nQuery Rewrite Hit@1 delta: ${result.deltas.rewriteHitAt1PercentagePoints.toFixed(2)} percentage points.\n\n`
    + `## Hit@1 by query group\n\n`
    + `| Group | Queries | Raw | Simulated policy |\n`
    + `|---|---:|---:|---:|\n`
    + groupRows
    + `\n\n`
    + `## Latency\n\n`
    + `| Operation | Count | Mean ms | P50 ms | P95 ms |\n`
    + `|---|---:|---:|---:|---:|\n`
    + Object.entries(result.latency).map(([name, summary]) => `| ${name} | ${summary.count} | ${formatMetric(summary.meanMs)} | ${formatMetric(summary.p50Ms)} | ${formatMetric(summary.p95Ms)} |`).join('\n')
    + `\n\n## Query details\n\n`
    + `| ID | Group | Original | Rewritten | Raw top-1 | Policy top-1 | Correct |\n`
    + `|---|---|---|---|---|---|---|\n`
    + queryRows
    + `\n`;
}

await loadBackendProviderInfo();
console.log(`Dataset: ${dataset.documents.length} documents, ${dataset.queries.length} queries`);
console.log(`Embedding model: ${embeddingModel}; chat model: ${chatModel}; rewrite: ${!skipRewrite}`);
console.log(`Provider mode: ${backendBase ? `application backend (${backendBase})` : 'direct API'}`);

const totalStarted = performance.now();
const documentEmbedding = await embedTexts(dataset.documents.map(document => document.text));

const rewrites = [];
const rewriteLatencies = [];
for (let index = 0; index < dataset.queries.length; index += 1) {
  const query = dataset.queries[index];
  if (skipRewrite) {
    rewrites.push(query.question);
  } else {
    const rewrite = await rewriteQuery(query);
    rewrites.push(rewrite.text);
    rewriteLatencies.push(rewrite.elapsedMs);
  }
  process.stdout.write(`Rewritten ${index + 1}/${dataset.queries.length}\r`);
}
process.stdout.write('\n');

const rawEmbedding = await embedTexts(dataset.queries.map(query => query.question));
const rewrittenEmbedding = skipRewrite
  ? { vectors: rawEmbedding.vectors, latenciesMs: [] }
  : await embedTexts(rewrites);

const rawRankings = [];
const rewrittenRankings = [];
const policyRankings = [];
const localRetrievalLatencies = [];

dataset.queries.forEach((query, index) => {
  const started = performance.now();
  const rawRanking = rankDocuments(rawEmbedding.vectors[index], documentEmbedding.vectors);
  const rewrittenRanking = rankDocuments(rewrittenEmbedding.vectors[index], documentEmbedding.vectors);
  const searchParams = resolveSearchParams(query.question);
  let policyRanking = rankDocuments(
    rewrittenEmbedding.vectors[index], documentEmbedding.vectors, searchParams.minScore,
  ).slice(0, searchParams.topK);
  if (policyRanking.length === 0) {
    policyRanking = rankDocuments(
      rawEmbedding.vectors[index], documentEmbedding.vectors, searchParams.minScore,
    ).slice(0, searchParams.topK);
  }
  localRetrievalLatencies.push(performance.now() - started);
  rawRankings.push({ query, ranking: rawRanking });
  rewrittenRankings.push({ query, ranking: rewrittenRanking });
  policyRankings.push({ query, ranking: policyRanking });
});

const metrics = {
  raw: evaluate(rawRankings),
  rewritten: evaluate(rewrittenRankings),
  simulatedPolicy: evaluate(policyRankings),
};
const metricsByGroup = {
  raw: evaluateByGroup(rawRankings),
  rewritten: evaluateByGroup(rewrittenRankings),
  simulatedPolicy: evaluateByGroup(policyRankings),
};

const result = {
  generatedAt: new Date().toISOString(),
  dataset: dataset.name,
  corpusSize: dataset.documents.length,
  queryCount: dataset.queries.length,
  queryGroups: Object.fromEntries([...new Set(dataset.queries.map(query => query.group))].map(group => [
    group,
    dataset.queries.filter(query => query.group === group).length,
  ])),
  embeddingModel,
  chatModel: skipRewrite ? null : chatModel,
  dimensions: documentEmbedding.vectors[0]?.length ?? 0,
  method: 'exact-cosine-in-memory',
  retrievalPolicy,
  metrics,
  metricsByGroup,
  deltas: {
    rewriteHitAt1PercentagePoints: (metrics.rewritten['hit@1'] - metrics.raw['hit@1']) * 100,
    simulatedPolicyHitAt1PercentagePoints:
      (metrics.simulatedPolicy['hit@1'] - metrics.raw['hit@1']) * 100,
  },
  latency: {
    documentEmbeddingBatch: latencySummary(documentEmbedding.latenciesMs),
    rawQueryEmbeddingBatch: latencySummary(rawEmbedding.latenciesMs),
    rewrittenQueryEmbeddingBatch: latencySummary(rewrittenEmbedding.latenciesMs),
    queryRewrite: latencySummary(rewriteLatencies),
    simulatedCosinePolicy: latencySummary(localRetrievalLatencies),
  },
  totalElapsedMs: performance.now() - totalStarted,
  queries: dataset.queries.map((query, index) => {
    const rawTop1 = rawRankings[index].ranking[0]?.id ?? '-';
    const policyTop1 = policyRankings[index].ranking[0]?.id ?? '-';
    return {
      id: query.id,
      group: query.group,
      question: query.question,
      rewritten: rewrites[index],
      relevant: query.relevant,
      rawTop1,
      rawTop1Score: rawRankings[index].ranking[0]?.score ?? null,
      rawTop5: rawRankings[index].ranking.slice(0, 5),
      policyTop1,
      policyTop1Score: policyRankings[index].ranking[0]?.score ?? null,
      policyTop5: policyRankings[index].ranking.slice(0, 5),
      policyCorrect: query.relevant.includes(policyTop1),
    };
  }),
};

const timestamp = result.generatedAt.replaceAll(':', '-').replace(/\.\d{3}Z$/, 'Z');
const outputDir = path.join(benchmarkDir, 'results');
await mkdir(outputDir, { recursive: true });
const jsonPath = path.join(outputDir, `${timestamp}.json`);
const markdownPath = path.join(outputDir, `${timestamp}.md`);
await writeFile(jsonPath, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
await writeFile(markdownPath, markdownReport(result), 'utf8');

console.log(JSON.stringify({
  metrics,
  metricsByGroup,
  deltas: result.deltas,
  latency: result.latency,
  totalElapsedMs: result.totalElapsedMs,
}, null, 2));
console.log(`JSON result: ${jsonPath}`);
console.log(`Markdown result: ${markdownPath}`);
