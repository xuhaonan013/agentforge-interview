from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import ir_measures


# The local benchmark controller intentionally caps embedding batches at 30/minute.
# Keep a little headroom and wait once rather than retrying a burst of rejected calls.
_EMBED_REQUESTS_IN_WINDOW = 0
_EMBED_WINDOW_STARTED = 0.0


MEASURES = [
    ir_measures.P @ 1,
    ir_measures.P @ 3,
    ir_measures.P @ 5,
    ir_measures.P @ 10,
    ir_measures.R @ 1,
    ir_measures.R @ 3,
    ir_measures.R @ 5,
    ir_measures.R @ 10,
    ir_measures.Success @ 1,
    ir_measures.Success @ 3,
    ir_measures.Success @ 5,
    ir_measures.Success @ 10,
    ir_measures.RR @ 10,
    ir_measures.nDCG @ 10,
    ir_measures.AP @ 10,
    ir_measures.Rprec,
]
PRIMARY_MEASURES = {
    "R@5": "recall@5",
    "RR@10": "mrr@10",
    "nDCG@10": "ndcg@10",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean_ms": round(statistics.fmean(values), 3) if values else None,
        "p50_ms": round(percentile(values, 0.50), 3) if values else None,
        "p95_ms": round(percentile(values, 0.95), 3) if values else None,
        "p99_ms": round(percentile(values, 0.99), 3) if values else None,
    }


def load_dataset(dataset_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_path = dataset_dir / "dataset.json"
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    documents = dataset["documents"]
    queries = dataset["queries"]
    document_ids = {document["id"] for document in documents}
    query_ids = set()
    for query in queries:
        if query["id"] in query_ids:
            raise ValueError(f"duplicate query id: {query['id']}")
        query_ids.add(query["id"])
        qrels = query.get("qrels", {})
        if query.get("answerable") and not qrels:
            raise ValueError(f"answerable query has no qrels: {query['id']}")
        if not query.get("answerable") and qrels:
            raise ValueError(f"no-answer query has qrels: {query['id']}")
        unknown = set(qrels) - document_ids
        if unknown:
            raise ValueError(f"unknown qrels for {query['id']}: {sorted(unknown)}")
    return dataset, documents, queries


# A small deterministic tokenizer keeps the lexical baseline dependency-free.
# Chinese text is represented by overlapping bigrams; Latin identifiers and
# numbers are kept intact so Java, SQL, Redis and model names remain searchable.
STOPWORDS = set("的了和是与在中为及将并一个一种可以进行通过使用用于如果如何什么哪些应该需要以及这其对从到等着后时" )


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens: list[str] = []
    for match in re.finditer(r"[a-z][a-z0-9_+#.-]*|\d+(?:\.\d+)?|[\u4e00-\u9fff]+", lowered):
        value = match.group(0)
        if value[0].isascii():
            if len(value) >= 2:
                tokens.append(value)
            continue
        compact = "".join(char for char in value if char not in STOPWORDS)
        tokens.extend(compact[index:index + 2] for index in range(max(0, len(compact) - 1)))
    return tokens


class BM25:
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.2, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(document["title"] + "\n" + document["text"]) for document in documents]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.average_length = statistics.fmean(self.doc_lengths) if self.doc_lengths else 1.0
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            self.document_frequency.update(set(tokens))

    def rank(self, text: str) -> list[tuple[str, float]]:
        query_tokens = Counter(tokenize(text))
        document_count = len(self.documents)
        scored: list[tuple[int, float]] = []
        for index, tokens in enumerate(self.doc_tokens):
            frequencies = Counter(tokens)
            score = 0.0
            length = self.doc_lengths[index] or 1
            for term, query_frequency in query_tokens.items():
                term_frequency = frequencies.get(term, 0)
                if not term_frequency:
                    continue
                df = self.document_frequency.get(term, 0)
                idf = math.log(1.0 + (document_count - df + 0.5) / (df + 0.5))
                denominator = term_frequency + self.k1 * (1 - self.b + self.b * length / self.average_length)
                score += idf * (term_frequency * (self.k1 + 1) / denominator) * min(query_frequency, 3)
            scored.append((index, score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [(self.documents[index]["id"], score) for index, score in scored]


def normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def post_json(url: str, body: dict[str, Any], token: str | None, timeout: float = 90.0) -> tuple[dict[str, Any], float]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Benchmark-Token"] = token
    for attempt in range(3):
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:800]
            if error.code == 429 and attempt < 2:
                print("benchmark backend returned HTTP 429; waiting 60s", flush=True)
                time.sleep(60)
                continue
            raise RuntimeError(f"HTTP {error.code} from benchmark backend: {detail}") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"benchmark backend unavailable: {error.reason}") from error
        elapsed_ms = (time.perf_counter() - started) * 1000
        if data.get("success", False):
            return data["data"], elapsed_ms
        message = str(data.get("message", data))
        if "请求过于频繁" in message and attempt < 2:
            print("benchmark backend rate limit; waiting 60s", flush=True)
            time.sleep(60)
            continue
        raise RuntimeError(f"benchmark backend rejected request: {message}")
    raise RuntimeError("benchmark backend retry limit exceeded")


def embed_backend(texts: list[str], backend_url: str, token: str, batch_size: int = 10) -> tuple[list[list[float]], list[float], int]:
    global _EMBED_REQUESTS_IN_WINDOW, _EMBED_WINDOW_STARTED
    vectors: list[list[float]] = []
    latencies: list[float] = []
    dimensions = 0
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        now = time.monotonic()
        if _EMBED_REQUESTS_IN_WINDOW == 0:
            _EMBED_WINDOW_STARTED = now
        elif _EMBED_REQUESTS_IN_WINDOW >= 28 and now - _EMBED_WINDOW_STARTED < 61:
            wait_seconds = 61 - (now - _EMBED_WINDOW_STARTED)
            print(f"embedding rate limit guard: waiting {wait_seconds:.0f}s", flush=True)
            time.sleep(wait_seconds)
            _EMBED_REQUESTS_IN_WINDOW = 0
            _EMBED_WINDOW_STARTED = time.monotonic()
        data, elapsed_ms = post_json(
            backend_url.rstrip("/") + "/api/benchmark/embeddings",
            {"texts": batch},
            token,
        )
        raw = data.get("vectors", [])
        if len(raw) != len(batch):
            raise RuntimeError(f"embedding response length {len(raw)} != request length {len(batch)}")
        vectors.extend(normalize_vector([float(value) for value in vector]) for vector in raw)
        dimensions = int(data.get("dimensions") or (len(raw[0]) if raw else 0))
        latencies.append(elapsed_ms)
        _EMBED_REQUESTS_IN_WINDOW += 1
        print(f"embedded {min(start + len(batch), len(texts))}/{len(texts)}", flush=True)
    return vectors, latencies, dimensions


def rank_with_embeddings(
    query_texts: list[str],
    documents: list[dict[str, Any]],
    backend_url: str,
    token: str,
) -> tuple[dict[str, list[tuple[str, float]]], dict[str, Any]]:
    document_vectors, embed_latencies, dimensions = embed_backend(
        [document["title"] + "\n" + document["text"] for document in documents],
        backend_url,
        token,
    )
    query_vectors, query_latencies, query_dimensions = embed_backend(query_texts, backend_url, token)
    if dimensions and query_dimensions and dimensions != query_dimensions:
        raise RuntimeError(f"document/query embedding dimensions differ: {dimensions} vs {query_dimensions}")
    rankings: dict[str, list[tuple[str, float]]] = {}
    for query_text, vector in zip(query_texts, query_vectors):
        del query_text
        ranked = sorted(
            ((document["id"], cosine(vector, document_vectors[index])) for index, document in enumerate(documents)),
            key=lambda item: (-item[1], item[0]),
        )
        # The caller replaces this temporary key with the query id in order.
        rankings.setdefault("__pending__", []).append(ranked)
    return {"__pending__": rankings["__pending__"]}, {
        "embedding_dimensions": dimensions or query_dimensions,
        "embedding_latency": latency_summary(embed_latencies + query_latencies),
        "embedding_request_count": len(embed_latencies) + len(query_latencies),
    }


def metric_key(measure: Any) -> str:
    return str(measure)


def aggregate_ir_metrics(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    aggregate = ir_measures.calc_aggregate(MEASURES, qrels, run)
    values = {metric_key(measure): float(value) for measure, value in aggregate.items()}
    per_query: dict[str, dict[str, float]] = defaultdict(dict)
    for item in ir_measures.iter_calc(MEASURES, qrels, run):
        per_query[item.query_id][metric_key(item.measure)] = float(item.value)
    return values, dict(per_query)


def bootstrap_ci(values: list[float], seed: int = 20260819, samples: int = 1000) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "low95": None, "high95": None, "n": 0}
    # Local deterministic PRNG keeps the report reproducible without numpy.
    import random

    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return {
        "mean": statistics.fmean(values),
        "low95": means[int(0.025 * samples)],
        "high95": means[int(0.975 * samples) - 1],
        "n": len(values),
    }


def bootstrap_primary(per_query: dict[str, dict[str, float]]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    for source_key, output_key in PRIMARY_MEASURES.items():
        result[output_key] = bootstrap_ci([
            values[source_key] for values in per_query.values() if source_key in values
        ])
    return result


def roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    # Equivalent to the Mann-Whitney statistic, with ties receiving half credit.
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    rank_sum = 0.0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        rank_sum += sum(average_rank for index in order[position:end] if labels[index])
        position = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def pr_auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    tp = 0
    fp = 0
    previous_recall = 0.0
    area = 0.0
    for index in order:
        if labels[index]:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def best_abstention_threshold(labels: list[bool], scores: list[float]) -> dict[str, float | None]:
    candidates = sorted(set(scores))
    best: tuple[float, float] | None = None
    for threshold in candidates:
        stats = abstention_stats(labels, scores, threshold)
        f1 = stats["abstention_f1"]
        if f1 is not None and (best is None or f1 > best[1]):
            best = (threshold, f1)
    if best is None:
        return {"threshold": None, "abstention_f1": None}
    return {"threshold": best[0], "abstention_f1": best[1]}


def abstention_stats(labels: list[bool], scores: list[float], threshold: float) -> dict[str, float | int | None]:
    # labels=True means answerable. A low top-1 score predicts abstention.
    predicted_abstain = [score <= threshold for score in scores]
    actual_abstain = [not label for label in labels]
    true_positive = sum(pred and actual for pred, actual in zip(predicted_abstain, actual_abstain))
    false_positive = sum(pred and not actual for pred, actual in zip(predicted_abstain, actual_abstain))
    false_negative = sum(not pred and actual for pred, actual in zip(predicted_abstain, actual_abstain))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall and precision + recall else 0.0
    no_answer_count = sum(actual_abstain)
    no_answer_accepted = sum(not pred and actual for pred, actual in zip(predicted_abstain, actual_abstain))
    return {
        "threshold": threshold,
        "predicted_abstain": sum(predicted_abstain),
        "actual_no_answer": no_answer_count,
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": f1,
        "false_positive_rate": false_positive / sum(labels) if sum(labels) else None,
        "no_answer_false_accept_rate": no_answer_accepted / no_answer_count if no_answer_count else None,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    primary = result["primary_bootstrap_95ci"]
    rows = [
        ("Recall@1", metrics.get("R@1")),
        ("Recall@3", metrics.get("R@3")),
        ("Recall@5", metrics.get("R@5")),
        ("Recall@10", metrics.get("R@10")),
        ("Precision@5", metrics.get("P@5")),
        ("Hit@1", metrics.get("Success@1")),
        ("Hit@3", metrics.get("Success@3")),
        ("Hit@5", metrics.get("Success@5")),
        ("Hit@10", metrics.get("Success@10")),
        ("MRR@10", metrics.get("RR@10")),
        ("nDCG@10", metrics.get("nDCG@10")),
        ("MAP@10", metrics.get("AP@10")),
        ("R-Precision", metrics.get("Rprec")),
    ]
    metric_rows = "\n".join(f"| {name} | {format_value(value)} |" for name, value in rows)
    ci_rows = "\n".join(
        f"| {name} | {format_value(value.get('mean'))} | {format_value(value.get('low95'))} | {format_value(value.get('high95'))} |"
        for name, value in primary.items()
    )
    group_rows = []
    for group, group_result in result["by_group"].items():
        group_rows.append(
            f"| {group} | {group_result['count']} | {format_value(group_result['metrics'].get('R@5'))} | "
            f"{format_value(group_result['metrics'].get('RR@10'))} | {format_value(group_result['metrics'].get('nDCG@10'))} |"
        )
    abstain = result["abstention"]
    return (
        "# RAG Retrieval Benchmark v2\n\n"
        f"Generated: `{result['generated_at']}`\n\n"
        f"Dataset: `{result['dataset_version']}`; {result['document_count']} documents, "
        f"{result['query_count']} queries ({result['answerable_count']} answerable, {result['no_answer_count']} no-answer).\n\n"
        f"Mode: **{result['mode']}**. Ranking is exact in-memory retrieval; it does not claim pgvector/HNSW production latency.\n\n"
        "> The qrels/gold answers were generated from runtime-provided documents and still require human review before a resume claim. "
        "This report is an offline regression baseline, not a production accuracy statement.\n\n"
        "## Retrieval metrics\n\n"
        "| Metric | Value |\n|---|---:|\n"
        f"{metric_rows}\n\n"
        "## Primary bootstrap intervals\n\n"
        "| Metric | Mean | 95% low | 95% high |\n|---|---:|---:|---:|\n"
        f"{ci_rows}\n\n"
        "## Metrics by query group\n\n"
        "| Group | Queries | Recall@5 | MRR@10 | nDCG@10 |\n|---|---:|---:|---:|---:|\n"
        + "\n".join(group_rows)
        + "\n\n## No-answer / abstention\n\n"
        "| Measure | Value |\n|---|---:|\n"
        f"| Threshold | {format_value(abstain.get('threshold'))} |\n"
        f"| Threshold source | {abstain.get('threshold_source', '-')} |\n"
        f"| Abstention precision | {format_value(abstain.get('abstention_precision'))} |\n"
        f"| Abstention recall | {format_value(abstain.get('abstention_recall'))} |\n"
        f"| Abstention F1 | {format_value(abstain.get('abstention_f1'))} |\n"
        f"| Abstention false-positive rate | {format_value(abstain.get('false_positive_rate'))} |\n"
        f"| No-answer false-accept rate | {format_value(abstain.get('no_answer_false_accept_rate'))} |\n"
        f"| Answerability AUROC | {format_value(result['answerability_auroc'])} |\n"
        f"| Answerability AUPRC | {format_value(result['answerability_auprc'])} |\n\n"
        "## Latency\n\n"
        "| Operation | Count | Mean ms | P50 ms | P95 ms | P99 ms |\n|---|---:|---:|---:|---:|---:|\n"
        + "\n".join(
            f"| {name} | {summary['count']} | {format_value(summary['mean_ms'])} | {format_value(summary['p50_ms'])} | "
            f"{format_value(summary['p95_ms'])} | {format_value(summary['p99_ms'])} |"
            for name, summary in result["latency"].items()
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="TREC/BEIR-style retrieval benchmark for dataset v2.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "v2")
    parser.add_argument("--mode", choices=("lexical", "embedding"), default="lexical")
    parser.add_argument("--backend-url", default=os.environ.get("BENCHMARK_BACKEND_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--benchmark-token", default=os.environ.get("BENCHMARK_TOKEN", "local-only-benchmark-token"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N queries after split filtering.")
    parser.add_argument("--document-limit", type=int, default=None, help="Restrict the corpus for a smoke test; omit for the full corpus.")
    parser.add_argument("--threshold", type=float, default=None, help="Top-1 score threshold for abstention.")
    parser.add_argument("--calibrate-from-dev", action="store_true", help="Select the abstention threshold on the dev split.")
    parser.add_argument("--rewrite", action="store_true", help="Reserved for a later rewrite adapter; raw query is used by default.")
    args = parser.parse_args()
    if args.rewrite:
        raise SystemExit("--rewrite is intentionally not implemented in v2 yet; raw-query evaluation avoids an extra model variable.")

    dataset, documents, all_queries = load_dataset(args.dataset_dir)
    full_document_count = len(documents)
    if args.document_limit is not None:
        if args.document_limit < 1:
            raise SystemExit("--document-limit must be positive")
        documents = documents[:args.document_limit]
    queries = [query for query in all_queries if query.get("split", "test") == args.split]
    if args.limit:
        queries = queries[:args.limit]
    if not queries:
        raise SystemExit(f"no queries found for split={args.split!r}")

    query_texts = [((query.get("history") or "") + "\n" + query["question"]).strip() for query in queries]
    rankings: dict[str, list[tuple[str, float]]] = {}
    latency: dict[str, dict[str, Any]] = {}
    calibration: dict[str, Any] | None = None
    if args.mode == "lexical":
        ranker = BM25(documents)
        ranking_times: list[float] = []
        for query, query_text in zip(queries, query_texts):
            started = time.perf_counter()
            rankings[query["id"]] = ranker.rank(query_text)
            ranking_times.append((time.perf_counter() - started) * 1000)
        latency["lexical_rank"] = latency_summary(ranking_times)
        dimensions = None
        if args.calibrate_from_dev and args.split != "dev":
            dev_queries = [query for query in all_queries if query.get("split") == "dev"]
            dev_labels = [bool(query.get("answerable")) for query in dev_queries]
            dev_scores = []
            for query in dev_queries:
                dev_query_text = ((query.get("history") or "") + "\n" + query["question"]).strip()
                dev_ranked = ranker.rank(dev_query_text)
                dev_scores.append(dev_ranked[0][1] if dev_ranked else 0.0)
            calibration = best_abstention_threshold(dev_labels, dev_scores)
    else:
        if args.calibrate_from_dev:
            raise SystemExit("--calibrate-from-dev currently supports lexical mode only; embed the dev split explicitly before tuning an embedding threshold.")
        try:
            pending, embedding_latency = rank_with_embeddings(query_texts, documents, args.backend_url, args.benchmark_token)
        except Exception as error:  # noqa: BLE001
            print(f"MODEL_RUN_BLOCKED: {error}", file=sys.stderr)
            print("The dataset and lexical baseline remain runnable; restore provider quota or configure another embedding provider.", file=sys.stderr)
            return 2
        for query, ranked in zip(queries, pending["__pending__"]):
            rankings[query["id"]] = ranked
        latency["embedding_request"] = embedding_latency["embedding_latency"]
        dimensions = embedding_latency["embedding_dimensions"]

    qrels = {
        query["id"]: {doc_id: int(grade) for doc_id, grade in query.get("qrels", {}).items()}
        for query in queries
        if query.get("answerable") and query.get("qrels")
    }
    run = {query_id: {doc_id: score for doc_id, score in ranked} for query_id, ranked in rankings.items()}
    answerable_ids = set(qrels)
    answerable_run = {query_id: run[query_id] for query_id in answerable_ids}
    metrics, per_query = aggregate_ir_metrics(qrels, answerable_run)
    primary_ci = bootstrap_primary(per_query)

    by_group: dict[str, Any] = {}
    for group in sorted({query["group"] for query in queries}):
        group_queries = [query for query in queries if query["group"] == group and query["id"] in answerable_ids]
        group_qrels = {query["id"]: qrels[query["id"]] for query in group_queries}
        group_run = {query["id"]: run[query["id"]] for query in group_queries}
        group_metrics, _ = aggregate_ir_metrics(group_qrels, group_run) if group_qrels else ({}, {})
        by_group[group] = {"count": len(group_queries), "metrics": group_metrics}

    labels: list[bool] = []
    top_scores: list[float] = []
    for query in queries:
        labels.append(bool(query.get("answerable")))
        top_scores.append(rankings[query["id"]][0][1] if rankings[query["id"]] else 0.0)
    threshold_source = "explicit" if args.threshold is not None else "default"
    if args.threshold is not None:
        threshold = args.threshold
    elif calibration and calibration.get("threshold") is not None:
        threshold = float(calibration["threshold"])
        threshold_source = "dev"
    elif args.mode == "lexical":
        threshold = 0.0
    else:
        threshold = 0.28
    abstention = abstention_stats(labels, top_scores, threshold)
    abstention["oracle_best_on_this_split"] = best_abstention_threshold(labels, top_scores)
    abstention["threshold_source"] = threshold_source
    abstention["dev_calibration"] = calibration

    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "dataset_version": dataset.get("version", "unknown"),
        "dataset_dir": str(args.dataset_dir),
        "dataset_files": {
            name: sha256_file(args.dataset_dir / name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset.json", "manifest.json")
            if (args.dataset_dir / name).exists()
        },
        "mode": args.mode,
        "embedding_dimensions": dimensions,
        "split": args.split,
        "document_count": len(documents),
        "corpus_document_count": full_document_count,
        "query_count": len(queries),
        "answerable_count": len(answerable_ids),
        "no_answer_count": sum(not query.get("answerable") for query in queries),
        "metrics": metrics,
        "primary_bootstrap_95ci": primary_ci,
        "by_group": by_group,
        "abstention": abstention,
        "answerability_auroc": roc_auc(labels, top_scores),
        "answerability_auprc": pr_auc(labels, top_scores),
        "latency": latency,
        "method": "BM25 lexical baseline" if args.mode == "lexical" else "provider embeddings + exact cosine in memory",
        "limitations": [
            "Qrels and gold answers were generated from runtime-provided documents and require human review.",
            f"This report evaluates the {args.split} split; the threshold source is {threshold_source}.",
            "Embedding mode ranks in memory and does not measure pgvector/HNSW production latency.",
        ],
    }
    output_dir = args.dataset_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{utc_now()}-{args.mode}"
    json_path = output_dir / f"{prefix}.json"
    markdown_path = output_dir / f"{prefix}.md"
    run_path = output_dir / f"{prefix}.run.tsv"
    json_path.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(result), encoding="utf-8")
    # TREC run format: query_id, iteration, doc_id, rank, score, tag.
    run_lines = [
        f"{query_id}\t0\t{doc_id}\t{rank + 1}\t{score:.8f}\tv2-{args.mode}"
        for query_id, ranked in rankings.items()
        for rank, (doc_id, score) in enumerate(ranked)
    ]
    run_path.write_text("\n".join(run_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "mode": args.mode,
        "documents": len(documents),
        "queries": len(queries),
        "answerable": len(answerable_ids),
        "no_answer": sum(not query.get("answerable") for query in queries),
        "recall@5": metrics.get("R@5"),
        "mrr@10": metrics.get("RR@10"),
        "ndcg@10": metrics.get("nDCG@10"),
        "map@10": metrics.get("AP@10"),
        "r_precision": metrics.get("Rprec"),
        "abstention_f1": abstention.get("abstention_f1"),
        "answerability_auroc": result["answerability_auroc"],
        "json": str(json_path),
        "markdown": str(markdown_path),
        "run": str(run_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
