# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ragas==0.4.3",
#   "langchain==0.3.27",
#   "langchain-core==0.3.79",
#   "langchain-community==0.3.31",
#   "langchain-openai==0.3.35",
#   "openai==2.54.0",
#   "httpx==0.28.1",
#   "instructor==1.15.4",
#   "pydantic==2.13.4",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import AsyncOpenAI

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from ragas import EvaluationDataset, SingleTurnSample  # noqa: E402
from ragas.cache import DiskCacheBackend  # noqa: E402
from ragas.embeddings.base import embedding_factory  # noqa: E402
from ragas.llms import llm_factory  # noqa: E402
from ragas.metrics.collections import (  # noqa: E402
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    FactualCorrectness,
    Faithfulness,
)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RETRIEVAL_DIR = HERE.parent / "rag-retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ragas evaluation for agentforge-interview")
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BENCHMARK_BACKEND_URL", "http://127.0.0.1:8080"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=RETRIEVAL_DIR / "dataset.json",
    )
    parser.add_argument(
        "--references",
        type=Path,
        default=HERE / "references.json",
    )
    parser.add_argument(
        "--rewrite-result",
        type=Path,
        default=None,
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--metric-timeout", type=float, default=180.0)
    parser.add_argument("--ids", nargs="*", help="Optional query IDs for a smoke run")
    parser.add_argument(
        "--resume-result",
        type=Path,
        help="Reuse collected cases from an existing JSON result",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume-result, score only missing or failed metrics",
    )
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


class BackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "X-Benchmark-Token": os.getenv(
                "BENCHMARK_TOKEN", "local-only-benchmark-token"
            )
        }
        self.client = httpx.Client(timeout=httpx.Timeout(180.0, connect=10.0))

    def close(self) -> None:
        self.client.close()

    def _unwrap(self, response: httpx.Response) -> Any:
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(
                f"Backend request failed: code={payload.get('code')} "
                f"message={payload.get('message')}"
            )
        return payload["data"]

    def provider_info(self) -> dict[str, Any]:
        payload = self._unwrap(self.client.get(f"{self.base_url}/api/llm-provider/list"))
        chat = next(item for item in payload if item.get("defaultChatProvider"))
        embedding = next(
            item for item in payload if item.get("defaultEmbeddingProvider")
        )
        return {
            "chat_model": chat["model"],
            "embedding_model": embedding["embeddingModel"],
            "embedding_dimensions": embedding["embeddingDimensions"],
        }

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 10):
            data = self._unwrap(
                self.client.post(
                    f"{self.base_url}/api/benchmark/embeddings",
                    json={"texts": texts[start : start + 10]},
                    headers=self.headers,
                )
            )
            vectors.extend(data["vectors"])
        return vectors

    def generate(self, system: str, user: str) -> str:
        data = self._unwrap(
            self.client.post(
                f"{self.base_url}/api/benchmark/generate",
                json={"system": system, "user": user},
                headers=self.headers,
            )
        )
        return data["content"].strip()


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def build_user_input(query: dict[str, Any]) -> str:
    history = query.get("history", "").strip()
    if not history:
        return query["question"]
    return f"{history}\n当前问题: {query['question']}"


def render_user_prompt(template: str, contexts: list[str], question: str) -> str:
    return template.replace("{context}", "\n\n---\n\n".join(contexts)).replace(
        "{question}", question
    )


def collect_cases(
    args: argparse.Namespace,
    backend: BackendClient,
    provider: dict[str, Any],
) -> list[dict[str, Any]]:
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    reference_data = json.loads(args.references.read_text(encoding="utf-8"))
    rewrite_path = args.rewrite_result
    if rewrite_path is None:
        candidates = sorted((RETRIEVAL_DIR / "results").glob("*.json"))
        if not candidates:
            raise FileNotFoundError(
                "No retrieval result found; pass --rewrite-result explicitly"
            )
        rewrite_path = candidates[-1]
        args.rewrite_result = rewrite_path
    rewrite_data = json.loads(rewrite_path.read_text(encoding="utf-8"))
    references: dict[str, str] = reference_data["references"]
    overrides: dict[str, list[str]] = reference_data.get("relevance_overrides", {})
    rewrites = {item["id"]: item["rewritten"] for item in rewrite_data["queries"]}

    documents = dataset["documents"]
    document_by_id = {item["id"]: item for item in documents}
    queries = dataset["queries"]
    if args.ids:
        requested = set(args.ids)
        queries = [item for item in queries if item["id"] in requested]
        missing = requested - {item["id"] for item in queries}
        if missing:
            raise ValueError(f"Unknown query IDs: {sorted(missing)}")

    missing_references = [item["id"] for item in queries if item["id"] not in references]
    if missing_references:
        raise ValueError(f"Missing reference answers: {missing_references}")

    print(f"Embedding {len(documents)} documents and {len(queries)} frozen queries...")
    document_vectors = [normalize(item) for item in backend.embed(
        [item["text"] for item in documents]
    )]
    retrieval_queries = [rewrites.get(item["id"], item["question"]) for item in queries]
    query_vectors = [normalize(item) for item in backend.embed(retrieval_queries)]

    system_template = (
        REPO_ROOT
        / "app"
        / "src"
        / "main"
        / "resources"
        / "prompts"
        / "knowledgebase-query-system.st"
    ).read_text(encoding="utf-8")
    user_template = (
        REPO_ROOT
        / "app"
        / "src"
        / "main"
        / "resources"
        / "prompts"
        / "knowledgebase-query-user.st"
    ).read_text(encoding="utf-8")

    cases: list[dict[str, Any]] = []
    for query, query_vector, retrieval_query in zip(
        queries, query_vectors, retrieval_queries, strict=True
    ):
        ranking = sorted(
            (
                {
                    "id": document["id"],
                    "score": cosine(query_vector, document_vectors[index]),
                }
                for index, document in enumerate(documents)
            ),
            key=lambda item: item["score"],
            reverse=True,
        )[: args.top_k]
        retrieved_ids = [item["id"] for item in ranking]
        contexts = [document_by_id[item_id]["text"] for item_id in retrieved_ids]
        reference_ids = overrides.get(query["id"], query["relevant"])
        cases.append(
            {
                "id": query["id"],
                "group": query["group"],
                "question": query["question"],
                "history": query.get("history", ""),
                "user_input": build_user_input(query),
                "retrieval_query": retrieval_query,
                "retrieved_context_ids": retrieved_ids,
                "retrieved_context_scores": [item["score"] for item in ranking],
                "retrieved_contexts": contexts,
                "reference_context_ids": reference_ids,
                "reference_contexts": [document_by_id[item_id]["text"] for item_id in reference_ids],
                "reference": references[query["id"]],
                "response": "",
                "models": provider,
            }
        )

    print(f"Generating {len(cases)} RAG responses with {provider['chat_model']}...")

    def generate_answer(case: dict[str, Any]) -> tuple[str, str, str | None]:
        prompt = render_user_prompt(
            user_template,
            case["retrieved_contexts"],
            case["user_input"],
        )
        try:
            return case["id"], backend.generate(system_template, prompt), None
        except Exception as error:  # noqa: BLE001
            return case["id"], "", f"{type(error).__name__}: {error}"

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(generate_answer, case): case for case in cases}
        completed = 0
        for future in as_completed(futures):
            query_id, answer, error = future.result()
            case = next(item for item in cases if item["id"] == query_id)
            case["response"] = answer
            if error is not None:
                case["response_error"] = error
            completed += 1
            print(f"  responses {completed}/{len(cases)}", flush=True)

    samples = [
        SingleTurnSample(
            user_input=item["user_input"],
            retrieved_contexts=item["retrieved_contexts"],
            reference_contexts=item["reference_contexts"],
            retrieved_context_ids=item["retrieved_context_ids"],
            reference_context_ids=item["reference_context_ids"],
            response=item["response"],
            reference=item["reference"],
        )
        for item in cases
    ]
    # Ragas 0.4.3 collection metrics use direct .ascore(); retain this dataset for
    # native schema validation and keep the same samples as the scoring source.
    evaluation_dataset = EvaluationDataset(samples=samples)
    evaluation_dataset.validate_samples(evaluation_dataset.samples)
    return cases


def flatten_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return "" if content is None else str(content)


def format_messages(messages: list[dict[str, Any]]) -> str:
    formatted = []
    for message in messages:
        role = message.get("role", "unknown")
        content = flatten_message_content(message.get("content"))
        tool_calls = message.get("tool_calls")
        if tool_calls:
            content += "\n" + json.dumps(tool_calls, ensure_ascii=False)
        formatted.append(f"[{role}]\n{content}")
    return "\n\n".join(formatted)


def extract_json_object(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
            return json.dumps(value, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    return cleaned


class OpenAIFacade:
    def __init__(self, backend: BackendClient, provider: dict[str, Any]):
        self.backend = backend
        self.provider = provider
        self.request_count = 0
        self.lock = threading.Lock()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}/v1"

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _next_request(self) -> None:
        with self.lock:
            self.request_count += 1

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        facade = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *args: Any) -> None:
                return

            def send_json(self, status: int, payload: dict[str, Any]) -> None:
                encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def do_POST(self) -> None:  # noqa: N802
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    path = urlparse(self.path).path
                    facade._next_request()
                    if path.endswith("/embeddings"):
                        self.handle_embeddings(payload)
                    elif path.endswith("/chat/completions"):
                        self.handle_chat(payload)
                    else:
                        self.send_json(404, {"error": {"message": "unknown endpoint"}})
                except Exception as error:  # noqa: BLE001
                    self.send_json(
                        500,
                        {"error": {"message": f"local facade failed: {error}"}},
                    )

            def handle_embeddings(self, payload: dict[str, Any]) -> None:
                inputs = payload.get("input", [])
                texts = [inputs] if isinstance(inputs, str) else list(inputs)
                vectors = facade.backend.embed(texts)
                self.send_json(
                    200,
                    {
                        "object": "list",
                        "model": facade.provider["embedding_model"],
                        "data": [
                            {"object": "embedding", "index": index, "embedding": vector}
                            for index, vector in enumerate(vectors)
                        ],
                        "usage": {"prompt_tokens": 0, "total_tokens": 0},
                    },
                )

            def handle_chat(self, payload: dict[str, Any]) -> None:
                messages = payload.get("messages", [])
                tools = payload.get("tools") or []
                conversation = format_messages(messages)
                if tools:
                    function = tools[0]["function"]
                    schema = json.dumps(
                        function.get("parameters", {}), ensure_ascii=False, indent=2
                    )
                    system = (
                        "You are a strict structured-output generator. Return exactly one "
                        "valid JSON object matching the supplied JSON Schema. Do not use "
                        "Markdown fences or add commentary."
                    )
                    user = (
                        f"Evaluator conversation:\n{conversation}\n\n"
                        f"Function: {function.get('name')}\n"
                        f"Description: {function.get('description', '')}\n"
                        f"JSON Schema:\n{schema}\n\nReturn the JSON object now."
                    )
                    raw = facade.backend.generate(system, user)
                    arguments = extract_json_object(raw)
                    message = {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{uuid.uuid4().hex[:20]}",
                                "type": "function",
                                "function": {
                                    "name": function["name"],
                                    "arguments": arguments,
                                },
                            }
                        ],
                    }
                    finish_reason = "tool_calls"
                else:
                    content = facade.backend.generate("", conversation)
                    message = {"role": "assistant", "content": content}
                    finish_reason = "stop"

                self.send_json(
                    200,
                    {
                        "id": f"chatcmpl-{uuid.uuid4().hex}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": facade.provider["chat_model"],
                        "choices": [
                            {
                                "index": 0,
                                "message": message,
                                "finish_reason": finish_reason,
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                        },
                    },
                )

        return Handler


async def evaluate_cases(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    facade: OpenAIFacade,
    provider: dict[str, Any],
) -> list[dict[str, Any]]:
    openai_client = AsyncOpenAI(
        api_key="local-ragas-benchmark",
        base_url=facade.base_url,
        timeout=180.0,
        max_retries=1,
    )
    fingerprint = hashlib.sha256()
    for path in [args.dataset, args.references, args.rewrite_result]:
        if path is not None:
            fingerprint.update(path.read_bytes())
    for path in [
        REPO_ROOT / "app" / "src" / "main" / "resources" / "prompts" / "knowledgebase-query-system.st",
        REPO_ROOT / "app" / "src" / "main" / "resources" / "prompts" / "knowledgebase-query-user.st",
    ]:
        fingerprint.update(path.read_bytes())
    fingerprint.update(json.dumps(provider, sort_keys=True).encode("utf-8"))
    fingerprint.update(str(args.top_k).encode("ascii"))
    cache = DiskCacheBackend(
        cache_dir=str(HERE / ".cache" / fingerprint.hexdigest()[:16])
    )
    judge = llm_factory(
        provider["chat_model"],
        provider="openai",
        client=openai_client,
        adapter="instructor",
        cache=cache,
        temperature=0,
    )
    embeddings = embedding_factory(
        provider="openai",
        model=provider["embedding_model"],
        client=openai_client,
        interface="modern",
        cache=cache,
    )
    metrics = {
        "faithfulness": Faithfulness(llm=judge),
        "factual_correctness": FactualCorrectness(llm=judge, mode="f1"),
        "context_recall": ContextRecall(llm=judge),
        "context_precision": ContextPrecisionWithReference(llm=judge),
        "answer_relevancy": AnswerRelevancy(
            llm=judge,
            embeddings=embeddings,
            strictness=3,
        ),
    }
    semaphore = asyncio.Semaphore(args.max_workers)
    progress_lock = asyncio.Lock()
    for case in cases:
        if case.get("response_error"):
            case.setdefault("metrics", {}).update(
                {
                    name: {
                        "value": None,
                        "reason": None,
                        "elapsed_ms": None,
                        "error": f"response_error: {case['response_error']}",
                    }
                    for name in metrics
                }
            )

    scheduled = [
        (case, metric_name)
        for case in cases
        for metric_name in metrics
        if not case.get("response_error")
        and (
            not args.retry_errors
        or metric_name not in case.get("metrics", {})
        or case["metrics"][metric_name].get("value") is None
        or case["metrics"][metric_name].get("error") is not None
        )
    ]
    completed = 0
    total = len(scheduled)
    if total == 0:
        await openai_client.close()
        return cases

    async def score(case: dict[str, Any], metric_name: str) -> dict[str, Any]:
        nonlocal completed
        metric = metrics[metric_name]
        started: float | None = None
        try:
            async with semaphore:
                started = time.perf_counter()
                if metric_name == "faithfulness":
                    result = await asyncio.wait_for(
                        metric.ascore(
                            user_input=case["user_input"],
                            response=case["response"],
                            retrieved_contexts=case["retrieved_contexts"],
                        ),
                        timeout=args.metric_timeout,
                    )
                elif metric_name == "factual_correctness":
                    result = await asyncio.wait_for(
                        metric.ascore(
                            response=case["response"],
                            reference=case["reference"],
                        ),
                        timeout=args.metric_timeout,
                    )
                elif metric_name == "context_recall":
                    result = await asyncio.wait_for(
                        metric.ascore(
                            user_input=case["user_input"],
                            retrieved_contexts=case["retrieved_contexts"],
                            reference=case["reference"],
                        ),
                        timeout=args.metric_timeout,
                    )
                elif metric_name == "context_precision":
                    result = await asyncio.wait_for(
                        metric.ascore(
                            user_input=case["user_input"],
                            reference=case["reference"],
                            retrieved_contexts=case["retrieved_contexts"],
                        ),
                        timeout=args.metric_timeout,
                    )
                else:
                    result = await asyncio.wait_for(
                        metric.ascore(
                            user_input=case["user_input"],
                            response=case["response"],
                        ),
                        timeout=args.metric_timeout,
                    )
            record = {
                "value": float(result.value),
                "reason": result.reason,
                "elapsed_ms": (time.perf_counter() - started) * 1000,
                "error": None,
            }
        except Exception as error:  # noqa: BLE001
            record = {
                "value": None,
                "reason": None,
                "elapsed_ms": (
                    (time.perf_counter() - started) * 1000
                    if started is not None
                    else None
                ),
                "error": f"{type(error).__name__}: {error}",
            }
        async with progress_lock:
            completed += 1
            print(f"  Ragas metrics {completed}/{total}", flush=True)
        return {"id": case["id"], "metric": metric_name, "result": record}

    tasks = [score(case, metric_name) for case, metric_name in scheduled]
    scored = await asyncio.gather(*tasks)
    await openai_client.close()

    by_id: dict[str, dict[str, Any]] = {case["id"]: {} for case in cases}
    for item in scored:
        by_id[item["id"]][item["metric"]] = item["result"]
    for case in cases:
        case.setdefault("metrics", {}).update(by_id[case["id"]])
    return cases


def aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = list(cases[0]["metrics"])
    metrics: dict[str, Any] = {}
    for name in metric_names:
        values = [
            item["metrics"][name]["value"]
            for item in cases
            if item["metrics"][name]["value"] is not None
        ]
        metrics[name] = {
            "mean": statistics.fmean(values) if values else None,
            "valid": len(values),
            "total": len(cases),
        }

    groups: dict[str, Any] = {}
    for group in sorted({item["group"] for item in cases}):
        group_cases = [item for item in cases if item["group"] == group]
        groups[group] = {}
        for name in metric_names:
            values = [
                item["metrics"][name]["value"]
                for item in group_cases
                if item["metrics"][name]["value"] is not None
            ]
            groups[group][name] = {
                "mean": statistics.fmean(values) if values else None,
                "valid": len(values),
                "total": len(group_cases),
            }
    return {"metrics": metrics, "groups": groups}


def format_score(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def write_results(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    provider: dict[str, Any],
    aggregate_result: dict[str, Any],
    facade_requests: int,
    started: float,
) -> tuple[Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    base = args.output_dir / timestamp
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "framework": {
            "name": "ragas",
            "version": importlib.metadata.version("ragas"),
            "langchain_community": importlib.metadata.version("langchain-community"),
        },
        "models": provider,
        "dataset": {
            "queries": len(cases),
            "top_k": args.top_k,
            "rewrite_source": str(args.rewrite_result.resolve()),
        },
        "aggregate": aggregate_result,
        "facade_requests": facade_requests,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "cases": cases,
    }

    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")
    markdown_path = base.with_suffix(".md")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    metric_names = list(aggregate_result["metrics"])
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            ["id", "group", "question", "retrieval_query", *metric_names]
        )
        for case in cases:
            writer.writerow(
                [
                    case["id"],
                    case["group"],
                    case["question"],
                    case["retrieval_query"],
                    *[case["metrics"][name]["value"] for name in metric_names],
                ]
            )

    metric_rows = "\n".join(
        f"| {name} | {format_score(data['mean'])} | {data['valid']}/{data['total']} |"
        for name, data in aggregate_result["metrics"].items()
    )
    group_rows = "\n".join(
        "| "
        + group
        + " | "
        + " | ".join(
            f"{format_score(values[name]['mean'])} ({values[name]['valid']}/{values[name]['total']})"
            for name in metric_names
        )
        + " |"
        for group, values in aggregate_result["groups"].items()
    )
    markdown = (
        "# Ragas Evaluation\n\n"
        f"Generated: {payload['generated_at']}\n\n"
        f"Framework: Ragas {payload['framework']['version']}\n\n"
        f"Models: {provider['chat_model']} / {provider['embedding_model']} "
        f"({provider['embedding_dimensions']} dimensions)\n\n"
        f"Dataset: {len(cases)} queries, top-{args.top_k} exact-cosine retrieval with "
        "frozen query rewrites.\n\n"
        "## Aggregate\n\n"
        "| Metric | Mean | Valid |\n"
        "|---|---:|---:|\n"
        f"{metric_rows}\n\n"
        "## By Query Group\n\n"
        "| Group | " + " | ".join(metric_names) + " |\n"
        "|---|" + "---:|" * len(metric_names) + "\n"
        f"{group_rows}\n\n"
        "## Scope\n\n"
        "Ragas scores answer grounding, factual agreement, context quality, and answer "
        "relevancy. This run uses a synthetic labelled corpus and in-memory exact cosine "
        "retrieval; it is not a pgvector/HNSW production benchmark. The same configured "
        "chat model generates answers and acts as evaluator, which can introduce judge bias.\n"
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, csv_path, markdown_path


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.max_workers < 1:
        raise ValueError("--max-workers must be positive")
    if args.retry_errors and args.resume_result is None:
        raise ValueError("--retry-errors requires --resume-result")

    started = time.perf_counter()
    backend = BackendClient(args.backend_url)
    facade: OpenAIFacade | None = None
    try:
        provider = backend.provider_info()
        print(
            f"Ragas 0.4.3; chat={provider['chat_model']}; "
            f"embedding={provider['embedding_model']}"
        )
        if args.resume_result is not None:
            previous = json.loads(args.resume_result.read_text(encoding="utf-8"))
            previous_models = previous["models"]
            if previous_models != provider:
                raise ValueError(
                    "Current provider configuration differs from the resumed result"
                )
            if args.rewrite_result is None:
                args.rewrite_result = Path(previous["dataset"]["rewrite_source"])
            cases = previous["cases"]
            print(f"Reusing {len(cases)} collected cases from {args.resume_result}")
        else:
            cases = collect_cases(args, backend, provider)
        facade = OpenAIFacade(backend, provider)
        facade.start()
        print(f"Local OpenAI facade: {facade.base_url}")
        cases = asyncio.run(evaluate_cases(args, cases, facade, provider))
        aggregate_result = aggregate(cases)
        paths = write_results(
            args,
            cases,
            provider,
            aggregate_result,
            facade.request_count,
            started,
        )
        print(json.dumps(aggregate_result, ensure_ascii=False, indent=2))
        print("Results:")
        for path in paths:
            print(f"  {path}")
    finally:
        if facade is not None:
            facade.close()
        backend.close()


if __name__ == "__main__":
    main()
