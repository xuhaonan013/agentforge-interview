# RAG Retrieval Benchmark

This directory contains reproducible evaluation utilities for the AgentForge Interview retrieval pipeline. It supports a dependency-light lexical baseline and an embedding-backed comparison.

## Metrics

The runner reports TREC/BEIR-style metrics through ir-measures:

- Precision/Recall and Hit at 1, 3, 5 and 10
- MRR@10, nDCG@10, MAP@10 and R-Precision
- Per-topic and per-query-group breakdowns
- Answerability AUROC/AUPRC and abstention precision/recall/F1
- No-answer false-accept rate and p50/p95/p99 latency

Use Recall@5, MRR@10 and nDCG@10 as the primary retrieval indicators. Report the dataset version, split and qrels definition beside every number.

## Build an authorized dataset

The dataset builder accepts a directory of PDFs that you are allowed to process. It records only filenames and SHA-256 checksums as provenance; it does not require committing the source PDFs.

~~~powershell
uv run --with pypdf python benchmarks\rag-retrieval\build_dataset_v2.py --source-dir D:\path\to\authorized-pdfs --output-dir benchmarks\rag-retrieval\v2 --target-documents 200 --target-queries 320
~~~

Review the generated labels and gold answers before using the result for a public claim. Keep the generated v2 files and reports outside version control unless the source material is licensed for redistribution.

## Run the lexical baseline

The repository includes a small synthetic sample for a dependency and metric smoke test:

~~~powershell
uv run --with-requirements benchmarks\rag-retrieval\requirements.txt python benchmarks\rag-retrieval\benchmark_v2.py --dataset-dir benchmarks\rag-retrieval\sample --mode lexical --split test
~~~

~~~powershell
uv run --with-requirements benchmarks\rag-retrieval\requirements.txt python benchmarks\rag-retrieval\benchmark_v2.py --mode lexical --split test
~~~

For threshold selection, calibrate on the dev split and evaluate on test:

~~~powershell
uv run --with-requirements benchmarks\rag-retrieval\requirements.txt python benchmarks\rag-retrieval\benchmark_v2.py --mode lexical --split test --calibrate-from-dev
~~~

## Run application embeddings

Enable the local benchmark adapter only for a local run:

~~~powershell
$env:BENCHMARK_TOKEN = "replace-with-a-random-local-token"
$env:APP_BENCHMARK_TOKEN = $env:BENCHMARK_TOKEN
docker compose -f docker-compose.yml -f benchmarks/rag-retrieval/docker-compose.benchmark.yml up -d --build app
$env:BENCHMARK_BACKEND_URL = "http://127.0.0.1:8080"
uv run --with-requirements benchmarks\rag-retrieval\requirements.txt python benchmarks\rag-retrieval\benchmark_v2.py --mode embedding --split test
~~~

Use --limit 1 --document-limit 1 for a provider smoke test. The adapter is bound to localhost and protected by the token; disable it after the run:

~~~powershell
docker compose up -d --force-recreate app
Remove-Item Env:BENCHMARK_BACKEND_URL -ErrorAction SilentlyContinue
~~~

The benchmark scripts read provider credentials from the environment or the application's encrypted configuration. They never write API keys to report files.

## Direct OpenAI-compatible mode

benchmark.mjs can call a compatible endpoint directly. Set OPENAI_API_KEY plus AI_MODEL and AI_EMBEDDING_MODEL, or set OPENAI_COMPATIBLE_BASE_URL and OPENAI_COMPATIBLE_API_KEY. Use only datasets for which you have redistribution rights.

Generated reports are intentionally ignored by Git.
