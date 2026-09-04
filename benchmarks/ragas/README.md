# Ragas Answer-Quality Evaluation

This directory provides an optional Ragas 0.4.3 evaluation path for generated answers. It reports:

- Faithfulness
- Factual Correctness
- Context Recall
- Context Precision with Reference
- Answer Relevancy

Retrieval metrics and answer-quality metrics are separate experiments. Keep the dataset, model names, judge model and run timestamp beside any reported score.

## Run locally

The application benchmark adapter is disabled by default. Enable it only for a local evaluation:

~~~powershell
$env:BENCHMARK_TOKEN = "replace-with-a-random-local-token"
$env:APP_BENCHMARK_TOKEN = $env:BENCHMARK_TOKEN
docker compose -f docker-compose.yml -f benchmarks/rag-retrieval/docker-compose.benchmark.yml up -d --build app
uv run benchmarks\ragas\evaluate.py --max-workers 3
docker compose up -d --force-recreate app
~~~

Run a smoke subset first:

~~~powershell
uv run benchmarks\ragas\evaluate.py --ids q01 q02 --max-workers 1
~~~

The evaluator reads the provider configuration through the local adapter. API keys stay in environment variables or the application's encrypted store and are never written to reports.

## Data and reproducibility

The default dataset path points to the local retrieval benchmark output. Generate it from documents you are licensed to process, then review references and retrieved contexts before evaluation. Generated JSON, CSV and Markdown reports are ignored by Git. Do not commit private resumes, source PDFs, conversations or credentials.

Ragas uses an LLM judge, so scores can vary by model and prompt. For a stronger claim, use an independent judge and report the number of evaluated samples, failed cases and confidence intervals.
