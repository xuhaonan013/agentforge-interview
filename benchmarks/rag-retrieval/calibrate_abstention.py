from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ir_measures

from benchmark_v2 import abstention_stats, best_abstention_threshold, load_dataset


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def top_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in ir_measures.read_trec_run(str(path)):
        score = float(item.score)
        # read_trec_run yields each query in rank order and exposes only
        # query_id/doc_id/score, so the first record is top-1.
        scores.setdefault(item.query_id, score)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate embedding abstention on dev and apply it to test.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "v2")
    parser.add_argument("--dev-run", type=Path, required=True)
    parser.add_argument("--test-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    dataset, _, queries = load_dataset(args.dataset_dir)
    by_id = {query["id"]: query for query in queries}
    dev_scores = top_scores(args.dev_run)
    test_scores = top_scores(args.test_run)
    dev_ids = [query["id"] for query in queries if query.get("split") == "dev"]
    test_ids = [query["id"] for query in queries if query.get("split") == "test"]
    missing_dev = set(dev_ids) - set(dev_scores)
    missing_test = set(test_ids) - set(test_scores)
    if missing_dev or missing_test:
        raise SystemExit(f"run is missing queries: dev={sorted(missing_dev)} test={sorted(missing_test)}")

    dev_labels = [bool(by_id[query_id]["answerable"]) for query_id in dev_ids]
    dev_top_scores = [dev_scores[query_id] for query_id in dev_ids]
    calibration = best_abstention_threshold(dev_labels, dev_top_scores)
    test_labels = [bool(by_id[query_id]["answerable"]) for query_id in test_ids]
    test_top_scores = [test_scores[query_id] for query_id in test_ids]
    test_stats = abstention_stats(test_labels, test_top_scores, float(calibration["threshold"]))
    output = args.output or args.dataset_dir / "results" / "embedding-abstention-calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ"),
        "dataset_version": dataset.get("version"),
        "dev_query_count": len(dev_ids),
        "test_query_count": len(test_ids),
        "dev_run": str(args.dev_run),
        "test_run": str(args.test_run),
        "dev_run_sha256": sha256(args.dev_run),
        "test_run_sha256": sha256(args.test_run),
        "dev_calibration": calibration,
        "test_abstention": test_stats,
        "interpretation": "Threshold selected on dev top-1 cosine score and applied once to test; no test threshold tuning.",
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = (
        "# Embedding abstention calibration\n\n"
        f"Dataset `{result['dataset_version']}`; dev {len(dev_ids)} queries, test {len(test_ids)} queries.\n\n"
        f"Threshold selected on dev: `{calibration['threshold']:.8f}` (dev F1 `{calibration['abstention_f1']:.4f}`).\n\n"
        "| Test measure | Value |\n|---|---:|\n"
        f"| Abstention precision | {test_stats['abstention_precision']:.4f} |\n"
        f"| Abstention recall | {test_stats['abstention_recall']:.4f} |\n"
        f"| Abstention F1 | {test_stats['abstention_f1']:.4f} |\n"
        f"| Abstention false-positive rate | {test_stats['false_positive_rate']:.4f} |\n"
        f"| No-answer false-accept rate | {test_stats['no_answer_false_accept_rate']:.4f} |\n\n"
        "> Threshold is calibrated on dev and applied to test. Retrieval quality metrics remain those in the original embedding run.\n"
    )
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(json.dumps({"json": str(output), "markdown": str(output.with_suffix('.md')), **test_stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
