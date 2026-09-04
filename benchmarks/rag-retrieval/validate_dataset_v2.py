from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import ir_measures


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate v2 corpus, queries and TREC qrels.")
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent / "v2")
    args = parser.parse_args()
    root = args.dataset_dir
    dataset = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    corpus = dataset["documents"]
    queries = dataset["queries"]
    document_ids = {document["id"] for document in corpus}
    errors: list[str] = []
    duplicate_ids = len(corpus) - len(document_ids)
    duplicate_texts = len(corpus) - len({document["text"] for document in corpus})
    if duplicate_ids:
        errors.append(f"duplicate document ids: {duplicate_ids}")
    if duplicate_texts:
        errors.append(f"duplicate document texts: {duplicate_texts}")

    query_ids: set[str] = set()
    qrels_by_query: dict[str, dict[str, int]] = {}
    for query in queries:
        query_id = query["id"]
        if query_id in query_ids:
            errors.append(f"duplicate query id: {query_id}")
        query_ids.add(query_id)
        qrels = {str(doc_id): int(grade) for doc_id, grade in query.get("qrels", {}).items()}
        qrels_by_query[query_id] = qrels
        if query.get("answerable") and not qrels:
            errors.append(f"answerable query without qrels: {query_id}")
        if not query.get("answerable") and qrels:
            errors.append(f"no-answer query has qrels: {query_id}")
        unknown = set(qrels) - document_ids
        if unknown:
            errors.append(f"unknown qrels in {query_id}: {sorted(unknown)}")
        if len(query.get("hardNegativeIds", [])) < 3:
            errors.append(f"too few hard negatives: {query_id}")
        if set(query.get("hardNegativeIds", [])) & set(qrels):
            errors.append(f"hard negative is relevant: {query_id}")

    try:
        parsed_qrels = list(ir_measures.read_trec_qrels(str(root / "qrels.tsv")))
    except Exception as error:  # noqa: BLE001
        errors.append(f"TREC qrels parse failed: {error}")
        parsed_qrels = []
    source_files = sorted({document["metadata"]["source_file"] for document in corpus})
    # Public datasets intentionally omit absolute source paths. When a local
    # build includes an opt-in source_path field, validate those files too.
    source_paths = [
        Path(document["metadata"]["source_path"])
        for document in corpus
        if document.get("metadata", {}).get("source_path")
    ]
    missing_sources = [str(path) for path in source_paths if not path.exists()]
    if missing_sources:
        errors.append(f"missing source files: {missing_sources}")

    report: dict[str, Any] = {
        "dataset_version": dataset.get("version"),
        "documents": len(corpus),
        "queries": len(queries),
        "answerable": sum(bool(query.get("answerable")) for query in queries),
        "no_answer": sum(not query.get("answerable") for query in queries),
        "qrels_records": len(parsed_qrels),
        "duplicate_document_ids": duplicate_ids,
        "duplicate_document_texts": duplicate_texts,
        "source_file_count": len(source_files),
        "topic_counts": dict(Counter(document["metadata"].get("topic", "unknown") for document in corpus)),
        "query_group_counts": dict(Counter(query.get("group", "unknown") for query in queries)),
        "split_counts": dict(Counter(query.get("split", "missing") for query in queries)),
        "file_sha256": {
            name: sha256(root / name)
            for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv", "dataset.json", "manifest.json")
            if (root / name).exists()
        },
        "errors": errors,
        "status": "ok" if not errors else "failed",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
