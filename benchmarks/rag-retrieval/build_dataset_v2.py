from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from pypdf import PdfReader


TOPIC_RULES = [
    ("rag", ("RAG", "向量", "PGvector", "pgvector", "检索", "知识库", "Embedding")),
    ("llm_prompt", ("大模型", "Prompt", "结构化", "Token", "Temperature", "Ollama", "API")),
    ("spring_ai", ("Spring Al", "Spring AI", "SpringAl", "模拟面试", "面试")),
    ("postgresql", ("PostgreSQL", "PGvector", "数据库")),
    ("redis", ("Redis", "Stream", "Lua", "限流", "缓存")),
    ("storage", ("RustFs", "RustFS", "S3", "对象存储", "文件上传")),
    ("document_processing", ("Tika", "PDF", "文件", "解析", "iText")),
    ("java_spring", ("Java", "Spring Boot", "Mapstruct", "MapStruct")),
    ("delivery", ("Docker", "环境搭建", "项目启动", "部署")),
    ("interview", ("面试题", "面试官", "简历", "项目经历")),
]

NO_ANSWER_QUERIES = [
    "如何用 Kubernetes Operator 自动修复跨区域 Cassandra 分片？",
    "如何在 Flink 中实现 exactly-once 的实时风控窗口？",
    "如何为 TensorFlow 训练任务设计 GPU 多租户调度？",
    "如何用 ClickHouse 设计十亿级广告实时归因系统？",
    "如何实现 Linux 内核 eBPF 网络观测探针？",
    "如何用 Rust Tokio 编写高性能 QUIC 代理？",
    "如何为 Kafka 设计跨集群 MirrorMaker2 容灾拓扑？",
    "如何在 Elasticsearch 中实现向量与地理围栏联合查询？",
    "如何设计 OAuth2 多租户统一身份认证中心？",
    "如何用 Prometheus 计算 Kubernetes SLO burn rate？",
    "如何实现多模态图像分割模型的 LoRA 微调？",
    "如何为 MongoDB 分片集群设计在线 schema migration？",
    "如何用 Terraform 管理 AWS EKS 蓝绿发布？",
    "如何构建基于 WebRTC SFU 的多人视频会议？",
    "如何用 Ray 进行大模型分布式推理并做 KV Cache 复用？",
    "如何设计 Stripe 支付的幂等与对账系统？",
    "如何用 OpenTelemetry 追踪跨服务异步事务？",
    "如何实现 gRPC 双向流的连接迁移？",
    "如何设计 IoT MQTT 设备影子与离线消息？",
    "如何在 Snowflake 中实现增量数据质量校验？",
    "如何用 Rust 编写高性能 PDF OCR 管道？",
    "如何设计区块链智能合约的重入攻击防护？",
    "如何在 Unity 中实现实时多人游戏状态同步？",
    "如何训练一个中文语音识别端到端模型？",
    "如何为 Redis Cluster 设计跨机房自动故障转移？",
    "如何使用 Neo4j 构建知识图谱路径推理？",
    "如何在 PostGIS 中做亿级轨迹数据的空间索引？",
    "如何使用 Apache Beam 统一批流处理作业？",
    "如何为 Java Netty 服务设计零拷贝文件传输？",
    "如何实现可验证凭证 DID 的签发和吊销？",
]


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def topic_for(name: str, text: str) -> str:
    haystack = f"{name}\n{text[:1200]}"
    for topic, keywords in TOPIC_RULES:
        if any(keyword in haystack for keyword in keywords):
            return topic
    return "general"


def split_sentences(text: str) -> list[str]:
    pieces = re.split(r"(?<=[。！？；!?;])\s*|\n+", text)
    return [piece.strip() for piece in pieces if len(piece.strip()) >= 18]


def chunk_pages(pages: list[tuple[int, str]], target: int = 1450, maximum: int = 2100) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    buffer: list[str] = []
    start_page: int | None = None
    end_page: int | None = None

    def flush() -> None:
        nonlocal buffer, start_page, end_page
        text = normalize_text("\n".join(buffer))
        if len(text) >= 300:
            chunks.append({"text": text, "page_start": start_page, "page_end": end_page})
        buffer = []
        start_page = None
        end_page = None

    for page_number, raw_text in pages:
        text = normalize_text(raw_text)
        if len(text) < 80:
            continue
        if start_page is None:
            start_page = page_number
        end_page = page_number
        if len(text) > maximum:
            for sentence in split_sentences(text):
                if buffer and len("\n".join(buffer)) + len(sentence) > target:
                    flush()
                    start_page = page_number
                if start_page is None:
                    start_page = page_number
                buffer.append(sentence)
            continue
        if buffer and len("\n".join(buffer)) + len(text) > target:
            flush()
            start_page = page_number
        buffer.append(text)
        if len("\n".join(buffer)) >= target:
            flush()
    flush()
    return chunks


def extract_sources(source_dir: Path) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    for pdf_path in sorted(source_dir.glob("*.pdf")):
        if "2027桐哥讲公专宝典" in pdf_path.name:
            continue
        raw_bytes = pdf_path.read_bytes()
        checksum = hashlib.sha256(raw_bytes).hexdigest()
        try:
            reader = PdfReader(str(pdf_path))
            pages = [(index + 1, page.extract_text() or "") for index, page in enumerate(reader.pages)]
        except Exception as error:  # noqa: BLE001
            print(f"skip {pdf_path.name}: {error}")
            continue
        text_chars = sum(len(text) for _, text in pages)
        if text_chars < 1000:
            print(f"skip low-text PDF {pdf_path.name}: {text_chars} chars")
            continue
        for index, chunk in enumerate(chunk_pages(pages), start=1):
            topic = topic_for(pdf_path.name, chunk["text"])
            sources.append(
                {
                    "id": f"pdf-{checksum[:10]}-{index:03d}",
                    "title": f"{pdf_path.stem} / 片段 {index}",
                    "text": chunk["text"],
                    "metadata": {
                        "topic": topic,
                        "source_file": pdf_path.name,
                        # Keep provenance portable and avoid leaking a local filesystem path.
                        # The source filename and SHA-256 are sufficient to identify a
                        # locally retained, authorized source document.
                        "source_section": f"pages {chunk['page_start']}-{chunk['page_end']}",
                        "page_start": chunk["page_start"],
                        "page_end": chunk["page_end"],
                        "source_sha256": checksum,
                        "license": "user-provided-local-material",
                    },
                }
            )
    return sources


def keywords(text: str) -> set[str]:
    latin = set(re.findall(r"[A-Za-z][A-Za-z0-9+#.-]{2,}", text.lower()))
    chinese = set(re.findall(r"[\u4e00-\u9fff]{2,6}", text))
    return latin | chinese


def first_claim(text: str) -> str:
    sentences = split_sentences(text)
    claim = sentences[0] if sentences else text
    return claim[:150].rstrip("，。；")


def make_gold_answer(text: str) -> str:
    claims = split_sentences(text)[:3]
    return "".join(claims)[:500]


def choose_negatives(
    docs: list[dict[str, Any]], relevant: set[str], topic: str, rng: random.Random
) -> list[str]:
    same_topic = [doc["id"] for doc in docs if doc["metadata"]["topic"] == topic and doc["id"] not in relevant]
    other = [doc["id"] for doc in docs if doc["id"] not in relevant and doc["id"] not in same_topic]
    rng.shuffle(same_topic)
    rng.shuffle(other)
    return (same_topic[:5] + other[:5])[:10]


def build_queries(docs: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        by_topic[doc["metadata"]["topic"]].append(doc)

    queries: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        topic = doc["metadata"]["topic"]
        claim = first_claim(doc["text"])
        if index % 4 == 0:
            question = f"{doc['title']}主要解决什么问题？"
            group = "direct"
        elif index % 4 == 1:
            question = f"在{topic}场景中，{claim}的关键做法是什么？"
            group = "paraphrase"
        elif index % 4 == 2:
            question = f"如果要落地{doc['title']}，应该关注哪些实现要点？"
            group = "code_config"
        else:
            question = f"为什么{claim[:75]}？"
            group = "direct"
        relevant = {doc["id"]}
        queries.append(
            {
                "id": f"v2-q{len(queries)+1:03d}",
                "group": group,
                "topic": topic,
                "question": question,
                "history": "",
                "answerable": True,
                "expectedBehavior": "answer",
                "goldAnswer": make_gold_answer(doc["text"]),
                "qrels": {doc["id"]: 2},
                "relevant": [doc["id"]],
                "hardNegativeIds": choose_negatives(docs, relevant, topic, rng),
                "difficulty": "medium" if group != "direct" else "easy",
                "split": "test",
            }
        )

    # Add paraphrases for a balanced subset of documents.
    for doc in docs[::5]:
        topic = doc["metadata"]["topic"]
        terms = sorted(keywords(doc["title"] + " " + doc["text"]))[:5]
        relevant = {doc["id"]}
        queries.append(
            {
                "id": f"v2-q{len(queries)+1:03d}",
                "group": "paraphrase",
                "topic": topic,
                "question": f"{topic}里涉及{'、'.join(terms)}时，通常怎样设计？",
                "history": "",
                "answerable": True,
                "expectedBehavior": "answer",
                "goldAnswer": make_gold_answer(doc["text"]),
                "qrels": {doc["id"]: 2},
                "relevant": [doc["id"]],
                "hardNegativeIds": choose_negatives(docs, relevant, topic, rng),
                "difficulty": "hard",
                "split": "test",
            }
        )

    # Multi-hop queries combine two related chunks and grade both as relevant.
    topic_groups = [group for group in by_topic.values() if len(group) >= 2]
    for pair_index in range(min(30, len(topic_groups) * 3)):
        group = topic_groups[pair_index % len(topic_groups)]
        first = group[(pair_index * 2) % len(group)]
        second = group[(pair_index * 2 + 1) % len(group)]
        relevant = {first["id"], second["id"]}
        topic = first["metadata"]["topic"]
        queries.append(
            {
                "id": f"v2-q{len(queries)+1:03d}",
                "group": "multi_hop",
                "topic": topic,
                "question": f"在{topic}方案中，如何同时考虑{first['title'].split(' / ')[0]}与{second['title'].split(' / ')[0]}？",
                "history": "",
                "answerable": True,
                "expectedBehavior": "answer",
                "goldAnswer": make_gold_answer(first["text"] + " " + second["text"]),
                "qrels": {first["id"]: 2, second["id"]: 2},
                "relevant": sorted(relevant),
                "hardNegativeIds": choose_negatives(docs, relevant, topic, rng),
                "difficulty": "hard",
                "split": "test",
            }
        )

    # Contextual follow-ups use the first claim as the prior turn.
    for doc in docs[::7][:30]:
        topic = doc["metadata"]["topic"]
        relevant = {doc["id"]}
        queries.append(
            {
                "id": f"v2-q{len(queries)+1:03d}",
                "group": "contextual",
                "topic": topic,
                "question": "那失败时应该怎么办？",
                "history": f"用户: {first_claim(doc['text'])}\n助手: 已经采用该方案。",
                "answerable": True,
                "expectedBehavior": "answer",
                "goldAnswer": make_gold_answer(doc["text"]),
                "qrels": {doc["id"]: 2},
                "relevant": [doc["id"]],
                "hardNegativeIds": choose_negatives(docs, relevant, topic, rng),
                "difficulty": "hard",
                "split": "test",
            }
        )

    for question in NO_ANSWER_QUERIES:
        topic = "out_of_corpus"
        queries.append(
            {
                "id": f"v2-q{len(queries)+1:03d}",
                "group": "no_answer",
                "topic": topic,
                "question": question,
                "history": "",
                "answerable": False,
                "expectedBehavior": "abstain",
                "goldAnswer": "知识库未覆盖该问题，应明确拒答。",
                "qrels": {},
                "relevant": [],
                "hardNegativeIds": choose_negatives(docs, set(), "general", rng),
                "difficulty": "hard",
                "split": "test",
            }
        )
    return queries


def assign_splits(queries: list[dict[str, Any]], seed: int, dev_fraction: float = 0.2) -> None:
    """Create a deterministic, group-stratified dev/test split in place."""
    rng = random.Random(seed + 1)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        by_group[query["group"]].append(query)
    for group_queries in by_group.values():
        rng.shuffle(group_queries)
        dev_count = max(1, round(len(group_queries) * dev_fraction))
        for index, query in enumerate(group_queries):
            query["split"] = "dev" if index < dev_count else "test"


def select_documents(documents: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    if len(documents) <= target:
        return documents
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents:
        by_source[document["metadata"]["source_file"]].append(document)
    selected: list[dict[str, Any]] = []
    sources = sorted(by_source)
    cursor = 0
    while len(selected) < target:
        source = sources[cursor % len(sources)]
        source_docs = by_source[source]
        if source_docs:
            selected.append(source_docs.pop(0))
        cursor += 1
        if cursor > target * len(sources) * 2:
            break
    return selected


def limit_queries(queries: list[dict[str, Any]], target: int | None) -> list[dict[str, Any]]:
    if target is None or len(queries) <= target:
        return queries
    if target < 1:
        raise ValueError("--target-queries must be positive")
    no_answer = [query for query in queries if not query.get("answerable")]
    answerable = [query for query in queries if query.get("answerable")]
    no_answer_count = min(len(no_answer), max(1, target // 10))
    selected = answerable[: max(0, target - no_answer_count)] + no_answer[:no_answer_count]
    return selected[:target]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing PDFs you are authorized to process.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "v2")
    parser.add_argument("--target-documents", type=int, default=200)
    parser.add_argument("--target-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260819)
    args = parser.parse_args()

    documents = select_documents(extract_sources(args.source_dir), args.target_documents)
    if len(documents) < 150:
        raise RuntimeError(f"Only {len(documents)} text chunks found; need at least 150")
    queries = limit_queries(build_queries(documents, args.seed), args.target_queries)
    assign_splits(queries, args.seed)
    for query in queries:
        query["qrels"] = {doc_id: int(grade) for doc_id, grade in query["qrels"].items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "corpus.jsonl"
    queries_path = args.output_dir / "queries.jsonl"
    qrels_path = args.output_dir / "qrels.tsv"
    dataset_path = args.output_dir / "dataset.json"
    manifest_path = args.output_dir / "manifest.json"
    corpus_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in documents) + "\n", encoding="utf-8")
    queries_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in queries) + "\n", encoding="utf-8")
    # TREC/BEIR qrels are whitespace-delimited records without a visible header.
    # A header makes ir-measures/read_trec_qrels interpret the word "grade" as an int.
    qrels_lines: list[str] = []
    for query in queries:
        for doc_id, grade in query["qrels"].items():
            qrels_lines.append(f"{query['id']}\t0\t{doc_id}\t{grade}")
    qrels_path.write_text("\n".join(qrels_lines) + "\n", encoding="utf-8")
    dataset = {
        "name": "agentforge-interview-pdf-rag-v2",
        "version": "v2.0.0",
        "description": "从运行时提供且已获授权的 PDF 提取的可追溯 RAG 检索评测集。",
        "seed": args.seed,
        "documents": documents,
        "queries": queries,
    }
    dataset_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "dataset_version": "v2.0.0",
        "seed": args.seed,
        "source_dir": "provided-at-runtime",
        "document_count": len(documents),
        "query_count": len(queries),
        "query_groups": {group: sum(query["group"] == group for query in queries) for group in sorted({query["group"] for query in queries})},
        "answerable_count": sum(query["answerable"] for query in queries),
        "no_answer_count": sum(not query["answerable"] for query in queries),
        "split_counts": {
            split: sum(query["split"] == split for query in queries)
            for split in sorted({query["split"] for query in queries})
        },
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [corpus_path, queries_path, qrels_path, dataset_path]
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
