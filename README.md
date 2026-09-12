# AgentForge Interview

项目维护者：[@xuhaonan013](https://github.com/xuhaonan013)

面向简历分析、知识库问答和模拟面试的 AI Agent 应用，重点验证多模型调用、Skill/Tool Calling、RAG 检索、异步任务和实时语音链路的工程化实现。

## 项目亮点

- **Agent 技能系统架构**：实现多层级 Skills，支持预设 Skill、JD 解析自定义 Skill；通过依赖注入提升出题专业性，三级优先级分配覆盖核心知识点。
- **结构化输出与智能重试**：使用 Schema 校验约束模型输出，结合错误反馈自动修复并重试，接入 Micrometer 记录调用成功率、耗时和失败原因。
- **生产级异步任务架构**：使用 Redis Stream Consumer Group 解耦简历分析、知识库向量化和面试评估，支持批量拉取、ACK、失败重试、Pending 自动认领和状态恢复。
- **RAG 检索架构设计**：搭建 BM25 + pgvector 混合检索，使用 HNSW、倒排参数、Query Rewrite、Rerank、metadata 过滤和原问题回退，覆盖短问句及多跳查询。
- **实时语音面试架构**：打通 WebSocket + Qwen3 ASR/TTS 首包链路，支持语音片段防抖合并、虚拟线程和冷却窗口，文字与语音共用统一评估服务。
- **可复现评测**：提供 TREC/BEIR 风格的 Recall@K、MRR@10、nDCG@10、MAP、R-Precision、Abstention F1、AUROC 和延迟分位数评测脚本。

## 技术栈

Java 25、Spring Boot 4.1、Spring AI 2.0、Tool Calling/Advisor、PostgreSQL/pgvector、Redis Stream、DashScope、Reactor/SSE、WebSocket、Apache Tika、Docker、Python、React、TypeScript。

## 最小启动

准备 Docker Desktop，并复制环境变量模板：

~~~powershell
git clone https://github.com/xuhaonan013/agentforge-interview.git
Set-Location agentforge-interview
Copy-Item .env.example .env
notepad .env
~~~

至少配置：

- AI_BAILIAN_API_KEY：模型、Embedding、ASR 和 TTS 服务密钥
- APP_AI_CONFIG_ENCRYPTION_KEY：Provider 密钥加密密钥

启动完整服务：

~~~powershell
docker compose up -d --build
~~~

访问 <http://localhost>。停止服务：

~~~powershell
docker compose down
~~~

本地开发只启动依赖服务：

~~~powershell
docker compose -f docker-compose.dev.yml up -d
.\gradlew :app:bootRun
Set-Location frontend
pnpm install
pnpm dev
~~~

## 评测入口

仓库内的 benchmarks/ 只保留评测脚本和合成 smoke 数据，不包含个人 PDF、真实简历、对话记录或运行结果。使用已获授权的文档生成数据集后，可运行：

~~~powershell
uv run --with-requirements benchmarks\rag-retrieval\requirements.txt python benchmarks\rag-retrieval\benchmark_v2.py --mode lexical --split test
~~~

详细参数见 [benchmarks/rag-retrieval/README.md](benchmarks/rag-retrieval/README.md)。
