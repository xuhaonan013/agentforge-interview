# AgentForge Interview

AgentForge Interview 是一个面向个人求职者的 AI 面试工作台：上传简历后生成结构化分析，基于知识库进行检索增强问答，并提供文字面试、实时语音面试和可追踪的异步评估流程。

## 功能概览

- 简历解析、异步 AI 评估与结构化报告导出
- PDF/DOCX/TXT 知识库导入、向量化、BM25 + 向量混合检索与重排
- Skill 驱动的题目生成、追问和难度配置
- 文字面试与 WebSocket 语音面试（ASR/TTS）
- 可在页面中管理 OpenAI 兼容的 LLM/Embedding Provider
- Redis Stream 异步任务、SHA-256 文件去重、S3 兼容对象存储
- Spring Boot Actuator、Swagger/OpenAPI 与可选 benchmark 适配器

## 技术栈

Java 25、Spring Boot、Spring AI、PostgreSQL + pgvector、Redis、RustFS/MinIO、React、TypeScript、Vite、Docker Compose。

## 快速启动

### 1. 准备环境

- Docker Desktop（包含 Compose）
- JDK 25（仅本地运行后端需要）
- Node.js 20+ 与 pnpm 10+（仅本地运行前端需要）
- 一个 OpenAI 兼容的模型服务。默认示例使用阿里云百炼，也可以在页面中改为其他 Provider。

复制环境变量模板并填写密钥。密钥只通过环境变量或页面配置保存，不要提交 .env：

~~~powershell
Set-Location D:\workspace\agentforge-interview
Copy-Item .env.example .env
notepad .env
~~~

至少设置 AI_BAILIAN_API_KEY 与随机生成的 APP_AI_CONFIG_ENCRYPTION_KEY。生产环境请使用密钥管理服务，并替换 Compose 中的默认数据库、Redis 和对象存储凭据。

### 2. 一键运行

~~~powershell
docker compose up -d --build
~~~

打开 http://localhost 使用前端，后端接口和 Swagger 分别位于 http://localhost:8080 与 http://localhost:8080/swagger-ui.html。

停止服务（保留数据卷）：

~~~powershell
docker compose down
~~~

清理本地数据卷：

~~~powershell
docker compose down -v
~~~

### 3. 本地开发模式

先只启动依赖服务：

~~~powershell
docker compose -f docker-compose.dev.yml up -d
~~~

然后分别启动后端和前端：

~~~powershell
.\gradlew :app:bootRun
~~~

~~~powershell
Set-Location frontend
pnpm install
pnpm dev
~~~

本地开发前端默认访问 http://localhost:5173。Provider 配置文件默认写入 %USERPROFILE%\.agentforge-interview\，不会污染源码目录。

## 配置说明

主要配置位于 app/src/main/resources/application.yml，常用环境变量包括：

| 变量 | 用途 |
| --- | --- |
| AI_BAILIAN_API_KEY | 默认 DashScope 兼容接口密钥 |
| APP_AI_CONFIG_ENCRYPTION_KEY | Provider 密钥加密密钥 |
| POSTGRES_HOST/PORT/DB/USER/PASSWORD | PostgreSQL 连接 |
| REDIS_HOST/PORT | Redis 连接 |
| APP_STORAGE_ENDPOINT/ACCESS_KEY/SECRET_KEY/BUCKET | S3 兼容存储 |
| AI_MODEL | 默认聊天模型 |

应用默认数据库名为 agentforge_interview，对象存储桶名为 agentforge-interview。如需接入 OpenAI 或其他兼容服务，可在页面的 Provider 管理中填写 Base URL、模型和 Embedding 配置。

## 评测

benchmarks/ 提供可选的检索与生成评测脚本：

- rag-retrieval/：TREC/BEIR 风格的 Recall@K、MRR、nDCG、MAP、R-Precision、延迟分位数和拒答指标
- ragas/：Context Recall/Precision、Faithfulness、Answer Relevancy、Factual Correctness

公开副本不携带从个人 PDF 抽取的原始语料、真实对话或运行结果。请在本地准备获得授权的数据集，再按目录中的说明生成数据并运行评测。密钥通过环境变量传入，脚本不会把密钥写入结果文件。

仓库附带一个不含个人资料的 12 文档 synthetic smoke dataset，可先用它验证评测链路：

~~~powershell
uv run --with-requirements benchmarks\rag-retrieval\requirements.txt python benchmarks\rag-retrieval\benchmark_v2.py --dataset-dir benchmarks\rag-retrieval\sample --mode lexical --split test
~~~

## 代码结构

~~~text
app/
  src/main/java/com/agentforge/
    common/          通用配置、AI 调用、异常与评估
    infrastructure/  文件、Redis、S3、导出与映射
    modules/         resume、interview、knowledgebase、voiceinterview、llmprovider
frontend/            React + TypeScript 客户端
docker/              PostgreSQL 初始化脚本
benchmarks/          可选离线评测工具
~~~

## 开发与验证

~~~powershell
.\gradlew :app:compileJava --no-daemon
.\gradlew :app:test --no-daemon
Set-Location frontend
pnpm build
~~~

提交前请确认没有 .env、密钥、个人简历、PDF 原文或生成结果：

~~~powershell
git status --short
git diff --check
~~~

## 上游来源与许可证

本项目是基于 [Snailclimb/interview-guide](https://github.com/Snailclimb/interview-guide) 的 AGPL-3.0 修改版，保留原许可证和必要的归属说明。当前版本命名为 AgentForge Interview，并对包名、配置、运行编排、前端界面、评测工具和文档进行了修改；具体说明见 NOTICE.md。

AGPL-3.0 要求发布修改后的完整对应源代码；通过网络向用户提供本服务时，也应向用户提供相应源代码。部分依赖（例如 iText）有独立的许可证要求，请在部署和再分发前分别核对。

## 贡献

欢迎提交 Issue 或 Pull Request。请不要上传密钥、个人简历、未获授权的文档内容或运行日志。
