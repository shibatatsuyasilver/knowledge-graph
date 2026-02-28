# Enterprise Gen-AI KG Prototype Workspace

This workspace uses:

- `uv` for Python dependency management
- MLX (`mlx_lm.server`) as default local LLM runtime
- Neo4j in Docker
- Shared LLM provider abstraction with `openai` default and `ollama` fallback

## Architecture

Three-layer Python backend + React frontend, all wired through Neo4j and a multi-provider LLM abstraction.

```
User (Browser)
  └── frontend/                    React + TypeScript UI
        └── /api proxy
              └── backend/         FastAPI (port 8000)
                    ├── api layer       routers: ingest, qa, llm_compat
                    ├── services layer  chunking, URL fetch, QA execution
                    └── llm_kg layer    entity extraction, NL→Cypher, LLM client
                          └── Neo4j     graph database (port 7687)
```

### Folder map

```
backend/
├── main.py                  FastAPI app assembly and router registration
├── logic.py                 Thin facade over services (backward-compat shim)
├── api/routers/
│   ├── ingest.py            Async endpoints: process_text / process_url / process_keyword
│   ├── qa.py                NL2Cypher query + general chat endpoints
│   └── llm_compat.py        OpenAI-compatible /v1/chat/completions passthrough
├── config/settings.py       Centralized env-var parsing (LLM, Neo4j, chunking, timeouts)
├── jobs/store.py            In-memory async job store with TTL
├── services/
│   ├── ingest/service.py    Text chunking, URL fetch (BeautifulSoup + DuckDuckGo)
│   └── qa/service.py        Cypher execution, LLM answer rewriting
├── llm_kg/                  Core intelligence layer
│   ├── llm_client.py        Multi-provider LLM abstraction (OpenAI-compat / Ollama / Gemini)
│   ├── kg_builder.py        Schema-constrained entity/relation extraction with JSON repair
│   ├── nl2cypher.py         Natural language → Cypher with repair loops and agentic mode
│   ├── e2e_runner.py        End-to-end pipeline: extract → Neo4j write → QA validation
│   ├── llm_api.py        Standalone lightweight HTTP service (GET /health, POST /api/chat)
│   ├── benchmark/           Evaluation framework (dataset builder, runner, scorer, reporter)
│   └── tests/               Unit tests for llm_kg module (mock Ollama + OpenAI servers)
├── tests/                   API and integration tests (test_main, test_logic, test_e2e_integration)
└── Dockerfile

frontend/src/
├── App.tsx                  Tab navigation: "Build KG" ↔ "Chat"
├── components/
│   ├── BuildKG.tsx          Ingestion UI (text / file / URL / keyword crawl tabs)
│   └── Chat.tsx             Chat UI with KG query mode and general chat mode
├── api.ts                   HTTP client + async job polling helpers
└── types.ts                 TypeScript interfaces matching all API contracts

demo/                        Standalone demo apps (kg_pipeline, llm_api)
scripts/                     start_local_with_gemini.sh — one-command local stack
docker-compose.yml           Services: backend :8000, frontend :8081, neo4j :7687
```

### Async ingest flow

All ingest operations (text / URL / keyword) run as background jobs:

```
POST /api/process_*_async/start     →  job_id
GET  /api/process_*_async/{job_id}  ←  frontend polls until done
```

Each job streams chunk-level status: `queued → processing → processed / skipped_by_limit / failed`.

### KG extraction schema

Fixed entity types: `Person · Organization · Location · Technology · Product · FiscalPeriod · FinancialMetric`

Fixed relation types: `FOUNDED_BY · CHAIRED_BY · HEADQUARTERED_IN · PRODUCES · SUPPLIES_TO · USES · COMPETES_WITH · HAS_FINANCIAL_METRIC · FOR_PERIOD`

Direction rules are enforced (e.g. `FOUNDED_BY` must be `Organization → Person`).

## Quick start

### 0) One-command local stack with Gemini key (recommended)

```bash
cd /Users/silver/Documents/鴻海
cp .env.local.example .env.local
# edit .env.local and fill GEMINI_API_KEY
bash scripts/start_local_with_gemini.sh
```

Notes:

- `scripts/start_local_with_gemini.sh` auto-loads `.env.local` and starts `neo4j + backend + frontend`.
- It also auto-handles common compose/container name conflicts for `genai-*` services.
- It validates `GEMINI_API_KEY` online before startup, so invalid keys fail fast.
- `.env.local` is ignored by `/Users/silver/Documents/鴻海/.gitignore` and should not be committed.

Important `.env.local` behavior:

- `docker compose` does **not** auto-read `.env.local` by default.
- If you start stack manually, use `docker compose --env-file .env.local ...` or run `bash scripts/start_local_with_gemini.sh`.
- Symptom of missing env injection: UI shows `Upstream service error: GEMINI_API_KEY is required when provider=gemini`.
- Quick check: `docker exec genai-backend sh -lc 'if [ -n "$GEMINI_API_KEY" ]; then echo ok; else echo missing; fi'`

### 1) Sync Python dependencies

```bash
cd /Users/silver/Documents/鴻海
uv sync --group backend --group llmkg --group demo --group dev
```

### 2) Start MLX server

```bash
uv run python -m mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit-DWQ-053125 \
  --host 0.0.0.0 \
  --port 8080 \
  --max-tokens 32768
```

### 3) Start Neo4j

```bash
docker compose up -d neo4j
```

### 4) Start backend API

```bash
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:8080/v1
export LLM_MODEL=mlx-community/Qwen3-8B-4bit-DWQ-053125
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 5) Run one-command real-model E2E

```bash
uv run bash backend/llm_kg/run_e2e_mlx_local.sh
```

## Environment variables (primary interface)

- `LLM_PROVIDER=openai|ollama`
- `LLM_MODEL=...`
- `EXTRACTION_MODEL=...` (optional; override model for entity/relation extraction)
- `NL2CYPHER_MODEL=...` (optional; override model for Cypher generation)
- `LLM_TIMEOUT_SECONDS=...`
- `LLM_TEMPERATURE=...`
- `LLM_MAX_TOKENS=...`
- `LLM_ERROR_DETAIL_MAX_CHARS=...` (`0` means no truncation; default `4000`)
- `INGEST_CHUNK_LIMIT=...` (optional; global default chunk cap, `0` means no limit)
- `EXTRACTION_ERROR_RAW_MAX_CHARS=...` (`0` means no truncation; default `4000`)
- `OPENAI_BASE_URL=http://localhost:8080/v1`
- `OPENAI_API_KEY=...` (optional for local MLX)
- `GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta`
- `GEMINI_API_KEY=...`
- `GEMINI_MODEL=gemini-3-pro-preview`
- `GEMINI_INPUT_TOKEN_LIMIT=1048576` (Gemini input token cap for chunking budget)
- `GEMINI_OUTPUT_TOKEN_LIMIT=65536` (Gemini `maxOutputTokens` cap)
- `GEMINI_TWO_PASS_EXTRACTION=1` (Gemini extraction runs Phase-1 entity inventory -> KG missing-entity prefill -> Phase-2 relation extraction)
- `CHUNK_SIZE_MODE=provider|token|char` (`provider` = Gemini uses token-based chunking)
- `CHUNK_SIZE_TOKENS=1048576` (token chunk size when token mode is enabled)
- `CHUNK_MIN_TOKENS=120`
- `KG_QA_USE_LLM=1` (default on; use LLM to rewrite KG answers)
- `KG_QA_MODEL=...` (optional override for KG answer rewriting)
- `KG_QA_MAX_TOKENS=1024`
- `KG_QA_TEMPERATURE=0.1`

Legacy compatibility:

- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`

Ollama dual-model example:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export EXTRACTION_MODEL=sam860/deepseek-r1-0528-qwen3:8b
export NL2CYPHER_MODEL=ministral-3:14b
```

Gemini extraction example:

```bash
export GEMINI_API_KEY=...
```

`docker-compose.yml` backend service defaults to `LLM_PROVIDER=ollama` with the two stage models above.

Chunk cap example (process only first 5 chunks per URL):

```json
{"keyword":"鴻海","max_results":10,"language":"zh-tw","chunk_limit":5,"extraction_provider":"gemini","extraction_model":"gemini-3-pro-preview"}
```

前端 `Build Knowledge Graph` 已支援 Extraction Provider/Model 選擇：

- `Ollama`：例如 `sam860/deepseek-r1-0528-qwen3:8b`
- `Gemini`：例如 `gemini-3-pro-preview`

`Build Knowledge Graph` 的 `Text / File / URL / Keyword Crawl` 目前都使用非同步任務（backend background job + 前端輪詢），處理中即可看到 chunk 狀態：

- `queued`
- `processing`
- `processed`
- `skipped_by_limit`
- `failed`

新增 async API routes（舊 sync routes 保留）：

- `POST /api/process_text_async/start`
- `GET /api/process_text_async/{job_id}`
- `POST /api/process_url_async/start`
- `GET /api/process_url_async/{job_id}`
- `POST /api/process_keyword_async/start`
- `GET /api/process_keyword_async/{job_id}`

請使用環境變數提供金鑰，不要把 API key 寫進程式碼。

## Ollama Thinking 模式

先檢查模型是否支援 thinking（看 `Capabilities` 是否包含 `thinking`）：

```bash
ollama show sam860/deepseek-r1-0528-qwen3:8b
ollama show deepseek-r1:8b
```

目前建議（本專案預設）：

- `sam860/deepseek-r1-0528-qwen3:8b` 不支援 thinking，請保持：
  - `OLLAMA_THINK=false`
  - `OLLAMA_THINK_JSON=false`
- 若切換到支援 thinking 的模型（例如 `deepseek-r1:8b`），才可開啟：
  - `OLLAMA_THINK=true`
  - `OLLAMA_THINK_JSON=true`

常見錯誤：

- `"MODEL_NAME" does not support thinking`
- 代表模型本身不支援，請關閉上面兩個變數，或改用支援 thinking 的模型。

正確開啟 thinking 範例（僅適用支援 thinking 的模型）：

```bash
curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:8b",
  "messages": [{"role":"user","content":"請先思考再回答"}],
  "think": true
}'
```

CLI 方式：

```bash
ollama run deepseek-r1:8b
# 進入互動後輸入
/set think true
```

## Benchmark (100 題 DeepSeek R1 vs Gemma3)

Benchmark config:

- `/Users/silver/Documents/鴻海/backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml`

Dataset build (30 graph seed + 70 gemini synth):

```bash
export GEMINI_API_KEY=...
uv run python -m backend.llm_kg.benchmark.dataset_builder \
  --config backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml
```

Benchmark run (2 models x 3 runs):

```bash
uv run python -m backend.llm_kg.benchmark.runner \
  --config backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml
```

Generate report for a run:

```bash
uv run python -m backend.llm_kg.benchmark.reporter \
  --run-dir backend/llm_kg/benchmark/runs/<timestamp>
```

Artifacts per run:

- `run_manifest.json`
- `per_question_scores.csv`
- `model_summary.csv`
- `summary.json`
- `report.md`
