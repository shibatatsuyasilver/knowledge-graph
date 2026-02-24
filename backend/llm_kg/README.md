# Gen-AI Knowledge QA + GraphDB Prototype (uv + MLX default)

This project now defaults to:

- Python environment managed by `uv`
- LLM provider via shared client (`openai` compatible by default)
- Local MLX server (`mlx_lm.server`) as default upstream
- Neo4j via Docker

Ollama is still supported as a fallback provider.

## Directory highlights

- `backend/llm_kg/llm_client.py`: shared provider-aware LLM client (`openai` / `ollama`)
- `backend/llm_kg/kg_builder.py`: entity/relation extraction + Neo4j upsert
- `backend/llm_kg/nl2cypher.py`: schema-grounded Text-to-Cypher generation
- `backend/llm_kg/llm_deploy.py`: FastAPI wrapper (`/api/chat`, `/health`)
- `backend/llm_kg/run_e2e_mlx_local.sh`: local one-command E2E runner (Neo4j + MLX + QA)

## 1) Install dependencies with uv

From workspace root (`/Users/silver/Documents/鴻海`):

```bash
uv sync --group backend --group llmkg --group demo --group dev
```

## 1.5) One-command local stack with Gemini key

```bash
cd /Users/silver/Documents/鴻海
cp .env.local.example .env.local
# edit .env.local and fill GEMINI_API_KEY
bash scripts/start_local_with_gemini.sh
```

The script auto-resolves common compose/container conflicts and ensures `GEMINI_API_KEY` is loaded into backend.

Important `.env.local` behavior:

- `docker compose` does **not** auto-read `.env.local` unless you pass `--env-file .env.local`.
- If you start services manually, use `docker compose --env-file .env.local ...` or run `bash scripts/start_local_with_gemini.sh`.
- If backend missed env injection, UI may show: `GEMINI_API_KEY is required when provider=gemini`.

## 2) Start MLX OpenAI-compatible server (default path)

```bash
uv run python -m mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit-DWQ-053125 \
  --host 0.0.0.0 \
  --port 8080 \
  --max-tokens 32768
```

## 3) Start Neo4j (Docker)

```bash
cd /Users/silver/Documents/鴻海
docker compose up -d neo4j
```

## 4) Start LLM API wrapper

```bash
cd /Users/silver/Documents/鴻海
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:8080/v1
export LLM_MODEL=mlx-community/Qwen3-8B-4bit-DWQ-053125
uv run uvicorn backend.llm_kg.llm_deploy:app --host 0.0.0.0 --port 8000
```

Quick test:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"請用繁中說明台積電與先進製程"}'
```

## 5) Build Knowledge Graph in Neo4j

```bash
cd /Users/silver/Documents/鴻海
uv run python -m backend.llm_kg.kg_builder
```

## 6) NL to Cypher demo

```bash
cd /Users/silver/Documents/鴻海
uv run python -m backend.llm_kg.nl2cypher
```

## 7) One-command local real-model E2E (requested)

```bash
cd /Users/silver/Documents/鴻海
uv run bash backend/llm_kg/run_e2e_mlx_local.sh
```

Expected success criteria:

- extracted `entities > 0`
- extracted `relations > 0`
- `qa_success >= MIN_QA_SUCCESS` (default `2`)
- exit code `0`

Stability defaults in `run_e2e_mlx_local.sh`:

- `EXTRACTION_JSON_MODE=strict_json` (conservative JSON-first extraction)
- `EXTRACTION_MAX_JSON_RETRIES=4` (total 5 extraction attempts)
- `E2E_ALLOW_FALLBACK_EXTRACT=1` (deterministic fallback payload when extraction fails/insufficient)
- `E2E_ALLOW_FALLBACK_QA=1` (template-based Cypher fallback when nl2cypher generation fails)
- `LLM_MAX_TOKENS=2048` (lower memory pressure on 16GB machines)

## Provider configuration

### Preferred: MLX OpenAI-compatible

- `LLM_PROVIDER=openai`
- `OPENAI_BASE_URL=http://localhost:8080/v1`
- `OPENAI_API_KEY=` (optional for local)
- `LLM_MODEL=mlx-community/Qwen3-8B-4bit-DWQ-053125`
- `LLM_TIMEOUT_SECONDS=180`
- `LLM_TEMPERATURE=0.1`
- `LLM_MAX_TOKENS=512` (or higher per task)
- `GEMINI_INPUT_TOKEN_LIMIT=1048576`
- `GEMINI_OUTPUT_TOKEN_LIMIT=65536`
- `GEMINI_TWO_PASS_EXTRACTION=1` (Gemini 兩階段抽取：先實體盤點補齊，再抽關係)
- `CHUNK_SIZE_MODE=provider` (`provider` means Gemini uses token-based chunking)
- `CHUNK_SIZE_TOKENS=1048576`
- `CHUNK_MIN_TOKENS=120`
- `KG_QA_USE_LLM=1`
- `KG_QA_MODEL=...` (optional)
- `KG_QA_MAX_TOKENS=1024`
- `KG_QA_TEMPERATURE=0.1`

### Fallback: Ollama

- `LLM_PROVIDER=ollama`
- `OLLAMA_BASE_URL=http://localhost:11434`
- `LLM_MODEL=deepseek-r1:8b` (global fallback model)
- `EXTRACTION_MODEL=sam860/deepseek-r1-0528-qwen3:8b`
- `NL2CYPHER_MODEL=ministral-3:14b`

Example:

```bash
export LLM_PROVIDER=ollama
export OLLAMA_BASE_URL=http://localhost:11434
export EXTRACTION_MODEL=sam860/deepseek-r1-0528-qwen3:8b
export NL2CYPHER_MODEL=ministral-3:14b
```

Legacy env vars (`OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS`, etc.) are still read for backward compatibility.

## Docker compose (llmkg)

`backend/llm_kg/docker-compose.llmkg.yml` no longer requires an Ollama service by default.
It expects Ollama running on host at `http://host.docker.internal:11434` and defaults to `LLM_PROVIDER=ollama`.

## Async ingest APIs

Existing sync routes are unchanged. New async routes:

- `POST /api/process_text_async/start`
- `GET /api/process_text_async/{job_id}`
- `POST /api/process_url_async/start`
- `GET /api/process_url_async/{job_id}`
- `POST /api/process_keyword_async/start`
- `GET /api/process_keyword_async/{job_id}`

`frontend Build Knowledge Graph` now uses async polling for Text/File/URL/Keyword.

## Benchmark: DeepSeek R1 vs Gemma3 (100 題)

Config:

- `backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml`

### 1) Build/freeze dataset

```bash
export GEMINI_API_KEY=...
uv run python -m backend.llm_kg.benchmark.dataset_builder \
  --config backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml
```

Frozen dataset output:

- `backend/llm_kg/benchmark/datasets/kgqa_zh_tw_100_v1.jsonl`

### 2) Run benchmark

```bash
uv run python -m backend.llm_kg.benchmark.runner \
  --config backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml
```

Runner performs:

1. 2 models (`deepseek-r1:8b`, `gemma3:12b`)
2. 3 runs per model
3. Extraction scoring (Triple-F1) + KG QA scoring (Accuracy)
4. Winner decision by KG QA accuracy mean

### 3) Regenerate markdown report

```bash
uv run python -m backend.llm_kg.benchmark.reporter \
  --run-dir backend/llm_kg/benchmark/runs/<timestamp>
```

Artifacts in each run dir:

- `run_manifest.json`
- `per_question_scores.csv`
- `model_summary.csv`
- `summary.json`
- `report.md`
