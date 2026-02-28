# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Enterprise GenAI Knowledge Graph Studio — a full-stack app for building knowledge graphs from text, URLs, and web searches, with natural language querying and chat. Stack: Python FastAPI backend + React/TypeScript frontend + Neo4j graph DB + multi-provider LLM abstraction.

## Commands

### Python / Backend

```bash
# Install all dependencies (use uv, not pip)
uv sync --group backend --group llmkg --group demo --group dev

# Start backend API (with MLX local LLM)
export LLM_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:8080/v1
export LLM_MODEL=mlx-community/Qwen3-8B-4bit-DWQ-053125
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest backend/tests/test_main.py

# Run a specific test
uv run pytest backend/tests/test_main.py::test_function_name

# Start MLX local LLM server (Apple Silicon)
uv run python -m mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit-DWQ-053125 \
  --host 0.0.0.0 --port 8080 --max-tokens 32768
```

### Frontend

```bash
cd frontend
npm install
npm run dev      # Dev server on :5173 with /api proxy to localhost:8000
npm run build    # TypeScript compile + Vite build
```

### Docker / Full Stack

```bash
# One-command startup with Gemini (recommended)
cp .env.local.example .env.local
# fill GEMINI_API_KEY in .env.local
bash scripts/start_local_with_gemini.sh

# Start only Neo4j
docker compose up -d neo4j

# IMPORTANT: docker compose does NOT auto-read .env.local
# Always use: docker compose --env-file .env.local ...
# or use the start script above
```

### Benchmarks

```bash
uv run python -m backend.llm_kg.benchmark.dataset_builder \
  --config backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml

uv run python -m backend.llm_kg.benchmark.runner \
  --config backend/llm_kg/benchmark/configs/benchmark_zh_tw_100.yaml

uv run python -m backend.llm_kg.benchmark.reporter \
  --run-dir backend/llm_kg/benchmark/runs/<timestamp>
```

## Architecture

### Backend (`backend/`)

Three-layer design:

1. **API layer** (`backend/api/routers/`) — FastAPI routers:
   - `ingest.py` — async endpoints for text/URL/keyword ingestion
   - `qa.py` — NL2Cypher querying and general chat
   - `llm_compat.py` — OpenAI-compatible `/v1/chat/completions` endpoint

2. **Services layer** (`backend/services/`) — business logic:
   - `services/ingest/ingest_service.py` — chunking, URL fetch via DuckDuckGo, token estimation
   - `services/qa/qa_service.py` — QA execution, LLM answer rewriting

3. **LLM/KG layer** (`backend/llm_kg/`) — core intelligence:
   - `llm_client.py` — multi-provider abstraction (OpenAI-compatible, Ollama, Gemini)
   - `kg_builder.py` — schema-constrained entity/relation extraction with JSON repair
   - `nl2cypher.py` — NL→Cypher generation with repair loops and agentic modes

**Async job pattern**: All ingest operations create background jobs. Frontend polls job status endpoints (`GET /api/process_*_async/{job_id}`). Job store is in-memory with TTL (`backend/jobs/store.py`).

**Config**: All settings come from environment variables parsed in `backend/config/settings.py`.

### Frontend (`frontend/src/`)

- `App.tsx` — tab navigation between "Build KG" and "Chat"
- `components/BuildKG.tsx` — KG ingestion UI (text, file, URL, keyword crawl tabs)
- `components/Chat.tsx` — chat UI supporting both KG query mode and general chat
- `api.ts` — HTTP client with async job polling helpers
- `types.ts` — TypeScript interfaces matching all API contracts

Vite dev server proxies `/api` → `localhost:8000`.

### Knowledge Graph Schema

Fixed entity types: `Person`, `Organization`, `Location`, `Technology`, `Product`, `FiscalPeriod`, `FinancialMetric`

Fixed relation types: `FOUNDED_BY`, `CHAIRED_BY`, `HEADQUARTERED_IN`, `PRODUCES`, `SUPPLIES_TO`, `USES`, `COMPETES_WITH`, `HAS_FINANCIAL_METRIC`, `FOR_PERIOD`

Direction rules are enforced (e.g., `FOUNDED_BY` must be `Organization→Person`).

## Key Environment Variables

```bash
# LLM provider selection
LLM_PROVIDER=openai|ollama|gemini
LLM_MODEL=...
EXTRACTION_MODEL=...     # optional override for entity/relation extraction
NL2CYPHER_MODEL=...      # optional override for Cypher generation
KG_QA_MODEL=...          # optional override for KG answer rewriting

# OpenAI-compatible (including local MLX)
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=...       # optional for local MLX

# Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3-pro-preview
GEMINI_TWO_PASS_EXTRACTION=1   # Phase-1 entity inventory → Phase-2 relation extraction

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Chunking
CHUNK_SIZE_MODE=provider|token|char
INGEST_CHUNK_LIMIT=5     # max chunks per source (0 = no limit)

# Ollama thinking mode — only enable for models that support it
OLLAMA_THINK=false
OLLAMA_THINK_JSON=false
```

## Notes

- Python version must be `>=3.11,<3.12` (enforced in `pyproject.toml`)
- `ARCHITECTURE.md` describes OpenClaw (a separate AI gateway project) — it is legacy documentation and does not describe this codebase
- `.env.local` is gitignored; copy from `.env.local.example` to set API keys locally
