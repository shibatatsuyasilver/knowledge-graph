#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.yml}"
NEO4J_SERVICE="${NEO4J_SERVICE:-neo4j}"

LLM_PROVIDER="${LLM_PROVIDER:-openai}"
LLM_MODEL="${LLM_MODEL:-mlx-community/Qwen3-8B-4bit-DWQ-053125}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-2048}"
LLM_TIMEOUT_SECONDS="${LLM_TIMEOUT_SECONDS:-180}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.1}"
MLX_CHAT_TEMPLATE_ARGS="${MLX_CHAT_TEMPLATE_ARGS:-{\"enable_thinking\":false}}"
MLX_READY_RETRIES="${MLX_READY_RETRIES:-900}"
MLX_READY_SLEEP_SECONDS="${MLX_READY_SLEEP_SECONDS:-2}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
EXTRACTION_JSON_MODE="${EXTRACTION_JSON_MODE:-strict_json}"
E2E_ALLOW_FALLBACK_EXTRACT="${E2E_ALLOW_FALLBACK_EXTRACT:-1}"
EXTRACTION_TIMEOUT_SECONDS="${EXTRACTION_TIMEOUT_SECONDS:-600}"
NL2CYPHER_TIMEOUT_SECONDS="${NL2CYPHER_TIMEOUT_SECONDS:-300}"
EXTRACTION_NUM_PREDICT="${EXTRACTION_NUM_PREDICT:-8192}"
NL2CYPHER_NUM_PREDICT="${NL2CYPHER_NUM_PREDICT:-128}"
EXTRACTION_MAX_JSON_RETRIES="${EXTRACTION_MAX_JSON_RETRIES:-4}"
EXTRACTION_MODEL="${EXTRACTION_MODEL:-$LLM_MODEL}"
NL2CYPHER_MODEL="${NL2CYPHER_MODEL:-$LLM_MODEL}"

export HF_HUB_DISABLE_XET

OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8080/v1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"

NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password}"

MLX_PID=""

cleanup() {
  if [[ -n "$MLX_PID" ]] && kill -0 "$MLX_PID" >/dev/null 2>&1; then
    echo "[cleanup] stopping mlx_lm.server (pid=$MLX_PID)"
    kill "$MLX_PID" || true
  fi
}
trap cleanup EXIT

init_compose_project_name() {
  if [[ -n "${COMPOSE_PROJECT_NAME:-}" ]]; then
    return 0
  fi

  local derived
  derived="$(basename "$ROOT_DIR" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//')"
  if [[ -z "$derived" ]]; then
    derived="genai-kg"
  fi
  export COMPOSE_PROJECT_NAME="$derived"
}

is_neo4j_reachable() {
  python - "$NEO4J_URI" <<'PY'
import socket
import sys
from urllib.parse import urlparse

uri = urlparse(sys.argv[1])
host = uri.hostname or "localhost"
port = uri.port or 7687

sock = socket.socket()
sock.settimeout(1.5)
try:
    sock.connect((host, port))
except Exception:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

start_neo4j() {
  init_compose_project_name

  if is_neo4j_reachable; then
    echo "[info] Neo4j already reachable at ${NEO4J_URI}, skipping compose startup."
    return 0
  fi

  local output=""
  local status=0
  set +e
  output="$(docker compose -f "$COMPOSE_FILE" up -d "$NEO4J_SERVICE" 2>&1)"
  status=$?
  set -e

  if [[ "$status" -eq 0 ]]; then
    echo "$output"
  else
    echo "$output" >&2

    local conflict_name=""
    conflict_name="$(echo "$output" | sed -n 's/.*container name "\/\([^"]*\)".*/\1/p' | head -n 1)"
    if [[ -n "$conflict_name" ]]; then
      echo "[warn] Container name conflict detected (${conflict_name}), attempting to reuse existing container."
      docker start "$conflict_name" >/dev/null 2>&1 || true
    fi
  fi

  if ! is_neo4j_reachable; then
    echo "[error] Neo4j is not reachable at ${NEO4J_URI} after compose startup/reuse attempt." >&2
    exit 1
  fi
}

wait_http_ok() {
  local url="$1"
  local retries="${2:-120}"
  local sleep_seconds="${3:-2}"
  local attempt=1
  while [[ "$attempt" -le "$retries" ]]; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
    attempt=$((attempt + 1))
  done
  return 1
}

echo "[1/4] Starting Neo4j via Docker Compose ($COMPOSE_FILE)..."
start_neo4j

if [[ "$LLM_PROVIDER" == "openai" ]]; then
  echo "[2/4] Ensuring MLX OpenAI-compatible server is available..."
  if wait_http_ok "${OPENAI_BASE_URL}/models" 1 1; then
    echo "[info] Existing OpenAI-compatible endpoint detected at ${OPENAI_BASE_URL}"
  else
    uv run mlx_lm.server \
      --model "$LLM_MODEL" \
      --host 0.0.0.0 \
      --port 8080 \
      --max-tokens "$LLM_MAX_TOKENS" \
      --chat-template-args "$MLX_CHAT_TEMPLATE_ARGS" \
      >/tmp/mlx_lm_server.log 2>&1 &
    MLX_PID="$!"
    if ! wait_http_ok "${OPENAI_BASE_URL}/models" "$MLX_READY_RETRIES" "$MLX_READY_SLEEP_SECONDS"; then
      echo "[error] mlx_lm.server failed to become ready. Logs: /tmp/mlx_lm_server.log" >&2
      exit 1
    fi
  fi
elif [[ "$LLM_PROVIDER" == "ollama" ]]; then
  echo "[2/4] Checking Ollama endpoint..."
  if ! wait_http_ok "${OLLAMA_BASE_URL}/api/tags" 60 2; then
    echo "[error] Ollama is not reachable at ${OLLAMA_BASE_URL}" >&2
    exit 1
  fi
else
  echo "[error] Unsupported LLM_PROVIDER: ${LLM_PROVIDER}" >&2
  exit 1
fi

echo "[3/4] Running end-to-end validation..."
(
  cd "$ROOT_DIR"
  export LLM_PROVIDER
  export LLM_MODEL
  export LLM_MAX_TOKENS
  export LLM_TIMEOUT_SECONDS
  export LLM_TEMPERATURE
  export OPENAI_BASE_URL
  export OPENAI_API_KEY
  export OLLAMA_BASE_URL
  export NEO4J_URI
  export NEO4J_USER
  export NEO4J_PASSWORD
  export EXTRACTION_JSON_MODE
  export E2E_ALLOW_FALLBACK_EXTRACT
  export EXTRACTION_TIMEOUT_SECONDS
  export NL2CYPHER_TIMEOUT_SECONDS
  export EXTRACTION_NUM_PREDICT
  export NL2CYPHER_NUM_PREDICT
  export EXTRACTION_MAX_JSON_RETRIES
  export EXTRACTION_MODEL
  export NL2CYPHER_MODEL
  uv run python -m backend.llm_kg.e2e_runner
)

echo "[4/4] E2E completed successfully."
