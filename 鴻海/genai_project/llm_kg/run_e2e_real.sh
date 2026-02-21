#!/usr/bin/env bash
set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.e2e.yml}"
KEEP_VOLUMES="${KEEP_VOLUMES:-1}"
LOCAL_OLLAMA_BASE_URL="${LOCAL_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
COMPOSE_OLLAMA_BASE_URL="${COMPOSE_OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
EXTRACTION_MODEL="${EXTRACTION_MODEL:-sam860/deepseek-r1-0528-qwen3:8b}"
NL2CYPHER_MODEL="${NL2CYPHER_MODEL:-ministral-3:14b}"

export OLLAMA_BASE_URL="$COMPOSE_OLLAMA_BASE_URL"
export EXTRACTION_MODEL
export NL2CYPHER_MODEL

cleanup() {
  if [[ "$KEEP_VOLUMES" == "1" ]]; then
    docker compose -f "$COMPOSE_FILE" down --remove-orphans
  else
    docker compose -f "$COMPOSE_FILE" down -v --remove-orphans
  fi
}
trap cleanup EXIT

require_local_ollama_model() {
  local model="$1"
  local base_url="$2"
  python - "$base_url" "$model" <<'PY'
import json
import sys
import urllib.request

base_url = sys.argv[1].rstrip("/")
model = sys.argv[2]

try:
    with urllib.request.urlopen(f"{base_url}/api/tags", timeout=8) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except Exception:
    print("OLLAMA_UNREACHABLE")
    sys.exit(2)

models = payload.get("models", [])
for item in models:
    if str(item.get("name", "")).strip() == model:
        sys.exit(0)

print("MODEL_MISSING")
sys.exit(1)
PY
}

echo "[1/5] Starting base services (neo4j)..."
docker compose -f "$COMPOSE_FILE" up -d neo4j

echo "[2/5] Checking local Ollama models..."
if require_local_ollama_model "$EXTRACTION_MODEL" "$LOCAL_OLLAMA_BASE_URL"; then
  :
else
  status=$?
  if [[ "$status" -eq 2 ]]; then
    echo "[error] Local Ollama is unreachable at ${LOCAL_OLLAMA_BASE_URL}" >&2
    echo "Please start Ollama first, then run again." >&2
  else
    echo "[error] Missing extraction model: ${EXTRACTION_MODEL}" >&2
    echo "Run locally: ollama pull ${EXTRACTION_MODEL}" >&2
  fi
  exit 1
fi

if require_local_ollama_model "$NL2CYPHER_MODEL" "$LOCAL_OLLAMA_BASE_URL"; then
  :
else
  status=$?
  if [[ "$status" -eq 2 ]]; then
    echo "[error] Local Ollama is unreachable at ${LOCAL_OLLAMA_BASE_URL}" >&2
    echo "Please start Ollama first, then run again." >&2
  else
    echo "[error] Missing NL2Cypher model: ${NL2CYPHER_MODEL}" >&2
    echo "Run locally: ollama pull ${NL2CYPHER_MODEL}" >&2
  fi
  exit 1
fi

echo "[3/5] Starting llm_api..."
docker compose -f "$COMPOSE_FILE" up -d llm_api

echo "[4/5] API smoke test..."
docker compose -f "$COMPOSE_FILE" run --rm api_smoke

echo "[5/5] End-to-end tester..."
docker compose -f "$COMPOSE_FILE" run --rm tester

echo "E2E test completed successfully."
