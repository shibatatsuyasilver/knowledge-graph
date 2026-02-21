#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/docker-compose.yml}"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env.local}"

container_exists() {
  local name="$1"
  docker ps -a --format '{{.Names}}' | grep -Fxq "$name"
}

container_running() {
  local name="$1"
  docker ps --format '{{.Names}}' | grep -Fxq "$name"
}

ensure_project_name() {
  if [[ -n "${COMPOSE_PROJECT_NAME:-}" ]]; then
    return 0
  fi

  local existing_project
  existing_project="$(docker inspect -f '{{ index .Config.Labels "com.docker.compose.project" }}' genai-backend 2>/dev/null || true)"
  if [[ -n "$existing_project" ]]; then
    COMPOSE_PROJECT_NAME="$existing_project"
  else
    COMPOSE_PROJECT_NAME="genaikg"
  fi
  export COMPOSE_PROJECT_NAME
}

ensure_network_exists() {
  local network_name="$1"
  if docker network inspect "$network_name" >/dev/null 2>&1; then
    return 0
  fi
  docker network create "$network_name" >/dev/null
}

ensure_neo4j_attached() {
  local network_name="$1"
  if ! container_exists "genai-neo4j"; then
    return 0
  fi

  if ! container_running "genai-neo4j"; then
    echo "[info] Starting existing genai-neo4j container..."
    docker start genai-neo4j >/dev/null
  fi

  if docker inspect -f '{{range $k, $v := .NetworkSettings.Networks}}{{println $k}}{{end}}' genai-neo4j | grep -Fxq "$network_name"; then
    return 0
  fi

  echo "[info] Connecting genai-neo4j to $network_name..."
  docker network connect --alias neo4j "$network_name" genai-neo4j >/dev/null
}

start_neo4j_if_missing() {
  if container_exists "genai-neo4j"; then
    echo "[info] Reusing existing genai-neo4j container."
    return 0
  fi
  echo "[info] Starting neo4j via docker compose..."
  docker compose -f "$COMPOSE_FILE" up -d neo4j >/dev/null
}

start_backend_frontend() {
  local output=""
  echo "[info] Starting backend/frontend..."
  if output="$(docker compose -f "$COMPOSE_FILE" up -d --build --no-deps backend frontend 2>&1)"; then
    echo "$output"
    return 0
  fi

  if echo "$output" | grep -Eq 'container name "/genai-(backend|frontend)" is already in use'; then
    echo "[warn] Found stale genai-backend/genai-frontend containers. Recreating..."
    docker rm -f genai-backend genai-frontend >/dev/null 2>&1 || true
    docker compose -f "$COMPOSE_FILE" up -d --build --no-deps backend frontend >/dev/null
    return 0
  fi

  echo "$output" >&2
  return 1
}

wait_backend_ready() {
  local attempts=90
  local i=0
  until curl -fsS http://localhost:8000/ >/dev/null 2>&1; do
    i=$((i + 1))
    if [[ "$i" -ge "$attempts" ]]; then
      echo "[error] Backend not ready at http://localhost:8000 after ${attempts}s." >&2
      return 1
    fi
    sleep 1
  done
}

load_env_file() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "[error] Missing env file: $ENV_FILE" >&2
    echo "[hint] Create it from template: cp $ROOT_DIR/.env.local.example $ROOT_DIR/.env.local" >&2
    return 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

validate_gemini_key() {
  local key="${GEMINI_API_KEY:-}"
  if [[ -z "$key" || "$key" == "REPLACE_WITH_YOUR_GEMINI_API_KEY" ]]; then
    echo "[error] GEMINI_API_KEY is empty in $ENV_FILE" >&2
    return 1
  fi
  if [[ "$key" == "test-local-key" ]]; then
    echo "[error] GEMINI_API_KEY is still test-local-key; please replace with a real Gemini key." >&2
    return 1
  fi
}

validate_gemini_key_online() {
  if [[ "${SKIP_GEMINI_KEY_VALIDATE:-0}" == "1" ]]; then
    echo "[info] Skip online Gemini key validation (SKIP_GEMINI_KEY_VALIDATE=1)."
    return 0
  fi

  python - "$GEMINI_API_KEY" <<'PY'
import json
import sys
import urllib.error
import urllib.parse
import urllib.request

key = sys.argv[1]
url = "https://generativelanguage.googleapis.com/v1beta/models?key=" + urllib.parse.quote_plus(key)
req = urllib.request.Request(url, method="GET")
try:
    with urllib.request.urlopen(req, timeout=12) as resp:
        status = getattr(resp, "status", 200)
        if status >= 400:
            print(f"[error] Gemini API key validation failed: HTTP {status}")
            sys.exit(1)
        print("[ok] Gemini API key validated online.")
        sys.exit(0)
except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="ignore")
    message = body
    try:
        parsed = json.loads(body)
        message = parsed.get("error", {}).get("message", "") or body
    except Exception:
        pass
    message = message.replace("\n", " ").strip()
    if len(message) > 280:
        message = message[:280] + "..."
    print(f"[error] Gemini API key validation failed: HTTP {exc.code} - {message}")
    sys.exit(1)
except Exception as exc:
    print(f"[warn] Online Gemini key validation skipped due to network error: {exc}")
    sys.exit(0)
PY
}

print_summary() {
  echo "[ok] Services are up."
  echo "[ok] Frontend: http://localhost"
  echo "[ok] Backend:  http://localhost:8000"
  echo "[ok] Neo4j:    http://localhost:7474"
  docker exec genai-backend sh -lc 'if [ -n "$GEMINI_API_KEY" ]; then echo "[ok] backend GEMINI_API_KEY loaded"; else echo "[error] backend GEMINI_API_KEY missing"; exit 1; fi'
}

main() {
  load_env_file
  validate_gemini_key
  validate_gemini_key_online
  ensure_project_name

  local network_name="${COMPOSE_PROJECT_NAME}_genai-net"
  ensure_network_exists "$network_name"
  start_neo4j_if_missing
  ensure_neo4j_attached "$network_name"
  start_backend_frontend
  ensure_neo4j_attached "$network_name"
  wait_backend_ready
  print_summary
}

main "$@"
