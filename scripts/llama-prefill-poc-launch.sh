#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PI_LLAMA_PREFILL_PORT:-8081}"
AGENT_DIR="${PI_CODING_AGENT_DIR:-/private/tmp/pi-dev/agent}"
SESSION_DIR="${PI_CODING_AGENT_SESSION_DIR:-/private/tmp/pi-dev/sessions-$(date +%Y%m%d-%H%M%S)}"
MODEL_PRESET="${PI_LLAMA_PREFILL_MODEL:-gemma}"

case "$MODEL_PRESET" in
	gemma)
		CONTEXT_WINDOW="${PI_LLAMA_PREFILL_CONTEXT_WINDOW:-32768}"
		;;
	qwen27b)
		CONTEXT_WINDOW="${PI_LLAMA_PREFILL_CONTEXT_WINDOW:-32768}"
		;;
	qwen35a3b)
		CONTEXT_WINDOW="${PI_LLAMA_PREFILL_CONTEXT_WINDOW:-32768}"
		;;
	*)
		echo "Unknown PI_LLAMA_PREFILL_MODEL: $MODEL_PRESET" >&2
		echo "Expected: gemma, qwen27b, or qwen35a3b" >&2
		exit 1
		;;
esac

mkdir -p "$AGENT_DIR" "$SESSION_DIR"

cat >"$AGENT_DIR/models.json" <<JSON
{
  "providers": {
    "llama-cpp": {
      "baseUrl": "http://127.0.0.1:${PORT}/v1",
      "api": "openai-completions",
      "apiKey": "llama.cpp",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false,
        "supportsUsageInStreaming": false,
        "maxTokensField": "max_tokens"
      },
      "models": [
        {
          "id": "local-model",
          "name": "llama.cpp local",
          "reasoning": false,
          "input": ["text"],
          "contextWindow": ${CONTEXT_WINDOW},
          "maxTokens": 8192,
          "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
        }
      ]
    }
  }
}
JSON

echo "Starting Pi llama prefill PoC"
echo "  agent dir:   $AGENT_DIR"
echo "  session dir: $SESSION_DIR"
echo "  llama url:   http://127.0.0.1:${PORT}/v1"

cd "$ROOT_DIR"
exec env \
	PI_CODING_AGENT_DIR="$AGENT_DIR" \
	PI_CODING_AGENT_SESSION_DIR="$SESSION_DIR" \
	PI_OFFLINE="${PI_OFFLINE:-1}" \
	PI_SKIP_VERSION_CHECK="${PI_SKIP_VERSION_CHECK:-1}" \
	PI_LLAMA_PREFILL_STATUS="${PI_LLAMA_PREFILL_STATUS:-1}" \
	./pi-test.sh --model llama-cpp/local-model "$@"
