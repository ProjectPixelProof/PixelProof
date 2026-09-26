#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Preparation only: no model calls and no credentials read.
uv run python -m scripts.release prepare --rq "${1:-1}" --agent claude \
  --id "${2:-claude_example}" "${@:3}"
