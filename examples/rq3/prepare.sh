#!/usr/bin/env bash
# Prepare only: no credentials or model calls. Use --help for options.
set -euo pipefail
cd "$(dirname "$0")/../.."
exec uv run --extra runner python -m scripts.release prepare --rq 3 "$@"
