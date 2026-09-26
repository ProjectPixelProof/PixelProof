#!/usr/bin/env bash
# Render the three reference worlds locally, with their inverse checks.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash examples/shared_examples/render.sh [OUTPUT_DIRECTORY]"
  echo "Default: artifacts/shared-examples; requires a new directory; no model calls."
  exit 0
fi
if [[ $# -gt 1 ]]; then echo "Expected at most one output directory." >&2; exit 2; fi
cd "$(dirname "$0")/../.."
output="${1:-artifacts/shared-examples}"
if [[ -e "$output" ]]; then echo "Output exists; choose a new directory." >&2; exit 2; fi
mkdir -p "$output"
for world in two_circles angle_acuteness counting_with_distractors; do
  uv run --extra runner python -m "worlds.$world.generate" --n 12 --seed 101 \
    --out-dir "$output/$world/images" --manifest "$output/$world/manifest.jsonl"
done
