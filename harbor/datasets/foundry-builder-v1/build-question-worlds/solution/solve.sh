#!/bin/bash
set -euo pipefail

mkdir -p /logs/artifacts/submission/candidates
cp /solution/portfolio.json /logs/artifacts/submission/portfolio.json
cp -R /solution/reference_marker /logs/artifacts/submission/candidates/reference_marker
python /solution/write_reference_metadata.py
python /logs/artifacts/submission/candidates/reference_marker/world/generate.py \
  --out /logs/artifacts/submission/candidates/reference_marker/evidence \
  --n 12 \
  --seed 101
python /logs/artifacts/submission/candidates/reference_marker/world/verify.py \
  --dataset /logs/artifacts/submission/candidates/reference_marker/evidence

if grep -Eq 'protocol = "question-world@0\.(3|4)\.0"' /workspace/TASK_SPEC.toml; then
  python /solution/write_reference_process.py
fi
