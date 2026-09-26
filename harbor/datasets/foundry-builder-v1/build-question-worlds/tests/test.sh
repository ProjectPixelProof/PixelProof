#!/bin/bash
set -u

mkdir -p /logs/verifier
if ! python /tests/check_verifier_boundary.py \
  --task-spec /tests/TASK_SPEC.toml \
  --report /logs/verifier/boundary_report.json; then
  cp /logs/verifier/boundary_report.json /logs/verifier/gate_report.json
  printf '%s\n' \
    '{"submission_valid":0.0,"mechanically_eligible_fraction":0.0,"all_mechanically_eligible":0.0,"registry_distinct_fraction":0.0,"process_trace_valid":0.0,"search_budget_compliant":0.0}' \
    > /logs/verifier/reward.json
  if [ -d /logs/artifacts/submission ]; then
    mkdir -p /logs/verifier/artifacts
    cp -R /logs/artifacts/submission /logs/verifier/artifacts/submission
  fi
  if [ -d /logs/artifacts/process ]; then
    mkdir -p /logs/verifier/artifacts
    cp -R /logs/artifacts/process /logs/verifier/artifacts/process
  fi
  mkdir -p /logs/verifier/artifacts
  cp /logs/verifier/gate_report.json /logs/verifier/artifacts/gate_report.json
  exit 0
fi

gatekeeper_args=(
  --submission /logs/artifacts/submission
  --task-spec /tests/TASK_SPEC.toml
  --registry /tests/historical_registry.jsonl
  --report /logs/verifier/gate_report.json
  --reward /logs/verifier/reward.json
)
protocol="$(
  python - <<'PY'
import tomllib
from pathlib import Path
print(tomllib.loads(Path("/tests/TASK_SPEC.toml").read_text())["protocol"])
PY
)"
if [ "$protocol" = "question-world@0.4.0" ]; then
  gatekeeper_args+=(
    --mechanism-memory /tests/mechanism_memory.jsonl
    --negative-memory /tests/negative_memory.jsonl
  )
  if [ -s /tests/profile_source_mechanisms.jsonl ]; then
    gatekeeper_args+=(
      --profile-source-memory /tests/profile_source_mechanisms.jsonl
    )
  fi
fi
if [ -s /tests/QUALITY_DIVERSITY_POLICY.json ]; then
  gatekeeper_args+=(
    --quality-diversity-policy /tests/QUALITY_DIVERSITY_POLICY.json
  )
fi

python /tests/candidate_gatekeeper.py \
  "${gatekeeper_args[@]}"
verifier_status=$?

if [ "$protocol" = "question-world@0.3.0" ] || [ "$protocol" = "question-world@0.4.0" ]; then
  python /tests/process_trace_gatekeeper.py \
    --process /logs/artifacts/process \
    --submission /logs/artifacts/submission \
    --task-spec /tests/TASK_SPEC.toml \
    --report /logs/verifier/process_report.json \
    --reward /logs/verifier/reward.json
fi

if [ -d /logs/artifacts/submission ]; then
  mkdir -p /logs/verifier/artifacts
  cp -R /logs/artifacts/submission /logs/verifier/artifacts/submission
fi
if [ -d /logs/artifacts/process ]; then
  mkdir -p /logs/verifier/artifacts
  cp -R /logs/artifacts/process /logs/verifier/artifacts/process
fi
if [ -f /logs/verifier/gate_report.json ]; then
  mkdir -p /logs/verifier/artifacts
  cp /logs/verifier/gate_report.json /logs/verifier/artifacts/gate_report.json
fi
if [ -f /logs/verifier/process_report.json ]; then
  mkdir -p /logs/verifier/artifacts
  cp /logs/verifier/process_report.json /logs/verifier/artifacts/process_report.json
fi
cp /logs/verifier/boundary_report.json /logs/verifier/artifacts/boundary_report.json

# A scientifically invalid candidate is a measured outcome, not a verifier
# infrastructure crash. candidate_gatekeeper writes zero-valued gates and returns
# zero whenever it could complete the assessment.
exit "$verifier_status"
