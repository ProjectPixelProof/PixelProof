#!/bin/bash
set -u

mkdir -p /logs/verifier
if ! python /tests/check_verifier_boundary.py \
  --task-spec /tests/TASK_SPEC.toml \
  --report /logs/verifier/boundary_report.json; then
  cp /logs/verifier/boundary_report.json /logs/verifier/gate_report.json
  printf '%s\n' \
    '{"submission_valid":0.0,"mechanically_eligible_fraction":0.0,"all_mechanically_eligible":0.0,"registry_distinct_fraction":0.0,"semantic_review_ready_fraction":0.0}' \
    > /logs/verifier/reward.json
  mkdir -p /logs/verifier/artifacts
  [ ! -d /logs/artifacts/submission ] || \
    cp -R /logs/artifacts/submission /logs/verifier/artifacts/submission
  [ ! -d /logs/artifacts/process ] || \
    cp -R /logs/artifacts/process /logs/verifier/artifacts/process
  cp /logs/verifier/gate_report.json /logs/verifier/artifacts/gate_report.json
  cp /logs/verifier/boundary_report.json /logs/verifier/artifacts/boundary_report.json
  exit 0
fi

gatekeeper_args=(
  --submission /logs/artifacts/submission
  --task-spec /tests/TASK_SPEC.toml
  --registry /tests/historical_registry.jsonl
  --mechanism-memory /tests/mechanism_memory.jsonl
  --negative-memory /tests/negative_memory.jsonl
  --report /logs/verifier/gate_report.json
  --reward /logs/verifier/reward.json
)
if [ -s /tests/profile_source_mechanisms.jsonl ]; then
  gatekeeper_args+=(
    --profile-source-memory /tests/profile_source_mechanisms.jsonl
  )
fi
if [ -s /tests/QUALITY_DIVERSITY_POLICY.json ]; then
  gatekeeper_args+=(
    --quality-diversity-policy /tests/QUALITY_DIVERSITY_POLICY.json
  )
fi

python /tests/candidate_gatekeeper.py \
  "${gatekeeper_args[@]}"
verifier_status=$?

mkdir -p /logs/verifier/artifacts
[ ! -d /logs/artifacts/submission ] || \
  cp -R /logs/artifacts/submission /logs/verifier/artifacts/submission
[ ! -d /logs/artifacts/process ] || \
  cp -R /logs/artifacts/process /logs/verifier/artifacts/process
cp /logs/verifier/gate_report.json /logs/verifier/artifacts/gate_report.json
cp /logs/verifier/boundary_report.json /logs/verifier/artifacts/boundary_report.json

exit "$verifier_status"
