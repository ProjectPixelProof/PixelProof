#!/bin/bash
set -u

mkdir -p /logs/verifier
if [ -d /logs/artifacts/auth-smoke ]; then
  mkdir -p /logs/verifier/artifacts
  cp -R /logs/artifacts/auth-smoke /logs/verifier/artifacts/auth-smoke
fi

if python /tests/verify_auth_smoke.py; then
  printf '{"submission_valid": 1.0}\n' > /logs/verifier/reward.json
  exit 0
fi

printf '{"submission_valid": 0.0}\n' > /logs/verifier/reward.json
exit 1
