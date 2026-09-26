#!/usr/bin/env bash
# Starts paid provider usage; requires a frozen, committed configuration and auth.
set -euo pipefail
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash examples/run_campaign.sh CAMPAIGN_ID"
  echo "Run campaigns/CAMPAIGN_ID.toml after authentication, freezing, and committing."
  echo "This starts provider usage. Outputs: artifacts/CAMPAIGN_ID/."
  exit 0
fi
if [[ $# -ne 1 || ! "$1" =~ ^[a-z][a-z0-9_]{2,63}$ ]]; then
  echo "Provide one campaign ID (3–64 lowercase letters/digits/underscores)." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
exec uv run --extra runner python -m scripts.release run \
  --campaign "campaigns/$1.toml" --confirm "$1"
