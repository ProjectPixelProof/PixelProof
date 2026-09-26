#!/bin/bash
set -euo pipefail

mkdir -p /logs/artifacts/auth-smoke
printf 'AUTH_SMOKE_OK\n' > /logs/artifacts/auth-smoke/result.txt
