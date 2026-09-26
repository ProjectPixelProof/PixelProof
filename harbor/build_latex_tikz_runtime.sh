#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTEXT="$ROOT/harbor/images/latex-tikz-runtime"
RUNTIME_IMAGE="${LATEX_TIKZ_RUNTIME_IMAGE:-question-foundry-latex-tikz-runtime:0.1}"
REPLAY_IMAGE="${LATEX_TIKZ_REPLAY_IMAGE:-question-foundry-candidate-replay-latex-tikz:0.1}"
MAX_RUNTIME_MIB="${LATEX_TIKZ_MAX_RUNTIME_MIB:-650}"
MAX_REPLAY_MIB="${LATEX_TIKZ_MAX_REPLAY_MIB:-800}"

docker build --pull --provenance=false -t "$RUNTIME_IMAGE" "$CONTEXT"
docker build --provenance=false -f "$CONTEXT/Dockerfile.replay" -t "$REPLAY_IMAGE" "$CONTEXT"

for specification in "$RUNTIME_IMAGE:$MAX_RUNTIME_MIB" "$REPLAY_IMAGE:$MAX_REPLAY_MIB"; do
  image="${specification%:*}"
  limit="${specification##*:}"
  bytes="$(
    docker history --format '{{.Size}}' "$image" | python3 -c '
import re, sys
units = {"B": 1, "kB": 1000, "MB": 1000**2, "GB": 1000**3}
total = 0.0
for raw in sys.stdin:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(B|kB|MB|GB)\n?", raw)
    if match is None:
        raise SystemExit(f"unrecognized Docker layer size: {raw!r}")
    total += float(match.group(1)) * units[match.group(2)]
print(round(total))'
  )"
  mib="$(( (bytes + 1048575) / 1048576 ))"
  image_id="$(docker inspect --type image --format '{{.Id}}' "$image")"
  printf '%s: %s bytes (%s MiB summed runnable layers; limit %s MiB; id %s)\n' \
    "$image" "$bytes" "$mib" "$limit" "$image_id"
  if (( mib > limit )); then
    printf 'image exceeds its bounded size policy: %s > %s MiB\n' "$mib" "$limit" >&2
    exit 1
  fi
done
