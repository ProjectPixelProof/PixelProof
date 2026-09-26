"""Append-only, agent-visible phase observations for audited foundry sessions.

These observations establish timing and ordering only. They never award a
scientific gate, protected novelty, or canonical admission.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = "candidate-phase-event-0.1.0"
DEFAULT_LEDGER = Path("/logs/artifacts/process/phase-events.jsonl")
KINDS = {
    "candidate_staged",
    "public_gate_started",
    "public_gate_finished",
    "repair_started",
    "envelope_check_started",
    "envelope_check_finished",
    "survivor_committed",
}

_HASH_EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
_HASH_EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def read_events(path: Path = DEFAULT_LEDGER) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"phase ledger line {line_number} is invalid JSON") from exc
        if not isinstance(event, dict) or event.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"phase ledger line {line_number} has an invalid schema")
        if event.get("sequence") != line_number or event.get("kind") not in KINDS:
            raise ValueError(f"phase ledger line {line_number} breaks sequence or kind")
        events.append(event)
    return events


def append_event(
    kind: str,
    *,
    candidate_ids: list[str],
    outcome: str | None = None,
    details: dict | None = None,
    path: Path = DEFAULT_LEDGER,
) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unsupported phase event: {kind}")
    if not candidate_ids or any(not isinstance(item, str) or not item for item in candidate_ids):
        raise ValueError("phase events require non-empty candidate IDs")
    events = read_events(path)
    event = {
        "schema_version": SCHEMA_VERSION,
        "sequence": len(events) + 1,
        "observed_at": utc_now(),
        "kind": kind,
        "candidate_ids": candidate_ids,
        "outcome": outcome,
        "details": details or {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, (json.dumps(event, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return event


def portfolio_ids(submission: Path) -> list[str]:
    payload = json.loads((submission / "portfolio.json").read_text(encoding="utf-8"))
    candidate_ids = payload.get("candidate_ids") if isinstance(payload, dict) else None
    if not isinstance(candidate_ids, list) or not all(
        isinstance(item, str) and item for item in candidate_ids
    ):
        raise ValueError("portfolio candidate_ids are unavailable for phase audit")
    return candidate_ids


def candidate_tree_sha256(candidate: Path) -> str:
    """Hash the committed candidate tree using the controller's exclusions."""

    files = {}
    for path in sorted(candidate.rglob("*")):
        relative = path.relative_to(candidate)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in _HASH_EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in _HASH_EXCLUDED_SUFFIXES:
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()
