"""Immutable run configuration, immutable plans, and append-only attempt events."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

from question_foundry.registry import canonical_json

from . import EVENT_SCHEMA_VERSION, PLAN_SCHEMA_VERSION, RUN_SCHEMA_VERSION

EVENTS_DIRECTORY = "events"
LEGACY_LEDGER_FILENAME = "events.jsonl"
TERMINAL_STATUSES = frozenset({"scored", "unparseable", "refused"})
RETRYABLE_STATUSES = frozenset({"attempt_started", "api_error"})


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def assert_secret_absent(root: Path, secret: str) -> None:
    """Fail closed if an in-memory credential was persisted under a run directory."""
    if not secret:
        raise ValueError("secret scan requires a non-empty in-memory pattern")
    pattern = secret.encode()
    for path in root.rglob("*"):
        if path.is_file() and pattern in path.read_bytes():
            raise RuntimeError(f"credential leak detected in evaluation artifact: {path}")


def write_new_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(payload) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def acquire_run_lock(run_dir: Path) -> Path:
    """Acquire the single-writer lock for one evaluation directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ".writer.lock"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(
            f"evaluation run already has a writer lock: {path}; "
            "do not run two evaluators in one --run-dir"
        ) from exc
    try:
        payload = canonical_json({"pid": os.getpid(), "created_at": utc_now()}) + "\n"
        os.write(descriptor, payload.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def release_run_lock(path: Path) -> None:
    path.unlink(missing_ok=False)


def initialize_run(run_dir: Path, configuration: dict) -> dict:
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run.json"
    comparable = dict(configuration)
    comparable["schema_version"] = RUN_SCHEMA_VERSION
    comparable["configuration_sha256"] = sha256_json(comparable)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        expected = {key: value for key, value in existing.items() if key != "created_at"}
        if expected != comparable:
            raise ValueError(
                "run configuration differs from existing immutable run: "
                f"{path}; use a new --run-dir"
            )
        return existing
    payload = dict(comparable)
    payload["created_at"] = utc_now()
    write_new_json(path, payload)
    return payload


def read_events(run_dir: Path) -> list[dict]:
    event_paths = sorted((run_dir / EVENTS_DIRECTORY).glob("event-*.json"))
    path = run_dir / LEGACY_LEDGER_FILENAME
    output = []
    if path.is_file():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            output.append(event)
    output.extend(json.loads(item.read_text(encoding="utf-8")) for item in event_paths)
    _validate_events(output, run_dir)
    return output


def _validate_events(events: list[dict], source: Path) -> None:
    event_ids = set()
    for number, event in enumerate(events, 1):
        if event.get("schema_version") != EVENT_SCHEMA_VERSION:
            raise ValueError(f"{source}:{number}: unsupported event schema")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or event_id in event_ids:
            raise ValueError(f"{source}:{number}: missing or duplicate event_id")
        event_ids.add(event_id)


def append_event(run_dir: Path, event: dict) -> None:
    payload = dict(event)
    payload["schema_version"] = EVENT_SCHEMA_VERSION
    payload.setdefault("created_at", utc_now())
    if not isinstance(payload.get("event_id"), str):
        raise ValueError("event requires event_id")
    event_dir = run_dir / EVENTS_DIRECTORY
    event_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(event_dir.glob("event-*.json"))
    duplicate_file = next(event_dir.glob(f"event-*-{payload['event_id']}.json"), None)
    legacy = run_dir / LEGACY_LEDGER_FILENAME
    duplicate_legacy = legacy.is_file() and any(
        item.get("event_id") == payload["event_id"] for item in read_events(run_dir)
    )
    if duplicate_file is not None or duplicate_legacy:
        raise ValueError(f"duplicate event_id: {payload['event_id']}")
    number = len(existing) + 1
    path = event_dir / f"event-{number:08d}-{payload['event_id']}.json"
    write_new_json(path, payload)


def latest_sample_statuses(events: list[dict]) -> dict[str, str]:
    statuses = {}
    for event in events:
        sample_key = event.get("sample_key")
        status = event.get("status")
        if isinstance(sample_key, str) and isinstance(status, str):
            statuses[sample_key] = status
    return statuses


def excluded_sample_keys(events: list[dict], *, retry_errors: bool) -> set[str]:
    statuses = latest_sample_statuses(events)
    if retry_errors:
        return {key for key, status in statuses.items() if status not in RETRYABLE_STATUSES}
    return set(statuses)


def plans(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "plans").glob("plan-*.json"))


def read_plan(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"unsupported plan schema: {path}")
    claimed = payload.get("plan_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "plan_sha256"}
    if not isinstance(claimed, str) or claimed != sha256_json(unsigned):
        raise ValueError(f"plan integrity check failed: {path}")
    return payload


def pending_plan(
    run_dir: Path, events: list[dict], *, retry_errors: bool
) -> tuple[Path, dict] | None:
    excluded = excluded_sample_keys(events, retry_errors=retry_errors)
    for path in plans(run_dir):
        payload = read_plan(path)
        if any(item["sample_key"] not in excluded for item in payload["samples"]):
            return path, payload
    return None


def create_plan(
    run_dir: Path,
    *,
    samples: list[dict],
    seed: int,
    sampling_method: str,
) -> tuple[Path, dict]:
    number = len(plans(run_dir)) + 1
    path = run_dir / "plans" / f"plan-{number:06d}.json"
    payload = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "plan_id": f"plan-{number:06d}",
        "seed": seed,
        "sampling_method": sampling_method,
        "sample_count": len(samples),
        "samples": samples,
        "created_at": utc_now(),
    }
    payload["plan_sha256"] = sha256_json(payload)
    write_new_json(path, payload)
    return path, payload


def event_summary(events: list[dict]) -> dict:
    statuses = latest_sample_statuses(events)
    counts = {
        status: sum(value == status for value in statuses.values())
        for status in sorted(set(statuses.values()))
    }
    scored = [event for event in events if event.get("status") == "scored"]
    latest_scored = {}
    for event in scored:
        latest_scored[event["sample_key"]] = event
    correct = sum(bool(event.get("correct")) for event in latest_scored.values())
    total = len(latest_scored)
    usage_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "reasoning_tokens": 0,
        "cost": 0.0,
    }
    usage_records = 0
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        usage_records += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens"):
            value = usage.get(key)
            if isinstance(value, int | float) and not isinstance(value, bool):
                usage_totals[key] += value
        if not isinstance(usage.get("reasoning_tokens"), int | float):
            details = usage.get("completion_tokens_details")
            nested_reasoning = (
                details.get("reasoning_tokens") if isinstance(details, dict) else None
            )
            if isinstance(nested_reasoning, int | float) and not isinstance(nested_reasoning, bool):
                usage_totals["reasoning_tokens"] += nested_reasoning
        cost = usage.get("cost")
        if isinstance(cost, int | float) and not isinstance(cost, bool):
            usage_totals["cost"] += cost
    return {
        "attempted_samples": len(statuses),
        "latest_status_counts": counts,
        "scored_samples": total,
        "correct_samples": correct,
        "accuracy": correct / total if total else None,
        "usage_records": usage_records,
        "usage_totals": usage_totals,
    }
