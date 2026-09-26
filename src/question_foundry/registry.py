"""Append-only registry and immutable seed-snapshot helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from pathlib import Path
from typing import Any

_CANDIDATE_TRANSIENT_PARTS = frozenset({".pytest_cache", "__pycache__"})
_CANDIDATE_TRANSIENT_SUFFIXES = frozenset({".pyc", ".pyo"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_tree(root: Path) -> str:
    files = {}
    for path in sorted(
        item for item in root.rglob("*") if item.is_file() and not item.is_symlink()
    ):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[path.relative_to(root).as_posix()] = digest
    return hashlib.sha256(canonical_json(files).encode()).hexdigest()


def sha256_candidate_tree(root: Path) -> str:
    """Hash authored candidate content without verifier/runtime cache files."""
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if any(part in _CANDIDATE_TRANSIENT_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in _CANDIDATE_TRANSIENT_SUFFIXES:
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashlib.sha256(canonical_json(files).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{number}: registry record must be an object")
        records.append(value)
    return records


def append_record(path: Path, record: dict, *, id_field: str = "record_id") -> None:
    record_id = record.get(id_field)
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"record requires non-empty {id_field}")
    existing = read_jsonl(path)
    if any(item.get(id_field) == record_id for item in existing):
        raise ValueError(f"duplicate {id_field}: {record_id}")
    path.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(record) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(descriptor, line.encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_seed_snapshot(
    *,
    base_seed_path: Path,
    admission_records: list[dict],
    snapshot_id: str,
    version: str,
) -> dict:
    base = tomllib.loads(base_seed_path.read_text(encoding="utf-8"))
    worlds = list(base["worlds"])
    admission_ids = []
    for record in admission_records:
        if record.get("disposition") != "promoted":
            continue
        if record.get("mechanically_eligible") is not True:
            raise ValueError(f"promoted admission {record.get('record_id')} is not eligible")
        world_id = record.get("world_id")
        if not isinstance(world_id, str) or not world_id:
            raise ValueError("promoted admission has no world_id")
        if world_id not in worlds:
            worlds.append(world_id)
        admission_ids.append(record["record_id"])
    return {
        "id": snapshot_id,
        "version": version,
        "status": "draft",
        "parent_seed_set": f"{base['id']}@{base['version']}",
        "worlds": worlds,
        "include": list(base["include"]),
        "admission_record_ids": admission_ids,
        "visibility": dict(base.get("visibility", {})),
    }


def seed_snapshot_toml(snapshot: dict) -> str:
    lines = [
        f"id = {json.dumps(snapshot['id'])}",
        f"version = {json.dumps(snapshot['version'])}",
        f"status = {json.dumps(snapshot['status'])}",
        f"parent_seed_set = {json.dumps(snapshot['parent_seed_set'])}",
        "worlds = " + json.dumps(snapshot["worlds"]),
        "include = " + json.dumps(snapshot["include"]),
        "admission_record_ids = " + json.dumps(snapshot["admission_record_ids"]),
        "",
        "[visibility]",
    ]
    for key, value in sorted(snapshot["visibility"].items()):
        lines.append(f"{key} = {str(bool(value)).lower()}")
    return "\n".join(lines) + "\n"
