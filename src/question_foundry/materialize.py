"""Deterministically materialize minimal Harbor inner-builder datasets.

The canonical repository is never mounted wholesale into an inner trial. This
module copies the static Harbor task plus only the protocol and seed files named
by a seed-set manifest, then hashes the resulting packet.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tomllib
from dataclasses import dataclass
from pathlib import Path

_WORLD_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_PROTOCOL_FILES = (
    "brief.md",
    "candidate.schema.json",
    "protocol.toml",
)


@dataclass(frozen=True)
class MaterializationResult:
    """Location and identity of one immutable inner-task packet."""

    root: Path
    dataset: Path
    manifest: Path
    packet_sha256: str
    file_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _packet_hash(files: dict[str, str]) -> str:
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def materialize_inner_dataset(
    *,
    repo_root: Path,
    seed_set_path: Path,
    output_root: Path,
    source_revision: str,
    source_dirty: bool,
) -> MaterializationResult:
    """Create a new minimal Harbor dataset and refuse to overwrite an old packet."""

    repo_root = repo_root.resolve()
    seed_set_path = seed_set_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"materialization target already exists: {output_root}")

    seed_set = tomllib.loads(seed_set_path.read_text(encoding="utf-8"))
    worlds = seed_set.get("worlds")
    include = seed_set.get("include")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError("seed set must declare a non-empty worlds list")
    if not isinstance(include, list) or not include:
        raise ValueError("seed set must declare a non-empty include list")
    if any(not isinstance(item, str) or not _WORLD_ID.fullmatch(item) for item in worlds):
        raise ValueError("seed set contains an invalid world ID")
    if any(
        not isinstance(item, str) or Path(item).name != item or item.startswith(".")
        for item in include
    ):
        raise ValueError("seed include entries must be plain visible filenames")
    if not isinstance(seed_set.get("id"), str) or not isinstance(seed_set.get("version"), str):
        raise ValueError("seed set must declare string id and version fields")

    template = repo_root / "harbor/datasets/inner-builder-v0"
    dataset = output_root / "inner-builder-v0"
    shutil.copytree(
        template,
        dataset,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )
    starter = dataset / "propose-question-world/environment/starter"

    protocol_source = repo_root / "foundry/protocols/question-world/v0.1"
    for name in _PROTOCOL_FILES:
        _copy_file(protocol_source / name, starter / "protocol" / name)
    _copy_file(seed_set_path, starter / "SEED_SET.toml")

    for world in worlds:
        for name in include:
            _copy_file(repo_root / "worlds" / world / name, starter / "seeds" / world / name)

    files: dict[str, str] = {}
    for path in sorted(item for item in dataset.rglob("*") if item.is_file()):
        relative = path.relative_to(output_root).as_posix()
        files[relative] = _sha256(path)
    packet_sha256 = _packet_hash(files)
    manifest_data = {
        "schema_version": "0.1.0",
        "kind": "inner-builder-dataset",
        "seed_set": f"{seed_set['id']}@{seed_set['version']}",
        "source_revision": source_revision,
        "source_dirty": source_dirty,
        "packet_sha256": packet_sha256,
        "files": files,
        "visibility": seed_set.get("visibility", {}),
    }
    manifest = output_root / "materialization.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return MaterializationResult(
        root=output_root,
        dataset=dataset,
        manifest=manifest,
        packet_sha256=packet_sha256,
        file_count=len(files),
    )
