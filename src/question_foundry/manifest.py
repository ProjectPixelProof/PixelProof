"""Task-agnostic ground-truth ledger for rendered world examples."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST_SCHEMA_VERSION = "manifest-0.3.0"


@dataclass(frozen=True)
class ExampleRecord:
    example_id: str
    image_path: str
    split: str
    seed: int
    world: str
    scene: dict
    targets: dict
    prompt_family: str
    prompt_text: str
    renderer_version: str
    schema_version: str = MANIFEST_SCHEMA_VERSION
    extra: dict = field(default_factory=dict)


def write_manifest(records: list[ExampleRecord], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dataclasses.asdict(record)) + "\n")


def read_manifest(path: str | Path) -> Iterator[ExampleRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            # Forward-only compatibility for imported historical manifests.
            if "world" not in payload and "harness" in payload:
                payload["world"] = payload.pop("harness")
            yield ExampleRecord(**payload)
