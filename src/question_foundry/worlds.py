"""Discover standalone visual question worlds under :mod:`worlds`.

The task-agnostic core never imports a named world. Discovery reads only
``world.toml``; instance code is imported by explicit runners outside core.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def worlds_root() -> Path:
    """Return the world-bank directory, overridable for tests and materializers."""

    configured = os.environ.get("VLM_WORLDS_DIR")
    return Path(configured) if configured else _REPO_ROOT / "worlds"


@dataclass(frozen=True)
class WorldInfo:
    name: str
    path: Path
    description: str
    method_refs: list[str]
    meta: dict


def discover_worlds(root: Path | None = None) -> list[WorldInfo]:
    """Enumerate directories containing a valid ``world.toml`` contract."""

    root = root or worlds_root()
    if not root.exists():
        return []
    discovered: list[WorldInfo] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        config = directory / "world.toml"
        if not config.is_file():
            continue
        meta = tomllib.loads(config.read_text(encoding="utf-8"))
        world = meta.get("world", {})
        discovered.append(
            WorldInfo(
                name=world.get("name", directory.name),
                path=directory,
                description=world.get("description", ""),
                method_refs=list(world.get("method_refs", world.get("ssot_refs", []))),
                meta=meta,
            )
        )
    return discovered


def get_world(name: str, root: Path | None = None) -> WorldInfo:
    for info in discover_worlds(root):
        if info.name == name:
            return info
    raise KeyError(f"no visual question world named {name!r} under {root or worlds_root()}")
