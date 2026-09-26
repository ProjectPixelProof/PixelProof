"""Load discovered worlds into :class:`~question_foundry.gates.WorldBundle`s and
run the answerability gates over the bank (SSOT §7.14).

Lives in the world layer (``worlds/_common``) rather than core because it imports
world modules by name — core (``src/question_foundry``) never imports a specific
world. The gate functions are task-agnostic and live in
``question_foundry.gates``; this module only constructs bundles from disk and is
shared by ``tests/test_gates.py`` and ``scripts/gates_report.py``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from question_foundry.gates import WorldBundle, run_all
from question_foundry.manifest import read_manifest
from question_foundry.worlds import discover_worlds, get_world

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDITS_DIR = REPO_ROOT / "docs" / "audits"


def gate_worlds() -> list[str]:
    """Every discovered world that declares a ``[gold]`` block.

    Do not filter on manifest presence here: a promoted world missing its smoke
    evidence must enter the matrix and fail closed, not disappear from the report.
    """
    return sorted(info.name for info in discover_worlds() if info.meta.get("gold"))


def _gallery_ready(world: str, records: list) -> bool:
    """Check that gallery inputs cover every declared prompt family."""
    try:
        from scripts.make_gallery import collect_scene_rows

        info = get_world(world)
        pooled = list(records)
        for split in info.meta["gold"].get("gallery_pool_splits", []):
            manifest = info.path / "data/manifests" / f"{split}.jsonl"
            if not manifest.is_file():
                return False
            pooled.extend(read_manifest(manifest))
        rows = collect_scene_rows(pooled)
        covered = {family for row in rows for family in row.family_golds}
        declared = set(importlib.import_module(f"worlds.{world}.prompts").PROMPT_FAMILIES)
        return bool(rows) and declared <= covered
    except Exception:
        return False


def _resolve_scene_from_dict(world: str, renderer):
    """Older worlds keep ``_scene_from_dict`` in ``verify.py``; newer ones expose
    ``renderer.scene_from_dict``. Mirror gold_recheck's resolution order."""
    try:
        verify = importlib.import_module(f"worlds.{world}.verify")
        if hasattr(verify, "_scene_from_dict"):
            return verify._scene_from_dict
    except ModuleNotFoundError:
        pass
    if hasattr(renderer, "scene_from_dict"):
        return renderer.scene_from_dict
    return lambda d: renderer.Scene(**d)


def load_bundle(world: str) -> WorldBundle:
    info = get_world(world)
    renderer = importlib.import_module(f"worlds.{world}.renderer")
    oracle = importlib.import_module(f"worlds.{world}.oracle")
    prompts = importlib.import_module(f"worlds.{world}.prompts")
    smoke = info.path / "data/manifests/smoke.jsonl"
    records = list(read_manifest(smoke)) if smoke.is_file() else []
    supplemental = []
    manifests_dir = info.path / "data/manifests"
    if manifests_dir.is_dir():
        for manifest in sorted(manifests_dir.glob("*.jsonl")):
            if manifest != smoke:
                supplemental.extend(read_manifest(manifest))
    return WorldBundle(
        name=world,
        renderer=renderer,
        oracle=oracle,
        prompts=prompts,
        gold=info.meta["gold"],
        smoke_records=records,
        supplemental_records=supplemental,
        path=info.path,
        audits_dir=AUDITS_DIR,
        scene_from_dict=_resolve_scene_from_dict(world, renderer),
    )


def run_world(world: str, rng) -> dict:
    bundle = load_bundle(world)
    gallery_ok = _gallery_ready(world, bundle.smoke_records)
    return run_all(bundle, rng, gallery_ok=gallery_ok)
