"""Candidate tests: contract, determinism, symmetries, erasure, exact evidence."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "world"
EVIDENCE = ROOT / "evidence"
sys.path.insert(0, str(WORLD))

import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402
import verify  # noqa: E402

STRESS_SEEDS = (101, 2027, 9929)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _scenes(seed: int, n: int = 24):
    return [renderer.sample_scene(seed * 1000 + i) for i in range(n)]


def test_templates_have_equal_ink_and_separation():
    inks = {
        name: sum(sum(row) for row in renderer._matrix(rows))
        for name, rows in renderer.TEMPLATES.items()
    }
    assert set(inks.values()) == {16}
    names = list(renderer.SYMBOL_NAMES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            gap = renderer._hamming(
                renderer._matrix(renderer.TEMPLATES[a]),
                renderer._matrix(renderer.TEMPLATES[b]),
            )
            assert gap >= 8, (a, b, gap)


def test_oracle_templates_match_renderer_templates():
    assert set(oracle.SYMBOL_TEMPLATES) == set(renderer.TEMPLATES)
    for name, rows in oracle.SYMBOL_TEMPLATES.items():
        mine = [[int(ch) for ch in row] for row in rows]
        assert mine == renderer._matrix(renderer.TEMPLATES[name]), name


def test_oracle_imports_no_world_code():
    text = (WORLD / "oracle.py").read_text(encoding="utf-8")
    for forbidden in ("import renderer", "import prompts", "import generate", "import verify"):
        assert forbidden not in text


@pytest.mark.parametrize("seed", STRESS_SEEDS)
def test_pixel_arm_agrees_with_analytic_gold(seed: int):
    for scene in _scenes(seed):
        gold = renderer.analytic_gold(scene)
        decision = oracle.decision_from_image(renderer.render(scene))
        if renderer.is_quarantined(scene):
            assert decision in (gold, oracle.ABSTAIN)
        else:
            assert decision == gold


@pytest.mark.parametrize("seed", STRESS_SEEDS)
def test_render_is_deterministic(seed: int):
    for scene in _scenes(seed, 6):
        assert _png(renderer.render(scene)) == _png(renderer.render(scene))
        assert renderer.sample_scene(seed) == renderer.sample_scene(seed)


def test_declared_symmetries_are_pixel_identical_and_gold_preserving():
    for scene in _scenes(101, 8):
        base_png = _png(renderer.render(scene))
        gold = renderer.analytic_gold(scene)
        transforms = renderer.latent_symmetries(scene)
        assert transforms
        for name, other in transforms:
            assert other != scene, name
            assert _png(renderer.render(other)) == base_png, name
            assert renderer.analytic_gold(other) == gold, name


def test_evidence_erasure_changes_or_collapses_the_decision():
    changed = 0
    for scene in _scenes(101, 12):
        gold = renderer.analytic_gold(scene)
        blanked = dict(scene)
        blanked["rows"] = ["." * renderer.COLS] * renderer.ROWS
        assert oracle.decision_from_image(renderer.render(blanked)) != gold or (
            renderer.analytic_gold(blanked) != gold
        )
        swapped = dict(scene)
        swapped["marker_codes"] = [code + 1 for code in scene["marker_codes"]]
        if renderer.analytic_gold(swapped) != gold:
            changed += 1
        assert oracle.decision_from_image(
            renderer.render(swapped)
        ) == renderer.analytic_gold(swapped)
    assert changed >= 10


def test_class_balance_and_boundary_coverage():
    scenes = _scenes(101)
    decisions = {renderer.analytic_gold(s) for s in scenes if not renderer.is_quarantined(s)}
    assert len(decisions) >= 2
    margins = [renderer.margin(s) for s in scenes]
    assert min(margins) < 6.0
    assert any(renderer.is_quarantined(s) for s in scenes)


def test_manifest_rows_are_complete_and_consistent():
    rows = [
        json.loads(line)
        for line in (EVIDENCE / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    required = {
        "example_id",
        "scene_id",
        "seed",
        "scene",
        "image_path",
        "prompt_family",
        "question",
        "decision",
        "answer",
        "candidates",
        "margin",
        "quarantined",
    }
    families = set()
    for row in rows:
        assert required <= set(row)
        assert (EVIDENCE / row["image_path"]).exists()
        families.add(row["prompt_family"])
    assert families == set(prompts.PROMPT_FAMILIES)


def test_exact_submitted_evidence_verifies():
    assert verify.check(EVIDENCE) == []


def test_corrupted_manifest_label_fails_verification(tmp_path: Path):
    target = tmp_path / "dataset"
    (target / "gallery").mkdir(parents=True)
    for png in (EVIDENCE / "gallery").glob("*.png"):
        (target / "gallery" / png.name).write_bytes(png.read_bytes())
    rows = [
        json.loads(line)
        for line in (EVIDENCE / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    wrong = {"ring": "wedge", "wedge": "ring", "zigzag": "arrow", "arrow": "slash", "slash": "zigzag"}
    rows[0]["decision"] = wrong[rows[0]["decision"]]
    (target / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )
    assert verify.check(target) != []


def test_corrupted_raster_fails_verification(tmp_path: Path):
    target = tmp_path / "dataset"
    (target / "gallery").mkdir(parents=True)
    for png in (EVIDENCE / "gallery").glob("*.png"):
        (target / "gallery" / png.name).write_bytes(png.read_bytes())
    (target / "manifest.jsonl").write_bytes((EVIDENCE / "manifest.jsonl").read_bytes())
    victim = target / "gallery" / "scene_0000.png"
    with Image.open(victim) as handle:
        image = handle.convert("RGB")
    image.paste((255, 255, 255), (0, 0, image.width, image.height // 2))
    image.save(victim)
    assert verify.check(target) != []


def test_generation_is_reproducible(tmp_path: Path):
    outs = []
    for i in range(2):
        out = tmp_path / f"run{i}"
        subprocess.run(
            [sys.executable, "generate.py", "--out", str(out), "--n", "4", "--seed", "2027"],
            cwd=WORLD,
            check=True,
        )
        outs.append(out)
    assert (outs[0] / "manifest.jsonl").read_bytes() == (outs[1] / "manifest.jsonl").read_bytes()
    for png in (outs[0] / "gallery").glob("*.png"):
        assert png.read_bytes() == (outs[1] / "gallery" / png.name).read_bytes()
    assert verify.check(outs[0]) == []
