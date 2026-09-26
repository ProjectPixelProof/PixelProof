"""Candidate tests for the interlocked-ring panel world.

They exercise the checked-in evidence dataset itself, not only a convenient
seed range: the submitted manifest is re-verified row by row, the pixel-only
arm is re-run over the stored PNGs, and every public stress seed is regenerated
and cross-checked analytically and from pixels.
"""

from __future__ import annotations

import ast
import collections
import copy
import json
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
WORLD = ROOT / "world"
EVIDENCE = ROOT / "evidence"
sys.path.insert(0, str(WORLD))

import generate as generate_module  # noqa: E402
import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402
import verify as verify_module  # noqa: E402

STRESS_SEEDS = (101, 2027, 9929)
STRESS_SCENES = 24
FORBIDDEN = {"renderer", "prompts", "generate", "verify"}


def _rows() -> list:
    return [
        json.loads(line)
        for line in (EVIDENCE / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_submitted_evidence_verifies() -> None:
    assert verify_module.verify(EVIDENCE) == []


def test_manifest_shape_and_required_keys() -> None:
    rows = _rows()
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
    assert rows
    for row in rows:
        assert required <= set(row)
        assert (EVIDENCE / row["image_path"]).is_file()
    families = {row["prompt_family"] for row in rows}
    assert families == set(prompts.PROMPT_FAMILIES)
    scenes = {row["scene_id"] for row in rows}
    assert len(rows) == len(scenes) * len(families)


def test_gallery_covers_every_answer_class() -> None:
    rows = [row for row in _rows() if row["prompt_family"] == "pf_panel"]
    counts = collections.Counter(row["decision"] for row in rows)
    assert set(counts) == set(prompts.candidates_for("pf_panel"))
    assert min(counts.values()) >= 2
    assert (EVIDENCE / "gallery" / "index.html").is_file()


def test_pixel_arm_recovers_gold_on_stored_images() -> None:
    for row in _rows():
        if row["prompt_family"] != "pf_panel":
            continue
        image = Image.open(EVIDENCE / row["image_path"]).convert("RGB")
        assert oracle.decision_from_image(image) == row["decision"]


def test_oracle_module_is_independent() -> None:
    tree = ast.parse((WORLD / "oracle.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & FORBIDDEN)
    code = oracle.decision_from_image.__code__
    assert code.co_varnames[: code.co_argcount] == ("image",)


@pytest.mark.parametrize("seed", STRESS_SEEDS)
def test_stress_seed_analytic_and_pixel_agree(seed: int) -> None:
    for index in range(STRESS_SCENES):
        scene = renderer.sample_scene(seed * 10_000 + index)
        image = renderer.render(scene)
        gold = renderer.analytic_gold(scene)
        assert gold in prompts.candidates_for("pf_panel")
        assert not renderer.is_quarantined(scene)
        assert renderer.margin(scene) >= renderer.QUARANTINE_MARGIN
        assert oracle.decision_from_image(image) == gold


def test_rendering_is_deterministic() -> None:
    scene = renderer.sample_scene(4242)
    first = renderer.render(scene)
    second = renderer.render(json.loads(json.dumps(scene)))
    assert ImageChops.difference(first, second).getbbox() is None


def test_declared_symmetries_are_pixel_identical_and_label_preserving() -> None:
    scene = renderer.sample_scene(777)
    base_image = renderer.render(scene)
    gold = renderer.analytic_gold(scene)
    transforms = renderer.latent_symmetries(scene)
    assert len(transforms) >= 1
    for name, alias in transforms:
        assert alias != scene, name
        assert ImageChops.difference(base_image, renderer.render(alias)).getbbox() is None, name
        assert renderer.analytic_gold(alias) == gold, name
        assert renderer.margin(alias) == pytest.approx(renderer.margin(scene)), name


def test_erasing_the_alternation_changes_the_answer_or_forces_abstention() -> None:
    """Letting the front ring win both crossings destroys the only evidence."""
    scene = renderer.sample_scene(2027 * 10_000 + 3)
    gold = renderer.analytic_gold(scene)
    panel = int(gold.split("-")[1]) - 1
    broken = copy.deepcopy(scene)
    broken["panels"][panel]["linked"] = False
    assert renderer.is_quarantined(broken)
    with pytest.raises(ValueError):
        renderer.analytic_gold(broken)
    assert oracle.decision_from_image(renderer.render(broken)) == oracle.AMBIGUOUS


def test_moving_the_alternation_moves_the_answer() -> None:
    scene = renderer.sample_scene(101 * 10_000 + 5)
    gold = renderer.analytic_gold(scene)
    panel = int(gold.split("-")[1]) - 1
    other = (panel + 2) % renderer.PANELS
    moved = copy.deepcopy(scene)
    moved["panels"][panel]["linked"] = False
    moved["panels"][other]["linked"] = True
    assert renderer.analytic_gold(moved) == f"panel-{other + 1}"
    assert oracle.decision_from_image(renderer.render(moved)) == f"panel-{other + 1}"


def test_corrupted_raster_is_not_silently_accepted() -> None:
    scene = renderer.sample_scene(9929 * 10_000 + 1)
    image = renderer.render(scene)
    blank = Image.new("RGB", image.size, (255, 255, 255))
    assert oracle.decision_from_image(blank) == oracle.AMBIGUOUS


def test_corrupted_manifest_label_fails_verification(tmp_path: Path) -> None:
    generate_module.build(tmp_path, n=3, seed=5)
    assert verify_module.verify(tmp_path) == []
    manifest = tmp_path / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    others = [c for c in prompts.candidates_for("pf_panel") if c != rows[0]["decision"]]
    rows[0]["decision"] = others[0]
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    assert verify_module.verify(tmp_path)


def test_quarantine_is_reachable_on_a_constructed_scene() -> None:
    """A near-tangent stacked pair leaves a sliver instead of a second piece."""
    scene = renderer.sample_scene(31)
    gold = renderer.analytic_gold(scene)
    panel_index = int(gold.split("-")[1]) - 1
    other = (panel_index + 1) % renderer.PANELS
    tight = copy.deepcopy(scene)
    ix0, iy0, ix1, iy1 = renderer.panel_inner(other)
    radius = 38
    cy = int((iy0 + iy1) / 2)
    cx = int(ix0) + radius + 4
    span = 2 * radius - 1
    tight["panels"][other]["rings"] = [
        {"cx": cx, "cy": cy, "r": radius, "color": 0},
        {"cx": cx + span, "cy": cy, "r": radius, "color": 1},
    ]
    assert renderer.margin(tight) < renderer.QUARANTINE_MARGIN
    assert renderer.is_quarantined(tight)
    assert oracle.decision_from_image(renderer.render(tight)) in (gold, oracle.AMBIGUOUS)


def test_prompt_families_are_constrained_and_declare_the_rules() -> None:
    for family, text in prompts.PROMPT_FAMILIES.items():
        options = prompts.candidates_for(family)
        assert len(options) == renderer.PANELS == len(set(options))
        assert "interlocked" in text and "exactly one break" in text
        assert "cross each other at exactly two places" in text
        for option in options:
            assert option in text
        with pytest.raises(ValueError):
            prompts.correct_answer(family, "panel-9")
    assert prompts.correct_answer("pf_position", "panel-1") == "leftmost"
    assert prompts.correct_answer("pf_position", "panel-5") == "rightmost"
