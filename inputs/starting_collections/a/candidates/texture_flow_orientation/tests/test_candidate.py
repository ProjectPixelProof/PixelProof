"""Tests for the texture_flow_orientation world."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from oracle import decision_from_image
from prompts import CLASSES, PROMPT_FAMILIES, candidates_for, correct_answer
from renderer import (
    analytic_gold,
    is_quarantined,
    latent_symmetries,
    margin,
    render,
    sample_scene,
)

WORLD = Path(__file__).resolve().parents[1] / "world"
GENERATE = WORLD / "generate.py"
VERIFY = WORLD / "verify.py"


def _tmp_dataset(tmp_path: Path, n: int = 30, seed: int = 7) -> Path:
    import subprocess
    import sys

    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, str(GENERATE), "--out", str(out), "--n", str(n), "--seed", str(seed)],
        check=True,
        cwd=str(WORLD),
    )
    return out


def test_prompt_apis_consistent():
    for family in PROMPT_FAMILIES:
        assert tuple(candidates_for(family)) == CLASSES
        for cls in CLASSES:
            assert correct_answer(family, cls) == cls


def test_analytic_gold_is_class():
    for seed in range(80):
        scene = sample_scene(seed)
        assert analytic_gold(scene) in CLASSES


def test_margin_nonnegative_and_quarantine():
    for seed in range(300):
        scene = sample_scene(seed)
        assert margin(scene) >= 0.0
        flagged = is_quarantined(scene)
        assert isinstance(flagged, bool)
        if flagged:
            assert margin(scene) < 3.0


def test_render_deterministic():
    scene = sample_scene(123)
    assert ImageChops.difference(render(scene), render(scene)).getbbox() is None


def test_pixel_oracle_matches_analytic_on_off_quarantine():
    checked = 0
    for seed in range(800):
        scene = sample_scene(seed)
        if is_quarantined(scene):
            continue
        decision = decision_from_image(render(scene))
        assert decision == analytic_gold(scene), f"seed {seed} -> {decision}"
        checked += 1
    assert checked >= 30


def test_latent_symmetry_pixel_identical_and_gold_preserved():
    exercised = 0
    for seed in range(40):
        scene = sample_scene(seed)
        image = render(scene)
        gold = analytic_gold(scene)
        for name, twin in latent_symmetries(scene):
            exercised += 1
            assert ImageChops.difference(image, render(twin)).getbbox() is None, name
            assert analytic_gold(twin) == gold, name
    assert exercised >= 1


def test_blank_and_wrong_answer_are_rejected(tmp_path):
    import subprocess
    import sys

    ds = _tmp_dataset(tmp_path)
    rows = [json.loads(line) for line in (ds / "manifest.jsonl").read_text().splitlines() if line.strip()]
    first = rows[0]
    wrong = list(first["candidates"])
    wrong.remove(first["answer"])
    replacement = wrong[0]

    def run_verify(dataset: Path) -> int:
        return subprocess.run(
            [sys.executable, str(VERIFY), "--dataset", str(dataset)],
            cwd=str(WORLD),
            capture_output=True,
        ).returncode

    assert run_verify(ds) == 0

    bad = list(json.loads(line) for line in (ds / "manifest.jsonl").read_text().splitlines() if line.strip())
    bad[0]["answer"] = replacement
    (ds / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in bad), encoding="utf-8"
    )
    assert run_verify(ds) != 0

    blank_ds = _tmp_dataset(tmp_path)
    for png in (blank_ds / "gallery").glob("*.png"):
        Image.new("RGB", (256, 256), "white").save(png)
    assert run_verify(blank_ds) != 0


def test_class_balance_in_generated_dataset(tmp_path):
    import subprocess
    import sys
    from collections import Counter

    ds = _tmp_dataset(tmp_path, n=48)
    families = PROMPT_FAMILIES.keys()
    counts = Counter()
    for line in (ds / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row["quarantined"]:
            counts[(row["prompt_family"], row["answer"])] += 1
    for family in families:
        classes = {ans for (fam, ans) in counts if fam == family}
        assert len(classes) >= 2, family
