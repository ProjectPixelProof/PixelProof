"""Tests for the radial_density_regime world."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from oracle import decision_from_image
from prompts import PROMPT_FAMILIES, candidates_for, correct_answer
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
ANSWERS = ("yes", "no")


def _tmp_dataset(tmp_path: Path, n: int = 30, seed: int = 7) -> Path:
    out = tmp_path / "ds"
    subprocess.run(
        [sys.executable, str(GENERATE), "--out", str(out), "--n", str(n), "--seed", str(seed)],
        check=True,
        cwd=str(WORLD),
    )
    return out


def test_prompt_apis_consistent():
    for family in PROMPT_FAMILIES:
        assert tuple(candidates_for(family)) == ANSWERS
        core_ans = correct_answer(family, "core")
        rim_ans = correct_answer(family, "rim")
        assert core_ans in ANSWERS and rim_ans in ANSWERS
        assert core_ans != rim_ans


def test_analytic_gold_is_regime_label():
    for seed in range(100):
        scene = sample_scene(seed)
        assert analytic_gold(scene) in {"core", "rim"}


def test_margin_nonnegative_and_quarantine_logic():
    for seed in range(300):
        scene = sample_scene(seed)
        assert margin(scene) >= 0.0
        assert isinstance(is_quarantined(scene), bool)
        if is_quarantined(scene):
            assert margin(scene) < 14.0


def test_render_deterministic():
    scene = sample_scene(123)
    assert ImageChops.difference(render(scene), render(scene)).getbbox() is None


def test_pixel_oracle_matches_analytic_on_off_quarantine():
    checked = 0
    for seed in range(800):
        scene = sample_scene(seed)
        if is_quarantined(scene):
            continue
        assert decision_from_image(render(scene)) == analytic_gold(scene), f"seed {seed}"
        checked += 1
    assert checked >= 20


def test_latent_symmetry_pixel_identical_and_gold_preserved():
    exercised = 0
    for seed in range(30):
        scene = sample_scene(seed)
        image = render(scene)
        gold = analytic_gold(scene)
        for name, twin in latent_symmetries(scene):
            exercised += 1
            assert ImageChops.difference(image, render(twin)).getbbox() is None, name
            assert analytic_gold(twin) == gold, name
    assert exercised >= 1


def test_blank_and_wrong_answer_are_rejected(tmp_path):
    ds = _tmp_dataset(tmp_path)
    rows = [json.loads(line) for line in (ds / "manifest.jsonl").read_text().splitlines() if line.strip()]
    first = rows[0]
    replacement = next(v for v in first["candidates"] if v != first["answer"])

    def run_verify(dataset: Path) -> int:
        return subprocess.run(
            [sys.executable, str(VERIFY), "--dataset", str(dataset)],
            cwd=str(WORLD),
            capture_output=True,
        ).returncode

    assert run_verify(ds) == 0

    bad = [json.loads(line) for line in (ds / "manifest.jsonl").read_text().splitlines() if line.strip()]
    bad[0]["answer"] = replacement
    (ds / "manifest.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in bad), encoding="utf-8"
    )
    assert run_verify(ds) != 0

    blank = _tmp_dataset(tmp_path)
    for png in (blank / "gallery").glob("*.png"):
        Image.new("RGB", (256, 256), "white").save(png)
    assert run_verify(blank) != 0


def test_class_balance_in_generated_dataset(tmp_path):
    ds = _tmp_dataset(tmp_path, n=40)
    counts = Counter()
    for line in (ds / "manifest.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row["quarantined"]:
            counts[(row["prompt_family"], row["answer"])] += 1
    for family in PROMPT_FAMILIES:
        classes = {ans for (fam, ans) in counts if fam == family}
        assert len(classes) >= 2, family
