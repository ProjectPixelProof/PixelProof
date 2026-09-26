from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import oracle
import renderer


ROOT = Path(__file__).resolve().parents[1]


def test_pixel_oracle_matches_balanced_scenes() -> None:
    decisions = set()
    for seed in range(36):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        decision = renderer.analytic_gold(scene)
        assert oracle.decision_from_image(image) == decision
        assert renderer.margin(scene) >= 2.0
        assert not renderer.is_quarantined(scene)
        decisions.add(decision)
    assert decisions == {"amber", "cyan", "violet"}


def test_deterministic_and_no_declared_aliases() -> None:
    scene = renderer.sample_scene(2027)
    assert renderer.sample_scene(2027) == scene
    assert renderer.render(scene).tobytes() == renderer.render(scene).tobytes()
    assert renderer.latent_symmetries(scene) == []


def test_blank_raster_is_rejected() -> None:
    with pytest.raises(oracle.OracleError):
        oracle.decision_from_image(Image.new("RGB", (256, 256), "white"))


def test_exact_submitted_evidence() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "world/verify.py"), "--dataset", str(ROOT / "evidence")],
        cwd=ROOT / "world",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
