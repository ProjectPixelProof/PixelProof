from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from oracle import decision_from_image
from renderer import LABELS, analytic_gold, render, sample_scene


ROOT = Path(__file__).resolve().parents[1]


def test_pixel_inverse_stress() -> None:
    observed = set()
    for seed in range(80):
        scene = sample_scene(seed)
        decision = analytic_gold(scene)
        assert decision_from_image(render(scene)) == decision
        observed.add(decision)
    assert observed == set(LABELS)


def test_blank_image_is_rejected() -> None:
    blank = Image.new("RGB", (640, 560), "white")
    try:
        decision_from_image(blank)
    except ValueError:
        pass
    else:
        raise AssertionError("blank image was accepted")


def test_exact_submitted_evidence() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "world/verify.py"), "--dataset", str(ROOT / "evidence")],
        cwd=ROOT / "world",
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_scene_has_no_answer_field() -> None:
    scene = sample_scene(17)
    assert "answer" not in scene and "label" not in scene and "decision" not in scene
