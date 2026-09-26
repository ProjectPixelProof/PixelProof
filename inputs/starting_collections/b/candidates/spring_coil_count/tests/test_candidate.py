from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

import oracle
import renderer


ROOT = Path(__file__).resolve().parents[1]


def test_stress_sample_pixel_agreement() -> None:
    seen = set()
    for seed in (101, 2027, 9929, 9930, 9931, 9932, 9933):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        assert oracle.decision_from_image(image) == renderer.analytic_gold(scene)
        assert renderer.margin(scene) >= 12.0
        seen.add(renderer.analytic_gold(scene))
    assert len(seen) >= 4


def test_blank_and_erasure_abstain() -> None:
    image = renderer.render(renderer.sample_scene(101))
    blank = Image.new("RGB", image.size, "white")
    try:
        oracle.decision_from_image(blank)
        assert False, "blank image should abstain"
    except ValueError:
        pass
    erased = image.copy()
    x0 = image.width // 2 - 24
    for x in range(x0, x0 + 48):
        for y in range(image.height):
            erased.putpixel((x, y), (250, 249, 246))
    try:
        oracle.decision_from_image(erased)
        assert False, "route erasure should abstain"
    except ValueError:
        pass


def test_exact_submitted_evidence() -> None:
    completed = subprocess.run(
        [sys.executable, str(ROOT / "world/verify.py"), "--dataset", str(ROOT / "evidence")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    rows = [json.loads(line) for line in (ROOT / "evidence/manifest.jsonl").read_text().splitlines()]
    assert len(rows) >= 10

