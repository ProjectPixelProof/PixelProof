from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

import oracle
import renderer
import verify

CANDIDATE = Path(__file__).resolve().parents[1]


def test_balanced_stress_sweep() -> None:
    seen = set()
    for seed in range(80):
        scene = renderer.sample_scene(seed)
        decision = renderer.analytic_gold(scene)
        image = renderer.render(scene)
        seen.add(decision)
        assert oracle.decision_from_image(image) == decision
        assert renderer.margin(scene) >= 8.0
        assert all(sum(row) == 2 for row in scene["rows"])
        assert all(sum(scene["rows"][r][c] for r in range(4)) == 2 for c in range(4))
    assert seen == set(renderer.DECISIONS)


def test_card_erasure_and_blank_abstain() -> None:
    image = renderer.render(renderer.sample_scene(101))
    assert oracle.decision_from_image(Image.new("RGB", image.size, "white")) == "abstain"
    damaged = image.copy()
    ImageDraw.Draw(damaged).rectangle((0, 0, 230, 210), fill=renderer.BACKGROUND)
    assert oracle.decision_from_image(damaged) == "abstain"


def test_declared_alias_and_exact_evidence() -> None:
    assert renderer.latent_symmetries(renderer.sample_scene(5)) == []
    assert verify.verify_dataset(CANDIDATE / "evidence") == []
    rows = [json.loads(line) for line in (CANDIDATE / "evidence/manifest.jsonl").read_text().splitlines()]
    assert rows
