from __future__ import annotations

import sys
from pathlib import Path

WORLD = Path(__file__).resolve().parents[1] / "world"
sys.path.insert(0, str(WORLD))

import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402


def test_analytic_and_pixel_decisions_agree() -> None:
    for seed in range(20):
        scene = renderer.sample_scene(seed)
        assert oracle.decision_from_image(renderer.render(scene)) == renderer.analytic_gold(scene)


def test_prompt_families_preserve_decision() -> None:
    for decision in ("yes", "no"):
        for family in prompts.PROMPT_FAMILIES:
            assert prompts.correct_answer(family, decision) == decision


def test_declared_symmetry_is_pixel_identical() -> None:
    scene = renderer.sample_scene(0)
    for _, twin in renderer.latent_symmetries(scene):
        assert renderer.render(scene).tobytes() == renderer.render(twin).tobytes()
        assert renderer.analytic_gold(scene) == renderer.analytic_gold(twin)
