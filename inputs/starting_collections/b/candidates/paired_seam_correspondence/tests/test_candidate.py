from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image


WORLD = Path(__file__).resolve().parents[1] / "world"
sys.path.insert(0, str(WORLD))

import oracle  # noqa: E402
import prompts  # noqa: E402
import renderer  # noqa: E402


def test_stress_samples_are_balanced_and_pixel_answerable() -> None:
    decisions = []
    for seed in range(101, 141):
        scene = renderer.sample_scene(seed)
        image = renderer.render(scene)
        decision = renderer.analytic_gold(scene)
        assert oracle.decision_from_image(image) == decision
        assert renderer.margin(scene) > 0
        assert not renderer.is_quarantined(scene)
        decisions.append(decision)
    assert set(decisions) == {"yes", "no"}


def test_render_is_deterministic_and_prompt_api_is_stable() -> None:
    scene = renderer.sample_scene(2027)
    first = renderer.render(scene)
    second = renderer.render(scene)
    assert first.tobytes() == second.tobytes()
    assert set(prompts.PROMPT_FAMILIES) == {"pf1_direct", "pf2_fit", "pf3_contour"}
    for family in prompts.PROMPT_FAMILIES:
        assert prompts.candidates_for(family) == ("yes", "no")


def test_missing_or_erased_component_is_not_answered() -> None:
    scene = renderer.sample_scene(2027)
    image = renderer.render(scene)
    pixels = image.load()
    for y in range(120, 400):
        for x in range(70, 250):
            pixels[x, y] = (255, 255, 255)
    with pytest.raises(ValueError):
        oracle.decision_from_image(image)


def test_oracle_does_not_accept_blank_raster() -> None:
    with pytest.raises(ValueError):
        oracle.decision_from_image(Image.new("RGB", (512, 512), "white"))
