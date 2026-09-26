from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

import oracle
import prompts
import renderer
import verify


def test_balanced_pixel_agreement() -> None:
    decisions = set()
    for seed in range(64):
        scene = renderer.sample_scene(seed)
        decision = renderer.analytic_gold(scene)
        decisions.add(decision)
        assert oracle.decision_from_image(renderer.render(scene)) == decision
        assert renderer.margin(scene) >= 1.0
        assert not renderer.is_quarantined(scene)
    assert decisions == {"yes", "no"}


def test_palette_translation_and_scale_preserve_label() -> None:
    scene = renderer.sample_scene(8)
    expected = renderer.analytic_gold(scene)
    for index, color in enumerate(renderer.PALETTE):
        twin = dict(scene)
        twin["ink_rgb"] = list(color)
        twin["center_x"] += (index - 2) * 5
        twin["center_y"] -= (index - 2) * 4
        twin["radius"] = (114.0, 120.0, 126.0)[index % 3]
        assert oracle.decision_from_image(renderer.render(twin)) == expected


def test_declared_alias_is_pixel_identical() -> None:
    for seed in (7, 8):
        scene = renderer.sample_scene(seed)
        expected = renderer.render(scene).tobytes()
        transforms = renderer.latent_symmetries(scene)
        assert transforms
        for _, twin in transforms:
            assert twin != scene
            assert renderer.render(twin).tobytes() == expected
            assert renderer.analytic_gold(twin) == renderer.analytic_gold(scene)


def test_erasing_one_disconnected_fragment_forces_abstention() -> None:
    scene = renderer.sample_scene(10)
    image = renderer.render(scene)
    mask = oracle._ink_mask(image)
    component = oracle._components(mask)[0]
    x0, y0 = component.min(axis=0)
    x1, y1 = component.max(axis=0)
    damaged = image.copy()
    ImageDraw.Draw(damaged).rectangle(
        (int(x0) - 3, int(y0) - 3, int(x1) + 3, int(y1) + 3),
        fill=tuple(scene["background_rgb"]),
    )
    with pytest.raises(oracle.OracleAbstention):
        oracle.decision_from_image(damaged)


def test_blank_raster_abstains() -> None:
    with pytest.raises(oracle.OracleAbstention):
        oracle.decision_from_image(Image.new("RGB", (768, 768), "white"))


def test_prompt_contract() -> None:
    for family in prompts.PROMPT_FAMILIES:
        assert prompts.candidates_for(family) == ("yes", "no")
        assert prompts.correct_answer(family, "yes") in {"yes", "no"}
        assert prompts.correct_answer(family, "no") in {"yes", "no"}


def test_exact_submitted_evidence() -> None:
    dataset = Path(__file__).resolve().parents[1] / "evidence"
    assert verify.verify_dataset(dataset) == []
