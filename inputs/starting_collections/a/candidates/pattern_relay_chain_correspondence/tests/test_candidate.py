from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image, ImageChops, ImageDraw

from oracle import decision_from_image
from renderer import (
    START_FILL,
    analytic_gold,
    is_quarantined,
    latent_symmetries,
    margin,
    render,
    sample_scene,
)
from verify import verify


@pytest.mark.parametrize("seed", [0, 1, 2, 5, 11, 101, 2027, 9929])
def test_pixel_inverse_agrees(seed):
    latent = sample_scene(seed)
    assert decision_from_image(render(latent)) == analytic_gold(latent)


def test_sampling_and_rendering_are_deterministic():
    assert sample_scene(101) == sample_scene(101)
    assert render(sample_scene(101)).tobytes() == render(sample_scene(101)).tobytes()


def test_answer_color_and_terminal_position_are_balanced():
    scenes = [sample_scene(seed) for seed in range(60)]
    answers = [analytic_gold(latent) for latent in scenes]
    assert {color: answers.count(color) for color in ("red", "green", "blue")} == {
        "red": 20,
        "green": 20,
        "blue": 20,
    }
    terminal_slots = []
    for latent in scenes:
        current = latent["start_code"]
        last = None
        while True:
            matches = [relay for relay in latent["relays"] if relay["input"] == current]
            if not matches:
                break
            assert len(matches) == 1
            last = matches[0]
            current = last["output"]
        terminal_slots.append(last["slot"])
        other_end = next(
            relay for relay in latent["relays"]
            if relay["output"] not in {item["input"] for item in latent["relays"]}
            and relay is not last
        )
        assert other_end["tag"] != last["tag"]
    assert {slot: terminal_slots.count(slot) for slot in range(6)} == {slot: 10 for slot in range(6)}


def test_runner_up_boundary_and_quarantine_are_explicit():
    assert margin(sample_scene(7)) == 1.0
    assert is_quarantined(sample_scene(7))
    assert margin(sample_scene(8)) >= 3.0
    assert not is_quarantined(sample_scene(8))


def test_declared_storage_alias_is_pixel_identical():
    latent = sample_scene(2027)
    twins = latent_symmetries(latent)
    assert twins
    for _, twin in twins:
        assert ImageChops.difference(render(latent), render(twin)).getbbox() is None
        assert analytic_gold(twin) == analytic_gold(latent)


def _transpose_code(code: int) -> int:
    transformed = 0
    for row in range(3):
        for column in range(3):
            if code & (1 << (row * 3 + column)):
                transformed |= 1 << (column * 3 + row)
    return transformed


def test_glyph_relabeling_and_card_permutation_preserve_answer():
    latent = sample_scene(101)
    transformed = deepcopy(latent)
    transformed["start_code"] = _transpose_code(transformed["start_code"])
    for relay in transformed["relays"]:
        relay["input"] = _transpose_code(relay["input"])
        relay["output"] = _transpose_code(relay["output"])
        relay["slot"] = (relay["slot"] + 2) % 6
    assert analytic_gold(transformed) == analytic_gold(latent)
    assert decision_from_image(render(transformed)) == analytic_gold(latent)


def test_palette_permutation_transforms_answer_label():
    latent = sample_scene(102)
    transformed = deepcopy(latent)
    mapping = {"red": "green", "green": "blue", "blue": "red"}
    for relay in transformed["relays"]:
        relay["tag"] = mapping[relay["tag"]]
    assert analytic_gold(transformed) == mapping[analytic_gold(latent)]
    assert decision_from_image(render(transformed)) == mapping[analytic_gold(latent)]


def test_start_evidence_erasure_abstains():
    image = render(sample_scene(101))
    draw = ImageDraw.Draw(image)
    draw.rectangle((365, 82, 414, 125), fill=START_FILL)
    with pytest.raises(ValueError):
        decision_from_image(image)


def test_blank_raster_abstains():
    with pytest.raises(ValueError):
        decision_from_image(Image.new("RGB", (780, 500), "white"))


def test_exact_submitted_evidence():
    root = Path(__file__).resolve().parents[1]
    assert verify(root / "evidence") == []
