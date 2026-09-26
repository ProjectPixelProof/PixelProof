from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from oracle import OracleError, PALETTE, decision_from_image
from renderer import analytic_gold, latent_symmetries, margin, render, sample_scene
from verify import verify_dataset


def test_determinism_balance_and_oracle() -> None:
    decisions = []
    for seed in range(18):
        first = sample_scene(seed)
        second = sample_scene(seed)
        assert first == second
        assert render(first).tobytes() == render(second).tobytes()
        decision = analytic_gold(first)
        assert decision_from_image(render(first)) == decision
        assert margin(first) == 1.0
        decisions.append(decision)
    assert {color: decisions.count(color) for color in set(decisions)} == {
        "red": 6,
        "blue": 6,
        "orange": 6,
    }


def test_record_order_alias_is_pixel_identical() -> None:
    original = sample_scene(44)
    transforms = latent_symmetries(original)
    assert transforms
    for _, twin in transforms:
        assert analytic_gold(twin) == analytic_gold(original)
        assert render(twin).tobytes() == render(original).tobytes()


def test_translation_and_distractor_palette_preserve_label() -> None:
    original = sample_scene(52)
    decision = analytic_gold(original)
    shifted = copy.deepcopy(original)
    for node in shifted["nodes"]:
        node["x"] += 9
        node["y"] -= 7
    assert analytic_gold(shifted) == decision_from_image(render(shifted)) == decision

    swapped = copy.deepcopy(original)
    distractors = [n for n in swapped["nodes"] if n["kind"] == "terminal" and n["color"] != decision]
    distractors[0]["color"], distractors[1]["color"] = distractors[1]["color"], distractors[0]["color"]
    assert analytic_gold(swapped) == decision_from_image(render(swapped)) == decision


def test_arrow_erasure_causes_abstention() -> None:
    image = render(sample_scene(61))
    arr = np.asarray(image).copy()
    purple = np.asarray(PALETTE["arrow"])
    background = np.asarray(PALETTE["background"])
    mask = np.max(np.abs(arr.astype(int) - purple), axis=2) < 25
    arr[mask] = background
    with pytest.raises(OracleError):
        decision_from_image(Image.fromarray(arr.astype(np.uint8), "RGB"))


def test_exact_submitted_evidence() -> None:
    evidence = Path(__file__).resolve().parents[1] / "evidence"
    assert verify_dataset(evidence) == []
