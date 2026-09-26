"""Tests for the global-elongation world."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "world"))

import renderer
import oracle


def test_scene_render_size():
    scene = renderer.sample_scene(1234)
    img = renderer.render(scene)
    assert img.size == (renderer.WIDTH, renderer.HEIGHT)


def test_all_four_axes_occur():
    seen = set()
    for seed in range(500):
        scene = renderer.sample_scene(seed * 97 + 3)
        seen.add(renderer.analytic_gold(scene))
    assert renderer.analytic_gold(renderer.sample_scene(5)) in {
        "horizontal",
        "rising",
        "vertical",
        "falling",
    }
    assert {"horizontal", "rising", "vertical", "falling"} <= seen


def test_oracle_matches_gold():
    for i in range(150):
        scene = renderer.sample_scene(1009 + i * 131)
        img = renderer.render(scene).convert("RGB")
        assert oracle.decision_from_image(img) == renderer.analytic_gold(scene)


def test_off_quarantine_balanced_over_seed():
    classes = set()
    for seed in range(200):
        scene = renderer.sample_scene(seed * 503 + 7)
        if not renderer.is_quarantined(scene):
            classes.add(renderer.analytic_gold(scene))
    assert len(classes) >= 2


def test_quarantine_on_circular_cloud():
    rng = np.random.default_rng(7)
    pts = (rng.normal(0, 1, (400, 2)) * 40).tolist()
    scene = {"dots": [list(p) for p in pts]}
    assert renderer.is_quarantined(scene)


def test_latent_symmetries_empty():
    assert renderer.latent_symmetries(renderer.sample_scene(5)) == []


def test_margin_nonnegative():
    for seed in range(100):
        scene = renderer.sample_scene(seed * 211 + 1)
        assert renderer.margin(scene) >= 0
