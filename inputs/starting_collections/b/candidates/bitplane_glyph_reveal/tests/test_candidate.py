import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "world"))

import renderer
import oracle
import prompts

from PIL import Image, ImageChops


def test_scene_render_and_gold():
    for i in range(20):
        scene = renderer.build_scene(noise=10000 + i, glyph=("A" if i % 2 == 0 else "B"))
        img = renderer.render(scene)
        assert img.size == (renderer.W, renderer.H)
        assert renderer.analytic_gold(scene) in ("A", "B")
        assert not renderer.is_quarantined(scene)


def test_pixel_oracle_matches_gold():
    for i in range(30):
        scene = renderer.build_scene(noise=20000 + i, glyph=("A" if i % 2 == 0 else "B"))
        img = renderer.render(scene)
        assert oracle.decision_from_image(img) == renderer.analytic_gold(scene)


def test_lsb_plane_recovers_glyph():
    scene = renderer.build_scene(noise=7, glyph="A")
    field = oracle._recover_bitfield(renderer.render(scene))
    ons = field[renderer.BOX_R0:renderer.BOX_R1, renderer.BOX_C0:renderer.BOX_C1]
    expected = oracle._template_mask("A")
    assert bool((ons & expected).sum())
    assert oracle.decision_from_image(renderer.render(scene)) == "A"


def test_latent_symmetry_empty():
    scene = renderer.build_scene(noise=7, glyph="A")
    assert list(renderer.latent_symmetries(scene)) == []


def test_determinism():
    a = renderer.render(renderer.build_scene(noise=123, glyph="B"))
    b = renderer.render(renderer.build_scene(noise=123, glyph="B"))
    assert ImageChops.difference(a, b).getbbox() is None


def test_prompt_contract():
    assert set(prompts.PROMPT_FAMILIES) == {
        "pf1_bitplane_ab",
        "pf2_lsb_ab",
        "pf3_sum_lowbit_ab",
    }
    for fam in prompts.PROMPT_FAMILIES:
        assert prompts.candidates_for(fam) == ("A", "B")
        assert prompts.correct_answer(fam, "A") == "A"
        assert prompts.correct_answer(fam, "B") == "B"
