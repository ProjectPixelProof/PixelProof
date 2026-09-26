from __future__ import annotations

from PIL import Image, ImageChops

import oracle
import prompts
import renderer


def test_deterministic_render_and_pixel_inverse() -> None:
    for seed in (100, 101, 102, 2027, 2028, 9929, 9930):
        first = renderer.sample_scene(seed)
        second = renderer.sample_scene(seed)
        assert first == second
        assert ImageChops.difference(renderer.render(first), renderer.render(second)).getbbox() is None
        decision = renderer.analytic_gold(first)
        assert decision in prompts.candidates_for("pf1_curvature_rank")
        assert oracle.decision_from_image(renderer.render(first)) == decision
        assert renderer.margin(first) == 2.0
        assert renderer.is_quarantined(first) is False


def test_prompt_families_share_the_constrained_choice() -> None:
    scene = renderer.sample_scene(101)
    decision = renderer.analytic_gold(scene)
    for family in prompts.PROMPT_FAMILIES:
        assert prompts.correct_answer(family, decision) == decision
        assert prompts.candidates_for(family) == ("A", "B")


def test_answer_identity_is_balanced_by_the_scene_sampler() -> None:
    decisions = [renderer.analytic_gold(renderer.sample_scene(seed)) for seed in range(100, 112)]
    assert decisions.count("A") == decisions.count("B") == 6


def test_candidate_panels_keep_the_same_profile_multiset() -> None:
    for seed in (100, 101, 2027, 9929):
        scene = renderer.sample_scene(seed)
        for panel in (scene["reference"], scene["candidate_a"], scene["candidate_b"]):
            assert {arc["color"] for arc in panel} == set(renderer.COLORS)
            assert {arc["profile"] for arc in panel} == set(range(4))
        assert sorted(arc["profile"] for arc in scene["candidate_a"]) == [0, 1, 2, 3]
        assert sorted(arc["profile"] for arc in scene["candidate_b"]) == [0, 1, 2, 3]


def test_blank_or_corrupted_raster_does_not_get_a_candidate_answer() -> None:
    scene = renderer.sample_scene(101)
    image = renderer.render(scene)
    blank = Image.new("RGB", image.size, "white")
    assert oracle.decision_from_image(blank) == "quarantine"
