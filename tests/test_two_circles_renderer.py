"""GPU-free tests for the two-circle world model (SSOT §7.3-§7.5).

Everything here runs from geometry + PIL only; no torch, no network.
"""

import math
import random

import pytest

from worlds.two_circles.renderer import (
    MARGIN_BINS,
    MIN_PROTRUSION_PX,
    RENDERER_VERSION,
    TwoCircleScene,
    bin_for_m,
    counterfactual_margin,
    render,
    sample_scene,
    scene_targets,
    translate_circle_1,
)


def _scene(m: float, r1: float = 50.0, r2: float = 70.0) -> TwoCircleScene:
    """Horizontal layout with an exact signed gap m = d - r1 - r2."""
    d = r1 + r2 + m
    return TwoCircleScene(c1=(100.0, 256.0), c2=(100.0 + d, 256.0), r1=r1, r2=r2)


class TestGeometry:
    def test_signed_gap_exact(self):
        scene = TwoCircleScene(c1=(100, 100), c2=(300, 100), r1=50, r2=70)
        assert scene.signed_gap == pytest.approx(80.0)

    @pytest.mark.parametrize(
        ("m", "label", "overlap_or_touch"),
        [
            (25.0, "separate", False),
            (0.0, "touching", True),
            (-25.0, "overlapping", True),
        ],
    )
    def test_label_and_binary(self, m, label, overlap_or_touch):
        scene = _scene(m)
        assert scene.label() == label
        assert scene.overlap_or_touch is overlap_or_touch

    def test_tangent_eps_widens_touching_band(self):
        assert _scene(0.5).label() == "separate"
        assert _scene(0.5).label(tangent_eps=1.0) == "touching"
        assert _scene(-0.5).label(tangent_eps=1.0) == "touching"

    def test_translate_is_pure_counterfactual(self):
        scene = _scene(10.0)
        moved = translate_circle_1(scene, dx=-15.0, dy=0.0)
        # z' = T(z, a): new scene reflects the action, original is untouched.
        assert moved.signed_gap == pytest.approx(counterfactual_margin(scene, -15.0, 0.0))
        assert scene.signed_gap == pytest.approx(10.0)
        assert moved.c1 == (scene.c1[0] - 15.0, scene.c1[1])

    def test_counterfactual_margin_matches_math(self):
        scene = _scene(10.0)
        # Move circle 1 straight toward circle 2 by 25px: m drops by exactly 25.
        assert counterfactual_margin(scene, 25.0, 0.0) == pytest.approx(-15.0)

    def test_scene_targets_consistent(self):
        scene = _scene(-5.0)
        t = scene_targets(scene, "near_overlap")
        assert t["m"] == pytest.approx(scene.signed_gap)
        assert t["margin_bin"] == "near_overlap"
        assert t["label"] == "overlapping"
        assert t["overlap_or_touch"] is True


class TestSampler:
    @pytest.mark.parametrize("bin_name", sorted(MARGIN_BINS))
    def test_sampled_m_within_bin(self, bin_name):
        rng = random.Random(42)
        lo, hi = MARGIN_BINS[bin_name]
        for _ in range(5):
            scene = sample_scene(rng, margin_bin=bin_name)
            assert lo - 1e-6 <= scene.signed_gap <= hi + 1e-6

    def test_explicit_m_is_honored(self):
        rng = random.Random(7)
        scene = sample_scene(rng, m=-12.5)
        assert scene.signed_gap == pytest.approx(-12.5, abs=1e-6)

    def test_circles_fit_canvas(self):
        rng = random.Random(3)
        pad = 6.0
        for _ in range(20):
            s = sample_scene(rng, pad=pad)
            for (x, y), r in [(s.c1, s.r1), (s.c2, s.r2)]:
                assert r + pad <= x <= s.width - r - pad
                assert r + pad <= y <= s.height - r - pad

    def test_no_near_containment(self):
        # Sub-MIN_PROTRUSION slivers are ambiguous stimuli and unverifiable by
        # the pixel oracle (SSOT §7.6), so the sampler must never emit them.
        rng = random.Random(11)
        for _ in range(20):
            s = sample_scene(rng, margin_bin="deep_overlap")
            assert math.dist(s.c1, s.c2) >= abs(s.r1 - s.r2) + MIN_PROTRUSION_PX

    def test_seed_determinism(self):
        a = sample_scene(random.Random(123), margin_bin="near_separate")
        b = sample_scene(random.Random(123), margin_bin="near_separate")
        assert a == b

    @pytest.mark.parametrize(
        ("m", "expected"),
        [
            (80.0, "far_separate"),
            (20.0, "far_separate"),  # shared endpoint -> earlier bin
            (10.0, "near_separate"),
            (1.0, "near_separate"),
            (0.0, "tangent"),
            (-1.0, "tangent"),
            (-10.0, "near_overlap"),
            (-20.0, "near_overlap"),
            (-80.0, "deep_overlap"),
        ],
    )
    def test_bin_for_m(self, m, expected):
        assert bin_for_m(m) == expected

    def test_bin_for_m_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bin_for_m(81.0)


class TestRender:
    def test_render_deterministic_bytes(self):
        scene = _scene(-10.0)
        im1, im2 = render(scene), render(scene)
        assert im1.size == (512, 512)
        assert im1.tobytes() == im2.tobytes()

    def test_overlay_text_changes_pixels(self):
        import dataclasses

        scene = _scene(-10.0)
        with_text = dataclasses.replace(scene, overlay_text="not touching")
        assert render(scene).tobytes() != render(with_text).tobytes()

    def test_noise_is_deterministic_per_scene(self):
        import dataclasses

        noisy = dataclasses.replace(_scene(-10.0), noise="low")
        assert render(noisy).tobytes() == render(noisy).tobytes()
        assert render(noisy).tobytes() != render(_scene(-10.0)).tobytes()

    def test_renderer_version_pinned(self):
        # Manifests join on this; bump it deliberately, not accidentally.
        assert RENDERER_VERSION == "two_circles-0.2.0"
