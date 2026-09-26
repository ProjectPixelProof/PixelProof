"""GPU-free tests for the angle-acuteness world model (design docs/designs/angle_acuteness.md).

Everything here runs from geometry + PIL only; no torch, no network.
"""

import math
import random

import pytest

from question_foundry.worlds import get_world
from worlds.angle_acuteness.renderer import (
    LENGTH_RANGE,
    MARGIN_BINS,
    RENDERER_VERSION,
    STROKE_RANGE,
    THETA_RANGE,
    AngleScene,
    bin_for_m,
    counterfactual_margin,
    ray_rect,
    render,
    rotate_ray2,
    sample_scene,
    scene_targets,
)


def _interior_angle(scene: AngleScene) -> float:
    v = (scene.vx, scene.vy)
    a = (scene.ray1_end[0] - v[0], scene.ray1_end[1] - v[1])
    b = (scene.ray2_end[0] - v[0], scene.ray2_end[1] - v[1])
    cos = (a[0] * b[0] + a[1] * b[1]) / (math.hypot(*a) * math.hypot(*b))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


class TestGeometry:
    @pytest.mark.parametrize("theta", [30.0, 60.0, 90.0, 120.0, 150.0])
    def test_signed_margin_and_interior_angle(self, theta):
        s = AngleScene(vx=256, vy=256, heading=33.0, theta=theta, len1=160, len2=150, stroke=7.0)
        assert s.signed_margin == pytest.approx(90.0 - theta)
        # The drawn rays really subtend theta degrees at the vertex.
        assert _interior_angle(s) == pytest.approx(theta, abs=1e-6)

    @pytest.mark.parametrize(
        ("theta", "label", "acute"),
        [(60.0, "acute", True), (90.0, "right", False), (120.0, "obtuse", False)],
    )
    def test_label_and_binary(self, theta, label, acute):
        s = AngleScene(vx=256, vy=256, heading=0.0, theta=theta, len1=150, len2=150, stroke=7.0)
        assert s.label() == label
        assert s.acute is acute

    def test_tangent_eps_widens_right_band(self):
        near = AngleScene(vx=256, vy=256, heading=0.0, theta=89.0, len1=150, len2=150, stroke=7.0)
        assert near.label() == "acute"
        assert near.label(tangent_eps=2.0) == "right"

    def test_counterfactual_margin_can_cross_boundary(self):
        s = AngleScene(vx=256, vy=256, heading=0.0, theta=80.0, len1=150, len2=150, stroke=7.0)
        assert s.acute
        assert counterfactual_margin(s, 20.0) == pytest.approx(-10.0)  # 80 -> 100 obtuse
        assert rotate_ray2(s, 20.0).signed_margin < 0

    def test_scene_targets_consistent(self):
        s = AngleScene(vx=256, vy=256, heading=0.0, theta=70.0, len1=150, len2=150, stroke=7.0)
        t = scene_targets(s, "far_acute")
        assert t["m"] == pytest.approx(20.0)
        assert t["theta"] == pytest.approx(70.0)
        assert t["margin_bin"] == "far_acute"
        assert t["label"] == "acute"
        assert t["acute"] is True


class TestSampler:
    @pytest.mark.parametrize("bin_name", sorted(MARGIN_BINS))
    def test_sampled_m_within_bin(self, bin_name):
        rng = random.Random(42)
        lo, hi = MARGIN_BINS[bin_name]
        for _ in range(10):
            s = sample_scene(rng, margin_bin=bin_name)
            assert lo - 1e-9 <= s.signed_margin <= hi + 1e-9
            assert THETA_RANGE[0] - 1e-9 <= s.theta <= THETA_RANGE[1] + 1e-9

    @pytest.mark.parametrize("m", [-45.0, -1.0, 0.0, 1.0, 45.0])
    def test_explicit_m_is_honored_exactly(self, m):
        s = sample_scene(random.Random(7), m=m)
        assert s.signed_margin == pytest.approx(m, abs=1e-9)
        assert s.theta == pytest.approx(90.0 - m, abs=1e-9)

    def test_lengths_and_stroke_in_range(self):
        rng = random.Random(3)
        for _ in range(30):
            s = sample_scene(rng)
            assert LENGTH_RANGE[0] - 1e-9 <= s.len1 <= LENGTH_RANGE[1] + 1e-9
            assert LENGTH_RANGE[0] - 1e-9 <= s.len2 <= LENGTH_RANGE[1] + 1e-9
            assert STROKE_RANGE[0] - 1e-9 <= s.stroke <= STROKE_RANGE[1] + 1e-9

    def test_strokes_fit_canvas(self):
        rng = random.Random(5)
        pad = 6.0
        for _ in range(30):
            s = sample_scene(rng, pad=pad)
            v = (s.vx, s.vy)
            for end in (s.ray1_end, s.ray2_end):
                for x, y in ray_rect(v, end, s.stroke):
                    assert pad <= x <= s.width - pad
                    assert pad <= y <= s.height - pad

    def test_seed_determinism(self):
        a = sample_scene(random.Random(123), margin_bin="near_acute")
        b = sample_scene(random.Random(123), margin_bin="near_acute")
        assert a == b

    @pytest.mark.parametrize(
        ("m", "expected"),
        [
            (60.0, "far_acute"),
            (10.0, "far_acute"),  # shared endpoint -> earlier bin
            (5.0, "near_acute"),
            (2.0, "near_acute"),
            (0.0, "right"),
            (-2.0, "right"),
            (-5.0, "near_obtuse"),
            (-10.0, "near_obtuse"),
            (-60.0, "far_obtuse"),
        ],
    )
    def test_bin_for_m(self, m, expected):
        assert bin_for_m(m) == expected

    def test_bin_for_m_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bin_for_m(61.0)


class TestRender:
    def test_render_deterministic_bytes(self):
        s = AngleScene(vx=256, vy=256, heading=17.0, theta=70.0, len1=160, len2=150, stroke=7.0)
        im1, im2 = render(s), render(s)
        assert im1.size == (512, 512)
        assert im1.tobytes() == im2.tobytes()

    def test_ink_color_present(self):
        import numpy as np

        arr = np.asarray(render(AngleScene(256, 256, 0.0, 70.0, 160, 150, 7.0)))
        assert (arr == (0, 0, 255)).all(axis=-1).any()  # blue ink

    def test_renderer_version_pinned(self):
        assert RENDERER_VERSION == "angle_acuteness-0.1.0"


class TestDiscovery:
    def test_meta_layer_discovers_instance(self):
        info = get_world("angle_acuteness")
        assert info.description
        assert (info.path / "world.toml").is_file()
        assert (info.path / "generate.py").is_file()
        assert (info.path / "renderer.py").is_file()
        assert info.meta["axes"]["decision_variable"].startswith("m ")
