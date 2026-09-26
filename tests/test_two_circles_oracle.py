"""Pixel oracle (SSOT §7.6): the render-vs-ground-truth trust anchor.

The agents driving this harness cannot look at images, so these tests are the
only guarantee that verify_example() (a) accepts correct renders across the
whole sampled regime and (b) rejects images that do not match their scene.
Tolerances are empirically pinned — v0.2.0 (nearest-color + exposed-rim): max
observed |m̂ - m| over 500 bin-sampled scenes is ~1.2px, including worst-case
sliver geometries at the sampler floor.
"""

import dataclasses
import math
import random

import pytest

from worlds.two_circles.oracle import (
    M_TOL_PX,
    MIN_ARC_DEG,
    UnsupportedSceneError,
    _lens_area,
    measure,
    verify_example,
)
from worlds.two_circles.renderer import (
    MARGIN_BINS,
    MIN_PROTRUSION_PX,
    TwoCircleScene,
    render,
    sample_scene,
)


def _scene(m: float, r1: float = 50.0, r2: float = 70.0, **style) -> TwoCircleScene:
    d = r1 + r2 + m
    return TwoCircleScene(c1=(150.0, 256.0), c2=(150.0 + d, 256.0), r1=r1, r2=r2, **style)


class TestMeasurement:
    def test_recovers_full_circles(self):
        scene = _scene(20.0)
        meas = measure(render(scene), scene.color1, scene.color2)
        assert meas.circle1.cx == pytest.approx(scene.c1[0], abs=1.0)
        assert meas.circle1.cy == pytest.approx(scene.c1[1], abs=1.0)
        assert meas.circle1.r == pytest.approx(scene.r1, abs=1.0)
        assert meas.circle2.r == pytest.approx(scene.r2, abs=1.0)
        assert meas.circle1.arc_deg == pytest.approx(360.0, abs=10.0)
        assert meas.m_hat == pytest.approx(20.0, abs=M_TOL_PX)

    @pytest.mark.parametrize("bin_name", sorted(MARGIN_BINS))
    def test_certifies_sampled_scenes(self, bin_name):
        rng = random.Random(99)
        for _ in range(5):
            scene = sample_scene(rng, margin_bin=bin_name)
            report = verify_example(render(scene), scene)
            assert report.ok, (bin_name, scene.signed_gap, report.failures)
            assert report.confidence == "high"
            assert report.abs_err <= M_TOL_PX

    def test_certifies_noise_and_overlay(self):
        for style in ({"noise": "moderate"}, {"overlay_text": "not touching"}):
            scene = sample_scene(random.Random(7), margin_bin="near_overlap", **style)
            report = verify_example(render(scene), scene)
            assert report.ok, (style, report.failures)

    def test_thin_arc_is_low_confidence_not_rejected(self):
        # sample_scene never emits sub-MIN_ARC_DEG slivers (at 512px the
        # measurability floor and the fit-conditioning floor nearly coincide),
        # but hand-built counterfactual scenes can. The circle-1 fit is then
        # ill-conditioned; the oracle must downgrade to sign/area checks
        # rather than hard-fail a correct render.
        scene = TwoCircleScene(
            c1=(400.0, 512.0), c2=(485.0, 512.0), r1=200.0, r2=280.0, width=1024, height=1024
        )
        image = render(scene)
        meas = measure(image, scene.color1, scene.color2)
        assert meas.circle1.arc_deg < MIN_ARC_DEG
        report = verify_example(image, scene)
        assert report.confidence == "low"
        assert report.ok, report.failures


class TestCorruptionCaught:
    """Render from a tampered scene, verify against the original ground truth."""

    @pytest.fixture()
    def base(self) -> TwoCircleScene:
        return sample_scene(random.Random(5), margin_bin="near_separate")

    @pytest.mark.parametrize(
        "tamper",
        [
            {"r1": lambda s: s.r1 + 4},
            {"r1": lambda s: s.r1 / 2},
            {"c1": lambda s: (s.c1[0] + 4, s.c1[1])},
            {"color1": lambda s: s.color2, "color2": lambda s: s.color1},
        ],
        ids=["radius+4px", "radius-halved", "center+4px", "colors-swapped"],
    )
    def test_scene_tampering(self, base, tamper):
        wrong = dataclasses.replace(base, **{k: f(base) for k, f in tamper.items()})
        report = verify_example(render(wrong), base)
        assert not report.ok

    def test_blank_image(self, base):
        from PIL import Image

        blank = Image.new("RGB", (base.width, base.height), "white")
        report = verify_example(blank, base)
        assert not report.ok
        assert "boundary pixels" in report.failures[0]

    def test_wrong_canvas_size(self, base):
        shrunk = render(base).resize((256, 256))
        report = verify_example(shrunk, base)
        assert not report.ok


class TestUnsupportedStyles:
    def test_outline_fill_raises(self):
        scene = _scene(10.0, fill="outline")
        with pytest.raises(UnsupportedSceneError):
            verify_example(render(scene), scene)

    def test_shared_color_raises(self):
        scene = _scene(10.0, color1="red", color2="red")
        with pytest.raises(UnsupportedSceneError):
            verify_example(render(scene), scene)


class TestLensArea:
    def test_disjoint_and_contained(self):
        assert _lens_area(30.0, 40.0, 80.0) == 0.0
        assert _lens_area(30.0, 40.0, 5.0) == pytest.approx(math.pi * 30.0**2)

    def test_unit_circles_at_unit_distance(self):
        # Classic closed form: 2 * (pi/3 - sqrt(3)/4).
        assert _lens_area(1.0, 1.0, 1.0) == pytest.approx(2 * (math.pi / 3 - math.sqrt(3) / 4))

    def test_matches_rendered_overlap_pixels(self):
        # End-to-end: analytic lens area vs actual hidden (occluded) blue ink.
        scene = _scene(-30.0)
        meas = measure(render(scene), scene.color1, scene.color2)
        hidden = math.pi * scene.r1**2 - meas.area1_px
        d = math.dist(scene.c1, scene.c2)
        assert hidden == pytest.approx(_lens_area(scene.r1, scene.r2, d), rel=0.05)


def test_sampler_floor_matches_oracle_capability():
    # MIN_PROTRUSION_PX is the contract between sampler and oracle: at the floor
    # the oracle must still certify at high confidence (empirically validated
    # for the radius range the sampler uses).
    for r1, r2 in [(30.0, 90.0), (35.0, 85.0), (90.0, 30.0)]:
        d = abs(r1 - r2) + MIN_PROTRUSION_PX
        scene = TwoCircleScene(c1=(200.0, 256.0), c2=(200.0 + d, 256.0), r1=r1, r2=r2)
        report = verify_example(render(scene), scene)
        assert report.ok and report.confidence == "high", (r1, r2, report)
