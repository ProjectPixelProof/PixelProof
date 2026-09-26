"""Angle-acuteness pixel oracle (SSOT §7.6): the render-vs-ground-truth trust anchor.

The agents driving this harness cannot look at images, so these tests are the only
guarantee that verify_example() (a) accepts correct renders across the whole
sampled regime and (b) rejects images that do not match their scene — including,
specifically, a tamper that flips the acute/obtuse label. Tolerances are empirically
pinned (see worlds/angle_acuteness/oracle.py docstring: max |θ̂ - θ| ~0.40deg
over 1500 bin-sampled + 400 worst-case scenes). The oracle splits the single
same-colour "V" mask into two rays by a two-line fit; it is NOT weakened to two
colours (design §4).
"""

import dataclasses
import math
import random

import pytest

from worlds.angle_acuteness.oracle import (
    ANGLE_TOL_DEG,
    ENDPOINT_TOL_PX,
    RIGHT_BAND_DEG,
    VERTEX_TOL_PX,
    UnsupportedSceneError,
    measure,
    verify_example,
)
from worlds.angle_acuteness.renderer import (
    MARGIN_BINS,
    AngleScene,
    render,
    sample_scene,
)


def _scene(theta: float, heading: float = 20.0) -> AngleScene:
    return AngleScene(
        vx=256.0, vy=256.0, heading=heading, theta=theta, len1=170.0, len2=160.0, stroke=7.0
    )


class TestMeasurement:
    @pytest.mark.parametrize("theta", [40.0, 78.0, 90.0, 102.0, 140.0])
    def test_recovers_angle_and_vertex(self, theta):
        scene = _scene(theta)
        meas = measure(
            render(scene), scene.color_ink, background="white", stroke_width=scene.stroke
        )
        assert meas.theta_hat == pytest.approx(theta, abs=ANGLE_TOL_DEG)
        assert math.hypot(meas.vertex[0] - scene.vx, meas.vertex[1] - scene.vy) <= VERTEX_TOL_PX

    @pytest.mark.parametrize("bin_name", sorted(MARGIN_BINS))
    def test_certifies_sampled_scenes(self, bin_name):
        rng = random.Random(99)
        for _ in range(8):
            scene = sample_scene(rng, margin_bin=bin_name)
            report = verify_example(render(scene), scene)
            assert report.ok, (bin_name, scene.signed_margin, report.failures)
            assert report.confidence == "high"

    def test_certifies_worst_case_short_thin(self):
        # The calibrated worst case: both rays at the 120px length floor, 5px stroke,
        # near the |m| boundary, many headings. Must still certify.
        rng = random.Random(3)
        for _ in range(20):
            m = rng.choice([-2.0, -1.0, 1.0, 2.0])
            scene = AngleScene(
                vx=256.0, vy=256.0, heading=rng.uniform(0, 360), theta=90.0 - m,
                len1=120.0, len2=120.0, stroke=5.0,
            )  # fmt: skip
            report = verify_example(render(scene), scene)
            assert report.ok, (m, report.failures)


class TestCorruptionCaught:
    """Render from a tampered scene, verify against the original ground truth."""

    @pytest.fixture()
    def base(self) -> AngleScene:
        # m = +12 (theta 78) is clearly off the quarantined right band (|m| <= 2), so
        # an acute->obtuse tamper must trip the label check.
        return _scene(78.0)

    @pytest.mark.parametrize(
        "tamper",
        [
            {"theta": lambda s: 110.0},  # acute -> obtuse
            {"stroke": lambda s: s.stroke * 1.8},
            {"len1": lambda s: s.len1 * 0.6},
        ],
        ids=["angle-to-obtuse", "stroke-thickened", "ray1-shortened"],
    )
    def test_scene_tampering(self, base, tamper):
        wrong = dataclasses.replace(base, **{k: f(base) for k, f in tamper.items()})
        report = verify_example(render(wrong), base)
        assert not report.ok

    def test_label_flip_caught_specifically(self, base):
        # Open the angle from acute (78) to obtuse (110): the acute/obtuse label
        # inverts. The oracle must raise a label-flip failure specifically.
        assert base.signed_margin > RIGHT_BAND_DEG  # precondition: off the boundary
        flipped = dataclasses.replace(base, theta=110.0)
        report = verify_example(render(flipped), base)
        assert not report.ok
        assert any("label flip" in f for f in report.failures), report.failures

    def test_blank_image(self, base):
        from PIL import Image

        blank = Image.new("RGB", (base.width, base.height), "white")
        report = verify_example(blank, base)
        assert not report.ok
        assert "pixels" in report.failures[0]

    def test_wrong_canvas_size(self, base):
        shrunk = render(base).resize((256, 256))
        report = verify_example(shrunk, base)
        assert not report.ok


class TestUnsupportedStyles:
    def test_shared_color_raises(self):
        scene = dataclasses.replace(_scene(70.0), color_ink="white")
        with pytest.raises(UnsupportedSceneError):
            verify_example(render(scene), scene)

    def test_hairline_stroke_raises(self):
        scene = dataclasses.replace(_scene(70.0), stroke=2.0)
        with pytest.raises(UnsupportedSceneError):
            verify_example(render(scene), scene)


def test_tolerances_are_the_calibrated_values():
    # Protected invariant (SSOT §7.6 rule 2): these were set empirically with >=2x
    # headroom over observed maxima (oracle.py docstring). A silent relaxation to
    # make a failing dataset pass must show up as a change to this test.
    assert ANGLE_TOL_DEG == 1.5
    assert VERTEX_TOL_PX == 2.0
    assert ENDPOINT_TOL_PX == 2.5
    assert RIGHT_BAND_DEG == 2.0
