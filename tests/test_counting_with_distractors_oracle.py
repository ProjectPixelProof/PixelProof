"""Counting-with-distractors pixel oracle (SSOT §7.6): the render-vs-ground-truth
trust anchor.

The agents driving this harness cannot look at images, so these tests are the only
guarantee that verify_example() (a) accepts correct renders across the whole
sampled regime and (b) rejects images that do not match their scene — including,
specifically, a tamper that changes the COUNT (the label the oracle re-measures
directly from pixels). Tolerances are empirically pinned (oracle.py CALIBRATION_NOTE:
2400 bin-sampled + 1320 sweep scenes, count exact in every one).
"""

import dataclasses
import math
import random

import pytest
from PIL import Image

from worlds.counting_with_distractors.oracle import (
    CENTER_TOL_PX,
    M_TOL_PX,
    MIN_DISK_RADIUS_PX,
    RADIUS_TOL_PX,
    RESIDUAL_TOL_PX,
    SPECKLE_TOL_PX,
    OracleFitError,
    UnsupportedSceneError,
    measure,
    verify_example,
)
from worlds.counting_with_distractors.renderer import (
    MARGIN_SWEEP,
    SAMPLED_BINS,
    CountingScene,
    render,
    sample_scene,
)


def _base() -> CountingScene:
    """A well-separated 3-target / 2-distractor scene (all gaps large)."""
    return CountingScene(
        target_disks=[(120.0, 120.0, 20.0), (250.0, 120.0, 20.0), (120.0, 300.0, 20.0)],
        distractor_disks=[(400.0, 400.0, 18.0), (400.0, 150.0, 18.0)],
        distractor_colors=["blue", "green"],
    )


class TestMeasurement:
    def test_recovers_count_disks_and_margin(self):
        scene = _base()
        meas = measure(render(scene), scene.color_target, scene.distractor_colors)
        assert len(meas.target_disks) == 3
        assert meas.distractor_counts == {"blue": 1, "green": 1}
        assert meas.m_hat == pytest.approx(scene.min_gap, abs=M_TOL_PX)
        # every recovered target centre matches an analytic one
        for tx, ty, tr in scene.target_disks:
            best = min(meas.target_disks, key=lambda f: math.hypot(f.cx - tx, f.cy - ty))
            assert math.hypot(best.cx - tx, best.cy - ty) <= CENTER_TOL_PX
            assert abs(best.r - tr) <= RADIUS_TOL_PX

    def test_base_scene_certifies(self):
        report = verify_example(render(_base()), _base())
        assert report.ok, report.failures
        assert report.confidence == "high"
        assert report.count == 3 and report.count_hat == 3

    @pytest.mark.parametrize("bin_name", sorted(SAMPLED_BINS))
    def test_certifies_sampled_scenes(self, bin_name):
        rng = random.Random(99)
        for _ in range(8):
            scene = sample_scene(rng, margin_bin=bin_name)
            report = verify_example(render(scene), scene)
            assert report.ok, (bin_name, scene.min_gap, report.failures)
            assert report.confidence == "high"
            assert report.count_hat == scene.count
            assert report.abs_err <= M_TOL_PX

    @pytest.mark.parametrize("m", [float(x) for x in MARGIN_SWEEP])
    def test_certifies_sweep_including_boundary_band(self, m):
        rng = random.Random(int(m) + 1)
        for _ in range(6):
            scene = sample_scene(rng, m=m)
            report = verify_example(render(scene), scene)
            assert report.ok, (m, scene.min_gap, report.failures)
            assert report.count_hat == scene.count


class TestCorruptionCaught:
    """Render from a tampered scene, verify against the original ground truth."""

    def test_miscount_extra_target_caught_specifically(self):
        base = _base()
        wrong = dataclasses.replace(
            base, target_disks=[*base.target_disks, (300.0, 350.0, 20.0)]
        )  # 4 red disks rendered, but ground truth says N=3
        report = verify_example(render(wrong), base)
        assert not report.ok
        assert report.count_hat == 4
        assert any("target count" in f for f in report.failures), report.failures

    def test_miscount_dropped_target_caught_specifically(self):
        base = _base()
        wrong = dataclasses.replace(base, target_disks=base.target_disks[:2])  # only 2 drawn
        report = verify_example(render(wrong), base)
        assert not report.ok
        assert report.count_hat == 2
        assert any("target count" in f for f in report.failures), report.failures

    def test_radius_tamper_caught(self):
        base = _base()
        big = [(base.target_disks[0][0], base.target_disks[0][1], 34.0), *base.target_disks[1:]]
        wrong = dataclasses.replace(base, target_disks=big)
        report = verify_example(render(wrong), base)
        assert not report.ok

    def test_color_swap_caught(self):
        base = _base()
        wrong = dataclasses.replace(
            base,
            color_target=base.distractor_colors[0],
            distractor_colors=[base.color_target, *base.distractor_colors[1:]],
        )
        report = verify_example(render(wrong), base)
        assert not report.ok

    def test_blank_image(self):
        base = _base()
        blank = Image.new("RGB", (base.width, base.height), "white")
        report = verify_example(blank, base)
        assert not report.ok
        assert "nothing to count" in report.failures[0]

    def test_wrong_canvas_size(self):
        base = _base()
        shrunk = render(base).resize((256, 256))
        report = verify_example(shrunk, base)
        assert not report.ok


class TestUnsupportedStyles:
    def test_shared_color_raises(self):
        scene = CountingScene(
            target_disks=[(120.0, 120.0, 20.0), (250.0, 120.0, 20.0)],
            distractor_disks=[(400.0, 400.0, 18.0)],
            distractor_colors=["red"],  # same as target colour
        )
        with pytest.raises(UnsupportedSceneError):
            verify_example(render(scene), scene)

    def test_hairline_radius_raises(self):
        scene = CountingScene(
            target_disks=[(120.0, 120.0, 20.0), (250.0, 120.0, 3.0)],  # 3px hairline
            distractor_disks=[(400.0, 400.0, 18.0)],
            distractor_colors=["blue"],
        )
        with pytest.raises(UnsupportedSceneError):
            verify_example(render(scene), scene)

    def test_missing_ink_is_fit_error_not_unsupported(self):
        # A scene the oracle *can* measure but whose red ink is absent -> OracleFitError
        # inside measure(), surfaced by verify_example as a failed report (not a raise).
        base = _base()
        blank = Image.new("RGB", (base.width, base.height), "white")
        # measure() itself raises OracleFitError; verify_example converts to a report.
        with pytest.raises(OracleFitError):
            measure(blank, base.color_target, base.distractor_colors)


def test_tolerances_are_the_calibrated_values():
    # Protected invariant (SSOT §7.6 rule 2): set empirically with >=2x headroom over
    # observed maxima (oracle.py CALIBRATION_NOTE). A silent relaxation to make a
    # failing dataset pass must show up as a change to this test.
    assert CENTER_TOL_PX == 1.5
    assert RADIUS_TOL_PX == 1.5
    assert M_TOL_PX == 3.0
    assert RESIDUAL_TOL_PX == 1.2
    assert SPECKLE_TOL_PX == 30
    assert MIN_DISK_RADIUS_PX == 8.0
