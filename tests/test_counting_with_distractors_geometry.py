"""GPU-free tests for the counting-with-distractors world model
(design docs/designs/counting_with_distractors.md).

Everything here runs from geometry + PIL only; no torch, no network.
"""

import random

import numpy as np
import pytest

from question_foundry.worlds import get_world
from worlds.counting_with_distractors.renderer import (
    BACKGROUND_SEP_PX,
    DISTRACTOR_PALETTE,
    FLOOR_PX,
    MARGIN_BINS,
    N_DISTRACTORS_RANGE,
    N_TARGETS_RANGE,
    RADIUS_RANGE,
    RENDERER_VERSION,
    SAMPLED_BINS,
    CountingScene,
    bin_for_m,
    gap,
    min_pairwise_gap,
    render,
    sample_scene,
    scene_targets,
)


def _scene(disks_targets, disks_distractors, colors) -> CountingScene:
    return CountingScene(
        target_disks=list(disks_targets),
        distractor_disks=list(disks_distractors),
        distractor_colors=list(colors),
    )


class TestGeometry:
    def test_gap_is_center_distance_minus_radii(self):
        a = (100.0, 100.0, 20.0)
        b = (100.0, 160.0, 15.0)  # centres 60 apart
        assert gap(a, b) == pytest.approx(60.0 - 20.0 - 15.0)

    def test_min_pairwise_gap_picks_the_closest(self):
        disks = [(0.0, 0.0, 10.0), (100.0, 0.0, 10.0), (0.0, 40.0, 10.0)]
        # gaps: (0,1)=80, (0,2)=20, (1,2)=hypot(100,40)-20~87.7 -> min 20
        assert min_pairwise_gap(disks) == pytest.approx(20.0)

    def test_min_pairwise_gap_needs_two(self):
        with pytest.raises(ValueError):
            min_pairwise_gap([(0.0, 0.0, 10.0)])

    def test_scene_count_and_min_gap(self):
        s = _scene(
            [(100.0, 100.0, 20.0), (200.0, 100.0, 20.0)],
            [(100.0, 250.0, 15.0)],
            ["blue"],
        )
        assert s.count == 2
        assert s.n_targets == 2
        assert s.n_distractors == 1
        # closest pair is the two targets: gap = 100 - 40 = 60
        assert s.min_gap == pytest.approx(60.0)

    def test_scene_targets_payload(self):
        s = _scene(
            [(100.0, 100.0, 20.0), (200.0, 100.0, 20.0), (100.0, 300.0, 18.0)],
            [(400.0, 400.0, 16.0)],
            ["green"],
        )
        t = scene_targets(s, "spaced")
        assert t == {
            "count": 3,
            "n_targets": 3,
            "n_distractors": 1,
            "min_gap": s.min_gap,
            "margin_bin": "spaced",
        }


class TestSampler:
    @pytest.mark.parametrize("bin_name", sorted(SAMPLED_BINS))
    def test_sampled_m_within_bin(self, bin_name):
        rng = random.Random(42)
        lo, hi = MARGIN_BINS[bin_name]
        lo = max(lo, FLOOR_PX)
        for _ in range(12):
            s = sample_scene(rng, margin_bin=bin_name)
            assert lo - 1e-6 <= s.min_gap <= hi + 1e-6

    @pytest.mark.parametrize("m", [1.0, 2.0, 5.0, 12.0, 40.0, 80.0])
    def test_explicit_m_is_honored(self, m):
        s = sample_scene(random.Random(7), m=m)
        assert s.min_gap == pytest.approx(m, abs=1e-6)

    def test_min_gap_is_a_single_distinct_close_pair(self):
        # Every non-closest pair is >= m + BACKGROUND_SEP_PX (design §2).
        rng = random.Random(11)
        for _ in range(20):
            s = sample_scene(rng, margin_bin="tight")
            disks = s.all_disks()
            gaps = sorted(
                gap(disks[i], disks[j]) for i in range(len(disks)) for j in range(i + 1, len(disks))
            )
            assert gaps[0] == pytest.approx(s.min_gap)
            if len(gaps) > 1:
                assert gaps[1] >= s.min_gap + BACKGROUND_SEP_PX - 1e-6

    def test_counts_and_radii_in_range(self):
        rng = random.Random(3)
        for _ in range(30):
            s = sample_scene(rng)
            assert N_TARGETS_RANGE[0] <= s.n_targets <= N_TARGETS_RANGE[1]
            assert N_DISTRACTORS_RANGE[0] <= s.n_distractors <= N_DISTRACTORS_RANGE[1]
            for _, _, r in s.all_disks():
                assert RADIUS_RANGE[0] - 1e-6 <= r <= RADIUS_RANGE[1] + 1e-6

    def test_distractor_colors_distinct_and_from_palette(self):
        rng = random.Random(4)
        for _ in range(30):
            s = sample_scene(rng)
            assert len(s.distractor_colors) == s.n_distractors
            assert len(set(s.distractor_colors)) == s.n_distractors  # distinct
            assert all(c in DISTRACTOR_PALETTE for c in s.distractor_colors)

    def test_all_disks_on_canvas_and_disjoint(self):
        rng = random.Random(5)
        pad = 6.0
        for _ in range(30):
            s = sample_scene(rng, pad=pad)
            for cx, cy, r in s.all_disks():
                assert pad <= cx - r and cx + r <= s.width - pad
                assert pad <= cy - r and cy + r <= s.height - pad
            assert s.min_gap > 0  # no overlap / touch

    def test_explicit_m_close_pair_is_cross_colour(self):
        # In sweep mode (explicit m) the min-gap pair is target-distractor, so
        # target disks are never that close (design §2).
        rng = random.Random(9)
        for _ in range(20):
            s = sample_scene(rng, m=1.5)
            if s.n_targets >= 2:
                tgaps = min(
                    gap(s.target_disks[i], s.target_disks[j])
                    for i in range(s.n_targets)
                    for j in range(i + 1, s.n_targets)
                )
                assert tgaps > s.min_gap + 1e-6

    def test_seed_determinism(self):
        a = sample_scene(random.Random(123), margin_bin="crowded")
        b = sample_scene(random.Random(123), margin_bin="crowded")
        assert a == b

    def test_counts_span_full_range_in_packable_bins(self):
        # N and D independent of m: the tight bin packs the whole range.
        rng = random.Random(2)
        ns, ds = set(), set()
        for _ in range(200):
            s = sample_scene(rng, margin_bin="tight")
            ns.add(s.n_targets)
            ds.add(s.n_distractors)
        assert ns == set(range(N_TARGETS_RANGE[0], N_TARGETS_RANGE[1] + 1))
        assert ds == set(range(N_DISTRACTORS_RANGE[0], N_DISTRACTORS_RANGE[1] + 1))

    @pytest.mark.parametrize(
        ("m", "expected"),
        [
            (120.0, "far_isolated"),
            (40.0, "far_isolated"),  # shared endpoint -> earlier bin
            (30.0, "spaced"),
            (20.0, "spaced"),
            (12.0, "crowded"),
            (8.0, "crowded"),
            (5.0, "tight"),
            (3.0, "tight"),
            (2.0, "boundary"),
            (0.0, "boundary"),
        ],
    )
    def test_bin_for_m(self, m, expected):
        assert bin_for_m(m) == expected

    def test_bin_for_m_out_of_range_raises(self):
        with pytest.raises(ValueError):
            bin_for_m(121.0)


class TestRender:
    def test_render_deterministic_bytes(self):
        s = _scene([(150.0, 150.0, 20.0), (300.0, 150.0, 22.0)], [(200.0, 350.0, 18.0)], ["blue"])
        im1, im2 = render(s), render(s)
        assert im1.size == (512, 512)
        assert im1.tobytes() == im2.tobytes()

    def test_target_and_distractor_colors_present(self):
        s = _scene([(150.0, 150.0, 24.0)], [(350.0, 350.0, 20.0)], ["blue"])
        arr = np.asarray(render(s))
        assert (arr == (255, 0, 0)).all(axis=-1).any()  # red target
        assert (arr == (0, 0, 255)).all(axis=-1).any()  # blue distractor

    def test_renderer_version_pinned(self):
        assert RENDERER_VERSION == "counting_with_distractors-0.1.0"


class TestDiscovery:
    def test_meta_layer_discovers_instance(self):
        info = get_world("counting_with_distractors")
        assert info.description
        assert (info.path / "world.toml").is_file()
        assert (info.path / "generate.py").is_file()
        assert (info.path / "renderer.py").is_file()
        assert info.meta["world"]["task"] == "counting"
        assert info.meta["axes"]["decision_variable"].startswith("m ")
        assert info.meta["axes"]["structural"].startswith("N objects")
