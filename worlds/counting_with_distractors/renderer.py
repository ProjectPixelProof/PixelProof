"""Counting-with-distractors world renderer (SSOT §7.2 "Counting with
distractors", §7.7; design docs/designs/counting_with_distractors.md).

Latent state z = {target disks, distractor disks, canvas}. ``N`` target-coloured
filled disks (N in [2, 7]) and ``D`` distractor disks (D in [1, 6]) of distinct
palette colours are drawn on a white canvas; no two disks overlap or touch. The
image is I = R(z) and the closed-form gold is simply

    count = N                       (the number of target-coloured disks)

a scene *parameter*, not a pixel measurement. The crowding margin is

    m = min over ALL pairs of disks (targets AND distractors) of
        ( dist(c_i, c_j) - r_i - r_j )          (canvas px, always > 0)

with large m => every disk isolated (easy to count) and small m => two disks
nearly touch. m is the SSOT §7.2 "distance-to-count-change" margin: the count is
render-robust exactly while every disk is separated, and becomes pixel-ambiguous
as m -> 0 (two same-colour rims fuse into one connected component). Unlike the
distance worlds, gold does not flip at m = 0 — m measures difficulty and
pixel-verifiability of the (fixed) count, not the sign of the answer (design §2).

This is the bank's first **aggregation / counting** world: a variable
object count, an integer-valued answer, and a margin defined over the whole set
(design §6). ``N``, ``D`` and the radii are sampled independently of ``m`` so no
proxy (total ink, object count) is monotone with the crowding difficulty
(shortcut block, SSOT §21.2).

Colour palette (design §1). The oracle labels pixels by nearest palette colour;
for that to be leak-free the target, every present distractor colour, and the
white background must be mutually well separated AND no ink colour may sit near
another ink's rim->white antialiasing segment. The palette below is chosen so
that every ink's rim->white blend labels only as that ink or white (verified in
scripts/calibrate_counting_with_distractors.py); min pairwise RGB distance 145.8.
Distractor colours are sampled WITHOUT replacement, so each present distractor
colour marks exactly one disk.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw

RENDERER_VERSION = "counting_with_distractors-0.1.0"

# Target (the counted colour) and the leak-free distractor palette (design §1).
COLOR_TARGET = "red"
DISTRACTOR_PALETTE: tuple[str, ...] = ("blue", "cyan", "green", "indigo", "magenta", "yellow")
BACKGROUND = "white"

# Object counts and radii — sampled independently of m (design §2, §6).
N_TARGETS_RANGE = (2, 7)  # inclusive
N_DISTRACTORS_RANGE = (1, 6)  # inclusive
RADIUS_RANGE = (14.0, 30.0)  # px; well above pixel scale (no visibility ambiguity)

# SSOT §7.4 analogue — positive-only crowding bins (the sign is fixed; difficulty
# rides on magnitude). ``boundary`` is the QUARANTINED band (SSOT §21.6): below
# ~3 px two same-colour antialiased rims fuse into one connected component and the
# oracle can no longer certify the count. It is approached from the crowded side
# only; the sampler floor is FLOOR_PX = 3 and bin-balanced generation never enters
# it (see SAMPLED_BINS). bin_for_m still classifies it for downstream analysis.
MARGIN_BINS: dict[str, tuple[float, float]] = {
    "far_isolated": (40.0, 120.0),
    "spaced": (20.0, 40.0),
    "crowded": (8.0, 20.0),
    "tight": (3.0, 8.0),
    "boundary": (0.0, 3.0),
}

# Bins the sampler actually produces (all >= FLOOR_PX; boundary is quarantined).
SAMPLED_BINS: tuple[str, ...] = ("far_isolated", "spaced", "crowded", "tight")

# SSOT §7.4 — fixed positive-only sweep for continuity analysis. It deliberately
# probes below the FLOOR into the quarantined band (3, 2, 1); to keep every sweep
# point oracle-certifiable there, the sweep places the closest (min-gap) pair as a
# TARGET-DISTRACTOR (cross-colour) pair, so target disks are never that close and
# the target component count stays exactly N even at m = 1 (design §2, §5).
MARGIN_SWEEP: list[float] = [80, 60, 40, 30, 20, 12, 8, 5, 3, 2, 1]

# Sampler floor: no two disks are placed closer than this in bin-balanced mode, so
# same-colour (target-target) pairs never fuse — the merge distance is ~1-2 px at
# 512^2 and this sits above it with headroom (design §5; verified in calibration).
FLOOR_PX = 3.0
# Background pairwise-gap floor above the close pair, so the min gap is a single
# distinct pair and the second-smallest gap is clearly larger (design §2).
BACKGROUND_SEP_PX = 6.0


def bin_for_m(m: float) -> str:
    """First bin (in MARGIN_BINS order) whose closed [lo, hi] range contains m.

    Shared endpoints (e.g. m=8 in both crowded and tight) resolve to the earlier
    (more isolated) bin deterministically; tests pin the edge cases.
    """
    for name, (lo, hi) in MARGIN_BINS.items():
        if lo <= m <= hi:
            return name
    raise ValueError(f"m={m} outside all margin bins {MARGIN_BINS}")


Disk = tuple[float, float, float]  # (cx, cy, r)


@dataclass(frozen=True)
class CountingScene:
    """Full latent visual state for one rendered example (design §1).

    ``target_disks`` and ``distractor_disks`` are (cx, cy, r) triples;
    ``distractor_colors`` is parallel to ``distractor_disks`` (distinct palette
    colours). No two disks overlap or touch.
    """

    target_disks: list[Disk]
    distractor_disks: list[Disk]
    distractor_colors: list[str]
    color_target: str = COLOR_TARGET
    width: int = 512
    height: int = 512
    background: str = BACKGROUND
    antialias: bool = True

    @property
    def n_targets(self) -> int:
        return len(self.target_disks)

    @property
    def n_distractors(self) -> int:
        return len(self.distractor_disks)

    @property
    def count(self) -> int:
        """Closed-form gold (design §1): the number of target-coloured disks."""
        return len(self.target_disks)

    def all_disks(self) -> list[Disk]:
        return list(self.target_disks) + list(self.distractor_disks)

    @property
    def min_gap(self) -> float:
        """m = min over ALL pairs of disks of (centre distance - r_i - r_j)."""
        return min_pairwise_gap(self.all_disks())

    def to_dict(self) -> dict:
        """JSON-friendly dict (tuples -> lists) for the manifest ledger."""
        d = dataclasses.asdict(self)
        d["target_disks"] = [list(t) for t in self.target_disks]
        d["distractor_disks"] = [list(t) for t in self.distractor_disks]
        d["distractor_colors"] = list(self.distractor_colors)
        return d


def gap(a: Disk, b: Disk) -> float:
    """Signed gap between two disks: centre distance minus the two radii."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) - a[2] - b[2]


def min_pairwise_gap(disks: list[Disk]) -> float:
    """Minimum signed gap over all unordered pairs (exact geometry; no pixels)."""
    if len(disks) < 2:
        raise ValueError("need at least two disks for a pairwise gap")
    return min(gap(disks[i], disks[j]) for i in range(len(disks)) for j in range(i + 1, len(disks)))


def _fits_canvas(c: Disk, w: int, h: int, pad: float) -> bool:
    cx, cy, r = c
    return pad <= cx - r and cx + r <= w - pad and pad <= cy - r and cy + r <= h - pad


def _try_place(
    rng: random.Random,
    radii: list[float],
    m_target: float,
    w: int,
    h: int,
    pad: float,
    pos_tries: int,
) -> tuple[list[Disk], tuple[int, int]] | None:
    """Place K disks: K-1 mutually >= (m_target + SEP) apart, then one at gap
    exactly m_target from a chosen background disk and >= (m_target + SEP) from
    every other. Returns (disks, close_pair_indices) or None if it does not fit.
    """
    k = len(radii)
    g_bg = m_target + BACKGROUND_SEP_PX
    disks: list[Disk] = []
    for i in range(k - 1):
        for _ in range(pos_tries):
            cx = rng.uniform(pad + radii[i], w - pad - radii[i])
            cy = rng.uniform(pad + radii[i], h - pad - radii[i])
            cand = (cx, cy, radii[i])
            if all(gap(cand, d) >= g_bg for d in disks):
                disks.append(cand)
                break
        else:
            return None
    r_last = radii[k - 1]
    for _ in range(pos_tries):
        j = rng.randrange(len(disks))
        jx, jy, jr = disks[j]
        theta = rng.uniform(0.0, 2.0 * math.pi)
        d = jr + m_target + r_last  # centre distance for exact gap m_target
        cand = (jx + d * math.cos(theta), jy + d * math.sin(theta), r_last)
        if not _fits_canvas(cand, w, h, pad):
            continue
        if all(gap(cand, disks[idx]) >= g_bg for idx in range(len(disks)) if idx != j):
            disks.append(cand)
            return disks, (j, k - 1)
    return None


def sample_scene(
    rng: random.Random,
    margin_bin: str | None = None,
    m: float | None = None,
    radius_range: tuple[float, float] = RADIUS_RANGE,
    pad: float = 6.0,
    canvas: tuple[int, int] = (512, 512),
    meta_tries: int = 400,
    pos_tries: int = 400,
    **style,
) -> CountingScene:
    """Sample a scene whose minimum pairwise disk gap is ~m (design §2).

    Give either an explicit target ``m`` (sweep mode) or a ``margin_bin`` name.
    ``N``, ``D`` and the radii are sampled independently of ``m``. When ``m`` is
    given explicitly (sweep) the closest pair is forced to be a target-distractor
    (cross-colour) pair so the target count stays exact even below FLOOR_PX; in
    bin-balanced mode (m >= FLOOR_PX) the closest pair may be any pair, so the hard
    target-target crowding case is represented.

    For dense scenes in the isolated bins the required background separation cannot
    be packed for a large object count; ``N``/``D`` are then resampled (an isolated
    scene simply has fewer disks — a geometric fact, not a shortcut), so within the
    packable bins ``N`` remains decorrelated from ``m``.
    """
    w, h = canvas
    if m is None:
        if margin_bin is None:
            margin_bin = rng.choice(SAMPLED_BINS)
        lo, hi = MARGIN_BINS[margin_bin]
        m_target = rng.uniform(max(lo, FLOOR_PX), hi)
        cross_close = False
    else:
        m_target = float(m)
        cross_close = True

    for _ in range(meta_tries):
        n_targets = rng.randint(*N_TARGETS_RANGE)
        n_distractors = rng.randint(*N_DISTRACTORS_RANGE)
        k = n_targets + n_distractors
        radii = [rng.uniform(*radius_range) for _ in range(k)]
        placed = _try_place(rng, radii, m_target, w, h, pad, pos_tries)
        if placed is None:
            continue
        disks, (j, last) = placed
        colors = rng.sample(DISTRACTOR_PALETTE, n_distractors)

        # Assign roles. Indices are disk positions in ``disks``.
        idxs = list(range(k))
        if cross_close:
            # Force the close pair (j, last) to be one target + one distractor.
            t_of_pair, d_of_pair = (j, last) if rng.random() < 0.5 else (last, j)
            rest = [i for i in idxs if i not in (j, last)]
            rng.shuffle(rest)
            target_idx = [t_of_pair] + rest[: n_targets - 1]
            distractor_idx = [d_of_pair] + rest[n_targets - 1 :]
        else:
            rng.shuffle(idxs)
            target_idx = idxs[:n_targets]
            distractor_idx = idxs[n_targets:]

        target_disks = [disks[i] for i in target_idx]
        distractor_disks = [disks[i] for i in distractor_idx]
        scene = CountingScene(
            target_disks=target_disks,
            distractor_disks=distractor_disks,
            distractor_colors=list(colors),
            width=w,
            height=h,
            **style,
        )
        return scene
    raise RuntimeError(
        f"could not place a scene at m={m_target:.2f} (bin={margin_bin}) in canvas {canvas}"
    )


def render(scene: CountingScene, supersample: int = 4) -> Image.Image:
    """Render I = R(z). Antialiasing via supersampling + LANCZOS downscale."""
    ss = supersample if scene.antialias else 1
    img = Image.new("RGB", (scene.width * ss, scene.height * ss), scene.background)
    draw = ImageDraw.Draw(img)
    for cx, cy, r in scene.target_disks:
        draw.ellipse(
            [cx * ss - r * ss, cy * ss - r * ss, cx * ss + r * ss, cy * ss + r * ss],
            fill=scene.color_target,
        )
    for (cx, cy, r), color in zip(scene.distractor_disks, scene.distractor_colors, strict=True):
        draw.ellipse(
            [cx * ss - r * ss, cy * ss - r * ss, cx * ss + r * ss, cy * ss + r * ss],
            fill=color,
        )
    if ss > 1:
        img = img.resize((scene.width, scene.height), Image.LANCZOS)
    return img


def latent_symmetries(scene: CountingScene, rng: random.Random):
    """Answerability gate 1 (SSOT §7.14): the target disks are interchangeable (same
    colour) and the distractor disks carry their colour with them, so the disks live
    in positional list slots with no meaning to the order. Permuting the target list,
    and the (distractor disk, colour) list together, redraws the identical picture and
    leaves the count (gold) unchanged. Catches any gold keyed to list position."""
    from worlds._common.symmetries import permute_list_group

    yield (
        "permute_disk_lists",
        permute_list_group(
            scene, [("target_disks",), ("distractor_disks", "distractor_colors")], rng
        ),
    )


def scene_targets(scene: CountingScene, margin_bin: str) -> dict:
    """Task-specific ground truth for one example, stored in ExampleRecord.targets."""
    return {
        "count": scene.count,
        "n_targets": scene.n_targets,
        "n_distractors": scene.n_distractors,
        "min_gap": scene.min_gap,
        "margin_bin": margin_bin,
    }
