"""Pixel-level render-verification oracle for the counting world (SSOT §7.6).

An independent inverse renderer: it re-measures the rendered image with code that
shares NO geometry with ``renderer.render()`` and re-derives the count, every disk
(centre, radius) and the crowding margin m̂ from pixels alone. ``verify_example()``
then checks the measurement against the analytic scene and fails loudly on
disagreement. Ground truth is derived from the latent scene z, the model sees
I = R(z), and the inner builders need not look at images (historical SSOT §7.6),
so every generated example must pass the oracle before it enters a manifest.

Separation of powers:

- ``measure()`` never sees the scene geometry — it is told only the ink colours
  (target colour + the list of distractor colours present + background), NEVER the
  disk centres, radii, or the count. It COUNTS connected components of each colour
  mask and fits circles to them (design §4). The ground-truth count N and the
  pixel-measured component count are computed by disjoint code and must agree —
  the sharpest possible independence.
- ``verify_example()`` sees both and applies the tolerance policy.

Method (v0.1.0): label every pixel by NEAREST palette colour — antialiased blend
pixels split at the ~50% point, which sits on the true boundary, so masks carry no
threshold bias. The palette (renderer.DISTRACTOR_PALETTE + target + white) is
chosen leak-free: every ink's rim->white blend labels only as that ink or white.
On each colour mask, extract 4-connectivity connected components (implemented here
with union-find; scipy is not a dependency). Components of area >= MIN_DISK_PX are
"disk candidates"; the residual sub-threshold ink ("speckle", a few px of LANCZOS
ringing, e.g. the red+blue -> magenta averaging pixel) must stay below
SPECKLE_TOL_PX. The target mask must have exactly N disk candidates and each
present distractor colour exactly one; each candidate must be disk-like (Kåsa
circle fit residual small and pixel area ~ pi r^2). The min pairwise gap m̂ is
recomputed from the recovered disk centres/radii over ALL colours.

Empirical calibration (scripts/calibrate_counting_with_distractors.py; re-run to
re-derive). Tolerances keep >= 2x headroom over the observed maxima per SSOT §7.6
rule 2; agents must not relax them to make a failing dataset pass — an oracle
failure means the renderer or the scene is wrong, not the oracle.

Calibration = 2400 bin-sampled scenes (all four packable bins, seeds fixed) plus
the full positive sweep down to m = 1 (cross-colour close pair); see
CALIBRATION_NOTE below for the observed maxima and chosen tolerances.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageColor

ORACLE_VERSION = "counting_with_distractors-oracle-0.1.0"

# Public visual specification, duplicated deliberately rather than imported from
# the renderer. A disagreement is exposed by the image/scene certification path
# instead of becoming a shared-code blind spot.
_TARGET_COLOR = "red"
_DISTRACTOR_COLORS = ("blue", "cyan", "green", "indigo", "magenta", "yellow")
_BACKGROUND = "white"

# -- measurement constants ----------------------------------------------------
# A real disk (radius >= 14) covers >= pi*14^2 ~= 615 ink px; sub-threshold
# components are LANCZOS ringing speckle (<= ~16 px observed). 300 sits well below
# the smallest disk and well above the speckle.
MIN_DISK_PX = 300
_MIN_BOUNDARY_PTS = 12  # need this many rim pixels to attempt a circle fit
# Below this radius a disk washes out under antialiasing; the sampler uses
# [14, 30] so this floor has wide headroom.
MIN_DISK_RADIUS_PX = 8.0

# -- verification policy (tolerances pinned by tests) -------------------------
# Calibration numbers: see CALIBRATION_NOTE below. Do not relax (SSOT §7.6 rule 2).
CENTER_TOL_PX = 1.5
RADIUS_TOL_PX = 1.5
M_TOL_PX = 3.0
# Disk area check: |pixels - pi*r^2| <= AREA_TOL_FRAC*pi*r^2 + AREA_TOL_EDGE_PX*2*pi*r
# (antialiasing eats ~a half-pixel band around the rim).
AREA_TOL_FRAC = 0.06
AREA_TOL_EDGE_PX = 1.5
# Kåsa RMS residual of a genuine disk rim (px).
RESIDUAL_TOL_PX = 1.2
# Sub-threshold ("speckle") ink tolerated per colour mask (px). Benign LANCZOS
# ringing is a few px (e.g. the red+blue -> magenta averaging pixel); a genuine
# colour leak paints a whole rim arc (>= ~88 px for the smallest disk) and is
# caught here or by the disk-likeness check.
SPECKLE_TOL_PX = 30
# Two palette colours (incl. background) closer than this in RGB are not safely
# separable by nearest-colour labelling -> UnsupportedSceneError.
COLOR_MIN_DIST_PX = 60.0

# Calibration evidence behind the tolerances above (REAL run of
# scripts/calibrate_counting_with_distractors.py: 2400 bin-sampled + 1320 sweep
# scenes, count exact in every one). Tolerances keep >= 2x headroom (SSOT §7.6 rule 2).
CALIBRATION_NOTE = (
    "bin+sweep calibration (scripts/calibrate_counting_with_distractors.py, n=3720, "
    "count_fail=0): max |c_hat-c| 0.269px -> CENTER_TOL_PX 1.5 (5.6x); max |r_hat-r| "
    "0.539px -> RADIUS_TOL_PX 1.5 (2.8x); max |m_hat-m| 1.094px -> M_TOL_PX 3.0 (2.7x); "
    "max Kåsa rms 0.317px -> RESIDUAL_TOL_PX 1.2 (3.8x); max speckle/colour 5px -> "
    "SPECKLE_TOL_PX 30 (6.0x); max area dev/tol 0.246x."
)


class UnsupportedSceneError(ValueError):
    """Scene style the oracle cannot measure (shared/near colours, hairline disk)."""


class OracleFitError(ValueError):
    """Pixel-level measurement impossibility (e.g. a colour's ink is missing).

    Unlike UnsupportedSceneError this is evidence about the image, so
    verify_example() converts it into a failed report instead of raising.
    """


@dataclass(frozen=True)
class DiskFit:
    color: str
    cx: float
    cy: float
    r: float
    area_px: int
    n_boundary: int
    rms_residual: float
    touches_border: bool


@dataclass(frozen=True)
class Measurement:
    """Everything the oracle can say about an image without seeing the scene."""

    target_disks: list[DiskFit]
    distractor_disks: list[DiskFit]  # flattened across colours
    distractor_counts: dict[str, int]  # colour -> # disk candidates
    speckle_px: dict[str, int]  # colour -> sub-threshold ink
    m_hat: float
    size: tuple[int, int]

    def all_disks(self) -> list[DiskFit]:
        return list(self.target_disks) + list(self.distractor_disks)


@dataclass(frozen=True)
class OracleReport:
    ok: bool
    confidence: str  # always "high" — disks are opaque and never occlude (design §4)
    count: int  # analytic N
    count_hat: int  # recovered target component count
    m: float
    m_hat: float
    abs_err: float
    failures: list[str]  # empty iff ok
    version: str = ORACLE_VERSION

    def to_extra(self) -> dict:
        """Compact dict stored in ExampleRecord.extra["oracle"]."""
        d = asdict(self)
        d["abs_err"] = round(d["abs_err"], 3)
        d["m_hat"] = round(d["m_hat"], 3)
        return d


# -- independent pixel geometry (no imports from renderer.py; SSOT §7.6) -------


def _label_pixels(arr: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    """Assign every pixel to its nearest palette colour (Euclidean RGB)."""
    dists = np.stack(
        [np.linalg.norm(arr - np.array(c, dtype=np.float32), axis=-1) for c in palette], axis=-1
    )
    return dists.argmin(axis=-1)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connectivity connected components via union-find (scipy-free).

    Returns a list of boolean masks (full image size), one per component.
    Operates on the ink bounding box for speed.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    sub = mask[y0:y1, x0:x1]
    h, w = sub.shape
    lin = np.arange(h * w).reshape(h, w)
    parent = list(range(h * w))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    mr = sub[:, :-1] & sub[:, 1:]
    for a, b in zip(lin[:, :-1][mr].tolist(), lin[:, 1:][mr].tolist(), strict=True):
        union(a, b)
    md = sub[:-1, :] & sub[1:, :]
    for a, b in zip(lin[:-1, :][md].tolist(), lin[1:, :][md].tolist(), strict=True):
        union(a, b)

    groups: dict[int, list[int]] = {}
    for p in lin[sub].tolist():
        groups.setdefault(find(p), []).append(p)

    comps: list[np.ndarray] = []
    for pix in groups.values():
        arr = np.array(pix)
        yy = arr // w + y0
        xx = arr % w + x0
        full = np.zeros_like(mask)
        full[yy, xx] = True
        comps.append(full)
    return comps


def _boundary_pixels(comp: np.ndarray) -> np.ndarray:
    """(N, 2) (x, y) pixel-centre coords of component pixels 4-adjacent to outside."""
    interior = _shift(comp, 1, 0) & _shift(comp, -1, 0) & _shift(comp, 0, 1) & _shift(comp, 0, -1)
    boundary = comp & ~interior
    return np.argwhere(boundary).astype(np.float64)[:, ::-1] + 0.5


def _shift(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """mask shifted by (dy, dx) with False fill (a neighbour-present indicator)."""
    out = np.zeros_like(mask)
    ys = slice(max(dy, 0), mask.shape[0] + min(dy, 0))
    xs = slice(max(dx, 0), mask.shape[1] + min(dx, 0))
    ysrc = slice(max(-dy, 0), mask.shape[0] + min(-dy, 0))
    xsrc = slice(max(-dx, 0), mask.shape[1] + min(-dx, 0))
    out[ys, xs] = mask[ysrc, xsrc]
    return out


def _kasa_fit(pts: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Algebraic circle fit; returns (cx, cy, r, per-point |residual|)."""
    a = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    b = (pts**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    return cx, cy, r, resid


def _fit_disk(comp: np.ndarray, color: str, size: tuple[int, int]) -> DiskFit:
    pts = _boundary_pixels(comp)
    if len(pts) < _MIN_BOUNDARY_PTS:
        raise OracleFitError(f"{color}: component has only {len(pts)} rim px — cannot fit")
    cx, cy, r, resid = _kasa_fit(pts)
    ys, xs = np.nonzero(comp)
    w, h = size
    touches = bool(xs.min() == 0 or ys.min() == 0 or xs.max() == w - 1 or ys.max() == h - 1)
    return DiskFit(
        color=color,
        cx=cx,
        cy=cy,
        r=r,
        area_px=int(comp.sum()),
        n_boundary=len(pts),
        rms_residual=float(np.sqrt((resid**2).mean())),
        touches_border=touches,
    )


def _disks_of_color(
    labels: np.ndarray, class_idx: int, color: str, size: tuple[int, int]
) -> tuple[list[DiskFit], int]:
    """Disk candidates (area >= MIN_DISK_PX) and total sub-threshold speckle px."""
    fits: list[DiskFit] = []
    speckle = 0
    for comp in _connected_components(labels == class_idx):
        area = int(comp.sum())
        if area >= MIN_DISK_PX:
            fits.append(_fit_disk(comp, color, size))
        else:
            speckle += area
    return fits, speckle


def measure(
    image: Image.Image,
    color_target: str,
    distractor_colors: list[str],
    *,
    background: str = "white",
) -> Measurement:
    """Inverse-render the image. Sees only the ink colours, never the geometry."""
    names = [color_target, *distractor_colors, background]
    rgbs = [ImageColor.getrgb(c) for c in names]
    for i in range(len(rgbs)):
        for j in range(i + 1, len(rgbs)):
            if math.dist(rgbs[i], rgbs[j]) < COLOR_MIN_DIST_PX:
                raise UnsupportedSceneError(
                    f"colours not separable: {names[i]}/{names[j]} "
                    f"(RGB dist {math.dist(rgbs[i], rgbs[j]):.0f} < {COLOR_MIN_DIST_PX})"
                )
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    labels = _label_pixels(arr, list(rgbs))
    size = image.size

    target_fits, target_speckle = _disks_of_color(labels, 0, color_target, size)
    if not target_fits:
        raise OracleFitError(f"target({color_target}): no disk-sized components — nothing to count")

    speckle = {color_target: target_speckle}
    distractor_fits: list[DiskFit] = []
    distractor_counts: dict[str, int] = {}
    for k, color in enumerate(distractor_colors, start=1):
        fits, sp = _disks_of_color(labels, k, color, size)
        distractor_fits.extend(fits)
        distractor_counts[color] = len(fits)
        speckle[color] = sp

    all_fits = target_fits + distractor_fits
    if len(all_fits) < 2:
        raise OracleFitError("fewer than two disks recovered — cannot compute a pairwise gap")
    m_hat = min(
        math.hypot(a.cx - b.cx, a.cy - b.cy) - a.r - b.r
        for i, a in enumerate(all_fits)
        for b in all_fits[i + 1 :]
    )
    return Measurement(
        target_disks=target_fits,
        distractor_disks=distractor_fits,
        distractor_counts=distractor_counts,
        speckle_px=speckle,
        m_hat=m_hat,
        size=size,
    )


def decision_from_image(image: Image.Image) -> int:
    """Answerability gate 2 (SSOT §7.14): the target count recovered from pixels
    ALONE (no scene). Labels by the world's fixed palette and counts disk-sized
    connected components of the target colour. The full distractor palette is
    pairwise-separable (min RGB dist ~146), so passing it all cleanly isolates the
    target mask regardless of which distractor colours a given scene used."""
    meas = measure(
        image,
        _TARGET_COLOR,
        list(_DISTRACTOR_COLORS),
        background=_BACKGROUND,
    )
    return len(meas.target_disks)


def _match_error(
    true_disks: list[tuple[float, float, float]], fits: list[DiskFit]
) -> tuple[float, float]:
    """Greedy nearest-centre match; returns (max centre error, max radius error).

    Counts are assumed equal (verified by the caller). Each true disk claims its
    nearest unused recovered fit — a bijection when the disks are well separated.
    """
    remaining = list(fits)
    max_c = max_r = 0.0
    for tx, ty, tr in true_disks:
        best = min(remaining, key=lambda f: math.hypot(f.cx - tx, f.cy - ty))
        remaining.remove(best)
        max_c = max(max_c, math.hypot(best.cx - tx, best.cy - ty))
        max_r = max(max_r, abs(best.r - tr))
    return max_c, max_r


def verify_example(image: Image.Image, scene: Any) -> OracleReport:
    """Check that the rendered image matches the analytic scene."""
    for _, _, r in scene.all_disks():
        if r < MIN_DISK_RADIUS_PX:
            raise UnsupportedSceneError(
                f"disk radius {r:.1f} washes out under antialiasing "
                f"(oracle needs >= {MIN_DISK_RADIUS_PX}px)"
            )

    m = scene.min_gap
    n = scene.count
    failures: list[str] = []
    try:
        meas = measure(
            image, scene.color_target, scene.distractor_colors, background=scene.background
        )
    except OracleFitError as e:
        return OracleReport(
            ok=False, confidence="high", count=n, count_hat=0, m=m, m_hat=math.nan,
            abs_err=math.nan, failures=[str(e)],
        )  # fmt: skip

    if meas.size != (scene.width, scene.height):
        failures.append(f"image size {meas.size} != scene canvas {(scene.width, scene.height)}")

    count_hat = len(meas.target_disks)
    if count_hat != n:
        failures.append(f"target count {count_hat} != N {n} (miscount / merge / split)")
    for color, cnt in meas.distractor_counts.items():
        if cnt != 1:
            failures.append(f"distractor colour {color} has {cnt} disks (expected 1)")

    for color, sp in meas.speckle_px.items():
        if sp > SPECKLE_TOL_PX:
            failures.append(f"colour {color} has {sp}px sub-threshold ink (> {SPECKLE_TOL_PX})")

    for fit in meas.all_disks():
        if fit.rms_residual > RESIDUAL_TOL_PX:
            failures.append(
                f"{fit.color} disk not disk-like: rms residual {fit.rms_residual:.2f} "
                f"(> {RESIDUAL_TOL_PX})"
            )
        expected = math.pi * fit.r**2
        tol = AREA_TOL_FRAC * expected + AREA_TOL_EDGE_PX * (2 * math.pi * fit.r)
        if abs(fit.area_px - expected) > tol:
            failures.append(
                f"{fit.color} disk area {fit.area_px}px vs pi*r^2 {expected:.0f}px (tol {tol:.0f})"
            )
        if fit.touches_border:
            failures.append(f"{fit.color} disk touches image border (clip/merge)")

    # Per-colour geometry vs the analytic disks (catches radius/position tampers a
    # global m̂ might miss). Only when the counts already line up.
    if count_hat == n:
        cerr, rerr = _match_error(scene.target_disks, meas.target_disks)
        if cerr > CENTER_TOL_PX:
            failures.append(f"target centre off by {cerr:.2f}px (> {CENTER_TOL_PX})")
        if rerr > RADIUS_TOL_PX:
            failures.append(f"target radius off by {rerr:.2f}px (> {RADIUS_TOL_PX})")
    for color in set(scene.distractor_colors):
        true = [
            d
            for d, c in zip(scene.distractor_disks, scene.distractor_colors, strict=True)
            if c == color
        ]
        got = [f for f in meas.distractor_disks if f.color == color]
        if len(true) == len(got):
            cerr, rerr = _match_error(true, got)
            if cerr > CENTER_TOL_PX:
                failures.append(f"{color} centre off by {cerr:.2f}px (> {CENTER_TOL_PX})")
            if rerr > RADIUS_TOL_PX:
                failures.append(f"{color} radius off by {rerr:.2f}px (> {RADIUS_TOL_PX})")

    if abs(meas.m_hat - m) > M_TOL_PX:
        failures.append(f"m_hat {meas.m_hat:.2f} vs m {m:.2f} (> {M_TOL_PX}px off)")
    if meas.m_hat <= 0:
        failures.append(f"m_hat {meas.m_hat:.2f} <= 0 (disks overlap/touch in pixels)")

    return OracleReport(
        ok=not failures,
        confidence="high",
        count=n,
        count_hat=count_hat,
        m=m,
        m_hat=meas.m_hat,
        abs_err=abs(meas.m_hat - m),
        failures=failures,
    )
