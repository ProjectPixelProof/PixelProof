"""Pixel-level render-verification oracle (SSOT §7.6).

An independent inverse renderer: it re-measures the rendered image with code
that shares no geometry with ``renderer.render()`` and re-estimates the circle
parameters and signed gap m̂ from pixels alone. ``verify_example()`` then checks
the measurement against the analytic scene and fails loudly on disagreement.

Why this exists: ground truth is derived from the latent scene z, the model
sees I = R(z), and the inner builders need not look at images
(SSOT §7.6). Without a pixel-level check, a renderer bug (swapped colors,
wrong radius, stroke eating the boundary) would silently poison every
downstream result while passing all determinism tests. Every generated
example must pass the oracle before it enters a manifest.

Separation of powers:

- ``measure()`` never sees the scene geometry — it only gets the two ink
  colors and whether an overlay strip must be ignored, so it cannot cheat.
- ``verify_example()`` sees both and applies the tolerance policy.

Method (v0.2.0): label every pixel by NEAREST palette color (circle colors +
background + black when an overlay is present) — antialiased blend pixels
split at the ~50% point, which sits on the true circle boundary, so masks
carry no threshold bias. Each circle's "exposed rim" is its ink pixels
4-adjacent to background: the occlusion boundary (ink-vs-ink) is excluded
structurally, so the two Kåsa fits are fully independent of each other. One
outlier-trimming refinement per fit. An independent pixel-count check compares
each mask's area against the analytic visible area (disk minus lens), which
stays sharp even when the visible arc of circle 1 is a sliver and the fit is
ill-conditioned.

The exposed-rim + nearest-color formulation was adopted from the VLMH1 agent
smoke test, whose independently-designed oracle was structurally cleaner than
v0.1.0's threshold masks + occlusion-band filtering (which coupled circle 1's
fit to circle 2's).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageColor

# 0.2.0: nearest-color labeling + exposed-rim fits (see module docstring).
ORACLE_VERSION = "two_circles-oracle-0.2.0"

# -- measurement constants ----------------------------------------------------
# Overlay text is drawn anchored at (w/2, 24) with a size-28 font (renderer.py);
# rim measurements ignore this top strip when an overlay is present (gray
# antialiased text edges can label as ink under nearest-color assignment).
OVERLAY_STRIP_PX = 48
# Kåsa refinement: drop rim points farther than this from the first fit.
_REFIT_RESIDUAL_PX = 2.0
_MIN_BOUNDARY_PTS = 12

# -- verification policy (tolerances pinned by tests/test_oracle.py) ----------
# High-confidence fits (visible arc >= MIN_ARC_DEG on both circles).
# v0.2.0 calibration (500 bin-sampled scenes + sampler-floor slivers):
# max |m̂-m| 1.19px in-bin / 1.22px at the sliver floor, max radius err 0.57px,
# max center err 0.26px. Tolerances keep >2x headroom for cross-platform
# antialiasing differences (datasets may be rendered on the box).
CENTER_TOL_PX = 2.0
RADIUS_TOL_PX = 2.0
M_TOL_PX = 3.0
# Below this visible-arc coverage the circle-1 fit is ill-conditioned; the
# report downgrades to confidence="low" and skips its parameter checks.
MIN_ARC_DEG = 60.0
# Label agreement is only required when |m| is clearly off the boundary
# (SSOT §21.6: the tangent band is ambiguous at pixel scale).
LABEL_BAND_PX = 1.5
# In low confidence, m̂ inherits circle-1 fit error; only the sign is checked,
# and only when the true |m| is clearly away from zero.
LOW_CONF_SIGN_BAND_PX = 6.0
# Area check: |pixels - analytic visible area| <= AREA_TOL_FRAC * analytic
# + AREA_TOL_RIM_PX * visible rim length (antialiasing eats ~a half-pixel band).
AREA_TOL_FRAC = 0.02
AREA_TOL_RIM_PX = 1.0


class UnsupportedSceneError(ValueError):
    """Scene style the oracle cannot measure yet (outline fill, shared colors)."""


class OracleFitError(ValueError):
    """Pixel-level measurement impossibility (e.g. a circle's ink is missing).

    Unlike UnsupportedSceneError this is evidence about the image, so
    verify_example() converts it into a failed report instead of raising.
    """


@dataclass(frozen=True)
class CircleFit:
    cx: float
    cy: float
    r: float
    n_points: int
    arc_deg: float  # angular coverage of inlier boundary points, degrees
    rms_residual: float


@dataclass(frozen=True)
class Measurement:
    """Everything the oracle can say about an image without seeing the scene."""

    circle1: CircleFit
    circle2: CircleFit
    m_hat: float
    area1_px: int  # visible (non-occluded) ink pixels per circle
    area2_px: int
    size: tuple[int, int]


@dataclass(frozen=True)
class OracleReport:
    ok: bool
    confidence: str  # "high" | "low" (low: circle-1 arc < MIN_ARC_DEG)
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


def _label_pixels(arr: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    """Assign every pixel to its nearest palette color (Euclidean RGB).

    Unlike a distance threshold, argmin labeling leaves no unclassified blend
    band: antialiased edge pixels split at the ~50% mix, which lies on the true
    circle boundary in expectation.
    """
    dists = np.stack(
        [np.linalg.norm(arr - np.array(c, dtype=np.float32), axis=-1) for c in palette], axis=-1
    )
    return dists.argmin(axis=-1)


def _exposed_rim(mask: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    """Ink pixels 4-adjacent to background — the disk's true outer arc.

    The occlusion boundary (ink touching the other disk's ink) is excluded
    structurally, so each circle's fit is independent of the other's.
    """
    near_bg = np.zeros_like(bg_mask)
    near_bg[1:] |= bg_mask[:-1]
    near_bg[:-1] |= bg_mask[1:]
    near_bg[:, 1:] |= bg_mask[:, :-1]
    near_bg[:, :-1] |= bg_mask[:, 1:]
    return mask & near_bg


def _rim_points(mask: np.ndarray, bg_mask: np.ndarray, ignore_top: int) -> np.ndarray:
    """(N, 2) array of (x, y) pixel-centre coordinates of exposed-rim pixels."""
    pts = np.argwhere(_exposed_rim(mask, bg_mask)).astype(np.float64)[:, ::-1] + 0.5
    if ignore_top:
        pts = pts[pts[:, 1] >= ignore_top]
    return pts


def _kasa_fit(pts: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Algebraic circle fit; returns (cx, cy, r, per-point |residual|)."""
    a = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
    b = (pts**2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    resid = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r)
    return cx, cy, r, resid


def _fit_circle(pts: np.ndarray, what: str) -> CircleFit:
    if len(pts) < _MIN_BOUNDARY_PTS:
        raise OracleFitError(f"{what}: only {len(pts)} boundary pixels — nothing to fit")
    cx, cy, r, resid = _kasa_fit(pts)
    inliers = pts[resid <= _REFIT_RESIDUAL_PX]
    if len(inliers) >= _MIN_BOUNDARY_PTS and len(inliers) < len(pts):
        cx, cy, r, resid = _kasa_fit(inliers)
        pts = inliers
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    arc_deg = 5.0 * len(np.unique((np.degrees(angles) // 5).astype(int)))
    return CircleFit(cx, cy, r, len(pts), arc_deg, float(np.sqrt((resid**2).mean())))


def measure(
    image: Image.Image,
    color1: str,
    color2: str,
    *,
    background: str = "white",
    has_overlay: bool = False,
) -> Measurement:
    """Inverse-render the image. Sees only ink colors, never the scene geometry.

    Each circle is fitted from its exposed rim (ink adjacent to background),
    so the two fits are independent — draw order does not matter.
    """
    rgbs = [ImageColor.getrgb(c) for c in (color1, color2, background)]
    if len(set(rgbs)) < 3:
        raise UnsupportedSceneError(f"colors not pairwise distinct: {color1}/{color2}/{background}")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    palette = list(rgbs)
    if has_overlay:
        palette.append((0, 0, 0))  # overlay text ink gets its own class
    labels = _label_pixels(arr, palette)
    mask1, mask2, bg = labels == 0, labels == 1, labels == 2
    strip = OVERLAY_STRIP_PX if has_overlay else 0

    fit1 = _fit_circle(_rim_points(mask1, bg, strip), f"circle1({color1})")
    fit2 = _fit_circle(_rim_points(mask2, bg, strip), f"circle2({color2})")

    m_hat = math.hypot(fit1.cx - fit2.cx, fit1.cy - fit2.cy) - fit1.r - fit2.r
    return Measurement(
        circle1=fit1,
        circle2=fit2,
        m_hat=m_hat,
        area1_px=int(mask1.sum()),
        area2_px=int(mask2.sum()),
        size=image.size,
    )


def decision_from_image(image: Image.Image) -> float:
    """Answerability gate 2 (SSOT §7.14): the signed gap recovered from pixels
    ALONE (no scene). Fits both disks from their exposed rims and returns
    m_hat = ||c1-c2|| - r1 - r2; the overlap decision is sign(m_hat). The palette
    is the world's fixed default (blue/red/white); the sign is orientation-free
    (the gap is symmetric), so no latent convention enters."""
    return measure(image, "blue", "red", background="white").m_hat


def _lens_area(r1: float, r2: float, d: float) -> float:
    """Area of intersection of two disks (standard two-segment formula)."""
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2) ** 2
    a1 = math.acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = math.acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    return r1 * r1 * (a1 - math.sin(2 * a1) / 2) + r2 * r2 * (a2 - math.sin(2 * a2) / 2)


def verify_example(image: Image.Image, scene: Any) -> OracleReport:
    """Check that the rendered image matches the analytic scene.

    Raises UnsupportedSceneError for styles the oracle cannot measure
    (outline fill, shared colors); returns a report with ok=False and the
    failed checks otherwise.
    """
    if scene.fill != "solid":
        raise UnsupportedSceneError(f"fill={scene.fill!r} not supported by oracle v1")

    m = scene.signed_gap
    failures: list[str] = []
    try:
        meas = measure(
            image,
            scene.color1,
            scene.color2,
            background=scene.background,
            has_overlay=scene.overlay_text is not None,
        )
    except OracleFitError as e:
        # Missing ink (e.g. a circle never drawn) is evidence about the image:
        # a verification failure, not an unsupported style.
        return OracleReport(
            ok=False, confidence="high", m=m, m_hat=math.nan, abs_err=math.nan,
            failures=[str(e)],
        )  # fmt: skip

    if meas.size != (scene.width, scene.height):
        failures.append(f"image size {meas.size} != scene canvas {(scene.width, scene.height)}")

    confidence = "high" if meas.circle1.arc_deg >= MIN_ARC_DEG else "low"

    def _check_fit(fit: CircleFit, c: tuple[float, float], r: float, name: str) -> None:
        dc = math.hypot(fit.cx - c[0], fit.cy - c[1])
        if dc > CENTER_TOL_PX:
            failures.append(f"{name} center off by {dc:.2f}px (> {CENTER_TOL_PX})")
        if abs(fit.r - r) > RADIUS_TOL_PX:
            failures.append(f"{name} radius {fit.r:.2f} vs {r:.2f} (> {RADIUS_TOL_PX}px off)")

    _check_fit(meas.circle2, scene.c2, scene.r2, "circle2")
    if confidence == "high":
        _check_fit(meas.circle1, scene.c1, scene.r1, "circle1")
        if abs(meas.m_hat - m) > M_TOL_PX:
            failures.append(f"m_hat {meas.m_hat:.2f} vs m {m:.2f} (> {M_TOL_PX}px off)")
        if abs(m) > LABEL_BAND_PX and (meas.m_hat <= 0) != (m <= 0):
            failures.append(f"label flip: m_hat {meas.m_hat:.2f} vs m {m:.2f}")
    elif abs(m) > LOW_CONF_SIGN_BAND_PX and (meas.m_hat <= 0) != (m <= 0):
        failures.append(f"low-confidence sign flip: m_hat {meas.m_hat:.2f} vs m {m:.2f}")

    # Independent of the fits: visible ink area vs analytic prediction. This is
    # the check that stays sharp when circle 1 is a sliver. Skipped under an
    # overlay (text may punch holes in either disk).
    if scene.overlay_text is None:
        d = math.dist(scene.c1, scene.c2)
        lens = _lens_area(scene.r1, scene.r2, d)
        for name, area_px, expected, rim in (
            ("circle1", meas.area1_px, math.pi * scene.r1**2 - lens, 2 * math.pi * scene.r1),
            ("circle2", meas.area2_px, math.pi * scene.r2**2, 2 * math.pi * scene.r2),
        ):
            tol = AREA_TOL_FRAC * expected + AREA_TOL_RIM_PX * rim
            if abs(area_px - expected) > tol:
                failures.append(
                    f"{name} visible area {area_px}px vs analytic {expected:.0f}px (tol {tol:.0f})"
                )

    return OracleReport(
        ok=not failures,
        confidence=confidence,
        m=m,
        m_hat=meas.m_hat,
        abs_err=abs(meas.m_hat - m),
        failures=failures,
    )
