"""Pixel-level render-verification oracle for the angle-acuteness world (SSOT §7.6).

An independent inverse renderer: it re-measures the rendered image with code that
shares no geometry with ``renderer.render()`` and re-estimates the interior angle
θ̂ (and hence m̂ = 90 − θ̂) from pixels alone. ``verify_example()`` then checks the
measurement against the analytic scene and fails loudly on disagreement.

Why this exists: ground truth is derived from the latent scene z, the model sees
I = R(z), and the inner builders need not look at images (historical SSOT §7.6).
Without a pixel-level check, a renderer bug (wrong angle, wrong heading, missing
ray) would silently poison every downstream result while passing all determinism
tests. Every generated example must pass the oracle before it enters a manifest.

Separation of powers:

- ``measure()`` never sees the scene geometry — it only gets the ink colour, the
  background, and the stroke width, so it cannot cheat.
- ``verify_example()`` sees both and applies the tolerance policy.

Method (v0.1.0): label every pixel by NEAREST palette colour (ink + background) —
one ink mask shaped like a "V". The design keeps BOTH rays the same colour (the
object is a single angle); this oracle splits the single mask into two rays by a
deterministic two-line fit rather than being weakened to two colours (design §4):
(1) RANSAC the dominant straight line through the ink (fixed-seed, so
deterministic); (2) RANSAC a second line through the residual pixels; (3) reassign
every ink pixel to its nearest line and refit each ray's direction by PCA,
excluding a small disk around the corner where the two strokes overlap; (4) the
vertex V̂ is the intersection of the two refit lines, each ray direction is
oriented away from V̂, and θ̂ is the angle between them. Each ray's far endpoint is
the extremal projection along its direction. m̂ = 90 − θ̂ is a fresh subtraction.

Empirical calibration (scripts/calibrate_angle_acuteness.py; re-run to re-derive).
Tolerances keep >= 2x headroom over the observed maxima per SSOT §7.6 rule 2;
agents must not relax them to make a failing dataset pass — an oracle failure means
the renderer or the scene is wrong, not the oracle.

Calibration = 1500 bin-sampled scenes (300/bin) over all headings + lengths, plus
400 forced worst-case scenes (both rays at the 120px length floor, 5px stroke
floor, θ at the |m| in [1,2]deg boundary, all headings):

- interior-angle recovery: max |θ̂ - θ| = 0.395deg -> ANGLE_TOL_DEG = 1.5   (3.8x)
- vertex recovery:         max |v̂ - v|   = 0.477px  -> VERTEX_TOL_PX = 2.0   (4.2x)
- ray-endpoint recovery:   max |ê - e|   = 0.689px  -> ENDPOINT_TOL_PX = 2.5  (3.6x)
- sign of m̂ (acute/obtuse): flipped only for true |m| <= 0.104deg
                           -> RIGHT_BAND_DEG = 2.0 (matches the quarantined right
                           bin; the label is certified for every non-right scene)
- ink area, frac*expected + edge_px*perimeter: max deviation = 0.09x the tol
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageColor

ORACLE_VERSION = "angle_acuteness-oracle-0.1.0"

# -- measurement constants ----------------------------------------------------
_MIN_INK_PX = 300  # a legal figure covers >= 2 * 120 * 5 = 1200 ink px
_MIN_STROKE_PX = 3.0  # below this a stroke washes out under antialiasing
_RANSAC_TRIALS = 400
_RANSAC_MIN_SEED_SEP = 40.0  # seed pairs must be this far apart to fix a direction
_SEED = 0  # fixed RNG seed -> deterministic split for a given pixel set

# -- verification policy (tolerances pinned by tests/test_angle_acuteness_oracle.py)
# Calibration numbers: module docstring. Do not relax (SSOT §7.6 rule 2).
ANGLE_TOL_DEG = 1.5
VERTEX_TOL_PX = 2.0
ENDPOINT_TOL_PX = 2.5
# The acute/obtuse label is only required to agree when |m| is clearly off the
# right-angle boundary (SSOT §21.6: the right band is ambiguous at pixel scale).
# Sign flip observed only at true |m| <= 0.10deg; 2.0 equals the quarantined right
# bin edge, so the label is certified for every non-right scene (|m| > 2deg).
RIGHT_BAND_DEG = 2.0
# Ink-area check: |pixels - (ℓ1+ℓ2)*s + overlap| <= AREA_TOL_FRAC * expected
# + AREA_TOL_EDGE_PX * perimeter (antialiasing eats ~a half-pixel band).
AREA_TOL_FRAC = 0.04
AREA_TOL_EDGE_PX = 1.5


class UnsupportedSceneError(ValueError):
    """Scene style the oracle cannot measure yet (shared colours, hairline stroke)."""


class OracleFitError(ValueError):
    """Pixel-level measurement impossibility (e.g. the ink cannot be split in two).

    Unlike UnsupportedSceneError this is evidence about the image, so
    verify_example() converts it into a failed report instead of raising.
    """


@dataclass(frozen=True)
class Measurement:
    """Everything the oracle can say about an image without seeing the scene."""

    vertex: tuple[float, float]
    end1: tuple[float, float]  # recovered ray far endpoints
    end2: tuple[float, float]
    theta_hat: float  # interior angle, degrees, in (0, 180)
    m_hat: float  # 90 - theta_hat
    ink_area_px: int
    size: tuple[int, int]


@dataclass(frozen=True)
class OracleReport:
    ok: bool
    confidence: str  # always "high" — a single opaque figure, no occlusion
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
        d["theta_hat"] = round(90.0 - d["m_hat"], 3)
        return d


# -- independent geometry (no imports from renderer.py; SSOT §7.6) ------------


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _polygon_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _stroke_rect(v: np.ndarray, e: np.ndarray, width: float) -> np.ndarray:
    """Corners of the butt-capped stroke rectangle from v to e (CCW)."""
    d = e - v
    u = d / np.linalg.norm(d)
    n = np.array([-u[1], u[0]]) * (width / 2.0)
    poly = np.array([v + n, e + n, e - n, v - n])
    x, y = poly[:, 0], poly[:, 1]
    if float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) < 0:
        poly = poly[::-1]
    return poly


def _clip_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip a convex polygon by a CCW convex polygon."""
    output = list(subject)
    for i in range(len(clip)):
        c1, c2 = clip[i], clip[(i + 1) % len(clip)]
        edge = c2 - c1
        inputs, output = output, []
        for j in range(len(inputs)):
            cur, nxt = inputs[j], inputs[(j + 1) % len(inputs)]
            cur_in = _cross2(edge, cur - c1) >= 0
            nxt_in = _cross2(edge, nxt - c1) >= 0
            if cur_in:
                output.append(cur)
            if cur_in != nxt_in:
                s = _cross2(edge, c1 - cur) / _cross2(edge, nxt - cur)
                output.append(cur + s * (nxt - cur))
        if not output:
            return np.empty((0, 2))
    return np.array(output)


def _stroke_overlap_area(v: np.ndarray, e1: np.ndarray, e2: np.ndarray, width: float) -> float:
    """Analytic area shared by the two stroke rectangles near the vertex."""
    inter = _clip_convex(_stroke_rect(v, e1, width), _stroke_rect(v, e2, width))
    return _polygon_area(inter) if len(inter) >= 3 else 0.0


# -- pixel measurement ---------------------------------------------------------


def _label_pixels(arr: np.ndarray, palette: list[tuple[int, int, int]]) -> np.ndarray:
    dists = np.stack(
        [np.linalg.norm(arr - np.array(c, dtype=np.float32), axis=-1) for c in palette], axis=-1
    )
    return dists.argmin(axis=-1)


def _pca_direction(pts: np.ndarray) -> np.ndarray:
    centered = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0]


def _ransac_line(pts: np.ndarray, tol: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean inlier mask of the highest-support straight line through ``pts``."""
    n = len(pts)
    best_mask = np.zeros(n, dtype=bool)
    best_count = 0
    for _ in range(_RANSAC_TRIALS):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        a, b = pts[i], pts[j]
        d = b - a
        if np.hypot(d[0], d[1]) < _RANSAC_MIN_SEED_SEP:
            continue
        u = d / np.linalg.norm(d)
        nrm = np.array([-u[1], u[0]])
        perp = np.abs((pts - a) @ nrm)
        mask = perp <= tol
        c = int(mask.sum())
        if c > best_count:
            best_count, best_mask = c, mask
    return best_mask


def _line_intersection(
    c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray
) -> np.ndarray:
    """Intersection of line (c1 + t d1) and (c2 + s d2)."""
    a = np.column_stack([d1, -d2])
    t, _ = np.linalg.solve(a, c2 - c1)
    return c1 + t * d1


def measure(
    image: Image.Image,
    color_ink: str,
    *,
    background: str = "white",
    stroke_width: float,
) -> Measurement:
    """Inverse-render the image. Sees only the ink colour + stroke, never the geometry."""
    rgbs = [ImageColor.getrgb(c) for c in (color_ink, background)]
    if len(set(rgbs)) < 2:
        raise UnsupportedSceneError(f"ink and background share a colour: {color_ink}/{background}")
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    labels = _label_pixels(arr, list(rgbs))
    pts = np.argwhere(labels == 0).astype(np.float64)[:, ::-1] + 0.5  # (x, y) centres
    if len(pts) < _MIN_INK_PX:
        raise OracleFitError(f"ink({color_ink}): only {len(pts)} pixels — nothing to fit")
    if len(pts) > 8000:  # deterministic thinning keeps RANSAC fast without bias
        pts = pts[:: len(pts) // 8000 + 1]

    rng = np.random.default_rng(_SEED)
    tol = stroke_width / 2.0 + 1.5
    mask1 = _ransac_line(pts, tol, rng)
    if mask1.sum() < _MIN_INK_PX // 3 or mask1.all():
        raise OracleFitError("could not fit a first ray line")
    rest = pts[~mask1]
    if len(rest) < _MIN_INK_PX // 3:
        raise OracleFitError("no residual pixels for a second ray line")
    mask2_rest = _ransac_line(rest, tol, np.random.default_rng(_SEED + 1))
    if mask2_rest.sum() < _MIN_INK_PX // 3:
        raise OracleFitError("could not fit a second ray line")

    # Rough directions/centroids, then reassign ALL points to the nearest line.
    c1, d1 = pts[mask1].mean(axis=0), _pca_direction(pts[mask1])
    c2 = rest[mask2_rest].mean(axis=0)
    d2 = _pca_direction(rest[mask2_rest])
    n1, n2 = np.array([-d1[1], d1[0]]), np.array([-d2[1], d2[0]])
    perp1 = np.abs((pts - c1) @ n1)
    perp2 = np.abs((pts - c2) @ n2)
    assign1 = perp1 <= perp2

    v = _line_intersection(c1, d1, c2, d2)
    # Refit each ray excluding the corner region where the strokes overlap.
    far = np.hypot(pts[:, 0] - v[0], pts[:, 1] - v[1]) > 2.5 * stroke_width
    grp1, grp2 = pts[assign1 & far], pts[(~assign1) & far]
    if len(grp1) < _MIN_INK_PX // 3 or len(grp2) < _MIN_INK_PX // 3:
        raise OracleFitError("degenerate two-line split (a ray has too few far pixels)")
    d1, d2 = _pca_direction(grp1), _pca_direction(grp2)
    v = _line_intersection(grp1.mean(axis=0), d1, grp2.mean(axis=0), d2)

    # Orient each direction away from the vertex, into its ray.
    if (grp1.mean(axis=0) - v) @ d1 < 0:
        d1 = -d1
    if (grp2.mean(axis=0) - v) @ d2 < 0:
        d2 = -d2
    len1 = float(((grp1 - v) @ d1).max())
    len2 = float(((grp2 - v) @ d2).max())
    end1 = v + len1 * d1
    end2 = v + len2 * d2

    cos = float(np.clip(d1 @ d2, -1.0, 1.0))
    theta_hat = math.degrees(math.acos(cos))
    return Measurement(
        vertex=(float(v[0]), float(v[1])),
        end1=(float(end1[0]), float(end1[1])),
        end2=(float(end2[0]), float(end2[1])),
        theta_hat=theta_hat,
        m_hat=90.0 - theta_hat,
        ink_area_px=int((labels == 0).sum()),
        size=image.size,
    )


def decision_from_image(image: Image.Image) -> float:
    """Answerability gate 2 (SSOT §7.14): the signed angular margin m̂ = 90 − θ̂ recovered
    from pixels ALONE (no scene). Splits the single V-shaped ink mask into two rays by the
    same deterministic two-line fit ``measure`` uses and returns 90 − θ̂; the acute/obtuse
    decision is sign(m̂). The ink colour is the world's fixed default (blue on white). The
    per-scene stroke width is unavailable without the scene, so a fixed representative
    width (the top of STROKE_RANGE) is passed: it only sets the RANSAC inlier band and the
    corner-exclusion radius, not the recovered ray directions, so θ̂ (and hence the SIGN
    that is the decision) is insensitive to it off the quarantined right band."""
    return measure(image, "blue", background="white", stroke_width=9.0).m_hat


def _match_endpoints(
    recovered: tuple[tuple[float, float], tuple[float, float]],
    declared: tuple[tuple[float, float], tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    r1, r2 = recovered
    d1, d2 = declared
    straight = math.dist(r1, d1) + math.dist(r2, d2)
    swapped = math.dist(r1, d2) + math.dist(r2, d1)
    return [(r1, d1), (r2, d2)] if straight <= swapped else [(r1, d2), (r2, d1)]


def verify_example(image: Image.Image, scene: Any) -> OracleReport:
    """Check that the rendered image matches the analytic scene."""
    if scene.stroke < _MIN_STROKE_PX:
        raise UnsupportedSceneError(
            f"stroke={scene.stroke} washes out under antialiasing "
            f"(oracle v1 needs >= {_MIN_STROKE_PX}px)"
        )

    m = scene.signed_margin
    failures: list[str] = []
    try:
        meas = measure(
            image, scene.color_ink, background=scene.background, stroke_width=scene.stroke
        )
    except OracleFitError as e:
        return OracleReport(
            ok=False, confidence="high", m=m, m_hat=math.nan, abs_err=math.nan,
            failures=[str(e)],
        )  # fmt: skip

    if meas.size != (scene.width, scene.height):
        failures.append(f"image size {meas.size} != scene canvas {(scene.width, scene.height)}")

    verr = math.hypot(meas.vertex[0] - scene.vx, meas.vertex[1] - scene.vy)
    if verr > VERTEX_TOL_PX:
        failures.append(f"vertex off by {verr:.2f}px (> {VERTEX_TOL_PX})")

    for recovered, target in _match_endpoints(
        (meas.end1, meas.end2), (scene.ray1_end, scene.ray2_end)
    ):
        err = math.dist(recovered, target)
        if err > ENDPOINT_TOL_PX:
            failures.append(
                f"ray endpoint {tuple(round(v, 1) for v in recovered)} off declared "
                f"{tuple(round(v, 1) for v in target)} by {err:.2f}px (> {ENDPOINT_TOL_PX})"
            )

    if abs(meas.theta_hat - scene.theta) > ANGLE_TOL_DEG:
        failures.append(
            f"theta_hat {meas.theta_hat:.2f} vs theta {scene.theta:.2f} (> {ANGLE_TOL_DEG} deg)"
        )
    if abs(m) > RIGHT_BAND_DEG and (meas.m_hat > 0) != (m > 0):
        failures.append(f"label flip: m_hat {meas.m_hat:.2f} vs m {m:.2f} (deg)")

    # Independent of the angle fit: total visible ink vs analytic stroke area.
    v = np.array([scene.vx, scene.vy])
    e1, e2 = np.array(scene.ray1_end), np.array(scene.ray2_end)
    overlap = _stroke_overlap_area(v, e1, e2, scene.stroke)
    expected = scene.len1 * scene.stroke + scene.len2 * scene.stroke - overlap
    perimeter = 2 * (scene.len1 + scene.len2 + 2 * scene.stroke)
    tol = AREA_TOL_FRAC * expected + AREA_TOL_EDGE_PX * perimeter
    if abs(meas.ink_area_px - expected) > tol:
        failures.append(
            f"ink area {meas.ink_area_px}px vs analytic {expected:.0f}px (tol {tol:.0f})"
        )

    return OracleReport(
        ok=not failures,
        confidence="high",
        m=m,
        m_hat=meas.m_hat,
        abs_err=abs(meas.m_hat - m),
        failures=failures,
    )
