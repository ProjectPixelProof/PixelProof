"""Angle-acuteness world renderer (SSOT §7.2, §7.7; design docs/designs/angle_acuteness.md).

Latent state z = {vertex, two rays, ink style, canvas}. Two straight strokes of the
SAME colour share the endpoint ``vertex``; the interior angle between them is θ
degrees. The image is I = R(z) and the signed angular margin is

    m = 90 - θ            (degrees from a right angle)

with m > 0 acute (θ < 90), m == 0 a right angle, and m < 0 obtuse (θ > 90). The
decision the world asks is *is the angle acute* — the bank's first ANGULAR
decision variable (measured in degrees, not pixels), orthogonal to every distance
margin (two_circles, lines, relative_length, point_in_polygon).

Both rays are the SAME colour on purpose (the object is a single angle, not two
rankable things); the oracle splits the single "V" mask into two rays by a
two-line fit — it is NOT weakened to two colours (design §4, SSOT §7.6).

Shortcut blocking (SSOT §21.2): the overall figure heading α, the two ray lengths,
the vertex position, and the stroke width are all randomized independently of the
label, so a model cannot shortcut on any single ray's orientation, on which ray is
"steeper", or on the figure's size. Only the relative angle θ carries the answer.
"""

from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw

RENDERER_VERSION = "angle_acuteness-0.1.0"

# Ray length / stroke sampled uniformly from these ranges (canvas px). Long rays
# reduce the heading-fit error (design §5): δ_head ~ atan(ε_perp / ℓ), so the
# ℓ >= 120px floor is the lever that bounds render-robustness.
LENGTH_RANGE = (120.0, 205.0)
STROKE_RANGE = (5.0, 9.0)
# Interior angle stays clear of the degenerate extremes (θ->0 collinear overlap,
# θ->180 straight line) that no oracle could resolve (design §2).
THETA_RANGE = (30.0, 150.0)

# SSOT §7.4 analogue — dense bins near the decision boundary (m = 0, i.e. θ = 90).
# Sign encodes acute (+) vs obtuse (-); magnitude (degrees) is the difficulty dial.
MARGIN_BINS: dict[str, tuple[float, float]] = {
    "far_acute": (10.0, 60.0),  # θ in [30, 80]
    "near_acute": (2.0, 10.0),  # θ in [80, 88]
    "right": (-2.0, 2.0),  # θ in [88, 92]
    "near_obtuse": (-10.0, -2.0),  # θ in [92, 100]
    "far_obtuse": (-60.0, -10.0),  # θ in [100, 150]
}

# SSOT §7.4 — fixed sweep for continuity analysis (degrees; all within |m| <= 60).
MARGIN_SWEEP: list[float] = [45, 30, 20, 10, 5, 2, 1, 0, -1, -2, -5, -10, -20, -30, -45]


def bin_for_m(m: float) -> str:
    """First bin (in MARGIN_BINS order) whose closed [lo, hi] range contains m."""
    for name, (lo, hi) in MARGIN_BINS.items():
        if lo <= m <= hi:
            return name
    raise ValueError(f"m={m} outside all margin bins {MARGIN_BINS}")


@dataclass(frozen=True)
class AngleScene:
    """Full latent visual state for one rendered example (design §1)."""

    vx: float
    vy: float
    heading: float  # α, degrees — direction of ray 1
    theta: float  # interior angle between the rays, degrees (in THETA_RANGE)
    len1: float
    len2: float
    stroke: float
    color_ink: str = "blue"
    width: int = 512
    height: int = 512
    background: str = "white"
    antialias: bool = True

    @property
    def signed_margin(self) -> float:
        """m = 90 - θ (degrees)."""
        return 90.0 - self.theta

    @property
    def ray1_end(self) -> tuple[float, float]:
        a = math.radians(self.heading)
        return (self.vx + self.len1 * math.cos(a), self.vy + self.len1 * math.sin(a))

    @property
    def ray2_end(self) -> tuple[float, float]:
        a = math.radians(self.heading + self.theta)
        return (self.vx + self.len2 * math.cos(a), self.vy + self.len2 * math.sin(a))

    def label(self, tangent_eps: float = 0.0) -> str:
        """Three-way label. Note SSOT §21.6: the right-angle boundary is ambiguous."""
        m = self.signed_margin
        if m > tangent_eps:
            return "acute"
        if m >= -tangent_eps:
            return "right"
        return "obtuse"

    @property
    def acute(self) -> bool:
        """Binary convention (design §1): acute = (θ < 90) = (m > 0), strict."""
        return self.signed_margin > 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def ray_rect(
    v: tuple[float, float], e: tuple[float, float], stroke: float
) -> list[tuple[float, float]]:
    """Four corners of a butt-capped stroke rectangle from vertex ``v`` to end ``e``.

    The rectangle spans exactly [v, e] along the ray with width ``stroke``, so the
    ink's extent along the ray equals the ray length (the property the oracle uses
    to recover length) and the near cap sits exactly at the vertex.
    """
    dx, dy = e[0] - v[0], e[1] - v[1]
    d = math.hypot(dx, dy)
    ux, uy = dx / d, dy / d
    nx, ny = -uy * stroke / 2.0, ux * stroke / 2.0
    return [
        (v[0] + nx, v[1] + ny),
        (e[0] + nx, e[1] + ny),
        (e[0] - nx, e[1] - ny),
        (v[0] - nx, v[1] - ny),
    ]


def _corners_fit(corners: list[tuple[float, float]], w: int, h: int, pad: float) -> bool:
    return all(pad <= x <= w - pad and pad <= y <= h - pad for x, y in corners)


def sample_scene(
    rng: random.Random,
    margin_bin: str | None = None,
    m: float | None = None,
    length_range: tuple[float, float] = LENGTH_RANGE,
    stroke_range: tuple[float, float] = STROKE_RANGE,
    pad: float = 6.0,
    canvas: tuple[int, int] = (512, 512),
    **style,
) -> AngleScene:
    """Sample a scene with a target angular margin m = 90 - θ (degrees).

    Either give an explicit target ``m`` (θ = 90 - m, honoured exactly) or a
    ``margin_bin`` name. Heading α, both ray lengths, stroke, and vertex position
    are sampled independently; the vertex is rejection-sampled until both butt-
    capped stroke rectangles fit the canvas. Nothing about the margin is imposed
    by pixels — θ is a scene parameter, m is pure arithmetic on it.
    """
    if m is None:
        if margin_bin is None:
            margin_bin = rng.choice(list(MARGIN_BINS))
        lo, hi = MARGIN_BINS[margin_bin]
        m = rng.uniform(lo, hi)
    theta = 90.0 - m
    if not (THETA_RANGE[0] - 1e-9 <= theta <= THETA_RANGE[1] + 1e-9):
        raise ValueError(f"θ={theta} (m={m}) outside {THETA_RANGE}")

    w, h = canvas
    for _ in range(20_000):
        heading = rng.uniform(0.0, 360.0)
        len1 = rng.uniform(*length_range)
        len2 = rng.uniform(*length_range)
        stroke = rng.uniform(*stroke_range)
        vx = rng.uniform(pad, w - pad)
        vy = rng.uniform(pad, h - pad)
        scene = AngleScene(
            vx=vx, vy=vy, heading=heading, theta=theta,
            len1=len1, len2=len2, stroke=stroke, width=w, height=h, **style,
        )  # fmt: skip
        v = (vx, vy)
        r1 = ray_rect(v, scene.ray1_end, stroke)
        r2 = ray_rect(v, scene.ray2_end, stroke)
        if _corners_fit(r1, w, h, pad) and _corners_fit(r2, w, h, pad):
            return scene
    raise RuntimeError(f"could not place an angle θ={theta:.2f} within canvas {canvas}")


def render(scene: AngleScene, supersample: int = 4) -> Image.Image:
    """Render I = R(z). Antialiasing is done by supersampling + LANCZOS downscale."""
    ss = supersample if scene.antialias else 1
    img = Image.new("RGB", (scene.width * ss, scene.height * ss), scene.background)
    draw = ImageDraw.Draw(img)
    v = (scene.vx, scene.vy)
    for end in (scene.ray1_end, scene.ray2_end):
        corners = ray_rect(v, end, scene.stroke)
        draw.polygon([(x * ss, y * ss) for x, y in corners], fill=scene.color_ink)
    if ss > 1:
        img = img.resize((scene.width, scene.height), Image.LANCZOS)
    return img


def rotate_ray2(scene: AngleScene, delta_theta: float) -> AngleScene:
    """Counterfactual action z' = T(z, a) with a = open/close the angle by delta_theta."""
    return dataclasses.replace(scene, theta=scene.theta + delta_theta)


def counterfactual_margin(scene: AngleScene, delta_theta: float) -> float:
    """m' = 90 - (θ + delta_theta) without rendering."""
    return rotate_ray2(scene, delta_theta).signed_margin


def latent_symmetries(scene: AngleScene, rng: random.Random):
    """Answerability gate 1 (SSOT §7.14): no nontrivial latent symmetry. The scene is
    a SINGLE angle figure (one vertex + two same-colour rays) parametrized by a heading
    α and the interior angle θ, not an interchangeable list of positional slots — there
    is nothing to permute. The only render-invariant relabeling (swapping the two rays)
    cannot be expressed in the (heading, interior-θ) parametrization without leaving the
    valid THETA_RANGE domain and changing signed_margin, so it is not a symmetry here.
    Every latent (vertex, heading, interior angle, both ray lengths, stroke) is directly
    reflected in the pixels, so the parametrization is minimal. Declared empty."""
    return iter(())


def scene_targets(scene: AngleScene, margin_bin: str) -> dict:
    """Task-specific ground truth for one example, stored in ExampleRecord.targets."""
    return {
        "m": scene.signed_margin,
        "theta": scene.theta,
        "margin_bin": margin_bin,
        "label": scene.label(),
        "acute": scene.acute,
    }
