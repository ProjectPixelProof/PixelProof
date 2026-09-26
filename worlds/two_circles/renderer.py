"""Two-circle collision world renderer (SSOT §7.3–§7.4).

Latent state z = {c1, c2, r1, r2, colors, canvas, style}. The image is I = R(z)
and the signed gap is

    m = ||c1 - c2||_2 - r1 - r2

with m > 0 separate, m == 0 touching, m < 0 overlapping.
"""

from __future__ import annotations

import dataclasses
import math
import random
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 0.2.0: per-scene noise seed (0.1.0 reused one fixed noise pattern for every image).
RENDERER_VERSION = "two_circles-0.2.0"

# SSOT §7.4 — dense bins near the decision boundary.
MARGIN_BINS: dict[str, tuple[float, float]] = {
    "far_separate": (20.0, 80.0),
    "near_separate": (1.0, 20.0),
    "tangent": (-1.0, 1.0),
    "near_overlap": (-20.0, -1.0),
    "deep_overlap": (-80.0, -20.0),
}

# SSOT §7.4 — fixed sweep for continuous analysis.
MARGIN_SWEEP: list[float] = [50, 40, 30, 20, 10, 5, 2, 1, 0, -1, -2, -5, -10, -20, -30, -40, -50]


def bin_for_m(m: float) -> str:
    """First bin (in MARGIN_BINS order) whose closed [lo, hi] range contains m.

    Shared endpoints (e.g. m=1 in both near_separate and tangent) resolve to the
    earlier bin deterministically; tests pin the edge cases.
    """
    for name, (lo, hi) in MARGIN_BINS.items():
        if lo <= m <= hi:
            return name
    raise ValueError(f"m={m} outside all margin bins {MARGIN_BINS}")


# Minimum distance the smaller circle must protrude past the larger one's rim.
# Below ~a few pixels the visible sliver is ambiguous at pixel scale and the
# oracle cannot certify the render (SSOT §7.6).
MIN_PROTRUSION_PX = 8.0


@dataclass(frozen=True)
class TwoCircleScene:
    """Full latent visual state for one rendered example (SSOT §7.3)."""

    c1: tuple[float, float]
    c2: tuple[float, float]
    r1: float
    r2: float
    color1: str = "blue"
    color2: str = "red"
    width: int = 512
    height: int = 512
    background: str = "white"
    fill: str = "solid"  # "solid" | "outline"
    stroke_width: int = 4
    antialias: bool = True
    noise: str = "none"  # "none" | "low" | "moderate"
    overlay_text: str | None = None  # adversarial text overlay (SSOT §7.5)

    @property
    def signed_gap(self) -> float:
        """m = ||c1 - c2|| - r1 - r2."""
        return math.dist(self.c1, self.c2) - self.r1 - self.r2

    def label(self, tangent_eps: float = 0.0) -> str:
        """Three-way label. Note SSOT §21.6: the boundary is ambiguous at pixel scale."""
        m = self.signed_gap
        if m < -tangent_eps:
            return "overlapping"
        if m <= tangent_eps:
            return "touching"
        return "separate"

    @property
    def overlap_or_touch(self) -> bool:
        """Binary version (SSOT §7.3): overlap_or_touch = (m <= 0)."""
        return self.signed_gap <= 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def sample_scene(
    rng: random.Random,
    margin_bin: str | None = None,
    m: float | None = None,
    radius_range: tuple[float, float] = (30.0, 90.0),
    pad: float = 6.0,
    canvas: tuple[int, int] = (512, 512),
    **style,
) -> TwoCircleScene:
    """Sample a scene with a target signed gap.

    Either give an explicit target ``m`` or a ``margin_bin`` name from MARGIN_BINS.
    Rejection-samples radii/placement so both circles fit fully on canvas and the
    center distance stays >= |r1 - r2| + MIN_PROTRUSION_PX: near-containment
    scenes where circle 1 barely peeks past circle 2 are ambiguous stimuli (the
    visible sliver approaches sub-pixel) and unverifiable by the pixel oracle
    (SSOT §7.6), so they are excluded like full containment is.
    """
    if m is None:
        if margin_bin is None:
            margin_bin = rng.choice(list(MARGIN_BINS))
        lo, hi = MARGIN_BINS[margin_bin]
        m = rng.uniform(lo, hi)

    w, h = canvas
    for _ in range(10_000):
        r1 = rng.uniform(*radius_range)
        r2 = rng.uniform(*radius_range)
        d = r1 + r2 + m
        if d < abs(r1 - r2) + MIN_PROTRUSION_PX:
            continue  # (near-)containment; resample radii
        x1 = rng.uniform(r1 + pad, w - r1 - pad)
        y1 = rng.uniform(r1 + pad, h - r1 - pad)
        theta = rng.uniform(0, 2 * math.pi)
        x2 = x1 + d * math.cos(theta)
        y2 = y1 + d * math.sin(theta)
        if r2 + pad <= x2 <= w - r2 - pad and r2 + pad <= y2 <= h - r2 - pad:
            return TwoCircleScene(
                c1=(x1, y1), c2=(x2, y2), r1=r1, r2=r2, width=w, height=h, **style
            )
    raise RuntimeError(f"could not place circles for m={m:.2f} within canvas {canvas}")


def render(scene: TwoCircleScene, supersample: int = 4) -> Image.Image:
    """Render I = R(z). Antialiasing is done by supersampling + LANCZOS downscale."""
    ss = supersample if scene.antialias else 1
    img = Image.new("RGB", (scene.width * ss, scene.height * ss), scene.background)
    draw = ImageDraw.Draw(img)
    circles = ((scene.c1, scene.r1, scene.color1), (scene.c2, scene.r2, scene.color2))
    for (cx, cy), r, color in circles:
        box = [(cx - r) * ss, (cy - r) * ss, (cx + r) * ss, (cy + r) * ss]
        if scene.fill == "solid":
            draw.ellipse(box, fill=color)
        else:
            draw.ellipse(box, outline=color, width=max(1, scene.stroke_width * ss))
    if ss > 1:
        img = img.resize((scene.width, scene.height), Image.LANCZOS)

    if scene.noise != "none":
        sigma = {"low": 3.0, "moderate": 8.0}[scene.noise]
        # Seed from the scene content: deterministic per scene, but not the same
        # pattern for every image (a constant pattern would itself be learnable).
        key = repr((scene.c1, scene.c2, scene.r1, scene.r2, scene.noise))
        rng = np.random.default_rng(zlib.crc32(key.encode()))
        arr = np.asarray(img, dtype=np.float32)
        arr += rng.normal(0.0, sigma, arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    if scene.overlay_text:
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default(size=28)
        draw.text((scene.width / 2, 24), scene.overlay_text, fill="black", anchor="mm", font=font)
    return img


def latent_symmetries(scene: TwoCircleScene, rng: random.Random):
    """Answerability gate 1 (SSOT §7.14): no nontrivial latent symmetry. The two
    circles occupy positional slots (``c1``/``c2``), but swapping the slots swaps
    the draw order, and where the disks OVERLAP the z-order is visible (one disk
    occludes the other), so the swap changes the pixels — it is not a render
    symmetry. (For separate disks it would be, but the overlap bins are the whole
    point of the world.) Every other latent (position, radius, colour) is directly
    reflected in the picture, so the parametrization is minimal. Declared empty."""
    return iter(())


def translate_circle_1(scene: TwoCircleScene, dx: float, dy: float) -> TwoCircleScene:
    """Counterfactual action z' = T(z, a) with a = translate(circle_1, [dx, dy])."""
    return dataclasses.replace(scene, c1=(scene.c1[0] + dx, scene.c1[1] + dy))


def counterfactual_margin(scene: TwoCircleScene, dx: float, dy: float) -> float:
    """m' = ||(c1 + delta) - c2|| - r1 - r2 without rendering."""
    return translate_circle_1(scene, dx, dy).signed_gap


def scene_targets(scene: TwoCircleScene, margin_bin: str) -> dict:
    """Task-specific ground truth for one example, stored in ExampleRecord.targets.

    The foundry manifest is task-agnostic; this is where the two-circle world
    declares what a downstream probe / eval / metric can join against.
    """
    return {
        "m": scene.signed_gap,
        "margin_bin": margin_bin,
        "label": scene.label(),
        "overlap_or_touch": scene.overlap_or_touch,
    }
