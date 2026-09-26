"""Radial-density-regime world: global radial texture-energy structure.

A field of dark dots on a white canvas. In one regime the dots are packed
toward the centre (a solid core); in the other they are packed toward the
outer rim (a surrounding ring). The decision is the emergent global Gestalt
of the whole dot field's radial density distribution.

The renderer owns scene sampling + rasterization + the structural analytic
gold. The independent pixel-only inverse arm (``oracle.py``) recovers the
same decision from the final PNG alone.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

WIDTH = 512
HEIGHT = 512
CX = WIDTH // 2
CY = HEIGHT // 2
BACKGROUND = (255, 255, 255)
DOT_COLOR = (35, 45, 90)  # dark navy ink
DOT_RADIUS = 5
N = 60
INNER_HI = 70   # core dots live inside r <= INNER_HI
OUTER_LO = 150  # rim dots live outside r >= OUTER_LO
BOUNDARY = 128  # radial boundary used by both the analytic gold and the oracle
CORE_FRACTION = 0.8  # probability a dot goes to the dominant region of its mode


def _sample_dots(rng, count: int, lo: int, hi: int) -> list[list[float]]:
    """Sample ``count`` dots uniformly inside the annulus lo <= r <= hi."""
    pts: list[list[float]] = []
    while len(pts) < count:
        x = rng.uniform(20, WIDTH - 20)
        y = rng.uniform(20, HEIGHT - 20)
        r = np.hypot(x - CX, y - CY)
        if lo <= r <= hi:
            pts.append([float(x), float(y)])
    return pts


def sample_scene(seed: int) -> dict:
    """Sample one latent scene deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    mode = "core" if int(rng.integers(0, 2)) == 0 else "rim"
    if mode == "core":
        n_inner = int(round(CORE_FRACTION * N))
        n_outer = N - n_inner
    else:
        n_outer = int(round(CORE_FRACTION * N))
        n_inner = N - n_outer
    inner = _sample_dots(rng, n_inner, 18, INNER_HI)
    outer = _sample_dots(rng, n_outer, OUTER_LO, 205)
    dots = inner + outer
    rng.shuffle(dots)
    order = list(range(len(dots)))
    rng.shuffle(order)
    return {"dots": dots, "dot_order": order}


def render(scene: dict) -> Image.Image:
    """Rasterize the latent scene (a deterministic anti-aliased-free PNG)."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(img)
    dots = scene["dots"]
    order = scene.get("dot_order", list(range(len(dots))))
    for idx in order:
        x, y = dots[idx]
        r = DOT_RADIUS
        d.ellipse([x - r, y - r, x + r, y + r], fill=DOT_COLOR)
    return img


def _inner_outer(scene: dict) -> tuple[int, int]:
    inner = 0
    outer = 0
    for (x, y) in scene["dots"]:
        if np.hypot(x - CX, y - CY) <= BOUNDARY:
            inner += 1
        else:
            outer += 1
    return inner, outer


def analytic_gold(scene: dict) -> str:
    """Structural gold: are the dots packed toward the centre (core) or rim?"""
    inner, outer = _inner_outer(scene)
    return "core" if inner >= outer else "rim"


def margin(scene: dict) -> float:
    """Commitment magnitude: the inner-vs-outer dot-count difference."""
    inner, outer = _inner_outer(scene)
    return float(abs(inner - outer))


def is_quarantined(scene: dict) -> bool:
    """Quarantine ambiguous cases where the inner/outer counts nearly tie."""
    return margin(scene) < 14.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """The dots do not overlap, so paint order is photometrically invisible.

    Rotating the declared ``dot_order`` produces a byte-identical raster while
    ``analytic_gold`` (which reads only the dot positions) is unchanged.
    """
    dots = scene["dots"]
    order = list(scene.get("dot_order", list(range(len(dots)))))
    twin_order = [order[-1]] + order[:-1]
    return [("rotated_dot_order", {"dots": dots, "dot_order": twin_order})]
