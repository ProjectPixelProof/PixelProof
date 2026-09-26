"""Renderer and analytic answer for the reflected-silhouette search world.

The answer is not stored as a target label.  It is the sector whose two visible
filled polygons use zero shear in the reflection construction; the renderer
draws that construction and the pixel oracle independently rediscovers it.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any

from PIL import Image, ImageDraw


RENDERER_VERSION = "reflected-silhouette-search-0.1.0"
WIDTH = 512
HEIGHT = 512
SECTORS = ("NW", "NE", "SW", "SE")
INK_PALETTE = ((25, 78, 142), (122, 38, 108), (26, 124, 94), (128, 67, 28))
BACKGROUND = (248, 247, 242)
GUIDE = (205, 211, 216)
SUPERSAMPLE = 4

# Each polygon is intentionally asymmetric and mildly concave.  A target pair
# is a reflected copy; a distractor is a determinant-one shear of that copy.
# The area-preserving shear blocks a simple equal-ink shortcut.
TEMPLATES: tuple[tuple[tuple[float, float], ...], ...] = (
    (
        (-31.0, -27.0),
        (18.0, -27.0),
        (31.0, -12.0),
        (12.0, -4.0),
        (24.0, 11.0),
        (8.0, 29.0),
        (-31.0, 29.0),
    ),
    (
        (-29.0, -30.0),
        (19.0, -30.0),
        (30.0, -14.0),
        (7.0, -2.0),
        (25.0, 9.0),
        (17.0, 28.0),
        (-29.0, 28.0),
    ),
    (
        (-30.0, -26.0),
        (13.0, -30.0),
        (30.0, -16.0),
        (9.0, -1.0),
        (26.0, 14.0),
        (4.0, 30.0),
        (-30.0, 25.0),
    ),
)


def _point_for_local(
    anchor: tuple[float, float], u: tuple[float, float], v: tuple[float, float], a: float, b: float
) -> tuple[float, float]:
    return (anchor[0] + a * u[0] + b * v[0], anchor[1] + a * u[1] + b * v[1])


def _panel_polygon(panel: dict[str, Any], side: int) -> list[tuple[float, float]]:
    angle = math.radians(float(panel["angle"]))
    u = (math.cos(angle), math.sin(angle))
    v = (-math.sin(angle), math.cos(angle))
    midpoint = (float(panel["cx"]), float(panel["cy"]))
    half = float(panel["distance"]) / 2.0
    anchor = (
        midpoint[0] + (-1.0 if side == 0 else 1.0) * half * u[0],
        midpoint[1] + (-1.0 if side == 0 else 1.0) * half * u[1],
    )
    template = TEMPLATES[int(panel["template"])]
    shear = float(panel["shear"])
    points = []
    for a, b in template:
        if side == 0:
            local_a, local_b = a, b
        else:
            # Reflection across the perpendicular-bisector axis negates u.
            # A shear parallel to u changes only distractors and has det 1.
            local_a, local_b = -a + shear * b, b
        points.append(_point_for_local(anchor, u, v, local_a, local_b))
    return points


def sample_scene(seed: int) -> dict:
    """Sample one balanced four-sector scene from an integer seed."""

    rng = random.Random(int(seed))
    target_index = rng.randrange(4)
    centers = {
        "NW": (128.0, 128.0),
        "NE": (384.0, 128.0),
        "SW": (128.0, 384.0),
        "SE": (384.0, 384.0),
    }
    panels: list[dict[str, Any]] = []
    for index, sector in enumerate(SECTORS):
        base_x, base_y = centers[sector]
        shear = 0.0 if index == target_index else rng.choice((-1.0, 1.0)) * rng.uniform(0.48, 0.78)
        panels.append(
            {
                "sector": sector,
                "cx": round(base_x + rng.uniform(-5.0, 5.0), 4),
                "cy": round(base_y + rng.uniform(-5.0, 5.0), 4),
                "angle": round(rng.uniform(38.0, 52.0), 4),
                "distance": round(rng.uniform(198.0, 208.0), 4),
                "template": rng.randrange(len(TEMPLATES)),
                "shear": round(shear, 4),
            }
        )
    return {
        "panels": panels,
        "width": WIDTH,
        "height": HEIGHT,
        "background": list(BACKGROUND),
        "ink": list(rng.choice(INK_PALETTE)),
        "guide": list(GUIDE),
    }


def render(scene: dict) -> Image.Image:
    """Render only the scene's visible panels, guides, and ink polygons."""

    width, height = int(scene["width"]), int(scene["height"])
    ss = SUPERSAMPLE
    image = Image.new("RGB", (width * ss, height * ss), tuple(scene["background"]))
    draw = ImageDraw.Draw(image)
    guide = tuple(scene["guide"])
    # Faint panel frames establish the public sector vocabulary but are not
    # saturated and therefore cannot enter the pixel inverse's ink mask.
    for x0, y0, x1, y1 in ((10, 10, 246, 246), (266, 10, 502, 246), (10, 266, 246, 502), (266, 266, 502, 502)):
        draw.rounded_rectangle(
            (x0 * ss, y0 * ss, x1 * ss, y1 * ss),
            radius=13 * ss,
            outline=guide,
            width=2 * ss,
        )
    ink = tuple(scene["ink"])
    for panel in scene["panels"]:
        for side in (0, 1):
            polygon = [(round(x * ss), round(y * ss)) for x, y in _panel_polygon(panel, side)]
            draw.polygon(polygon, fill=ink)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def analytic_gold(scene: dict) -> str:
    """Select the unique panel with the smallest visible construction shear."""

    ranked = sorted((abs(float(panel["shear"])), str(panel["sector"])) for panel in scene["panels"])
    if len(ranked) < 2 or ranked[0][0] != 0.0 or ranked[1][0] <= 0.34:
        raise ValueError("scene does not have a unique reflected-pair winner")
    return ranked[0][1]


def margin(scene: dict) -> float:
    ranked = sorted(abs(float(panel["shear"])) for panel in scene["panels"])
    if len(ranked) < 2:
        return 0.0
    return float(ranked[1] - ranked[0])


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < 0.34


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    twin = copy.deepcopy(scene)
    twin["panels"] = list(reversed(twin["panels"]))
    return [("reverse_unordered_panel_records", twin)]
