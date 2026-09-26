"""Renderer for the paired-seam correspondence question world."""

from __future__ import annotations

import math
import random
from typing import Any

from PIL import Image, ImageDraw


CANVAS = (512, 512)
BASE_TILE_WIDTH = 80.0
BASE_TILE_HEIGHT = 156.0
SUPERSAMPLE = 4
MATCH_THRESHOLD = 4.0
PALETTE = ("#176b78", "#a94832", "#66509b", "#3d7f4c", "#a16c19")


def _shift_profile(profile: list[float], shift: int) -> list[float]:
    """Cyclically reorder the internal knots while preserving their multiset."""
    amount = int(shift) % len(profile)
    if amount == 0:
        return list(profile)
    return list(profile[amount:]) + list(profile[:amount])


def _right_profile(scene: dict[str, Any]) -> list[float]:
    profile = [float(value) for value in scene["profile"]]
    if bool(scene["match"]):
        return profile
    return _shift_profile(profile, int(scene["mismatch_shift"]))


def _profile_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return math.inf
    return sum(abs(a - b) for a, b in zip(left, right)) / len(left)


def sample_scene(seed: int) -> dict[str, Any]:
    """Sample one balanced, deterministic two-tile scene."""
    rng = random.Random(int(seed))
    match = int(seed) % 2 == 0

    # The same value multiset is used in both classes.  Only its ordered
    # sequence changes in the negative class, preventing area and roughness
    # from identifying the answer.
    levels = [-18.0, 15.0, -9.0, 19.0, -14.0, 8.0, -12.0, 16.0, 5.0]
    while True:
        rng.shuffle(levels)
        profile = [round(value * rng.uniform(0.92, 1.08), 3) for value in levels]
        mismatch_shift = rng.choice((2, 3, 4))
        if match or _profile_distance(profile, _shift_profile(profile, mismatch_shift)) >= 9.0:
            break

    return {
        "match": match,
        "profile": profile,
        "mismatch_shift": int(mismatch_shift),
        # Keep the two components in distinct ablation lobes: the left tile
        # stays in the x=1..2 grid neighborhood and the right tile in x=5..6.
        # The group still occupies one central regional band, not the canvas.
        "x": round(rng.uniform(80.0, 84.0), 3),
        "y": round(rng.uniform(174.0, 218.0), 3),
        "gap": round(rng.uniform(181.0, 183.0), 3),
        "scale": round(rng.uniform(0.97, 1.03), 4),
        "color": rng.choice(PALETTE),
    }


def _edge_samples(profile: list[float]) -> list[tuple[float, float]]:
    values = [0.0, *[float(value) for value in profile], 0.0]
    last = len(values) - 1
    return [(index / last, value) for index, value in enumerate(values)]


def _tile_polygons(
    x: float,
    y: float,
    width: float,
    height: float,
    profile: list[float],
    scale: float,
    left_tile: bool,
    supersample: int,
) -> list[tuple[int, int]]:
    points: list[tuple[float, float]] = []
    if left_tile:
        points.extend(((x, y), (x + width, y)))
        for fraction, displacement in _edge_samples(profile):
            points.append((x + width + displacement * scale, y + fraction * height))
        points.extend(((x + width, y + height), (x, y + height)))
    else:
        points.extend(((x, y), (x + width, y), (x + width, y + height), (x, y + height)))
        for fraction, displacement in reversed(_edge_samples(profile)):
            points.append((x + displacement * scale, y + fraction * height))
    return [(int(round(px * supersample)), int(round(py * supersample))) for px, py in points]


def render(scene: dict[str, Any]) -> Image.Image:
    """Render two disconnected filled tiles with a visible facing seam pair."""
    width, height = CANVAS
    ss = SUPERSAMPLE
    image = Image.new("RGB", (width * ss, height * ss), "white")
    draw = ImageDraw.Draw(image)
    scale = float(scene["scale"])
    tile_width = BASE_TILE_WIDTH * scale
    tile_height = BASE_TILE_HEIGHT * scale
    x_left = float(scene["x"])
    y = float(scene["y"])
    x_right = x_left + tile_width + float(scene["gap"])
    left_profile = [float(value) for value in scene["profile"]]
    right_profile = _right_profile(scene)
    color = str(scene["color"])

    left_points = _tile_polygons(
        x_left,
        y,
        tile_width,
        tile_height,
        left_profile,
        scale,
        True,
        ss,
    )
    right_points = _tile_polygons(
        x_right,
        y,
        tile_width,
        tile_height,
        right_profile,
        scale,
        False,
        ss,
    )
    draw.polygon(left_points, fill=color)
    draw.polygon(right_points, fill=color)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def analytic_gold(scene: dict[str, Any]) -> str:
    """The declared relation is the only analytic label input."""
    return "yes" if bool(scene["match"]) else "no"


def margin(scene: dict[str, Any]) -> float:
    """Signed separation from the visible profile-comparison boundary."""
    left = [float(value) for value in scene["profile"]]
    right = _right_profile(scene)
    distance = _profile_distance(left, right)
    signed = MATCH_THRESHOLD - distance if bool(scene["match"]) else distance - MATCH_THRESHOLD
    return round(float(signed), 6)


def is_quarantined(scene: dict[str, Any]) -> bool:
    """Exclude only profiles too close to the pixel comparison boundary."""
    left = [float(value) for value in scene["profile"]]
    distance = _profile_distance(left, _right_profile(scene))
    return 2.0 <= distance <= 6.0


def latent_symmetries(scene: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """No distinct latent state is pixel-identical under this parametrization."""
    return []
