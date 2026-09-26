"""Deterministic renderer and analytic arm for three-fragment contour completion."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

CANVAS_SIZE = 768
CELL = CANVAS_SIZE // 8
BASE_RADIUS = 120.0
PALETTE = (
    (31, 102, 176),
    (177, 62, 91),
    (23, 137, 112),
    (117, 73, 175),
    (191, 96, 28),
)
BACKGROUNDS = ((250, 249, 246), (247, 250, 252), (251, 248, 252))
DECORATION = (211, 213, 212)


def _base_geometry(phase: int, row: int, col: int) -> tuple[tuple[float, float], list[tuple[float, float]]]:
    """Return a circle center and three lattice-centered tangent points.

    Each orientation occupies three nonadjacent cells inside a 3x3 block.  The
    points lie on one radius-120 circle; changing radius scales them about that
    center while retaining the same topology.
    """

    x0 = col * CELL + CELL / 2.0
    y0 = row * CELL + CELL / 2.0
    if phase == 0:  # base above, apex below
        points = [(x0, y0), (x0 + 2 * CELL, y0), (x0 + CELL, y0 + 2 * CELL)]
        center = (x0 + CELL, y0 + 0.75 * CELL)
    elif phase == 1:  # base below, apex above
        points = [(x0, y0 + 2 * CELL), (x0 + 2 * CELL, y0 + 2 * CELL), (x0 + CELL, y0)]
        center = (x0 + CELL, y0 + 1.25 * CELL)
    elif phase == 2:  # base left, apex right
        points = [(x0, y0), (x0, y0 + 2 * CELL), (x0 + 2 * CELL, y0 + CELL)]
        center = (x0 + 0.75 * CELL, y0 + CELL)
    else:  # base right, apex left
        points = [(x0 + 2 * CELL, y0), (x0 + 2 * CELL, y0 + 2 * CELL), (x0, y0 + CELL)]
        center = (x0 + 1.25 * CELL, y0 + CELL)
    return center, points


def sample_scene(seed: int) -> dict:
    """Sample one balanced scene; even seeds are coherent and odd seeds are not."""

    rng = random.Random(int(seed))
    phase = rng.randrange(4)
    row = rng.randrange(1, 5)
    col = rng.randrange(1, 5)
    center, base_points = _base_geometry(phase, row, col)
    radius = rng.choice((114.0, 120.0, 126.0))
    scale = radius / BASE_RADIUS
    # Store the translated center; render recomputes the tangent points from the
    # public phase and lattice anchor encoded by this center.
    modes = [0, 0, 0]
    if int(seed) % 2:
        modes[rng.randrange(3)] = 1
    order = [0, 1, 2]
    rng.shuffle(order)
    return {
        "canvas_size": CANVAS_SIZE,
        "center_x": round(center[0], 6),
        "center_y": round(center[1], 6),
        "radius": radius,
        "arc_span": rng.choice((29.0, 31.0, 33.0)),
        "phase": phase,
        "stroke_width": rng.choice((5, 6, 7)),
        "ink_rgb": list(rng.choice(PALETTE)),
        "background_rgb": list(rng.choice(BACKGROUNDS)),
        "fragment_order": order,
        "fragment_modes": modes,
    }


def _canonical_offsets(phase: int) -> list[tuple[float, float]]:
    if phase == 0:
        return [(-96.0, -72.0), (96.0, -72.0), (0.0, 120.0)]
    if phase == 1:
        return [(-96.0, 72.0), (96.0, 72.0), (0.0, -120.0)]
    if phase == 2:
        return [(-72.0, -96.0), (-72.0, 96.0), (120.0, 0.0)]
    return [(72.0, -96.0), (72.0, 96.0), (-120.0, 0.0)]


def _fragment_points(scene: dict, index: int) -> list[tuple[float, float]]:
    cx = float(scene["center_x"])
    cy = float(scene["center_y"])
    radius = float(scene["radius"])
    offset = _canonical_offsets(int(scene["phase"]))[index]
    unit = (offset[0] / BASE_RADIUS, offset[1] / BASE_RADIUS)
    midpoint = (cx + radius * unit[0], cy + radius * unit[1])
    radial_angle = math.atan2(unit[1], unit[0])
    local_center = (cx, cy)
    if int(scene["fragment_modes"][index]) == 1:
        # A half-turn about the fragment midpoint preserves its size and ink but
        # reverses the visible bend.  Its continuation circle is now opposite.
        local_center = (2.0 * midpoint[0] - cx, 2.0 * midpoint[1] - cy)
        radial_angle += math.pi
    span = math.radians(float(scene["arc_span"]))
    points: list[tuple[float, float]] = []
    for step in range(65):
        angle = radial_angle - span / 2.0 + span * step / 64.0
        points.append(
            (
                local_center[0] + radius * math.cos(angle),
                local_center[1] + radius * math.sin(angle),
            )
        )
    return points


def render(scene: dict) -> Image.Image:
    size = int(scene["canvas_size"])
    background = tuple(int(value) for value in scene["background_rgb"])
    image = Image.new("RGB", (size, size), background)
    draw = ImageDraw.Draw(image)
    # Pale remote marks make absolute location and empty-canvas heuristics inert;
    # their zero chroma keeps them outside the contour oracle's segmentation.
    phase = int(scene["phase"])
    for index in range(6):
        x = 34 + ((index * 173 + phase * 41) % (size - 68))
        y = 38 + ((index * 227 + phase * 67) % (size - 76))
        draw.line((x - 8, y, x + 8, y), fill=DECORATION, width=2)
        draw.line((x, y - 8, x, y + 8), fill=DECORATION, width=2)
    ink = tuple(int(value) for value in scene["ink_rgb"])
    width = int(scene["stroke_width"])
    for index in scene["fragment_order"]:
        draw.line(_fragment_points(scene, int(index)), fill=ink, width=width, joint="curve")
    return image


def analytic_gold(scene: dict) -> str:
    return "yes" if all(int(value) == 0 for value in scene["fragment_modes"]) else "no"


def margin(scene: dict) -> float:
    """Runner-up gap: full reversal is two continuation radii from agreement."""

    return 2.0 if any(int(value) for value in scene["fragment_modes"]) else 1.0


def is_quarantined(scene: dict) -> bool:
    modes = [int(value) for value in scene["fragment_modes"]]
    return len(modes) != 3 or sum(modes) not in {0, 1}


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    twin = dict(scene)
    twin["fragment_order"] = list(reversed([int(value) for value in scene["fragment_order"]]))
    return [("reverse_disjoint_fragment_draw_order", twin)]
