"""Deterministic renderer and analytic gold for the compact spur world.

The analytic state is deliberately expressed as route control points.  The
pixel arm in oracle.py independently reconstructs the route from the PNG.
"""

from __future__ import annotations

import math
import random
from PIL import Image, ImageDraw

W = H = 512
BG = (250, 250, 248)
INK = (38, 43, 51)
GREEN = (38, 166, 91)
RED = (204, 72, 72)
AMBER = (226, 151, 35)
VIOLET = (113, 83, 181)
GRAY = (198, 204, 211)
WHITE = (250, 250, 248)

PROMPT_FAMILIES = ("pf1_spur_ownership", "pf2_anonymous_branch", "pf3_route_topology")


def _rotate(point: tuple[float, float], angle: float) -> tuple[float, float]:
    x, y = point
    c, s = math.cos(angle), math.sin(angle)
    return (x * c - y * s, x * s + y * c)


def _transform(point: tuple[float, float], scene: dict) -> tuple[float, float]:
    x, y = _rotate(point, scene["rotation"])
    if scene["mirror"]:
        x = -x
    return (x + scene["position"][0], y + scene["position"][1])


def _polyline(base: list[tuple[float, float]], scene: dict) -> list[tuple[float, float]]:
    return [_transform(p, scene) for p in base]


def _scene_geometry(scene: dict) -> dict:
    # All decisive coordinates live in one compact local tile.  The main route
    # starts at green, forks at the central hub, and ends at amber/violet.
    trunk = [(-42, 18), (-25, 9), (0, 0)]
    amber_arm = [(0, 0), (24, -8), (42, -14), (48, -16)]
    violet_arm = [(0, 0), (24, 8), (42, 14), (48, 16)]
    spur_arm = [(24, -8), (31, -35), (34, -72)]
    if scene["spur_terminal"] == "amber":
        spur = spur_arm
    else:
        spur = [(24, 8), (31, 35), (34, 72)]
    return {
        "trunk": _polyline(trunk, scene),
        "amber_arm": _polyline(amber_arm, scene),
        "violet_arm": _polyline(violet_arm, scene),
        "spur": _polyline(spur, scene),
    }


def sample_scene(seed: int) -> dict:
    rng = random.Random(int(seed))
    # The local tile is translated and rotated, while arm identity, terminal
    # colors, and spur identity are counterbalanced independently.
    angle = rng.choice([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    mirror = bool(rng.randrange(2))
    position = (rng.uniform(230, 282), rng.uniform(230, 282))
    spur_terminal = "amber" if rng.randrange(2) == 0 else "violet"
    # A commutative draw-order field is exposed for the alias check.  It has no
    # effect on the raster because ornaments are drawn in sorted order.
    decor_order = ["square", "circle"]
    if rng.randrange(2):
        decor_order.reverse()
    return {
        "spur_terminal": spur_terminal,
        "position": [round(position[0], 4), round(position[1], 4)],
        "rotation": angle,
        "mirror": mirror,
        "style": "ivory_darkroute_v1",
        "decor_order": decor_order,
    }


def _draw_polyline(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], width: int) -> None:
    draw.line(points, fill=INK, width=width, joint="curve")


def _draw_marker(draw: ImageDraw.ImageDraw, center: tuple[float, float], color: tuple[int, int, int]) -> None:
    x, y = center
    r = 10
    draw.ellipse((x - r - 3, y - r - 3, x + r + 3, y + r + 3), fill=WHITE, outline=WHITE, width=2)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=WHITE, width=3)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=WHITE)


def render(scene: dict) -> Image.Image:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    geo = _scene_geometry(scene)
    for key in ("trunk", "amber_arm", "violet_arm", "spur"):
        _draw_polyline(draw, geo[key], 16)
    # Endpoint markers are painted after the route, preserving visible route
    # attachment through a short dark neck and a white marker halo.
    green = geo["trunk"][0]
    amber = geo["amber_arm"][-1]
    violet = geo["violet_arm"][-1]
    _draw_marker(draw, green, GREEN)
    _draw_marker(draw, amber, AMBER)
    _draw_marker(draw, violet, VIOLET)

    # Remote ornaments are intentionally identical, small, and nondiagnostic.
    # Their locations are fixed relative to the page and have no scene access.
    for kind in sorted(scene["decor_order"]):
        if kind == "circle":
            draw.ellipse((30, 30, 50, 50), outline=GRAY, width=3)
        else:
            draw.rectangle((455, 30, 477, 52), outline=GRAY, width=3)
    return image


def analytic_gold(scene: dict) -> str:
    return "yes" if scene["spur_terminal"] == "amber" else "no"


def margin(scene: dict) -> float:
    # Minimum visible branch-separation clearance in pixels: intentionally large.
    return 22.0


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < 12.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    twin = dict(scene)
    twin["decor_order"] = list(reversed(scene["decor_order"]))
    return [("commutative_ornament_order", twin)]
