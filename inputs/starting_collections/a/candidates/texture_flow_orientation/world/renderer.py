"""Texture-flow orientation world.

A field of short dark strokes is scattered across the canvas. Most strokes are
drawn aligned to one shared global orientation (a gentle vector flow); the
remainder are decoys pointing in unrelated directions. The question asks in
which of the four directions the whole texture field predominantly aligns:
horizontal, rising (uphill to the right), vertical, or falling (downhill to
the right). The answer is an emergent aggregate of the entire stroke field --
the dominant orientation bucket of the whole set -- not of any single stroke,
its ink, its length, or its absolute position.
"""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

CANVAS = 256
BACKGROUND = (255, 255, 255)
STROKE_COLOR = (30, 35, 90)
STROKE_WIDTH = 3

CLASSES = ("horizontal", "rising", "vertical", "falling")

_BUCKET_CENTER = {
    "horizontal": 0.0,
    "rising": 45.0,
    "vertical": 90.0,
    "falling": 135.0,
}

_DECORATOR = ("decorator", "free")

QUARANTINE_MARGIN = 3.0


def _category(angle_deg: float) -> str:
    """Classify an orientation angle in [0,180) into one of the four buckets."""
    a = angle_deg % 180.0
    best = None
    best_dist = 1e18
    for cls, center in _BUCKET_CENTER.items():
        d = abs((a - center + 180.0) % 180.0)
        if d > 90.0:
            d = 180.0 - d
        if d < best_dist:
            best_dist = d
            best = cls
    return best


def _dominant(strokes: list[dict]) -> str:
    counts = {cls: 0 for cls in CLASSES}
    for stroke in strokes:
        counts[_category(stroke["angle_deg"])] += 1
    return max(CLASSES, key=lambda cls: counts[cls])


def _draw_endpoints(stroke: dict) -> tuple[float, float, float, float]:
    cx, cy, ang, length = stroke["x"], stroke["y"], stroke["angle_deg"], stroke["length"]
    a = math.radians(ang)
    dx = math.cos(a) * length / 2.0
    dy = math.sin(a) * length / 2.0
    return cx - dx, cy - dy, cx + dx, cy + dy


def _separated(pending, strokes) -> bool:
    px, py, pl = pending["x"], pending["y"], pending["length"]
    for other in strokes:
        ox, oy, ol = other["x"], other["y"], other["length"]
        dx = px - ox
        dy = py - oy
        need = (pl + ol) / 2.0 + 7.0
        if dx * dx + dy * dy < need * need:
            return False
    return True


def sample_scene(seed: int) -> dict:
    rng = random.Random(seed)
    dominant = rng.choice(CLASSES)
    total = rng.randint(16, 22)
    nt = rng.uniform(0.72, 0.85)
    dominant_count = max(6, int(round(total * nt)))
    n_deco = total - dominant_count

    strokes: list[dict] = []
    attempts = 0
    placed = 0
    max_attempts = total * 200 + 500
    angle_jitter = rng.uniform(3.0, 6.0)

    while placed < total and attempts < max_attempts:
        attempts += 1
        x = rng.uniform(22.0, CANVAS - 22.0)
        y = rng.uniform(22.0, CANVAS - 22.0)
        length = rng.uniform(12.0, 18.0)
        if placed < dominant_count:
            center = _BUCKET_CENTER[dominant]
            ang = center + rng.uniform(-angle_jitter, angle_jitter)
        else:
            ang = rng.uniform(0.0, 180.0)
        stroke = {"x": x, "y": y, "angle_deg": ang % 180.0, "length": length}
        if not _separated(stroke, strokes):
            continue
        strokes.append(stroke)
        placed += 1

    draw_order = list(range(len(strokes)))
    rng.shuffle(draw_order)

    return {
        "strokes": strokes,
        "draw_order": draw_order,
        "dominant_count": dominant_count,
        "total": len(strokes),
    }


def render(scene: dict) -> Image.Image:
    image = Image.new("RGB", (CANVAS, CANVAS), BACKGROUND)
    draw = ImageDraw.Draw(image)
    for index in scene["draw_order"]:
        stroke = scene["strokes"][index]
        x0, y0, x1, y1 = _draw_endpoints(stroke)
        draw.line((x0, y0, x1, y1), fill=STROKE_COLOR, width=STROKE_WIDTH)
    return image


def analytic_gold(scene: dict) -> str:
    return _dominant(scene["strokes"])


def margin(scene: dict) -> float:
    counts = {cls: 0 for cls in CLASSES}
    for stroke in scene["strokes"]:
        counts[_category(stroke["angle_deg"])] += 1
    ordered = sorted(counts.values(), reverse=True)
    return float(ordered[0] - ordered[1])


def is_quarantined(scene: dict) -> bool:
    if margin(scene) < QUARANTINE_MARGIN:
        return True
    return False


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    twin = dict(scene)
    twin["draw_order"] = list(scene["draw_order"])
    twin["draw_order"].append(twin["draw_order"].pop(0))
    return [("draw_order_rotated", twin)]
