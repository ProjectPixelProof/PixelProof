"""Deterministic renderer and analytic arm for Nine-Tooth Route Rasterization."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw

WIDTH = HEIGHT = 512
SYMBOLS = ("X", "plus", "T", "L")
TEMPLATES = {
    "X": (1, 0, 1, 0, 1, 0, 1, 0, 1),
    "plus": (0, 1, 0, 1, 1, 1, 0, 1, 0),
    "T": (1, 1, 1, 0, 1, 0, 0, 1, 0),
    "L": (1, 0, 0, 1, 0, 0, 1, 1, 1),
}
INKS = ((31, 48, 72), (62, 35, 68), (25, 67, 57))
BACKGROUNDS = ((250, 248, 242), (246, 249, 250), (250, 246, 249))
X_SLOTS = (82, 210, 338)
Y_BASES = (96, 224, 352)


def sample_scene(seed: int) -> dict:
    rng = random.Random(int(seed) ^ 0x6A09E667)
    symbol = SYMBOLS[int(seed) % len(SYMBOLS)]
    step = rng.choice((9, 10, 11))
    amplitude = rng.choice((10, 11, 12))
    x0 = rng.choice(X_SLOTS) + rng.randint(-2, 2)
    y0 = rng.choice(Y_BASES) + rng.randint(-1, 1)
    reverse = bool(rng.getrandbits(1))
    decorations = []
    for _ in range(7):
        # Pale remote marks are nuisance-only and kept well away from the route badge.
        for _attempt in range(100):
            x, y = rng.randint(28, 484), rng.randint(28, 484)
            route_mid_x = x0 + (9 * step) / 2
            if abs(x - route_mid_x) > 105 or abs(y - y0) > 65:
                decorations.append([x, y, rng.randint(2, 4), rng.randrange(3)])
                break
    return {
        "bits": list(TEMPLATES[symbol]),
        "x0": x0,
        "y0": y0,
        "step": step,
        "amplitude": amplitude,
        "stroke": rng.choice((4, 5, 6)),
        "reverse": reverse,
        "ink": list(rng.choice(INKS)),
        "background": list(rng.choice(BACKGROUNDS)),
        "decorations": decorations,
    }


def _route_points(scene: dict) -> list[tuple[float, float]]:
    bits = scene["bits"]
    x0, y0 = float(scene["x0"]), float(scene["y0"])
    step, amp = float(scene["step"]), float(scene["amplitude"])
    points = [(x0, y0)]
    for index, bit in enumerate(bits):
        left = x0 + index * step
        points.append((left + step / 2.0, y0 - amp if int(bit) else y0 + amp))
        points.append((left + step, y0))
    if scene["reverse"]:
        axis = x0 + 9 * step
        points = [(axis - (x - x0), y) for x, y in points]
    return points


def render(scene: dict) -> Image.Image:
    scale = 4
    image = Image.new("RGB", (WIDTH * scale, HEIGHT * scale), tuple(scene["background"]))
    draw = ImageDraw.Draw(image)
    x0, y0 = int(scene["x0"]), int(scene["y0"])
    step, amp = int(scene["step"]), int(scene["amplitude"])
    route_width = 9 * step

    # Remote, low-contrast distractors are independent of the label.
    pastel = ((214, 205, 226), (204, 222, 215), (230, 216, 195))
    for x, y, radius, color_index in scene["decorations"]:
        draw.ellipse(
            ((x-radius)*scale, (y-radius)*scale, (x+radius)*scale, (y+radius)*scale),
            fill=pastel[int(color_index)],
        )

    frame = (x0 - 12, y0 - amp - 12, x0 + route_width + 12, y0 + amp + 12)
    draw.rounded_rectangle(tuple(v * scale for v in frame), radius=10*scale,
                           fill=(237, 234, 228), outline=(181, 174, 166), width=2*scale)
    draw.line([(x0*scale, y0*scale), ((x0+route_width)*scale, y0*scale)],
              fill=(202, 196, 187), width=1*scale)

    points = [(int(round(x*scale)), int(round(y*scale))) for x, y in _route_points(scene)]
    draw.line(points, fill=tuple(scene["ink"]), width=int(scene["stroke"])*scale,
              joint="curve")

    start = points[0]
    end = points[-1]
    direction = 1 if end[0] > start[0] else -1
    tri = [
        (start[0] - direction*7*scale, start[1] - 7*scale),
        (start[0] - direction*7*scale, start[1] + 7*scale),
        (start[0] + direction*6*scale, start[1]),
    ]
    draw.polygon(tri, fill=(232, 157, 43), outline=(145, 88, 18))
    r = 7 * scale
    draw.ellipse((end[0]-r, end[1]-r, end[0]+r, end[1]+r),
                 fill=(45, 174, 168), outline=(20, 130, 130), width=scale)
    # Keep the route itself continuously visible through both caps. Besides
    # helping a human trace it, this makes continuity an observable pixel fact.
    draw.line(points, fill=tuple(scene["ink"]), width=int(scene["stroke"])*scale,
              joint="curve")
    return image.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)


def _classify(bits) -> str:
    key = tuple(int(value) for value in bits)
    matches = [name for name, template in TEMPLATES.items() if key == template]
    if len(matches) != 1:
        raise ValueError("bits do not encode one declared symbol")
    return matches[0]


def analytic_gold(scene: dict) -> str:
    return _classify(scene["bits"])


def margin(scene: dict) -> float:
    bits = tuple(int(value) for value in scene["bits"])
    distances = sorted(sum(a != b for a, b in zip(bits, template)) for template in TEMPLATES.values())
    return float(distances[1] - distances[0])


def is_quarantined(scene: dict) -> bool:
    return False


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    return []
