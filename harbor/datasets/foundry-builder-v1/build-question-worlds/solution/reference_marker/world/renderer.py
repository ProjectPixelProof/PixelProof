from __future__ import annotations

import random

from PIL import Image, ImageDraw

WIDTH = 160
HEIGHT = 120


def sample_scene(seed: int) -> dict:
    rng = random.Random(seed)
    reference_y = 60
    offset = 12 + (seed % 25)
    marker_y = reference_y - offset if seed % 2 == 0 else reference_y + offset
    return {
        "marker_x": rng.randint(30, WIDTH - 30),
        "marker_y": marker_y,
        "reference_y": reference_y,
        "latent_tag": seed % 7,
    }


def render(scene: dict) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    y = int(scene["reference_y"])
    draw.line((10, y, WIDTH - 10, y), fill=(100, 100, 100), width=3)
    x = int(scene["marker_x"])
    marker_y = int(scene["marker_y"])
    draw.ellipse((x - 6, marker_y - 6, x + 6, marker_y + 6), fill=(220, 20, 20))
    return image


def analytic_gold(scene: dict) -> str:
    return "yes" if scene["marker_y"] < scene["reference_y"] else "no"


def margin(scene: dict) -> float:
    return float(scene["reference_y"] - scene["marker_y"])


def is_quarantined(scene: dict) -> bool:
    return abs(margin(scene)) <= 3.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    twin = dict(scene)
    twin["latent_tag"] = (int(scene["latent_tag"]) + 1) % 7
    return [("unused_latent_tag", twin)]
