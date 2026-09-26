"""Deterministic latent sampler and raster renderer for spring_coil_count."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw


def sample_scene(seed: int) -> dict:
    rng = random.Random(int(seed))
    return {
        "coil_count": 3 + (int(seed) % 5),
        "amplitude": rng.randint(52, 66),
        "wire_width": rng.randint(5, 7),
        "vertical_shift": rng.randint(-18, 18),
        "horizontal_margin": rng.randint(50, 66),
        "canvas_width": 768,
        "canvas_height": 384,
    }


def _trace_points(scene: dict, scale: int) -> list[tuple[float, float]]:
    count = int(scene["coil_count"])
    amplitude = float(scene["amplitude"]) * scale
    shift = float(scene["vertical_shift"]) * scale
    margin = float(scene["horizontal_margin"]) * scale
    width = int(scene["canvas_width"]) * scale
    height = int(scene["canvas_height"]) * scale
    center = height / 2.0 + shift
    usable = width - 2.0 * margin
    # A cosine begins and ends at the visible top phase, so every cycle is full.
    samples = max(800, count * 220)
    points = []
    for index in range(samples + 1):
        t = index / samples
        x = margin + usable * t
        y = center - amplitude * math.cos(2.0 * math.pi * count * t)
        points.append((x, y))
    return points


def render(scene: dict) -> Image.Image:
    scale = 3
    width = int(scene["canvas_width"])
    height = int(scene["canvas_height"])
    image = Image.new("RGB", (width * scale, height * scale), (250, 249, 246))
    draw = ImageDraw.Draw(image)
    points = _trace_points(scene, scale)
    wire_width = int(scene["wire_width"]) * scale

    # Neutral end mounts make the object read as a spring without carrying label evidence.
    x0 = int(float(scene["horizontal_margin"]) * scale)
    x1 = int((width - float(scene["horizontal_margin"])) * scale)
    center = int((height / 2.0 + float(scene["vertical_shift"])) * scale)
    amplitude = int(float(scene["amplitude"]) * scale)
    mount_y = center - amplitude
    plate_half_h = 34 * scale
    plate_w = 18 * scale
    for x in (x0, x1):
        draw.rounded_rectangle(
            (x - plate_w, mount_y - plate_half_h, x + plate_w, mount_y + plate_half_h),
            radius=8 * scale,
            fill=(195, 199, 203),
            outline=(91, 98, 104),
            width=2 * scale,
        )

    # A dark under-stroke gives crisp, independently segmentable blue pixels.
    draw.line(points, fill=(20, 66, 112), width=wire_width + 4 * scale, joint="curve")
    draw.line(points, fill=(35, 139, 230), width=wire_width, joint="curve")
    radius = wire_width // 2
    for point in (points[0], points[-1]):
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=(35, 139, 230),
        )

    return image.resize((width, height), Image.Resampling.LANCZOS)


def analytic_gold(scene: dict) -> str:
    return str(int(scene["coil_count"]))


def margin(scene: dict) -> float:
    period = (float(scene["canvas_width"]) - 2.0 * float(scene["horizontal_margin"])) / float(
        scene["coil_count"]
    )
    return float(
        min(
            float(scene["amplitude"]) - 5.0 * float(scene["wire_width"]),
            period / 2.0 - 2.0 * float(scene["wire_width"]),
        )
    )


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < 12.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    return []
