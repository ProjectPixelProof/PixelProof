"""Deterministic latent sampler and raster renderer for run-bar decompression."""

from __future__ import annotations

import random
from typing import Iterable

from PIL import Image, ImageDraw

LABELS = ("zigzag", "key", "anchor", "bell")

# Four separately recognizable 5x5 glyphs.  Each has ten black cells and
# thirteen total horizontal runs; 2x scaling therefore preserves both total ink
# and the number of visible run tokens across answer identities.
BASE_TEMPLATES = {
    "zigzag": ("00110", "01100", "00110", "00011", "00011"),
    "key": ("11000", "01000", "11110", "01100", "00100"),
    "anchor": ("00100", "00100", "00100", "11111", "10001"),
    "bell": ("00100", "01110", "11011", "10001", "00000"),
}

PALETTES = (
    ((36, 40, 48), (244, 201, 93)),
    ((41, 50, 92), (240, 170, 110)),
    ((54, 45, 72), (174, 218, 185)),
)
BACKGROUND = (249, 246, 238)
CYAN = (0, 151, 167)
MAGENTA = (216, 27, 96)
GRID = (206, 211, 207)
BAR_WIDTH = 16
BAR_GAP = 7
HEIGHT_UNIT = 3


def _expanded_template(label: str) -> list[list[int]]:
    base = BASE_TEMPLATES[label]
    return [
        [int(bit) for bit in row for _ in range(2)]
        for row in base
        for _ in range(2)
    ]


TEMPLATES = {label: _expanded_template(label) for label in LABELS}


def _runs(row: Iterable[int]) -> list[list[int]]:
    values = list(row)
    out: list[list[int]] = []
    for value in values:
        if out and out[-1][0] == value:
            out[-1][1] += 1
        else:
            out.append([int(value), 1])
    return out


def _decode(encoded: list[list[list[int]]]) -> list[list[int]]:
    rows: list[list[int]] = []
    for lane in encoded:
        row: list[int] = []
        for value, length in lane:
            row.extend([int(value)] * int(length))
        if len(row) != 10:
            raise ValueError("every decoded lane must contain exactly ten cells")
        rows.append(row)
    if len(rows) != 10:
        raise ValueError("the scene must contain exactly ten lanes")
    return rows


def _scores(decoded: list[list[int]]) -> list[tuple[int, str]]:
    scores = []
    for label in LABELS:
        template = TEMPLATES[label]
        matches = sum(
            int(decoded[y][x] == template[y][x]) for y in range(10) for x in range(10)
        )
        scores.append((matches, label))
    return sorted(scores, reverse=True)


def sample_scene(seed: int) -> dict:
    """Return a balanced deterministic scene without storing an answer field."""
    rng = random.Random(int(seed))
    label = LABELS[int(seed) % len(LABELS)]
    grid = [row[:] for row in TEMPLATES[label]]

    # Some seeds include a balanced four-cell corruption.  It preserves total
    # ink while producing real, still-decodable variation in the decision gap.
    if int(seed) % 5 == 0:
        ones = [(y, x) for y in range(10) for x in range(10) if grid[y][x] == 1]
        zeros = [(y, x) for y in range(10) for x in range(10) if grid[y][x] == 0]
        rng.shuffle(ones)
        rng.shuffle(zeros)
        for y, x in ones[:2]:
            grid[y][x] = 0
        for y, x in zeros[:2]:
            grid[y][x] = 1

    return {
        "encoded": [_runs(row) for row in grid],
        "x_shift": rng.randint(-18, 18),
        "palette_index": rng.randrange(len(PALETTES)),
        "canvas_width": 640,
        "canvas_height": 560,
    }


def render(scene: dict) -> Image.Image:
    encoded = scene["encoded"]
    x_shift = int(scene["x_shift"])
    palette_index = int(scene["palette_index"])
    width = int(scene["canvas_width"])
    height = int(scene["canvas_height"])
    dark, pale = PALETTES[palette_index]

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    left, right, top = 34, width - 35, 26
    lane_height = 50
    bottom = top + 10 * lane_height
    draw.rectangle((left, top, right, bottom), outline=CYAN, width=4)

    for lane_index, runs in enumerate(encoded):
        lane_top = top + lane_index * lane_height
        baseline = lane_top + 41
        # Visible integer height lattice and per-lane baseline are fiducials,
        # not hidden coordinates required by the inverse.
        for step in range(1, 11):
            y = baseline - step * HEIGHT_UNIT
            draw.line((left + 5, y, right - 5, y), fill=GRID, width=1)
        draw.line((left + 5, baseline, right - 5, baseline), fill=MAGENTA, width=2)

        token_span = len(runs) * BAR_WIDTH + (len(runs) - 1) * BAR_GAP
        x = (width - token_span) // 2 + x_shift
        for value, length in runs:
            bar_height = int(length) * HEIGHT_UNIT
            color = dark if int(value) else pale
            draw.rectangle((x, baseline - bar_height, x + BAR_WIDTH - 1, baseline - 1), fill=color)
            x += BAR_WIDTH + BAR_GAP

    return image


def analytic_gold(scene: dict) -> str:
    return _scores(_decode(scene["encoded"]))[0][1]


def margin(scene: dict) -> float:
    ranked = _scores(_decode(scene["encoded"]))
    return float(ranked[0][0] - ranked[1][0])


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < 12.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    return []
