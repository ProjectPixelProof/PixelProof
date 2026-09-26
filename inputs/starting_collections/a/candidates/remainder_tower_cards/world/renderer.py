"""Deterministic sampler and raster renderer for remainder tower cards."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

SIZE = (512, 512)
BACKGROUND = (248, 246, 240)
AMBER = (238, 166, 42)
PALETTES = (
    ((84, 52, 135), (29, 65, 85)),
    ((102, 55, 128), (36, 82, 71)),
    ((76, 61, 145), (92, 54, 45)),
)
DECISIONS = ("hourglass", "diamond", "falling blocks", "rising blocks")
TEMPLATES = {
    "hourglass": ((1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)),
    "diamond": ((0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0)),
    "falling blocks": ((1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 1, 1)),
    "rising blocks": ((0, 0, 1, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 0)),
}
SLOTS = ((16, 16), (352, 16), (16, 364), (352, 364))
CARD_W, CARD_H = 144, 132


def sample_scene(seed: int) -> dict:
    rng = random.Random(int(seed))
    label = DECISIONS[int(seed) % len(DECISIONS)]
    slot_order = [1, 2, 3, 4]
    rng.shuffle(slot_order)
    jitter = [[rng.randint(-6, 6), rng.randint(-6, 6)] for _ in SLOTS]
    return {
        "rows": [list(row) for row in TEMPLATES[label]],
        "slot_order": slot_order,
        "palette": rng.randrange(len(PALETTES)),
        "jitter": jitter,
    }


def _decoded_rows(scene: dict) -> tuple[tuple[int, ...], ...]:
    rows = tuple(tuple(int(value) for value in row) for row in scene["rows"])
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("rows must form a 4 by 4 tile")
    return rows


def _distances(rows: tuple[tuple[int, ...], ...]) -> list[tuple[int, str]]:
    return sorted(
        (
            sum(int(a != b) for row_a, row_b in zip(rows, template) for a, b in zip(row_a, row_b)),
            label,
        )
        for label, template in TEMPLATES.items()
    )


def analytic_gold(scene: dict) -> str:
    ranked = _distances(_decoded_rows(scene))
    if ranked[0][0] != 0 or ranked[0][0] == ranked[1][0]:
        return "abstain"
    return ranked[0][1]


def margin(scene: dict) -> float:
    ranked = _distances(_decoded_rows(scene))
    return float(ranked[1][0] - ranked[0][0])


def is_quarantined(scene: dict) -> bool:
    try:
        return analytic_gold(scene) == "abstain" or margin(scene) < 8.0
    except (TypeError, ValueError, KeyError):
        return True


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    return []


def render(scene: dict) -> Image.Image:
    rows = _decoded_rows(scene)
    slot_order = [int(value) for value in scene["slot_order"]]
    jitter = [[int(v) for v in pair] for pair in scene["jitter"]]
    if sorted(slot_order) != [1, 2, 3, 4] or len(jitter) != 4:
        raise ValueError("card ranks must be a permutation with four jitters")
    frame, tower = PALETTES[int(scene["palette"])]
    image = Image.new("RGB", SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)

    for slot, rank in enumerate(slot_order):
        dx, dy = jitter[slot]
        x0, y0 = SLOTS[slot][0] + dx, SLOTS[slot][1] + dy
        x1, y1 = x0 + CARD_W - 1, y0 + CARD_H - 1
        draw.rectangle((x0, y0, x1, y1), fill=frame)
        draw.rectangle((x0 + 6, y0 + 6, x1 - 6, y1 - 6), fill=BACKGROUND)
        draw.rectangle((x0 + 6, y0 + 34, x1 - 6, y0 + 37), fill=frame)

        dot_y = y0 + 20
        dot_start = x0 + CARD_W // 2 - ((rank - 1) * 18) // 2
        for index in range(rank):
            cx = dot_start + index * 18
            draw.ellipse((cx - 5, dot_y - 5, cx + 5, dot_y + 5), fill=AMBER)

        baseline = y0 + 119
        for column, bit in enumerate(rows[rank - 1]):
            height = 3 if bit else 4
            xa = x0 + 13 + column * 32
            for unit in range(height):
                ya = baseline - unit * 16 - 12
                draw.rectangle((xa, ya, xa + 22, ya + 12), fill=tower)
        draw.line((x0 + 12, baseline + 3, x1 - 12, baseline + 3), fill=(170, 166, 158), width=2)
    return image
