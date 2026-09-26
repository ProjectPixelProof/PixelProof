"""Deterministic Pillow world for content-addressed visual relay chains."""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 780, 500
BACKGROUND = (250, 248, 244)
CARD_FILL = (238, 243, 248)
START_FILL = (255, 237, 248)
OUTLINE = (44, 51, 62)
INK = (22, 27, 34)
GRID = (177, 187, 199)
MAGENTA = (205, 38, 128)
TAG_RGB = {
    "red": (218, 67, 70),
    "green": (37, 160, 94),
    "blue": (52, 109, 207),
}
CARD_W, CARD_H = 226, 118
SLOT_ORIGINS = ((24, 180), (277, 180), (530, 180), (24, 342), (277, 342), (530, 342))


def _hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def _base_codes(rng: random.Random) -> list[int]:
    """Choose eight legible 3x3 codes separated by at least three cells."""
    pool = [value for value in range(1, 511) if 2 <= value.bit_count() <= 7]
    rng.shuffle(pool)
    chosen: list[int] = []
    for value in pool:
        if all(_hamming(value, prior) >= 3 for prior in chosen):
            chosen.append(value)
            if len(chosen) == 8:
                return chosen
    raise RuntimeError("could not construct a separated glyph codebook")


def sample_scene(seed: int) -> dict:
    """Sample two three-hop chains and select the first with a visible start key."""
    rng = random.Random(int(seed))
    codes = _base_codes(rng)

    # Every seventh scene exposes the one-cell runner-up boundary. The exact
    # matching remains unique, but those scenes are quarantined from headline use.
    if seed % 7 == 0:
        target_query = codes[1]
        occupied = set(codes)
        bit_order = list(range(9))
        rng.shuffle(bit_order)
        for bit in bit_order:
            candidate = target_query ^ (1 << bit)
            if candidate not in occupied - {codes[4]} and candidate not in (0, 511):
                codes[4] = candidate
                break
        else:
            raise RuntimeError("could not make a one-cell runner-up")

    first = codes[:4]
    second = codes[4:]
    records = [
        {"input": first[0], "output": first[1]},
        {"input": first[1], "output": first[2]},
        {"input": first[2], "output": first[3]},
        {"input": second[0], "output": second[1]},
        {"input": second[1], "output": second[2]},
        {"input": second[2], "output": second[3]},
    ]

    desired = ("red", "green", "blue")[seed % 3]
    other_terminal = ("red", "green", "blue")[(seed + 1) % 3]
    remaining = ["red", "red", "green", "green", "blue", "blue"]
    remaining.remove(desired)
    remaining.remove(other_terminal)
    rng.shuffle(remaining)
    colors: list[str | None] = [None] * 6
    colors[2] = desired
    colors[5] = other_terminal
    for index, color in zip((0, 1, 3, 4), remaining):
        colors[index] = color

    slots = list(range(6))
    rng.shuffle(slots)
    wanted_slot = seed % 6
    holder = slots.index(wanted_slot)
    slots[2], slots[holder] = slots[holder], slots[2]
    for index, record in enumerate(records):
        record["tag"] = colors[index]
        record["slot"] = slots[index]
        record["jitter"] = [rng.randint(-5, 5), rng.randint(-4, 4)]

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "start_code": first[0],
        "relays": records,
    }


def _trace(start_code: int, relays: list[dict]) -> tuple[str, list[int]]:
    current = int(start_code)
    visited: set[int] = set()
    path: list[int] = []
    last_tag: str | None = None
    for _ in range(len(relays) + 1):
        matches = [index for index, relay in enumerate(relays) if int(relay["input"]) == current]
        if not matches:
            if last_tag is None:
                raise ValueError("start glyph has no relay")
            return last_tag, path
        if len(matches) != 1:
            raise ValueError("glyph match is not unique")
        index = matches[0]
        if index in visited:
            raise ValueError("relay cycle does not terminate")
        visited.add(index)
        path.append(index)
        relay = relays[index]
        last_tag = str(relay["tag"])
        current = int(relay["output"])
    raise ValueError("relay chain exceeded the record count")


def analytic_gold(scene: dict) -> str:
    """Compute the terminal tag by joining the visible records by glyph equality."""
    decision, _ = _trace(scene["start_code"], scene["relays"])
    return decision


def margin(scene: dict) -> float:
    """Smallest Hamming gap to a competing input at every followed query."""
    inputs = [int(record["input"]) for record in scene["relays"]]
    current = int(scene["start_code"])
    gaps: list[int] = []
    visited: set[int] = set()
    for _ in range(len(inputs) + 1):
        matches = [index for index, value in enumerate(inputs) if value == current]
        competitors = [
            _hamming(current, value)
            for index, value in enumerate(inputs)
            if not matches or index != matches[0]
        ]
        if competitors:
            gaps.append(min(competitors))
        if not matches:
            break
        if len(matches) != 1 or matches[0] in visited:
            return 0.0
        index = matches[0]
        visited.add(index)
        current = int(scene["relays"][index]["output"])
    return float(min(gaps)) if gaps else 0.0


def is_quarantined(scene: dict) -> bool:
    return margin(scene) <= 1.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """Exercise the irrelevant storage ordering of visibly slotted records."""
    twin = dict(scene)
    twin["relays"] = [
        {key: (list(value) if isinstance(value, list) else value) for key, value in relay.items()}
        for relay in reversed(scene["relays"])
    ]
    return [("reverse_record_storage", twin)]


def _draw_pattern(draw: ImageDraw.ImageDraw, x: int, y: int, code: int) -> None:
    for row in range(3):
        for column in range(3):
            x0 = x + column * 12
            y0 = y + row * 12
            bit = row * 3 + column
            fill = INK if code & (1 << bit) else None
            draw.rectangle((x0, y0, x0 + 9, y0 + 9), fill=fill, outline=GRID, width=1)


def _center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str) -> None:
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1 - width) // 2, (y0 + y1 - height) // 2), text, fill=INK, font=font)


def render(scene: dict) -> Image.Image:
    image = Image.new("RGB", (int(scene["width"]), int(scene["height"])), BACKGROUND)
    draw = ImageDraw.Draw(image)

    start_x, start_y, start_w, start_h = 291, 38, 198, 106
    draw.rectangle(
        (start_x, start_y, start_x + start_w - 1, start_y + start_h - 1),
        fill=START_FILL,
        outline=MAGENTA,
        width=4,
    )
    _center_text(draw, (start_x, start_y + 5, start_x + start_w, start_y + 24), "START")
    _draw_pattern(draw, start_x + 82, start_y + 49, int(scene["start_code"]))

    for relay in scene["relays"]:
        base_x, base_y = SLOT_ORIGINS[int(relay["slot"])]
        x = base_x + int(relay["jitter"][0])
        y = base_y + int(relay["jitter"][1])
        draw.rectangle(
            (x, y, x + CARD_W - 1, y + CARD_H - 1),
            fill=CARD_FILL,
            outline=OUTLINE,
            width=4,
        )
        _draw_pattern(draw, x + 20, y + 22, int(relay["input"]))
        _draw_pattern(draw, x + 170, y + 22, int(relay["output"]))
        arrow_y = y + 39
        draw.line((x + 76, arrow_y, x + 148, arrow_y), fill=INK, width=4)
        draw.polygon(
            ((x + 148, arrow_y), (x + 137, arrow_y - 8), (x + 137, arrow_y + 8)),
            fill=INK,
        )
        color = TAG_RGB[str(relay["tag"])]
        draw.rounded_rectangle(
            (x + 88, y + 82, x + 138, y + 102), radius=8, fill=color, outline=OUTLINE, width=2
        )
    return image
