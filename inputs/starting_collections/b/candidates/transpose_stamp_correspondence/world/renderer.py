"""Deterministic renderer for the compact transpose-stamp world."""

from __future__ import annotations

import itertools
import random

from PIL import Image, ImageDraw

CANVAS = 256
PLAQUE_SIZE = 46
TILE_SIZE = 19
ANSWERS = ("amber", "cyan", "violet")

RGB = {
    "background": (255, 255, 255),
    "plaque": (218, 224, 229),
    "off": (247, 249, 250),
    "ink": (50, 63, 78),
    "black": (24, 29, 35),
    "amber": (230, 155, 32),
    "cyan": (18, 157, 181),
    "violet": (137, 78, 191),
}

TILE_OFFSETS = ((3, 3), (24, 3), (3, 24), (24, 24))


def _transpose_bits(bits: str) -> str:
    return "".join(bits[c * 3 + r] for r in range(3) for c in range(3))


def _hamming(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right, strict=True))


def sample_scene(seed: int) -> dict:
    """Sample one balanced scene from an integer seed."""

    rng = random.Random((int(seed) ^ 0x5A17C9E3) & 0xFFFFFFFFFFFFFFFF)
    patterns = [
        "".join(bits)
        for bits in itertools.product("01", repeat=9)
        if bits.count("1") == 4 and "".join(bits) != _transpose_bits("".join(bits))
    ]
    reference = patterns[rng.randrange(len(patterns))]
    target = _transpose_bits(reference)

    pool = [
        bits
        for bits in ("".join(v) for v in itertools.product("01", repeat=9))
        if bits.count("1") == 4
        and bits not in {reference, target}
        and _hamming(bits, target) >= 2
    ]
    rng.shuffle(pool)
    distractors: list[str] = []
    for bits in pool:
        if all(_hamming(bits, other) >= 2 for other in distractors):
            distractors.append(bits)
        if len(distractors) == 2:
            break
    if len(distractors) != 2:
        raise RuntimeError("could not sample distinct equal-ink distractors")

    target_slot = (int(seed) // 3) % 3
    candidates = distractors[:]
    candidates.insert(target_slot, target)

    answer_color = ANSWERS[int(seed) % 3]
    remaining_colors = [color for color in ANSWERS if color != answer_color]
    if rng.randrange(2):
        remaining_colors.reverse()
    tag_colors: list[str | None] = [None, None, None]
    tag_colors[target_slot] = answer_color
    fill_iter = iter(remaining_colors)
    for index in range(3):
        if tag_colors[index] is None:
            tag_colors[index] = next(fill_iter)

    return {
        "reference": reference,
        "candidates": candidates,
        "tag_colors": [str(value) for value in tag_colors],
        "anchor_x": rng.randint(12, CANVAS - PLAQUE_SIZE - 12),
        "anchor_y": rng.randint(12, CANVAS - PLAQUE_SIZE - 12),
        "ref_slot": (int(seed) // 9) % 4,
    }


def _draw_tile(draw: ImageDraw.ImageDraw, x: int, y: int, tag: str, bits: str) -> None:
    draw.rectangle((x, y, x + TILE_SIZE - 1, y + TILE_SIZE - 1), fill=RGB[tag])
    draw.rectangle((x + 2, y + 2, x + 16, y + 16), fill=RGB["off"])
    for row in range(3):
        for col in range(3):
            left = x + 3 + col * 5
            top = y + 3 + row * 5
            fill = RGB["ink"] if bits[row * 3 + col] == "1" else RGB["off"]
            draw.rectangle((left, top, left + 3, top + 3), fill=fill)


def render(scene: dict) -> Image.Image:
    """Render the scene; every scene field is visibly expressed."""

    reference = scene["reference"]
    candidates = scene["candidates"]
    tag_colors = scene["tag_colors"]
    anchor_x = int(scene["anchor_x"])
    anchor_y = int(scene["anchor_y"])
    ref_slot = int(scene["ref_slot"])

    image = Image.new("RGB", (CANVAS, CANVAS), RGB["background"])
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (anchor_x, anchor_y, anchor_x + PLAQUE_SIZE - 1, anchor_y + PLAQUE_SIZE - 1),
        fill=RGB["plaque"],
    )
    candidate_index = 0
    for slot, (dx, dy) in enumerate(TILE_OFFSETS):
        if slot == ref_slot:
            tag, bits = "black", reference
        else:
            tag = str(tag_colors[candidate_index])
            bits = str(candidates[candidate_index])
            candidate_index += 1
        _draw_tile(draw, anchor_x + dx, anchor_y + dy, tag, bits)
    return image


def analytic_gold(scene: dict) -> str:
    """Return the color bound to the unique exact transpose."""

    reference = scene["reference"]
    candidates = scene["candidates"]
    tag_colors = scene["tag_colors"]
    target = _transpose_bits(reference)
    matches = [index for index, bits in enumerate(candidates) if bits == target]
    if len(matches) != 1:
        raise ValueError("scene has no unique transpose match")
    return str(tag_colors[matches[0]])


def margin(scene: dict) -> float:
    """Runner-up Hamming distance from the exact target."""

    target = _transpose_bits(str(scene["reference"]))
    distances = sorted(_hamming(target, str(bits)) for bits in scene["candidates"])
    if not distances or distances[0] != 0:
        return 0.0
    return float(distances[1])


def is_quarantined(scene: dict) -> bool:
    try:
        target = _transpose_bits(str(scene["reference"]))
        matches = sum(str(bits) == target for bits in scene["candidates"])
        return matches != 1 or margin(scene) < 2.0
    except (KeyError, TypeError, ValueError):
        return True


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """No intended pixel-identical aliases exist for this explicit layout."""

    return []
