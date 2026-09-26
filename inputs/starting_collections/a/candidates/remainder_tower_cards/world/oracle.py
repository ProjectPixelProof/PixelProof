"""Independent image-only inverse for the remainder-card transform."""

from __future__ import annotations

import numpy as np
from PIL import Image

FRAME_COLORS = ((84, 52, 135), (102, 55, 128), (76, 61, 145))
TOWER_COLORS = ((29, 65, 85), (36, 82, 71), (92, 54, 45))
AMBER = (238, 166, 42)
TEMPLATES = {
    "hourglass": ((1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 1, 0), (1, 0, 0, 1)),
    "diamond": ((0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0)),
    "falling blocks": ((1, 1, 0, 0), (1, 1, 0, 0), (0, 0, 1, 1), (0, 0, 1, 1)),
    "rising blocks": ((0, 0, 1, 1), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 0)),
}


def _near_any(arr: np.ndarray, colors: tuple[tuple[int, int, int], ...], tolerance: int = 2) -> np.ndarray:
    masks = [np.max(np.abs(arr.astype(np.int16) - np.array(color, dtype=np.int16)), axis=2) <= tolerance for color in colors]
    return np.logical_or.reduce(masks)


def _components(mask: np.ndarray, minimum: int = 1) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    remaining = set(int(value) for value in np.flatnonzero(mask))
    found: list[tuple[int, int, int, int, int]] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 0
        min_x = max_x = start % width
        min_y = max_y = start // width
        while stack:
            value = stack.pop()
            y, x = divmod(value, width)
            size += 1
            min_x, max_x = min(min_x, x), max(max_x, x)
            min_y, max_y = min(min_y, y), max(max_y, y)
            for neighbor in (value - 1, value + 1, value - width, value + width):
                if neighbor in remaining:
                    ny, nx = divmod(neighbor, width)
                    if abs(nx - x) + abs(ny - y) == 1:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
        if size >= minimum:
            found.append((min_x, min_y, max_x, max_y, size))
    return found


def _read_card(arr: np.ndarray, box: tuple[int, int, int, int, int]) -> tuple[int, tuple[int, ...]] | None:
    x0, y0, x1, y1, frame_size = box
    if (x1 - x0 + 1, y1 - y0 + 1) != (144, 132) or not (3650 <= frame_size <= 3750):
        return None
    crop = arr[y0:y1 + 1, x0:x1 + 1]

    amber_mask = _near_any(crop, (AMBER,))
    dots = _components(amber_mask, minimum=20)
    if not dots or any(not (90 <= item[4] <= 110) for item in dots):
        return None
    rank = len(dots)
    if rank not in (1, 2, 3, 4):
        return None

    tower_mask = _near_any(crop, TOWER_COLORS)
    boxes = _components(tower_mask, minimum=20)
    if len(boxes) != 14:
        return None
    if any((b[2] - b[0] + 1, b[3] - b[1] + 1, b[4]) != (23, 13, 299) for b in boxes):
        return None
    groups: dict[int, int] = {}
    for xa, _ya, xb, _yb, _size in boxes:
        center = (xa + xb) // 2
        groups[center] = groups.get(center, 0) + 1
    if len(groups) != 4:
        return None
    heights = [count for _center, count in sorted(groups.items())]
    if any(height not in (3, 4) for height in heights):
        return None
    return rank, tuple(int(height % 3 == 0) for height in heights)


def decision_from_image(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if arr.shape != (512, 512, 3):
        return "abstain"
    frame_mask = _near_any(arr, FRAME_COLORS)
    cards = _components(frame_mask, minimum=1000)
    if len(cards) != 4:
        return "abstain"
    decoded: dict[int, tuple[int, ...]] = {}
    for card in cards:
        result = _read_card(arr, card)
        if result is None or result[0] in decoded:
            return "abstain"
        decoded[result[0]] = result[1]
    if set(decoded) != {1, 2, 3, 4}:
        return "abstain"
    rows = tuple(decoded[rank] for rank in (1, 2, 3, 4))
    ranked = sorted(
        (sum(int(a != b) for ra, rb in zip(rows, template) for a, b in zip(ra, rb)), label)
        for label, template in TEMPLATES.items()
    )
    if ranked[0][0] != 0 or ranked[1][0] - ranked[0][0] < 8:
        return "abstain"
    return ranked[0][1]
