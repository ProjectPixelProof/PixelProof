"""Independent pixel-only inverse for the transpose-stamp world."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

BACKGROUND = (255, 255, 255)
PLAQUE_SIZE = 46
TILE_SIZE = 19
INK = (50, 63, 78)
TAG_RGB = {
    "black": (24, 29, 35),
    "amber": (230, 155, 32),
    "cyan": (18, 157, 181),
    "violet": (137, 78, 191),
}
ALLOWED = {
    BACKGROUND,
    (218, 224, 229),
    (247, 249, 250),
    INK,
    *TAG_RGB.values(),
}


class OracleError(ValueError):
    """Raised when a raster does not contain one intact inspection plaque."""


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for start_y, start_x in np.argwhere(mask):
        sy, sx = int(start_y), int(start_x)
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        queue = deque([(sy, sx)])
        points: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        result.append(points)
    return result


def _tag_box(array: np.ndarray, color: tuple[int, int, int]) -> tuple[int, int, int, int]:
    mask = np.all(array == np.asarray(color, dtype=np.uint8), axis=2)
    components = _components(mask)
    if len(components) != 1:
        raise OracleError("each border color must form exactly one component")
    points = components[0]
    ys = [point[0] for point in points]
    xs = [point[1] for point in points]
    box = min(xs), min(ys), max(xs), max(ys)
    if box[2] - box[0] + 1 != TILE_SIZE or box[3] - box[1] + 1 != TILE_SIZE:
        raise OracleError("colored border has the wrong visible extent")
    expected_ring = TILE_SIZE * TILE_SIZE - (TILE_SIZE - 4) * (TILE_SIZE - 4)
    if len(points) != expected_ring:
        raise OracleError("colored border is incomplete")
    return box


def _read_grid(array: np.ndarray, box: tuple[int, int, int, int]) -> str:
    left, top, _, _ = box
    values = []
    for row in range(3):
        for col in range(3):
            x0 = left + 3 + col * 5
            y0 = top + 3 + row * 5
            patch = array[y0 : y0 + 4, x0 : x0 + 4]
            ink_count = int(np.all(patch == np.asarray(INK, dtype=np.uint8), axis=2).sum())
            if ink_count not in {0, 16}:
                raise OracleError("cell interior is damaged or ambiguous")
            values.append("1" if ink_count == 16 else "0")
    if values.count("1") != 4:
        raise OracleError("each grid must contain exactly four filled cells")
    return "".join(values)


def _transpose(bits: str) -> str:
    return "".join(bits[c * 3 + r] for r in range(3) for c in range(3))


def decision_from_image(image: Image.Image) -> str:
    """Recover the matched border color using only pixels and visible structure."""

    array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise OracleError("expected an RGB raster")
    colors = {tuple(int(v) for v in row) for row in array.reshape(-1, 3)}
    if not colors <= ALLOWED:
        raise OracleError("raster contains an undeclared color")

    foreground = np.any(array != np.asarray(BACKGROUND, dtype=np.uint8), axis=2)
    points = np.argwhere(foreground)
    if not len(points):
        raise OracleError("inspection plaque is absent")
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0)
    if int(right - left + 1) != PLAQUE_SIZE or int(bottom - top + 1) != PLAQUE_SIZE:
        raise OracleError("inspection plaque has the wrong extent")
    crop = array[int(top) : int(bottom) + 1, int(left) : int(right) + 1]
    if np.any(np.all(crop == np.asarray(BACKGROUND, dtype=np.uint8), axis=2)):
        raise OracleError("inspection plaque is not one intact area")

    boxes = {name: _tag_box(crop, rgb) for name, rgb in TAG_RGB.items()}
    all_boxes = list(boxes.values())
    if len(set(all_boxes)) != 4:
        raise OracleError("tag borders do not identify four distinct grids")
    centers = sorted(((box[0] + box[2]) // 2, (box[1] + box[3]) // 2) for box in all_boxes)
    if len({x for x, _ in centers}) != 2 or len({y for _, y in centers}) != 2:
        raise OracleError("grids do not form the declared two-by-two plaque")

    reference = _read_grid(crop, boxes["black"])
    target = _transpose(reference)
    matches = [
        name for name in ("amber", "cyan", "violet") if _read_grid(crop, boxes[name]) == target
    ]
    if len(matches) != 1:
        raise OracleError("no unique pixel-level transpose match")
    return matches[0]
