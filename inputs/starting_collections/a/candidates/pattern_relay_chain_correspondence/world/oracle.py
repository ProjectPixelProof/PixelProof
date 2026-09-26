"""Independent inverse arm using only RGB pixels and visible design conventions."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

CARD_FILL = (238, 243, 248)
START_FILL = (255, 237, 248)
TAG_RGB = {
    "red": (218, 67, 70),
    "green": (37, 160, 94),
    "blue": (52, 109, 207),
}


def _components(mask: np.ndarray, minimum: int) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found: list[tuple[int, int, int, int, int]] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if seen[start_y, start_x]:
            continue
        queue = deque([(int(start_x), int(start_y))])
        seen[start_y, start_x] = True
        area = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while queue:
            x, y = queue.popleft()
            area += 1
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= nx < width and 0 <= ny < height and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((nx, ny))
        if area >= minimum:
            found.append((min_x, min_y, max_x, max_y, area))
    return found


def _read_pattern(pixels: np.ndarray, x: int, y: int) -> int:
    code = 0
    for row in range(3):
        for column in range(3):
            cx = x + column * 12 + 4
            cy = y + row * 12 + 4
            patch = pixels[cy - 2 : cy + 3, cx - 2 : cx + 3]
            dark = np.all(patch < 90, axis=2)
            if int(dark.sum()) >= 18:
                code |= 1 << (row * 3 + column)
    if code in (0, 511):
        raise ValueError("glyph is missing or degenerate")
    return code


def _read_tag(pixels: np.ndarray, bounds: tuple[int, int, int, int, int]) -> str:
    x0, y0, x1, y1, _ = bounds
    crop = pixels[y0 : y1 + 1, x0 : x1 + 1]
    counts = {
        name: int(np.all(crop == np.asarray(rgb, dtype=np.uint8), axis=2).sum())
        for name, rgb in TAG_RGB.items()
    }
    ordered = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    if ordered[0][1] < 300 or ordered[0][1] == ordered[1][1]:
        raise ValueError("relay color tag is absent or ambiguous")
    return ordered[0][0]


def decision_from_image(image: Image.Image) -> str:
    """Parse cards, perform exact visual joins, and return the terminal tag."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("expected an RGB raster")

    relay_mask = np.all(pixels == np.asarray(CARD_FILL, dtype=np.uint8), axis=2)
    relay_boxes = _components(relay_mask, 10000)
    if len(relay_boxes) != 6:
        raise ValueError(f"expected six relay cards, found {len(relay_boxes)}")
    start_mask = np.all(pixels == np.asarray(START_FILL, dtype=np.uint8), axis=2)
    start_boxes = _components(start_mask, 8000)
    if len(start_boxes) != 1:
        raise ValueError(f"expected one start card, found {len(start_boxes)}")

    sx0, sy0, _, _, _ = start_boxes[0]
    start = _read_pattern(pixels, sx0 - 4 + 82, sy0 - 4 + 49)

    records: list[tuple[int, int, str]] = []
    for bounds in relay_boxes:
        x0, y0, _, _, _ = bounds
        outer_x, outer_y = x0 - 4, y0 - 4
        left = _read_pattern(pixels, outer_x + 20, outer_y + 22)
        right = _read_pattern(pixels, outer_x + 170, outer_y + 22)
        records.append((left, right, _read_tag(pixels, bounds)))

    current = start
    visited: set[int] = set()
    terminal: str | None = None
    for _ in range(len(records) + 1):
        matches = [index for index, record in enumerate(records) if record[0] == current]
        if not matches:
            if terminal is None:
                raise ValueError("start pattern has no exact input match")
            return terminal
        if len(matches) != 1:
            raise ValueError("input pattern match is not unique")
        index = matches[0]
        if index in visited:
            raise ValueError("relay chain is cyclic")
        visited.add(index)
        _, current, terminal = records[index]
    raise ValueError("relay chain does not terminate")
