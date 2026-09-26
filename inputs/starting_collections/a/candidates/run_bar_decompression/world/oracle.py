"""Independent image-only inverse for the run-bar world."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

LABELS = ("zigzag", "key", "anchor", "bell")
BASE = {
    "zigzag": ("00110", "01100", "00110", "00011", "00011"),
    "key": ("11000", "01000", "11110", "01100", "00100"),
    "anchor": ("00100", "00100", "00100", "11111", "10001"),
    "bell": ("00100", "01110", "11011", "10001", "00000"),
}
TEMPLATES = {
    name: [[int(bit) for bit in row for _ in range(2)] for row in rows for _ in range(2)]
    for name, rows in BASE.items()
}
DARKS = np.array(((36, 40, 48), (41, 50, 92), (54, 45, 72)), dtype=np.int16)
PALES = np.array(((244, 201, 93), (240, 170, 110), (174, 218, 185)), dtype=np.int16)
CYAN = np.array((0, 151, 167), dtype=np.int16)
MAGENTA = np.array((216, 27, 96), dtype=np.int16)


def _near(arr: np.ndarray, colors: np.ndarray, tolerance: int = 10) -> np.ndarray:
    delta = np.abs(arr.astype(np.int16)[..., None, :] - colors)
    return np.any(np.max(delta, axis=-1) <= tolerance, axis=-1)


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        queue = deque([(int(y0), int(x0))])
        seen[y0, x0] = True
        xs, ys = [], []
        while queue:
            y, x = queue.popleft()
            xs.append(x)
            ys.append(y)
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        found.append((min(xs), min(ys), max(xs), max(ys), len(xs)))
    return found


def _decoded_grid(image: Image.Image) -> list[list[int]]:
    arr = np.asarray(image.convert("RGB"))
    cyan_mask = _near(arr, CYAN[None, :])
    cy, cx = np.nonzero(cyan_mask)
    if len(cx) < 100:
        raise ValueError("missing cyan lane frame")
    left, right, top, bottom = int(cx.min()), int(cx.max()), int(cy.min()), int(cy.max())
    if right - left < 300 or bottom - top < 300:
        raise ValueError("cyan frame is too small")

    magenta = _near(arr, MAGENTA[None, :])
    row_counts = magenta[:, left : right + 1].sum(axis=1)
    baseline_rows = np.flatnonzero(row_counts > (right - left) * 0.6)
    groups = []
    for y in baseline_rows:
        if not groups or y > groups[-1][-1] + 1:
            groups.append([int(y)])
        else:
            groups[-1].append(int(y))
    baselines = [round(sum(group) / len(group)) for group in groups]
    if len(baselines) != 10:
        raise ValueError(f"expected ten magenta baselines, found {len(baselines)}")

    dark_mask = _near(arr, DARKS)
    pale_mask = _near(arr, PALES)
    rows = []
    prior = top
    for baseline in baselines:
        lane_mask = (dark_mask | pale_mask)[prior:baseline, left + 1 : right]
        pieces = [c for c in _components(lane_mask) if c[4] >= 20]
        pieces.sort(key=lambda c: c[0])
        if not pieces:
            raise ValueError("lane contains no run bars")
        row = []
        for x0, y0, x1, y1, area in pieces:
            absolute = arr[prior + y0 : prior + y1 + 1, left + 1 + x0 : left + 2 + x1]
            dark_votes = int(_near(absolute, DARKS).sum())
            pale_votes = int(_near(absolute, PALES).sum())
            value = int(dark_votes > pale_votes)
            length = int(round((y1 - y0 + 1) / 3.0))
            if not 1 <= length <= 10:
                raise ValueError("bar height is outside the declared grid")
            row.extend([value] * length)
        if len(row) != 10:
            raise ValueError(f"decoded lane has {len(row)} cells rather than ten")
        rows.append(row)
        prior = baseline + 1
    return rows


def decision_from_image(image: Image.Image) -> str:
    decoded = _decoded_grid(image)
    ranked = []
    for label in LABELS:
        score = sum(
            int(decoded[y][x] == TEMPLATES[label][y][x])
            for y in range(10)
            for x in range(10)
        )
        ranked.append((score, label))
    ranked.sort(reverse=True)
    if ranked[0][0] < 70 or ranked[0][0] == ranked[1][0]:
        raise ValueError("decoded raster is ambiguous or too corrupted")
    return ranked[0][1]
