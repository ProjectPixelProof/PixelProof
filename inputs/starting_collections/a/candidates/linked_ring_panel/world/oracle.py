"""Pixel-only inverse arm for the interlocked-ring panel world.

It receives a rendered PNG and nothing else: no scene, no renderer, no gold. It
rules the grey frames to recover the five panel interiors, segments the flat ink
colours inside each panel, labels four-connected components per colour, and
calls a panel interlocked when each of its two ink colours survives as exactly
one connected curve while the two colours touch (so the rings really do cross).
A stacked pair leaves the ring behind in two pieces, so it scores 1 + 2 instead.
The decision is the unique interlocked panel; anything else abstains.
"""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

ORACLE_VERSION = "linked_ring_panel-oracle-0.1.0"
AMBIGUOUS = "ambiguous"

PANELS = 5
GREY_TOL = 14
GREY_LO = 90
GREY_HI = 190
WHITE_LO = 235
MIN_COMPONENT_PX = 10
MIN_RING_PX = 120


def _masks(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(grey frame mask, coloured ink mask) from flat RGB pixels."""
    chan_max = arr.max(axis=2).astype(np.int16)
    chan_min = arr.min(axis=2).astype(np.int16)
    spread = chan_max - chan_min
    mean = arr.mean(axis=2)
    grey = (spread <= GREY_TOL) & (mean >= GREY_LO) & (mean <= GREY_HI)
    white = (spread <= GREY_TOL) & (mean > WHITE_LO)
    ink = ~grey & ~white
    return grey, ink


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, stop) index pairs."""
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(flags)))
    return out


def _components(mask: np.ndarray) -> list[int]:
    """Sizes of four-connected components of a boolean mask."""
    height, width = mask.shape
    seen = np.zeros_like(mask)
    sizes: list[int] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        size = 0
        queue = deque([(sy, sx)])
        seen[sy, sx] = True
        while queue:
            y, x = queue.popleft()
            size += 1
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        sizes.append(size)
    return sizes


def _touching(a: np.ndarray, b: np.ndarray) -> bool:
    """True when the two masks are four-adjacent somewhere (the rings cross)."""
    near = np.zeros_like(b)
    near[1:] |= b[:-1]
    near[:-1] |= b[1:]
    near[:, 1:] |= b[:, :-1]
    near[:, :-1] |= b[:, 1:]
    return bool((a & near).any())


def _panel_interiors(grey: np.ndarray) -> list[tuple[int, int, int, int]] | None:
    """Rule the grey frames into five panel interiors, or give up."""
    col_runs = _runs(grey.sum(axis=0) > 20)
    row_runs = _runs(grey.sum(axis=1) > 20)
    if len(col_runs) != 2 * PANELS or len(row_runs) != 2:
        return None
    top, bottom = row_runs[0][1], row_runs[1][0]
    if bottom - top < 20:
        return None
    boxes = []
    for i in range(PANELS):
        left = col_runs[2 * i][1]
        right = col_runs[2 * i + 1][0]
        if right - left < 20:
            return None
        boxes.append((left, top, right, bottom))
    return boxes


def _panel_is_interlocked(sub_ink: np.ndarray, sub: np.ndarray) -> bool | None:
    """None when the panel is unreadable, else whether each colour is unbroken."""
    if not sub_ink.any():
        return None
    colors = np.unique(sub[sub_ink].reshape(-1, 3), axis=0)
    if len(colors) != 2:
        return None
    masks = []
    for color in colors:
        mask = sub_ink & np.all(sub == color.reshape(1, 1, 3), axis=2)
        if int(mask.sum()) < MIN_RING_PX:
            return None
        masks.append(mask)
    if not _touching(masks[0], masks[1]):
        return None
    counts = [
        sum(1 for size in _components(mask) if size >= MIN_COMPONENT_PX) for mask in masks
    ]
    return counts == [1, 1]


def decision_from_image(image: Image.Image) -> str:
    """Recover the interlocked panel from pixels alone, or abstain."""
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    grey, ink = _masks(arr)
    boxes = _panel_interiors(grey)
    if boxes is None:
        return AMBIGUOUS
    winners = []
    for index, (x0, y0, x1, y1) in enumerate(boxes):
        verdict = _panel_is_interlocked(ink[y0:y1, x0:x1], arr[y0:y1, x0:x1])
        if verdict is None:
            return AMBIGUOUS
        if verdict:
            winners.append(index)
    if len(winners) != 1:
        return AMBIGUOUS
    return f"panel-{winners[0] + 1}"
