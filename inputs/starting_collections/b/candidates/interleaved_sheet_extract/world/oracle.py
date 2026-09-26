"""Pixel-only inverse arm for the interleaved-sheet world.

Receives one rendered PNG and nothing else.  It locates the printed frame
fiducial, reads the dot/ring marker beside each lattice row, executes the
declared extraction (keep dot-marked rows, in order, pushed together), and
matches the resulting 7x7 block against the five symbol names published in the
question.  It shares no code, no constants of the scene, and no geometry with
the renderer; the symbol templates are re-transcribed here from the verbal
descriptions in the question text.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

LATTICE_ROWS = 14
LATTICE_COLS = 7
KEEP_ROWS = 7
ABSTAIN = "abstain"

SYMBOL_TEMPLATES = {
    "ring": [
        "0000000",
        "0111110",
        "0100010",
        "0100010",
        "0100010",
        "0111110",
        "0000000",
    ],
    "wedge": [
        "0000000",
        "1111111",
        "0111110",
        "0011100",
        "0001000",
        "0000000",
        "0000000",
    ],
    "zigzag": [
        "1111110",
        "0000010",
        "0000100",
        "0001000",
        "0010000",
        "0100000",
        "0011111",
    ],
    "arrow": [
        "1000001",
        "1100011",
        "0110110",
        "0011100",
        "0001000",
        "0001000",
        "0001000",
    ],
    "slash": [
        "1100000",
        "1110000",
        "0111000",
        "0011100",
        "0000110",
        "0000011",
        "0000001",
    ],
}


def _classes(arr: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    flat = arr.reshape(-1, 3)
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    background = colors[int(counts.argmax())]
    out = []
    for color, count in zip(colors, counts):
        if count < 900:
            continue
        if int(np.abs(color.astype(int) - background.astype(int)).sum()) < 120:
            continue
        mask = np.all(arr == color, axis=2)
        ys, xs = np.nonzero(mask)
        box = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        area = max(1, (box[2] - box[0] + 1) * (box[3] - box[1] + 1))
        out.append(
            {
                "mask": mask,
                "box": box,
                "count": int(count),
                "fill": float(count) / area,
                "area": area,
            }
        )
    return background, out


def decision_from_image(image: Image.Image) -> str:
    arr = np.asarray(image.convert("RGB"))
    _, classes = _classes(arr)
    if len(classes) < 2:
        return ABSTAIN

    hollow = [c for c in classes if c["fill"] < 0.2]
    if not hollow:
        return ABSTAIN
    frame = max(hollow, key=lambda c: c["area"])
    fx0, fy0, fx1, fy1 = frame["box"]
    if fx1 - fx0 < 80 or fy1 - fy0 < 160:
        return ABSTAIN

    rest = [c for c in classes if c is not frame]
    inside = []
    outside = []
    for c in rest:
        bx0, by0, bx1, by1 = c["box"]
        if bx0 >= fx0 and bx1 <= fx1 and by0 >= fy0 and by1 <= fy1:
            inside.append(c)
        else:
            outside.append(c)
    if not inside or not outside:
        return ABSTAIN
    cells = max(inside, key=lambda c: c["count"])
    marks = max(outside, key=lambda c: c["count"])

    row_edges = np.linspace(fy0, fy1 + 1, LATTICE_ROWS + 1)
    col_edges = np.linspace(fx0, fx1 + 1, LATTICE_COLS + 1)

    mark_mask = marks["mask"][:, :fx0]
    if mark_mask.size == 0:
        return ABSTAIN
    weights = []
    for r in range(LATTICE_ROWS):
        top = int(round(row_edges[r]))
        bottom = int(round(row_edges[r + 1]))
        weights.append(int(mark_mask[top:bottom, :].sum()))
    if min(weights) < 40:
        return ABSTAIN

    order = sorted(range(LATTICE_ROWS), key=lambda i: weights[i])
    gaps = [
        (weights[order[i + 1]] - weights[order[i]], i)
        for i in range(LATTICE_ROWS - 1)
    ]
    best_gap, split = max(gaps)
    if best_gap < 40 or split != LATTICE_ROWS - KEEP_ROWS - 1:
        return ABSTAIN
    solid = set(order[split + 1 :])

    block = []
    for r in range(LATTICE_ROWS):
        if r not in solid:
            continue
        cy = 0.5 * (row_edges[r] + row_edges[r + 1])
        line = []
        for c in range(LATTICE_COLS):
            cx = 0.5 * (col_edges[c] + col_edges[c + 1])
            y0 = int(round(cy)) - 5
            x0 = int(round(cx)) - 5
            patch = cells["mask"][y0 : y0 + 11, x0 : x0 + 11]
            if patch.size == 0:
                return ABSTAIN
            line.append(1 if patch.mean() > 0.5 else 0)
        block.append(line)
    if len(block) != KEEP_ROWS:
        return ABSTAIN

    ranked = sorted(
        (
            sum(
                1
                for r in range(KEEP_ROWS)
                for c in range(LATTICE_COLS)
                if block[r][c] != int(rows[r][c])
            ),
            name,
        )
        for name, rows in SYMBOL_TEMPLATES.items()
    )
    return ranked[0][1]
