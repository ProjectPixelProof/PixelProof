"""Independent pixel-only inverse arm for bitplane_glyph_reveal.

Receives ONLY the final PNG. It re-discovers the hidden glyph by isolating the
least-significant bit plane (value & 1) of every pixel, block-averaging up to
the declared 32x32 grid, and classifying the recovered shape against the two
fixed block-glyph candidates A and B declared by the public question contract.
No scene data and no analytic gold are read.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

ORACLE_VERSION = "bitplane_glyph_reveal-oracle-0.1.0"

NROWS = 32
NCOLS = 32
CELL = 16
BOX_R0, BOX_R1 = 8, 23
BOX_C0, BOX_C1 = 10, 22

TEMPLATE_A = [
    list("............"),
    list("....##......"),
    list("....##......"),
    list("...#..#....."),
    list("...#..#....."),
    list("..#....#...."),
    list("..#....#...."),
    list(".#......#..."),
    list(".#......#..."),
    list(".########..."),
    list("#........#.."),
    list("#........#.."),
    list("#........#.."),
    list("............"),
    list("............"),
]
TEMPLATE_B = [
    list("............"),
    list("##########.."),
    list("#........#.."),
    list("#........#.."),
    list("#........#.."),
    list("##########.."),
    list("#........#.."),
    list("#........#.."),
    list("#........#.."),
    list("##########.."),
    list("............"),
    list("............"),
    list("............"),
    list("............"),
    list("............"),
]
TEMPLATES = {"A": TEMPLATE_A, "B": TEMPLATE_B}


def _template_mask(glyph: str) -> np.ndarray:
    grid = np.array([[c == "#" for c in row] for row in TEMPLATES[glyph]], dtype=bool)
    return grid


def _recover_bitfield(img: Image.Image) -> np.ndarray:
    """(NROWS, NCOLS) boolean glyph field recovered from the LSB plane."""
    arr = np.asarray(img.convert("L"), dtype=np.int32)
    bits = arr & 1
    pooled = (
        bits.reshape(NROWS, CELL, NCOLS, CELL).sum(axis=(1, 3))
    )
    return pooled > (CELL * CELL // 2)


def decision_from_image(image: Image.Image) -> str:
    field = _recover_bitfield(image)
    crop = field[BOX_R0:BOX_R1, BOX_C0:BOX_C1]
    scores = {}
    for glyph, tmpl in TEMPLATES.items():
        tp = _template_mask(glyph)
        on = int(tp.sum())
        scores[glyph] = float((crop & tp).sum()) / max(1, on)
    return "A" if scores["A"] >= scores["B"] else "B"


def reveal_margin(image: Image.Image) -> float:
    """Decision margin recovered from pixels alone (diagnostic / quarantine aid)."""
    field = _recover_bitfield(image)
    crop = field[BOX_R0:BOX_R1, BOX_C0:BOX_C1]
    scores = {}
    for glyph, tmpl in TEMPLATES.items():
        tp = _template_mask(glyph)
        on = int(tp.sum())
        scores[glyph] = float((crop & tp).sum()) / max(1, on)
    return abs(scores["A"] - scores["B"])
