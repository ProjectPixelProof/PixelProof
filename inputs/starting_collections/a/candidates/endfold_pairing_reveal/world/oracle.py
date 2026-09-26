"""Pixel-only inverse arm for the end-fold pairing reveal world.

Receives one rasterized PNG and the public question contract, nothing else. It
locates the visible fiducials (the single teal cord, its round beads, and the
black corner brackets on one end bead), executes the declared raster operation
(trace the cord, fold it end to end, mark agreeing pairs), and classifies the
resulting 5x5 pattern against the four declared glyphs with a reported margin.

It imports no renderer, prompt, generator, verifier, or gold code and never
receives latent scene data.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

ORACLE_VERSION = "endfold_pairing_reveal-oracle-0.1.0"

ABSTAIN = "abstain"
GRID = 5
BEADS = 50
PAIRS = BEADS // 2

_CORD_RGB = np.array([26, 132, 128], dtype=np.int16)
_COLOR_TOL = 40
_GRAY_TOL = 12
_GRAY_LO = 30
_GRAY_HI = 244
_BLACK_HI = 40


def _glyph_masks() -> dict[str, tuple[int, ...]]:
    def mask(cells) -> tuple[int, ...]:
        chosen = frozenset(cells)
        return tuple(
            1 if (i // GRID, i % GRID) in chosen else 0 for i in range(GRID * GRID)
        )

    return {
        "plus": mask([(2, c) for c in range(GRID)] + [(r, 2) for r in range(GRID)]),
        "ex": mask([(i, i) for i in range(GRID)] + [(i, GRID - 1 - i) for i in range(GRID)]),
        "tee": mask([(0, c) for c in range(GRID)] + [(r, 2) for r in range(1, GRID)]),
        "ell": mask([(r, 0) for r in range(GRID)] + [(GRID - 1, c) for c in range(1, GRID)]),
    }


GLYPHS = _glyph_masks()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Four-connected components of a boolean mask, as index arrays."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    out: list[np.ndarray] = []
    height, width = mask.shape
    current = 0
    for y in range(height):
        row = mask[y]
        if not row.any():
            continue
        for x in np.nonzero(row)[0]:
            if labels[y, x]:
                continue
            current += 1
            stack = [(int(y), int(x))]
            labels[y, x] = current
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width:
                        if mask[ny, nx] and not labels[ny, nx]:
                            labels[ny, nx] = current
                            stack.append((ny, nx))
            out.append(np.array(pixels, dtype=np.int32))
    return out


def _dilated_any(mask: np.ndarray, ys: np.ndarray, xs: np.ndarray, reach: int) -> bool:
    height, width = mask.shape
    y0 = max(0, int(ys.min()) - reach)
    y1 = min(height, int(ys.max()) + reach + 1)
    x0 = max(0, int(xs.min()) - reach)
    x1 = min(width, int(xs.max()) + reach + 1)
    return bool(mask[y0:y1, x0:x1].any())


def _decision(image: Image.Image) -> tuple[str, float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    cord = (np.abs(rgb - _CORD_RGB).max(axis=2) <= _COLOR_TOL)
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    value = rgb.mean(axis=2)
    gray = (spread <= _GRAY_TOL) & (value >= _GRAY_LO) & (value <= _GRAY_HI)
    black = (spread <= _GRAY_TOL) & (value < _BLACK_HI)

    if not cord.any():
        return ABSTAIN, 0.0

    beads = []
    for pixels in _components(gray):
        if not (60 <= len(pixels) <= 900):
            continue
        ys, xs = pixels[:, 0], pixels[:, 1]
        height = int(ys.max() - ys.min()) + 1
        width = int(xs.max() - xs.min()) + 1
        if max(height, width) > 26 or abs(height - width) > 4:
            continue
        if not _dilated_any(cord, ys, xs, 3):
            continue
        beads.append(
            {
                "cy": float(ys.mean()),
                "cx": float(xs.mean()),
                "tone": float(value[ys, xs].mean()),
                "n": len(pixels),
            }
        )
    if len(beads) != BEADS:
        return ABSTAIN, 0.0

    ink = cord | gray
    edges: list[tuple[float, int, int]] = []
    for i in range(BEADS):
        for j in range(i + 1, BEADS):
            ax, ay = beads[i]["cx"], beads[i]["cy"]
            bx, by = beads[j]["cx"], beads[j]["cy"]
            dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
            if dist <= 1.0 or dist > 140.0:
                continue
            covered = True
            for step in range(9):
                t = 0.3 + 0.05 * step
                py = int(round(ay + (by - ay) * t))
                px = int(round(ax + (bx - ax) * t))
                if not ink[py, px]:
                    covered = False
                    break
            if covered:
                edges.append((dist, i, j))
    if not edges:
        return ABSTAIN, 0.0

    pitch = min(edge[0] for edge in edges)
    kept = [edge for edge in edges if edge[0] <= 1.5 * pitch]
    adjacency: dict[int, list[int]] = {i: [] for i in range(BEADS)}
    for _, i, j in kept:
        adjacency[i].append(j)
        adjacency[j].append(i)
    degrees = sorted(len(v) for v in adjacency.values())
    if degrees != [1, 1] + [2] * (BEADS - 2):
        return ABSTAIN, 0.0

    ends = [i for i in range(BEADS) if len(adjacency[i]) == 1]
    if not black.any():
        return ABSTAIN, 0.0
    bys, bxs = np.nonzero(black)
    mark = (float(bys.mean()), float(bxs.mean()))
    start = min(
        ends,
        key=lambda i: (beads[i]["cy"] - mark[0]) ** 2 + (beads[i]["cx"] - mark[1]) ** 2,
    )

    order = [start]
    previous = -1
    while len(order) < BEADS:
        current = order[-1]
        nxt = [n for n in adjacency[current] if n != previous]
        if len(nxt) != 1:
            return ABSTAIN, 0.0
        previous = current
        order.append(nxt[0])
    if len(set(order)) != BEADS:
        return ABSTAIN, 0.0

    tones = [beads[i]["tone"] for i in order]
    split = (min(tones) + max(tones)) / 2.0
    clearance = min(abs(tone - split) for tone in tones)
    bits = [1 if tone > split else 0 for tone in tones]
    marks = tuple(1 if bits[k] == bits[BEADS - 1 - k] else 0 for k in range(PAIRS))

    scored = sorted(
        (sum(1 for a, b in zip(marks, mask) if a != b), name)
        for name, mask in GLYPHS.items()
    )
    best, runner_up = scored[0], scored[1]
    if best[0] != 0 or runner_up[0] - best[0] < 1:
        return ABSTAIN, float(clearance)
    return best[1], float(clearance)


def decision_from_image(image: Image.Image) -> str:
    """Glyph name recovered from the pixels alone, or 'abstain'."""
    return _decision(image)[0]


def decision_with_margin(image: Image.Image) -> tuple[str, float]:
    """Glyph plus the observed per-bead tone clearance in luma units."""
    return _decision(image)
