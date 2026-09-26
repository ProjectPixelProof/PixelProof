"""Independent pixel-only oracle for ``stitch_face_alternation``.

Receives one RGB PIL image and nothing else: no latent scene, no renderer,
prompt, generator, verifier or analytic-gold import.  It re-derives the decision
from the raster by

1. classifying tape, thread and tag ink by colour;
2. picking the unique tape component that actually has thread stubs on both of
   its long edges (remote scraps have none);
3. reading the thread stub positions just outside each tape edge and pairing
   them in x order into five crossings;
4. probing the tape interior along each crossing's chord: thread ink present ->
   the thread passes in FRONT, tape only -> BEHIND;
5. locating the crimson tag to fix the walk direction, then reporting whether
   the walked front/behind sequence strictly alternates and what the first
   crossing's face is.

Anything unreadable (missing tape, wrong stub count, ambiguous chord, missing or
centred tag) returns ``ABSTAIN``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

ORACLE_VERSION = "stitch_face_alternation-oracle-0.1.0"
ABSTAIN = "abstain"

_THREAD = (26, 42, 88)
_TAPE = (198, 168, 120)
_TAG = (196, 32, 48)
_TOL = 46

_N_CROSSINGS = 5
_PROBE_LO = 0.30
_PROBE_HI = 0.70


def _mask(arr: np.ndarray, rgb) -> np.ndarray:
    ref = np.array(rgb, dtype=np.int16)
    return (np.abs(arr.astype(np.int16) - ref).max(axis=2) <= _TOL)


def _components(mask: np.ndarray, min_px: int = 40):
    """4-connected components of a boolean mask, as (size, y0, x0, y1, x1)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys.tolist(), xs.tolist()):
        if seen[sy, sx]:
            continue
        q = deque([(sy, sx)])
        seen[sy, sx] = True
        size = 0
        y0 = y1 = sy
        x0 = x1 = sx
        while q:
            y, x = q.popleft()
            size += 1
            y0, y1 = min(y0, y), max(y1, y)
            x0, x1 = min(x0, x), max(x1, x)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        if size >= min_px:
            out.append((size, y0, x0, y1, x1))
    return out


def _clusters(xs: list[int], gap: int = 3) -> list[float]:
    """Group sorted x positions into clusters, returning cluster centres."""
    if not xs:
        return []
    xs = sorted(xs)
    groups = [[xs[0]]]
    for v in xs[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [sum(g) / len(g) for g in groups]


def _stub_centres(thread: np.ndarray, y_lo: int, y_hi: int, x0: int, x1: int) -> list[float]:
    h, w = thread.shape
    y_lo = max(0, y_lo)
    y_hi = min(h - 1, y_hi)
    if y_hi < y_lo:
        return []
    band = thread[y_lo:y_hi + 1, max(0, x0):min(w, x1 + 1)]
    cols = np.nonzero(band.any(axis=0))[0]
    return [c + max(0, x0) for c in _clusters(cols.tolist())]


def decision_from_image(image) -> str:
    arr = np.asarray(image.convert("RGB"))
    if arr.ndim != 3:
        return ABSTAIN
    h, w = arr.shape[:2]
    tape = _mask(arr, _TAPE)
    thread = _mask(arr, _THREAD)
    tag = _mask(arr, _TAG)

    # --- 1. the stitched tape strip: the only tape with thread on both edges ---
    # FRONT crossings sever the strip into pieces; heal those thin cuts with a
    # short horizontal dilation.  Remote tape pieces stay well separated.
    healed = tape.copy()
    for k in range(1, 7):
        healed[:, k:] |= tape[:, :-k]
        healed[:, :-k] |= tape[:, k:]

    scored = []
    for comp in _components(healed, min_px=120):
        _, y0, x0, y1, x1 = comp
        above = _stub_centres(thread, y0 - 6, y0 - 2, x0, x1)
        below = _stub_centres(thread, y1 + 2, y1 + 6, x0, x1)
        scored.append((min(len(above), len(below)), comp, above, below))
    scored.sort(key=lambda s: -s[0])
    if not scored or scored[0][0] < _N_CROSSINGS:
        return ABSTAIN
    if len(scored) > 1 and scored[1][0] > 0:
        return ABSTAIN            # more than one candidate strip: unreadable
    best = scored[0]

    _, (_, ty0, tx0, ty1, tx1), above, below = best
    if len(above) != _N_CROSSINGS or len(below) != _N_CROSSINGS:
        return ABSTAIN
    if ty1 - ty0 < 8:
        return ABSTAIN

    # --- 2. face of each crossing, read inside the tape along its chord ---
    y_a = ty0 - 4.0
    y_b = ty1 + 4.0
    faces: list[str] = []
    for xa, xb in zip(sorted(above), sorted(below)):
        hits = 0
        total = 0
        for i in range(9):
            t = _PROBE_LO + (_PROBE_HI - _PROBE_LO) * i / 8.0
            yy = ty0 + t * (ty1 - ty0)
            xx = xa + (yy - y_a) / (y_b - y_a) * (xb - xa)
            iy, ix = int(round(yy)), int(round(xx))
            if not (0 <= iy < h and 0 <= ix < w):
                return ABSTAIN
            total += 1
            win = thread[max(0, iy - 1):iy + 2, max(0, ix - 2):ix + 3]
            if win.any():
                hits += 1
        if hits >= total - 1:
            faces.append("front")
        elif hits == 0:
            faces.append("behind")
        else:
            return ABSTAIN

    # --- 3. the crimson tag fixes the walk direction ---
    tag_comps = _components(tag, min_px=20)
    if len(tag_comps) != 1:
        return ABSTAIN
    _, gy0, gx0, gy1, gx1 = tag_comps[0]
    tag_x = (gx0 + gx1) / 2.0
    left_x, right_x = min(above + below), max(above + below)
    d_left = abs(tag_x - left_x)
    d_right = abs(tag_x - right_x)
    if abs(d_left - d_right) < 6.0:
        return ABSTAIN
    walk = faces if d_left < d_right else list(reversed(faces))

    alternating = all(walk[i] != walk[i + 1] for i in range(len(walk) - 1))
    return f"{'alternating' if alternating else 'broken'}|{walk[0]}"
