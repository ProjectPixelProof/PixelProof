"""Independent image-only oracle for the compact unmarked-spur route."""

from __future__ import annotations

import collections
import math
from PIL import Image
import numpy as np

_BG = np.array([250, 250, 248], dtype=np.float32)
_INK = np.array([38, 43, 51], dtype=np.float32)
_GREEN = np.array([38, 166, 91], dtype=np.float32)
_AMBER = np.array([226, 151, 35], dtype=np.float32)
_VIOLET = np.array([113, 83, 181], dtype=np.float32)


def _nearest(arr: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.linalg.norm(arr - color, axis=2) < 95.0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        q = [(y, x)]
        seen[y, x] = True
        comp = []
        while q:
            cy, cx = q.pop()
            comp.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        out.append(comp)
    return out


def _centroid(comp: list[tuple[int, int]]) -> tuple[float, float]:
    ys = np.asarray([p[0] for p in comp], dtype=float)
    xs = np.asarray([p[1] for p in comp], dtype=float)
    return (float(xs.mean()), float(ys.mean()))


def _thin(mask: np.ndarray) -> np.ndarray:
    """Vectorized Zhang-Suen thinning, independently coded for the route mask."""
    img = mask.copy().astype(bool)
    changed = True
    h, w = img.shape
    while changed:
        changed = False
        for phase in (0, 1):
            p = [
                img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:], img[2:, 2:],
                img[2:, 1:-1], img[2:, :-2], img[1:-1, :-2], img[:-2, :-2],
            ]
            b = sum(p)
            a = sum((~p[i]) & p[(i + 1) % 8] for i in range(8))
            base = img[1:-1, 1:-1] & (b >= 2) & (b <= 6) & (a == 1)
            if phase == 0:
                ok = ~(p[0] & p[2] & p[4]) & ~(p[2] & p[4] & p[6])
            else:
                ok = ~(p[0] & p[2] & p[6]) & ~(p[0] & p[4] & p[6])
            remove = base & ok
            if np.any(remove):
                changed = True
                inner = img[1:-1, 1:-1]
                inner[remove] = False
    return img


def _nearest_route(point: tuple[float, float], coords: np.ndarray) -> float:
    x, y = point
    return float(np.min((coords[:, 1] - y) ** 2 + (coords[:, 0] - x) ** 2) ** 0.5)


def decision_from_image(image: Image.Image) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    dark = np.linalg.norm(arr - _INK, axis=2) < 72.0
    dark_comps = [c for c in _components(dark) if len(c) > 500]
    if len(dark_comps) != 1:
        raise ValueError("route component is not uniquely visible")
    route = np.zeros(dark.shape, dtype=bool)
    for y, x in dark_comps[0]:
        route[y, x] = True
    # Colored center estimates come from their own masks, not from renderer state.
    markers = {}
    for name, color in (("green", _GREEN), ("amber", _AMBER), ("violet", _VIOLET)):
        comps = [c for c in _components(_nearest(arr, color)) if len(c) > 80]
        if len(comps) != 1:
            raise ValueError(f"missing or ambiguous {name} marker")
        markers[name] = _centroid(comps[0])

    # The anonymous dead-end is the route portion farthest from every colored
    # endpoint. The endpoint radii are balanced while the spur is intentionally
    # longer, making this a visible protrusion test independent of coordinates.
    coords = np.argwhere(route)
    marker_array = np.asarray(list(markers.values()), dtype=float)
    nearest_marker_distance = np.min(
        np.sqrt(
            (coords[:, 1, None] - marker_array[None, :, 0]) ** 2
            + (coords[:, 0, None] - marker_array[None, :, 1]) ** 2
        ),
        axis=1,
    )
    cutoff = float(np.percentile(nearest_marker_distance, 99.5))
    far = coords[nearest_marker_distance >= cutoff]
    if len(far) < 12:
        raise ValueError("anonymous endpoint has no stable support")
    spur_end = np.mean(far, axis=0)

    # The three colored centers form a visible, translation/rotation-invariant
    # estimate of the fork's local neighborhood. Compare spur direction with
    # the two terminal directions and require a large decision margin.
    hub = marker_array.mean(axis=0)
    hy, hx = float(hub[1]), float(hub[0])
    spur_vec = np.array([spur_end[1] - hx, spur_end[0] - hy])
    norms = np.linalg.norm(spur_vec)
    if norms < 20:
        raise ValueError("spur arm has no stable direction")
    scores = {}
    for name in ("amber", "violet"):
        vec = np.array([markers[name][0] - hx, markers[name][1] - hy])
        vnorm = np.linalg.norm(vec)
        scores[name] = float(np.dot(spur_vec, vec) / (norms * vnorm))
    if abs(scores["amber"] - scores["violet"]) < 0.28:
        raise ValueError("arm ownership margin is ambiguous")
    return "yes" if scores["amber"] > scores["violet"] else "no"
