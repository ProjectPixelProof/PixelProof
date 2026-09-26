"""Independent inverse arm using only final RGB pixels and the public contract."""

from __future__ import annotations

import numpy as np
from PIL import Image

TEMPLATES = {
    "X": (1, 0, 1, 0, 1, 0, 1, 0, 1),
    "plus": (0, 1, 0, 1, 1, 1, 0, 1, 0),
    "T": (1, 1, 1, 0, 1, 0, 0, 1, 0),
    "L": (1, 0, 0, 1, 0, 0, 1, 1, 1),
}


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    groups = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        points = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        groups.append(np.asarray(points, dtype=np.int32))
    return groups


def _cap_centroid(arr: np.ndarray, kind: str) -> tuple[float, float]:
    red, green, blue = (arr[..., index].astype(np.int16) for index in range(3))
    if kind == "amber":
        mask = (red > 170) & (green > 90) & (green < 205) & (blue < 100) & ((red-green) > 35)
    else:
        mask = (green > 115) & (blue > 110) & (red < 100) & (np.abs(green-blue) < 65)
    groups = [group for group in _components(mask) if len(group) >= 35]
    if len(groups) != 1:
        raise ValueError(f"missing or ambiguous {kind} route cap")
    y, x = groups[0].mean(axis=0)
    return float(x), float(y)


def measure_from_image(image: Image.Image) -> dict:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    # All allowed route inks are dark in every RGB channel; frames, caps, and
    # nuisance marks are deliberately outside this public segmentation band.
    dark = np.max(arr, axis=2) < 115
    groups = [group for group in _components(dark) if len(group) >= 80]
    if len(groups) != 1:
        raise ValueError("route is missing, interrupted, or not a single connected trace")
    route = groups[0]
    ys = route[:, 0].astype(float)
    xs = route[:, 1].astype(float)
    x_min, x_max = float(xs.min()), float(xs.max())
    span = x_max - x_min
    if not 70.0 <= span <= 110.0 or (ys.max() - ys.min()) < 16.0:
        raise ValueError("route extent is inconsistent with nine readable teeth")

    amber_x, _ = _cap_centroid(arr, "amber")
    teal_x, _ = _cap_centroid(arr, "teal")
    if abs(amber_x - teal_x) < 60.0:
        raise ValueError("route direction is ambiguous")

    # Every declared glyph has five up and four down teeth of equal visible
    # length. The route-pixel median is therefore a robust, image-only estimate
    # of their shared baseline, unaffected by the differently shaped caps.
    baseline = float(np.median(ys))
    decoded = []
    clearances = []
    for index in range(9):
        center = x_min + (index + 0.5) * span / 9.0
        half_window = span / 45.0
        sample = ys[np.abs(xs - center) <= half_window]
        if len(sample) < 5:
            raise ValueError("a route tooth is missing")
        offset = baseline - float(np.median(sample))
        clearances.append(abs(offset))
        if abs(offset) < 3.5:
            raise ValueError("a route tooth has ambiguous vertical sign")
        decoded.append(1 if offset > 0 else 0)
    if amber_x > teal_x:
        decoded.reverse()

    scored = sorted(
        (sum(a != b for a, b in zip(decoded, template)), name)
        for name, template in TEMPLATES.items()
    )
    if scored[0][0] != 0 or scored[1][0] - scored[0][0] < 2:
        raise ValueError("decoded raster is not an unambiguous declared symbol")
    return {
        "decision": scored[0][1],
        "bits": tuple(decoded),
        "runner_up_margin": float(scored[1][0] - scored[0][0]),
        "minimum_tooth_clearance": float(min(clearances)),
    }


def decision_from_image(image: Image.Image) -> str:
    return str(measure_from_image(image)["decision"])
