"""Independent pixel-only inverse arm for the spring coil count."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Return the largest 4-connected component without renderer knowledge."""
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y, x in zip(*np.nonzero(mask), strict=True):
        if seen[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        while queue:
            yy, xx = queue.popleft()
            pixels.append((yy, xx))
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = yy + dy, xx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if len(pixels) > len(best):
            best = pixels
    if len(best) < 1000:
        raise ValueError("no substantial connected blue wire")
    component = np.zeros_like(mask, dtype=bool)
    yy, xx = zip(*best, strict=True)
    component[np.array(yy), np.array(xx)] = True
    return component


def decision_from_image(image: Image.Image) -> str:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    # Hue/chroma test includes both the bright face and dark blue edge, but not gray mounts.
    blue_mask = (blue >= 88) & (blue - red >= 42) & (blue - green >= 8)
    component = _largest_component(blue_mask)
    ys, xs = np.nonzero(component)
    x0, x1 = int(xs.min()), int(xs.max())
    if x1 - x0 < image.width * 0.68:
        raise ValueError("wire does not span the image")

    centerline = np.full(x1 - x0 + 1, np.nan, dtype=float)
    for offset, x in enumerate(range(x0, x1 + 1)):
        column_y = np.nonzero(component[:, x])[0]
        if len(column_y):
            centerline[offset] = float(np.median(column_y))
    # This strict continuity check is also the evidence-erasure boundary: removing
    # any routed interval makes the inverse arm abstain instead of guessing a count.
    if np.isnan(centerline).any():
        raise ValueError("blue route is interrupted")

    kernel = np.ones(9, dtype=float) / 9.0
    smooth = np.convolve(np.pad(centerline, (4, 4), mode="edge"), kernel, mode="valid")
    low, high = float(np.percentile(smooth, 4)), float(np.percentile(smooth, 96))
    amplitude = (high - low) / 2.0
    if amplitude < image.height * 0.10:
        raise ValueError("vertical coil amplitude is too small")
    midpoint = (low + high) / 2.0

    bottom = smooth > midpoint + 0.30 * amplitude
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(bottom.tolist() + [False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= 8:
                runs.append((start, index))
            start = None
    count = len(runs)
    if count < 2 or count > 9:
        raise ValueError("implausible full-coil count")

    widths = [end - start for start, end in runs]
    gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
    if min(widths) < 10 or (gaps and min(gaps) < 16):
        raise ValueError("coil-count runner-up margin is too small")
    # Both visible endpoints must occupy the same top phase. This prevents a
    # partial end cycle from being silently rounded into the count.
    if smooth[0] > midpoint - 0.45 * amplitude or smooth[-1] > midpoint - 0.45 * amplitude:
        raise ValueError("endpoint phases do not expose complete cycles")
    return str(count)

