"""Independent image-only inverse for paired seam correspondence."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


def _foreground(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    return np.max(np.abs(array - 255), axis=2) > 30


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found: list[np.ndarray] = []
    for y, x in np.argwhere(mask):
        iy, ix = int(y), int(x)
        if seen[iy, ix]:
            continue
        stack = [(iy, ix)]
        seen[iy, ix] = True
        points: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny in range(cy - 1, cy + 2):
                for nx in range(cx - 1, cx + 2):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and mask[ny, nx]
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        found.append(np.asarray(points, dtype=np.int32))
    return sorted(found, key=len, reverse=True)


def _component_mask(component: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    result[component[:, 0], component[:, 1]] = True
    return result


def _box(component: np.ndarray) -> tuple[int, int, int, int]:
    y0 = int(component[:, 0].min())
    y1 = int(component[:, 0].max())
    x0 = int(component[:, 1].min())
    x1 = int(component[:, 1].max())
    return x0, y0, x1, y1


def _edge_trace(component: np.ndarray, shape: tuple[int, int], side: str) -> np.ndarray:
    mask = _component_mask(component, shape)
    x0, y0, x1, y1 = _box(component)
    height = y1 - y0 + 1
    trace: list[float] = []
    for fraction in np.linspace(0.08, 0.92, 25):
        target = int(round(y0 + fraction * (height - 1)))
        candidates: list[int] = []
        for row in range(max(y0, target - 2), min(y1, target + 2) + 1):
            xs = np.flatnonzero(mask[row])
            if len(xs):
                candidates.extend(int(value) for value in xs)
        if not candidates:
            raise ValueError("tile edge is incomplete")
        trace.append(float(max(candidates) if side == "right" else min(candidates)))
    return np.asarray(trace, dtype=np.float64)


def _profile_residual(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left - float(left.mean())
    right_centered = right - float(right.mean())
    return float(np.mean(np.abs(left_centered - right_centered)))


def decision_from_image(image: Image.Image) -> str:
    """Recover the answer from only the rasterized foreground geometry."""
    mask = _foreground(image)
    components = _components(mask)
    substantial = [component for component in components if len(component) >= 9000]
    if len(substantial) != 2:
        raise ValueError("expected two intact tile components")

    first, second = sorted(substantial[:2], key=lambda item: float(item[:, 1].mean()))
    left_box = _box(first)
    right_box = _box(second)
    left_width = left_box[2] - left_box[0] + 1
    right_width = right_box[2] - right_box[0] + 1
    left_height = left_box[3] - left_box[1] + 1
    right_height = right_box[3] - right_box[1] + 1
    if min(left_width, right_width, left_height, right_height) < 80:
        raise ValueError("tile components are too small")
    if abs(left_height - right_height) > 5:
        raise ValueError("tile heights are not aligned")
    if right_box[0] - left_box[2] < 12:
        raise ValueError("tile gap is not visibly separated")

    left_fill = len(first) / float(left_width * left_height)
    right_fill = len(second) / float(right_width * right_height)
    if min(left_fill, right_fill) < 0.70:
        raise ValueError("one tile has been materially erased")
    area_ratio = max(len(first), len(second)) / float(min(len(first), len(second)))
    if area_ratio > 1.18:
        raise ValueError("the two tile supports are imbalanced")

    left_trace = _edge_trace(first, mask.shape, "right")
    right_trace = _edge_trace(second, mask.shape, "left")
    residual = _profile_residual(left_trace, right_trace)
    if not math.isfinite(residual):
        raise ValueError("profile residual is not finite")
    return "yes" if residual < 4.0 else "no"
