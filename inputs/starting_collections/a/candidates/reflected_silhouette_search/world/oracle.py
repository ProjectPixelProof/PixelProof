"""Independent pixel-only inverse for reflected silhouette pairs.

Only the final RGB raster is accepted.  No renderer constants, scene fields, or
analytic answer are imported.  The inverse segments saturated ink, extracts two
components in each public sector, builds the visible perpendicular-bisector axis
from their centroids, and scores bidirectional silhouette overlap after reflection.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np
from PIL import Image


_SECTORS = ("NW", "NE", "SW", "SE")
_MIN_AREA = 500
_MATCH_RADIUS = 3
_MIN_WINNER_MARGIN = 0.045


def _ink_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    spread = array.max(axis=2) - array.min(axis=2)
    # The faint panel frames are neutral gray; the glyph ink is saturated.
    return (spread > 34.0) & (array.min(axis=2) < 190.0)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for y0, x0 in np.argwhere(mask):
        y0, x0 = int(y0), int(x0)
        if seen[y0, x0]:
            continue
        queue: deque[tuple[int, int]] = deque([(y0, x0)])
        seen[y0, x0] = True
        points: list[tuple[int, int]] = []
        while queue:
            y, x = queue.popleft()
            points.append((y, x))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if len(points) >= _MIN_AREA:
            components.append(np.asarray(points, dtype=np.float64))
    return components


def _centroid(points_yx: np.ndarray) -> np.ndarray:
    # Return x,y to make the reflection equations read naturally.
    return np.asarray([points_yx[:, 1].mean(), points_yx[:, 0].mean()], dtype=np.float64)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            output |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return output


def _reflected(points_xy: np.ndarray, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    direction = second - first
    length = float(np.linalg.norm(direction))
    if length < 40.0:
        raise ValueError("component centers are too close")
    unit = direction / length
    midpoint = (first + second) / 2.0
    offsets = points_xy - midpoint
    # Reflection across the line perpendicular to center-to-center direction.
    return points_xy - 2.0 * np.outer(offsets @ unit, unit)


def _one_way_match(points_xy: np.ndarray, target_mask: np.ndarray, source_mask: np.ndarray) -> float:
    reflected = points_xy
    rounded = np.rint(reflected).astype(np.int64)
    valid = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < target_mask.shape[1])
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < target_mask.shape[0])
    )
    hits = np.zeros(len(rounded), dtype=bool)
    ys = rounded[valid, 1]
    xs = rounded[valid, 0]
    hits[valid] = target_mask[ys, xs]
    return float(hits.mean()) if len(hits) else 0.0


def _pair_score(first_yx: np.ndarray, second_yx: np.ndarray) -> float:
    first_xy = first_yx[:, ::-1]
    second_xy = second_yx[:, ::-1]
    first_center = _centroid(first_yx)
    second_center = _centroid(second_yx)
    reflected_first = _reflected(first_xy, first_center, second_center)
    reflected_second = _reflected(second_xy, second_center, first_center)
    height = 512
    width = 512
    first_mask = np.zeros((height, width), dtype=bool)
    second_mask = np.zeros((height, width), dtype=bool)
    first_mask[first_yx[:, 0].astype(int), first_yx[:, 1].astype(int)] = True
    second_mask[second_yx[:, 0].astype(int), second_yx[:, 1].astype(int)] = True
    first_support = _dilate(first_mask, _MATCH_RADIUS)
    second_support = _dilate(second_mask, _MATCH_RADIUS)
    # Use a small dilation for antialias/raster rounding, but score all pixels.
    one = _one_way_match(reflected_first, second_support, second_support)
    two = _one_way_match(reflected_second, first_support, first_support)
    return 1.0 - 0.5 * (one + two)


def decision_from_image(image: Image.Image) -> str:
    """Return the unique reflected-pair sector, or abstain by raising ValueError."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    mask = _ink_mask(rgb)
    all_components = _components(mask)
    by_sector: dict[str, list[np.ndarray]] = {sector: [] for sector in _SECTORS}
    for component in all_components:
        center = _centroid(component)
        column = "E" if center[0] >= width / 2 else "W"
        row = "S" if center[1] >= height / 2 else "N"
        by_sector[row + column].append(component)
    scores: dict[str, float] = {}
    # Destructive ablations of irrelevant sectors are intentionally tolerated:
    # a complete local pair is still a valid measurable candidate.  Removing
    # either member of the true pair removes that candidate, so the returned
    # winner changes or the inverse abstains.  This is what makes the causal
    # support multipart rather than the entire eight-glyph gallery.
    for sector, components in by_sector.items():
        if len(components) == 2:
            scores[sector] = _pair_score(components[0], components[1])
    ranked = sorted((score, sector) for sector, score in scores.items())
    if not ranked:
        raise ValueError("no complete glyph pair remains")
    if ranked[0][0] > 0.12:
        raise ValueError(f"no low-mismatch pair remains: {ranked}")
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < _MIN_WINNER_MARGIN:
        raise ValueError(f"reflection winner is ambiguous: {ranked}")
    return ranked[0][1]
