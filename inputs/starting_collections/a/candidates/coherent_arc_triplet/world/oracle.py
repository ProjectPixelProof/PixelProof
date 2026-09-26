"""Independent PNG-only inverse arm for collective circular continuation."""

from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


class OracleAbstention(ValueError):
    """The raster does not expose three clean measurable contour components."""


def _ink_mask(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.int16)
    chroma = array.max(axis=2) - array.min(axis=2)
    darkness = 255 - array.mean(axis=2)
    return (chroma >= 42) & (darkness >= 35)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    output: list[np.ndarray] = []
    for start_y, start_x in np.argwhere(mask):
        sy, sx = int(start_y), int(start_x)
        if seen[sy, sx]:
            continue
        queue: deque[tuple[int, int]] = deque([(sy, sx)])
        seen[sy, sx] = True
        points: list[tuple[float, float]] = []
        while queue:
            y, x = queue.popleft()
            points.append((float(x), float(y)))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dx or dy):
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        if len(points) >= 45:
            output.append(np.asarray(points, dtype=np.float64))
    return sorted(output, key=lambda points: float(points[:, 0].mean()))


def _centerline(component: np.ndarray) -> np.ndarray:
    origin = component.mean(axis=0)
    covariance = np.cov((component - origin).T)
    values, vectors = np.linalg.eigh(covariance)
    tangent = vectors[:, int(np.argmax(values))]
    normal = np.asarray([-tangent[1], tangent[0]])
    local = component - origin
    along = local @ tangent
    across = local @ normal
    # Median across the stroke in narrow one-pixel tangent bins estimates the
    # drawn centerline without renderer parameters or palette knowledge.
    bins = np.rint(along).astype(int)
    samples: list[tuple[float, float]] = []
    for key in range(int(bins.min()), int(bins.max()) + 1):
        selected = bins == key
        if int(selected.sum()) >= 2:
            samples.append((float(np.median(along[selected])), float(np.median(across[selected]))))
    if len(samples) < 22:
        raise OracleAbstention("a curve fragment is too incomplete to fit")
    local_line = np.asarray(samples, dtype=np.float64)
    return origin + local_line[:, :1] * tangent + local_line[:, 1:] * normal


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = points[:, 0]
    y = points[:, 1]
    design = np.column_stack((x, y, np.ones(len(points))))
    target = -(x * x + y * y)
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    center = -0.5 * coefficients[:2]
    radius_sq = float(center @ center - coefficients[2])
    if not np.isfinite(radius_sq) or radius_sq <= 100.0:
        raise OracleAbstention("circle fit is degenerate")
    radius = float(np.sqrt(radius_sq))
    radial = np.linalg.norm(points - center, axis=1)
    residual = float(np.sqrt(np.mean((radial - radius) ** 2)))
    if not np.isfinite(radius) or radius < 35.0 or radius > 360.0 or residual > 2.2:
        raise OracleAbstention("curve fragment has no stable circular continuation")
    return center, radius, residual


def _collective_score(image: Image.Image) -> float:
    components = _components(_ink_mask(image))
    if len(components) != 3:
        raise OracleAbstention(f"expected three contour fragments, found {len(components)}")
    fits = [_fit_circle(_centerline(component)) for component in components]
    centers = np.asarray([item[0] for item in fits])
    radii = np.asarray([item[1] for item in fits])
    median_radius = float(np.median(radii))
    pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    center_disagreement = float(pairwise.max() / median_radius)
    radius_disagreement = float((radii.max() - radii.min()) / median_radius)
    return center_disagreement + 0.35 * radius_disagreement


def decision_from_image(image: Image.Image) -> str:
    """Return the common-circle decision from final pixels and nothing else."""

    score = _collective_score(image)
    # Calibration leaves a wide empty band: coherent scenes score below 0.45,
    # while a fully reversed fragment scores above 1.45.  Interventions landing
    # in the band are explicitly treated as ambiguous rather than guessed.
    if score <= 0.70:
        return "yes"
    if score >= 1.10:
        return "no"
    raise OracleAbstention("collective continuation residual is ambiguous")
