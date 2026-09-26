"""Independent pixel-only inverse for the colored arc-curvature world.

The inverse sees only the final RGB image.  It segments the four public palette
colors inside each labeled panel, estimates each open stroke's endpoint chord
from a PCA axis, measures the median centerline departure from that chord, and
compares the resulting color-to-curvature ranks.  It does not import or inspect
the renderer, latent scene, prompt, generator, or analytic gold.
"""

from __future__ import annotations

import math

import numpy as np
from PIL import Image


ORACLE_VERSION = "colored-arc-curvature-oracle-0.1.0"

COLORS = {
    "coral": (214, 70, 77),
    "gold": (225, 153, 34),
    "teal": (28, 153, 145),
    "indigo": (71, 94, 190),
}
PANEL_BOXES = (
    (24, 64, 304, 528),
    (340, 64, 620, 528),
    (656, 64, 936, 528),
)


def _mask_pixels(array: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    target = np.asarray(color, dtype=np.int32)
    delta = array.astype(np.int32) - target
    distance_sq = np.sum(delta * delta, axis=2)
    return distance_sq <= 52 * 52


def _panel_array(array: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    # The crop excludes the public title/subtitle while retaining every arc.
    return array[top + 70 : bottom - 12, left + 8 : right - 8]


def _bow_ratio(points: np.ndarray) -> float | None:
    """Estimate absolute centerline sagitta divided by endpoint-chord length."""

    if points.shape[0] < 90:
        return None
    centered = points - np.mean(points, axis=0, keepdims=True)
    covariance = np.cov(centered.T)
    if not np.all(np.isfinite(covariance)):
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = centered @ axis
    lo, hi = np.percentile(projections, (2.0, 98.0))
    span = float(hi - lo)
    if span < 18.0:
        return None
    edge_band = max(3.0, span * 0.09)
    first = points[projections <= lo + edge_band]
    last = points[projections >= hi - edge_band]
    if first.shape[0] < 8 or last.shape[0] < 8:
        return None
    endpoint0 = np.median(first, axis=0)
    endpoint1 = np.median(last, axis=0)
    chord = endpoint1 - endpoint0
    chord_length = float(np.linalg.norm(chord))
    if chord_length < 20.0:
        return None

    relative = points - endpoint0
    along = relative @ chord / (chord_length * chord_length)
    signed_perpendicular = (chord[0] * relative[:, 1] - chord[1] * relative[:, 0]) / chord_length

    medians: list[float] = []
    for lower, upper in zip(np.linspace(0.12, 0.84, 19), np.linspace(0.16, 0.88, 19), strict=True):
        selected = signed_perpendicular[(along >= lower) & (along < upper)]
        if selected.size >= 8:
            medians.append(float(np.median(selected)))
    if len(medians) < 8:
        return None
    bow = max(abs(value) for value in medians)
    ratio = bow / chord_length
    return float(ratio) if math.isfinite(ratio) else None


def _panel_signature(panel: np.ndarray) -> dict[str, int] | None:
    values: dict[str, float] = {}
    for name, color in COLORS.items():
        mask = _mask_pixels(panel, color)
        ys, xs = np.nonzero(mask)
        if len(xs) < 90:
            return None
        points = np.column_stack((xs.astype(float), ys.astype(float)))
        bow = _bow_ratio(points)
        if bow is None:
            return None
        values[name] = bow
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    gaps = [ordered[index + 1][1] - ordered[index][1] for index in range(3)]
    if min(gaps) < 0.025:
        return None
    return {name: rank for rank, (name, _) in enumerate(ordered)}


def _mismatch(reference: dict[str, int], candidate: dict[str, int]) -> int:
    return sum(reference[name] != candidate[name] for name in COLORS)


def decision_from_image(image: Image.Image) -> str:
    """Return ``A``, ``B``, or ``quarantine`` using only image pixels."""

    array = np.asarray(image.convert("RGB"))
    signatures = [_panel_signature(_panel_array(array, box)) for box in PANEL_BOXES]
    if any(signature is None for signature in signatures):
        return "quarantine"
    reference, candidate_a, candidate_b = signatures
    assert reference is not None and candidate_a is not None and candidate_b is not None
    score_a = _mismatch(reference, candidate_a)
    score_b = _mismatch(reference, candidate_b)
    if score_a == score_b or min(score_a, score_b) != 0:
        return "quarantine"
    return "A" if score_a < score_b else "B"
