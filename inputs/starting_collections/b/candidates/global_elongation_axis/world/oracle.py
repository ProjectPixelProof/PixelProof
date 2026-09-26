"""Independent pixel-only inverse arm for the global-elongation world.

Recovers the decision (along which direction is the whole dot cloud stretched?)
from the final rasterized PNG alone. It detects the ink pixels itself (any
saturated, non-white pixel), treats the cloud as a set of equal-mass marks, and
computes the principal variance axis of the ink mass. Because the dots are all
the same radius, each mark contributes the same isotropic smear, which does not
rotate the principal axis; the recovered axis therefore matches the latent
cloud's principal stretch direction.

This module imports no renderer / prompts / generate / verify / analytic-gold
code and never receives the latent scene.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_BG_EDGE = 235
_MIN_INK = 30


def _ink_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    return (r < _BG_EDGE) & (g < _BG_EDGE) & (b < _BG_EDGE)


def _principal_angle(arr: np.ndarray) -> float:
    mask = _ink_mask(arr)
    ys, xs = np.nonzero(mask)
    if xs.size < _MIN_INK:
        return 0.0
    pts = np.stack([xs, ys], axis=1).astype(float)
    c = pts - pts.mean(axis=0)
    cov = np.cov(c, rowvar=False)
    if np.ndim(cov) == 0:
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    return float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)


def decision_from_image(image: Image.Image) -> str:
    """Return the principal elongation axis among horizontal/rising/vertical/falling."""
    arr = np.asarray(image.convert("RGB")).astype(int)
    theta = _principal_angle(arr)
    if theta < 22.5 or theta >= 157.5:
        return "horizontal"
    if theta < 67.5:
        return "rising"
    if theta < 112.5:
        return "vertical"
    return "falling"
