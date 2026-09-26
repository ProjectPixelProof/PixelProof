"""Independent pixel-only inverse arm for the radial-density-regime world.

Recovers the decision (dots packed toward the centre core, or toward the outer
rim) from the final rasterized PNG alone. It detects the dark dot ink, counts
how much of that ink lies inside versus outside a central disc centred on the
canvas, and decides by majority.

This module imports no renderer / prompts / generate / verify / analytic-gold
code and never receives the latent scene.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

_BOUNDARY = 128.0


def _dark_mask(arr: np.ndarray) -> np.ndarray:
    """Logical mask of the dark dot ink (clearly non-background pixels)."""
    r = arr[..., 0].astype(int)
    g = arr[..., 1].astype(int)
    b = arr[..., 2].astype(int)
    return (r < 150) & (g < 150) & (b < 160)


def decision_from_image(image: Image.Image) -> str:
    """Return 'core' if the dark ink is denser toward the centre, else 'rim'."""
    arr = np.asarray(image.convert("RGB"))
    mask = _dark_mask(arr)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return "rim"
    h, w = arr.shape[:2]
    cx = w / 2.0
    cy = h / 2.0
    dr = np.hypot(xs - cx, ys - cy)
    inner = int((dr <= _BOUNDARY).sum())
    outer = int((dr > _BOUNDARY).sum())
    return "core" if inner >= outer else "rim"
