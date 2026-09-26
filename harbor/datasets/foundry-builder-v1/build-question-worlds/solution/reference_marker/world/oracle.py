from __future__ import annotations

import numpy as np
from PIL import Image


def decision_from_image(image: Image.Image) -> str:
    pixels = np.asarray(image.convert("RGB"))
    red = (pixels[:, :, 0] > 180) & (pixels[:, :, 1] < 80) & (pixels[:, :, 2] < 80)
    gray = (
        (pixels[:, :, 0] >= 80)
        & (pixels[:, :, 0] <= 120)
        & (pixels[:, :, 1] >= 80)
        & (pixels[:, :, 1] <= 120)
        & (pixels[:, :, 2] >= 80)
        & (pixels[:, :, 2] <= 120)
    )
    if not red.any() or not gray.any():
        raise ValueError("required marker or reference pixels are missing")
    marker_y = float(np.nonzero(red)[0].mean())
    reference_y = float(np.nonzero(gray)[0].mean())
    return "yes" if marker_y < reference_y else "no"
