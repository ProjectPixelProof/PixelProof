"""Standalone pixel-only inverse arm for texture_flow_orientation.

Receives only a PIL RGB image. Recovers every dark stroke as a connected
component, fits each stroke's principal (long) axis by PCA of its pixel
coordinates, classifies each stroke into one of four orientation buckets
(horizontal / rising / vertical / falling), and answers with the bucket that
dominates the whole field. Returns the sentinel "ABSTAIN" when the image does
not contain a usable stroke field.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

MIN_COMPONENTS = 8
_BUCKET_CENTER = (0.0, 45.0, 90.0, 135.0)
_CLASSES = ("horizontal", "rising", "vertical", "falling")


def _category(angle_deg: float) -> str:
    a = angle_deg % 180.0
    best = 0
    best_dist = 1e18
    for i, center in enumerate(_BUCKET_CENTER):
        d = abs((a - center + 180.0) % 180.0)
        if d > 90.0:
            d = 180.0 - d
        if d < best_dist:
            best_dist = d
            best = i
    return _CLASSES[best]


def _components(mask):
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return []
    parent = {}

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (y, x) in zip(ys.tolist(), xs.tolist()):
        key = (y, x)
        parent[key] = key
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if (ny, nx) in parent:
                union((ny, nx), key)

    comp = {}
    for (y, x) in zip(ys.tolist(), xs.tolist()):
        root = find((y, x))
        comp.setdefault(root, []).append((y, x))
    return list(comp.values())


def _stroke_angle(points):
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        return None
    mu = pts.mean(axis=0)
    centered = pts - mu
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    angle = np.degrees(np.arctan2(principal[0], principal[1])) % 180.0
    return float(angle)


def decision_from_image(image: Image.Image) -> str:
    image = image.convert("RGB")
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        return "ABSTAIN"
    gray = arr.astype(np.float64).mean(axis=2)
    mask = gray < 128.0
    comps = _components(mask)
    if len(comps) < MIN_COMPONENTS:
        return "ABSTAIN"

    angles = [_stroke_angle(points) for points in comps if _stroke_angle(points) is not None]
    if len(angles) < MIN_COMPONENTS:
        return "ABSTAIN"

    counts = {cls: 0 for cls in _CLASSES}
    for a in angles:
        counts[_category(a)] += 1
    return max(_CLASSES, key=lambda cls: counts[cls])
