"""Global-elongation world: recover the principal stretch axis of a dot cloud.

A single cloud of equal-sized dark dots fills the white canvas. The cloud as a
whole is elongated along exactly one of four principal directions: horizontal,
rising (uphill to the right), vertical, or falling (downhill to the right).
The decision is the global Gestalt property recovered from the arrangement: the
principal axis along which the whole scatter cloud is stretched out.

The renderer owns scene sampling + rasterization + the structural analytic
gold. The independent pixel-only inverse arm (``oracle.py``) recovers the same
elongation axis from the final PNG alone by computing the principal variance
axis of the ink mass.

The reconstruction is deliberately monotone: the answer depends on the
second-moment (variance) structure of the whole cloud, never on a single dot,
the ink count, the ink colour, the total ink, or the cloud centre position.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

WIDTH = 512
HEIGHT = 512
BACKGROUND = (255, 255, 255)
INK = (45, 65, 150)  # single dark blue ink; colour never cues the answer.

DOT_R = 5
N_LO = 68
N_HI = 92
SIGMA_MAJOR = 72.0
ELONGATION = 3.4
CENTER_OFFSET = 32.0
ANGLE_JITTER = 3.5
NUM_DOTS = 36  # not a cue; dots are sampled continuously

# Class centres (degrees) and the boundaries between them.
CLASS_CENTERS = {"horizontal": 2.0, "rising": 45.0, "vertical": 90.0, "falling": 135.0}
_BOUNDS = (22.5, 67.5, 112.5, 157.5)
_QUARANTINE_DEG = 5.0
_MIN_RATIO = 1.8


def _ang_dist(a: float, b: float) -> float:
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def sample_scene(seed: int) -> dict:
    """Sample one latent scene deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    classes = list(CLASS_CENTERS)
    axis = classes[int(rng.integers(0, len(classes)))]
    center = CLASS_CENTERS[axis] + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)
    n = int(rng.integers(N_LO, N_HI + 1))
    offset_x = rng.uniform(-CENTER_OFFSET, CENTER_OFFSET)
    offset_y = rng.uniform(-CENTER_OFFSET, CENTER_OFFSET)

    sigma_minor = SIGMA_MAJOR / ELONGATION
    base = rng.normal(0.0, 1.0, (n, 2)) * np.array([SIGMA_MAJOR, sigma_minor])
    theta = np.radians(center)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = base @ rot.T
    cx = WIDTH / 2.0 + offset_x
    cy = HEIGHT / 2.0 + offset_y
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts = np.clip(pts, 0.0, float(WIDTH - 1))

    dots = [[round(float(x), 3), round(float(y), 3)] for x, y in pts]
    return {"dots": dots, "axis": axis, "angle": float(round(center, 4))}


def render(scene: dict) -> Image.Image:
    """Rasterize the latent scene (a deterministic PNG)."""
    img = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND)
    d = ImageDraw.Draw(img)
    for x, y in scene["dots"]:
        d.ellipse([x - DOT_R, y - DOT_R, x + DOT_R, y + DOT_R], fill=INK)
    return img


def principal_angle(points: list) -> float:
    """Principal variance axis of a point set, in degrees in [0, 180)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or len(pts) < 3:
        return 0.0
    c = pts - pts.mean(axis=0)
    if c.shape[0] < 2:
        return 0.0
    cov = np.cov(c, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]])
    elif cov.shape != (2, 2):
        return 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    v = eigvecs[:, int(np.argmax(eigvals))]
    ang = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    return ang


def _classify(theta: float) -> str:
    if theta < 22.5 or theta >= 157.5:
        return "horizontal"
    if theta < 67.5:
        return "rising"
    if theta < 112.5:
        return "vertical"
    return "falling"


def elongation_ratio(scene: dict) -> float:
    pts = np.asarray(scene["dots"], dtype=float)
    if len(pts) < 3:
        return 1.0
    c = pts - pts.mean(axis=0)
    cov = np.cov(c, rowvar=False)
    if np.ndim(cov) == 0:
        return 1.0
    eigvals = np.linalg.eigvalsh(np.atleast_2d(cov))
    if eigvals.size < 2:
        return 1.0
    return float(max(eigvals) / min(eigvals)) if min(eigvals) > 1e-9 else float("inf")


def analytic_gold(scene: dict) -> str:
    """Structural gold: the principal elongation axis of the cloud."""
    return _classify(principal_angle(scene["dots"]))


def margin(scene: dict) -> float:
    """Commitment magnitude: angular gap (degrees) to the nearest axis boundary."""
    theta = principal_angle(scene["dots"])
    return float(min(_ang_dist(theta, b) for b in _BOUNDS))


def is_quarantined(scene: dict) -> bool:
    """Quarantine near-boundary or non-elongated (circular / erased) clouds."""
    if elongation_ratio(scene) < _MIN_RATIO:
        return True
    return margin(scene) < _QUARANTINE_DEG


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """The scene is injectively rendered; no latent aliases are declared."""
    return []
