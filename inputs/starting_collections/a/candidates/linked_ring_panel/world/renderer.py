"""Interlocked-ring panel world.

Five framed panels each hold two opaque rings that cross at exactly two places.
Where they cross, the ring drawn later hides a short piece of the other one. In
exactly one panel the pair is INTERLOCKED: the under ring is repainted over the
top ring at one of the two crossings, so each ring shows exactly one break and
each stays a single connected curve. Everywhere else the pair is merely STACKED:
the same ring is in front at both crossings, so the ring behind is broken twice
and falls into two pieces.

Latent state z = {canvas, palette offset, per-panel ring pair, draw order,
linked flag, patched-crossing side}. The analytic gold reads only the linked
flags; the runner-up margin is the length in pixels of the shortest surviving
fragment of a stacked panel's under ring, i.e. how plainly the weakest
distractor advertises its second break.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
from PIL import Image

RENDERER_VERSION = "linked_ring_panel-0.1.0"

PANELS = 5
PANEL_W = 200
PANEL_H = 200
GAP = 10
MARGIN_PX = 10
FRAME = 2
INNER_PAD = 5
CANVAS_W = 2 * MARGIN_PX + PANELS * PANEL_W + (PANELS - 1) * GAP
CANVAS_H = 2 * MARGIN_PX + PANEL_H
STROKE = 7.0
PATCH_ARC_PX = 16.0

BG_RGB = (255, 255, 255)
FRAME_RGB = (128, 128, 128)
PALETTE = [
    (198, 32, 40),
    (28, 84, 190),
    (24, 132, 64),
    (150, 52, 168),
    (198, 110, 16),
    (20, 140, 160),
]

MIN_FRAGMENT_PX = 26.0
QUARANTINE_MARGIN = 8.0
R_MIN = 33
R_MAX = 45


def panel_box(index: int) -> tuple[int, int, int, int]:
    """Outer pixel box (x0, y0, x1, y1) of panel ``index``, x1/y1 exclusive."""
    x0 = MARGIN_PX + index * (PANEL_W + GAP)
    return (x0, MARGIN_PX, x0 + PANEL_W, MARGIN_PX + PANEL_H)


def panel_inner(index: int) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = panel_box(index)
    pad = FRAME + INNER_PAD
    return (x0 + pad, y0 + pad, x1 - pad, y1 - pad)


def _order(panel: dict) -> tuple[dict, dict]:
    """(under ring, over ring): the over ring is painted second, hiding the other."""
    rings = panel["rings"]
    top = int(panel["top"])
    return rings[1 - top], rings[top]


def _crossing(under: dict, over: dict) -> dict | None:
    """Pixel geometry of the two crossings, measured from the under ring.

    Returns the crossing angle between the two curves, the arc length of the
    under ring that the over ring hides at one crossing (``cut``), the two
    crossing angles as seen from the under ring's centre, and the length of the
    shorter surviving fragment of the under ring when it is hidden at BOTH
    crossings (the stacked case).
    """
    ux, uy, ur = float(under["cx"]), float(under["cy"]), float(under["r"])
    ox, oy, orad = float(over["cx"]), float(over["cy"]), float(over["r"])
    d = math.hypot(ox - ux, oy - uy)
    if not (abs(ur - orad) < d < ur + orad):
        return None
    cos_theta = (ur * ur + orad * orad - d * d) / (2.0 * ur * orad)
    theta = math.acos(max(-1.0, min(1.0, cos_theta)))
    sin_theta = math.sin(theta)
    if sin_theta < 1e-6:
        return None
    cut = STROKE / sin_theta
    cos_alpha = (d * d + ur * ur - orad * orad) / (2.0 * d * ur)
    alpha = math.acos(max(-1.0, min(1.0, cos_alpha)))
    psi = math.atan2(oy - uy, ox - ux)
    phis = (psi - alpha, psi + alpha)
    points = [(ux + ur * math.cos(p), uy + ur * math.sin(p)) for p in phis]
    span = min(2.0 * alpha, 2.0 * math.pi - 2.0 * alpha)
    fragment = ur * span - cut
    return {
        "theta": theta,
        "cut": cut,
        "phis": phis,
        "points": points,
        "fragment": fragment,
    }


def _panel_fragment(panel: dict) -> float:
    """Shorter surviving fragment of this panel's under ring, in pixels."""
    geo = _crossing(*_order(panel))
    return 0.0 if geo is None else float(geo["fragment"])


def _patch_phi(under: dict, over: dict, patch_lower: bool) -> float:
    """Angle (from the under ring's centre) of the crossing that is repainted.

    The crossing is chosen by its own screen position — lower or upper on the
    canvas — so the choice does not depend on which slot holds which ring.
    """
    geo = _crossing(under, over)
    phis, points = geo["phis"], geo["points"]
    order = sorted(range(2), key=lambda i: (points[i][1], points[i][0]))
    return phis[order[1] if patch_lower else order[0]]


def _sample_panel(rng: random.Random, index: int, linked: bool) -> dict:
    ix0, iy0, ix1, iy1 = panel_inner(index)
    cx_mid, cy_mid = (ix0 + ix1) / 2.0, (iy0 + iy1) / 2.0
    for _ in range(20000):
        r0 = rng.randint(R_MIN, R_MAX)
        r1 = rng.randint(R_MIN, R_MAX)
        theta = math.radians(rng.uniform(50.0, 115.0))
        d = round(math.sqrt(r0 * r0 + r1 * r1 - 2.0 * r0 * r1 * math.cos(theta)))
        if not (abs(r0 - r1) + 10 < d < r0 + r1 - 10):
            continue
        psi = rng.uniform(0.0, 2.0 * math.pi)
        mx = cx_mid + rng.uniform(-7.0, 7.0)
        my = cy_mid + rng.uniform(-7.0, 7.0)
        c0 = (round(mx - d / 2.0 * math.cos(psi)), round(my - d / 2.0 * math.sin(psi)))
        c1 = (round(mx + d / 2.0 * math.cos(psi)), round(my + d / 2.0 * math.sin(psi)))
        rings = [
            {"cx": c0[0], "cy": c0[1], "r": r0, "color": 0},
            {"cx": c1[0], "cy": c1[1], "r": r1, "color": 0},
        ]
        halo = STROKE / 2.0 + 1.0
        if any(
            ring["cx"] - ring["r"] - halo < ix0
            or ring["cx"] + ring["r"] + halo > ix1
            or ring["cy"] - ring["r"] - halo < iy0
            or ring["cy"] + ring["r"] + halo > iy1
            for ring in rings
        ):
            continue
        # Both draw orders must stay legible: the fragment test is applied to
        # whichever ring ends up underneath, so the sampler is order-blind.
        geos = [_crossing(rings[0], rings[1]), _crossing(rings[1], rings[0])]
        if any(g is None for g in geos):
            continue
        if min(g["fragment"] for g in geos) < MIN_FRAGMENT_PX:
            continue
        colors = rng.sample(range(len(PALETTE)), 2)
        rings[0]["color"], rings[1]["color"] = colors
        return {
            "rings": rings,
            "top": rng.randrange(2),
            "linked": bool(linked),
            "patch_lower": bool(rng.randrange(2)),
        }
    raise RuntimeError(f"could not place a crossing ring pair in panel {index}")


def sample_scene(seed: int) -> dict:
    """Sample one balanced scene: exactly one interlocked panel."""
    rng = random.Random(seed * 7919 + 13)
    winner = rng.randrange(PANELS)
    panels = [_sample_panel(rng, i, linked=(i == winner)) for i in range(PANELS)]
    return {
        "canvas": [CANVAS_W, CANVAS_H],
        "palette": rng.randrange(len(PALETTE)),
        "panels": panels,
    }


def _rgb(color_index: int, palette_offset: int) -> tuple[int, int, int]:
    return PALETTE[(int(color_index) + int(palette_offset)) % len(PALETTE)]


def render(scene: dict) -> Image.Image:
    """I = R(z). Flat colours, no antialiasing: every ink pixel is exact."""
    width, height = int(scene["canvas"][0]), int(scene["canvas"][1])
    arr = np.full((height, width, 3), BG_RGB, dtype=np.uint8)
    offset = int(scene["palette"])
    for index in range(PANELS):
        x0, y0, x1, y1 = panel_box(index)
        arr[y0 : y0 + FRAME, x0:x1] = FRAME_RGB
        arr[y1 - FRAME : y1, x0:x1] = FRAME_RGB
        arr[y0:y1, x0 : x0 + FRAME] = FRAME_RGB
        arr[y0:y1, x1 - FRAME : x1] = FRAME_RGB
    for index, panel in enumerate(scene["panels"]):
        x0, y0, x1, y1 = panel_box(index)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        yy = yy.astype(np.float64)
        xx = xx.astype(np.float64)
        under, over = _order(panel)
        sub = arr[y0:y1, x0:x1]

        def annulus(ring: dict) -> np.ndarray:
            dist = np.hypot(xx - float(ring["cx"]), yy - float(ring["cy"]))
            return np.abs(dist - float(ring["r"])) <= STROKE / 2.0

        mask_under = annulus(under)
        sub[mask_under] = _rgb(under["color"], offset)
        sub[annulus(over)] = _rgb(over["color"], offset)
        if panel["linked"]:
            phi = _patch_phi(under, over, bool(panel["patch_lower"]))
            ang = np.arctan2(yy - float(under["cy"]), xx - float(under["cx"]))
            delta = np.abs((ang - phi + math.pi) % (2.0 * math.pi) - math.pi)
            wedge = delta <= PATCH_ARC_PX / float(under["r"])
            sub[mask_under & wedge] = _rgb(under["color"], offset)
    return Image.fromarray(arr, mode="RGB")


def analytic_gold(scene: dict) -> str:
    linked = [i for i, panel in enumerate(scene["panels"]) if panel["linked"]]
    if len(linked) != 1:
        raise ValueError(f"scene has {len(linked)} interlocked panels, expected exactly 1")
    return f"panel-{linked[0] + 1}"


def margin(scene: dict) -> float:
    """Runner-up decision margin, in pixels.

    The runner-up is the stacked panel that comes closest to reading as
    interlocked: the one whose under ring keeps the shortest surviving fragment
    between its two breaks. Once that fragment vanishes the stacked pair also
    shows one connected curve per colour and the panels tie.
    """
    fragments = [
        _panel_fragment(panel) for panel in scene["panels"] if not panel["linked"]
    ]
    return float(min(fragments)) if fragments else 0.0


def is_quarantined(scene: dict) -> bool:
    linked = sum(1 for panel in scene["panels"] if panel["linked"])
    return linked != 1 or margin(scene) < QUARANTINE_MARGIN


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """Declared latent aliases: pixel-identical, gold-preserving rewrites of z."""
    out: list[tuple[str, dict]] = []

    rotated = copy.deepcopy(scene)
    rotated["palette"] = int(rotated["palette"]) + len(PALETTE)
    out.append(("palette_period", rotated))

    swapped = copy.deepcopy(scene)
    for panel in swapped["panels"]:
        panel["rings"] = [panel["rings"][1], panel["rings"][0]]
        panel["top"] = 1 - int(panel["top"])
    out.append(("ring_slot_swap", swapped))

    flipped = copy.deepcopy(scene)
    for panel in flipped["panels"]:
        if not panel["linked"]:
            panel["patch_lower"] = not bool(panel["patch_lower"])
    out.append(("stacked_patch_side", flipped))
    return out
