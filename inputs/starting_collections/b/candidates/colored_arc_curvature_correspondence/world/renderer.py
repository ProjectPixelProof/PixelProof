"""Renderer and analytic label for the colored arc-curvature correspondence world.

The latent scene contains three visible panels.  Each panel has four isolated
colored quadratic arcs.  Their profile values control normalized sagitta
(maximum centerline bow divided by endpoint-chord length).  The label compares
the complete color-to-profile mapping, not a panel position or a raw size.
"""

from __future__ import annotations

import math
import random
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


RENDERER_VERSION = "colored-arc-curvature-correspondence-0.1.0"
WIDTH = 960
HEIGHT = 560
BACKGROUND = (249, 249, 246)
PANEL_FILL = (255, 255, 252)
PANEL_BORDER = (171, 175, 181)
LABEL_INK = (42, 47, 54)
SUBTITLE_INK = (109, 113, 120)

COLOR_RGB: dict[str, tuple[int, int, int]] = {
    "coral": (214, 70, 77),
    "gold": (225, 153, 34),
    "teal": (28, 153, 145),
    "indigo": (71, 94, 190),
}
COLORS = tuple(COLOR_RGB)

PANEL_BOXES = {
    "reference": (24, 64, 304, 528),
    "A": (340, 64, 620, 528),
    "B": (656, 64, 936, 528),
}
PANEL_ORDER = ("reference", "A", "B")
PROFILE_RATIOS = (0.14, 0.23, 0.32, 0.41)
PROFILE_VALUES = frozenset(range(4))
SLOT_Y = (208.0, 390.0)
SLOT_X_OFFSETS = (78.0, 202.0)


def _profile_ratio(profile: int) -> float:
    return PROFILE_RATIOS[int(profile)]


def _signature(panel: list[dict]) -> dict[str, int]:
    return {str(arc["color"]): int(arc["profile"]) for arc in panel}


def _scores(scene: dict) -> tuple[int, int]:
    reference = _signature(scene["reference"])
    return tuple(
        sum(reference.get(color) != _signature(panel).get(color) for color in COLORS)
        for panel in (scene["candidate_a"], scene["candidate_b"])
    )  # type: ignore[return-value]


def _panel_arcs(
    rng: random.Random,
    panel_number: int,
    profile_to_color: dict[int, str],
) -> list[dict]:
    left = PANEL_BOXES[PANEL_ORDER[panel_number]][0]
    slots = [(left + offset, y) for y in SLOT_Y for offset in SLOT_X_OFFSETS]
    rng.shuffle(slots)
    arcs: list[dict] = []
    for profile, (cx, cy) in zip(range(4), slots, strict=True):
        arcs.append(
            {
                "color": str(profile_to_color[profile]),
                "profile": int(profile),
                "cx": round(float(cx + rng.uniform(-7.0, 7.0)), 4),
                "cy": round(float(cy + rng.uniform(-8.0, 8.0)), 4),
                "angle": round(float(rng.uniform(-math.pi, math.pi)), 6),
                "chord": round(float(rng.uniform(78.0, 92.0)), 4),
                "stroke": round(float(rng.uniform(7.0, 10.0)), 4),
            }
        )
    rng.shuffle(arcs)
    return arcs


def sample_scene(seed: int) -> dict:
    """Return one deterministic, balanced scene for an integer seed."""

    rng = random.Random(int(seed))
    shuffled_colors = list(COLORS)
    rng.shuffle(shuffled_colors)
    reference_map = {
        profile: shuffled_colors[profile] for profile in range(len(COLORS))
    }

    wrong_map = dict(reference_map)
    first, second = rng.sample(range(4), 2)
    wrong_map[first], wrong_map[second] = wrong_map[second], wrong_map[first]

    correct_is_a = int(seed) % 2 == 0
    reference = _panel_arcs(rng, 0, reference_map)
    candidate_a = _panel_arcs(rng, 1, reference_map if correct_is_a else wrong_map)
    candidate_b = _panel_arcs(rng, 2, wrong_map if correct_is_a else reference_map)

    return {
        "width": WIDTH,
        "height": HEIGHT,
        "background": list(BACKGROUND),
        "reference": reference,
        "candidate_a": candidate_a,
        "candidate_b": candidate_b,
    }


def _bezier_points(arc: dict, supersample: int) -> list[tuple[int, int]]:
    cx = float(arc["cx"])
    cy = float(arc["cy"])
    angle = float(arc["angle"])
    chord = float(arc["chord"])
    ux, uy = math.cos(angle), math.sin(angle)
    nx, ny = -uy, ux
    half = chord / 2.0
    p0 = (cx - ux * half, cy - uy * half)
    p1 = (cx + ux * half, cy + uy * half)
    sagitta = chord * _profile_ratio(int(arc["profile"]))
    control = (cx + nx * sagitta * 2.0, cy + ny * sagitta * 2.0)
    points: list[tuple[int, int]] = []
    for index in range(49):
        t = index / 48.0
        mt = 1.0 - t
        x = mt * mt * p0[0] + 2.0 * mt * t * control[0] + t * t * p1[0]
        y = mt * mt * p0[1] + 2.0 * mt * t * control[1] + t * t * p1[1]
        points.append((int(round(x * supersample)), int(round(y * supersample))))
    return points


def _font(size: int, supersample: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size * supersample
        )
    except OSError:
        return ImageFont.load_default()


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    panel: Iterable[dict],
    name: str,
    box: tuple[int, int, int, int],
    supersample: int,
) -> None:
    left, top, right, bottom = box
    scale = supersample
    draw.rounded_rectangle(
        tuple(int(value * scale) for value in box),
        radius=14 * scale,
        fill=PANEL_FILL,
        outline=PANEL_BORDER,
        width=2 * scale,
    )
    title = "REFERENCE" if name == "reference" else name
    title_font = _font(18, scale)
    small_font = _font(12, scale)
    title_box = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_box[2] - title_box[0]
    draw.text(
        (int((left + right) * scale / 2 - title_width / 2), int((top + 22) * scale)),
        title,
        fill=LABEL_INK,
        font=title_font,
    )
    subtitle = "normalized bow ranks"
    subtitle_box = draw.textbbox((0, 0), subtitle, font=small_font)
    subtitle_width = subtitle_box[2] - subtitle_box[0]
    draw.text(
        (int((left + right) * scale / 2 - subtitle_width / 2), int((top + 52) * scale)),
        subtitle,
        fill=SUBTITLE_INK,
        font=small_font,
    )

    for arc in panel:
        color = COLOR_RGB[str(arc["color"])]
        points = _bezier_points(arc, scale)
        width = max(1, int(round(float(arc["stroke"]) * scale)))
        draw.line(points, fill=color, width=width, joint="curve")
        radius = width / 2.0
        for x, y in (points[0], points[-1]):
            draw.ellipse(
                (int(x - radius), int(y - radius), int(x + radius), int(y + radius)),
                fill=color,
            )


def render(scene: dict) -> Image.Image:
    """Rasterize only fields present in the scene using deterministic Pillow drawing."""

    supersample = 4
    width = int(scene["width"])
    height = int(scene["height"])
    background = tuple(int(value) for value in scene["background"])
    image = Image.new("RGB", (width * supersample, height * supersample), background)
    draw = ImageDraw.Draw(image)
    panels = (scene["reference"], scene["candidate_a"], scene["candidate_b"])
    for panel, name in zip(panels, PANEL_ORDER, strict=True):
        _draw_panel(draw, panel, name, PANEL_BOXES[name], supersample)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def analytic_gold(scene: dict) -> str:
    """Select the unique candidate whose color-to-profile map equals the reference."""

    score_a, score_b = _scores(scene)
    if score_a == score_b or min(score_a, score_b) != 0:
        raise ValueError("candidate curvature profiles are not uniquely matched")
    return "A" if score_a < score_b else "B"


def margin(scene: dict) -> float:
    """Return the absolute mismatch-score gap between the candidate profiles."""

    score_a, score_b = _scores(scene)
    return float(abs(score_a - score_b))


def _well_formed_panel(panel: object) -> bool:
    if not isinstance(panel, list) or len(panel) != 4:
        return False
    colors: set[str] = set()
    profiles: set[int] = set()
    for arc in panel:
        if not isinstance(arc, dict):
            return False
        try:
            color = str(arc["color"])
            profile = int(arc["profile"])
            cx = float(arc["cx"])
            cy = float(arc["cy"])
            angle = float(arc["angle"])
            chord = float(arc["chord"])
            stroke = float(arc["stroke"])
        except (KeyError, TypeError, ValueError):
            return False
        if color not in COLOR_RGB or color in colors or profile not in PROFILE_VALUES:
            return False
        if profile in profiles or not all(math.isfinite(value) for value in (cx, cy, angle, chord, stroke)):
            return False
        if chord < 70.0 or stroke < 5.0:
            return False
        colors.add(color)
        profiles.add(profile)
    return colors == set(COLORS) and profiles == PROFILE_VALUES


def is_quarantined(scene: dict) -> bool:
    """Reject malformed, tied, or non-unique finite-resolution profile cases."""

    try:
        if any(
            not _well_formed_panel(scene[key])
            for key in ("reference", "candidate_a", "candidate_b")
        ):
            return True
        score_a, score_b = _scores(scene)
        return score_a == score_b or min(score_a, score_b) != 0 or abs(score_a - score_b) < 2
    except (KeyError, TypeError, ValueError):
        return True


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """No hidden aliases are claimed; all label-bearing arc records are visible."""

    return []
