"""Renderer for the end-fold pairing reveal world.

One teal cord runs as a single unbranching thin route across the whole canvas.
It carries 50 round beads, each filled with one of exactly two grays. Folding
the cord end to end pairs bead k with bead 49-k; a pair is marked when its two
beads carry the same gray. The 25 pair verdicts, taken in pair order, spell one
abstract glyph on a 5x5 grid.

Every decisive comparison couples a bead near one end of the cord with a bead
near the other end, so no local neighbourhood of the picture carries the answer:
the first 25 beads alone are an unbiased coin sequence and so are the last 25.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

RENDERER_VERSION = "endfold_pairing_reveal-renderer-0.1.0"

CANVAS = 640
LATTICE = 21
PITCH = 28
ORIGIN = (CANVAS - (LATTICE - 1) * PITCH) // 2
BEADS = 50
PAIRS = BEADS // 2
GRID = 5
CORD_W = 6
BEAD_R = 8
BRACKET_LEN = 9
BRACKET_W = 3
BRACKET_GAP = 5
KEY_SW = 26

QUARANTINE_CLEARANCE = 17.0

BACKGROUND = (255, 255, 255)
CORD = (26, 132, 128)
DECOY = (203, 197, 226)
INK = (0, 0, 0)
KEY_FRAME = (98, 104, 168)

GLYPHS = ("ell", "ex", "plus", "tee")


def _glyph_cells(name: str) -> frozenset[tuple[int, int]]:
    if name == "plus":
        return frozenset([(2, c) for c in range(GRID)] + [(r, 2) for r in range(GRID)])
    if name == "ex":
        return frozenset(
            [(i, i) for i in range(GRID)] + [(i, GRID - 1 - i) for i in range(GRID)]
        )
    if name == "tee":
        return frozenset([(0, c) for c in range(GRID)] + [(r, 2) for r in range(1, GRID)])
    if name == "ell":
        return frozenset([(r, 0) for r in range(GRID)] + [(GRID - 1, c) for c in range(1, GRID)])
    raise ValueError(f"unknown glyph {name!r}")


GLYPH_MASKS: dict[str, tuple[int, ...]] = {
    name: tuple(
        1 if (index // GRID, index % GRID) in _glyph_cells(name) else 0
        for index in range(GRID * GRID)
    )
    for name in GLYPHS
}


def _hamming(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def hamming_margin(scene: dict) -> int:
    """Smallest Hamming distance from the revealed mask to any other glyph."""
    marks = _marks_from_tones(tuple(scene["tones"]))
    gold = analytic_gold(scene)
    return min(_hamming(marks, GLYPH_MASKS[name]) for name in GLYPHS if name != gold)


def _neighbours(node: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    x, y = node
    return ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))


def _free(node: tuple[int, int], used: set, current: tuple[int, int]) -> bool:
    """A node is free when it is unused and touches no used node but `current`."""
    x, y = node
    if not (0 <= x < LATTICE and 0 <= y < LATTICE):
        return False
    if node in used:
        return False
    for other in _neighbours(node):
        if other != current and other in used:
            return False
    return True


def _route(rng: random.Random) -> list[tuple[int, int]] | None:
    """A sparse staircase route: long straight legs, one thin trace corner to corner.

    Legs alternate between the two axes and mostly advance toward the opposite
    corner, so the cord reaches across the whole canvas while leaving most of it
    empty. The realised node list is then checked to be self-avoiding and
    non-touching, which keeps consecutive beads the only ink-joined pairs.
    """
    for _ in range(600):
        sx = 1 if rng.random() < 0.5 else -1
        sy = 1 if rng.random() < 0.5 else -1
        x = 0 if sx > 0 else LATTICE - 1
        y = 0 if sy > 0 else LATTICE - 1
        path = [(x, y)]
        axis = 0 if rng.random() < 0.5 else 1
        while len(path) < BEADS:
            remaining = BEADS - len(path)
            leg = min(remaining, rng.choice((3, 4, 4, 5, 5, 6)))
            forward = rng.random() < 0.86
            step = (sx if axis == 0 else sy) * (1 if forward else -1)
            for _ in range(leg):
                x, y = path[-1]
                nxt = (x + step, y) if axis == 0 else (x, y + step)
                if not (0 <= nxt[0] < LATTICE and 0 <= nxt[1] < LATTICE):
                    break
                path.append(nxt)
            axis = 1 - axis
            if len(path) == BEADS:
                break
            if len(path) > 1 and path[-1] == path[-2]:
                break
            if len(path) < 2:
                break
            if len(path) < BEADS and len(set(path)) != len(path):
                break
        if len(path) != BEADS or not _non_touching(path):
            continue
        if _spread_ok(path):
            return path
    return None


def _non_touching(path: list[tuple[int, int]]) -> bool:
    """No two non-consecutive nodes coincide or share a lattice edge."""
    if len(set(path)) != len(path):
        return False
    index = {node: i for i, node in enumerate(path)}
    for i, node in enumerate(path):
        for other in _neighbours(node):
            j = index.get(other)
            if j is not None and abs(i - j) != 1:
                return False
    return True


def _spread_ok(path: list[tuple[int, int]]) -> bool:
    """The route must span the canvas and separate most folded partners."""
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    if max(xs) - min(xs) < LATTICE - 4 or max(ys) - min(ys) < LATTICE - 4:
        return False
    half = LATTICE / 2.0
    quadrants = [0, 0, 0, 0]
    for x, y in path:
        quadrants[(1 if x >= half else 0) + (2 if y >= half else 0)] += 1
    if min(quadrants) < 3:
        return False
    far = 0
    for k in range(PAIRS):
        a, b = path[k], path[BEADS - 1 - k]
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 8:
            far += 1
    return far >= 19


def _marks_from_tones(tones: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 if tones[k] == tones[BEADS - 1 - k] else 0 for k in range(PAIRS))


def _alternative_masks(tones: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Pairings a shortcut reader might try instead of the declared fold."""
    stacked = tuple(1 if tones[k] == tones[k + PAIRS] else 0 for k in range(PAIRS))
    adjacent = tuple(1 if tones[2 * k] == tones[2 * k + 1] else 0 for k in range(PAIRS))
    first_half = tuple(tones[:PAIRS])
    last_half = tuple(tones[PAIRS:])
    reversed_last = tuple(reversed(last_half))
    return [stacked, adjacent, first_half, last_half, reversed_last]


def _tones_for(rng: random.Random, glyph: str) -> tuple[int, ...]:
    """Bead tones whose folded pair verdicts spell `glyph` and nothing cheaper."""
    target = GLYPH_MASKS[glyph]
    for _ in range(2000):
        head = [rng.randrange(2) for _ in range(PAIRS)]
        tones = [0] * BEADS
        for k in range(PAIRS):
            tones[k] = head[k]
            tones[BEADS - 1 - k] = head[k] if target[k] else 1 - head[k]
        candidate = tuple(tones)
        if _marks_from_tones(candidate) != target:
            continue
        if any(
            mask in GLYPH_MASKS.values() for mask in _alternative_masks(candidate)
        ):
            continue
        return candidate
    raise RuntimeError("could not sample decoy-free bead tones")


def _node_xy(node: tuple[int, int]) -> tuple[int, int]:
    return (ORIGIN + node[0] * PITCH, ORIGIN + node[1] * PITCH)


def _point_segment_distance(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    length = dx * dx + dy * dy
    if length <= 0.0:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length))
    cx, cy = ax + t * dx, ay + t * dy
    return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5


def key_box(corner: int) -> tuple[int, int, int, int]:
    """Pixel box of the two-swatch tone key for one of the four corners."""
    left = corner in (0, 2)
    top = corner in (0, 1)
    x0 = 14 if left else CANVAS - 14 - 2 * KEY_SW
    y0 = 14 if top else CANVAS - 14 - KEY_SW
    return (x0, y0, x0 + 2 * KEY_SW, y0 + KEY_SW)


def free_key_corners(scene: dict) -> list[int]:
    """Corners whose key box keeps a clear margin from the decisive cord."""
    origin = int(scene["origin"])
    pitch = int(scene["pitch"])
    points = [(origin + n[0] * pitch, origin + n[1] * pitch) for n in scene["route"]]
    free = []
    for corner in range(4):
        x0, y0, x1, y1 = key_box(corner)
        clear = True
        for index in range(len(points) - 1):
            for probe in range(9):
                t = probe / 8.0
                px = points[index][0] + (points[index + 1][0] - points[index][0]) * t
                py = points[index][1] + (points[index + 1][1] - points[index][1]) * t
                if x0 - 18 <= px <= x1 + 18 and y0 - 18 <= py <= y1 + 18:
                    clear = False
                    break
            if not clear:
                break
        if clear:
            free.append(corner)
    return free


def _decoys(rng: random.Random, route: list[tuple[int, int]]) -> list[dict]:
    points = [_node_xy(node) for node in route]
    decoys: list[dict] = []
    for _ in range(200):
        if len(decoys) >= 3:
            break
        x0 = rng.randrange(30, CANVAS - 30)
        y0 = rng.randrange(30, CANVAS - 30)
        length = rng.randrange(60, 130)
        horizontal = rng.random() < 0.5
        x1 = x0 + (length if horizontal else 0)
        y1 = y0 + (0 if horizontal else length)
        if not (30 <= x1 <= CANVAS - 30 and 30 <= y1 <= CANVAS - 30):
            continue
        clear = True
        for index in range(len(points) - 1):
            for probe in (0.0, 0.25, 0.5, 0.75, 1.0):
                probe_pt = (x0 + (x1 - x0) * probe, y0 + (y1 - y0) * probe)
                if _point_segment_distance(probe_pt, points[index], points[index + 1]) < 26:
                    clear = False
                    break
            if not clear:
                break
        if not clear:
            continue
        for other in decoys:
            if abs(other["x0"] - x0) < 34 and abs(other["y0"] - y0) < 34:
                clear = False
        if clear:
            decoys.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1})
    return decoys


def sample_scene(seed: int) -> dict:
    """Sample one scene; every field controls visible pixels."""
    rng = random.Random(0x5E11 ^ (seed * 2654435761 % (2**31)))
    route = None
    for _ in range(60):
        route = _route(rng)
        if route is not None:
            break
    if route is None:
        raise RuntimeError(f"no admissible cord route for seed {seed}")
    probe = {"origin": ORIGIN, "pitch": PITCH, "route": [list(n) for n in route]}
    free = free_key_corners(probe)
    for _ in range(60):
        if free:
            break
        route = _route(rng)
        if route is None:
            raise RuntimeError(f"no admissible cord route for seed {seed}")
        probe = {"origin": ORIGIN, "pitch": PITCH, "route": [list(n) for n in route]}
        free = free_key_corners(probe)
    if not free:
        raise RuntimeError(f"no clear tone-key corner for seed {seed}")
    glyph = GLYPHS[rng.randrange(len(GLYPHS))]
    tones = _tones_for(rng, glyph)
    dark = rng.randrange(52, 92)
    gap = rng.randrange(30, 121)
    light = dark + gap
    key_corner = free[rng.randrange(len(free))]
    return {
        "canvas": CANVAS,
        "pitch": PITCH,
        "origin": ORIGIN,
        "cord_w": CORD_W,
        "bead_r": BEAD_R,
        "route": [list(node) for node in route],
        "tones": list(tones),
        "dark": dark,
        "light": light,
        "key_corner": key_corner,
        "decoys": _decoys(rng, route),
    }


def analytic_gold(scene: dict) -> str:
    """The glyph spelled by the folded pair verdicts."""
    marks = _marks_from_tones(tuple(scene["tones"]))
    for name in GLYPHS:
        if GLYPH_MASKS[name] == marks:
            return name
    raise ValueError("sampled tones do not spell a declared glyph")


def margin(scene: dict) -> float:
    """Per-bead tone clearance: half the rendered gray gap, in luma units."""
    return abs(float(scene["light"]) - float(scene["dark"])) / 2.0


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < QUARANTINE_CLEARANCE


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """No latent aliases: every scene field is rendered (see candidate.json)."""
    return []


def _tone_rgb(scene: dict, bit: int) -> tuple[int, int, int]:
    value = int(scene["light"]) if bit else int(scene["dark"])
    return (value, value, value)


def render(scene: dict) -> Image.Image:
    canvas = int(scene["canvas"])
    image = Image.new("RGB", (canvas, canvas), BACKGROUND)
    draw = ImageDraw.Draw(image)
    origin = int(scene["origin"])
    pitch = int(scene["pitch"])
    radius = int(scene["bead_r"])
    width = int(scene["cord_w"])

    for decoy in scene["decoys"]:
        draw.line(
            (decoy["x0"], decoy["y0"], decoy["x1"], decoy["y1"]), fill=DECOY, width=width
        )
        for end in ((decoy["x0"], decoy["y0"]), (decoy["x1"], decoy["y1"])):
            draw.ellipse(
                (end[0] - radius, end[1] - radius, end[0] + radius, end[1] + radius),
                fill=DECOY,
            )

    points = [
        (origin + node[0] * pitch, origin + node[1] * pitch) for node in scene["route"]
    ]
    for index in range(len(points) - 1):
        a, b = points[index], points[index + 1]
        draw.line((a[0], a[1], b[0], b[1]), fill=CORD, width=width)

    for point, bit in zip(points, scene["tones"]):
        draw.ellipse(
            (point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius),
            fill=_tone_rgb(scene, int(bit)),
        )

    sx, sy = points[0]
    reach = radius + BRACKET_GAP
    for dx in (-1, 1):
        for dy in (-1, 1):
            cx, cy = sx + dx * reach, sy + dy * reach
            draw.line((cx, cy, cx + dx * BRACKET_LEN, cy), fill=INK, width=BRACKET_W)
            draw.line((cx, cy, cx, cy + dy * BRACKET_LEN), fill=INK, width=BRACKET_W)

    key_x, key_y, _, _ = key_box(int(scene["key_corner"]))
    for slot, bit in enumerate((0, 1)):
        x0 = key_x + slot * KEY_SW
        draw.rectangle(
            (x0, key_y, x0 + KEY_SW - 1, key_y + KEY_SW - 1), fill=_tone_rgb(scene, bit)
        )
    draw.rectangle(
        (key_x - 1, key_y - 1, key_x + 2 * KEY_SW, key_y + KEY_SW),
        outline=KEY_FRAME,
        width=2,
    )
    return image
