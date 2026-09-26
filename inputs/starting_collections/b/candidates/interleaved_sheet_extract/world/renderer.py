"""Interleaved-sheet extraction world.

Two 7x7 sheets of dark cells are interleaved row by row into one printed
14x7 lattice.  A filled dot or a hollow ring is printed beside every lattice
row and states which sheet that row belongs to.  Keeping the dot-marked rows
in order and pushing them together restores one sheet, whose dark cells match
one of five named symbols.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

ROWS = 14
COLS = 7
BLOCK = 7
PITCH = 44
CELL = 34
FRAME_W = 3
MARK_R = 11
MARK_DX = 30
BASE_X = 66
BASE_Y = 36
CANVAS = (420, 706)
FILL_CHARS = "1#"

SYMBOL_NAMES = ("ring", "wedge", "zigzag", "arrow", "slash")

TEMPLATES = {
    "ring": (
        ".......",
        ".#####.",
        ".#...#.",
        ".#...#.",
        ".#...#.",
        ".#####.",
        ".......",
    ),
    "wedge": (
        ".......",
        "#######",
        ".#####.",
        "..###..",
        "...#...",
        ".......",
        ".......",
    ),
    "zigzag": (
        "######.",
        ".....#.",
        "....#..",
        "...#...",
        "..#....",
        ".#.....",
        "..#####",
    ),
    "arrow": (
        "#.....#",
        "##...##",
        ".##.##.",
        "..###..",
        "...#...",
        "...#...",
        "...#...",
    ),
    "slash": (
        "##.....",
        "###....",
        ".###...",
        "..###..",
        "....##.",
        ".....##",
        "......#",
    ),
}

PALETTES = (
    {
        "bg": (246, 245, 241),
        "grid": (214, 212, 206),
        "cell": (28, 32, 60),
        "frame": (188, 84, 32),
        "mark": (36, 96, 104),
    },
    {
        "bg": (243, 246, 244),
        "grid": (208, 214, 210),
        "cell": (46, 26, 54),
        "frame": (32, 96, 168),
        "mark": (150, 60, 24),
    },
    {
        "bg": (247, 244, 246),
        "grid": (212, 208, 212),
        "cell": (24, 52, 34),
        "frame": (140, 44, 120),
        "mark": (60, 64, 140),
    },
)


def _matrix(rows) -> list[list[int]]:
    return [[1 if ch in FILL_CHARS else 0 for ch in row] for row in rows]


def _hamming(a: list[list[int]], b: list[list[int]]) -> int:
    return sum(
        1
        for r in range(len(a))
        for c in range(len(a[0]))
        if a[r][c] != b[r][c]
    )


def _scores(block: list[list[int]]) -> list[tuple[int, str]]:
    return sorted(
        (_hamming(block, _matrix(TEMPLATES[name])), name) for name in SYMBOL_NAMES
    )


def _perturb(base: list[list[int]], swaps: int, rng: random.Random) -> list[list[int]]:
    grid = [row[:] for row in base]
    on = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if grid[r][c]]
    off = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if not grid[r][c]]
    rng.shuffle(on)
    rng.shuffle(off)
    for i in range(swaps):
        r0, c0 = on[i]
        r1, c1 = off[i]
        grid[r0][c0] = 0
        grid[r1][c1] = 1
    return grid


def _sheet(name: str, swaps: int, rng: random.Random, boundary: bool) -> list[list[int]]:
    base = _matrix(TEMPLATES[name])
    for _ in range(24):
        grid = _perturb(base, swaps, rng)
        ranked = _scores(grid)
        gap = ranked[1][0] - ranked[0][0]
        if boundary:
            if gap <= 1:
                return grid
        elif ranked[0][1] == name and gap >= 2:
            return grid
    return base


def _nearest_other(name: str) -> tuple[str, int]:
    ranked = sorted(
        (_hamming(_matrix(TEMPLATES[name]), _matrix(TEMPLATES[other])), other)
        for other in SYMBOL_NAMES
        if other != name
    )
    return ranked[0][1], ranked[0][0]


def _blend(a_name: str, b_name: str, steps: int, rng: random.Random) -> list[list[int]]:
    """Move `steps` ink-preserving cell pairs from sheet A towards sheet B."""
    a = _matrix(TEMPLATES[a_name])
    b = _matrix(TEMPLATES[b_name])
    grid = [row[:] for row in a]
    drop = [
        (r, c) for r in range(BLOCK) for c in range(BLOCK) if a[r][c] and not b[r][c]
    ]
    gain = [
        (r, c) for r in range(BLOCK) for c in range(BLOCK) if b[r][c] and not a[r][c]
    ]
    rng.shuffle(drop)
    rng.shuffle(gain)
    for i in range(min(steps, len(drop), len(gain))):
        grid[drop[i][0]][drop[i][1]] = 0
        grid[gain[i][0]][gain[i][1]] = 1
    return grid


def _marker_bits(rng: random.Random) -> list[int]:
    while True:
        bits = [1] * BLOCK + [0] * BLOCK
        rng.shuffle(bits)
        runs = sum(1 for i in range(ROWS - 1) if bits[i] != bits[i + 1])
        if runs >= 6:
            return bits


def sample_scene(seed: int) -> dict:
    """Sample one latent scene.  Every stored field is printed in the raster."""
    rng = random.Random(seed * 7919 + 13)
    target = SYMBOL_NAMES[seed % len(SYMBOL_NAMES)]
    mode = seed % 8
    if mode == 7:
        # Tied boundary scene: halfway between the two closest templates.
        target = "wedge" if (seed // 8) % 2 == 0 else "arrow"
        partner, distance = _nearest_other(target)
        grid_a = _blend(target, partner, distance // 4, rng)
    elif mode == 3:
        # Near-boundary scene: gold still `target`, runner-up gap of two to four.
        partner, distance = _nearest_other(target)
        grid_a = _blend(target, partner, (distance - 2) // 4, rng)
    else:
        grid_a = _sheet(target, rng.randint(0, 2), rng, False)
    decoy = rng.choice([s for s in SYMBOL_NAMES if s != target])
    grid_b = _sheet(decoy, rng.randint(0, 2), rng, False)
    bits = _marker_bits(rng)
    glyph = rng.choice(FILL_CHARS)

    rows: list[str] = []
    ia = ib = 0
    for bit in bits:
        if bit:
            source = grid_a[ia]
            ia += 1
        else:
            source = grid_b[ib]
            ib += 1
        rows.append("".join(glyph if v else "." for v in source))

    marker_codes = [bit + 2 * rng.randint(0, 3) for bit in bits]
    return {
        "rows": rows,
        "marker_codes": marker_codes,
        "palette": rng.randrange(len(PALETTES)),
        "jitter_x": rng.randint(-8, 8),
        "jitter_y": rng.randint(-8, 8),
    }


def extracted_block(scene: dict) -> list[list[int]]:
    """The declared transform: keep dot-marked rows in order, push them together."""
    kept = [
        scene["rows"][i]
        for i, code in enumerate(scene["marker_codes"])
        if code % 2 == 1
    ]
    return _matrix(kept)


def analytic_gold(scene: dict) -> str:
    return _scores(extracted_block(scene))[0][1]


def margin(scene: dict) -> float:
    ranked = _scores(extracted_block(scene))
    return float(ranked[1][0] - ranked[0][0])


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < 2.0


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """Pixel-identical, gold-preserving relabelings of the latent scene."""
    shifted = dict(scene)
    shifted["marker_codes"] = [code + 2 for code in scene["marker_codes"]]

    swapped = dict(scene)
    other = {"1": "#", "#": "1"}
    swapped["rows"] = [
        "".join(other.get(ch, ch) for ch in row) for row in scene["rows"]
    ]
    return [("marker_code_shift", shifted), ("fill_glyph_swap", swapped)]


def render(scene: dict) -> Image.Image:
    pal = PALETTES[scene["palette"]]
    img = Image.new("RGB", CANVAS, pal["bg"])
    draw = ImageDraw.Draw(img)
    ox = BASE_X + scene["jitter_x"]
    oy = BASE_Y + scene["jitter_y"]

    fx0, fy0 = ox - 1, oy - 1
    fx1, fy1 = ox + COLS * PITCH, oy + ROWS * PITCH
    draw.rectangle([fx0, fy0, fx1, fy1], outline=pal["frame"], width=FRAME_W)

    inset = (PITCH - CELL) // 2
    for r, row in enumerate(scene["rows"]):
        for c, ch in enumerate(row):
            x0 = ox + c * PITCH + inset
            y0 = oy + r * PITCH + inset
            box = [x0, y0, x0 + CELL - 1, y0 + CELL - 1]
            if ch in FILL_CHARS:
                draw.rectangle(box, fill=pal["cell"])
            else:
                draw.rectangle(box, outline=pal["grid"], width=1)

    for r, code in enumerate(scene["marker_codes"]):
        cx = fx0 - MARK_DX
        cy = oy + r * PITCH + PITCH // 2
        box = [cx - MARK_R, cy - MARK_R, cx + MARK_R, cy + MARK_R]
        if code % 2 == 1:
            draw.ellipse(box, fill=pal["mark"])
        else:
            draw.ellipse(box, outline=pal["mark"], width=3)
    return img
