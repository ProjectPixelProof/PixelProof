"""bitplane_glyph_reveal: least-significant-bit plane glyph reveal world.

Declared transform: take the least-significant bit plane (bit 0) of every
pixel's grayscale value. Each pixel's 8-bit value is a uniformly random 0..255
noise draw whose lowest bit is forced to encode a hidden block glyph: within
glyph cells the bit is 1, outside it is 0. The high 7 bits are independent per-
pixel random noise, so the direct image is pure gray speckle in which the
symbol is invisible. Isolating bit-plane 0 recovers the glyph, which is either
the block letter A or the block letter B. The question asks which letter the
bit-plane reveals.

Latent scene z = {noise, glyph, cell}. label = glyph.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

RENDERER_VERSION = "bitplane_glyph_reveal-0.1.0"

W = 512
H = 512
CELL = 16
NROWS = H // CELL          # 32
NCOLS = W // CELL          # 32

# Fixed block-glyph templates: 15 rows x 12 columns of cells.
TEMPLATE_A = [
    list("............"),
    list("....##......"),
    list("....##......"),
    list("...#..#....."),
    list("...#..#....."),
    list("..#....#...."),
    list("..#....#...."),
    list(".#......#..."),
    list(".#......#..."),
    list(".########..."),
    list("#........#.."),
    list("#........#.."),
    list("#........#.."),
    list("............"),
    list("............"),
]
TEMPLATE_B = [
    list("............"),
    list("##########.."),
    list("#........#.."),
    list("#........#.."),
    list("#........#.."),
    list("##########.."),
    list("#........#.."),
    list("#........#.."),
    list("#........#.."),
    list("##########.."),
    list("............"),
    list("............"),
    list("............"),
    list("............"),
    list("............"),
]
TEMPLATES = {"A": TEMPLATE_A, "B": TEMPLATE_B}

# Glyph box placement in block-grid coordinates.
BOX_R0, BOX_R1 = 8, 8 + 15    # rows [8, 23)
BOX_C0, BOX_C1 = 10, 10 + 12  # cols [10, 22)
Q_TOL = 0.15  # quarantine threshold on the A/B correlation margin


def _template_mask(glyph: str) -> np.ndarray:
    """(NROWS, NCOLS) boolean: glyph template placed in the block grid."""
    grid = TEMPLATES[glyph]
    mask = np.zeros((NROWS, NCOLS), dtype=bool)
    mask[BOX_R0:BOX_R1, BOX_C0:BOX_C1] = np.array(
        [[c == "#" for c in row] for row in grid], dtype=bool
    )
    return mask


def glyph_field(glyph: str) -> np.ndarray:
    """(NROWS, NCOLS) integer bit-plane: 1 where the glyph is on, 0 elsewhere."""
    return _template_mask(glyph).astype(np.uint8)


def build_scene(noise: int, glyph: str, cell: int = CELL) -> dict:
    return {"noise": int(noise), "glyph": glyph, "cell": int(cell)}


def sample_scene(seed: int) -> dict:
    """One deterministic scene from a scalar seed (interface requirement)."""
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    glyph = "A" if rng.integers(0, 2) else "B"
    return build_scene(noise=int(seed), glyph=glyph)


def render(scene: dict) -> Image.Image:
    """Render I = R(z). Pixel value = random 7-bit high part shifted, OR bit 0.

    value = (noise7 << 1) | glyph_bit, where noise7 ~ U[0,127] per pixel. The
    displayed grayscale field is uniform random speckle; only bit 0 carries the
    glyph.
    """
    cell = int(scene.get("cell", CELL))
    nrows, ncols = H // cell, W // cell
    field = glyph_field(scene["glyph"])
    if (nrows, ncols) != (NROWS, NCOLS):
        # support alternate cell sizes by block-pooling the canonical glyph field
        ys = np.linspace(0, NROWS, nrows, endpoint=False).astype(int)
        xs = np.linspace(0, NCOLS, ncols, endpoint=False).astype(int)
        field = field[np.ix_(ys, xs)] if nrows > 0 and ncols > 0 else field
    up = np.repeat(np.repeat(field, cell, axis=0), cell, axis=1)[:H, :W]
    rng = np.random.default_rng(int(scene["noise"]) & 0xFFFFFFFF)
    noise7 = rng.integers(0, 128, size=(H, W)).astype(np.uint16)
    value = ((noise7 << 1) | up.astype(np.uint16)).astype(np.uint8)
    img = Image.fromarray(value, mode="L").convert("RGB")
    return img


def _bitfield_from_scene(scene: dict) -> np.ndarray:
    cell = int(scene.get("cell", CELL))
    nrows, ncols = H // cell, W // cell
    field = glyph_field(scene["glyph"])
    if (nrows, ncols) != (NROWS, NCOLS):
        ys = np.linspace(0, NROWS, nrows, endpoint=False).astype(int)
        xs = np.linspace(0, NCOLS, ncols, endpoint=False).astype(int)
        field = field[np.ix_(ys, xs)] if nrows > 0 and ncols > 0 else field
    return field


def margin(scene: dict) -> float:
    """A/B template correlation margin on the recovered bit-field, analytic."""
    field = _bitfield_from_scene(scene)
    chosen = _template_mask(scene["glyph"])
    other = _template_mask("B" if scene["glyph"] == "A" else "A")
    c_filled = int(chosen.sum())
    o_filled = int(other.sum())
    corr_chosen = float((field[chosen]).sum()) / max(1, c_filled)
    corr_other = float((field & other).sum()) / max(1, o_filled)
    return round(corr_chosen - corr_other, 6)


def analytic_gold(scene: dict) -> str:
    return scene["glyph"]


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < Q_TOL


def latent_symmetries(scene: dict):
    """not_applicable: the parametrization is minimal and fully visible. Every
    latent field (the per-pixel random noise, the answer glyph, the block size)
    directly changes distinct pixels of the raster, so no distinct latent vector
    reproduces the same image and no latent is hidden from rendering. Declared
    empty."""
    return iter(())
