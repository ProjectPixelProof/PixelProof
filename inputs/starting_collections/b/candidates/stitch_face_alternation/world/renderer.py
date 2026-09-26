"""Renderer for ``stitch_face_alternation``.

Latent scene: one thin thread stitched through one flat tape strip inside a
small local inset, crossing the tape exactly five times.  At each crossing the
thread is drawn either in FRONT of the tape (visible straight across it) or
BEHIND it (hidden by the tape, leaving a visible gap).  A small crimson tag
marks one end of the thread and fixes the walk direction.

Decision (analytic gold) = ``"<pattern>|<first_face>"`` where

* ``pattern``    = ``alternating`` iff, walking the thread from its tagged end,
  the five front/behind states strictly take turns; otherwise ``broken``;
* ``first_face`` = the state of the first crossing met from the tagged end.

The canonical prior under test is the running-stitch prior ("a stitch alternates
over and under"); it is recorded per scene but never used as gold.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

RENDERER_VERSION = "stitch_face_alternation-0.1.0"

WIDTH = 512
HEIGHT = 512
BACKGROUND = (255, 255, 255)
THREAD_RGB = (26, 42, 88)
TAPE_RGB = (198, 168, 120)
TAG_RGB = (196, 32, 48)

THREAD_W = 4
TAG_SIDE_PX = 9
N_CROSSINGS = 5

# Margin floors: every geometric clearance is measured against these, so the
# reported margin is zero exactly at the visible ambiguity event.
PITCH_FLOOR = THREAD_W + 6.0   # neighbouring crossings merge below this pitch
AMP_FLOOR = 8.0                # thread stub above/below the tape vanishes
BAND_FLOOR = 10.0              # a BEHIND gap stops being a visible interruption
CLEAR_FLOOR = 12.0             # a remote scrap starts touching the stitched inset

QUARANTINE_MARGIN = 6.0
TIGHT_BIN = (6.0, 12.0)

FACES = ("front", "behind")


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def sample_scene(seed: int) -> dict:
    """Deterministic latent scene for ``seed`` (JSON-serialisable dict)."""
    rng = random.Random(int(seed) * 2654435761 % (2**63))

    tight = rng.random() < 0.34
    m0 = rng.uniform(*TIGHT_BIN) if tight else rng.uniform(13.0, 24.0)
    limiting = rng.choice(("pitch", "amp", "band"))

    def slack() -> float:
        return m0 + rng.uniform(3.0, 12.0)

    pitch = PITCH_FLOOR + (m0 if limiting == "pitch" else min(slack(), 22.0))
    amp = AMP_FLOOR + (m0 if limiting == "amp" else min(slack(), 30.0))
    band = BAND_FLOOR + (m0 if limiting == "band" else min(slack(), 26.0))
    clear = CLEAR_FLOOR + m0 + rng.uniform(4.0, 16.0)

    pitch, amp, band, clear = (round(v, 3) for v in (pitch, amp, band, clear))

    # walk-order face sequence: control alternates, counterfactual repeats once.
    variant = "control" if rng.random() < 0.5 else "counterfactual"
    f0 = rng.randrange(2)
    walk = [FACES[(f0 + k) % 2] for k in range(N_CROSSINGS)]
    if variant == "counterfactual":
        r = rng.randrange(1, N_CROSSINGS)
        walk[r] = walk[r - 1]

    x_dir = 1 if rng.random() < 0.5 else -1      # path direction along x
    tag_at = 0 if rng.random() < 0.5 else 2 * N_CROSSINGS - 1  # tagged path end
    first_above = rng.random() < 0.5             # is vertex 0 above the tape?

    faces = list(walk) if tag_at == 0 else list(reversed(walk))

    tape_w = pitch * (N_CROSSINGS - 1) + 20.0
    pad = 14.0
    ox = rng.uniform(pad, WIDTH - tape_w - pad)             # tape left edge
    oy = rng.uniform(pad + amp, HEIGHT - pad - amp - band)  # tape top edge

    tx0, tx1 = ox, ox + tape_w
    ty0, ty1 = oy, oy + band
    y_top, y_bot = ty0 - amp, ty1 + amp

    # squared meander: five straight crossings of the tape, joined by runs that
    # stay clear of the tape, so every crossing is a clean vertical chord.
    xs = [tx0 + 10.0 + i * pitch for i in range(N_CROSSINGS)]
    if x_dir < 0:
        xs = list(reversed(xs))
    verts = []
    at_top = first_above
    for x in xs:
        verts.append([round(x, 3), round(y_top if at_top else y_bot, 3)])
        verts.append([round(x, 3), round(y_bot if at_top else y_top, 3)])
        at_top = not at_top

    inset = [tx0 - 6.0, ty0 - amp - 8.0, tx1 + 6.0, ty1 + amp + 8.0]
    distractors = _sample_distractors(rng, inset, band, clear)

    return {
        "verts": verts,
        "faces": faces,
        "tag_at": tag_at,
        "tape": [round(tx0, 3), round(ty0, 3), round(tx1, 3), round(ty1, 3)],
        "pitch": pitch,
        "amp": amp,
        "band": band,
        "clear": clear,
        "thread_w": THREAD_W,
        "tag_side_px": TAG_SIDE_PX,
        "distractors": distractors,
        "width": WIDTH,
        "height": HEIGHT,
        "variant": variant,
        "canonical_prior_answer": "alternating",
    }


def _sample_distractors(rng: random.Random, inset: list[float], band: float,
                        clear: float) -> list[dict]:
    """Remote, nondiagnostic marks: loose thread scraps and bare tape pieces.

    Every scrap keeps at least ``clear`` px from the stitched inset and from any
    other item, so no remote mark can be mistaken for the stitched tape.
    """
    boxes = [list(inset)]
    out: list[dict] = []
    want = rng.randrange(3, 6)
    for _ in range(600):
        if len(out) >= want:
            break
        kind = "scrap" if rng.random() < 0.6 else "tape"
        w = rng.uniform(34.0, 52.0) if kind == "scrap" else rng.uniform(28.0, 48.0)
        h = rng.uniform(30.0, 46.0) if kind == "scrap" else band
        x = rng.uniform(8.0, WIDTH - w - 8.0)
        y = rng.uniform(8.0, HEIGHT - h - 8.0)
        box = [x, y, x + w, y + h]
        if any(_box_gap(box, b) < clear for b in boxes):
            continue
        boxes.append(box)
        if kind == "tape":
            out.append({"kind": "tape", "rect": [round(v, 3) for v in box]})
        else:
            pts = [
                [round(x, 3), round(y + h * rng.uniform(0.0, 0.35), 3)],
                [round(x + w * rng.uniform(0.35, 0.65), 3), round(y + h, 3)],
                [round(x + w, 3), round(y + h * rng.uniform(0.0, 0.35), 3)],
            ]
            out.append({"kind": "scrap", "pts": pts})
    return out


def _box_gap(a: list[float], b: list[float]) -> float:
    dx = max(a[0] - b[2], b[0] - a[2], 0.0)
    dy = max(a[1] - b[3], b[1] - a[3], 0.0)
    if dx == 0.0 and dy == 0.0:
        return 0.0
    return (dx * dx + dy * dy) ** 0.5


# --------------------------------------------------------------------------- #
# canonicalisation, gold, margin
# --------------------------------------------------------------------------- #
def _canon(scene: dict) -> tuple[list[list[float]], list[str], str]:
    """Return (vertices left-to-right, faces left-to-right, tag side).

    An open thread has no intrinsic storage direction, so everything the
    renderer and gold use is derived from the physical left-to-right layout.
    """
    verts = [list(map(float, v)) for v in scene["verts"]]
    faces = list(scene["faces"])
    tag_at = int(scene["tag_at"])
    if verts[0][0] > verts[-1][0]:
        verts = list(reversed(verts))
        faces = list(reversed(faces))
        tag_at = len(verts) - 1 - tag_at
    tag_side = "left" if tag_at == 0 else "right"
    return verts, faces, tag_side


def _walk_faces(scene: dict) -> list[str]:
    _, faces, tag_side = _canon(scene)
    return faces if tag_side == "left" else list(reversed(faces))


def analytic_gold(scene: dict) -> str:
    walk = _walk_faces(scene)
    alternating = all(walk[i] != walk[i + 1] for i in range(len(walk) - 1))
    return f"{'alternating' if alternating else 'broken'}|{walk[0]}"


def margin(scene: dict) -> float:
    """Minimum visible clearance (canvas px) to any ambiguity event."""
    return round(
        min(
            float(scene["pitch"]) - PITCH_FLOOR,
            float(scene["amp"]) - AMP_FLOOR,
            float(scene["band"]) - BAND_FLOOR,
            float(scene["clear"]) - CLEAR_FLOOR,
        ),
        4,
    )


def margin_bin(scene: dict) -> str:
    m = margin(scene)
    if m < QUARANTINE_MARGIN:
        return "quarantine"
    if m < TIGHT_BIN[1]:
        return "tight"
    if m < 18.0:
        return "moderate"
    return "wide"


def is_quarantined(scene: dict) -> bool:
    return margin(scene) < QUARANTINE_MARGIN


def latent_symmetries(scene: dict) -> list[tuple[str, dict]]:
    """One nontrivial pixel-identical alias: read the same thread backwards.

    Reversing ``verts`` and ``faces`` together and remapping ``tag_at`` names the
    identical physical stitch traced from its other end.  The renderer draws from
    the canonical left-to-right layout, so the raster is byte-identical and gold
    must not move; a gold rule keyed to list position instead of to the tagged
    physical end would fail here.
    """
    alias = dict(scene)
    alias["verts"] = list(reversed([list(v) for v in scene["verts"]]))
    alias["faces"] = list(reversed(list(scene["faces"])))
    alias["tag_at"] = len(scene["verts"]) - 1 - int(scene["tag_at"])
    return [("thread_read_backwards", alias)]


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _crossing_runs(verts: list[list[float]]) -> list[tuple[list[float], list[float]]]:
    """The straight tape-crossing runs of the meander, in the given vertex order."""
    runs = []
    for a, b in zip(verts, verts[1:]):
        if abs(a[0] - b[0]) < 1e-6:
            runs.append((a, b))
    return runs


def _seg_clip_y(a: list[float], b: list[float], lo: float, hi: float):
    """Clip segment a->b (which spans lo..hi in y) to the y band [lo, hi]."""
    (x0, y0), (x1, y1) = a, b
    if y1 == y0:
        return None
    pts = []
    for yy in (lo, hi):
        t = (yy - y0) / (y1 - y0)
        if -0.05 <= t <= 1.05:
            t = min(max(t, 0.0), 1.0)
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    if len(pts) != 2:
        return None
    return pts


def render(scene: dict) -> Image.Image:
    img = Image.new("RGB", (int(scene["width"]), int(scene["height"])), BACKGROUND)
    draw = ImageDraw.Draw(img)
    tw = int(scene["thread_w"])
    verts, faces, _ = _canon(scene)
    tx0, ty0, tx1, ty1 = (float(v) for v in scene["tape"])

    # remote nondiagnostic marks first (they never touch the stitched inset)
    for d in scene["distractors"]:
        if d["kind"] == "tape":
            x0, y0, x1, y1 = (float(v) for v in d["rect"])
            draw.rectangle([x0, y0, x1, y1], fill=TAPE_RGB)
        else:
            pts = [(float(p[0]), float(p[1])) for p in d["pts"]]
            draw.line(pts, fill=THREAD_RGB, width=tw, joint="curve")

    # the thread, then the tape over it, then every FRONT crossing back on top
    draw.line([(v[0], v[1]) for v in verts], fill=THREAD_RGB, width=tw, joint="curve")
    draw.rectangle([tx0, ty0, tx1, ty1], fill=TAPE_RGB)
    for (a, b), face in zip(_crossing_runs(verts), faces):
        if face != "front":
            continue
        clip = _seg_clip_y(a, b, ty0 - 2.0, ty1 + 2.0)
        if clip is not None:
            draw.line(clip, fill=THREAD_RGB, width=tw)

    # the crimson tag on the marked end of the thread
    tag_at = int(scene["tag_at"])
    tag_v = [float(c) for c in scene["verts"][tag_at]]
    s = float(scene["tag_side_px"]) / 2.0
    draw.rectangle([tag_v[0] - s, tag_v[1] - s, tag_v[0] + s, tag_v[1] + s], fill=TAG_RGB)
    return img
