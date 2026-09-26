> Migrated from the VLMH15 three-seed baseline. Historical SSOT section references are provenance; current authority is `docs/verification-boundary.md`.

# Question world: `angle_acuteness`

An angular-perception world. Two straight strokes of the **same colour** share a
vertex; the signed angular margin is

```
m = 90 - θ      (degrees)   (m > 0 acute, m = 0 right angle, m < 0 obtuse)
```

We can vary `m` continuously and ask *is the angle acute* — the bank's first
**angular** decision variable, measured in **degrees** rather than pixels. Every
other built or assigned world has a length/distance margin (signed gap, crossing
margin, length difference, containment distance); angular perception is a distinct
visual primitive, so this cell tests availability/continuity of a quantity the
distance worlds cannot reach. It probes the core *available-but-unused* failure
(SSOT §1.6) for a continuous *angle*.

## What this instance owns

| File | Role |
|---|---|
| `renderer.py` | scene sampling + rendering (design §1), degree-margin bins, butt-capped `ray_rect` strokes, counterfactual `rotate_ray2`, `scene_targets()` |
| `prompts.py` | 5 prompt families (§7.5; lexical acute/smaller-than-right/sharp, polarity obtuse, MC) + constrained answer sets + gold mapping |
| `oracle.py` | pixel oracle (§7.6): independent two-line-fit inverse renderer certifying image ↔ scene |
| `generate.py` | dataset + manifest generator (paired scene × family design, oracle-certified) |
| `verify.py` | re-certify an existing dataset against its manifest (CI/transfer gate) |
| `world.toml` | machine-readable metadata for the meta layer, incl. `[axes]` |
| `data/manifests/` | tracked JSONL ground-truth ledgers |
| `data/rendered/` | generated PNGs (git-ignored; regenerated on demand) |

The core/world boundary is load-bearing: nothing here imports another world, and
core imports nothing from here.

## Margin bins (§7.4)

Dense near the decision boundary `m = 0` (θ = 90°) — never sample only easy angles:

```
far_acute    m in [ 10,  60]   (θ in [30,  80])
near_acute   m in [  2,  10]   (θ in [80,  88])
right        m in [ -2,   2]   (θ in [88,  92])   # QUARANTINED (SSOT §21.6)
near_obtuse  m in [-10,  -2]   (θ in [92, 100])
far_obtuse   m in [-60, -10]   (θ in [100,150])
```

`renderer.MARGIN_SWEEP` gives the fixed degree sweep (+45 … 0 … −45) for continuity
analysis. Ray lengths are sampled in `[120, 205]` px (a **floor** that bounds the
heading-fit error — the render-robustness lever, design §5), stroke `[5, 9]` px,
heading and vertex position all randomized independently of the label. Only the
relative angle θ carries the answer (a shortcut block, SSOT §21.2).

## Oracle (§7.6)

An independent inverse renderer sharing no geometry with `render()`. Both rays are
the **same colour**, so the oracle splits the single "V" ink mask into two rays by
a deterministic two-line fit (fixed-seed RANSAC to bootstrap, then nearest-line
reassignment + PCA refit excluding the corner overlap) — it is **not** weakened to
two colours. The vertex is the intersection of the two fitted lines; `θ̂` is the
angle between the ray directions; `m̂ = 90 − θ̂`. It certifies the interior angle,
the vertex, both ray endpoints (heading + length), the acute/obtuse label
off-boundary, and an independent ink-area check. Tolerances are **empirically
calibrated** (`scripts/calibrate_angle_acuteness.py`; 1500 bin-sampled + 400
worst-case scenes; max |θ̂−θ| 0.40°, sign flip only at true |m| ≤ 0.10°) and pinned
by `tests/test_angle_acuteness_oracle.py` with ≥2× headroom. See the `oracle.py`
docstring for the full calibration table.

## Generate

Run as a module from the repo root. Records = scenes × prompt families (all
families share each scene's image — paired framing comparisons, §8.6); every image
is pixel-oracle-certified before its records are written (§7.6):

```bash
uv run python -m worlds.angle_acuteness.generate --n 200 --split smoke     # 200 scenes -> 1000 records
uv run python -m worlds.angle_acuteness.generate --n 20 --sweep --split sweep  # 20 scenes per fixed margin
uv run python -m worlds.angle_acuteness.verify worlds/angle_acuteness/data/manifests/smoke.jsonl
uv run python -m worlds.angle_acuteness.verify worlds/angle_acuteness/data/manifests/smoke.jsonl --repair
```

## Developing a successor

New candidates are created in isolated Harbor tasks. Only mechanically reviewed
and human-admitted candidates are imported into `worlds/`.
