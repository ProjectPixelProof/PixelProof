> Migrated from the VLMH15 three-seed baseline. Historical SSOT section references are provenance; current authority is `docs/verification-boundary.md`.

# Question world: `counting_with_distractors`

A counting / aggregation world. `N` target-coloured (red) filled disks and `D`
distractor disks of distinct palette colours are drawn on white; no two disks
overlap or touch. The closed-form gold is simply

```
count = N            (the number of red disks — a scene parameter, not a pixel measurement)
```

and the crowding margin is

```
m = min over ALL pairs of disks (targets AND distractors) of (centre distance - r_i - r_j)   (px, always > 0)
```

We ask *how many red dots are there* (and threshold framings) and vary `m` to
control the difficulty and pixel-verifiability of that count. This is the
bank's **first aggregation / counting world**: `two_circles` is fixed-arity with
a binary answer;
`counting_with_distractors` carries a variable object count, an integer answer, and
a margin defined over the whole set (design §6). It probes whether the model
preserves and causally uses a continuous *crowding* variable — the core
*available-but-unused* failure (historical SSOT §1.6) — and exercises the foundry's ability to
carry a non-yes/no constrained answer set end-to-end.

## What this instance owns

| File | Role |
|---|---|
| `renderer.py` | scene sampling + rendering (design §1), crowding-margin bins, close-pair placement, exact pairwise-gap geometry, `scene_targets()` |
| `prompts.py` | 5 prompt families (§7.5; lexical how-many/count, threshold >3 / <=3 / ==4) + constrained answer sets + gold keyed on the count N |
| `oracle.py` | pixel oracle (§7.6): independent connected-component inverse renderer that re-counts the disks and re-measures the margin |
| `generate.py` | dataset + manifest generator (paired scene × family design, oracle-certified) |
| `verify.py` | re-certify an existing dataset against its manifest (CI/transfer gate) |
| `world.toml` | machine-readable metadata for the meta layer, incl. `[axes]` |
| `data/manifests/` | tracked JSONL ground-truth ledgers |
| `data/rendered/` | generated PNGs (git-ignored; regenerated on demand) |

The core/world boundary is load-bearing: nothing here imports another world, and
core imports nothing from here.

## Colour palette (design §1)

Target `red`; distractors sampled WITHOUT replacement from
`("blue", "cyan", "green", "indigo", "magenta", "yellow")`; background `white`.
The palette is chosen **leak-free** for the oracle's nearest-colour labelling: on a
white canvas the only antialiased boundaries are ink↔white, and every ink's
rim→white blend labels only as that ink or white (no third colour captures the
rim). Min pairwise RGB distance is 145.8. A single benign artifact remains — the
red+blue → magenta averaging pixel — which is a few px of sub-threshold speckle the
oracle tolerates (see below).

## Crowding bins (§7.4)

Positive-only — the sign is fixed (all disks disjoint), difficulty rides on the
magnitude of the closest gap:

```
far_isolated  m in [ 40, 120]
spaced        m in [ 20,  40]
crowded       m in [  8,  20]
tight         m in [  3,   8]
boundary      m in [  0,   3]     # QUARANTINED (SSOT §21.6): below ~1-2 px two same-colour rims fuse
```

The sampler floor is `FLOOR_PX = 3`; bin-balanced generation rotates only over the
four packable bins (`SAMPLED_BINS`) and never enters the quarantined band, which is
the oracle's blind spot by design. `bin_for_m` still classifies `boundary` for
downstream analysis. `N`, `D` and the radii (`[14, 30]` px) are sampled
independently of `m` so no proxy (object count, total ink) is monotone with the
crowding difficulty (shortcut block, SSOT §21.2). Because a scene whose *minimum*
gap is large is necessarily sparse, the isolated bins pack fewer disks for the
largest counts — a geometric fact, not a shortcut; the scientifically-interesting
tight/crowded bins pack the full `N ∈ [2,7]` range with `m` decorrelated from `N`.

`renderer.MARGIN_SWEEP` gives the fixed positive sweep (`80 … 3, 2, 1`) for
continuity analysis. It deliberately probes below the floor into the quarantined
band; there the closest (min-gap) pair is forced to be a **target–distractor**
(cross-colour) pair, so target disks are never that close and the target component
count stays exactly `N` even at `m = 1` — every sweep point remains oracle-certified.

## Oracle (§7.6)

An independent inverse renderer sharing no geometry with `render()`. It labels
pixels by nearest palette colour, extracts **4-connectivity connected components**
of each colour mask (union-find; scipy is not a dependency), takes components of
area ≥ `MIN_DISK_PX` as disk candidates, and fits each with a Kåsa circle. It
asserts the target mask has exactly `N` disk candidates and each present distractor
colour exactly one (the label itself, not just a margin, is re-measured from
pixels), that each candidate is disk-like (fit residual + pixel area vs `π r²`),
that no candidate touches the border, that sub-threshold "speckle" ink stays below
`SPECKLE_TOL_PX` (benign LANCZOS ringing; a real colour leak paints a whole rim
arc and is caught), and it recomputes the minimum pairwise gap `m̂` from the
recovered disks and checks `|m̂ − m| ≤ M_TOL_PX` and `m̂ > 0`.

Tolerances are **empirically calibrated**
(`scripts/calibrate_counting_with_distractors.py`; 2400 bin-sampled + full sweep
scenes) and pinned by `tests/test_counting_with_distractors_oracle.py` with ≥2×
headroom. See the `oracle.py` docstring `CALIBRATION_NOTE` for the full table.

## Generate

Run as a module from the repo root. Records = scenes × prompt families (all
families share each scene's image — paired framing comparisons, §8.6); every image
is pixel-oracle-certified before its records are written (§7.6):

```bash
uv run python -m worlds.counting_with_distractors.generate --n 200 --split smoke     # 200 scenes -> 1000 records
uv run python -m worlds.counting_with_distractors.generate --n 20 --sweep --split sweep  # 20 scenes per fixed margin
uv run python -m worlds.counting_with_distractors.verify worlds/counting_with_distractors/data/manifests/smoke.jsonl
uv run python -m worlds.counting_with_distractors.verify worlds/counting_with_distractors/data/manifests/smoke.jsonl --repair
```

## Developing a successor

New candidates are created in isolated Harbor tasks. Only mechanically reviewed
and human-admitted candidates are imported into `worlds/`.
