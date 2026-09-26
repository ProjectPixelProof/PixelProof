> Migrated from the VLMH15 three-seed baseline. Historical SSOT section references are provenance; current authority is `docs/verification-boundary.md`.

# Question world: `two_circles`

The minimal first testbed. Two circles are separate, tangent, or overlapping;
the signed gap is

```
m = ||c1 - c2||_2 - r1 - r2      (m > 0 separate, m = 0 touching, m < 0 overlapping)
```

We can vary `m` continuously and ask whether the model's hidden states and output
logits change smoothly and monotonically — the core probe of the
*available-but-unused* visual-state failure (SSOT §1.6).

## What this instance owns

| File | Role |
|---|---|
| `renderer.py` | scene sampling + rendering (§7.3–7.4), margin bins, counterfactual translate, `scene_targets()` |
| `prompts.py` | 9 prompt families (§7.5; incl. the lexical×polarity 2×2 pf7–pf9, SSOT §7.2 axis 4) + adversarial variants, constrained answer sets, ground-truth mapping |
| `oracle.py` | pixel oracle (§7.6): independent inverse renderer certifying image ↔ scene |
| `generate.py` | dataset + manifest generator (paired scene × family design, oracle-certified) |
| `verify.py` | re-certify an existing dataset against its manifest (CI/transfer gate) |
| `world.toml` | machine-readable metadata for the meta layer |
| `data/manifests/` | tracked JSONL ground-truth ledgers |
| `data/rendered/` | generated PNGs (git-ignored; regenerated on the box) |

It depends on the task-agnostic core only for the manifest schema. Downstream
model scoring and instrumentation are intentionally outside this migration.

## Margin bins (§7.4)

Dense near the decision boundary — never sample only easy examples:

```
far_separate  m in [ 20,  80]
near_separate m in [  1,  20]
tangent       m in [ -1,   1]
near_overlap  m in [-20,  -1]
deep_overlap  m in [-80, -20]
```

`renderer.MARGIN_SWEEP` gives the fixed sweep (+50 … 0 … −50) for continuity analysis.

## Generate

Run as a module from the repo root. Records = scenes × prompt families (all
families share each scene's image — paired framing comparisons, §8.6); every
image is pixel-oracle-certified before its records are written (§7.6):

```bash
uv run python -m worlds.two_circles.generate --n 200 --split smoke     # 200 scenes -> 1200 records
uv run python -m worlds.two_circles.generate --n 200 --split lexical \
  --families pf7_no_gap_yesno,pf8_not_overlapping_yesno,pf9_apart_yesno
uv run python -m worlds.two_circles.generate --n 10000 --split pilot
uv run python -m worlds.two_circles.generate --n 30 --sweep --split sweep  # 30 scenes per fixed margin
uv run python -m worlds.two_circles.verify worlds/two_circles/data/manifests/smoke.jsonl
uv run python -m worlds.two_circles.verify worlds/two_circles/data/manifests/smoke.jsonl --repair
```

## Developing a successor

New candidates are created in isolated Harbor tasks, not directly under
`worlds/`. After mechanical review and human admission, an accepted bundle can
be imported here with its provenance and tests.
