# Paper semantic candidate B: Tracing

> **Author choice pending.** This is `paper-semantic@0.3.0:tracing`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Follow a visible path, stroke, curve, or network to recover connectivity, order, terminal identity, bend count, or geodesic length.

## Frozen candidate seed directions

- **`blind_network_farthest`** (visual_bias@0.1.0:visual_bias_tasks): “Which terminal is farthest from green along the lines?”
- **`blind_required_node`** (visual_bias@0.1.0:visual_bias_tasks): “Which colored node lies on the green-to-red route?”
- **`figure_trace_terminal`** (latex@0.1.0:dense_scientific_figures): “Which colored terminal belongs to the green-started trace?”
- **`path_bend_comparison`** (hypothesis@0.1.0:path_tracing): “Between the two ringed routes, which route has fewer bends from start to terminal?”
- **`path_ordered_checkpoint`** (hypothesis@0.1.0:path_tracing): “Starting at the ringed port, which marked checkpoint is encountered first along the path?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`seeded-sha256-rank` from an eligible pool of
13. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must segment the relevant strokes and markers, reconstruct a connectivity graph or ordered centerline, and then compute the requested terminal, order, bend, or along-path distance.

The renderer and inverse arm must be independent: the inverse may receive only
the final PNG and public question contract. Generate balanced randomized scenes,
report a runner-up decision margin, and abstain or quarantine ambiguous cases.
Question wording must declare every rule needed to solve the image.

## Protected validity requirements

- Analytic scene answer and pixel-only answer must agree over deterministic
  stress samples before submission.
- Counterbalance answer identity against color, position, size, count, total
  ink, panel order, and other cheap local cues.
- Label-preserving palette, translation, and distractor permutations must
  preserve the answer; declared label-transforming symmetries must transform it
  exactly as specified.
- Evidence erasure must change the answer, cause abstention, or materially
  collapse the decision margin.
- Reject unresolved crossings, broken centerlines, touching markers, ambiguous branch ownership, and path comparisons without a stable pixel margin.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
