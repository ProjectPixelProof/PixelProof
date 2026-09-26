# Paper semantic candidate B: Topology

> **Author choice pending.** This is `paper-semantic@0.3.0:topology`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Recover enclosure, holes, nesting depth, region connectivity, cycles, and the connectivity effect of removing a visible link.

## Frozen candidate seed directions

- **`blind_cycle_link`** (visual_bias@0.1.0:visual_bias_tasks): “Which colored link lies on a loop?”
- **`blind_maze_panels`** (visual_bias@0.1.0:visual_bias_tasks): “Which maze connects green to red: left, center, or right?”
- **`math_boundary_depth`** (scientific@0.2.0:diagrammatic_mathematics): “Which colored dot is enclosed by the same number of black loops as green?”
- **`topology_bridge_removal`** (hypothesis@0.1.0:global_topology): “If the highlighted bridge is removed, do the two markers remain connected?”
- **`topology_signature_match`** (hypothesis@0.1.0:global_topology): “Which candidate has the same numbers of connected components and holes as the reference?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`seeded-sha256-rank` from an eligible pool of
9. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must segment barriers or regions, construct component, enclosure, and adjacency relations, and compute the topological predicate without renderer state.

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
- Reject one-pixel bridges, near-tangent contours, open loops, marker-boundary contacts, and topology that changes under ordinary lossless decoding or small raster perturbations.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
