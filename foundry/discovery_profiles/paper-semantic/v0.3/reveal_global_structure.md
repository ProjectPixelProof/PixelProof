# Paper semantic candidate B: Reveal: global structure

> **Author choice pending.** This is `paper-semantic@0.3.0:reveal_global_structure`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Recover a benign abstract symbol or relation that emerges from global Gestalt organization: negative space, contour grouping, texture energy, continuation, constellation structure, or symmetry.

## Frozen candidate seed directions

- **`blind_symmetry_axis`** (visual_bias@0.1.0:visual_bias_tasks): “Which colored line is a symmetry axis?”
- **`dot_constellation_icon`** (perceptual-reveal@0.1.0:global_structure): “Which symbol do the connected dots outline?”
- **`negative_space_cavity`** (perceptual-reveal@0.1.0:global_structure): “Which shape is formed by the empty region?”
- **`occluded_continuation`** (perceptual-reveal@0.1.0:global_structure): “Which symbol continues behind the bars?”
- **`texture_energy_silhouette`** (perceptual-reveal@0.1.0:global_structure): “Which shape is hidden in the texture?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`seeded-sha256-rank` from an eligible pool of
6. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must reconstruct the declared global evidence—foreground complement, contour graph, texture-energy field, continuation graph, proximity graph, or symmetry score—and classify only the resulting structure.

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
- Reject natural-language payloads, metadata, tiny local codes, ambiguous shapes, fixed templates, and cases where evidence erasure does not remove or materially weaken the answer.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
