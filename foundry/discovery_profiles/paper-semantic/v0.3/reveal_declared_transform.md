# Paper semantic candidate B: Reveal: declared transform

> **Author choice pending.** This is `paper-semantic@0.3.0:reveal_declared_transform`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Mentally execute a transformation explicitly declared by the question—rectify, unwrap, reassemble, difference, or blur—and read the benign abstract symbol that appears.

## Frozen candidate seed directions

- **`framed_perspective_anamorph`** (perceptual-reveal@0.1.0:declared_transform): “Which symbol appears when the marked quadrilateral is viewed front-on?”
- **`hybrid_lowpass_icon`** (perceptual-reveal@0.1.0:declared_transform): “Which symbol appears after blurring?”
- **`polar_ring_unwrap`** (perceptual-reveal@0.1.0:declared_transform): “Which symbol appears when the ring is unwrapped?”
- **`symmetry_difference_icon`** (perceptual-reveal@0.1.0:declared_transform): “Which symbol appears in the left-right difference?”
- **`visible_tile_reconstruction`** (perceptual-reveal@0.1.0:declared_transform): “Which symbol do the marked panels form?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`census` from an eligible pool of
5. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must locate visible fiducials, execute the declared raster transform, extract the transformed evidence, and classify the resulting symbol with a reported margin.

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
- Reject missing fiducials, transform ambiguity, unstable resampling, natural-language payloads, hidden keys, metadata channels, and symbols recoverable without the declared operation.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
