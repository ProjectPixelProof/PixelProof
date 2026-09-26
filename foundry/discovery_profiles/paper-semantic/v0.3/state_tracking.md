# Paper semantic candidate B: State tracking

> **Author choice pending.** This is `paper-semantic@0.3.0:state_tracking`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Follow identity, position, and state changes across ordered static panels or a visible sequence of operations.

## Frozen candidate seed directions

- **`state_atomic_change`** (hypothesis@0.1.0:state_tracking): “Which object underwent the single declared visual change between the two panels?”
- **`state_cross_panel_identity`** (hypothesis@0.1.0:state_tracking): “Where does the initially ringed object appear in the final panel?”
- **`state_first_region_entry`** (hypothesis@0.1.0:state_tracking): “Which labeled object makes the earliest outside-to-inside transition into the shaded region across the ordered panels?”
- **`state_grid_toggles`** (hypothesis@0.1.0:state_tracking): “What is the final state of the ringed cell after the shown row and column toggles?”
- **`state_visible_swaps`** (hypothesis@0.1.0:state_tracking): “Starting from the ringed slot, where does the token finish after the visible swaps?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`census` from an eligible pool of
5. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must parse panel order, recover the declared per-step events, replay them from the marked initial state, and return the final identity, position, or state.

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
- Reject ambiguous panel order, unresolved swaps, occluded identities, simultaneous undeclared changes, and any sequence with more than one consistent replay.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
