# Paper semantic candidate B: Prior conflict

> **Author choice pending.** This is `paper-semantic@0.3.0:prior_conflict`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Report visible truth when the image may contradict a familiar canonical prior, using matched control and counterfactual variants with identical wording.

## Frozen candidate seed directions

- **`bias_checker_columns`** (blind_bias@0.1.0:vlms_are_biased): “How many columns are in the checkerboard?”
- **`bias_solar_planets`** (blind_bias@0.1.0:vlms_are_biased): “How many blue planets are visible?”
- **`bias_stop_sign_sides`** (blind_bias@0.1.0:vlms_are_biased): “How many sides does the STOP sign have?”
- **`bias_traffic_lamps`** (blind_bias@0.1.0:vlms_are_biased): “How many lamps are in the traffic signal?”
- **`bias_week_columns`** (blind_bias@0.1.0:vlms_are_biased): “How many columns are in the WEEK grid?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`seeded-sha256-rank` from an eligible pool of
10. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must measure the requested visible count, orientation, position, or target directly from the raster. The canonical prior is recorded for analysis but is never used as gold.

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
- Reject unmatched control/counterfactual styling, ambiguous counts or pointers, low-resolution marks, and any variant whose wording, palette, scale, or layout predicts whether it is canonical.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
