# Paper semantic candidate B: Search

> **Author choice pending.** This is `paper-semantic@0.3.0:search`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Find the unique element satisfying a visible property or matching a reference among structured distractors.

## Frozen candidate seed directions

- **`search_closed_contour`** (hypothesis@0.1.0:visual_search): “Which quadrant contains the only completely closed contour?”
- **`search_conjunction`** (hypothesis@0.1.0:visual_search): “Which region contains the object matching both attributes of the visual reference?”
- **`search_contour_in_clutter`** (hypothesis@0.1.0:visual_search): “Which sector contains the exact reference contour embedded in the clutter?”
- **`search_guided_count_comparison`** (hypothesis@0.1.0:visual_search): “Which half contains more elements matching the orientation shown in the reference box?”
- **`search_orientation_oddball`** (hypothesis@0.1.0:visual_search): “Which quadrant contains the uniquely oriented dark bar?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`census` from an eligible pool of
5. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must enumerate candidate regions, measure the declared visual attributes or contour match for every candidate, and require a unique winner with a runner-up margin.

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
- Reject accidental single-feature pop-out, tied candidates, incomplete contours, answer-position imbalance, and targets identifiable from count, color, or salience alone.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
