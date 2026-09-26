# Paper semantic candidate B: Correspondence

> **Author choice pending.** This is `paper-semantic@0.3.0:correspondence`, one of two
> deterministic five-seed alternatives. It is not the selected confirmatory
> treatment. Choose a final version before registering any paper campaign.

Design one **new, concise, code-renderable visual question** in the semantic
family below. The answer must be determined from the final pixels, and the
candidate must include a separately implemented pixel-only inverse arm.

## Target computation

Match parts, landmarks, or attributes across panels after a visible transformation, re-arrangement, or structural completion.

## Frozen candidate seed directions

- **`correspondence_attribute_binding`** (hypothesis@0.1.0:correspondence_binding): “What halo color belongs to the object matching the visual reference?”
- **`correspondence_attributed_graph`** (hypothesis@0.1.0:correspondence_binding): “Which candidate preserves the same colored-part adjacency graph as the reference?”
- **`correspondence_patch_completion`** (hypothesis@0.1.0:correspondence_binding): “Which candidate patch exactly completes the missing window so that every contour continues across the seam?”
- **`correspondence_rigid_multipart`** (hypothesis@0.1.0:correspondence_binding): “Which candidate is the same multipart object as the reference after rotation?”
- **`equation_fraction_binding`** (latex@0.1.0:structured_equations): “Which colored fraction has a numerator matching the green denominator?”

These five directions were selected after the signed exclusion pass using
`ascending-sha256(seed NUL profile_id NUL seed_id)` with sampling seed
`paper-semantic-v0.3.0-2026-08-09`. Selection mode for this profile is
`census` from an eligible pool of
5. The source references identify historical
direction provenance only. The builder does not receive executable prototypes,
images, latent scenes, gold answers, oracle outputs, or author-review packets.

## Required construction and inverse arm

The pixel arm must extract local parts and their relational structure, solve the cross-panel assignment, and read the requested attribute only after correspondence is established.

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
- Reject non-unique assignments, unmatched part counts, seam ambiguity, attribute leakage, or cases in which color or position alone identifies the answer.

## Forbidden shortcuts

Do not use a VLM judge as the inverse arm. Do not read renderer state, analytic
gold, source data, TeX/PDF objects, filenames, metadata, hidden payloads, exact
coordinates, or answer-specific styling. Do not reproduce a seed layout or
decision rule with superficial palette or geometry changes. Human review—not
the proposing agent—chooses the final paper profile version.
