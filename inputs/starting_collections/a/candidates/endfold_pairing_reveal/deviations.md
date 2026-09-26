# Deviations

No deviations from the protocol v0.4 candidate contract are claimed.

Implementation notes that are intentional, not deviations:

- The optional LaTeX/TikZ renderer was not used. The scene is built from flat,
  non-antialiased polyline and ellipse fills so that the cord is one exact flat
  teal and every bead one exact flat gray, and the raster is byte-reproducible.
- `latent_alias.mode` is `not_applicable`: `latent_symmetries(scene)` returns an
  empty list because every latent field is rendered and analytic gold reads only
  `tones`, the 50 bead grays in walking order, which the raster exposes exactly.
  This is declared as `not_applicable`, not as a passing symmetry gate. Walk
  reversal is not a latent alias: end-to-end folding is symmetric under
  reversal, so the reversed scene renders a different raster with the same gold,
  and a test asserts that the pixel arm still recovers it.
- The pixel oracle can return `abstain`. Abstention is never a gold decision; it
  is the required behaviour when the teal cord does not resolve into exactly 50
  bead components adjacent to the cord, when the traced component is branched or
  does not have exactly two ends, when the bracketed start bead is not
  identifiable, or when the folded pattern is not exactly one declared glyph.
- The support geometry is ablation aware by design and asserted by tests rather
  than declared in metadata. On all 72 public stress scenes the tests check that
  the cord and its beads form a single connected trace, that this trace spans at
  least 70% of the canvas in both axes while staying at least 20 px from every
  edge, and that it fills less than 20% of its own bounding box, so the decisive
  support is one thin elongated component reaching across widely separated parts
  of the picture rather than a compact blob. Separate tests check that whiting
  out the left, right, top, or bottom half of the picture forces abstention,
  that erasing any of three distinct interior stretches of the cord forces
  abstention, and that erasing the tone key leaves the answer unchanged.
- The two-swatch key is placed in a corner the cord does not reach, so it never
  slices into the decisive trace. It makes the per-bead tone boundary visible to
  a reader, but the pixel oracle deliberately does not use it: it splits the 50
  bead readings
  at the midpoint of their own observed range. The key is therefore decoration
  for the oracle and is not part of the decisive support.
- Scene sampling rejects bead tones for which the raw first half, the raw second
  half, the reversed second half, the half-on-half pairing, or the neighbour
  pairing would already spell a declared glyph, so the cheap alternatives to the
  declared fold cannot recover the answer. Cheap local cues are counterbalanced:
  every glyph marks exactly 9 of 25 positions, every scene has 50 beads, 49 cord
  legs and 16 unmarked pairs, bead and cord ink are constant, and the dark-bead
  totals of the four answers overlap.
- The two prompt families share one image per scene and one decision variable,
  as the contract requires; each has at least two off-quarantine answer classes
  in the submitted evidence.
