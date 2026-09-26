# Deviations

No deviations from the question-world@0.4.0 candidate contract are declared.

- The required schema versions, manifest keys, prompt APIs, and world interface
  are followed exactly.
- The latent-alias gate uses `tested_transforms` with a declared pixel-identical
  `dot_order` permutation; the analytic gold reads only `dots`, so
  `label_inputs == analytic-gold scene access`.
- The LaTeX renderer was not used; the radial dot field is rendered directly
  with Pillow because the discrete radial-density Gestalt reads best as plain
  dots.
