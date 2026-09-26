# Public scientific gates under protocol v0.4

1. **Safe artifact:** ordinary bounded files only; no path escapes, credentials,
   symlinks, devices, or writes outside declared output.
2. **Contract:** candidate and portfolio schemas, IDs, interfaces, fingerprints,
   contrasts, and declared memories are valid.
3. **Executable:** tests, generation, exact-evidence verification, and standalone
   verification execute offline.
4. **Deterministic:** identical inputs produce byte-identical images and manifests.
5. **Analytic consistency:** stored decisions, margins, and answers equal
   independently recomputed consequences of the scene.
6. **Render fidelity:** re-rendering a stored scene reproduces the raster.
7. **Sign-blind oracle:** the decision is recoverable from the image alone.
8. **Oracle independence:** the image path receives no scene and imports no
   renderer, prompt, generator, verifier, or gold code.
9. **Latent alias:** this gate is tri-state:
   - `pass`: nontrivial declared pixel-identical transforms preserve gold;
   - `not_applicable`: the renderer declares no latent aliases, returns no
     transforms, exposes all label inputs, passes a rendered-field audit, and
     produces no sampled pixel collision with different gold;
   - `fail`: a declaration is inconsistent or an alias changes the answer.
   Both `pass` and `not_applicable` are mechanically acceptable and are reported
   distinctly.
10. **Observable boundary:** the image exposes the decision boundary.
11. **Quarantine:** finite-resolution boundary cases are explicit and excluded
    from headline claims.
12. **Class balance:** each prompt family has at least two off-quarantine classes.
13. **Corruption coverage:** altered labels and materially corrupted rasters fail.
14. **Gallery coverage:** actual images cover prompt families, classes, and
    near-boundary cases.
15. **Exact evidence:** the protected probe executes over the submitted evidence,
    in addition to hidden deterministic stress seeds.
16. **Semantic review readiness:** the controlled fingerprint is valid, the
    independently retrieved nearest neighbor is addressed, and sibling
    similarities are disclosed.

Gates 1–15 determine mechanical eligibility, with latent-alias `not_applicable`
accepted but never relabeled as `pass`. Gate 16 and semantic duplicate risk are
diagnostics for blinded review; they do not mechanically certify novelty.

Process integrity, ranking-pass compliance, and search budgets measure the outer
loop. They never compensate for a failed scientific gate.
