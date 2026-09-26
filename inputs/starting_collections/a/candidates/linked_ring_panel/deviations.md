# Deviations

None from the protocol v0.4 candidate contract.

Notes that are not deviations:

- The optional offline LaTeX/TikZ renderer was deliberately not used. The whole
  decision is which ring's ink survives unbroken at a crossing, so every ink
  pixel must be an exact flat colour under my own control; the raster is
  composed directly with NumPy masks and no antialiasing, and `latex_used` is
  false in `provenance.json`.
- `margin` is reported only by the analytic arm, in pixels. The pixel arm
  returns a decision and abstains as `ambiguous`; it writes no margin into the
  manifest.
- No sampled scene is quarantined: the sampler requires every stacked pair to
  keep an under-ring fragment of at least 26 px between its two breaks, so the
  analytic margin never falls below that. The quarantine and abstention paths
  are implemented and exercised on constructed scenes in
  `tests/test_candidate.py`.
