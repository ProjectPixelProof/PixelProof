# Deviations for global_elongation_axis

No deviations from protocol v0.4 requirements.

- Candidate is the sole `exactly_one_per_session` entry.
- The independent pixel-only inverse arm (`world/oracle.py`) imports only numpy/PIL and
  never receives the latent scene or analytic gold.
- `latent_alias.mode` is `not_applicable`: rendering is injective over sampled scenes
  and `latent_symmetries` returns an empty list.
- The decision is the principal second-moment (variance) stretch axis of the whole dot
  cloud, recovered from the ink mass in the raster; scenes whose axis lies within the
  5-degree quarantine band of a class boundary, or whose cloud is not elongated
  (major/minor variance ratio below 1.8), are quarantined.
