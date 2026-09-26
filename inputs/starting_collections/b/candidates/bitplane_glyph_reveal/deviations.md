# Deviations

- None from the question-world@0.4.0 executable candidate contract.
- The LaTeX/TikZ renderer was not used: the declared transform is an integer
  least-significant-bit-plane slice best expressed with direct numpy value
  arithmetic, which keeps the renderer and the independent inverse arm crisp,
  deterministic, and free of resampling instability.
- The answer symbols A and B are fixed block letters declared in the question
  and prompt contract; they are not derived from per-scene latent data.
