# Deviations

None that affect the contract. Implementation notes:

- The optional LaTeX/TikZ renderer was not used. The world is a plain lattice of
  square cells plus two marker glyphs, whose exact fill colours and aliasing-free
  edges are easier to control through Pillow, and the pixel arm depends on that
  colour separation rather than on any typeset structure.
- The five candidate symbols are named and described in words in the question
  rather than printed as a legend, so the pixel arm carries its own independent
  transcription of those verbal descriptions. A test asserts that transcription
  agrees cell for cell with the renderer's independently written templates.
- All five symbol templates use exactly 16 dark cells, and the discarded decoy
  sheet is always a different symbol, so total ink is constant across answers.
