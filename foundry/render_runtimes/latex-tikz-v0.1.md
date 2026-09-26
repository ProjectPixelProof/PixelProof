# LaTeX/TikZ renderer capability v0.1

This Harbor task includes an offline LaTeX renderer in addition to the normal
Python, NumPy, and Pillow tools. It is an optional rendering mechanism, not a
different candidate contract or a shortcut around the pixel-only oracle.

## Supported surface

- `pdflatex` with core LaTeX, AMS math, recommended fonts and packages;
- TikZ/PGF and PGFPlots from `texlive-pictures`;
- ordinary tables, `booktabs`, `graphicx`, `xcolor`, and `geometry`;
- `render-latex-png SOURCE.tex OUTPUT.png` for one-page PDF-to-PNG rendering.

The wrapper fixes the rasterizer, requires one PDF page, disables shell escape,
uses paranoid TeX file-I/O policy, and imposes a compilation timeout. Compilation
must succeed with Harbor networking disabled. Unsupported packages must cause a
candidate to fail; never download packages or vendor an executable.

## Candidate boundary

The renderer may derive `.tex` from the sampled latent scene and rasterize it.
The analytic gold may use the latent scene. The independent inverse arm receives
only the final PNG and must not read `.tex`, PDF objects, auxiliary files,
renderer code, latent parameters, or analytic gold. Generated datasets must
retain the final PNGs; temporary TeX/PDF files are implementation artifacts.

Use a single page, explicit page geometry, fixed fonts and sizes, and short
labels. Avoid external commands, `\write18`, system fonts, remote assets,
time-dependent values, and packages outside the installed set.
