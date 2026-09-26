# Deviations

None from the protocol v0.4 candidate contract.

Notes that are not deviations:

- The optional LaTeX/TikZ renderer was available but deliberately unused: the
  decision rests on exact occlusion of thin thread ink by flat tape ink, and a
  direct Pillow raster keeps every crossing state colour-classifiable without a
  PDF rasterisation stage.
- Rendering is done at 1x with no antialiasing so that tape, thread and tag ink
  stay exactly colour-separable for the independent pixel-only oracle. Marks are
  well above pixel scale (thread width 4 px, tape band >= 16 px, tag 9 px).
- The canonical running-stitch prior is stored per scene as
  `canonical_prior_answer` for analysis only; analytic gold never reads it.
- Developmental descriptor preview (`pixel-oracle-support@0.5.0`, own gallery,
  status `complete`, nonempty-scene fraction 1.0): the measured cell is
  `support_scale=focal|support_shape=compact`, i.e. it matches the requested
  scale but not the requested `pathlike` shape. The decisive support is one
  connected component (`connected_components=1`) with median elongation ~2.4 and
  bounding-box fill 1.0: the stitched strip and its thread stubs are thin in
  absolute pixels but, at the 8x8 ablation grid, occupy a filled two-row block
  rather than a one-cell-thick chain. Reaching a one-cell-thick chain would
  require either a strip long enough to leave the focal scale or a diagonally
  routed strip with a rotation-general pixel arm; that redesign was not attempted
  inside this session's budget. This is reported as measured, not relabelled: no
  metadata field claims the requested cell.
