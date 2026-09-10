# TerrainTrace

TerrainTrace is a reproducible terrain-reconstruction lab. It treats
super-resolution as a measurement problem rather than an image-beautification
contest: a model must reduce elevation error, preserve slope and edge
structure, and remain consistent with the observed low-resolution cells.

## Why this is more than a resize demo

- **Grouped evaluation:** synthetic terrain families are held out by region
  group, so a train patch cannot quietly reappear in validation or test.
- **Observation consistency:** every reconstruction is projected back to the
  observed area means; the report exposes any remaining low-resolution drift.
- **Structure-aware metrics:** elevation RMSE/MAE, slope error, gradient
  correlation, Laplacian edge error, and low-resolution consistency are shown
  together.
- **Transparent baselines:** nearest-cell and bilinear interpolation are
  compared with a small radial spectral-detail prior fitted on training cases
  only. The prior is bounded and never sees test targets.
- **Reproducible evidence:** benchmark and multi-seed sweep commands emit
  JSON/CSV/Markdown plus SHA-256 manifests; repeated runs are byte-identical.
- **Clean-room boundary:** all surfaces are invented analytic fields. No
  customer/teacher raster, checkpoint, screenshot, or source implementation is
  included.
