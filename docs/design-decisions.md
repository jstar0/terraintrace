# TerrainTrace design decisions and limits

TerrainTrace is a synthetic terrain-reconstruction benchmark, not a geospatial or remote-sensing product.

## Decisions embodied in the implementation

1. **Splits are region-group disjoint.** Rejected alternative: random patch splits that let related terrain appear in train and test.
2. **Spectral priors fit training cases only.** Rejected alternative: estimate a reconstruction prior from validation or test terrain.
3. **Nearest and bilinear methods remain explicit baselines.** Rejected alternative: claim a structure-aware method without simple comparisons.
4. **Metrics include slope, edge, gradient and observed-cell consistency alongside RMSE.** Rejected alternative: optimize pixel error only.
5. **Seed sweeps and hash-bound artifacts expose fixture repeatability.** Rejected alternative: publish a single favorable reconstruction.

## Known weaknesses

1. **Terrain, degradation, regions, and observations are synthetic.**
2. **The spectral prior is a small benchmark method, not a learned super-resolution model.**
3. **Metrics do not establish map accuracy, elevation datum validity, or field safety.**
4. **Area-mean degradation omits sensor noise, occlusion, registration and geospatial projection errors.**
5. **Grouped synthetic regions do not prove geographic generalization.**
6. **No external DEM, raster IO, coordinate system, provenance, or visualization workflow is included.**
7. **Seed stability is fixture evidence, not remote-sensing performance.**

## Interview-safe description

Describe TerrainTrace as a reproducible synthetic structure-aware reconstruction benchmark with group-isolated evaluation.
