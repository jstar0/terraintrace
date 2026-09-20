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

## Quick start

```bash
uv sync --locked --dev
pnpm --dir web install --frozen-lockfile
pnpm --dir web build
uv run pytest
uv run --with ruff ruff check src tests
uv lock --check
uv build
uv run terraintrace benchmark --output-dir /tmp/terraintrace-benchmark
uv run terraintrace sweep --output-dir /tmp/terraintrace-sweep --seeds 17,29,41
uv run terraintrace case-export --output /tmp/terraintrace-case.zip --seed 17 --case-index 0
```

Start the local demonstration server after the package/UI build:

```bash
uv run terraintrace serve --host 127.0.0.1 --port 5206
```

After `pnpm --dir web build`, the `serve` command serves both the API and the
browser workbench at <http://127.0.0.1:5206/>. The wheel includes the built
static client. User datasets, runs, NPY arrays, and downloadable archives stay
under `runtime/terraintrace/` in the directory from which `serve` is started.

For live UI development, keep the API running in one terminal and start Vite in
another:

Terminal A:

```bash
uv run terraintrace serve --host 127.0.0.1 --port 5206
```

Terminal B:

```bash
pnpm --dir web dev
```

Open <http://127.0.0.1:5266/> for the proxied development client. The packaged
client at port 5206 is the local demonstration path.

## Browser workflow

Choose **打开合成示例** to save the fixed-seed dataset as a versioned local
revision. Select nearest, bilinear, and the train-only spectral prior, then
start a run. Completed cases appear by train/validation/test split. Each case
loads the server's full-resolution arrays: benchmark maps share one elevation
range, signed residual maps share one zero-centered range, and zoom, pan, and
the cursor stay aligned across panels.

Click **导入 NPZ** to add benchmark or LR-only input. Benchmark files contain
exactly one `high_res` array and need `id`, `group`, `split`, `scale`,
`pixel_size`, and `elevation_unit`; inference accepts one `low_res` array with
`id`, `scale`, `pixel_size`, and `elevation_unit`. The metadata editor shows a
starter template. Import validates the file and split boundary before creating
a new dataset revision. Benchmark scale comes from the saved case metadata;
the reconstruction methods are deterministic and do not take an unused
per-run random seed.

Benchmark metadata follows the selected file order:

```json
[
  {
    "id": "ridge-01",
    "group": "north-ridge",
    "split": "train",
    "scale": 4,
    "pixel_size": 30,
    "elevation_unit": "m"
  }
]
```

LR-only inference uses one object:

```json
{
  "id": "observation-01",
  "scale": 4,
  "pixel_size": 30,
  "elevation_unit": "m"
}
```

In **仅推断** mode, the workbench displays the LR observation as a block view
alongside nearest and bilinear outputs. It deliberately hides true-value
residuals and reports only LR consistency. For benchmark cases, drag or place
profile endpoints A/B on the maps, sample the shared path, and download the
same server-returned samples as CSV. The map context menu can set either
endpoint; with a handle selected, arrow keys nudge it by one cell. Press `0`
to fit the map, `[` / `]` to switch cases, and `Escape` to leave point/ROI mode.

The **全案例指标** view keeps each method and metric separate and sorts by the
chosen method and metric, without inventing a composite score. ROI min/max/mean
is labelled exploratory and never replaces full-case metrics. A benchmark
case archive includes the original target (when present), observed LR and
reconstruction arrays, metadata, metrics, shared-path profile CSV, static
HTML, and per-file digests; download is registered only after the archive
reopens and passes verification.

When a spectral prior is present, the detail rail also shows its train-only
radial gain curve, clip range, compatible input, and training array hashes.

The fixed seed 17 test split contains a reproducible metric tradeoff: in case
`2-03`, spectral-prior RMSE is about 1.6% lower than bilinear, while its edge
RMSE is about 8.4% higher. The workbench surfaces the actual test case so the
maps and profile can be inspected beside those figures. This is a synthetic
example, not evidence of real-terrain accuracy.

The local API is under `/api/v1`: `POST /terrain-datasets`, `GET
/terrain-datasets/{id}/{revision}`, `POST /reconstruction-runs`, job status,
events and cancel routes, run cases/metrics/arrays, server-side profiles and
ROI, and registered archive creation/download. Error responses use the shared
`error.code`, `error.message`, `field_errors`, and `retryable` envelope.

`benchmark` writes:

```text
benchmark.json    split metadata, method metrics, and per-case rows
per_case.csv      one row per split/case/method evaluation
report.md         test-summary table and leakage/consistency boundary
manifest.json     SHA-256 and byte sizes for every artifact
```

`sweep` repeats the complete grouped benchmark for explicit seeds and reports
mean/population-standard-deviation metrics. It is a stability descriptor, not a
confidence interval.

`case-export` writes one actual held-out test case: observed LR and all
reconstruction arrays, shared-path profile CSV, metadata, static HTML, and a
digest manifest. The command reopens the ZIP before reporting success.

## Model and evaluation contract

The default fixture contains twelve `48 × 48` synthetic high-resolution fields
at `×4` scale. Four cases each belong to one of three disjoint synthetic region
groups: `ridge-basin` (train), `coastal-shelf` (validation), and
`valley-plateau` (test). Low-resolution observations are non-overlapping area
means, not arbitrary image thumbnails.

The spectral prior estimates a bounded radial Fourier-amplitude gain from the
four training targets. At inference it modifies the bilinear spectrum, then
projects the result back to the observed cell means. This is intentionally
small enough to inspect line by line; it is not presented as a replacement for
an accepted terrain SR model or a physical sensor simulator.

## Evidence and limits

All arrays, groups, and metrics are synthetic. The benchmark does not claim
real terrain accuracy, cross-sensor generalization, official elevation data,
or operational value. A real-data experiment would need source/product
licenses, CRS and vertical-datum checks, NoData handling, spatially grouped
splits, and an independently frozen test manifest before any external claim.
See [`docs/architecture.md`](docs/architecture.md) and
[`docs/benchmark.md`](docs/benchmark.md) for the contracts.

MIT licensed. Contributions must preserve grouped splits, synthetic or openly
licensed data, deterministic seeds, and tests for both elevation and structure
metrics.
