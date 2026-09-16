# Contributing to TerrainTrace

Keep the benchmark synthetic, grouped by terrain family, and reproducible.

```bash
uv sync --locked --dev
uv run pytest
uv run --with ruff ruff check src tests
uv lock --check
uv build
uv run terraintrace benchmark --output-dir /tmp/terraintrace-benchmark
uv run terraintrace sweep --output-dir /tmp/terraintrace-sweep --seeds 17,29,41
```

New reconstruction methods need tests for shape, low-resolution consistency,
group leakage, deterministic output, and at least one structure metric. Do not
add customer rasters, teacher inputs, downloaded checkpoints, credentials, or
unreviewed third-party model code.
