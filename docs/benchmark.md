# Benchmark contract

Run two identical benchmark directories and compare every public artifact:

```bash
uv run terraintrace benchmark --output-dir /tmp/terraintrace-a --seed 17
uv run terraintrace benchmark --output-dir /tmp/terraintrace-b --seed 17
cmp /tmp/terraintrace-a/benchmark.json /tmp/terraintrace-b/benchmark.json
cmp /tmp/terraintrace-a/per_case.csv /tmp/terraintrace-b/per_case.csv
cmp /tmp/terraintrace-a/report.md /tmp/terraintrace-b/report.md
cmp /tmp/terraintrace-a/manifest.json /tmp/terraintrace-b/manifest.json
```

Run the seed stability check:

```bash
uv run terraintrace sweep --output-dir /tmp/terraintrace-sweep-a --seeds 17,29,41
uv run terraintrace sweep --output-dir /tmp/terraintrace-sweep-b --seeds 17,29,41
cmp /tmp/terraintrace-sweep-a/sweep.json /tmp/terraintrace-sweep-b/sweep.json
cmp /tmp/terraintrace-sweep-a/sweep.csv /tmp/terraintrace-sweep-b/sweep.csv
cmp /tmp/terraintrace-sweep-a/report.md /tmp/terraintrace-sweep-b/report.md
cmp /tmp/terraintrace-sweep-a/manifest.json /tmp/terraintrace-sweep-b/manifest.json
```

The test split is a held-out synthetic region group. The spectral prior is fit
only on the four training cases. Each method is evaluated on the same observed
low-resolution cells and reports both target error and area-mean consistency.

The benchmark is a reproducibility contract, not a wall-clock performance
claim, a physical terrain simulator, or evidence of real cross-source
generalization. Any future real-data result must name the product/license,
CRS/vertical datum, grouped manifest, seed, metric implementation, and device.
