"""Benchmark orchestration and hash-bound artifact writers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .fixture import TerrainDataset, create_dataset
from .methods import fit_spectral_prior, reconstruct_all
from .metrics import MetricSet, aggregate_metrics, evaluate_reconstruction

METHOD_ORDER = ("nearest", "bilinear", "spectral_prior")
METRIC_FIELDS = (
    "rmse",
    "mae",
    "slopeMae",
    "gradientCorrelation",
    "gradientCorrelationReason",
    "edgeRmse",
    "lowResolutionRmse",
    "areaConsistencyMax",
)


def benchmark_payload(seed: int = 17) -> dict[str, Any]:
    """Run validation and group-held-out test evaluation on the fixed fixture."""

    dataset = create_dataset(seed)
    prior = fit_spectral_prior(dataset.train)
    rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for split_name, cases in (("validation", dataset.validation), ("test", dataset.test)):
        split_summary: dict[str, dict[str, float]] = {}
        for method in METHOD_ORDER:
            metrics: list[MetricSet] = []
            for case in cases:
                prediction = reconstruct_all(case, prior)[method]
                metric = evaluate_reconstruction(prediction, case)
                metrics.append(metric)
                rows.append(
                    {
                        "split": split_name,
                        "caseId": case.id,
                        "group": case.group,
                        "method": method,
                        **metric.as_dict(),
                    }
                )
            split_summary[method] = aggregate_metrics(metrics).as_dict()
        summary[split_name] = split_summary
    test_summary = summary["test"]
    nearest_baseline = test_summary["nearest"]
    bilinear_baseline = test_summary["bilinear"]
    improvements = {
        method: {
            "rmseDeltaVsNearest": round(nearest_baseline["rmse"] - metrics["rmse"], 8),
            "rmseDeltaVsBilinear": round(bilinear_baseline["rmse"] - metrics["rmse"], 8),
            "slopeMaeDeltaVsNearest": round(nearest_baseline["slopeMae"] - metrics["slopeMae"], 8),
            "edgeRmseDeltaVsNearest": round(nearest_baseline["edgeRmse"] - metrics["edgeRmse"], 8),
        }
        for method, metrics in test_summary.items()
    }
    return {
        "benchmarkVersion": 1,
        "seed": seed,
        "scale": dataset.scale,
        "splitCounts": {
            "train": len(dataset.train),
            "validation": len(dataset.validation),
            "test": len(dataset.test),
        },
        "groups": {
            "train": [case.group for case in dataset.train],
            "validation": [case.group for case in dataset.validation],
            "test": [case.group for case in dataset.test],
        },
        "prior": {
            "bins": len(prior.gains),
            "radiusCenters": [round(float(value), 8) for value in prior.radius_centers],
            "gains": [round(float(value), 8) for value in prior.gains],
        },
        "methods": list(METHOD_ORDER),
        "summary": summary,
        "testImprovementsVsNearest": improvements,
        "rows": rows,
    }


def write_benchmark(output_dir: str | Path, seed: int = 17) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = benchmark_payload(seed)
    write_text(directory / "benchmark.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_rows_csv(directory / "per_case.csv", payload["rows"])
    write_text(directory / "report.md", markdown_report(payload))
    manifest = {
        "schemaVersion": 1,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
            for path in sorted(directory.glob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    write_text(directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return payload


def write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ("split", "caseId", "group", "method", *METRIC_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TerrainTrace benchmark",
        "",
        "A deterministic synthetic terrain reconstruction benchmark with group-held-out evaluation.",
        "",
        f"- seed / scale: `{payload['seed']} / x{payload['scale']}`",
        f"- split counts: `{payload['splitCounts']}`",
        f"- train-only spectral prior bins: `{payload['prior']['bins']}`",
        "",
        "## Test summary",
        "",
        "| method | RMSE | slope MAE | 2D gradient corr. | edge RMSE | LR RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in payload["methods"]:
        metrics = payload["summary"]["test"][method]
        gradient = (
            f"{metrics['gradientCorrelation']:.6f}"
            if metrics["gradientCorrelation"] is not None
            else f"N/A ({metrics['gradientCorrelationReason']})"
        )
        lines.append(
            f"| {method} | {metrics['rmse']:.6f} | {metrics['slopeMae']:.6f} | "
            f"{gradient} | {metrics['edgeRmse']:.6f} | "
            f"{metrics['lowResolutionRmse']:.9f} |"
        )
    lines.extend(
        [
            "",
            "## Split boundary",
            "",
            "Training groups fit the spectral prior; validation and test groups are disjoint from training and from "
            "each other.",
            "The low-resolution consistency metric is reported separately so a sharper-looking result cannot hide "
            "observation drift.",
            "",
            "All arrays and metrics are synthetic. This report is a reproducibility contract, not a claim about real "
            "terrain, sensors, or operational elevation accuracy.",
            "",
        ]
    )
    return "\n".join(lines)


def dataset_from_payload(payload: dict[str, Any]) -> TerrainDataset:
    """Provide a small type-checking hook for callers that need the fixture."""

    seed = payload.get("seed")
    if not isinstance(seed, int):
        raise ValueError("benchmark payload seed must be an integer")
    return create_dataset(seed)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
