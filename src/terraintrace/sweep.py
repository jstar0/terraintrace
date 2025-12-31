"""Multi-seed stability artifacts for TerrainTrace."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Any

from .benchmark import METHOD_ORDER, benchmark_payload


def parse_seed_list(value: str) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("seeds must be a comma-separated list of integers")
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("seeds must be a comma-separated list of integers") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def seed_sweep_payload(seeds: Iterable[int]) -> dict[str, Any]:
    """Evaluate the grouped fixture across explicit independent seeds."""

    seed_tuple = tuple(seeds)
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_tuple):
        raise ValueError("seeds must be integers")
    runs: list[dict[str, Any]] = []
    for seed in seed_tuple:
        payload = benchmark_payload(seed)
        test_summary = payload["summary"]["test"]
        runs.append(
            {
                "seed": seed,
                **{f"{method}Rmse": test_summary[method]["rmse"] for method in METHOD_ORDER},
                **{f"{method}SlopeMae": test_summary[method]["slopeMae"] for method in METHOD_ORDER},
                "spectralRmseDeltaVsBilinear": payload["testImprovementsVsNearest"]["spectral_prior"][
                    "rmseDeltaVsBilinear"
                ],
            }
        )
    numeric_fields = tuple(field for field in runs[0] if field != "seed")
    aggregate: dict[str, dict[str, float]] = {}
    for field in numeric_fields:
        values = [float(run[field]) for run in runs]
        mean = fmean(values)
        variance = fmean((value - mean) ** 2 for value in values)
        aggregate[field] = {"mean": round(mean, 8), "pstdev": round(sqrt(variance), 8)}
    return {
        "sweepVersion": 1,
        "seeds": list(seed_tuple),
        "methods": list(METHOD_ORDER),
        "runs": runs,
        "aggregate": aggregate,
    }


def write_sweep(output_dir: str | Path, seeds: Iterable[int]) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = seed_sweep_payload(seeds)
    write_text(directory / "sweep.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(directory / "sweep.csv", payload["runs"])
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


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    if not runs:
        raise ValueError("runs must not be empty")
    fields = tuple(runs[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# TerrainTrace seed sweep",
        "",
        "The grouped synthetic terrain fixture evaluated across independent seeds.",
        "",
        f"- seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        "",
        "| seed | nearest RMSE | bilinear RMSE | spectral-prior RMSE | spectral-prior slope MAE |",
        "|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        lines.append(
            f"| {run['seed']} | {run['nearestRmse']:.6f} | {run['bilinearRmse']:.6f} | "
            f"{run['spectral_priorRmse']:.6f} | {run['spectral_priorSlopeMae']:.6f} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for field, summary in payload["aggregate"].items():
        lines.append(f"- `{field}` — mean `{summary['mean']:.8f}`, population σ `{summary['pstdev']:.8f}`")
    lines.extend(
        [
            "",
            "Seed variation is reported as a stability descriptor over invented surfaces, not a confidence interval or "
            "a claim about real sensor generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
