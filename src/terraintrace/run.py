"""Array-backed reconstruction outputs and geometry helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .methods import (
    SpectralPrior,
    reconstruct_all,
    reconstruct_bilinear,
    reconstruct_inference,
    reconstruct_nearest,
)
from .metrics import MetricSet, evaluate_reconstruction
from .model import FloatArray, InferenceCase, TerrainCase, low_resolution_rmse


@dataclass(frozen=True, slots=True)
class ArrayArtifact:
    method: str
    values: FloatArray
    sha256: str
    value_range: tuple[float, float]
    low_resolution_rmse: float

    def metadata(self) -> dict[str, object]:
        return {
            "method": self.method,
            "shape": list(self.values.shape),
            "dtype": str(self.values.dtype),
            "sha256": self.sha256,
            "value_range": list(self.value_range),
            "low_resolution_rmse": self.low_resolution_rmse,
        }


@dataclass(frozen=True, slots=True)
class CaseRun:
    case_id: str
    mode: str
    arrays: dict[str, ArrayArtifact]
    metrics: dict[str, MetricSet | None]


def run_benchmark_case(
    case: TerrainCase,
    prior: SpectralPrior | None,
    methods: tuple[str, ...] = ("nearest", "bilinear", "spectral_prior"),
) -> CaseRun:
    allowed = {"nearest", "bilinear", "spectral_prior"}
    if not methods or not set(methods) <= allowed:
        raise ValueError("unsupported benchmark reconstruction method")
    if "spectral_prior" in methods and prior is None:
        raise ValueError("spectral_prior requires a fitted training prior")
    all_outputs: dict[str, FloatArray] = {}
    if "nearest" in methods:
        all_outputs["nearest"] = reconstruct_nearest(case)
    if "bilinear" in methods:
        all_outputs["bilinear"] = reconstruct_bilinear(case)
    if "spectral_prior" in methods and prior is not None:
        all_outputs["spectral_prior"] = reconstruct_all(case, prior)["spectral_prior"]
    outputs = {method: all_outputs[method] for method in methods}
    return CaseRun(
        case.id,
        "benchmark",
        {method: _artifact(method, values, case.low_res, case.scale) for method, values in outputs.items()},
        {method: evaluate_reconstruction(values, case) for method, values in outputs.items()},
    )


def run_inference_case(case: InferenceCase, methods: tuple[str, ...] = ("nearest", "bilinear")) -> CaseRun:
    outputs = reconstruct_inference(case, methods)
    return CaseRun(
        case.id,
        "inference",
        {method: _artifact(method, values, case.low_res, case.scale) for method, values in outputs.items()},
        {method: None for method in outputs},
    )


def profile(
    case_run: CaseRun,
    points: tuple[tuple[float, float], tuple[float, float]],
    samples: int = 128,
    pixel_size: float = 1.0,
    reference: FloatArray | None = None,
) -> dict[str, object]:
    if samples < 2 or samples > 2048 or pixel_size <= 0:
        raise ValueError("profile samples and pixel_size are invalid")
    start, end = points
    shape = next(iter(case_run.arrays.values())).values.shape
    if not (
        0 <= start[0] < shape[0] and 0 <= start[1] < shape[1] and 0 <= end[0] < shape[0] and 0 <= end[1] < shape[1]
    ):
        raise ValueError("profile points must be inside the reconstruction grid")
    rows = np.linspace(start[0], end[0], samples)
    cols = np.linspace(start[1], end[1], samples)
    distance = np.hypot(rows - rows[0], cols - cols[0]) * pixel_size
    series = {name: _sample(row.values, rows, cols).tolist() for name, row in case_run.arrays.items()}
    if reference is not None:
        if reference.shape != shape:
            raise ValueError("profile reference must match the reconstruction grid")
        series["high_res"] = _sample(reference, rows, cols).tolist()
    return {
        "points": [list(start), list(end)],
        "samples": samples,
        "distance": distance.tolist(),
        "series": series,
    }


def roi_metrics(case_run: CaseRun, bounds: tuple[int, int, int, int]) -> dict[str, object]:
    """Return local descriptive statistics without overwriting full-case evaluation metrics."""
    top, left, bottom, right = bounds
    shape = next(iter(case_run.arrays.values())).values.shape
    if not (0 <= top < bottom <= shape[0] and 0 <= left < right <= shape[1]):
        raise ValueError("ROI bounds must be ordered inside the reconstruction grid")
    return {
        "exploratory": True,
        "bounds": list(bounds),
        "methods": {
            method: {
                "min": float(artifact.values[top:bottom, left:right].min()),
                "max": float(artifact.values[top:bottom, left:right].max()),
                "mean": float(artifact.values[top:bottom, left:right].mean()),
            }
            for method, artifact in case_run.arrays.items()
        },
    }


def _artifact(method: str, values: FloatArray, low_res: FloatArray, scale: int) -> ArrayArtifact:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return ArrayArtifact(
        method,
        array,
        hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
        (float(array.min()), float(array.max())),
        low_resolution_rmse(array, low_res, scale),
    )


def _sample(values: FloatArray, rows: FloatArray, cols: FloatArray) -> FloatArray:
    top, left = np.floor(rows).astype(int), np.floor(cols).astype(int)
    bottom, right = np.minimum(top + 1, values.shape[0] - 1), np.minimum(left + 1, values.shape[1] - 1)
    vertical, horizontal = rows - top, cols - left
    return (
        values[top, left] * (1 - vertical) * (1 - horizontal)
        + values[bottom, left] * vertical * (1 - horizontal)
        + values[top, right] * (1 - vertical) * horizontal
        + values[bottom, right] * vertical * horizontal
    )
