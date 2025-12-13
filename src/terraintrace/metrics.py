"""Structure-aware metrics for synthetic terrain reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean

import numpy as np

from .model import FloatArray, TerrainCase, area_downsample, low_resolution_rmse


@dataclass(frozen=True, slots=True)
class MetricSet:
    """Metrics kept together so reports cannot silently mix protocols."""

    rmse: float
    mae: float
    slope_mae: float
    gradient_correlation: float | None
    gradient_correlation_reason: str | None
    edge_rmse: float
    low_resolution_rmse: float
    area_consistency_max: float

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "rmse": round(self.rmse, 8),
            "mae": round(self.mae, 8),
            "slopeMae": round(self.slope_mae, 8),
            "gradientCorrelation": round(self.gradient_correlation, 8)
            if self.gradient_correlation is not None
            else None,
            "gradientCorrelationReason": self.gradient_correlation_reason,
            "edgeRmse": round(self.edge_rmse, 8),
            "lowResolutionRmse": round(self.low_resolution_rmse, 8),
            "areaConsistencyMax": round(self.area_consistency_max, 8),
        }


def evaluate_reconstruction(prediction: FloatArray, case: TerrainCase) -> MetricSet:
    """Evaluate elevation fidelity, terrain structure, and LR consistency."""

    predicted = np.asarray(prediction, dtype=np.float64)
    truth = case.high_res
    if predicted.shape != truth.shape:
        raise ValueError("prediction shape must match the high-resolution target")
    if not np.isfinite(predicted).all():
        raise ValueError("prediction must contain only finite values")
    difference = predicted - truth
    truth_gradient = _gradient(truth, case.pixel_size)
    predicted_gradient = _gradient(predicted, case.pixel_size)
    truth_slope = np.hypot(truth_gradient[0], truth_gradient[1])
    predicted_slope = np.hypot(predicted_gradient[0], predicted_gradient[1])
    truth_edges = _laplacian(truth, case.pixel_size)
    predicted_edges = _laplacian(predicted, case.pixel_size)
    reduced = area_downsample(predicted, case.scale)
    consistency = float(np.max(np.abs(reduced - case.low_res)))
    gradient_correlation, gradient_reason = _correlation(
        np.concatenate((predicted_gradient[0].ravel(), predicted_gradient[1].ravel())),
        np.concatenate((truth_gradient[0].ravel(), truth_gradient[1].ravel())),
    )
    return MetricSet(
        rmse=float(np.sqrt(np.mean(difference**2))),
        mae=float(np.mean(np.abs(difference))),
        slope_mae=float(np.mean(np.abs(predicted_slope - truth_slope))),
        gradient_correlation=gradient_correlation,
        gradient_correlation_reason=gradient_reason,
        edge_rmse=float(np.sqrt(np.mean((predicted_edges - truth_edges) ** 2))),
        low_resolution_rmse=low_resolution_rmse(predicted, case.low_res, case.scale),
        area_consistency_max=consistency,
    )


def aggregate_metrics(metrics: list[MetricSet] | tuple[MetricSet, ...]) -> MetricSet:
    """Average metrics over cases without dropping structure diagnostics."""

    if not metrics:
        raise ValueError("at least one metric set is required")
    correlations = [metric.gradient_correlation for metric in metrics]
    missing_correlation = next(
        (metric.gradient_correlation_reason for metric in metrics if metric.gradient_correlation is None),
        None,
    )
    return MetricSet(
        rmse=fmean(metric.rmse for metric in metrics),
        mae=fmean(metric.mae for metric in metrics),
        slope_mae=fmean(metric.slope_mae for metric in metrics),
        gradient_correlation=fmean(value for value in correlations if value is not None)
        if missing_correlation is None
        else None,
        gradient_correlation_reason=missing_correlation,
        edge_rmse=fmean(metric.edge_rmse for metric in metrics),
        low_resolution_rmse=fmean(metric.low_resolution_rmse for metric in metrics),
        area_consistency_max=max(metric.area_consistency_max for metric in metrics),
    )


def _gradient(values: FloatArray, pixel_size: float) -> tuple[FloatArray, FloatArray]:
    return tuple(np.gradient(values, pixel_size, edge_order=1))  # type: ignore[return-value]


def _laplacian(values: FloatArray, pixel_size: float) -> FloatArray:
    gradient_x, gradient_y = _gradient(values, pixel_size)
    second_x = np.gradient(gradient_x, pixel_size, axis=0, edge_order=1)
    second_y = np.gradient(gradient_y, pixel_size, axis=1, edge_order=1)
    return second_x + second_y


def _correlation(left: FloatArray, right: FloatArray) -> tuple[float | None, str | None]:
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return None, "zero_gradient_variance"
    return float(np.dot(left_centered, right_centered) / denominator), None
