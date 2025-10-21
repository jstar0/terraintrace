"""TerrainTrace: a synthetic, reproducible terrain reconstruction lab."""

from .benchmark import benchmark_payload, write_benchmark
from .fixture import TerrainDataset, create_dataset
from .methods import (
    SpectralPrior,
    fit_spectral_prior,
    reconstruct_bilinear,
    reconstruct_nearest,
    reconstruct_spectral,
)
from .metrics import MetricSet, aggregate_metrics, evaluate_reconstruction
from .model import (
    Reconstruction,
    TerrainCase,
    area_downsample,
    bilinear_upsample,
    low_resolution_rmse,
    nearest_upsample,
    project_area_mean,
)
from .sweep import parse_seed_list, seed_sweep_payload, write_sweep

__all__ = [
    "MetricSet",
    "Reconstruction",
    "SpectralPrior",
    "TerrainCase",
    "TerrainDataset",
    "aggregate_metrics",
    "area_downsample",
    "benchmark_payload",
    "bilinear_upsample",
    "create_dataset",
    "evaluate_reconstruction",
    "fit_spectral_prior",
    "low_resolution_rmse",
    "nearest_upsample",
    "parse_seed_list",
    "project_area_mean",
    "reconstruct_bilinear",
    "reconstruct_nearest",
    "reconstruct_spectral",
    "seed_sweep_payload",
    "write_benchmark",
    "write_sweep",
]
