"""Small deterministic reconstruction methods and a train-only spectral prior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import (
    FloatArray,
    InferenceCase,
    TerrainCase,
    area_downsample,
    bilinear_upsample,
    nearest_upsample,
    project_area_mean,
    validate_scale,
)


@dataclass(frozen=True, slots=True, eq=False)
class SpectralPrior:
    """Radial high-frequency gain learned only from training cases."""

    radius_centers: FloatArray
    gains: FloatArray
    scale: int

    def __post_init__(self) -> None:
        validate_scale(self.scale)
        radii = np.asarray(self.radius_centers, dtype=np.float64)
        values = np.asarray(self.gains, dtype=np.float64)
        if radii.ndim != 1 or values.ndim != 1 or len(radii) != len(values) or len(radii) < 2:
            raise ValueError("spectral prior needs matching one-dimensional radius/gain arrays")
        if not np.isfinite(radii).all() or not np.isfinite(values).all():
            raise ValueError("spectral prior values must be finite")
        if np.any(np.diff(radii) <= 0) or np.any(radii < 0):
            raise ValueError("spectral prior radii must be strictly increasing and non-negative")
        if np.any(values <= 0):
            raise ValueError("spectral prior gains must be positive")
        object.__setattr__(self, "radius_centers", _readonly(radii))
        object.__setattr__(self, "gains", _readonly(values))


def reconstruct_nearest(case: TerrainCase) -> FloatArray:
    """Nearest-cell baseline, which exactly preserves each area mean."""

    return nearest_upsample(case.low_res, case.scale)


def reconstruct_bilinear(case: TerrainCase) -> FloatArray:
    """Bilinear baseline projected back to the observed area means."""

    interpolated = bilinear_upsample(case.low_res, case.scale)
    return project_area_mean(interpolated, case.low_res, case.scale)


def fit_spectral_prior(cases: tuple[TerrainCase, ...], *, bins: int = 12) -> SpectralPrior:
    """Estimate a bounded radial detail gain from paired training surfaces.

    The prior sees high-resolution targets only in this explicit fitting call;
    validation and test cases never influence it.  A median ratio is used so
    one unusually sharp synthetic surface cannot dominate the gain curve.
    """

    if not cases:
        raise ValueError("at least one training case is required")
    if bins < 2 or bins > 64:
        raise ValueError("bins must be between 2 and 64")
    first = cases[0]
    if any(case.scale != first.scale or case.high_res.shape != first.high_res.shape for case in cases):
        raise ValueError("all training cases must share shape and scale")
    height, width = first.high_res.shape
    frequency = np.sqrt(np.fft.fftfreq(height)[:, None] ** 2 + np.fft.fftfreq(width)[None, :] ** 2)
    edges = np.linspace(0.0, float(frequency.max()) + 1e-12, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    ratios: list[np.ndarray] = []
    for case in cases:
        base = reconstruct_bilinear(case)
        high_power = np.abs(np.fft.fft2(case.high_res))
        base_power = np.abs(np.fft.fft2(base))
        ratio = np.divide(high_power, base_power + 1e-8)
        case_gains = np.ones(bins, dtype=np.float64)
        for index in range(bins):
            mask = (frequency >= edges[index]) & (frequency < edges[index + 1])
            if np.any(mask):
                case_gains[index] = float(np.median(ratio[mask]))
        ratios.append(case_gains)
    gains = np.median(np.stack(ratios, axis=0), axis=0)
    # Keep the observation-compatible low frequencies untouched and bound the
    # learned detail multiplier to avoid synthetic ringing explosions.
    gains = np.clip(gains, 0.65, 1.55)
    gains[centers < 0.08] = 1.0
    return SpectralPrior(centers, gains, first.scale)


def reconstruct_spectral(case: TerrainCase, prior: SpectralPrior) -> FloatArray:
    """Apply a train-only radial detail prior and restore area consistency."""

    if case.scale != prior.scale:
        raise ValueError("case and spectral prior scales do not match")
    base = reconstruct_bilinear(case)
    height, width = base.shape
    frequency = np.sqrt(np.fft.fftfreq(height)[:, None] ** 2 + np.fft.fftfreq(width)[None, :] ** 2)
    gain_grid = np.interp(
        frequency.ravel(), prior.radius_centers, prior.gains, left=float(prior.gains[0]), right=float(prior.gains[-1])
    ).reshape(frequency.shape)
    enhanced = np.fft.ifft2(np.fft.fft2(base) * gain_grid).real
    return project_area_mean(enhanced, case.low_res, case.scale)


def reconstruct_all(case: TerrainCase, prior: SpectralPrior) -> dict[str, FloatArray]:
    """Return the benchmark methods in a stable order."""

    return {
        "nearest": reconstruct_nearest(case),
        "bilinear": reconstruct_bilinear(case),
        "spectral_prior": reconstruct_spectral(case, prior),
    }


def reconstruct_inference(case: InferenceCase, methods: tuple[str, ...]) -> dict[str, FloatArray]:
    """Reconstruct an LR-only request without inventing a hidden HR evaluation target."""
    supported = {"nearest", "bilinear"}
    requested = set(methods)
    if not requested or not requested <= supported:
        raise ValueError("LR-only inference supports nearest and bilinear only")
    outputs: dict[str, FloatArray] = {}
    if "nearest" in requested:
        outputs["nearest"] = nearest_upsample(case.low_res, case.scale)
    if "bilinear" in requested:
        outputs["bilinear"] = project_area_mean(bilinear_upsample(case.low_res, case.scale), case.low_res, case.scale)
    return outputs


def check_area_consistency(values: FloatArray, case: TerrainCase, tolerance: float = 1e-9) -> bool:
    """Return whether a prediction preserves the observed low-resolution means."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    reduced = area_downsample(values, case.scale)
    return bool(np.max(np.abs(reduced - case.low_res)) <= tolerance)


def _readonly(values: FloatArray) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array
