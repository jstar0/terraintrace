"""Typed array boundaries and deterministic terrain resampling primitives."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

import numpy as np

FloatArray: TypeAlias = np.ndarray


def _readonly(values: FloatArray | list[list[float]]) -> FloatArray:
    array = np.array(values, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _validate_grid(values: FloatArray, *, name: str) -> None:
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError(f"{name} must be a two-dimensional grid with at least 2x2 cells")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def validate_scale(scale: int) -> None:
    if not isinstance(scale, int) or isinstance(scale, bool) or not 2 <= scale <= 16:
        raise ValueError("scale must be an integer between 2 and 16")


@dataclass(frozen=True, slots=True, eq=False)
class TerrainCase:
    """One synthetic high-resolution terrain surface and its split group."""

    id: str
    group: str
    high_res: FloatArray
    scale: int = 4
    pixel_size: float = 1.0

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("case id must be identifier-like")
        if not self.group or not self.group.replace("-", "").replace("_", "").isalnum():
            raise ValueError("case group must be identifier-like")
        validate_scale(self.scale)
        if not isfinite(self.pixel_size) or self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive and finite")
        array = np.asarray(self.high_res, dtype=np.float64)
        _validate_grid(array, name="high_res")
        if array.shape[0] % self.scale or array.shape[1] % self.scale:
            raise ValueError("high_res dimensions must be divisible by scale")
        object.__setattr__(self, "high_res", _readonly(array))

    @property
    def low_res_shape(self) -> tuple[int, int]:
        return (self.high_res.shape[0] // self.scale, self.high_res.shape[1] // self.scale)

    @property
    def low_res(self) -> FloatArray:
        return area_downsample(self.high_res, self.scale)


@dataclass(frozen=True, slots=True, eq=False)
class InferenceCase:
    """A low-resolution-only request that intentionally contains no target surface."""

    id: str
    low_res: FloatArray
    scale: int
    pixel_size: float
    elevation_unit: str

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("inference id must be identifier-like")
        if self.scale not in {2, 4, 8}:
            raise ValueError("inference scale must be one of 2, 4, or 8")
        if not isfinite(self.pixel_size) or self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive and finite")
        if not self.elevation_unit.strip():
            raise ValueError("elevation_unit is required")
        array = np.asarray(self.low_res, dtype=np.float64)
        _validate_grid(array, name="low_res")
        output_shape = (array.shape[0] * self.scale, array.shape[1] * self.scale)
        if min(output_shape) < 16 or max(output_shape) > 512:
            raise ValueError("inference output dimensions must be within 16..512")
        object.__setattr__(self, "low_res", _readonly(array))

    @property
    def high_res_shape(self) -> tuple[int, int]:
        return (self.low_res.shape[0] * self.scale, self.low_res.shape[1] * self.scale)


@dataclass(frozen=True, slots=True, eq=False)
class Reconstruction:
    """A method output tied to one case and its scale."""

    method: str
    case_id: str
    values: FloatArray

    def __post_init__(self) -> None:
        if not self.method or not self.case_id:
            raise ValueError("method and case_id are required")
        array = np.asarray(self.values, dtype=np.float64)
        _validate_grid(array, name="reconstruction")
        if not np.isfinite(array).all():
            raise ValueError("reconstruction must contain only finite values")
        object.__setattr__(self, "values", _readonly(array))


def area_downsample(high_res: FloatArray, scale: int) -> FloatArray:
    """Average non-overlapping ``scale × scale`` cells."""

    validate_scale(scale)
    array = np.asarray(high_res, dtype=np.float64)
    _validate_grid(array, name="high_res")
    if array.shape[0] % scale or array.shape[1] % scale:
        raise ValueError("grid dimensions must be divisible by scale")
    height, width = array.shape
    return array.reshape(height // scale, scale, width // scale, scale).mean(axis=(1, 3))


def nearest_upsample(low_res: FloatArray, scale: int) -> FloatArray:
    """Expand each low-resolution cell into a constant block."""

    validate_scale(scale)
    array = np.asarray(low_res, dtype=np.float64)
    _validate_grid(array, name="low_res")
    return np.repeat(np.repeat(array, scale, axis=0), scale, axis=1)


def bilinear_upsample(low_res: FloatArray, scale: int) -> FloatArray:
    """Perform deterministic center-to-center bilinear interpolation."""

    validate_scale(scale)
    array = np.asarray(low_res, dtype=np.float64)
    _validate_grid(array, name="low_res")
    height, width = array.shape
    target_height, target_width = height * scale, width * scale
    source_x = np.arange(height, dtype=np.float64)
    source_y = np.arange(width, dtype=np.float64)
    target_x = np.linspace(0.0, height - 1.0, target_height)
    target_y = np.linspace(0.0, width - 1.0, target_width)
    horizontal = np.vstack([np.interp(target_y, source_y, row) for row in array])
    vertical = np.vstack([np.interp(target_x, source_x, horizontal[:, column]) for column in range(target_width)]).T
    return vertical


def project_area_mean(reconstruction: FloatArray, low_res: FloatArray, scale: int) -> FloatArray:
    """Project a reconstruction back onto the observed area means."""

    validate_scale(scale)
    prediction = np.asarray(reconstruction, dtype=np.float64)
    observed = np.asarray(low_res, dtype=np.float64)
    _validate_grid(prediction, name="reconstruction")
    _validate_grid(observed, name="low_res")
    expected_shape = (observed.shape[0] * scale, observed.shape[1] * scale)
    if prediction.shape != expected_shape:
        raise ValueError("reconstruction shape does not match low_res and scale")
    residual = observed - area_downsample(prediction, scale)
    return prediction + np.repeat(np.repeat(residual, scale, axis=0), scale, axis=1)


def low_resolution_rmse(reconstruction: FloatArray, low_res: FloatArray, scale: int) -> float:
    """Return RMSE after reducing a reconstruction to the observed grid."""

    reduced = area_downsample(reconstruction, scale)
    observed = np.asarray(low_res, dtype=np.float64)
    if reduced.shape != observed.shape:
        raise ValueError("reconstruction and low_res shapes do not match")
    return float(np.sqrt(np.mean((reduced - observed) ** 2)))
