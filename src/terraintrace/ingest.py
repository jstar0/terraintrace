"""Strict local NPZ/metadata intake for TerrainTrace v1."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal

import numpy as np

from .model import InferenceCase, TerrainCase

SplitName = Literal["train", "validation", "test"]
_MAX_NPZ_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class BenchmarkImport:
    case: TerrainCase
    split: SplitName
    elevation_unit: str
    array_sha256: str


def load_benchmark_npz(path: str | Path, metadata: dict[str, object]) -> BenchmarkImport:
    """Load one HR benchmark case after validating the NPZ archive and metadata contract."""
    values = _load_array(path, key="high_res")
    split = _metadata_split(metadata)
    scale, pixel_size, unit = _common_metadata(metadata)
    _validate_hr_shape(values, scale)
    case = TerrainCase(
        _metadata_string(metadata, "id"),
        _metadata_string(metadata, "group"),
        values,
        scale=scale,
        pixel_size=pixel_size,
    )
    return BenchmarkImport(case, split, unit, _array_sha256(values))


def load_inference_npz(path: str | Path, metadata: dict[str, object]) -> InferenceCase:
    """Load an LR-only inference request; metadata must not smuggle in a split or HR target."""
    if "split" in metadata or "group" in metadata:
        raise ValueError("inference metadata must not include benchmark split or group")
    values = _load_array(path, key="low_res")
    scale, pixel_size, unit = _common_metadata(metadata)
    return InferenceCase(_metadata_string(metadata, "id"), values, scale, pixel_size, unit)


def validate_benchmark_collection(entries: tuple[BenchmarkImport, ...]) -> None:
    if not entries or len(entries) > 60:
        raise ValueError("benchmark dataset must contain 1..60 cases")
    groups: dict[str, SplitName] = {}
    arrays: dict[str, SplitName] = {}
    ids: set[str] = set()
    for entry in entries:
        if entry.case.id in ids:
            raise ValueError(f"duplicate case id {entry.case.id}")
        ids.add(entry.case.id)
        previous_group = groups.setdefault(entry.case.group, entry.split)
        if previous_group != entry.split:
            raise ValueError(f"group {entry.case.group} crosses splits")
        previous_array = arrays.setdefault(entry.array_sha256, entry.split)
        if previous_array != entry.split:
            raise ValueError("exact duplicate array crosses splits")


def parse_metadata_json(text: str) -> dict[str, object]:
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("metadata must be valid JSON") from error
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    return metadata


def _load_array(path: str | Path, *, key: str) -> np.ndarray:
    archive_path = Path(path)
    if archive_path.suffix.lower() != ".npz" or not archive_path.is_file():
        raise ValueError("input must be an existing .npz file")
    _validate_zip(archive_path)
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            if set(archive.files) != {key}:
                raise ValueError(f"NPZ must contain exactly one {key!r} array")
            values = np.asarray(archive[key], dtype=np.float64)
    except (OSError, ValueError) as error:
        raise ValueError(f"invalid NPZ: {error}") from error
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("array must be finite, two-dimensional, and contain no NoData values")
    return values


def _validate_zip(path: Path) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            total = sum(entry.file_size for entry in archive.infolist())
    except zipfile.BadZipFile as error:
        raise ValueError("input is not a valid NPZ ZIP archive") from error
    if total > _MAX_NPZ_BYTES:
        raise ValueError("NPZ uncompressed content exceeds 32 MiB")


def _common_metadata(metadata: dict[str, object]) -> tuple[int, float, str]:
    scale = metadata.get("scale")
    if not isinstance(scale, int) or isinstance(scale, bool) or scale not in {2, 4, 8}:
        raise ValueError("metadata.scale must be 2, 4, or 8")
    pixel_size = metadata.get("pixel_size")
    if (
        not isinstance(pixel_size, (int, float))
        or isinstance(pixel_size, bool)
        or not isfinite(pixel_size)
        or pixel_size <= 0
    ):
        raise ValueError("metadata.pixel_size must be positive and finite")
    return scale, float(pixel_size), _metadata_string(metadata, "elevation_unit")


def _metadata_split(metadata: dict[str, object]) -> SplitName:
    split = metadata.get("split")
    if split not in {"train", "validation", "test"}:
        raise ValueError("metadata.split must be train, validation, or test")
    return split


def _metadata_string(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"metadata.{key} is required")
    return value.strip()


def _validate_hr_shape(values: np.ndarray, scale: int) -> None:
    if min(values.shape) < 16 or max(values.shape) > 512:
        raise ValueError("high_res dimensions must be within 16..512")
    if values.shape[0] % scale or values.shape[1] % scale:
        raise ValueError("high_res dimensions must be divisible by metadata.scale")


def _array_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.float64)
    return hashlib.sha256(canonical.tobytes()).hexdigest()
