"""Reproducible TerrainTrace case archives with post-write integrity verification."""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .model import InferenceCase, TerrainCase
from .run import CaseRun, profile


def write_case_archive(path: Path, source: TerrainCase | InferenceCase, result: CaseRun) -> dict[str, object]:
    if source.id != result.case_id:
        raise ValueError("source and run case ids must match")
    arrays = io.BytesIO()
    archive_arrays = {"low_res": source.low_res, **{name: row.values for name, row in result.arrays.items()}}
    if isinstance(source, TerrainCase):
        archive_arrays["high_res"] = source.high_res
    np.savez(arrays, **archive_arrays)
    metadata = {
        "case_id": source.id,
        "mode": result.mode,
        "scale": source.scale,
        "pixel_size": source.pixel_size,
        "elevation_unit": getattr(source, "elevation_unit", "synthetic elevation units"),
        "arrays": {name: row.metadata() for name, row in result.arrays.items()},
        "metrics": {name: metric.as_dict() if metric else None for name, metric in result.metrics.items()},
    }
    shape = (source.low_res.shape[0] * source.scale, source.low_res.shape[1] * source.scale)
    sampled = profile(
        result,
        ((0, 0), (shape[0] - 1, shape[1] - 1)),
        pixel_size=source.pixel_size,
        reference=source.high_res if isinstance(source, TerrainCase) else None,
    )
    members = {
        "arrays.npz": arrays.getvalue(),
        "metadata.json": _json(metadata),
        "profile.csv": _profile_csv(sampled),
        "report.html": _html(metadata),
    }
    manifest = {
        "schema_version": 1,
        "files": {name: hashlib.sha256(value).hexdigest() for name, value in members.items()},
    }
    members["manifest.json"] = _json(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)
    return verify_case_archive(path)


def verify_case_archive(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for name, expected in manifest["files"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"artifact digest mismatch for {name}")
        with np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            if "low_res" not in arrays.files or len(arrays.files) < 2:
                raise ValueError("artifact arrays are incomplete")
        return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _profile_csv(sampled: dict[str, object]) -> bytes:
    handle = io.StringIO()
    names = sorted(sampled["series"])
    writer = csv.DictWriter(handle, fieldnames=("sample", "distance", *names), lineterminator="\n")
    writer.writeheader()
    for index, distance in enumerate(sampled["distance"]):
        writer.writerow(
            {"sample": index, "distance": distance, **{name: sampled["series"][name][index] for name in names}}
        )
    return handle.getvalue().encode()


def _html(metadata: dict[str, Any]) -> bytes:
    title = f"TerrainTrace {metadata['case_id']}"
    return (
        f"<!doctype html><meta charset=utf-8><title>{html.escape(title)}</title>"
        f"<h1>{html.escape(title)}</h1><pre>{html.escape(json.dumps(metadata, indent=2))}</pre>"
    ).encode()


def _json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
