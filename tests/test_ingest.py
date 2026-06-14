import json

import numpy as np
import pytest

from terraintrace.ingest import (
    load_benchmark_npz,
    load_inference_npz,
    parse_metadata_json,
    validate_benchmark_collection,
)


def _write_npz(path, key: str, values: np.ndarray) -> None:
    np.savez(path, **{key: values})


def _benchmark_metadata(case_id: str, group: str, split: str) -> dict[str, object]:
    return {
        "id": case_id,
        "group": group,
        "split": split,
        "scale": 4,
        "pixel_size": 30,
        "elevation_unit": "m",
    }


def test_npz_benchmark_and_lr_only_inputs_have_distinct_contracts(tmp_path) -> None:
    hr_path = tmp_path / "hr.npz"
    lr_path = tmp_path / "lr.npz"
    high_res = np.arange(256, dtype=np.float64).reshape(16, 16)
    _write_npz(hr_path, "high_res", high_res)
    _write_npz(lr_path, "low_res", high_res.reshape(4, 4, 4, 4).mean(axis=(1, 3)))
    benchmark = load_benchmark_npz(hr_path, _benchmark_metadata("case-a", "ridge", "train"))
    inference = load_inference_npz(
        lr_path,
        {"id": "observation-a", "scale": 4, "pixel_size": 30, "elevation_unit": "m"},
    )
    assert benchmark.case.high_res.shape == (16, 16)
    assert benchmark.array_sha256
    assert inference.high_res_shape == (16, 16)
    assert not hasattr(inference, "high_res")


def test_collection_rejects_groups_or_exact_arrays_crossing_splits(tmp_path) -> None:
    path = tmp_path / "same.npz"
    _write_npz(path, "high_res", np.arange(256, dtype=np.float64).reshape(16, 16))
    train = load_benchmark_npz(path, _benchmark_metadata("one", "shared", "train"))
    group_crossing = load_benchmark_npz(path, _benchmark_metadata("two", "shared", "test"))
    with pytest.raises(ValueError, match="group"):
        validate_benchmark_collection((train, group_crossing))
    array_crossing = load_benchmark_npz(path, _benchmark_metadata("three", "other", "test"))
    with pytest.raises(ValueError, match="duplicate"):
        validate_benchmark_collection((train, array_crossing))


def test_npz_rejects_nonfinite_or_wrong_schema_and_metadata_is_explicit(tmp_path) -> None:
    path = tmp_path / "bad.npz"
    _write_npz(path, "other", np.ones((16, 16)))
    with pytest.raises(ValueError, match="exactly one"):
        load_benchmark_npz(path, _benchmark_metadata("bad", "group", "train"))
    _write_npz(path, "high_res", np.full((16, 16), np.nan))
    with pytest.raises(ValueError, match="NoData"):
        load_benchmark_npz(path, _benchmark_metadata("bad", "group", "train"))
    assert parse_metadata_json(json.dumps(_benchmark_metadata("ok", "group", "test")))["id"] == "ok"
    with pytest.raises(ValueError, match="JSON object"):
        parse_metadata_json("[]")
