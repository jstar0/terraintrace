import numpy as np
import pytest

from terraintrace.fixture import create_dataset
from terraintrace.model import (
    InferenceCase,
    TerrainCase,
    area_downsample,
    bilinear_upsample,
    low_resolution_rmse,
    nearest_upsample,
    project_area_mean,
)


def test_grouped_fixture_is_deterministic_and_leakage_free() -> None:
    first = create_dataset(17)
    second = create_dataset(17)
    assert first.scale == 4
    assert {case.group for case in first.train}.isdisjoint(case.group for case in first.validation)
    assert {case.group for case in first.validation}.isdisjoint(case.group for case in first.test)
    assert [case.id for case in first.all_cases()] == [case.id for case in second.all_cases()]
    assert all(
        np.array_equal(left.high_res, right.high_res)
        for left, right in zip(first.all_cases(), second.all_cases(), strict=True)
    )


def test_area_downsample_uses_non_overlapping_cell_means() -> None:
    high_res = np.arange(16, dtype=np.float64).reshape(4, 4)
    expected = np.array([[2.5, 4.5], [10.5, 12.5]])
    assert np.array_equal(area_downsample(high_res, 2), expected)


def test_resampling_shapes_and_area_projection() -> None:
    case = create_dataset(17).test[0]
    nearest = nearest_upsample(case.low_res, case.scale)
    bilinear = bilinear_upsample(case.low_res, case.scale)
    projected = project_area_mean(bilinear, case.low_res, case.scale)
    assert nearest.shape == case.high_res.shape
    assert bilinear.shape == case.high_res.shape
    assert low_resolution_rmse(projected, case.low_res, case.scale) < 1e-10


def test_terrain_case_copies_and_freezes_input() -> None:
    values = np.ones((8, 8), dtype=np.float64)
    case = TerrainCase("demo", "group", values, scale=4)
    values[0, 0] = 99
    assert case.high_res[0, 0] == 1
    with pytest.raises(ValueError):
        case.high_res[0, 0] = 2


def test_invalid_case_and_grid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TerrainCase("demo", "group", np.ones((7, 8)), scale=4)
    with pytest.raises(ValueError, match="finite"):
        area_downsample(np.array([[np.nan, 1], [1, 1]]), 2)
    with pytest.raises(ValueError, match="scale"):
        nearest_upsample(np.ones((2, 2)), 1)


def test_projection_rejects_shape_mismatch() -> None:
    case = create_dataset(17).test[0]
    with pytest.raises(ValueError, match="shape"):
        project_area_mean(np.ones((3, 3)), case.low_res, case.scale)


def test_lr_only_inference_case_has_no_target_and_validates_output_contract() -> None:
    case = InferenceCase("lr-demo", np.ones((4, 4)), scale=4, pixel_size=30, elevation_unit="m")
    assert case.high_res_shape == (16, 16)
    assert not hasattr(case, "high_res")
    with pytest.raises(ValueError, match="elevation_unit"):
        InferenceCase("bad", np.ones((4, 4)), scale=4, pixel_size=30, elevation_unit=" ")
