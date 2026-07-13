import numpy as np
import pytest

from terraintrace.fixture import create_dataset
from terraintrace.metrics import aggregate_metrics, evaluate_reconstruction
from terraintrace.model import TerrainCase


def test_perfect_reconstruction_has_zero_error_and_unit_correlation() -> None:
    case = create_dataset(17).test[0]
    metrics = evaluate_reconstruction(case.high_res, case)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.slope_mae == pytest.approx(0.0)
    assert metrics.edge_rmse == pytest.approx(0.0)
    assert metrics.low_resolution_rmse == pytest.approx(0.0)
    assert metrics.gradient_correlation == pytest.approx(1.0)
    assert metrics.area_consistency_max == pytest.approx(0.0)


def test_aggregate_keeps_worst_area_consistency_bound() -> None:
    case = create_dataset(17).test[0]
    first = evaluate_reconstruction(case.high_res, case)
    second = evaluate_reconstruction(case.high_res + 0.5, case)
    aggregate = aggregate_metrics((first, second))
    assert aggregate.rmse == pytest.approx(0.25)
    assert aggregate.area_consistency_max == pytest.approx(0.5)


def test_metrics_reject_shape_and_nonfinite_predictions() -> None:
    case = create_dataset(17).test[0]
    with pytest.raises(ValueError, match="shape"):
        evaluate_reconstruction(np.ones((3, 3)), case)
    bad = case.high_res.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        evaluate_reconstruction(bad, case)


def test_zero_gradient_has_explicit_unavailable_2d_correlation() -> None:
    case = TerrainCase("flat", "flat-group", np.zeros((16, 16)), scale=4)
    metrics = evaluate_reconstruction(case.high_res, case)
    assert metrics.gradient_correlation is None
    assert metrics.gradient_correlation_reason == "zero_gradient_variance"
    assert metrics.as_dict()["gradientCorrelation"] is None
