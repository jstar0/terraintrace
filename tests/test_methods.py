from dataclasses import replace

import numpy as np
import pytest

from terraintrace.fixture import create_dataset
from terraintrace.methods import (
    SpectralPrior,
    check_area_consistency,
    fit_spectral_prior,
    reconstruct_all,
    reconstruct_inference,
    reconstruct_spectral,
)
from terraintrace.model import InferenceCase, low_resolution_rmse


def test_train_only_spectral_prior_is_stable_and_bounded() -> None:
    dataset = create_dataset(17)
    first = fit_spectral_prior(dataset.train)
    second = fit_spectral_prior(dataset.train)
    assert np.array_equal(first.radius_centers, second.radius_centers)
    assert np.array_equal(first.gains, second.gains)
    assert np.all((first.gains >= 0.65) & (first.gains <= 1.55))
    assert first.gains[0] == 1.0


def test_all_methods_are_finite_and_observation_consistent() -> None:
    dataset = create_dataset(17)
    prior = fit_spectral_prior(dataset.train)
    for case in dataset.test:
        outputs = reconstruct_all(case, prior)
        assert set(outputs) == {"nearest", "bilinear", "spectral_prior"}
        for values in outputs.values():
            assert values.shape == case.high_res.shape
            assert np.isfinite(values).all()
            assert check_area_consistency(values, case)


def test_prior_rejects_empty_or_incompatible_training_cases() -> None:
    dataset = create_dataset(17)
    with pytest.raises(ValueError, match="training"):
        fit_spectral_prior(())
    with pytest.raises(ValueError, match="shape and scale"):
        fit_spectral_prior((dataset.train[0], replace(dataset.train[0], scale=2)))


def test_spectral_prior_scale_mismatch_is_explicit() -> None:
    dataset = create_dataset(17)
    prior = fit_spectral_prior(dataset.train)
    other = type(dataset.test[0])(
        dataset.test[0].id,
        dataset.test[0].group,
        dataset.test[0].high_res,
        scale=2,
    )
    with pytest.raises(ValueError, match="scales"):
        reconstruct_spectral(other, prior)


def test_prior_constructor_rejects_non_monotonic_or_non_positive_gains() -> None:
    with pytest.raises(ValueError, match="increasing"):
        SpectralPrior(np.array([0.2, 0.1]), np.array([1.0, 1.0]), 4)
    with pytest.raises(ValueError, match="positive"):
        SpectralPrior(np.array([0.1, 0.2]), np.array([1.0, 0.0]), 4)


def test_lr_only_inference_reconstructs_without_hr_metrics_or_spectral_prior() -> None:
    case = InferenceCase("lr", np.arange(16, dtype=np.float64).reshape(4, 4), 4, 10, "m")
    outputs = reconstruct_inference(case, ("nearest", "bilinear"))
    assert set(outputs) == {"nearest", "bilinear"}
    assert all(values.shape == case.high_res_shape for values in outputs.values())
    assert all(low_resolution_rmse(values, case.low_res, case.scale) < 1e-10 for values in outputs.values())
    with pytest.raises(ValueError, match="LR-only"):
        reconstruct_inference(case, ("spectral_prior",))
