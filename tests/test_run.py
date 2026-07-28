import numpy as np

from terraintrace.fixture import create_dataset
from terraintrace.methods import fit_spectral_prior
from terraintrace.model import InferenceCase
from terraintrace.run import profile, roi_metrics, run_benchmark_case, run_inference_case


def test_real_array_outputs_keep_lr_consistency_and_inference_omits_metrics() -> None:
    dataset = create_dataset(17)
    benchmark = run_benchmark_case(dataset.test[0], fit_spectral_prior(dataset.train))
    assert set(benchmark.arrays) == {"nearest", "bilinear", "spectral_prior"}
    assert all(row.low_resolution_rmse < 1e-8 for row in benchmark.arrays.values())
    assert all(metric is not None for metric in benchmark.metrics.values())
    inference = run_inference_case(InferenceCase("lr", np.ones((4, 4)), 4, 30, "m"))
    assert all(metric is None for metric in inference.metrics.values())


def test_profile_uses_one_shared_path_for_every_method() -> None:
    dataset = create_dataset(17)
    result = run_benchmark_case(dataset.test[0], fit_spectral_prior(dataset.train))
    sampled = profile(result, ((0, 0), (47, 47)), samples=8, pixel_size=30)
    assert len(sampled["distance"]) == 8
    assert all(len(series) == 8 for series in sampled["series"].values())
    roi = roi_metrics(result, (2, 3, 12, 14))
    assert roi["exploratory"] is True
