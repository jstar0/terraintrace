import io
import json
import zipfile

import numpy as np

from terraintrace.export import verify_case_archive, write_case_archive
from terraintrace.fixture import create_dataset
from terraintrace.methods import fit_spectral_prior
from terraintrace.run import run_benchmark_case


def test_case_archive_reopens_arrays_profile_and_manifest(tmp_path) -> None:
    dataset = create_dataset(17)
    case = dataset.test[0]
    path = tmp_path / "case.zip"
    proof = write_case_archive(path, case, run_benchmark_case(case, fit_spectral_prior(dataset.train)))
    assert proof == verify_case_archive(path)
    with zipfile.ZipFile(path) as archive:
        assert {"arrays.npz", "metadata.json", "profile.csv", "report.html", "manifest.json"} == set(archive.namelist())
        with np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            assert {"low_res", "high_res", "nearest", "bilinear", "spectral_prior"} == set(arrays.files)
            np.testing.assert_array_equal(arrays["high_res"], case.high_res)
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["metrics"]["bilinear"]["rmse"] is not None
