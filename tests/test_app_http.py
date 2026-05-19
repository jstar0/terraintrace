from __future__ import annotations

import hashlib
import io
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from http.client import HTTPMessage
from pathlib import Path

import numpy as np
import pytest
import uvicorn

from terraintrace.app.main import create_app
from terraintrace.export import verify_case_archive
from terraintrace.fixture import create_dataset
from terraintrace.model import InferenceCase, area_downsample, low_resolution_rmse


@pytest.fixture
def local_api(tmp_path: Path) -> Iterator[str]:
    listening_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listening_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listening_socket.bind(("127.0.0.1", 0))
    listening_socket.listen()
    host, port = listening_socket.getsockname()
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path), log_level="critical"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listening_socket]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert server.started, "local TerrainTrace API did not start"
    try:
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        listening_socket.close()
        assert not thread.is_alive(), "local TerrainTrace API did not stop"


def test_http_benchmark_and_inference_run_lifecycle(local_api: str, tmp_path: Path) -> None:
    dataset = create_dataset(17)
    benchmark_rows = [
        ("train", dataset.train[0]),
        ("validation", dataset.validation[0]),
        ("test", dataset.test[0]),
    ]
    benchmark_metadata = [
        {
            "id": case.id,
            "group": case.group,
            "split": split,
            "scale": case.scale,
            "pixel_size": case.pixel_size,
            "elevation_unit": "synthetic elevation units",
        }
        for split, case in benchmark_rows
    ]
    fields = [
        ("mode", "benchmark"),
        ("metadata", json.dumps(benchmark_metadata)),
        ("dataset_id", "http_benchmark"),
    ]
    files = [("files", f"{case.id}.npz", _npz_bytes(high_res=case.high_res)) for _, case in benchmark_rows]
    status, _, body = _request(local_api, "POST", "/api/v1/terrain-datasets", *_multipart(fields, files))
    assert status == 200, body.decode()
    created = json.loads(body)
    assert (created["dataset_id"], created["revision"], len(created["cases"])) == ("http_benchmark", 1, 3)
    status, _, body = _request(local_api, "GET", "/api/v1/terrain-datasets/http_benchmark/1")
    assert status == 200
    assert json.loads(body)["manifest"]["mode"] == "benchmark"

    status, _, body = _json_request(
        local_api,
        "POST",
        "/api/v1/reconstruction-runs",
        {
            "dataset_id": "http_benchmark",
            "dataset_revision": 1,
            "mode": "benchmark",
            "methods": ["nearest", "bilinear", "spectral_prior"],
            "metrics_protocol": "terrain_v2",
        },
    )
    assert status == 202
    queued = json.loads(body)
    benchmark_run = _wait_for_job(local_api, queued["job_id"])
    assert benchmark_run["status"] == "succeeded", benchmark_run

    status, _, body = _request(local_api, "GET", f"/api/v1/runs/{queued['run_id']}/cases")
    assert status == 200
    run_cases = json.loads(body)
    assert run_cases["status"] == "succeeded"
    assert {row["split"] for row in run_cases["cases"]} == {"train", "validation", "test"}
    test_case = next(case for case in dataset.test if case.id == run_cases["cases"][-1]["case_id"])

    status, _, body = _request(local_api, "GET", f"/api/v1/runs/{queued['run_id']}/cases/{test_case.id}/metrics")
    metrics = json.loads(body)
    assert status == 200
    assert metrics["protocol"] == "terrain_v2"
    assert metrics["metrics"]["bilinear"]["rmse"] >= 0

    status, headers, body = _request(
        local_api,
        "GET",
        f"/api/v1/runs/{queued['run_id']}/cases/{test_case.id}/arrays?method=bilinear",
    )
    assert status == 200
    reconstructed = np.load(io.BytesIO(body), allow_pickle=False)
    assert reconstructed.shape == test_case.high_res.shape
    assert low_resolution_rmse(reconstructed, test_case.low_res, test_case.scale) < 1e-8
    assert (
        headers["X-Terrain-Array-SHA256"] == hashlib.sha256(np.ascontiguousarray(reconstructed).tobytes()).hexdigest()
    )
    status, _, body = _request(
        local_api,
        "GET",
        f"/api/v1/runs/{queued['run_id']}/cases/{test_case.id}/arrays?method=bilinear&kind=difference",
    )
    assert status == 200
    np.testing.assert_allclose(np.load(io.BytesIO(body), allow_pickle=False), reconstructed - test_case.high_res)

    status, _, body = _json_request(
        local_api,
        "POST",
        f"/api/v1/runs/{queued['run_id']}/profiles",
        {"case_id": test_case.id, "points": [[0, 0], [47, 47]], "methods": ["nearest", "bilinear"], "samples": 8},
    )
    sampled = json.loads(body)
    assert status == 200
    assert len(sampled["distance"]) == 8
    assert set(sampled["series"]) == {"nearest", "bilinear", "high_res"}
    assert all(len(series) == 8 for series in sampled["series"].values())

    status, _, body = _json_request(
        local_api,
        "POST",
        f"/api/v1/runs/{queued['run_id']}/roi",
        {"case_id": test_case.id, "bounds": [2, 3, 12, 14], "methods": ["bilinear"]},
    )
    roi = json.loads(body)
    assert status == 200
    assert roi["exploratory"] is True
    assert "bilinear" in roi["methods"]

    status, _, body = _request(
        local_api,
        "POST",
        f"/api/v1/runs/{queued['run_id']}/exports?{urllib.parse.urlencode({'case_id': test_case.id})}",
    )
    exported = json.loads(body)
    assert status == 200
    status, _, body = _request(local_api, "GET", f"/api/v1/artifacts/{exported['artifact_id']}/download")
    assert status == 200
    archive_path = tmp_path / "benchmark-export.zip"
    archive_path.write_bytes(body)
    assert verify_case_archive(archive_path)["sha256"] == exported["sha256"]
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        with np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            assert {"low_res", "high_res", "nearest", "bilinear", "spectral_prior"} == set(arrays.files)
            np.testing.assert_allclose(arrays["high_res"], test_case.high_res)
        metadata = json.loads(archive.read("metadata.json"))
        assert metadata["mode"] == "benchmark"
        assert metadata["metrics"]["bilinear"]["rmse"] == metrics["metrics"]["bilinear"]["rmse"]

    inference_source = dataset.test[0]
    inference = InferenceCase(
        id="lr-only",
        low_res=area_downsample(inference_source.high_res, inference_source.scale),
        scale=inference_source.scale,
        pixel_size=inference_source.pixel_size,
        elevation_unit="synthetic elevation units",
    )
    inference_metadata = {
        "id": inference.id,
        "scale": inference.scale,
        "pixel_size": inference.pixel_size,
        "elevation_unit": inference.elevation_unit,
    }
    fields = [("mode", "inference"), ("metadata", json.dumps(inference_metadata)), ("dataset_id", "http_inference")]
    files = [("files", "observation.npz", _npz_bytes(low_res=inference.low_res))]
    status, _, body = _request(local_api, "POST", "/api/v1/terrain-datasets", *_multipart(fields, files))
    assert status == 200, body.decode()
    inference_dataset = json.loads(body)
    assert inference_dataset["revision"] == 1
    status, _, body = _json_request(
        local_api,
        "POST",
        "/api/v1/reconstruction-runs",
        {
            "dataset_id": "http_inference",
            "dataset_revision": 1,
            "mode": "inference",
            "methods": ["nearest", "bilinear"],
        },
    )
    assert status == 202
    inference_queued = json.loads(body)
    assert _wait_for_job(local_api, inference_queued["job_id"])["status"] == "succeeded"
    status, _, body = _request(
        local_api,
        "GET",
        f"/api/v1/runs/{inference_queued['run_id']}/cases/{inference.id}/metrics",
    )
    assert status == 200
    inference_metrics = json.loads(body)["metrics"]
    assert inference_metrics == {"nearest": None, "bilinear": None}
    status, _, body = _request(
        local_api,
        "GET",
        f"/api/v1/runs/{inference_queued['run_id']}/cases/{inference.id}/arrays?kind=high_res",
    )
    assert status == 404
    assert json.loads(body)["error"]["code"] == "NOT_FOUND"
    status, _, body = _request(
        local_api,
        "GET",
        f"/api/v1/runs/{inference_queued['run_id']}/cases/{inference.id}/arrays?method=bilinear&kind=difference",
    )
    assert status == 404
    assert json.loads(body)["error"]["code"] == "NOT_FOUND"
    status, _, body = _json_request(
        local_api,
        "POST",
        f"/api/v1/runs/{inference_queued['run_id']}/profiles",
        {"points": [[0, 0], [47, 47]]},
    )
    assert status == 422
    validation_error = json.loads(body)["error"]
    assert validation_error["code"] == "INVALID_REQUEST"
    assert any(item["path"] == "body.case_id" for item in validation_error["field_errors"])

    status, _, body = _request(
        local_api,
        "POST",
        f"/api/v1/runs/{inference_queued['run_id']}/exports?{urllib.parse.urlencode({'case_id': inference.id})}",
    )
    inference_export = json.loads(body)
    assert status == 200
    status, _, body = _request(local_api, "GET", f"/api/v1/artifacts/{inference_export['artifact_id']}/download")
    assert status == 200
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        with np.load(io.BytesIO(archive.read("arrays.npz")), allow_pickle=False) as arrays:
            assert "high_res" not in arrays.files
            assert {"low_res", "nearest", "bilinear"} == set(arrays.files)
        assert json.loads(archive.read("metadata.json"))["metrics"] == {"nearest": None, "bilinear": None}


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _multipart(fields: list[tuple[str, str]], files: list[tuple[str, str, bytes]]) -> tuple[bytes, dict[str, str]]:
    boundary = "terraintrace-test-boundary"
    chunks: list[bytes] = []
    for name, value in fields:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for name, filename, content in files:
        chunks.extend(
            [
                f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: application/octet-stream\r\n\r\n".encode(),
                content,
                b"\r\n",
            ]
        )
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), {"Content-Type": f"multipart/form-data; boundary={boundary}"}


def _request(
    base: str,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, HTTPMessage, bytes]:
    request = urllib.request.Request(base + path, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def _json_request(base: str, method: str, path: str, payload: dict[str, object]) -> tuple[int, HTTPMessage, bytes]:
    return _request(
        base,
        method,
        path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )


def _wait_for_job(base: str, job_id: str) -> dict[str, object]:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        status, _, body = _request(base, "GET", f"/api/v1/jobs/{job_id}")
        assert status == 200, body.decode()
        result = json.loads(body)
        if result["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
            return result
        time.sleep(0.025)
    pytest.fail(f"job {job_id} did not finish within 10 seconds")
