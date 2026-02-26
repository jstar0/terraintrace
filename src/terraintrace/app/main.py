from __future__ import annotations

import hashlib
import io
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Event, Thread
from typing import Annotated, Literal
from uuid import uuid4

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from terraintrace.app.store import LocalStore, QueueFullError
from terraintrace.export import write_case_archive
from terraintrace.fixture import TerrainDataset, create_dataset
from terraintrace.ingest import (
    BenchmarkImport,
    load_benchmark_npz,
    load_inference_npz,
    validate_benchmark_collection,
)
from terraintrace.methods import SpectralPrior, fit_spectral_prior
from terraintrace.metrics import MetricSet
from terraintrace.model import InferenceCase
from terraintrace.run import ArrayArtifact, CaseRun, profile, roi_metrics, run_benchmark_case, run_inference_case

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
METHODS = {"nearest", "bilinear", "spectral_prior"}


class ProfileInput(BaseModel):
    case_id: str
    points: tuple[tuple[float, float], tuple[float, float]]
    methods: list[str] | None = None
    samples: int = Field(default=128, ge=2, le=2048)


class RoiInput(BaseModel):
    case_id: str
    bounds: tuple[int, int, int, int]
    methods: list[str] | None = None


class RunInput(BaseModel):
    dataset_id: str
    dataset_revision: int = Field(ge=1)
    mode: Literal["benchmark", "inference"]
    methods: list[str] = Field(min_length=1)
    metrics_protocol: Literal["terrain_v2"] = "terrain_v2"


class RunWorker:
    """Single local worker; case boundaries are durable progress/cancellation points."""

    def __init__(self, store: LocalStore, runtime_dir: Path) -> None:
        self.store = store
        self.runtime_dir = runtime_dir
        self.wake = Event()
        self.stop = Event()
        self.thread = Thread(target=self._loop, daemon=True, name="terraintrace-runner")

    def start(self) -> None:
        self.store.interrupt_running_jobs()
        self.thread.start()

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=3)

    def submit(self) -> None:
        self.wake.set()

    def _loop(self) -> None:
        while not self.stop.is_set():
            item = self.store.claim_job()
            if item is None:
                self.wake.wait(0.25)
                self.wake.clear()
                continue
            job_id, run_id, request, total = item
            self._execute(job_id, run_id, request, total)

    def _execute(self, job_id: str, run_id: str, request: dict[str, object], total: int) -> None:
        run_record = self.store.run(run_id)
        run_payload = run_record["payload"]
        try:
            dataset_record = self.store.dataset(str(request["dataset_id"]), int(request["dataset_revision"]))
            manifest = dataset_record["manifest"]
            entries = manifest["cases"]
            mode = str(dataset_record["mode"])
            loaded = [_load_manifest_case(self.runtime_dir, row, mode) for row in entries]
            if mode == "benchmark":
                imported = tuple(row for row in loaded if isinstance(row, BenchmarkImport))
                validate_benchmark_collection(imported)
                train = tuple(row.case for row in imported if row.split == "train")
                prior = fit_spectral_prior(train) if "spectral_prior" in request["methods"] else None
                _validate_prior_compatibility(imported, prior)
            else:
                prior = None
            case_results: list[dict[str, object]] = []
            done = 0
            for source in loaded:
                if mode == "benchmark":
                    result = run_benchmark_case(source.case, prior, tuple(request["methods"]))
                    input_shape = source.case.low_res.shape
                    split = source.split
                    group = source.case.group
                else:
                    result = run_inference_case(source, tuple(request["methods"]))
                    input_shape = source.low_res.shape
                    split = "inference"
                    group = None
                arrays_meta: dict[str, object] = {}
                for method, artifact in result.arrays.items():
                    rel = Path("runs") / run_id / result.case_id / f"{method}.npy"
                    _write_npy_atomic(self.runtime_dir / rel, artifact.values)
                    arrays_meta[method] = {**artifact.metadata(), "path": str(rel)}
                case_record = {
                    "case_id": result.case_id,
                    "group": group,
                    "split": split,
                    "status": "succeeded",
                    "input_shape": list(input_shape),
                    "arrays": arrays_meta,
                    "metrics": {name: metric.as_dict() if metric else None for name, metric in result.metrics.items()},
                }
                case_results.append(case_record)
                done += 1
                run_payload["cases"] = list(case_results)
                run_payload["status"] = "running"
                if self.store.cancel_requested(job_id):
                    run_payload["status"] = "cancelled"
                    self.store.update_run(run_id, run_payload)
                    self.store.update_job(job_id, "cancelled", done)
                    return
                self.store.update_run(run_id, run_payload)
                self.store.update_job(job_id, "running", done)
                if self.store.cancel_requested(job_id):
                    run_payload["status"] = "cancelled"
                    self.store.update_run(run_id, run_payload)
                    self.store.update_job(job_id, "cancelled", done)
                    return
            if self.store.cancel_requested(job_id):
                run_payload["status"] = "cancelled"
                self.store.update_run(run_id, run_payload)
                self.store.update_job(job_id, "cancelled", done)
                return
            run_payload["status"] = "succeeded"
            run_payload["prior"] = _prior_metadata(prior, loaded, manifest)
            self.store.update_run(run_id, run_payload)
            self.store.update_job(job_id, "succeeded", done)
        except Exception as error:
            run_payload["status"] = "failed"
            run_payload["error"] = str(error)
            self.store.update_run(run_id, run_payload)
            self.store.update_job(job_id, "failed", len(run_payload.get("cases", [])), error=str(error))


def create_app(runtime_dir: Path = Path("runtime/terraintrace")) -> FastAPI:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = runtime_dir.resolve()
    store = LocalStore(runtime_dir / "terraintrace.sqlite")
    worker = RunWorker(store, runtime_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        worker.start()
        try:
            yield
        finally:
            worker.close()
            if not worker.thread.is_alive():
                store.close()

    app = FastAPI(title="TerrainTrace local lab", version="1", lifespan=lifespan)
    app.state.store = store
    app.state.worker = worker
    app.state.runtime_dir = runtime_dir
    static_dir = Path(__file__).resolve().parent.parent / "static"

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, error: HTTPException) -> JSONResponse:
        detail = error.detail if isinstance(error.detail, dict) else {}
        message = str(detail.get("message", error.detail if isinstance(error.detail, str) else "请求处理失败"))
        default_code = {
            404: "NOT_FOUND",
            413: "PAYLOAD_TOO_LARGE",
            422: "INVALID_INPUT",
            429: "QUEUE_FULL",
        }.get(error.status_code, "REQUEST_FAILED")
        body = {
            "error": {
                "code": str(detail.get("code", default_code)),
                "message": message,
                "field_errors": detail.get("field_errors", []),
                "retryable": error.status_code == 429 or error.status_code >= 500,
            }
        }
        return JSONResponse(status_code=error.status_code, content=body)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        body = {
            "error": {
                "code": "INVALID_REQUEST",
                "message": "请求字段不合法",
                "field_errors": [
                    {
                        "path": ".".join(str(part) for part in issue["loc"]),
                        "message": issue["msg"],
                    }
                    for issue in error.errors()
                ],
                "retryable": False,
            }
        }
        return JSONResponse(status_code=422, content=body)

    @app.get("/api/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "runtime": str(runtime_dir)}

    if (static_dir / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="terraintrace-assets")

        @app.get("/", include_in_schema=False)
        def frontend() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    @app.get("/api/v1/demo/cases")
    def demo_cases(seed: int = 17) -> dict[str, object]:
        dataset = create_dataset(seed)
        return {"seed": seed, "cases": _fixture_manifest(dataset)}

    @app.post("/api/v1/demo/datasets")
    def save_demo_dataset(seed: int = 17) -> dict[str, object]:
        dataset = create_dataset(seed)
        dataset_id = f"demo_{seed}"
        manifest = _persist_fixture(runtime_dir, dataset_id, dataset)
        revision = store.save_dataset(dataset_id, "benchmark", manifest)
        return {"dataset_id": dataset_id, "revision": revision, "cases": len(manifest["cases"])}

    @app.post("/api/v1/terrain-datasets")
    async def import_dataset(
        files: Annotated[list[UploadFile], File()],
        mode: Literal["benchmark", "inference"] = Form(...),
        metadata: str = Form(...),
        dataset_id: str | None = Form(default=None),
    ) -> dict[str, object]:
        if not files or (mode == "inference" and len(files) != 1) or len(files) > 60:
            raise HTTPException(422, detail={"message": "inference needs one NPZ; benchmark accepts 1..60"})
        parsed = _parse_metadata_collection(metadata, mode, len(files))
        dataset_id = _validate_dataset_id(dataset_id) if dataset_id else f"ds_{uuid4().hex[:12]}"
        staging = runtime_dir / ".staging"
        assets = runtime_dir / "assets"
        staging.mkdir(parents=True, exist_ok=True)
        assets.mkdir(parents=True, exist_ok=True)
        staged: list[Path] = []
        moved: list[Path] = []
        try:
            imported_rows: list[BenchmarkImport | InferenceCase] = []
            manifest_cases: list[dict[str, object]] = []
            for index, (upload, case_metadata) in enumerate(zip(files, parsed, strict=True)):
                asset_id = f"asset_{uuid4().hex}.npz"
                stage_path = staging / asset_id
                staged.append(stage_path)
                size = await _save_upload_limited(upload, stage_path)
                asset_sha256 = hashlib.sha256(stage_path.read_bytes()).hexdigest()
                source = (
                    load_benchmark_npz(stage_path, case_metadata)
                    if mode == "benchmark"
                    else load_inference_npz(stage_path, case_metadata)
                )
                asset_path = assets / asset_id
                os.replace(stage_path, asset_path)
                moved.append(asset_path)
                relative = str(asset_path.relative_to(runtime_dir))
                if isinstance(source, BenchmarkImport):
                    imported_rows.append(source)
                    case = source.case
                    row = {
                        "id": case.id,
                        "group": case.group,
                        "split": source.split,
                        "scale": case.scale,
                        "pixel_size": case.pixel_size,
                        "elevation_unit": source.elevation_unit,
                        "shape": list(case.high_res.shape),
                        "array_sha256": source.array_sha256,
                        "asset_sha256": asset_sha256,
                        "asset_path": relative,
                        "display_name": upload.filename or f"case-{index}.npz",
                        "bytes": size,
                    }
                else:
                    imported_rows.append(source)
                    row = {
                        "id": source.id,
                        "scale": source.scale,
                        "pixel_size": source.pixel_size,
                        "elevation_unit": source.elevation_unit,
                        "shape": list(source.low_res.shape),
                        "asset_sha256": asset_sha256,
                        "asset_path": relative,
                        "display_name": upload.filename or "inference.npz",
                        "bytes": size,
                    }
                manifest_cases.append(row)
            if mode == "benchmark":
                validate_benchmark_collection(tuple(row for row in imported_rows if isinstance(row, BenchmarkImport)))
                _validate_uniform_dataset(imported_rows)
            manifest = {"mode": mode, "cases": manifest_cases}
            revision = store.save_dataset(dataset_id, mode, manifest)
        except Exception as error:
            for path in (*staged, *moved):
                path.unlink(missing_ok=True)
            if isinstance(error, HTTPException):
                raise
            raise HTTPException(422, detail={"message": str(error)}) from error
        return {"dataset_id": dataset_id, "revision": revision, "mode": mode, "cases": manifest_cases}

    @app.get("/api/v1/terrain-datasets/{dataset_id}/{revision}")
    def get_dataset(dataset_id: str, revision: int) -> dict[str, object]:
        try:
            return store.dataset(dataset_id, revision)
        except KeyError as error:
            raise HTTPException(404, detail={"message": str(error)}) from error

    @app.post("/api/v1/reconstruction-runs", status_code=202)
    def create_reconstruction_run(request: RunInput) -> dict[str, object]:
        try:
            dataset = store.dataset(request.dataset_id, request.dataset_revision)
        except KeyError as error:
            raise HTTPException(404, detail={"message": str(error)}) from error
        mode = str(dataset["mode"])
        if mode != request.mode:
            raise HTTPException(422, detail={"message": "run mode must match saved dataset mode"})
        if not request.methods or set(request.methods) - METHODS or len(set(request.methods)) != len(request.methods):
            raise HTTPException(422, detail={"message": "methods must be unique supported names"})
        if mode == "inference" and "spectral_prior" in request.methods:
            raise HTTPException(422, detail={"message": "spectral_prior requires benchmark training cases"})
        if mode == "inference" and len(dataset["manifest"]["cases"]) != 1:
            raise HTTPException(422, detail={"message": "inference dataset must contain one observation"})
        if (
            mode == "benchmark"
            and "spectral_prior" in request.methods
            and not any(row.get("split") == "train" for row in dataset["manifest"]["cases"])
        ):
            raise HTTPException(422, detail={"message": "spectral_prior requires at least one train case"})
        config = request.model_dump()
        config_sha256 = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        input_sha256 = hashlib.sha256(
            json.dumps(dataset["manifest"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        run_payload: dict[str, object] = {
            "status": "queued",
            "config": config,
            "config_sha256": config_sha256,
            "input_sha256": input_sha256,
            "engine_version": "0.1.0",
            "mode": mode,
            "metrics_protocol": request.metrics_protocol,
            "cases": [],
        }
        try:
            run_id, job_id = store.create_run_job(
                request.dataset_id,
                request.dataset_revision,
                run_payload,
                config,
                len(dataset["manifest"]["cases"]),
            )
        except KeyError as error:
            raise HTTPException(404, detail={"message": str(error)}) from error
        except QueueFullError as error:
            raise HTTPException(429, detail={"message": str(error)}) from error
        worker.submit()
        return {"run_id": run_id, "job_id": job_id, "status": "queued", "dataset_revision": request.dataset_revision}

    @app.get("/api/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, object]:
        try:
            return store.job(job_id)
        except KeyError as error:
            raise HTTPException(404, detail={"message": str(error)}) from error

    @app.post("/api/v1/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, object]:
        try:
            return store.cancel_job(job_id)
        except KeyError as error:
            raise HTTPException(404, detail={"message": str(error)}) from error

    @app.get("/api/v1/jobs/{job_id}/events")
    def job_events(job_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
        _require_job(store, job_id)

        def stream():
            seq = after
            while True:
                for event in store.events(job_id, seq):
                    seq = int(event["seq"])
                    yield f"id: {seq}\nevent: {event['kind']}\ndata: {json.dumps(event['payload'])}\n\n"
                status = store.job(job_id)["status"]
                if status in {"succeeded", "failed", "cancelled", "interrupted"}:
                    return
                time.sleep(0.1)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        return _require_run(store, run_id)

    @app.get("/api/v1/runs/{run_id}/cases")
    def run_cases(run_id: str) -> dict[str, object]:
        record = _require_run(store, run_id)
        return {
            "run_id": run_id,
            "status": record["payload"].get("status"),
            "cases": record["payload"].get("cases", []),
        }

    @app.get("/api/v1/runs/{run_id}/cases/{case_id}/metrics")
    def case_metrics(run_id: str, case_id: str) -> dict[str, object]:
        case = _require_case(store, run_id, case_id)
        return {
            "case_id": case_id,
            "protocol": _require_run(store, run_id)["payload"]["metrics_protocol"],
            "status": case["status"],
            "metrics": case["metrics"],
        }

    @app.get("/api/v1/runs/{run_id}/cases/{case_id}/arrays")
    def case_array(
        run_id: str, case_id: str, method: str | None = None, kind: str = "reconstruction", display: bool = False
    ) -> Response:
        run = _require_run(store, run_id)
        case_record = _require_case(store, run_id, case_id)
        manifest_case = _manifest_case(store, run, case_id)
        if kind in {"difference", "absolute_error"}:
            if run["payload"]["mode"] == "inference":
                raise HTTPException(404, detail={"message": "inference has no high_res target for error arrays"})
            if method is None or method not in case_record["arrays"]:
                raise HTTPException(404, detail={"message": "reconstruction method not found"})
            source = _load_manifest_case(runtime_dir, manifest_case, "benchmark")
            if not isinstance(source, BenchmarkImport):
                raise HTTPException(404, detail={"message": "high_res target is unavailable"})
            reconstruction = _load_registered_array(runtime_dir, case_record["arrays"][method])
            residual = reconstruction - source.case.high_res
            values = np.abs(residual) if kind == "absolute_error" else residual
            array_name = kind
        elif kind in {"low_res", "high_res"}:
            if kind == "high_res" and run["payload"]["mode"] == "inference":
                raise HTTPException(404, detail={"message": "inference has no high_res target"})
            source = _load_manifest_case(runtime_dir, manifest_case, str(run["payload"]["mode"]))
            if isinstance(source, BenchmarkImport):
                values = source.case.high_res if kind == "high_res" else source.case.low_res
            else:
                values = source.low_res
            array_name = kind
        else:
            if method is None or method not in case_record["arrays"]:
                raise HTTPException(404, detail={"message": "reconstruction method not found"})
            metadata = case_record["arrays"][method]
            values = _load_registered_array(runtime_dir, metadata)
            array_name = method
        if display:
            values = _display_grid(values)
        buffer = io.BytesIO()
        np.save(buffer, values, allow_pickle=False)
        digest = hashlib.sha256(buffer.getvalue()).hexdigest()
        header_meta = case_record["arrays"].get(array_name, {})
        headers = {
            "X-Terrain-Shape": json.dumps(list(values.shape)),
            "X-Terrain-Dtype": str(values.dtype),
            "X-Terrain-SHA256": digest,
            "X-Terrain-Array-SHA256": hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest(),
            "X-Terrain-Value-Range": json.dumps([float(np.min(values)), float(np.max(values))]),
            "X-Terrain-Display-Grid": "true" if display else "false",
        }
        if header_meta and "low_resolution_rmse" in header_meta:
            headers["X-Terrain-LR-RMSE"] = str(header_meta["low_resolution_rmse"])
        return Response(buffer.getvalue(), media_type="application/x-npy", headers=headers)

    @app.post("/api/v1/runs/{run_id}/profiles")
    def run_profile(run_id: str, request: ProfileInput) -> dict[str, object]:
        record = _require_run(store, run_id)
        case_id = request.case_id
        case_record = _require_case(store, run_id, case_id)
        result = _case_run_from_record(runtime_dir, case_record, str(record["payload"]["mode"]), request.methods)
        case = _load_manifest_case(runtime_dir, _manifest_case(store, record, case_id), record["payload"]["mode"])
        pixel_size = case.case.pixel_size if isinstance(case, BenchmarkImport) else case.pixel_size
        reference = case.case.high_res if isinstance(case, BenchmarkImport) else None
        return profile(result, request.points, request.samples, pixel_size, reference)

    @app.post("/api/v1/runs/{run_id}/roi")
    def run_roi(run_id: str, request: RoiInput) -> dict[str, object]:
        record = _require_run(store, run_id)
        case_record = _require_case(store, run_id, request.case_id)
        return roi_metrics(
            _case_run_from_record(runtime_dir, case_record, str(record["payload"]["mode"]), request.methods),
            request.bounds,
        )

    @app.post("/api/v1/runs/{run_id}/exports")
    def export_case(run_id: str, case_id: str) -> dict[str, object]:
        record = _require_run(store, run_id)
        case_record = _require_case(store, run_id, case_id)
        source = _load_manifest_case(runtime_dir, _manifest_case(store, record, case_id), record["payload"]["mode"])
        result = _case_run_from_record(runtime_dir, case_record, str(record["payload"]["mode"]))
        path = runtime_dir / "artifacts" / f"{run_id}-{uuid4().hex[:8]}.zip"
        proof = write_case_archive(path, source.case if isinstance(source, BenchmarkImport) else source, result)
        artifact_id = store.register_artifact(
            run_id, case_id, str(path.relative_to(runtime_dir)), str(proof["sha256"]), int(proof["bytes"])
        )
        return {"artifact_id": artifact_id, **proof}

    @app.get("/api/v1/artifacts/{artifact_id}/download")
    def download_artifact(artifact_id: str) -> FileResponse:
        try:
            artifact = store.artifact(artifact_id)
        except KeyError as error:
            raise HTTPException(404, detail={"message": str(error)}) from error
        path = _safe_path(runtime_dir, str(artifact["path"]))
        if not path.is_file():
            raise HTTPException(404, detail={"message": "artifact file missing"})
        return FileResponse(path, filename=f"terraintrace-{artifact_id}.zip", media_type="application/zip")

    return app


def _parse_metadata_collection(text: str, mode: str, count: int) -> list[dict[str, object]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise HTTPException(422, detail={"message": "metadata must be valid JSON"}) from error
    if mode == "benchmark":
        if not isinstance(value, list) or len(value) != count or not all(isinstance(row, dict) for row in value):
            raise HTTPException(422, detail={"message": "benchmark metadata must be a matching JSON array"})
        return value
    if not isinstance(value, dict) or count != 1:
        raise HTTPException(422, detail={"message": "inference needs one metadata object and one NPZ"})
    return [value]


async def _save_upload_limited(upload: UploadFile, path: Path) -> int:
    total = 0
    with path.open("wb") as handle:
        while chunk := await upload.read(64 * 1024):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(413, detail={"message": "NPZ file exceeds 32 MiB"})
            handle.write(chunk)
    if total == 0:
        raise HTTPException(422, detail={"message": "NPZ file is empty"})
    return total


def _validate_dataset_id(value: str | None) -> str:
    if value is None or not value or len(value) > 64 or not all(char.isalnum() or char in "-_" for char in value):
        raise HTTPException(422, detail={"message": "dataset_id must contain only letters, digits, - and _"})
    return value


def _validate_uniform_dataset(entries: list[BenchmarkImport | InferenceCase]) -> None:
    scales = {entry.case.scale if isinstance(entry, BenchmarkImport) else entry.scale for entry in entries}
    pixels = {entry.case.pixel_size if isinstance(entry, BenchmarkImport) else entry.pixel_size for entry in entries}
    units = {entry.elevation_unit if isinstance(entry, BenchmarkImport) else entry.elevation_unit for entry in entries}
    if len(scales) != 1 or len(pixels) != 1 or len(units) != 1:
        raise ValueError("dataset cases must share scale, pixel_size, and elevation_unit")


def _fixture_manifest(dataset: TerrainDataset) -> list[dict[str, object]]:
    return [
        {"id": case.id, "group": case.group, "split": split, "shape": list(case.high_res.shape)}
        for split, rows in (("train", dataset.train), ("validation", dataset.validation), ("test", dataset.test))
        for case in rows
    ]


def _persist_fixture(runtime_dir: Path, dataset_id: str, dataset: TerrainDataset) -> dict[str, object]:
    assets = runtime_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, object]] = []
    split_by_id = {
        case.id: split
        for split, rows in (("train", dataset.train), ("validation", dataset.validation), ("test", dataset.test))
        for case in rows
    }
    for case in dataset.all_cases():
        filename = f"asset_{uuid4().hex}.npz"
        path = assets / filename
        np.savez_compressed(path, high_res=case.high_res)
        cases.append(
            {
                "id": case.id,
                "group": case.group,
                "split": split_by_id[case.id],
                "scale": case.scale,
                "pixel_size": case.pixel_size,
                "elevation_unit": "synthetic elevation units",
                "shape": list(case.high_res.shape),
                "array_sha256": hashlib.sha256(np.ascontiguousarray(case.high_res).tobytes()).hexdigest(),
                "asset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "asset_path": str(path.relative_to(runtime_dir)),
                "display_name": f"{case.id}.npz",
            }
        )
    return {"seed": int(dataset_id.removeprefix("demo_")), "mode": "benchmark", "cases": cases}


def _safe_path(runtime_dir: Path, relative: str) -> Path:
    root = runtime_dir.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("stored file path escaped runtime directory")
    return path


def _load_manifest_case(runtime_dir: Path, row: dict[str, object], mode: str) -> BenchmarkImport | InferenceCase:
    path = _safe_path(runtime_dir, str(row["asset_path"]))
    metadata = {key: row[key] for key in ("id", "scale", "pixel_size", "elevation_unit")}
    if mode == "benchmark":
        metadata.update({"group": row["group"], "split": row["split"]})
        return load_benchmark_npz(path, metadata)
    return load_inference_npz(path, metadata)


def _validate_prior_compatibility(entries: tuple[BenchmarkImport, ...], prior: SpectralPrior | None) -> None:
    if prior is None:
        return
    train = [entry for entry in entries if entry.split == "train"]
    if not train:
        raise ValueError("spectral prior requires at least one train case")
    first = train[0]
    signature = (first.case.high_res.shape, first.case.scale, first.case.pixel_size, first.elevation_unit)
    if any(
        (entry.case.high_res.shape, entry.case.scale, entry.case.pixel_size, entry.elevation_unit) != signature
        for entry in entries
    ):
        raise ValueError("spectral prior requires compatible shape, scale, pixel_size, and elevation unit")


def _prior_metadata(
    prior: SpectralPrior | None, loaded: list[BenchmarkImport | InferenceCase], manifest: dict[str, object]
) -> dict[str, object] | None:
    if prior is None:
        return None
    train = [row for row in loaded if isinstance(row, BenchmarkImport) and row.split == "train"]
    return {
        "bins": len(prior.gains),
        "radius_centers": prior.radius_centers.tolist(),
        "gains": prior.gains.tolist(),
        "clip": [0.65, 1.55],
        "train_case_hashes": [row.array_sha256 for row in train],
        "scale": prior.scale,
        "compatible_input": {
            "shape": list(train[0].case.high_res.shape),
            "pixel_size": train[0].case.pixel_size,
            "elevation_unit": train[0].elevation_unit,
        },
    }


def _write_npy_atomic(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_npy(runtime_dir: Path, relative: str) -> np.ndarray:
    path = _safe_path(runtime_dir, relative)
    with path.open("rb") as handle:
        result = np.load(handle, allow_pickle=False)
    if not isinstance(result, np.ndarray) or result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("registered array artifact is invalid")
    return result


def _display_grid(values: np.ndarray, limit: int = 256) -> np.ndarray:
    rows = np.linspace(0, values.shape[0] - 1, min(limit, values.shape[0])).round().astype(int)
    cols = np.linspace(0, values.shape[1] - 1, min(limit, values.shape[1])).round().astype(int)
    return values[np.ix_(rows, cols)]


def _require_job(store: LocalStore, job_id: str) -> dict[str, object]:
    try:
        return store.job(job_id)
    except KeyError as error:
        raise HTTPException(404, detail={"message": str(error)}) from error


def _require_run(store: LocalStore, run_id: str) -> dict[str, object]:
    try:
        return store.run(run_id)
    except KeyError as error:
        raise HTTPException(404, detail={"message": str(error)}) from error


def _require_case(store: LocalStore, run_id: str, case_id: str) -> dict[str, object]:
    record = _require_run(store, run_id)
    case = next((row for row in record["payload"].get("cases", []) if row.get("case_id") == case_id), None)
    if case is None:
        raise HTTPException(404, detail={"message": "case result not found"})
    return case


def _manifest_case(store: LocalStore, run: dict[str, object], case_id: str) -> dict[str, object]:
    dataset = store.dataset(run["dataset_id"], int(run["dataset_revision"]))
    case = next((row for row in dataset["manifest"]["cases"] if row.get("id") == case_id), None)
    if case is None:
        raise HTTPException(404, detail={"message": "case not in dataset revision"})
    return case


def _case_run_from_record(
    runtime_dir: Path, case: dict[str, object], mode: str, methods: list[str] | None = None
) -> CaseRun:
    selected = methods or list(case["arrays"].keys())
    arrays: dict[str, ArrayArtifact] = {}
    metrics: dict[str, MetricSet | None] = {}
    for method in selected:
        if method not in case["arrays"]:
            raise HTTPException(404, detail={"message": f"method {method} not found for case"})
        meta = case["arrays"][method]
        values = _load_registered_array(runtime_dir, meta)
        arrays[method] = ArrayArtifact(
            method, values, meta["sha256"], tuple(meta["value_range"]), meta["low_resolution_rmse"]
        )
        metrics[method] = _metric_from_record(case["metrics"].get(method))
    return CaseRun(case["case_id"], mode, arrays, metrics)


def _metric_from_record(value: object) -> MetricSet | None:
    if not isinstance(value, dict):
        return None
    correlation = value["gradientCorrelation"]
    return MetricSet(
        rmse=float(value["rmse"]),
        mae=float(value["mae"]),
        slope_mae=float(value["slopeMae"]),
        gradient_correlation=float(correlation) if correlation is not None else None,
        gradient_correlation_reason=value["gradientCorrelationReason"],
        edge_rmse=float(value["edgeRmse"]),
        low_resolution_rmse=float(value["lowResolutionRmse"]),
        area_consistency_max=float(value["areaConsistencyMax"]),
    )


def _load_registered_array(runtime_dir: Path, metadata: dict[str, object]) -> np.ndarray:
    values = _load_npy(runtime_dir, str(metadata["path"]))
    digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    if (
        digest != metadata["sha256"]
        or list(values.shape) != metadata["shape"]
        or str(values.dtype) != metadata["dtype"]
        or [float(values.min()), float(values.max())] != metadata["value_range"]
    ):
        raise HTTPException(500, detail={"message": "registered reconstruction array failed integrity check"})
    return values
