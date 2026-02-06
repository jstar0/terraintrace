"""SQLite persistence boundary for local TerrainTrace datasets and runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from threading import RLock
from uuid import uuid4


class QueueFullError(RuntimeError):
    """Raised when five jobs are already waiting in the local queue."""


class LocalStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = RLock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS datasets (
              dataset_id TEXT NOT NULL, revision INTEGER NOT NULL,
              mode TEXT NOT NULL, manifest TEXT NOT NULL, created_at INTEGER NOT NULL,
              PRIMARY KEY(dataset_id, revision)
            );
            CREATE TABLE IF NOT EXISTS runs (
              run_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL,
              dataset_revision INTEGER NOT NULL, payload TEXT NOT NULL,
              created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, status TEXT NOT NULL,
              payload TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
              total INTEGER NOT NULL DEFAULT 0, error TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
              created_at INTEGER NOT NULL,
              updated_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_events (
              job_id TEXT NOT NULL, seq INTEGER NOT NULL, kind TEXT NOT NULL,
              payload TEXT NOT NULL, created_at INTEGER NOT NULL,
              PRIMARY KEY(job_id, seq)
            );
            CREATE TABLE IF NOT EXISTS artifacts (
              artifact_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, case_id TEXT NOT NULL,
              path TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
              created_at INTEGER NOT NULL
            );
            """
        )
        job_columns = {row[1] for row in self.connection.execute("PRAGMA table_info(jobs)")}
        if "cancel_requested" not in job_columns:
            self.connection.execute("ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0")

    def save_dataset(self, dataset_id: str, mode: str, manifest: dict[str, object]) -> int:
        with self.lock, self.connection:
            row = self.connection.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM datasets WHERE dataset_id=?", (dataset_id,)
            ).fetchone()
            revision = int(row[0]) + 1
            self.connection.execute(
                "INSERT INTO datasets(dataset_id, revision, mode, manifest, created_at) VALUES (?, ?, ?, ?, ?)",
                (dataset_id, revision, mode, json.dumps(manifest, sort_keys=True), int(time.time())),
            )
        return revision

    def dataset(self, dataset_id: str, revision: int) -> dict[str, object]:
        with self.lock:
            row = self.connection.execute(
                "SELECT mode, manifest FROM datasets WHERE dataset_id=? AND revision=?", (dataset_id, revision)
            ).fetchone()
        if row is None:
            raise KeyError("dataset revision not found")
        return {"dataset_id": dataset_id, "revision": revision, "mode": row[0], "manifest": json.loads(row[1])}

    def save_run(self, dataset_id: str, revision: int, payload: dict[str, object]) -> str:
        run_id = f"run_{uuid4().hex[:12]}"
        with self.lock, self.connection:
            exists = self.connection.execute(
                "SELECT 1 FROM datasets WHERE dataset_id=? AND revision=?", (dataset_id, revision)
            ).fetchone()
            if exists is None:
                raise KeyError("dataset revision not found")
            self.connection.execute(
                "INSERT INTO runs(run_id, dataset_id, dataset_revision, payload, created_at) VALUES (?, ?, ?, ?, ?)",
                (run_id, dataset_id, revision, json.dumps(payload, sort_keys=True), int(time.time())),
            )
        return run_id

    def create_run_job(
        self,
        dataset_id: str,
        revision: int,
        run_payload: dict[str, object],
        job_payload: dict[str, object],
        total: int,
    ) -> tuple[str, str]:
        """Persist a run and its job together, enforcing the local queue limit."""
        run_id = f"run_{uuid4().hex[:12]}"
        job_id = f"job_{uuid4().hex[:12]}"
        now = int(time.time())
        payload = {**run_payload, "job_id": job_id}
        with self.lock, self.connection:
            exists = self.connection.execute(
                "SELECT 1 FROM datasets WHERE dataset_id=? AND revision=?", (dataset_id, revision)
            ).fetchone()
            if exists is None:
                raise KeyError("dataset revision not found")
            queued = self.connection.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
            if queued >= 5:
                raise QueueFullError("local job queue is full (maximum 5 queued jobs)")
            self.connection.execute(
                "INSERT INTO runs(run_id,dataset_id,dataset_revision,payload,created_at) VALUES(?,?,?,?,?)",
                (run_id, dataset_id, revision, json.dumps(payload, sort_keys=True), now),
            )
            self.connection.execute(
                "INSERT INTO jobs(job_id,run_id,status,payload,done,total,created_at,updated_at) "
                "VALUES(?,?,'queued',?,0,?,?,?)",
                (job_id, run_id, json.dumps(job_payload, sort_keys=True), total, now, now),
            )
            self._event(job_id, "queued", {"done": 0, "total": total})
        return run_id, job_id

    def run(self, run_id: str) -> dict[str, object]:
        with self.lock:
            row = self.connection.execute(
                "SELECT dataset_id, dataset_revision, payload FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            raise KeyError("run not found")
        return {"run_id": run_id, "dataset_id": row[0], "dataset_revision": row[1], "payload": json.loads(row[2])}

    def update_run(self, run_id: str, payload: dict[str, object]) -> None:
        with self.lock, self.connection:
            cursor = self.connection.execute(
                "UPDATE runs SET payload=? WHERE run_id=?", (json.dumps(payload, sort_keys=True), run_id)
            )
            if cursor.rowcount == 0:
                raise KeyError("run not found")

    def create_job(self, run_id: str, payload: dict[str, object], total: int) -> str:
        job_id = f"job_{uuid4().hex[:12]}"
        now = int(time.time())
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO jobs(job_id,run_id,status,payload,done,total,created_at,updated_at) "
                "VALUES(?,?,'queued',?,0,?,?,?)",
                (job_id, run_id, json.dumps(payload, sort_keys=True), total, now, now),
            )
            self._event(job_id, "queued", {"done": 0, "total": total})
        return job_id

    def claim_job(self) -> tuple[str, str, dict[str, object], int] | None:
        with self.lock, self.connection:
            row = self.connection.execute(
                "SELECT job_id,run_id,payload,total FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job_id, run_id, payload, total = row
            self.connection.execute(
                "UPDATE jobs SET status='running',updated_at=? WHERE job_id=?", (int(time.time()), job_id)
            )
            run = self.connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is not None:
                run_payload = json.loads(run[0])
                run_payload["status"] = "running"
                self.connection.execute(
                    "UPDATE runs SET payload=? WHERE run_id=?", (json.dumps(run_payload, sort_keys=True), run_id)
                )
            self._event(job_id, "running", {"done": 0, "total": total})
        return str(job_id), str(run_id), json.loads(payload), int(total)

    def cancel_job(self, job_id: str) -> dict[str, object]:
        with self.lock, self.connection:
            row = self.connection.execute("SELECT run_id,status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError("job not found")
            run_id, status = row
            if status == "queued":
                self.connection.execute(
                    "UPDATE jobs SET status='cancelled',updated_at=? WHERE job_id=?", (int(time.time()), job_id)
                )
                run = self.connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is not None:
                    payload = json.loads(run[0])
                    payload["status"] = "cancelled"
                    self.connection.execute(
                        "UPDATE runs SET payload=? WHERE run_id=?", (json.dumps(payload, sort_keys=True), run_id)
                    )
                self._event(job_id, "cancelled", {"done": 0})
            elif status in {"running", "cancelling"}:
                self.connection.execute(
                    "UPDATE jobs SET status='cancelling',cancel_requested=1,updated_at=? WHERE job_id=?",
                    (int(time.time()), job_id),
                )
                run = self.connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is not None:
                    payload = json.loads(run[0])
                    payload["status"] = "cancelling"
                    self.connection.execute(
                        "UPDATE runs SET payload=? WHERE run_id=?", (json.dumps(payload, sort_keys=True), run_id)
                    )
                self._event(job_id, "cancelling", {"message": "finish current case"})
        return self.job(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        with self.lock:
            row = self.connection.execute("SELECT cancel_requested FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return bool(row and row[0])

    def update_job(self, job_id: str, status: str, done: int, *, error: str | None = None) -> None:
        with self.lock, self.connection:
            row = self.connection.execute(
                "SELECT total,cancel_requested FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError("job not found")
            effective_status = "cancelling" if row[1] and status not in {"succeeded", "failed", "cancelled"} else status
            self.connection.execute(
                "UPDATE jobs SET status=?,done=?,error=?,updated_at=? WHERE job_id=?",
                (effective_status, done, error, int(time.time()), job_id),
            )
            self._event(
                job_id,
                effective_status
                if effective_status in {"succeeded", "failed", "cancelled", "interrupted"}
                else "progress",
                {"done": done, "total": row[0], "error": error, "status": effective_status},
            )

    def job(self, job_id: str) -> dict[str, object]:
        with self.lock:
            row = self.connection.execute(
                "SELECT run_id,status,done,total,error FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise KeyError("job not found")
        return {
            "job_id": job_id,
            "run_id": row[0],
            "status": row[1],
            "progress": {"done": row[2], "total": row[3]},
            "error": row[4],
        }

    def interrupt_running_jobs(self) -> None:
        with self.lock, self.connection:
            rows = self.connection.execute(
                "SELECT job_id,run_id,done FROM jobs WHERE status IN ('running','cancelling')"
            ).fetchall()
            for job_id, run_id, done in rows:
                self.connection.execute(
                    "UPDATE jobs SET status='interrupted',updated_at=? WHERE job_id=?", (int(time.time()), job_id)
                )
                run = self.connection.execute("SELECT payload FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is not None:
                    payload = json.loads(run[0])
                    payload["status"] = "interrupted"
                    self.connection.execute(
                        "UPDATE runs SET payload=? WHERE run_id=?", (json.dumps(payload, sort_keys=True), run_id)
                    )
                self._event(job_id, "interrupted", {"done": done, "message": "server restarted"})

    def register_artifact(self, run_id: str, case_id: str, path: str, sha256: str, size: int) -> str:
        artifact_id = f"artifact_{uuid4().hex[:12]}"
        with self.lock, self.connection:
            self.connection.execute(
                "INSERT INTO artifacts(artifact_id,run_id,case_id,path,sha256,bytes,created_at) VALUES(?,?,?,?,?,?,?)",
                (artifact_id, run_id, case_id, path, sha256, size, int(time.time())),
            )
        return artifact_id

    def artifact(self, artifact_id: str) -> dict[str, object]:
        with self.lock:
            row = self.connection.execute(
                "SELECT run_id,case_id,path,sha256,bytes FROM artifacts WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError("artifact not found")
        return {
            "artifact_id": artifact_id,
            "run_id": row[0],
            "case_id": row[1],
            "path": row[2],
            "sha256": row[3],
            "bytes": row[4],
        }

    def events(self, job_id: str, after: int = 0) -> list[dict[str, object]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT seq,kind,payload FROM job_events WHERE job_id=? AND seq>? ORDER BY seq",
                (job_id, after),
            ).fetchall()
        return [{"seq": row[0], "kind": row[1], "payload": json.loads(row[2])} for row in rows]

    def _event(self, job_id: str, kind: str, payload: dict[str, object]) -> None:
        seq = self.connection.execute(
            "SELECT COALESCE(MAX(seq),0)+1 FROM job_events WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        self.connection.execute(
            "INSERT INTO job_events(job_id,seq,kind,payload,created_at) VALUES(?,?,?,?,?)",
            (job_id, int(seq), kind, json.dumps(payload, sort_keys=True), int(time.time())),
        )

    def close(self) -> None:
        with self.lock:
            self.connection.close()
