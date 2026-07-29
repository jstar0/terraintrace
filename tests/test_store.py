import pytest

from terraintrace.app.store import LocalStore, QueueFullError


def test_local_store_initializes_wal_dataset_and_run_tables(tmp_path) -> None:
    store = LocalStore(tmp_path / "terraintrace.sqlite")
    names = {row[0] for row in store.connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"datasets", "runs"} <= names
    assert store.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store.save_dataset("terrain_local", "benchmark", {"cases": 1}) == 1
    assert store.save_dataset("terrain_local", "benchmark", {"cases": 2}) == 2
    assert store.dataset("terrain_local", 2)["manifest"] == {"cases": 2}


def test_run_and_job_are_persisted_together_and_queued_cancel_updates_run(tmp_path) -> None:
    store = LocalStore(tmp_path / "terraintrace.sqlite")
    store.save_dataset("terrain_local", "benchmark", {"cases": 1})

    run_id, job_id = store.create_run_job(
        "terrain_local", 1, {"status": "queued", "cases": []}, {"methods": ["bilinear"]}, 1
    )
    assert store.run(run_id)["payload"]["job_id"] == job_id
    assert store.job(job_id)["status"] == "queued"

    cancelled = store.cancel_job(job_id)
    assert cancelled["status"] == "cancelled"
    assert store.run(run_id)["payload"]["status"] == "cancelled"
    assert [event["kind"] for event in store.events(job_id)] == ["queued", "cancelled"]


def test_queue_limit_and_restart_mark_job_and_run_interrupted(tmp_path) -> None:
    store = LocalStore(tmp_path / "terraintrace.sqlite")
    store.save_dataset("terrain_local", "benchmark", {"cases": 1})
    for _ in range(5):
        store.create_run_job("terrain_local", 1, {"status": "queued", "cases": []}, {}, 1)
    with pytest.raises(QueueFullError):
        store.create_run_job("terrain_local", 1, {"status": "queued", "cases": []}, {}, 1)

    claimed = store.claim_job()
    assert claimed is not None
    job_id, run_id = claimed[0:2]
    assert store.run(run_id)["payload"]["status"] == "running"
    store.interrupt_running_jobs()
    assert store.job(job_id)["status"] == "interrupted"
    assert store.run(run_id)["payload"]["status"] == "interrupted"
    assert store.events(job_id)[-1]["kind"] == "interrupted"
