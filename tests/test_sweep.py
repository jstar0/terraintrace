import hashlib
import json

import pytest

from terraintrace.sweep import parse_seed_list, seed_sweep_payload, write_sweep


def test_parse_seed_list_is_strict() -> None:
    assert parse_seed_list("17, 29,41") == (17, 29, 41)
    with pytest.raises(ValueError, match="unique"):
        parse_seed_list("17,17")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_seed_list("17,bad")


def test_sweep_is_deterministic_and_reports_variation() -> None:
    first = seed_sweep_payload((17, 29))
    second = seed_sweep_payload((17, 29))
    assert first == second
    assert len(first["runs"]) == 2
    assert first["aggregate"]["spectral_priorRmse"]["pstdev"] >= 0


def test_sweep_artifacts_have_stable_manifest(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_sweep(first, (17, 29))
    write_sweep(second, (17, 29))
    for name in ("sweep.json", "sweep.csv", "report.md", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    for entry in manifest["files"]:
        assert entry["sha256"] == hashlib.sha256((first / entry["name"]).read_bytes()).hexdigest()


def test_sweep_rejects_empty_or_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="at least one"):
        seed_sweep_payload(())
    with pytest.raises(ValueError, match="unique"):
        seed_sweep_payload((1, 1))
