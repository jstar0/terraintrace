import hashlib
import json
import subprocess
import sys

from terraintrace.benchmark import benchmark_payload, write_benchmark


def test_benchmark_payload_has_grouped_rows_and_method_summaries() -> None:
    payload = benchmark_payload(17)
    assert payload["benchmarkVersion"] == 1
    assert payload["splitCounts"] == {"train": 4, "validation": 4, "test": 4}
    assert payload["groups"]["train"] != payload["groups"]["test"]
    assert payload["methods"] == ["nearest", "bilinear", "spectral_prior"]
    assert len(payload["rows"]) == 2 * 4 * 3
    assert set(payload["summary"]["test"]) == set(payload["methods"])
    assert "rmseDeltaVsBilinear" in payload["testImprovementsVsNearest"]["spectral_prior"]


def test_benchmark_payload_is_byte_stable_as_json() -> None:
    first = json.dumps(benchmark_payload(17), indent=2, sort_keys=True)
    second = json.dumps(benchmark_payload(17), indent=2, sort_keys=True)
    assert first == second


def test_write_benchmark_emits_hash_bound_artifacts(tmp_path) -> None:
    write_benchmark(tmp_path, 17)
    expected = {"benchmark.json", "per_case.csv", "report.md", "manifest.json"}
    assert {path.name for path in tmp_path.iterdir()} == expected
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for entry in manifest["files"]:
        content = (tmp_path / entry["name"]).read_bytes()
        assert entry["bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()
    assert "## Test summary" in (tmp_path / "report.md").read_text()


def test_write_benchmark_is_byte_stable(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_benchmark(first, 29)
    write_benchmark(second, 29)
    for name in ("benchmark.json", "per_case.csv", "report.md", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_case_export_cli_writes_a_reopenable_case_archive(tmp_path) -> None:
    output = tmp_path / "case.zip"
    completed = subprocess.run(
        [sys.executable, "-m", "terraintrace.cli", "case-export", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert output.is_file()
    assert payload["case_id"]
    assert payload["bytes"] == output.stat().st_size
