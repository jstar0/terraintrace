"""Command-line entry point for TerrainTrace artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .app.main import create_app
from .benchmark import write_benchmark
from .export import write_case_archive
from .fixture import create_dataset
from .methods import fit_spectral_prior
from .run import run_benchmark_case
from .sweep import parse_seed_list, write_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="terraintrace", description="Synthetic, reproducible terrain reconstruction benchmark"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="evaluate grouped validation and test cases")
    benchmark.add_argument("--output-dir", default="artifacts/benchmark")
    benchmark.add_argument("--seed", type=int, default=17)
    sweep = subparsers.add_parser("sweep", help="repeat the benchmark across explicit seeds")
    sweep.add_argument("--output-dir", default="artifacts/sweep")
    sweep.add_argument("--seeds", default="17,29,41")
    case_export = subparsers.add_parser("case-export", help="export one real synthetic test-case archive")
    case_export.add_argument("--output", default="artifacts/case.zip")
    case_export.add_argument("--seed", type=int, default=17)
    case_export.add_argument("--case-index", type=int, default=0)
    serve = subparsers.add_parser("serve", help="run the local TerrainTrace API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=5206)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "benchmark":
        payload = write_benchmark(args.output_dir, args.seed)
        print(
            json.dumps(
                {
                    "output_dir": args.output_dir,
                    "seed": payload["seed"],
                    "test_methods": payload["methods"],
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "sweep":
        payload = write_sweep(args.output_dir, parse_seed_list(args.seeds))
        print(
            json.dumps(
                {"output_dir": args.output_dir, "runs": len(payload["runs"]), "seeds": payload["seeds"]},
                sort_keys=True,
            )
        )
        return 0
    if args.command == "case-export":
        dataset = create_dataset(args.seed)
        if not 0 <= args.case_index < len(dataset.test):
            parser.error("--case-index must select a test case")
        case = dataset.test[args.case_index]
        proof = write_case_archive(Path(args.output), case, run_benchmark_case(case, fit_spectral_prior(dataset.train)))
        print(json.dumps({"output": args.output, "case_id": case.id, **proof}, sort_keys=True))
        return 0
    if args.command == "serve":
        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
