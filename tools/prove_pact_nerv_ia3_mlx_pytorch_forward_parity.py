#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prove Pact-NeRV-IA3 MLX export forward parity against PyTorch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.pact_nerv_ia3_export_parity import (  # noqa: E402
    PactNervIa3ExportParityError,
    prove_pact_nerv_ia3_mlx_pytorch_forward_parity,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", required=True, type=Path)
    parser.add_argument("--pt-out", type=Path)
    parser.add_argument("--pair-indices", default="0,1,2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output-height", type=int, default=24)
    parser.add_argument("--output-width", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _pair_indices(raw: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise PactNervIa3ExportParityError("--pair-indices must be comma-separated ints") from exc
    if not values:
        raise PactNervIa3ExportParityError("--pair-indices must not be empty")
    return values


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report_out = _resolve(args.report_out)
        pt_out = _resolve(args.pt_out) if args.pt_out is not None else None
        report = prove_pact_nerv_ia3_mlx_pytorch_forward_parity(
            pair_indices=_pair_indices(args.pair_indices),
            output_pt_path=pt_out,
            seed=args.seed,
            tolerance=args.tolerance,
            output_height=args.output_height,
            output_width=args.output_width,
            overwrite_pt=args.overwrite,
        )
        expected_existing_sha256 = (
            sha256_file(report_out) if report_out.is_file() and args.overwrite else None
        )
        write = write_json_artifact(
            report_out,
            report,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        PactNervIa3ExportParityError,
        ValueError,
    ) as exc:
        print(f"FATAL: Pact-NeRV-IA3 export parity failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "pact_nerv_ia3_mlx_pytorch_forward_parity_cli_result.v1",
                "report_out": str(args.report_out),
                "pt_out": str(args.pt_out) if args.pt_out is not None else None,
                "parity_passed": report["parity_passed"],
                "max_abs_diff_255": report["max_abs_diff_255"],
                "bytes_written": write.bytes_written,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
