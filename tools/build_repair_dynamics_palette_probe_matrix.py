#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a repair-dynamics palette probe matrix from a work order."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.repair_dynamics_palette_probe import (  # noqa: E402
    RepairDynamicsPaletteProbeError,
    build_repair_dynamics_palette_probe_matrix,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-order", required=True, type=Path)
    parser.add_argument("--matrix-out", required=True, type=Path)
    parser.add_argument("--probe-output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps", "mlx", "gpu"), default="mlx")
    parser.add_argument("--n-pairs", type=int, default=48)
    parser.add_argument("--max-modes", type=int, default=96)
    parser.add_argument("--max-interaction-modes", type=int, default=16)
    parser.add_argument("--max-counterfactual-modes", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        work_order_path = _resolve(args.work_order)
        work_order = json.loads(work_order_path.read_text(encoding="utf-8"))
        if not isinstance(work_order, dict):
            raise RepairDynamicsPaletteProbeError("work order must be a JSON object")
        matrix = build_repair_dynamics_palette_probe_matrix(
            work_order=work_order,
            work_order_path=args.work_order,
            probe_output_dir=args.probe_output_dir,
            device=args.device,
            n_pairs=args.n_pairs,
            max_modes=args.max_modes,
            max_interaction_modes=args.max_interaction_modes,
            max_counterfactual_modes=args.max_counterfactual_modes,
        )
        matrix_out = _resolve(args.matrix_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if matrix_out.exists() and args.overwrite:
            existing_text = matrix_out.read_text(encoding="utf-8")
            next_text = json_text(matrix)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(matrix_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                matrix_out,
                matrix,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (ArtifactWriteError, OSError, RepairDynamicsPaletteProbeError, ValueError) as exc:
        print(
            f"FATAL: repair dynamics palette probe matrix failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": "repair_dynamics_palette_probe_matrix_cli_result.v1",
                "matrix_out": str(args.matrix_out),
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
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
