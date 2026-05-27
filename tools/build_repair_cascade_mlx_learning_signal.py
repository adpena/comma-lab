#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a repair-campaign learning signal from a cascade MLX probe result."""

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

from comma_lab.scheduler.repair_cascade_mlx_probe_queue import (  # noqa: E402
    RepairCascadeMlxProbeQueueError,
    build_repair_cascade_mlx_learning_signal,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-result", required=True, type=Path)
    parser.add_argument(
        "--learning-signal-out",
        "--signal-out",
        dest="learning_signal_out",
        required=True,
        type=Path,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result_path = _resolve(args.probe_result)
        probe_result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(probe_result, dict):
            raise RepairCascadeMlxProbeQueueError(
                "probe result must be a JSON object"
            )
        signal = build_repair_cascade_mlx_learning_signal(
            probe_result=probe_result,
            probe_result_path=args.probe_result,
            repo_root=REPO_ROOT,
        )
        signal_out = _resolve(args.learning_signal_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if signal_out.exists() and args.overwrite:
            existing_text = signal_out.read_text(encoding="utf-8")
            next_text = json_text(signal)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(signal_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                signal_out,
                signal,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCascadeMlxProbeQueueError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(
            f"FATAL: repair cascade MLX learning signal failed: {exc}",
            file=sys.stderr,
        )
        return 2
    print(
        json_text(
            {
                "schema": "repair_cascade_mlx_learning_signal_cli_result.v1",
                "probe_result": str(args.probe_result),
                "learning_signal_out": str(args.learning_signal_out),
                "typed_response_id": signal["typed_response_id"],
                "family_id": signal["family_id"],
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
