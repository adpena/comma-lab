#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only grouped MLX acquisition batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.mlx_acquisition_batch import (  # noqa: E402
    MLXAcquisitionBatchError,
    build_mlx_acquisition_batch_from_selection,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-effective-spend-triage-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--set-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        batch = build_mlx_acquisition_batch_from_selection(
            _load_mapping(args.mlx_effective_spend_triage_selection),
            source_path=_repo_rel(args.mlx_effective_spend_triage_selection, args.repo_root),
            set_size=args.set_size,
            limit=args.limit,
        )
        write_json_artifact(
            args.output,
            batch,
            allow_overwrite=args.allow_overwrite,
            expected_existing_sha256=args.expected_output_sha256,
        )
    except (
        ArtifactWriteError,
        MLXAcquisitionBatchError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema": batch["schema"],
                "operation_set_count": batch["operation_set_count"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
