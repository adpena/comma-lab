#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority MLX-local probe spec for one repair cascade."""

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
    build_repair_cascade_mlx_probe_spec,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-payload", required=True, type=Path)
    parser.add_argument("--cascade-id", required=True)
    parser.add_argument("--probe-spec-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source_path = _resolve(args.source_payload)
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source_payload, dict):
            raise RepairCascadeMlxProbeQueueError("source payload must be a JSON object")
        spec = build_repair_cascade_mlx_probe_spec(
            source_payload=source_payload,
            source_payload_path=args.source_payload,
            cascade_id=args.cascade_id,
            repo_root=REPO_ROOT,
        )
        spec_out = _resolve(args.probe_spec_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if spec_out.exists() and args.overwrite:
            existing_text = spec_out.read_text(encoding="utf-8")
            next_text = json_text(spec)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(spec_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                spec_out,
                spec,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCascadeMlxProbeQueueError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair cascade MLX probe spec build failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_cascade_mlx_probe_spec_cli_result.v1",
                "source_payload": str(args.source_payload),
                "cascade_id": args.cascade_id,
                "probe_spec_out": str(args.probe_spec_out),
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
