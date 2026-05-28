#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a deterministic byte-transform executor for one repair family."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.repair_family_byte_transform_executor import (  # noqa: E402
    RepairFamilyByteTransformExecutorError,
    build_repair_family_byte_transform_execution_report,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-materializer-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-report-out", required=True, type=Path)
    parser.add_argument("--replay-bundle-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairFamilyByteTransformExecutorError(f"{path} must be a JSON object")
    return payload


def _write_json_idempotent(
    path: Path,
    payload: dict[str, object],
    *,
    overwrite: bool,
) -> tuple[int, bool]:
    expected_existing_sha256 = None
    skipped = False
    if path.exists() and overwrite:
        existing_text = path.read_text(encoding="utf-8")
        next_text = json_text(payload)
        if existing_text == next_text:
            skipped = True
        else:
            expected_existing_sha256 = sha256_file(path)
    if skipped:
        return 0, True
    write_result = write_json_artifact(
        path,
        payload,
        allow_overwrite=overwrite,
        expected_existing_sha256=expected_existing_sha256,
    )
    return write_result.bytes_written, False


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest_path = _resolve(args.family_materializer_manifest)
        report, replay_bundle = build_repair_family_byte_transform_execution_report(
            family_materializer_manifest=_load_json(manifest_path),
            family_materializer_manifest_path=args.family_materializer_manifest,
            output_dir=args.output_dir,
            replay_argv=[
                ".venv/bin/python",
                "tools/run_repair_family_byte_transform_executor.py",
                "--family-materializer-manifest",
                str(args.family_materializer_manifest),
                "--output-dir",
                str(args.output_dir),
                "--execution-report-out",
                str(args.execution_report_out),
                "--replay-bundle-out",
                str(args.replay_bundle_out),
                "--overwrite",
            ],
            invocation_argv=sys.argv,
            repo_root=REPO_ROOT,
            allow_overwrite=bool(args.overwrite),
        )
        report_out = _resolve(args.execution_report_out)
        replay_out = _resolve(args.replay_bundle_out)
        report_bytes, report_skipped = _write_json_idempotent(
            report_out,
            report,
            overwrite=bool(args.overwrite),
        )
        replay_bytes, replay_skipped = _write_json_idempotent(
            replay_out,
            replay_bundle,
            overwrite=bool(args.overwrite),
        )
    except (
        ArtifactWriteError,
        OSError,
        RepairFamilyByteTransformExecutorError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair family byte transform failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_family_byte_transform_executor_cli_result.v1",
                "family_materializer_manifest": str(args.family_materializer_manifest),
                "execution_report_out": str(args.execution_report_out),
                "replay_bundle_out": str(args.replay_bundle_out),
                "family_id": report.get("family_id"),
                "typed_response_id": report.get("typed_response_id"),
                "byte_transform_delta_emitted": (
                    report.get("byte_transform_delta_emitted") is True
                ),
                "byte_closed_candidate_emitted": (
                    report.get("byte_closed_candidate_emitted") is True
                ),
                "component_response_replayed": (
                    report.get("component_response_replayed") is True
                ),
                "execution_report_bytes_written": report_bytes,
                "replay_bundle_bytes_written": replay_bytes,
                "skipped_identical_existing_execution_report": report_skipped,
                "skipped_identical_existing_replay_bundle": replay_skipped,
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
