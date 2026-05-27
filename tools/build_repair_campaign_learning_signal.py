#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planner-consumable repair campaign learning signal."""

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

from tac.optimization.repair_campaign_learning_signal import (  # noqa: E402
    RepairCampaignLearningSignalError,
    build_repair_campaign_learning_signal,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-report", required=True, type=Path)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--replay-bundle", required=True, type=Path)
    parser.add_argument(
        "--signal-out",
        "--learning-signal-out",
        dest="signal_out",
        required=True,
        type=Path,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairCampaignLearningSignalError(f"{label} must be a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        score_report_path = _resolve(args.score_report)
        probe_path = _resolve(args.probe)
        replay_bundle_path = _resolve(args.replay_bundle)
        score_report = _load_json_object(score_report_path, label="score report")
        probe = _load_json_object(probe_path, label="stackability probe")
        replay_bundle = _load_json_object(
            replay_bundle_path,
            label="stackability replay bundle",
        )
        signal = build_repair_campaign_learning_signal(
            score_report_path=args.score_report,
            probe_path=args.probe,
            replay_bundle_path=args.replay_bundle,
            score_report=score_report,
            probe=probe,
            replay_bundle=replay_bundle,
            repo_root=REPO_ROOT,
        )
        signal_out = _resolve(args.signal_out)
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
        RepairCampaignLearningSignalError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair campaign learning signal failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_learning_signal_cli_result.v1",
                "score_report": str(args.score_report),
                "probe": str(args.probe),
                "replay_bundle": str(args.replay_bundle),
                "signal_out": str(args.signal_out),
                "source_artifact_count": len(signal["source_artifacts"]),
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
