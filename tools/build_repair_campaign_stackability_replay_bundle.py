#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic replay bundle for a repair stackability probe."""

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

from tac.optimization.repair_campaign_replay_bundle import (  # noqa: E402
    RepairCampaignReplayBundleError,
    build_repair_campaign_stackability_replay_bundle,
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
    parser.add_argument("--bundle-out", required=True, type=Path)
    parser.add_argument(
        "--probe-command-json",
        required=True,
        help="JSON argv list that reruns the stackability probe.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairCampaignReplayBundleError(f"{label} must be a JSON object")
    return payload


def _load_argv(text: str) -> list[str]:
    payload = json.loads(text)
    if not isinstance(payload, list) or not payload:
        raise RepairCampaignReplayBundleError("--probe-command-json must be a non-empty list")
    return [str(item) for item in payload]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        score_report_path = _resolve(args.score_report)
        probe_path = _resolve(args.probe)
        score_report = _load_json_object(score_report_path, label="score report")
        probe = _load_json_object(probe_path, label="stackability probe")
        replay_argv = _load_argv(args.probe_command_json)
        bundle = build_repair_campaign_stackability_replay_bundle(
            score_report_path=args.score_report,
            probe_path=args.probe,
            score_report=score_report,
            probe=probe,
            replay_argv=replay_argv,
            invocation_argv=[sys.executable, __file__, *sys.argv[1:]],
            repo_root=REPO_ROOT,
        )
        bundle_out = _resolve(args.bundle_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if bundle_out.exists() and args.overwrite:
            existing_text = bundle_out.read_text(encoding="utf-8")
            next_text = json_text(bundle)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(bundle_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                bundle_out,
                bundle,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairCampaignReplayBundleError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: repair stackability replay bundle failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_stackability_replay_bundle_cli_result.v1",
                "score_report": str(args.score_report),
                "probe": str(args.probe),
                "bundle_out": str(args.bundle_out),
                "hash_manifest_sha256": bundle["hash_manifest_sha256"],
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
