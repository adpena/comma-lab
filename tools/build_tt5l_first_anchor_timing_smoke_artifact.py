#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L first-anchor timing-smoke custody artifact."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.optimization.l5_staircase_v2 import (  # noqa: E402
    LANE_ID,
    TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH,
    TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID,
    TT5L_FIRST_ANCHOR_TIMING_SMOKE_SCHEMA,
    TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH,
    tt5l_first_anchor_timing_smoke_status,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative_or_original(path: Path, *, repo_root: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(repo_root))
    except ValueError:
        return str(path)


def _parse_command_argv(value: str) -> list[str]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("command argv must be a JSON array") from exc
    if (
        not isinstance(loaded, list)
        or not loaded
        or not all(isinstance(item, str) and item.strip() for item in loaded)
    ):
        raise argparse.ArgumentTypeError("command argv must be a non-empty string array")
    return loaded


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be a positive finite float") from exc
    if not parsed > 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root used for path custody validation.",
    )
    parser.add_argument(
        "--result-artifact",
        required=True,
        help="Repo-local timing-smoke result artifact to hash and bind.",
    )
    parser.add_argument("--provider", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--provider-call-id", required=True)
    parser.add_argument("--elapsed-seconds", required=True, type=_positive_float)
    rate_group = parser.add_mutually_exclusive_group(required=True)
    rate_group.add_argument("--seconds-per-epoch", type=_positive_float)
    rate_group.add_argument("--seconds-per-candidate", type=_positive_float)
    parser.add_argument(
        "--command-argv-json",
        required=True,
        type=_parse_command_argv,
        help="Exact timing-smoke command argv as a JSON string array.",
    )
    parser.add_argument(
        "--output-json",
        default=TT5L_FIRST_ANCHOR_TIMING_SMOKE_ARTIFACT_PATH,
        help="Output artifact path.",
    )
    return parser


def _payload_from_args(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    result_artifact = (repo_root / args.result_artifact).resolve()
    if not result_artifact.is_file():
        raise FileNotFoundError(f"result artifact missing: {result_artifact}")
    payload: dict[str, Any] = {
        "schema": TT5L_FIRST_ANCHOR_TIMING_SMOKE_SCHEMA,
        "lane_id": LANE_ID,
        "predicate_id": TT5L_FIRST_ANCHOR_TIMING_SMOKE_PREDICATE_ID,
        "predicate_passed": True,
        "required_axes": ["contest_cpu", "contest_cuda"],
        "provider": args.provider,
        "hardware": args.hardware,
        "provider_call_id": args.provider_call_id,
        "command_argv": args.command_argv_json,
        "elapsed_seconds": args.elapsed_seconds,
        "result_artifact_path": _repo_relative_or_original(
            result_artifact,
            repo_root=repo_root,
        ),
        "result_artifact_sha256": _sha256_file(result_artifact),
        "generated_by_tool": TT5L_FIRST_ANCHOR_TIMING_SMOKE_TOOL_PATH,
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if args.seconds_per_epoch is not None:
        payload["seconds_per_epoch"] = args.seconds_per_epoch
    if args.seconds_per_candidate is not None:
        payload["seconds_per_candidate"] = args.seconds_per_candidate
    return payload


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    try:
        payload = _payload_from_args(args, repo_root=repo_root)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"[tt5l-timing-smoke] FATAL: {exc}", file=sys.stderr)
        return 2

    status = tt5l_first_anchor_timing_smoke_status(repo_root=repo_root)
    if status["artifact_valid"] is not True:
        print(json.dumps(status, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps({"artifact_path": str(output_path), "status": status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
