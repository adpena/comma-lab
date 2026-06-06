#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert HiNeRV scorer-domain bootstrap telemetry into ActionEffect JSONL.

This is the scorer-domain-bootstrap sibling of the target-region birth receipt
path already emitted by the live runner.  It does not invent a birth receipt and
does not mint score authority: only fields present in the training artifact are
copied into ``tac.action_effect.v1``.  Missing exact Pose/byte/survival endpoints
become explicit blockers on the row so launch gates and planners can fail
closed while still seeing the accepted update as a typed action.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.action_effect import append_action_effect, validate_action_effect_payload  # noqa: E402
from tac.analysis.hinerv_scorer_bootstrap_action_effect import (  # noqa: E402
    build_action_effect_from_hinerv_scorer_bootstrap_artifact,
)
from tac.repo_io import sha256_file  # noqa: E402

CONVERSION_SCHEMA = "tac.hinerv_scorer_bootstrap_action_effect_conversion.v1"


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} JSON malformed at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-artifact", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--consumer", default="nerv_long_run_launch_gate")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        artifact_path = args.training_artifact.expanduser().resolve(strict=False)
        artifact = _load_json_object(artifact_path, label="training artifact")
        effect = build_action_effect_from_hinerv_scorer_bootstrap_artifact(
            artifact,
            training_artifact_path=artifact_path,
            consumer=args.consumer,
        )
        record = append_action_effect(effect, args.output_jsonl)
        validation = validate_action_effect_payload(record)
    except (OSError, ValueError, TypeError) as exc:
        print(f"FATAL: could not convert HiNeRV scorer bootstrap to ActionEffect: {exc}", file=sys.stderr)
        return 2

    summary = {
        "schema": CONVERSION_SCHEMA,
        "training_artifact": artifact_path.as_posix(),
        "training_artifact_sha256": sha256_file(artifact_path),
        "output_jsonl": args.output_jsonl.as_posix(),
        "action_id": record["action_id"],
        "authority": record["authority"],
        "delta_score_nonrate": record["delta_score_nonrate"],
        "delta_score_total": record["delta_score_total"],
        "delta_bytes": record["delta_bytes"],
        "value_per_byte": record["value_per_byte"],
        "blockers": record["blockers"],
        "validation": validation,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True) + "\n", end="")
    return 0 if validation.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
