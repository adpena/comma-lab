#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Diff MLX master-gradient replay bundles or sidecars."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPLAY_DIFF_SCHEMA = "mlx_master_gradient_replay_diff.v1"
COMPARISON_KEYS = (
    ("archive.sha256", ("archive", "sha256"), ("archive_sha256",)),
    ("output.npy_sha256", ("output", "npy_sha256"), ("npy_sha256",)),
    ("output.npy_shape", ("output", "npy_shape"), ("npy_shape",)),
    ("determinism.seed", ("determinism", "seed"), ("determinism", "seed")),
    ("git_head", ("git_head",), ("git_head",)),
    ("calibration.ready_for_exact_eval_dispatch", ("calibration_gate", "ready_for_exact_eval_dispatch"), ("calibration_gate", "ready_for_exact_eval_dispatch")),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _lookup(payload: Mapping[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value: Any = payload
        ok = True
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                ok = False
                break
            value = value[key]
        if ok:
            return value
    return None


def diff_replay_payloads(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for label, left_path, right_path in COMPARISON_KEYS:
        left_value = _lookup(left, left_path, right_path)
        right_value = _lookup(right, left_path, right_path)
        matched = left_value == right_value
        comparisons.append(
            {
                "field": label,
                "matched": matched,
                "left": left_value,
                "right": right_value,
            }
        )
        if not matched:
            mismatches.append(label)

    left_env = _lookup(left, ("environment", "env"))
    right_env = _lookup(right, ("environment", "env"))
    if isinstance(left_env, Mapping) and isinstance(right_env, Mapping):
        changed_env = sorted(
            key
            for key in set(left_env) | set(right_env)
            if left_env.get(key) != right_env.get(key)
        )
    else:
        changed_env = []
        if left_env != right_env:
            mismatches.append("environment.env")
    return {
        "schema": REPLAY_DIFF_SCHEMA,
        "matched": not mismatches and not changed_env,
        "mismatches": mismatches,
        "changed_environment_keys": changed_env,
        "comparisons": comparisons,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare two MLX master-gradient replay bundles or sidecars."
    )
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when replay-critical fields or captured environment differ.",
    )
    args = parser.parse_args(argv)

    try:
        diff = diff_replay_payloads(_load_json(args.left), _load_json(args.right))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[mlx-replay-diff] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(diff, indent=2, sort_keys=True))
    return 1 if args.strict and diff["matched"] is not True else 0


if __name__ == "__main__":
    raise SystemExit(main())
