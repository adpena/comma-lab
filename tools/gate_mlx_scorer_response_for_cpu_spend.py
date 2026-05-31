#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Gate expensive local CPU scoring from a non-authoritative MLX response."""

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

from tac.repo_io import sha256_file, write_json  # noqa: E402

SCHEMA = "mlx_response_cpu_spend_gate.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-score", required=True, type=float)
    parser.add_argument(
        "--max-score-delta",
        type=float,
        default=0.0,
        help="Allow CPU gate when mlx_score - baseline_score <= this value.",
    )
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    response = _read_object(args.mlx_response)
    blockers: list[str] = []
    if response.get("schema_version") != "mlx_scorer_response.v1":
        blockers.append("mlx_response_schema_mismatch")
    for field, expected in FALSE_AUTHORITY.items():
        if response.get(field) is not expected:
            blockers.append(f"mlx_response_{field}_not_false")
    score = _float_or_none(response.get("canonical_score"))
    if score is None:
        blockers.append("mlx_response_canonical_score_missing")
        score = float("inf")
    samples = response.get("n_samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < int(args.min_samples):
        blockers.append("mlx_response_insufficient_samples")
    delta = float(score) - float(args.baseline_score)
    if delta > float(args.max_score_delta):
        blockers.append("mlx_response_not_within_cpu_spend_band")
    cpu_gate_allowed = not blockers
    payload = {
        "schema": SCHEMA,
        "mlx_response": {
            "path": str(args.mlx_response),
            "sha256": sha256_file(args.mlx_response),
        },
        "baseline_score": float(args.baseline_score),
        "max_score_delta": float(args.max_score_delta),
        "mlx_score": float(score),
        "mlx_score_delta_vs_baseline": float(delta),
        "n_samples": samples,
        "cpu_gate_allowed": cpu_gate_allowed,
        "blockers": blockers,
        "allowed_next_step": (
            "local_cpu_component_spot_check" if cpu_gate_allowed else None
        ),
        "mlx_is_acquisition_only": True,
        "requires_local_cpu_before_exact_auth": True,
        **FALSE_AUTHORITY,
    }
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite gate output: {args.output}")
    write_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "cpu_gate_allowed": cpu_gate_allowed}, sort_keys=True))
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected JSON object")
    return payload


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
