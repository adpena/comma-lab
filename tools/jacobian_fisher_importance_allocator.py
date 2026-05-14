#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the generic Jacobian/Fisher importance allocator on JSON inputs.

The tool is intentionally thin: it reads byte/error candidate curves and an
importance manifest, calls
``tac.optimization.jacobian_fisher_importance_allocator``, and writes a
planning-only manifest.  It does not load the scorer, build archives, launch
GPU work, or emit score/rank/promotion/kill evidence.  CPU/MPS/proxy inputs
remain allocator priors only until a byte-closed archive returns exact CUDA auth
eval.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.jacobian_fisher_importance_allocator import (  # noqa: E402
    ImportanceConfig,
    build_importance_allocation_manifest,
)
from tac.repo_io import read_json, repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/jacobian_fisher_importance_allocator.py"


def _payload_section(payload: Any, keys: tuple[str, ...]) -> Any | None:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _curves_from_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        for key in ("candidate_curves", "curves", "per_tensor_curves"):
            if key in payload:
                return payload[key]
    return payload


def _importance_inputs(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise SystemExit("importance JSON must be an object")
    per_weight = _payload_section(
        payload,
        ("per_weight_importance", "per_weight", "weights_by_tensor"),
    )
    per_tensor = _payload_section(
        payload,
        ("per_tensor_importance", "per_tensor", "tensor_importance", "importance"),
    )
    if per_weight is None and per_tensor is None:
        reserved = {
            "schema",
            "format",
            "metadata",
            "tool",
            "boundary_mass",
            "texture_capacity",
            "film_grain_capacity",
        }
        per_tensor = {key: value for key, value in payload.items() if key not in reserved}
    return {
        "per_tensor_importance": per_tensor,
        "per_weight_importance": per_weight,
        "boundary_mass": _payload_section(payload, ("boundary_mass",)),
        "texture_capacity": _payload_section(
            payload,
            ("texture_capacity", "film_grain_capacity", "capacity"),
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--curves-json", type=Path, required=True)
    parser.add_argument("--importance-json", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target-distortion", type=float)
    group.add_argument("--byte-budget", type=int)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--fisher-beta", type=float, default=1.0)
    parser.add_argument("--boundary-alpha", type=float, default=0.5)
    parser.add_argument("--texture-capacity-alpha", type=float, default=0.25)
    parser.add_argument("--min-weight", type=float, default=1e-9)
    parser.add_argument("--max-weight", type=float, default=1e9)
    parser.add_argument("--target-mean", type=float, default=1.0)
    parser.add_argument(
        "--per-weight-reducer",
        choices=("mean", "sum", "max", "rms"),
        default="mean",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    curves_payload = read_json(args.curves_json)
    importance_payload = read_json(args.importance_json)
    importance_inputs = _importance_inputs(importance_payload)
    config = ImportanceConfig(
        fisher_beta=args.fisher_beta,
        boundary_alpha=args.boundary_alpha,
        texture_capacity_alpha=args.texture_capacity_alpha,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        target_mean=args.target_mean,
        per_weight_reducer=args.per_weight_reducer,
    )
    manifest = build_importance_allocation_manifest(
        _curves_from_payload(curves_payload),
        per_tensor_importance=importance_inputs["per_tensor_importance"],
        per_weight_importance=importance_inputs["per_weight_importance"],
        boundary_mass=importance_inputs["boundary_mass"],
        texture_capacity=importance_inputs["texture_capacity"],
        target_distortion=args.target_distortion,
        byte_budget=args.byte_budget,
        config=config,
        producer_tool=TOOL_NAME,
        extra_inputs={
            "curves_json": repo_relative(args.curves_json, REPO_ROOT),
            "curves_json_sha256": sha256_file(args.curves_json),
            "importance_json": repo_relative(args.importance_json, REPO_ROOT),
            "importance_json_sha256": sha256_file(args.importance_json),
        },
    )
    write_json(args.output_json, manifest)
    allocation = manifest["allocation"]
    print(f"manifest: {args.output_json}")
    print(
        "allocation:"
        f" objective={allocation['objective']}"
        f" bytes={int(allocation['total_bytes']):,}"
        f" weighted_rms_error={float(allocation['weighted_rms_error']):.6f}"
    )
    print(
        "status: planning-only; CPU/MPS/proxy evidence is not score/rank/"
        "promotion/kill evidence"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
