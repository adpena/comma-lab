#!/usr/bin/env python3
# ruff: noqa: E402
"""Phase A2 sensitivity-weighted lossy coarsening.

This is the Track 1 Phase A2 local actuator: it takes PR101 quantized weight
symbols, converts an existing score-sensitivity artifact into per-tensor
allocator weights, then delegates K-selection to the canonical
``tac.optimization`` Lagrangian allocator.

The tool is intentionally CPU-only and fail-closed:

* no scorer import or GPU/remote dispatch path exists;
* diagnostic/stub sensitivity can produce only a local planning artifact;
* every manifest emitted here has no score, promotion, rank/kill, or exact-eval
  dispatch authority.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
prepend_paths(REPO_ROOT / "tools")

from pr101_lossy_coarsening_analytical import encode_with_per_tensor_K

from tac.codec.cost_curves import TensorBlob
from tac.optimization.beta_fisher_lossy_weights import (
    MODULE_NAME as WEIGHT_MODULE_NAME,
)
from tac.optimization.beta_fisher_lossy_weights import (
    ScoreWeightConfig,
    TensorWeightTarget,
    build_tensor_weight_rows,
    load_sensitivity_map_for_weight_export,
    load_tensor_scalar_json,
    select_weighted_k_allocations,
)
from tac.pr101_split_brotli_codec import (
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)
from tac.repo_io import repo_relative, sha256_file, write_json

TOOL_NAME = "tools/sensitivity_weighted_lossy_coarsening.py"
SCHEMA_VERSION = "phase_a2_sensitivity_weighted_lossy_coarsening.v1"
PACKET_LADDER_BUILDER = "tools/build_a2_sensitivity_weighted_pr101_packet.py"
DEFAULT_STATE_DICT = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex"
    / "pr101_decoder_state_dict.pt"
)
DEFAULT_SENSITIVITY_MAP = (
    REPO_ROOT
    / "experiments/results/sensitivity_map_pr106_20260504_claude"
    / "sensitivity_map_stub.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments/results/track1_phase_a2_sensitivity_pr101"
BASE_DISPATCH_BLOCKERS = [
    "cpu_local_allocator_proxy_only",
    "no_byte_closed_runtime_packet_built",
    "no_exact_cuda_auth_eval",
    "no_contest_cpu_auth_eval",
    "score_sensitivity_artifact_must_be_certified_before_promotion",
]
FALSE_AUTHORITY_FIELDS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def _utc_ts() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _output_manifest_path(output: Path | None) -> Path:
    if output is None:
        return DEFAULT_OUTPUT_ROOT / _utc_ts() / "manifest.json"
    if output.suffix.lower() == ".json":
        return output
    return output / "manifest.json"


def _validate_rms_targets(values: list[float]) -> list[float]:
    if not values:
        raise ValueError("at least one RMS/rel_err target is required")
    out: list[float] = []
    for value in values:
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise ValueError("RMS/rel_err targets must be finite non-negative numbers")
        out.append(float(value))
    return out


def _collect_pr101_targets_and_blobs(
    state_dict_path: Path,
) -> tuple[list[TensorWeightTarget], list[TensorBlob]]:
    if not state_dict_path.is_file():
        raise FileNotFoundError(f"state_dict not found: {state_dict_path}")
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"{state_dict_path}: expected tensor state_dict")

    targets: list[TensorWeightTarget] = []
    blobs: list[TensorBlob] = []
    for name, shape in FIXED_STATE_SCHEMA:
        if name not in state:
            raise ValueError(f"state_dict missing tensor {name!r}")
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise ValueError(f"state_dict entry {name!r} is not a tensor")
        qt = _quantize_tensor(name, tensor, n_quant=N_QUANT)
        symbols = qt.q_i8.astype(np.int32).flatten()
        targets.append(TensorWeightTarget(name=name, shape=tuple(shape), symbols=symbols))
        blobs.append(TensorBlob(name=name, raw=symbols))
    return targets, blobs


def _make_joint_encoder(blobs: list[TensorBlob]):
    def hook(selections: list[dict[str, Any]]) -> dict[str, Any]:
        selected_ks = [int(selection["K"]) for selection in selections]
        encoded = encode_with_per_tensor_K(blobs, selected_ks)
        return {
            "total_bytes": int(encoded["archive_bytes"]),
            "rel_err": float(encoded["rel_err"]),
            "selected_Ks": selected_ks,
            "payload_brotli_bytes": int(encoded["payload_brotli_bytes"]),
            "side_info_bytes": int(encoded["side_info_bytes"]),
            "archive_overhead_bytes": int(encoded["archive_overhead_bytes"]),
            "abs_err_sum": float(encoded["abs_err_sum"]),
            "abs_orig_sum": float(encoded["abs_orig_sum"]),
        }

    return hook


def _flag_blocked_manifest(
    *,
    args: argparse.Namespace,
    output_json: Path,
    reason: str,
    blockers: list[str],
    status: str = "blocked_fail_closed",
) -> dict[str, Any]:
    manifest = {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_utc": _utc_iso(),
        "phase": "A2",
        "decision": "A2",
        "substrate": args.substrate,
        "target": "cpu_local",
        "status": status,
        "reason": reason,
        "evidence_grade": "blocked",
        "evidence_semantics": "no_score_no_dispatch",
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "cpu_only": True,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "downstream_selected_Ks_can_change_charged_bits": False,
        "dispatch_blockers": sorted(set(BASE_DISPATCH_BLOCKERS + blockers)),
        "inputs": {
            "state_dict": repo_relative(args.state_dict, REPO_ROOT),
            "sensitivity_map": (
                repo_relative(args.sensitivity_map, REPO_ROOT)
                if args.sensitivity_map is not None
                else None
            ),
        },
    }
    write_json(output_json, manifest)
    return manifest


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.substrate != "pr101":
        raise ValueError("only --substrate pr101 is supported for Phase A2")
    if args.max_K < 1:
        raise ValueError("--max-K must be >= 1")
    rms_targets = _validate_rms_targets(args.rms_targets)
    k_range = list(range(1, int(args.max_K) + 1))

    targets, blobs = _collect_pr101_targets_and_blobs(args.state_dict)
    sensitivities, sensitivity_metadata, sensitivity_status = load_sensitivity_map_for_weight_export(
        args.sensitivity_map,
        allow_diagnostic=args.allow_diagnostic_sensitivity,
    )
    boundary_mass = (
        load_tensor_scalar_json(args.boundary_mass_json, value_key="boundary_mass")
        if args.boundary_mass_json is not None
        else {}
    )
    film_grain_capacity = (
        load_tensor_scalar_json(args.film_grain_capacity_json, value_key="film_grain_capacity")
        if args.film_grain_capacity_json is not None
        else {}
    )

    config = ScoreWeightConfig(
        fisher_beta=args.fisher_beta,
        boundary_alpha=args.boundary_alpha,
        film_grain_alpha=args.film_grain_alpha,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        missing_sensitivity_weight=args.missing_sensitivity_weight,
        use_variance_as_film_grain_capacity=not args.disable_variance_film_grain,
    )
    tensor_weights = build_tensor_weight_rows(
        targets,
        sensitivities,
        config=config,
        boundary_mass=boundary_mass,
        film_grain_capacity=film_grain_capacity,
    )
    weighted_weights = [float(value) for value in tensor_weights["allocator_input"]["weights"]]
    joint_encoder = _make_joint_encoder(blobs)
    weighted_allocations = select_weighted_k_allocations(
        blobs,
        weighted_weights,
        rms_targets=rms_targets,
        k_range=k_range,
        joint_encoder=joint_encoder,
    )
    uniform_allocations = select_weighted_k_allocations(
        blobs,
        [1.0] * len(blobs),
        rms_targets=rms_targets,
        k_range=k_range,
        joint_encoder=joint_encoder,
    )

    baseline_lossless = encode_with_per_tensor_K(blobs, [1] * len(blobs))
    sensitivity_blockers = list(sensitivity_status["metadata_blockers"])
    diagnostic_sensitivity = bool(sensitivity_blockers)
    blockers = list(BASE_DISPATCH_BLOCKERS)
    blockers.extend(tensor_weights["blockers"])
    if diagnostic_sensitivity:
        blockers.append("diagnostic_or_stub_sensitivity_map_not_score_authority")

    comparisons: list[dict[str, Any]] = []
    by_target_uniform = {
        float(row["rms_target"]): row for row in uniform_allocations
    }
    for row in weighted_allocations:
        target = float(row["rms_target"])
        uniform = by_target_uniform[target]
        comparisons.append(
            {
                "rms_target": target,
                "weighted_total_bytes": int(row["total_bytes"]),
                "uniform_total_bytes": int(uniform["total_bytes"]),
                "weighted_minus_uniform_bytes": int(row["total_bytes"]) - int(uniform["total_bytes"]),
                "weighted_rel_err": float(row["rel_err"]),
                "uniform_rel_err": float(uniform["rel_err"]),
            }
        )

    status = (
        "completed_local_diagnostic"
        if diagnostic_sensitivity
        else "completed_local_sensitivity_weighted_allocation"
    )
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "created_utc": _utc_iso(),
        "phase": "A2",
        "decision": "A2",
        "lane_id": "track1_phase_a2_sensitivity_quant",
        "substrate": args.substrate,
        "target": "cpu_local",
        "status": status,
        "evidence_grade": "CPU-local allocator proxy",
        "evidence_semantics": "sensitivity_weighted_K_selection_no_score_no_dispatch",
        **FALSE_AUTHORITY_FIELDS,
        "dispatch_attempted": False,
        "remote_dispatch_allowed": False,
        "cpu_only": True,
        "scorer_loaded": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "downstream_selected_Ks_can_change_charged_bits": True,
        "family_falsified": False,
        "falsification_scope": "none_local_allocator_proxy_only",
        "dispatch_blockers": sorted(set(blockers)),
        "implementation": {
            "weight_module": WEIGHT_MODULE_NAME,
            "allocator_module": "tac.optimization.lagrangian_per_tensor_allocation",
            "cost_curve_module": "tac.codec.cost_curves",
            "joint_encoder_source": "tools/pr101_lossy_coarsening_analytical.encode_with_per_tensor_K",
            "byte_closed_packet_ladder_builder": PACKET_LADDER_BUILDER,
        },
        "packet_ladder_builder": {
            "tool": PACKET_LADDER_BUILDER,
            "selected_k_schedule_field": "weighted_k_allocations[].selected_Ks",
            "status_from_this_manifest": "not_built_by_k_selection_tool",
            "closure_contract": (
                "Builder must emit archive bytes/SHA/member manifest and packet-local "
                "runtime custody before any exact-eval dispatch consideration."
            ),
            "authority": "no_score_no_dispatch_until_packet_builder_manifest_and_exact_auth_eval",
        },
        "inputs": {
            "state_dict": repo_relative(args.state_dict, REPO_ROOT),
            "state_dict_sha256": sha256_file(args.state_dict),
            "sensitivity_map": repo_relative(args.sensitivity_map, REPO_ROOT),
            "sensitivity_map_sha256": sha256_file(args.sensitivity_map),
            "boundary_mass_json": (
                repo_relative(args.boundary_mass_json, REPO_ROOT)
                if args.boundary_mass_json is not None
                else None
            ),
            "film_grain_capacity_json": (
                repo_relative(args.film_grain_capacity_json, REPO_ROOT)
                if args.film_grain_capacity_json is not None
                else None
            ),
            "rms_targets": rms_targets,
            "K_range": [k_range[0], k_range[-1]],
        },
        "sensitivity_artifact": {
            **sensitivity_status,
            "metadata": dict(sensitivity_metadata),
        },
        "baseline_lossless": {
            "archive_bytes": int(baseline_lossless["archive_bytes"]),
            "rel_err": float(baseline_lossless["rel_err"]),
            "selected_Ks": [1] * len(blobs),
        },
        "weight_config": config.to_dict(),
        "tensor_weights": tensor_weights,
        "weighted_k_allocations": weighted_allocations,
        "uniform_k_allocations": uniform_allocations,
        "weighted_vs_uniform": comparisons,
        "next_action": (
            "Replace diagnostic/stub sensitivity with a certified PR101 component-sensitivity "
            "map, then feed weighted_k_allocations[].selected_Ks into a byte-closed runtime "
            "packet builder before exact CUDA/CPU auth eval."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--substrate", default="pr101", choices=["pr101"])
    parser.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
    parser.add_argument("--sensitivity-map", type=Path, default=DEFAULT_SENSITIVITY_MAP)
    parser.add_argument("--boundary-mass-json", type=Path)
    parser.add_argument("--film-grain-capacity-json", type=Path)
    parser.add_argument("--allow-diagnostic-sensitivity", action="store_true")
    parser.add_argument("--rms-budget", type=float, default=0.05)
    parser.add_argument("--rms-targets", type=float, nargs="+")
    parser.add_argument("--max-K", type=int, default=64)
    parser.add_argument("--fisher-beta", type=float, default=1.0)
    parser.add_argument("--boundary-alpha", type=float, default=0.5)
    parser.add_argument("--film-grain-alpha", type=float, default=0.25)
    parser.add_argument("--min-weight", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=float, default=1e6)
    parser.add_argument("--missing-sensitivity-weight", type=float, default=1.0)
    parser.add_argument("--disable-variance-film-grain", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write a fail-closed preflight manifest without loading tensors.",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Document that this invocation is intentionally local CPU only.",
    )
    args = parser.parse_args(argv)
    if args.rms_targets is None:
        args.rms_targets = [args.rms_budget]
    if args.output is not None and args.output_json is not None:
        parser.error("use only one of --output or --output-json")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = _output_manifest_path(args.output_json or args.output)
    if args.dry_run:
        _flag_blocked_manifest(
            args=args,
            output_json=output_json,
            reason="dry_run_no_allocator_execution",
            blockers=["dry_run_no_tensor_or_sensitivity_validation"],
            status="dry_run_fail_closed",
        )
        print(f"manifest: {output_json}")
        return 0
    try:
        manifest = build_manifest(args)
    except Exception as exc:
        _flag_blocked_manifest(
            args=args,
            output_json=output_json,
            reason=str(exc),
            blockers=["a2_tool_input_or_validation_failed"],
        )
        print(f"blocked manifest: {output_json}")
        print(f"reason: {exc}")
        return 2

    write_json(output_json, manifest)
    preferred = manifest["weighted_k_allocations"][0]
    print(f"manifest: {output_json}")
    print(
        "selected_Ks:"
        f" target={preferred['rms_target']:.6f}"
        f" bytes={preferred['total_bytes']:,}"
        f" rel_err={preferred['rel_err']:.6f}"
    )
    if manifest["sensitivity_artifact"]["metadata_blockers"]:
        print("sensitivity: diagnostic/stub markers present; manifest is not score authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
