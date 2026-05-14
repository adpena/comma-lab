#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build beta-Fisher score-aware weights for lossy-coarsening K allocation.

This is a CPU-only bridge from existing sensitivity/Fisher artifacts to the
existing Lagrangian/no-dead-K allocator path. It exports:

* ``allocator_input.weights`` in FIXED_STATE_SCHEMA order;
* selected per-tensor ``K`` vectors for each requested rel_err target;
* an integration note pointing at the no-dead-K builder constant that should
  consume the selected vector before a byte-closed archive is rebuilt.

No scorer is loaded, no archive is promoted, and no GPU job is launched.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from pr101_lossy_coarsening_analytical import encode_with_per_tensor_K  # noqa: E402

from tac.codec.cost_curves import TensorBlob  # noqa: E402
from tac.optimization.beta_fisher_lossy_weights import (  # noqa: E402
    MODULE_NAME,
    SCHEMA_VERSION,
    ScoreWeightConfig,
    TensorWeightTarget,
    build_tensor_weight_rows,
    load_sensitivity_map_for_weight_export,
    load_tensor_scalar_json,
    select_weighted_k_allocations,
)
from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)
from tac.repo_io import repo_relative, sha256_file, write_json  # noqa: E402

TOOL_NAME = "tools/build_beta_fisher_lossy_coarsening_weights.py"
EVIDENCE_GRADE = "[CPU-planning beta-Fisher lossy-coarsening weight export]"
DEFAULT_STATE_DICT = (
    REPO_ROOT
    / "experiments/results/pr101_codecop_sweep_20260507_codex"
    / "pr101_decoder_state_dict.pt"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "reports/raw/beta_fisher_lossy_coarsening_weights"
DISPATCH_BLOCKERS = [
    "weight_export_only_no_byte_closed_archive",
    "selected_Ks_not_yet_encoded_in_no_dead_k_runtime_packet",
    "requires_static_archive_preflight",
    "requires_exact_cuda_auth_eval_before_score_claim",
]


def _utc_ts() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _utc_iso() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _collect_pr101_targets_and_blobs(state_dict_path: Path) -> tuple[list[TensorWeightTarget], list[TensorBlob]]:
    if not state_dict_path.is_file():
        raise SystemExit(f"state_dict not found: {state_dict_path}")
    state = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    targets: list[TensorWeightTarget] = []
    blobs: list[TensorBlob] = []
    for name, shape in FIXED_STATE_SCHEMA:
        if name not in state:
            raise SystemExit(f"state_dict missing tensor {name!r}")
        qt = _quantize_tensor(name, state[name], n_quant=N_QUANT)
        raw = qt.q_i8.astype(np.int32).flatten()
        targets.append(TensorWeightTarget(name=name, shape=tuple(shape), symbols=raw))
        blobs.append(TensorBlob(name=name, raw=raw))
    return targets, blobs


def _make_joint_encoder(blobs: list[TensorBlob]):
    def hook(selections: list[dict[str, Any]]) -> dict[str, Any]:
        selected_ks = [int(selection["K"]) for selection in selections]
        encoded = encode_with_per_tensor_K(blobs, selected_ks)
        return {
            "total_bytes": int(encoded["archive_bytes"]),
            "rel_err": float(encoded["rel_err"]),
            "Ks": selected_ks,
            "payload_brotli_bytes": int(encoded["payload_brotli_bytes"]),
            "side_info_bytes": int(encoded["side_info_bytes"]),
            "archive_overhead_bytes": int(encoded["archive_overhead_bytes"]),
            "abs_err_sum": float(encoded["abs_err_sum"]),
            "abs_orig_sum": float(encoded["abs_orig_sum"]),
        }

    return hook


def build_manifest(
    *,
    state_dict_path: Path,
    sensitivity_map_path: Path,
    boundary_mass_json: Path | None,
    film_grain_capacity_json: Path | None,
    allow_diagnostic_sensitivity: bool,
    config: ScoreWeightConfig,
    rms_targets: list[float],
    max_k: int,
) -> dict[str, Any]:
    targets, blobs = _collect_pr101_targets_and_blobs(state_dict_path)
    sensitivities, sensitivity_metadata, sensitivity_status = load_sensitivity_map_for_weight_export(
        sensitivity_map_path,
        allow_diagnostic=allow_diagnostic_sensitivity,
    )
    boundary_mass = (
        load_tensor_scalar_json(boundary_mass_json, value_key="boundary_mass")
        if boundary_mass_json is not None
        else {}
    )
    film_grain_capacity = (
        load_tensor_scalar_json(film_grain_capacity_json, value_key="film_grain_capacity")
        if film_grain_capacity_json is not None
        else {}
    )

    weights_payload = build_tensor_weight_rows(
        targets,
        sensitivities,
        config=config,
        boundary_mass=boundary_mass,
        film_grain_capacity=film_grain_capacity,
    )
    allocator_weights = [float(value) for value in weights_payload["allocator_input"]["weights"]]
    k_range = list(range(1, int(max_k) + 1))
    weighted_allocations = select_weighted_k_allocations(
        blobs,
        allocator_weights,
        rms_targets=rms_targets,
        k_range=k_range,
        joint_encoder=_make_joint_encoder(blobs),
    )

    diagnostic_sensitivity = bool(sensitivity_status["metadata_blockers"])
    blockers = list(DISPATCH_BLOCKERS)
    blockers.extend(weights_payload["blockers"])
    if diagnostic_sensitivity:
        blockers.append("diagnostic_or_stub_sensitivity_map_not_score_authority")

    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "module": MODULE_NAME,
        "created_utc": _utc_iso(),
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": "cpu_allocator_weight_export_no_score_no_dispatch",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "score_affecting_payload_changed": False,
        "charged_bits_changed": False,
        "downstream_selected_Ks_can_change_charged_bits": True,
        "family_falsified": False,
        "falsification_scope": "none_weight_export_only",
        "dispatch_blockers": sorted(set(blockers)),
        "inputs": {
            "state_dict": repo_relative(state_dict_path, REPO_ROOT),
            "state_dict_sha256": sha256_file(state_dict_path),
            "sensitivity_map": repo_relative(sensitivity_map_path, REPO_ROOT),
            "sensitivity_map_sha256": sha256_file(sensitivity_map_path),
            "boundary_mass_json": (
                repo_relative(boundary_mass_json, REPO_ROOT)
                if boundary_mass_json is not None
                else None
            ),
            "film_grain_capacity_json": (
                repo_relative(film_grain_capacity_json, REPO_ROOT)
                if film_grain_capacity_json is not None
                else None
            ),
            "rms_targets": [float(value) for value in rms_targets],
            "K_range": [k_range[0], k_range[-1]],
        },
        "sensitivity_artifact": {
            **sensitivity_status,
            "metadata": dict(sensitivity_metadata),
        },
        "weight_config": config.to_dict(),
        "tensor_weights": weights_payload,
        "weighted_k_allocations": weighted_allocations,
        "integration_point": {
            "target_tool": "tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py",
            "target_symbol": "ADMM_PATH_B_STEP6_KS",
            "next_code_change": (
                "Add an optional --score-weights-json/--selected-Ks-json argument "
                "that reads weighted_k_allocations[rms_target].selected_Ks from this "
                "manifest, then rebuilds the no-dead-K archive with the selected vector."
            ),
            "preferred_rms_target": 0.0386,
            "selected_Ks_field": "weighted_k_allocations[].selected_Ks",
            "allocator_weights_field": "tensor_weights.allocator_input.weights",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dict", type=Path, default=DEFAULT_STATE_DICT)
    parser.add_argument("--sensitivity-map", type=Path, required=True)
    parser.add_argument("--boundary-mass-json", type=Path)
    parser.add_argument("--film-grain-capacity-json", type=Path)
    parser.add_argument("--allow-diagnostic-sensitivity", action="store_true")
    parser.add_argument("--fisher-beta", type=float, default=1.0)
    parser.add_argument("--boundary-alpha", type=float, default=0.5)
    parser.add_argument("--film-grain-alpha", type=float, default=0.25)
    parser.add_argument("--min-weight", type=float, default=1e-6)
    parser.add_argument("--max-weight", type=float, default=1e6)
    parser.add_argument("--missing-sensitivity-weight", type=float, default=1.0)
    parser.add_argument("--disable-variance-film-grain", action="store_true")
    parser.add_argument("--rms-targets", type=float, nargs="+", default=[0.0386])
    parser.add_argument("--max-K", type=int, default=64)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_K < 1:
        raise SystemExit("--max-K must be >= 1")
    config = ScoreWeightConfig(
        fisher_beta=args.fisher_beta,
        boundary_alpha=args.boundary_alpha,
        film_grain_alpha=args.film_grain_alpha,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        missing_sensitivity_weight=args.missing_sensitivity_weight,
        use_variance_as_film_grain_capacity=not args.disable_variance_film_grain,
    )
    manifest = build_manifest(
        state_dict_path=args.state_dict,
        sensitivity_map_path=args.sensitivity_map,
        boundary_mass_json=args.boundary_mass_json,
        film_grain_capacity_json=args.film_grain_capacity_json,
        allow_diagnostic_sensitivity=args.allow_diagnostic_sensitivity,
        config=config,
        rms_targets=list(args.rms_targets),
        max_k=args.max_K,
    )
    output_json = args.output_json
    if output_json is None:
        output_json = DEFAULT_OUTPUT_ROOT / _utc_ts() / "manifest.json"
    write_json(output_json, manifest)
    print(f"manifest: {output_json}")
    preferred = next(
        (
            item
            for item in manifest["weighted_k_allocations"]
            if abs(float(item["rms_target"]) - 0.0386) < 1e-12
        ),
        manifest["weighted_k_allocations"][0],
    )
    print(
        "selected_Ks:"
        f" target={preferred['rms_target']:.4f}"
        f" bytes={preferred['total_bytes']:,}"
        f" rel_err={preferred['rel_err']:.6f}"
    )
    if manifest["sensitivity_artifact"]["metadata_blockers"]:
        print("sensitivity: diagnostic/stub markers present; manifest is not score authority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
