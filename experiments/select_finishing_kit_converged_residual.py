# SPDX-License-Identifier: MIT
"""Select the Track-A finishing-kit residual that survives converged revalidation.

This is a cheap, deterministic analysis pass over the expensive frozen-scorer
artifacts:

* ``finishing_kit_convergence_revalidation_RESULT.json`` (camera-float refit,
  two slices plus cross-slice transfer)
* optional ``finishing_kit_production_path_verify_RESULT.json`` (post-round
  production path through ``driver.kit_aware_exact_eval``)

The 2026-06-13 under-power audit showed that the old mid-basin PR98+T10
``-0.058`` headline shrinks to about ``-0.003`` on the converged decoder. The
same-slice affine still helps locally, but cross-slice transfer prefers the
PR98-only residual. This script makes that conservative decision reproducible
without rerunning the scorer.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tac.torch_vehicle.distortion_finishing_kit import (
    CONVERGED_RESIDUAL_PR98_BIAS,
    CONVERGED_RESIDUAL_PR98_PROVENANCE,
)

_ADVISORY = "[contest-CPU advisory] NON-PROMOTABLE"
_DEFAULT_REVALIDATION = Path(".omx/research/finishing_kit_convergence_revalidation_RESULT.json")
_DEFAULT_PRODUCTION = Path(".omx/research/finishing_kit_production_path_verify_RESULT.json")
_IDENTITY_SCALE = ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))


def _load_optional_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _same_matrix(a: Any, b: Any) -> bool:
    return tuple(tuple(float(v) for v in row) for row in a) == tuple(
        tuple(float(v) for v in row) for row in b
    )


def _worst_negative(deltas: list[float]) -> float | None:
    negative = [float(v) for v in deltas if float(v) < 0.0]
    if len(negative) != len(deltas):
        return None
    return max(negative)


def select_residual(
    reval: dict[str, Any],
    *,
    production_verify: dict[str, Any] | None = None,
    revalidation_path: str,
    production_path: str | None,
) -> dict[str, Any]:
    s1 = reval["converged_slice1_refit"]
    s2 = reval["converged_slice2_refit"]
    transfer = reval["cross_slice_transfer_s1_to_s2"]
    verdict = reval.get("verdict", {})

    s1_pr98_bias = s1["pr98_refit"]["best_bias_frame_channel"]
    s2_pr98_bias = s2["pr98_refit"]["best_bias_frame_channel"]
    expected_bias = [list(row) for row in CONVERGED_RESIDUAL_PR98_BIAS]
    bias_consistent = (
        _same_matrix(s1_pr98_bias, s2_pr98_bias)
        and _same_matrix(s1_pr98_bias, expected_bias)
    )

    candidate_deltas: dict[str, float] = {
        "pr98_camera_float_slice1": float(s1["pr98_refit"]["pr98_delta_vs_base"]),
        "pr98_camera_float_slice2": float(s2["pr98_refit"]["pr98_delta_vs_base"]),
        "pr98_camera_float_transfer_s1_to_s2": float(
            transfer["applied_pr98"]["pr98_delta_vs_base"]
        ),
        "full_affine_camera_float_slice1": float(s1["full_kit_delta_vs_base"]),
        "full_affine_camera_float_slice2": float(s2["full_kit_delta_vs_base"]),
        "full_affine_camera_float_transfer_s1_to_s2": float(
            transfer["applied_full_kit"]["full_kit_delta_vs_base"]
        ),
    }
    if production_verify is not None:
        candidate_deltas["pr98_production_post_round_n24"] = float(
            production_verify["pr98_delta_vs_off_production"]
        )
        candidate_deltas["full_affine_production_post_round_n24"] = float(
            production_verify["full_delta_vs_off_production"]
        )

    pr98_surfaces = [
        candidate_deltas["pr98_camera_float_slice1"],
        candidate_deltas["pr98_camera_float_slice2"],
        candidate_deltas["pr98_camera_float_transfer_s1_to_s2"],
    ]
    if "pr98_production_post_round_n24" in candidate_deltas:
        pr98_surfaces.append(candidate_deltas["pr98_production_post_round_n24"])

    full_transfer_is_weaker = (
        candidate_deltas["full_affine_camera_float_transfer_s1_to_s2"]
        > candidate_deltas["pr98_camera_float_transfer_s1_to_s2"]
    )
    retained_fraction = float(verdict.get("retained_fraction_vs_mid_basin", 1.0))
    shrinks = verdict.get("verdict") == "SHRINKS" or retained_fraction < 0.25
    banked_delta = _worst_negative(pr98_surfaces)
    selected = bias_consistent and shrinks and banked_delta is not None

    selection_status = "selected_pr98_residual_only" if selected else "blocked_no_residual_selection"
    if not bias_consistent:
        selection_status = "blocked_inconsistent_pr98_bias"
    elif not shrinks:
        selection_status = "blocked_mid_basin_collapse_not_confirmed"
    elif banked_delta is None:
        selection_status = "blocked_pr98_not_negative_on_all_required_surfaces"

    return {
        "schema": "track_a_finishing_kit_converged_residual_selection.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": _ADVISORY,
        "source_revalidation": revalidation_path,
        "source_production_verify": production_path,
        "selection_status": selection_status,
        "selected_kit": {
            "name": "converged_residual_pr98_only",
            "factory": "DistortionKitConfig.from_converged_residual_pr98()",
            "default_off": True,
            "archive_bytes_when_off": 0,
            "section_bytes_when_enabled": 54,
            "scale_frame_channel": [list(row) for row in _IDENTITY_SCALE],
            "bias_frame_channel": expected_bias,
            "provenance": CONVERGED_RESIDUAL_PR98_PROVENANCE,
            "banked_advisory_distortion_delta": banked_delta,
            "promotion_blocker": (
                "Re-fit on final n=600 converged decoder and run byte-closed "
                "same-runtime CPU/CUDA exact replay before any frontier claim."
            ),
        },
        "candidate_deltas": candidate_deltas,
        "decision": {
            "mid_basin_full_delta": float(verdict.get("mid_basin_n24_full_kit_delta", 0.0)),
            "retained_fraction_vs_mid_basin": retained_fraction,
            "full_affine_rejected": bool(full_transfer_is_weaker),
            "full_affine_rejection_reason": (
                "same-slice affine helps, but slice1->slice2 transfer is weaker "
                "than PR98-only; do not bank T10 as a default assumption"
                if full_transfer_is_weaker
                else "full affine was not rejected by transfer in this artifact"
            ),
        },
        "rejected_assumptions": [
            "mid_basin_-0.058_full_kit_gain",
            "canonical_pr98_frame1_green_bias_transfer",
            "t10_affine_as_default_banked_gain",
            "s12_as_render_base_byte_savings",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revalidation-json", type=Path, default=_DEFAULT_REVALIDATION)
    p.add_argument("--production-verify-json", type=Path, default=_DEFAULT_PRODUCTION)
    p.add_argument("--no-production-verify", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    reval = json.loads(args.revalidation_json.read_text())
    prod_path = None if args.no_production_verify else args.production_verify_json
    prod = _load_optional_json(prod_path)
    result = select_residual(
        reval,
        production_verify=prod,
        revalidation_path=str(args.revalidation_json),
        production_path=str(prod_path) if prod is not None else None,
    )

    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        out = Path(f".omx/research/finishing_kit_converged_residual_selection_{stamp}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        f"[select-finishing-kit] {result['selection_status']} "
        f"banked_delta={result['selected_kit']['banked_advisory_distortion_delta']} "
        f"-> {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
