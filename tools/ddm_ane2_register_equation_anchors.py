"""Append ddm_ane2's split-point anchors to ``scorer_fp16_drift_by_axis_v1``.

ane1 registered the equation with two all-fp16 anchors.  This arm measured the
same axis across a SPLIT-POINT ladder and a per-op sensitivity profile, which
tests something the two endpoint anchors could not: whether the drift is
DISTRIBUTED over the op sequence (curable by moving the fp16/fp32 boundary) or
BORN at one place (curable only by holding that place at fp32).

Every anchor here carries its own predicted value, so the residual is a real
count of how wrong the prediction was, not a fit.

Run:  .venv/bin/python tools/ddm_ane2_register_equation_anchors.py --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations import EmpiricalAnchor  # noqa: E402
from tac.canonical_equations.registry import (  # noqa: E402
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar  # noqa: E402

EQUATION_ID = "scorer_fp16_drift_by_axis_v1"
MEMO = ".omx/research/ddm_ane2_engineer_the_precision_drift_20260905.md"
STORE = Path("/Volumes/VertigoDataTier/pact/ddm_ane2_precision")


def _provenance(memo_sha: str):
    return build_provenance_for_research_sidecar(
        sidecar_path=MEMO,
        reactivation_criteria=(
            "re-measure when coremltools, the macOS build, or the frozen scorer "
            f"weights change (memo sha256 at registration: {memo_sha})"
        ),
        measurement_axis="[macOS-CPU/ANE advisory]",
        hardware_substrate="m5_max_ane_coremltools_9_0",
    )


def build_anchors(memo_sha: str) -> list[EmpiricalAnchor]:
    """The three anchors this arm's measurements actually support."""
    provenance = _provenance(memo_sha)
    ladder = json.loads((STORE / "screen/ladder_posenet_n120.json").read_text())
    sensitivity = json.loads((STORE / "screen/sensitivity_posenet_n120.json").read_text())
    mirror = json.loads((STORE / "stage2/mirror_posenet_n120.json").read_text())

    by_k = {r["k_fp32_tail_ops"]: r for r in ladder["rungs"]}
    k0 = by_k[0]["fidelity"]["self_mse_median"]
    k192 = by_k[192]["fidelity"]["self_mse_median"]
    # PREDICTED by the random-walk model pre-registered before the ladder ran:
    # error ~ sqrt(N_fp16), so MSE ~ N_fp16.  At k=192, 94 of 286 ops remain fp16.
    predicted_k192 = k0 * (94.0 / 286.0)

    groups = {r["group"]: r["fidelity"]["self_mse_median"] for r in sensitivity["rows"]}
    g0 = groups[0]
    rest = sum(v for g, v in groups.items() if g != 0)
    # PREDICTED by "the drift is distributed": each of 16 equal groups carries
    # roughly 1/16 of the total, so group 0's share is 1/16 = 0.0625.
    g0_share = g0 / (g0 + rest)

    best_mirror = min(mirror["rows"], key=lambda r: r["fidelity"]["self_mse_median"])
    mirror_mse = best_mirror["fidelity"]["self_mse_median"]

    return [
        EmpiricalAnchor(
            anchor_id="ane2_posenet_fp32_tail_split_does_not_reduce_drift_n120_20260905",
            measurement_utc="2026-09-05T15:20:00Z",
            inputs={
                "architecture_class": "posenet_fastvit_t12",
                "backend": "coreml_mixed_fp16_prefix_fp32_tail",
                "compute_ops": 286,
                "k_fp32_tail_ops": 192,
                "fp16_op_fraction": by_k[192]["fp16_op_fraction"],
                "ane_op_fraction": by_k[192]["placement"]["CPU_AND_NE"]["ane_op_fraction"],
                "reading_axis": "mean_squared_error_of_a_near_zero_residual",
                "pairs": ladder["pairs"],
                "frames": "gt_n600.npz stratified n120",
            },
            predicted_output={
                "self_mse_median": predicted_k192,
                "model": "random walk over fp16 ops: MSE proportional to N_fp16",
            },
            empirical_output={
                "self_mse_median": k192,
                "self_mse_at_k0": k0,
                "ratio_k192_over_k0": k192 / k0,
            },
            residual=abs(k192 - predicted_k192) / predicted_k192,
            source_artifact=str(STORE / "screen/ladder_posenet_n120.json"),
            measurement_method="n120_stratified_gt_frames_vs_cpu_torch_fp32_1thread",
            provenance=provenance,
        ),
        EmpiricalAnchor(
            anchor_id="ane2_posenet_drift_is_born_in_the_first_18_ops_n120_20260905",
            measurement_utc="2026-09-05T15:24:00Z",
            inputs={
                "architecture_class": "posenet_fastvit_t12",
                "backend": "coreml_one_op_group_fp16_rest_fp32",
                "groups": 16,
                "group_0_ordinals": "0:18",
                "reading_axis": "mean_squared_error_of_a_near_zero_residual",
                "pairs": sensitivity["pairs"],
            },
            predicted_output={
                "group_0_share_of_total_drift": 1.0 / 16.0,
                "model": "drift distributed evenly over the op sequence",
            },
            empirical_output={
                "group_0_share_of_total_drift": g0_share,
                "group_0_self_mse": g0,
                "sum_of_other_15_groups_self_mse": rest,
            },
            residual=abs(g0_share - 1.0 / 16.0) / (1.0 / 16.0),
            source_artifact=str(STORE / "screen/sensitivity_posenet_n120.json"),
            measurement_method="n120_stratified_gt_frames_one_group_fp16_at_a_time",
            provenance=provenance,
        ),
        EmpiricalAnchor(
            anchor_id="ane2_posenet_fp32_head_is_the_cure_n120_20260905",
            measurement_utc="2026-09-05T15:40:00Z",
            inputs={
                "architecture_class": "posenet_fastvit_t12",
                "backend": "coreml_fp32_head_fp16_tail",
                "label": best_mirror["label"],
                "fp32_ops": best_mirror["fp32_ops"],
                "fp16_op_fraction": best_mirror["fp16_op_fraction"],
                "ane_op_fraction": best_mirror["placement"]["CPU_AND_NE"]["ane_op_fraction"],
                "reading_axis": "mean_squared_error_of_a_near_zero_residual",
                "pairs": mirror["pairs"],
            },
            predicted_output={
                "self_mse_median": k0,
                "model": "the tail ladder's flatness read as 'no split helps'",
            },
            empirical_output={
                "self_mse_median": mirror_mse,
                "reduction_vs_all_fp16": k0 / mirror_mse if mirror_mse else float("inf"),
            },
            residual=abs(mirror_mse - k0) / k0,
            source_artifact=str(STORE / "stage2/mirror_posenet_n120.json"),
            measurement_method="n120_stratified_gt_frames_first_n_ops_fp32_rest_fp16",
            provenance=provenance,
        ),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memo-sha256", default="0" * 64)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)

    anchors = build_anchors(args.memo_sha256)
    for anchor in anchors:
        print(f"{anchor.anchor_id}: residual={anchor.residual:.4f}")
        print(f"  predicted={anchor.predicted_output}")
        print(f"  empirical={anchor.empirical_output}")
    if not args.apply:
        print("\n(dry run -- pass --apply to append to the registry)")
        return 0
    for anchor in anchors:
        update_equation_with_empirical_anchor(
            EQUATION_ID,
            anchor,
            agent="claude",
            subagent_id="ddm_ane2",
            notes="ddm_ane2 split-point ladder + per-op sensitivity + fp32-head mirror",
        )
        print(f"appended {anchor.anchor_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
