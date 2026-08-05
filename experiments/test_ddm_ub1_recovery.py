# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.subset_selection import MODE_PREFIX, MODE_STRATIFIED

from experiments import ddm_gr1_granularity_rerace as gr1
from experiments import ddm_ub1_pose_family_923_harness as pose923


def test_gr1_prefix_selection_preserves_historical_order() -> None:
    sel, rec = gr1.build_selection_scope(8, "prefix", 20260805, n_bootstrap=100)

    assert sel.indices == tuple(range(8))
    assert rec["selection"]["pair_selection"] == MODE_PREFIX
    assert rec["selection_args"]["selection_seed"] is None
    assert rec["scorer_forwards_run"] == 0


def test_gr1_stratified_n48_selection_is_seg_matched() -> None:
    sel, rec = gr1.build_selection_scope(48, "stratified", 20260805, n_bootstrap=100)

    assert len(sel.indices) == 48
    assert rec["selection"]["pair_selection"] == MODE_STRATIFIED
    assert rec["selection"]["indices"] == list(sel.indices)
    assert rec["selection"]["population_matched"] is True
    assert rec["selection"]["governing_ratios"][0]["name"] == "seg_flip_density"
    assert rec["scope_proof"]["axis"] == "seg"


def test_pose923_plan_is_scorer_free_and_records_recovery_boundaries() -> None:
    rec = pose923.build_fire_orders()

    assert rec["schema"] == "ddm_ub1.pose_family_923_fire_orders.v1"
    assert rec["scorer_runs_by_ub1"] == 0
    assert rec["selection"]["pair_selection"] == "stratified_blocks"
    assert rec["selection"]["n"] == 120
    statuses = {row["id"]: row["status"] for row in rec["fire_orders"]}
    assert statuses["pose_carrier_arms_stratified_n120_retest"] == "READY_REBUILT_FROM_RECEIPT_NOT_RUN"
    assert statuses["pose_mladder_depthwarp_a0_stratified_n120_retest"].startswith("PARTIAL_READY")
    assert statuses["pose_l2_truedepth_stratified_n120_retest"] == "BLOCKED_DEPTH_CACHE_ABSENT"
    assert statuses["pose_stratified_texture_stratified_n120_retest"] == "BLOCKED_TEXTURE_GRID_ABSENT"
    assert rec["checkpoint"]["xi_effective_available"] is True
    assert rec["gt_cache"]["sha256_status"].startswith("skipped_large_input")


def test_pose923_checkpoint_carries_xi_effective_table() -> None:
    xi = pose923.load_xi_effective()

    assert xi.shape == (600, 6)
    assert xi.dtype.kind == "f"
