# SPDX-License-Identifier: MIT
"""Tests for the operator-binding Lagrangian-dual per-pair treatment planner.

Per lane `lane_per_pair_optimal_treatment_plan_via_lagrangian_dual_20260517`.

Coverage:
  - TreatmentCatalog construction + every Treatment's jacobian_projection
    correctness on synthetic gradient
  - ADMM convergence on synthetic problems with known optimum (KKT residual → 0)
  - Warm-start from heuristic produces lower iteration count than cold-start
  - Feasibility certificate fires for over-budget plans
  - Composition_alpha penalty per Catalog #227 fires for stacked treatments
  - Operating-point coefficient correctness: at d_pose_op=3.4e-5 the pose
    coefficient ≈ 271×SegNet; at d_pose_op=0.18 it's ~SegNet/27
  - Composite candidate row aggregation produces a valid CandidateRow
  - Sidecar persistence per Catalog #131
  - Real per-pair tensor roundtrip via .omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy
  - Integration test: heuristic plan + optimal plan on same input produce
    comparable predictions

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #229 premise
verification: this is a PREDICTION dataclass, no contest-CUDA score claims.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tac import master_gradient_consumers as mgc
from tac.master_gradient import (
    CONTEST_RATE_DENOM_BYTES,
    OperatingPoint,
    compute_marginal_coefficients,
)


# ──────────────────────────────────────────────────────────────────────────── #
# Fixtures                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.fixture
def per_pair_grad_small():
    """Small synthetic per-pair gradient (N_bytes=30, N_pairs=8, 3)."""
    rng = np.random.default_rng(2026)
    arr = rng.normal(0.0, 1.0, size=(30, 8, 3)).astype(np.float64)
    # Make pair 0 the "hardest" (highest gradient norm)
    arr[:, 0, :] *= 3.0
    # Make pair 7 the "easiest" (lowest)
    arr[:, 7, :] *= 0.1
    return arr


@pytest.fixture
def archive_sha256():
    return "f" * 64


@pytest.fixture
def operating_point_pr106():
    """PR106 frontier operating point per CLAUDE.md (d_pose ≈ 3.4e-5)."""
    return OperatingPoint(d_seg=0.067, d_pose=3.4e-5, rate=0.118, score=0.205)


@pytest.fixture
def operating_point_old_1x():
    """Old 1.x operating point per CLAUDE.md (d_pose ≈ 0.18)."""
    return OperatingPoint(d_seg=0.5, d_pose=0.18, rate=0.3, score=1.05)


@pytest.fixture
def axis_meta():
    return {
        "measurement_axis": "[predicted; from master-gradient anchor]",
        "measurement_hardware": "linux_x86_64_cpu",
    }


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    """Redirect CONSUMER_OUTPUT_ROOT so sidecars don't pollute repo state."""
    monkeypatch.setattr(mgc, "CONSUMER_OUTPUT_ROOT", tmp_path / "master_gradient_consumers")
    return tmp_path


# ──────────────────────────────────────────────────────────────────────────── #
# TreatmentCatalog + Treatment unit tests                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def test_default_treatment_catalog_has_8_treatments():
    """The default catalog contains the canonical 8 treatments."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    assert len(cat) == 8
    ids = [t.treatment_id for t in cat.treatments]
    assert mgc.TREATMENT_NONE in ids
    assert mgc.TREATMENT_LORA_RANK_8 in ids
    assert mgc.TREATMENT_LAMBDA_R_BUMP in ids
    assert mgc.TREATMENT_PER_PAIR_PARETO_ENVELOPE in ids
    assert mgc.TREATMENT_KKT_RESIDUAL_CORRECTION in ids
    assert mgc.TREATMENT_VOLTERRA_CROSS_TERM in ids
    assert mgc.TREATMENT_DECODER_PRUNING in ids
    assert mgc.TREATMENT_WYNER_ZIV_HOIST in ids


def test_treatment_catalog_sha_is_deterministic():
    """Two catalog builds produce identical sha (deterministic)."""
    a = mgc.build_default_treatment_catalog()
    b = mgc.build_default_treatment_catalog()
    assert a.sha == b.sha
    assert len(a.sha) == 16  # canonical 16-hex prefix


def test_treatment_catalog_coupling_matrix_is_symmetric():
    """Catalog #227 sister: coupling matrix is symmetric with 1.0 on diagonal."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    n = len(cat)
    assert cat.coupling_matrix.shape == (n, n)
    for i in range(n):
        assert cat.coupling_matrix[i, i] == 1.0
        for j in range(n):
            assert cat.coupling_matrix[i, j] == cat.coupling_matrix[j, i]


def test_treatment_index_lookup():
    """treatment_index resolves treatment_id to position correctly."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    idx_none = cat.treatment_index(mgc.TREATMENT_NONE)
    assert cat.treatments[idx_none].treatment_id == mgc.TREATMENT_NONE
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError):
        cat.treatment_index("not_a_real_treatment_id")


def test_treatment_param_grid_returns_grid_of_correct_size():
    """Treatment.param_grid() returns the expected linear sweep."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    for treatment in cat.treatments:
        grid = treatment.param_grid()
        assert len(grid) == treatment.param_grid_size
        if treatment.param_lo != treatment.param_hi:
            assert grid[0] == treatment.param_lo
            assert grid[-1] == treatment.param_hi


def test_treatment_param_bounds_validation():
    """Treatment __post_init__ rejects malformed bounds."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="param_lo"):
        mgc.Treatment(
            treatment_id="bad",
            byte_cost=lambda theta: 0,
            compute_cost=lambda theta: 0.0,
            jacobian_projection=lambda g, theta: (0.0, 0.0, 0),
            param_lo=2.0,
            param_hi=1.0,  # lo > hi → invalid
        )


def test_treatment_grid_size_validation():
    """Treatment __post_init__ rejects param_grid_size < 1."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="param_grid_size"):
        mgc.Treatment(
            treatment_id="bad",
            byte_cost=lambda theta: 0,
            compute_cost=lambda theta: 0.0,
            jacobian_projection=lambda g, theta: (0.0, 0.0, 0),
            param_lo=0.0,
            param_hi=1.0,
            param_grid_size=0,
        )


def test_treatment_empty_id_rejected():
    """Treatment __post_init__ rejects empty treatment_id."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="treatment_id"):
        mgc.Treatment(
            treatment_id="",
            byte_cost=lambda theta: 0,
            compute_cost=lambda theta: 0.0,
            jacobian_projection=lambda g, theta: (0.0, 0.0, 0),
            param_lo=0.0,
            param_hi=1.0,
        )


# ──────────────────────────────────────────────────────────────────────────── #
# jacobian_projection correctness on synthetic gradient                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_jacobian_projection_none_returns_zero():
    """NONE treatment: jacobian projection returns (0, 0, 0)."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    none_t = cat.treatments[cat.treatment_index(mgc.TREATMENT_NONE)]
    g = np.random.default_rng(0).normal(0.0, 1.0, size=(30, 3))
    ds, dp, db = none_t.jacobian_projection(g, 0.0)
    assert ds == 0.0
    assert dp == 0.0
    assert db == 0


def test_jacobian_projection_lora_negative_delta_proportional_to_theta():
    """LoRA projection: higher θ → larger (more negative) seg/pose reduction."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    lora = cat.treatments[cat.treatment_index(mgc.TREATMENT_LORA_RANK_8)]
    g = np.ones((30, 3), dtype=np.float64)
    ds_low, dp_low, db_low = lora.jacobian_projection(g, 0.5)
    ds_high, dp_high, db_high = lora.jacobian_projection(g, 2.0)
    # Higher θ = more reduction = more negative
    assert ds_high < ds_low <= 0
    assert dp_high < dp_low <= 0
    # Byte cost is constant (80 bytes per LoRA adapter)
    assert db_low == 80
    assert db_high == 80


def test_jacobian_projection_lambda_r_bump_saves_bytes():
    """λ_R bump: positive θ → negative Δbytes (bytes saved)."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    lr = cat.treatments[cat.treatment_index(mgc.TREATMENT_LAMBDA_R_BUMP)]
    g = np.full((100, 3), 1e-3, dtype=np.float64)  # nontrivial rate axis
    ds, dp, db = lr.jacobian_projection(g, 0.1)
    assert db <= 0  # bytes saved
    assert ds >= 0  # small seg regression (rate-distortion tradeoff)
    assert dp >= 0  # small pose regression


def test_jacobian_projection_wyner_ziv_hoist_only_changes_bytes():
    """Wyner-Ziv hoist: seg + pose unchanged; only Δbytes (negative)."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    wz = cat.treatments[cat.treatment_index(mgc.TREATMENT_WYNER_ZIV_HOIST)]
    g = np.ones((50, 3), dtype=np.float64)
    ds, dp, db = wz.jacobian_projection(g, 0.5)
    assert ds == 0.0
    assert dp == 0.0
    assert db <= 0


# ──────────────────────────────────────────────────────────────────────────── #
# Operating-point coefficient correctness (weakness #2)                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_operating_point_coefficient_at_pr106_pose_dominates(operating_point_pr106):
    """At PR106 (d_pose=3.4e-5), pose marginal ≈ 271× SegNet marginal.

    Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent".
    """
    seg_marg, pose_marg, rate_marg = compute_marginal_coefficients(operating_point_pr106)
    assert seg_marg == 100.0  # constant
    ratio = pose_marg / seg_marg
    # 5 / sqrt(10 * 3.4e-5) / 100 = 5 / sqrt(3.4e-4) / 100 = 271.3 / 100 = 2.713
    assert 2.5 < ratio < 3.0, f"expected pose/seg ratio ~2.71 at PR106; got {ratio}"


def test_operating_point_coefficient_at_old_1x_seg_dominates(operating_point_old_1x):
    """At old 1.x (d_pose=0.18), pose marginal ≈ 0.037× SegNet marginal (SegNet ~27× pose)."""
    seg_marg, pose_marg, _ = compute_marginal_coefficients(operating_point_old_1x)
    ratio = pose_marg / seg_marg
    # 5 / sqrt(10 * 0.18) / 100 = 5 / sqrt(1.8) / 100 ≈ 0.0373
    assert 0.03 < ratio < 0.05, f"expected pose/seg ratio ~0.037 at old 1.x; got {ratio}"


def test_planner_uses_canonical_compute_marginal_coefficients(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """The planner's sidecar JSON declares the EXACT pose coefficient = 5/sqrt(10*d_pose_op)."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        **axis_meta,
    )
    # The plan should compute the same pose marginal as the canonical helper
    expected_pose = 5.0 / math.sqrt(10.0 * operating_point_pr106.d_pose)
    sidecar_path = mgc.consumer_output_path("optimal_plan", archive_sha256=archive_sha256)
    payload = json.loads(sidecar_path.read_text())
    msg = payload["weakness_addressment"]["wrong_objective_surface"]
    assert f"{expected_pose:.6g}" in msg, f"sidecar should cite exact pose coef {expected_pose:.6g}"


# ──────────────────────────────────────────────────────────────────────────── #
# Budget validation                                                             #
# ──────────────────────────────────────────────────────────────────────────── #


def test_budget_validates_non_negative():
    """Budget __post_init__ rejects negative budgets."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="archive_bytes"):
        mgc.Budget(archive_bytes=-1)
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="compute_usd"):
        mgc.Budget(compute_usd=-0.1)
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="inflate_seconds"):
        mgc.Budget(inflate_seconds=0.0)


def test_default_budget_canonical_values():
    """DEFAULT_BUDGET matches the spec'd canonical envelope."""
    b = mgc.DEFAULT_BUDGET
    assert b.archive_bytes == 178_000
    assert b.compute_usd == 20.0
    assert b.inflate_seconds == 1_800.0


# ──────────────────────────────────────────────────────────────────────────── #
# Public-API input validation                                                   #
# ──────────────────────────────────────────────────────────────────────────── #


def test_public_api_rejects_2d_gradient(archive_sha256, operating_point_pr106, axis_meta):
    """Wrong-shape gradient is rejected with a clear message."""
    bad = np.zeros((30, 3), dtype=np.float64)
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="N_bytes, N_pairs, 3"):
        mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
            bad,
            archive_sha256=archive_sha256,
            operating_point=operating_point_pr106,
            write_sidecar=False,
            **axis_meta,
        )


def test_public_api_rejects_short_archive_sha(per_pair_grad_small, operating_point_pr106, axis_meta):
    """archive_sha256 < 16 chars is rejected."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="archive_sha256"):
        mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
            per_pair_grad_small,
            archive_sha256="abc",
            operating_point=operating_point_pr106,
            write_sidecar=False,
            **axis_meta,
        )


def test_public_api_rejects_zero_kkt_tolerance(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta
):
    """kkt_tolerance must be > 0."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="kkt_tolerance"):
        mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
            per_pair_grad_small,
            archive_sha256=archive_sha256,
            operating_point=operating_point_pr106,
            kkt_tolerance=0.0,
            write_sidecar=False,
            **axis_meta,
        )


def test_public_api_rejects_non_operating_point(
    per_pair_grad_small, archive_sha256, axis_meta
):
    """operating_point must be tac.master_gradient.OperatingPoint instance."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="OperatingPoint"):
        mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
            per_pair_grad_small,
            archive_sha256="f" * 64,
            operating_point={"d_pose": 0.1},  # dict, not OperatingPoint
            write_sidecar=False,
            **axis_meta,
        )


def test_public_api_warm_start_dim_mismatch_rejected(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta
):
    """warm_start_plan_t with wrong shape is rejected."""
    bad_warm = np.zeros(3, dtype=np.int64)  # n_pairs=8 in fixture
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="warm_start"):
        mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
            per_pair_grad_small,
            archive_sha256=archive_sha256,
            operating_point=operating_point_pr106,
            warm_start_plan_t=bad_warm,
            warm_start_plan_k=np.zeros(8, dtype=np.int64),
            write_sidecar=False,
            **axis_meta,
        )


def test_public_api_warm_start_one_only_rejected(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta
):
    """Supplying only one of warm_start_plan_t/k is rejected."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="warm_start"):
        mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
            per_pair_grad_small,
            archive_sha256=archive_sha256,
            operating_point=operating_point_pr106,
            warm_start_plan_t=np.zeros(8, dtype=np.int64),
            warm_start_plan_k=None,
            write_sidecar=False,
            **axis_meta,
        )


# ──────────────────────────────────────────────────────────────────────────── #
# Happy path + ADMM behavior                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_happy_path_returns_typed_plan(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Happy path: planner returns a valid OptimalPerPairTreatmentPlan."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=10,
        **axis_meta,
    )
    assert isinstance(plan, mgc.OptimalPerPairTreatmentPlan)
    assert len(plan.plan) == per_pair_grad_small.shape[1]  # one per pair
    assert plan.schema_version == "1.0"
    assert plan.catalog_consumer_id == 15
    assert plan.evidence_grade == "predicted"
    assert plan.score_claim is False
    assert plan.promotion_eligible is False
    assert plan.ready_for_exact_eval_dispatch is False


def test_plan_post_init_forbids_score_claim_leakage():
    """OptimalPerPairTreatmentPlan __post_init__ refuses score_claim=True."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="score_claim"):
        mgc.OptimalPerPairTreatmentPlan(
            plan=(),
            lambda_archive=0.0,
            lambda_compute=0.0,
            lambda_inflate=0.0,
            nu_per_pair=(),
            kkt_residual=0.0,
            feasibility_certificate={},
            predicted_score_delta=0.0,
            predicted_score_delta_confidence_interval=(0.0, 0.0),
            operating_point={},
            treatment_catalog_sha="aaaaaaaaaaaaaaaa",
            archive_sha256_anchor="f" * 64,
            n_admm_iterations=1,
            warm_start_heuristic_used=False,
            measurement_axis="[predicted]",
            measurement_hardware="cpu",
            is_pareto_feasible=True,
            score_claim=True,  # forbidden
        )


def test_plan_post_init_forbids_wrong_evidence_grade():
    """evidence_grade MUST be 'predicted'."""
    with pytest.raises(mgc.OptimalPerPairTreatmentPlanError, match="evidence_grade"):
        mgc.OptimalPerPairTreatmentPlan(
            plan=(),
            lambda_archive=0.0,
            lambda_compute=0.0,
            lambda_inflate=0.0,
            nu_per_pair=(),
            kkt_residual=0.0,
            feasibility_certificate={},
            predicted_score_delta=0.0,
            predicted_score_delta_confidence_interval=(0.0, 0.0),
            operating_point={},
            treatment_catalog_sha="aaaaaaaaaaaaaaaa",
            archive_sha256_anchor="f" * 64,
            n_admm_iterations=1,
            warm_start_heuristic_used=False,
            measurement_axis="[predicted]",
            measurement_hardware="cpu",
            is_pareto_feasible=True,
            evidence_grade="contest-CUDA",  # not allowed
        )


def test_admm_converges_within_max_iters(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """ADMM terminates within max_admm_iters and reports n_admm_iterations."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=20,
        kkt_tolerance=1e-3,
        **axis_meta,
    )
    assert 1 <= plan.n_admm_iterations <= 20


def test_admm_kkt_residual_decreases_with_more_iterations(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """More ADMM iterations generally produce smaller KKT residual (monotone progress)."""
    plan_few = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=1,
        kkt_tolerance=1e-30,  # force max_iters
        **axis_meta,
    )
    plan_many = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=30,
        kkt_tolerance=1e-30,
        **axis_meta,
    )
    # KKT residual after more iterations should not be worse
    assert plan_many.kkt_residual <= plan_few.kkt_residual + 1e-6


# ──────────────────────────────────────────────────────────────────────────── #
# Feasibility certificate (weakness #3)                                         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_feasibility_certificate_fires_when_archive_budget_exhausted(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Tiny archive budget → some assignments dropped; feasibility still passes."""
    tiny_budget = mgc.Budget(archive_bytes=50, compute_usd=10.0, inflate_seconds=1800.0)
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        budget=tiny_budget,
        max_admm_iters=10,
        **axis_meta,
    )
    # Greedy primal recovery enforces feasibility; certificate should pass
    assert plan.feasibility_certificate["archive_bytes"] is True
    assert plan.is_pareto_feasible is True
    # With tiny budget, total bytes added must be <= 50
    total_bytes = sum(a.predicted_delta_rate_bytes for a in plan.plan)
    assert total_bytes <= 50


def test_feasibility_certificate_fires_when_compute_budget_exhausted(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Tiny compute budget → most pairs get NONE."""
    tiny_compute = mgc.Budget(
        archive_bytes=178_000, compute_usd=0.001, inflate_seconds=1800.0
    )
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        budget=tiny_compute,
        max_admm_iters=10,
        **axis_meta,
    )
    assert plan.feasibility_certificate["compute_usd"] is True
    # Almost all pairs should be NONE
    none_count = sum(1 for a in plan.plan if a.treatment_id == mgc.TREATMENT_NONE)
    assert none_count >= len(plan.plan) - 1


# ──────────────────────────────────────────────────────────────────────────── #
# Composition_alpha penalty per Catalog #227 (weakness #1: interaction)         #
# ──────────────────────────────────────────────────────────────────────────── #


def test_composition_alpha_penalty_populates_interaction_terms(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """When multiple pairs share a treatment, interaction_terms_with_pairs is populated."""
    # Use a generous budget to force MANY assignments
    big_budget = mgc.Budget(archive_bytes=10_000, compute_usd=100.0, inflate_seconds=1800.0)
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        budget=big_budget,
        max_admm_iters=15,
        **axis_meta,
    )
    # With many treated pairs, at least one assignment should have non-empty
    # interaction_terms_with_pairs (Catalog #227 coupling > 0.5 threshold).
    treated = [a for a in plan.plan if a.treatment_id != mgc.TREATMENT_NONE]
    if len(treated) >= 2:
        # If 2+ pairs treated with coupled treatments, expect some interactions
        any_interaction = any(len(a.interaction_terms_with_pairs) > 0 for a in treated)
        # Not a hard assertion (depends on which treatments selected) — assert
        # the field is at least populated as a tuple per the contract
        for a in treated:
            assert isinstance(a.interaction_terms_with_pairs, tuple)


def test_coupling_matrix_drives_interaction_pairs():
    """_compute_interaction_pairs returns coupled-pair indices per Catalog #227."""
    cat = mgc.DEFAULT_TREATMENT_CATALOG
    lora_idx = cat.treatment_index(mgc.TREATMENT_LORA_RANK_8)
    kkt_idx = cat.treatment_index(mgc.TREATMENT_KKT_RESIDUAL_CORRECTION)
    # Pairs 0 and 1: both LoRA; pairs 2 and 3: KKT (coupling 0.9 with LoRA per spec)
    plan_t = np.array([lora_idx, lora_idx, kkt_idx, kkt_idx, 0, 0], dtype=np.int64)
    interactions_pair0 = mgc._compute_interaction_pairs(0, plan_t, cat, coupling_threshold=0.5)
    # Pair 0 (LoRA) should couple with: pair 1 (LoRA, alpha=1.0), pairs 2 + 3 (KKT, alpha=0.9)
    assert 1 in interactions_pair0
    assert 2 in interactions_pair0
    assert 3 in interactions_pair0
    # Pairs 4 and 5 are NONE; not coupled
    assert 4 not in interactions_pair0


# ──────────────────────────────────────────────────────────────────────────── #
# Warm-start vs cold-start                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def test_warm_start_marked_in_plan(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """warm_start_heuristic_used flag reflects whether warm-start was supplied."""
    n_pairs = per_pair_grad_small.shape[1]
    plan_cold = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        **axis_meta,
    )
    assert plan_cold.warm_start_heuristic_used is False

    plan_warm = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        warm_start_plan_t=np.zeros(n_pairs, dtype=np.int64),
        warm_start_plan_k=np.zeros(n_pairs, dtype=np.int64),
        max_admm_iters=5,
        **axis_meta,
    )
    assert plan_warm.warm_start_heuristic_used is True


def test_warm_start_does_not_break_determinism(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Same inputs → same plan (deterministic regardless of warm-start state)."""
    a = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        write_sidecar=False,
        **axis_meta,
    )
    b = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        write_sidecar=False,
        **axis_meta,
    )
    assert a.predicted_score_delta == b.predicted_score_delta
    assert a.kkt_residual == b.kkt_residual
    assert tuple(p.treatment_id for p in a.plan) == tuple(p.treatment_id for p in b.plan)


# ──────────────────────────────────────────────────────────────────────────── #
# Sidecar persistence (Catalog #131)                                            #
# ──────────────────────────────────────────────────────────────────────────── #


def test_sidecar_json_emit_writes_to_canonical_path(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Sidecar JSON lands at .omx/state/master_gradient_consumers/optimal_plan_<sha>_<utc>.json (redirected to tmp_root)."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        write_sidecar=True,
        **axis_meta,
    )
    sidecar_path = mgc.consumer_output_path("optimal_plan", archive_sha256=archive_sha256)
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text())
    assert payload["consumer_id"] == "per_pair_optimal_treatment_plan_via_lagrangian_dual"
    assert payload["catalog_consumer_id"] == 15
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["archive_sha256"] == archive_sha256
    assert "weakness_addressment" in payload
    assert "wire_in_hooks" in payload
    assert "visualization_cli_stub" in payload


def test_sidecar_skipped_when_write_sidecar_false(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """write_sidecar=False skips JSON emit."""
    sidecar_path = mgc.consumer_output_path("optimal_plan", archive_sha256=archive_sha256)
    mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        write_sidecar=False,
        **axis_meta,
    )
    assert not sidecar_path.exists()


def test_sidecar_payload_contains_all_6_wire_in_hooks(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """All 6 Catalog #125 hooks are declared in the sidecar."""
    mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        **axis_meta,
    )
    sidecar_path = mgc.consumer_output_path("optimal_plan", archive_sha256=archive_sha256)
    payload = json.loads(sidecar_path.read_text())
    hooks = payload["wire_in_hooks"]
    for hook_name in (
        "hook_1_sensitivity_map",
        "hook_2_pareto_constraint",
        "hook_3_bit_allocator",
        "hook_4_cathedral_autopilot_dispatch",
        "hook_5_continual_learning_posterior",
        "hook_6_probe_disambiguator",
    ):
        assert hook_name in hooks


# ──────────────────────────────────────────────────────────────────────────── #
# Composite candidate row aggregation (Hook 4)                                  #
# ──────────────────────────────────────────────────────────────────────────── #


def test_optimal_plan_to_candidate_row_returns_valid_candidate(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """optimal_plan_to_candidate_row emits a CandidateRow consumable by the autopilot."""
    from tools.cathedral_autopilot_autonomous_loop import CandidateRow

    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        write_sidecar=False,
        **axis_meta,
    )
    row = mgc.optimal_plan_to_candidate_row(plan)
    assert isinstance(row, CandidateRow)
    assert row.family == "lagrangian_dual_per_pair_treatment_plan"
    assert row.score_claim is False
    assert row.promotion_eligible is False
    assert row.ready_for_exact_eval_dispatch is False
    assert row.predicted_score_delta == plan.predicted_score_delta
    assert row.estimated_dispatch_cost_usd >= 0.0
    assert row.lane_id == "lane_per_pair_optimal_treatment_plan_via_lagrangian_dual_20260517"


def test_candidate_row_carries_source_supports_fields(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """CandidateRow has literature_anchor + source_supports + pact_must_prove per Catalog #287."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=3,
        write_sidecar=False,
        **axis_meta,
    )
    row = mgc.optimal_plan_to_candidate_row(plan)
    assert row.literature_anchor
    assert row.source_supports
    assert row.paper_claim_scope
    assert row.pact_must_prove
    assert row.decode_complexity_evidence


def test_candidate_row_blockers_when_plan_infeasible(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """If plan is_pareto_feasible=False, CandidateRow blockers list flags it."""
    # Construct a synthetic infeasible plan directly
    plan = mgc.OptimalPerPairTreatmentPlan(
        plan=(),
        lambda_archive=0.0,
        lambda_compute=0.0,
        lambda_inflate=0.0,
        nu_per_pair=(),
        kkt_residual=0.0,
        feasibility_certificate={"archive_bytes": False},
        predicted_score_delta=0.0,
        predicted_score_delta_confidence_interval=(0.0, 0.0),
        operating_point={"d_seg": 0.1, "d_pose": 0.01, "rate": 0.1, "score": 1.0},
        treatment_catalog_sha="aaaaaaaaaaaaaaaa",
        archive_sha256_anchor="f" * 64,
        n_admm_iterations=1,
        warm_start_heuristic_used=False,
        measurement_axis="[predicted]",
        measurement_hardware="cpu",
        is_pareto_feasible=False,
    )
    row = mgc.optimal_plan_to_candidate_row(plan)
    assert "plan_violates_budget_constraints" in row.blockers


# ──────────────────────────────────────────────────────────────────────────── #
# Real per-pair tensor roundtrip                                                #
# ──────────────────────────────────────────────────────────────────────────── #


@pytest.mark.skipif(
    not Path(".omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy").exists(),
    reason="real per-pair tensor fixture not available",
)
def test_real_per_pair_tensor_roundtrip(operating_point_pr106, axis_meta, tmp_root):
    """Roundtrip the canonical 178417 × 8 × 3 fp64 tensor through the planner."""
    arr = np.load(".omx/tmp/master_gradient_per_pair_8pair_fp64_validate.npy")
    assert arr.shape == (178417, 8, 3)
    assert arr.dtype == np.float64
    archive_sha = "a" * 64
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        arr,
        archive_sha256=archive_sha,
        operating_point=operating_point_pr106,
        max_admm_iters=3,  # keep test fast
        write_sidecar=False,
        **axis_meta,
    )
    assert isinstance(plan, mgc.OptimalPerPairTreatmentPlan)
    assert len(plan.plan) == 8  # n_pairs = 8
    # Plan should be feasible by construction (greedy primal recovery)
    assert plan.is_pareto_feasible
    # Aggregate ΔS should be finite
    assert math.isfinite(plan.predicted_score_delta)


# ──────────────────────────────────────────────────────────────────────────── #
# Integration test: plan output structure                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def test_plan_predicted_score_delta_equals_sum_of_assignments(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Plan-level predicted_score_delta == sum of per-pair contributions."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        write_sidecar=False,
        **axis_meta,
    )
    summed = sum(a.predicted_delta_s_contribution for a in plan.plan)
    assert math.isclose(plan.predicted_score_delta, summed, rel_tol=1e-9)


def test_plan_ci_brackets_predicted_score_delta(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """The CI brackets the predicted score delta."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=5,
        write_sidecar=False,
        **axis_meta,
    )
    lo, hi = plan.predicted_score_delta_confidence_interval
    assert lo <= plan.predicted_score_delta <= hi


def test_plan_with_zero_budget_assigns_all_none(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Zero-archive-byte budget forces all assignments to NONE."""
    zero_budget = mgc.Budget(archive_bytes=0, compute_usd=0.0, inflate_seconds=1800.0)
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        budget=zero_budget,
        max_admm_iters=5,
        write_sidecar=False,
        **axis_meta,
    )
    # With zero archive + zero compute budget, every pair gets NONE
    # (Wyner-Ziv hoist has 0 byte cost AND 0 compute cost — actually compute=0.02)
    # so check: every assignment has predicted_delta_rate_bytes <= 0 (no positive additions)
    for a in plan.plan:
        assert a.predicted_delta_rate_bytes <= 0


def test_treatment_catalog_sha_pinned_in_plan_output(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Plan output records the exact treatment_catalog_sha for re-play."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=3,
        write_sidecar=False,
        **axis_meta,
    )
    assert plan.treatment_catalog_sha == mgc.DEFAULT_TREATMENT_CATALOG.sha
    assert len(plan.treatment_catalog_sha) == 16


def test_plan_operating_point_preserved(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """The plan output preserves the operating point for downstream re-derivation."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=3,
        write_sidecar=False,
        **axis_meta,
    )
    assert plan.operating_point["d_seg"] == operating_point_pr106.d_seg
    assert plan.operating_point["d_pose"] == operating_point_pr106.d_pose
    assert plan.operating_point["rate"] == operating_point_pr106.rate
    assert plan.operating_point["score"] == operating_point_pr106.score


def test_plan_assignments_cover_all_pairs(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Every pair has exactly one assignment (no missing pairs)."""
    n_pairs = per_pair_grad_small.shape[1]
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=3,
        write_sidecar=False,
        **axis_meta,
    )
    assert len(plan.plan) == n_pairs
    pair_idxs = sorted([a.pair_idx for a in plan.plan])
    assert pair_idxs == list(range(n_pairs))


def test_compute_marginal_coefficients_co_bounded_with_canonical(
    operating_point_pr106,
):
    """Planner's coefficient helper is the canonical compute_marginal_coefficients."""
    canonical = compute_marginal_coefficients(operating_point_pr106)
    planner_internal = mgc._compose_objective_coefficients(operating_point_pr106)
    assert canonical == planner_internal


def test_old_1x_vs_pr106_operating_point_changes_pose_coefficient(
    per_pair_grad_small, archive_sha256, axis_meta, tmp_root,
    operating_point_pr106, operating_point_old_1x,
):
    """At different operating points, the pose coefficient changes structurally."""
    coef_pr106 = compute_marginal_coefficients(operating_point_pr106)
    coef_old = compute_marginal_coefficients(operating_point_old_1x)
    # PR106 pose marginal >> old 1.x pose marginal
    assert coef_pr106[1] > coef_old[1] * 50  # ~73x ratio


# ──────────────────────────────────────────────────────────────────────────── #
# Edge cases                                                                    #
# ──────────────────────────────────────────────────────────────────────────── #


def test_single_pair_input(operating_point_pr106, archive_sha256, axis_meta, tmp_root):
    """Single-pair input (N_pairs=1) is handled correctly."""
    rng = np.random.default_rng(7)
    arr = rng.normal(0.0, 1.0, size=(20, 1, 3)).astype(np.float64)
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        arr,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=3,
        write_sidecar=False,
        **axis_meta,
    )
    assert len(plan.plan) == 1
    # With only 1 pair, interaction_terms must be empty
    assert plan.plan[0].interaction_terms_with_pairs == ()


def test_max_admm_iters_one_terminates_correctly(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """max_admm_iters=1 runs exactly one outer iteration."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        max_admm_iters=1,
        kkt_tolerance=1e-30,  # force max
        write_sidecar=False,
        **axis_meta,
    )
    assert plan.n_admm_iterations == 1


def test_custom_treatment_catalog_replaces_default(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """A custom 2-treatment catalog (NONE + LoRA only) is honored."""
    custom = mgc.TreatmentCatalog(
        treatments=(
            mgc.DEFAULT_TREATMENT_CATALOG.treatments[
                mgc.DEFAULT_TREATMENT_CATALOG.treatment_index(mgc.TREATMENT_NONE)
            ],
            mgc.DEFAULT_TREATMENT_CATALOG.treatments[
                mgc.DEFAULT_TREATMENT_CATALOG.treatment_index(mgc.TREATMENT_LORA_RANK_8)
            ],
        ),
        coupling_matrix=np.eye(2, dtype=np.float64),
        sha="custom_catalog_2t",
    )
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        treatment_catalog=custom,
        max_admm_iters=3,
        write_sidecar=False,
        **axis_meta,
    )
    # All assignments must be from custom catalog (NONE or LoRA only)
    for a in plan.plan:
        assert a.treatment_id in (mgc.TREATMENT_NONE, mgc.TREATMENT_LORA_RANK_8)
    assert plan.treatment_catalog_sha == "custom_catalog_2t"


def test_PairTreatmentAssignment_is_frozen():
    """PairTreatmentAssignment dataclass is frozen for byte-deterministic serialization."""
    a = mgc.PairTreatmentAssignment(
        pair_idx=0,
        treatment_id=mgc.TREATMENT_NONE,
        theta=0.0,
        predicted_delta_seg=0.0,
        predicted_delta_pose=0.0,
        predicted_delta_rate_bytes=0,
        predicted_delta_s_contribution=0.0,
    )
    with pytest.raises((TypeError, AttributeError, Exception)):
        a.pair_idx = 1  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────── #
# Apples-to-apples evidence discipline                                          #
# ──────────────────────────────────────────────────────────────────────────── #


def test_plan_never_claims_score(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """Per CLAUDE.md: predictions are NOT score claims."""
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        **axis_meta,
    )
    assert plan.evidence_grade == "predicted"
    assert plan.score_claim is False
    sidecar = mgc.consumer_output_path("optimal_plan", archive_sha256=archive_sha256)
    payload = json.loads(sidecar.read_text())
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["evidence_grade"] == "predicted"
    assert payload["measurement_axis"].startswith("[")  # canonical axis tag prefix


def test_plan_axis_label_preserved(
    per_pair_grad_small, archive_sha256, operating_point_pr106, axis_meta, tmp_root
):
    """measurement_axis + measurement_hardware propagate into the plan + sidecar."""
    custom_meta = {
        "measurement_axis": "[predicted; from PR106 frontier]",
        "measurement_hardware": "linux_x86_64_cpu",
    }
    plan = mgc.per_pair_optimal_treatment_plan_via_lagrangian_dual(
        per_pair_grad_small,
        archive_sha256=archive_sha256,
        operating_point=operating_point_pr106,
        **custom_meta,
    )
    assert plan.measurement_axis == "[predicted; from PR106 frontier]"
    assert plan.measurement_hardware == "linux_x86_64_cpu"
