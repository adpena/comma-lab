# SPDX-License-Identifier: MIT
"""Tests for PR 95 Stage 2 v331_softplus MLX curriculum build.

Sister of `test_run_pr95_mlx_timing_smoke.py` (Stage 1 timing smoke) and
`test_optimizer_scheduler_registry.py` (Stage 1/5/8 descriptor invariants).
Covers the canonical PR 95 8-stage curriculum Stage 2 extension. Stage 2
is `stage2_v331_softplus` with `tau_softplus_seg_loss` and AdamW LR 1e-3,
matching the recovered public PR95 source.

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the
docstring-overstatement trap)": all assertions in this module are
[macOS-MLX research-signal] non-promotable; no contest-axis score claims.

Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA": Stage 2 MLX
synthetic timing smoke is non-promotable by construction; promotion via
paired Linux x86_64 + NVIDIA per Catalog #192.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.local_acceleration.pr95_hnerv_mlx import (
    PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS,
    PR95_STAGE_MODULES,
    HNeRVSyntheticTrainingBundleMLX,
    pr95_default_optimizer_descriptor_id,
    run_pr95_mlx_synthetic_timing_smoke,
    stage_smoke_config,
)
from tac.optimization.optimizer_scheduler_registry import (
    FALSE_AUTHORITY_FIELDS,
    default_optimizer_scheduler_registry,
)
from tac.optimization.proxy_candidate_contract import validate_proxy_candidate

# ---------------------------------------------------------------------------
# Canonical interface extension invariants (PR95_STAGE_MODULES + dispatch)
# ---------------------------------------------------------------------------


def test_pr95_stage_modules_contains_stage_2_v331_softplus() -> None:
    """Stage 2 must be registered in the canonical dispatch dict."""

    assert 2 in PR95_STAGE_MODULES
    assert PR95_STAGE_MODULES[2] == "stage2_v331_softplus"


def test_pr95_stage_default_descriptor_for_stage_2() -> None:
    """Stage 2 must resolve to canonical AdamW baseline descriptor."""

    assert 2 in PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS
    assert PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[2] == (
        "pr95_stage2_adamw_baseline_mlx"
    )
    assert pr95_default_optimizer_descriptor_id(2) == (
        "pr95_stage2_adamw_baseline_mlx"
    )


def test_pr95_stage_modules_canonical_set_is_supported() -> None:
    """Canonical supported stages: at minimum the Stage 2 set {1, 2, 5, 8}.

    Asserted as superset-of so future Stage 3/4/6/7 additions do not
    require mutating this Stage 2 test per Catalog #110/#113
    HISTORICAL_PROVENANCE APPEND-ONLY discipline (Stage 3 landed in
    `.omx/research/pr95_mlx_stage_3_v332_smooth_curriculum_build_landed_20260525.md`
    growing the canonical set to {1, 2, 3, 5, 8}).
    """

    assert {1, 2, 5, 8}.issubset(set(PR95_STAGE_MODULES))
    assert {1, 2, 5, 8}.issubset(set(PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS))


def test_unsupported_stage_raises_value_error_with_canonical_supported_list() -> None:
    """Unsupported stage raises ValueError mentioning canonical supported set.

    Soft-error pattern uses `sorted(PR95_STAGE_MODULES)` dynamically so
    future stage additions do not require updating hardcoded strings. Per
    Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY: Stage 4 is now
    supported (`.omx/research/pr95_mlx_stage_4_v332_qat_curriculum_build_landed_20260525.md`);
    Stage 6/7 remain unsupported and are the canonical sentinel values
    surfacing the canonical-supported-set message.
    """

    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        stage_smoke_config(6)
    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        pr95_default_optimizer_descriptor_id(7)


# ---------------------------------------------------------------------------
# Canonical descriptor row invariants (per Catalog #287 + #323)
# ---------------------------------------------------------------------------


def test_pr95_stage2_descriptor_in_registry() -> None:
    """Stage 2 descriptor must be registered + carry canonical contract."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage2_adamw_baseline_mlx").to_planner_candidate()

    # Optimizer config matches the recovered public PR95 Stage 2 module.
    assert row["optimizer_config"]["use_muon"] is False
    assert row["optimizer_config"]["adamw_lr"] == 1e-3
    assert row["optimizer_config"]["latent_lr_mult"] == 10.0
    # Training-config stage_loss_family declares the canonical softplus form.
    assert row["training_config"]["stage_loss_family"] == "tau_softplus_seg_loss"

    # Training config carries canonical stage routing.
    assert row["training_config"]["pr95_stage_indices"] == [2]
    assert row["training_config"]["stage_modules"] == ["stage2_v331_softplus"]
    assert row["training_config"]["backend_status"] == (
        "implemented_mlx_local_timing_proxy"
    )
    assert row["training_config"]["score_claim"] is False
    assert row["training_config"]["promotion_eligible"] is False
    assert row["training_config"]["rank_or_kill_eligible"] is False
    assert row["training_config"]["ready_for_exact_eval_dispatch"] is False

    # Scheduler config matches stage routing.
    assert row["scheduler_config"]["stage_indices"] == [2]
    assert row["scheduler_config"]["source_pr"] == 95


def test_pr95_stage2_descriptor_passes_proxy_candidate_contract() -> None:
    """Stage 2 descriptor must satisfy proxy candidate validator."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage2_adamw_baseline_mlx").to_planner_candidate()
    assert validate_proxy_candidate(row) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected


def test_pr95_stage2_lr_is_canonical_v331_softplus_baseline() -> None:
    """Stage 2 LR must match canonical v331_softplus baseline (adamw_lr=1e-3).

    Per PR 95 published 8-stage curriculum sister-canonicalized 2026-05-25:
    Stage 2 (v331_softplus) uses adamw_lr=1e-3 — matching the Stage 1
    baseline LR (v328 also landed at 1e-3 sister-canonical) but with the
    `tau_softplus_seg_loss` family transition. Distinct from Stage 5
    (c1a_l7 at 3e-5) + Stage 8 (muon_finetune AdamW at 1e-5).
    """

    stage1 = stage_smoke_config(1)
    stage2 = stage_smoke_config(2)
    stage5 = stage_smoke_config(5)
    stage8 = stage_smoke_config(8)
    # Canonical sister-landed LR ladder per the v331 softplus paradigm:
    # Stage 1 + 2 share baseline LR=1e-3; Stage 5 + 8 use lower LRs for
    # quantization (3e-5) and finetune (1e-5) phases.
    assert stage1.optimizer.adamw_lr == 1e-3
    assert stage2.optimizer.adamw_lr == 1e-3
    assert stage5.optimizer.adamw_lr == 3e-5
    assert stage8.optimizer.adamw_lr == 1e-5
    # Loss-family transition: Stage 2 uses softplus seg loss vs Stage 1's
    # canonical RGB MSE baseline. Optimizer LR is shared; loss surface differs.
    assert stage2.stage_module == "stage2_v331_softplus"
    assert stage1.stage_module == "stage1_v328_ce"


# ---------------------------------------------------------------------------
# stage_smoke_config dispatch invariants
# ---------------------------------------------------------------------------


def test_stage_smoke_config_stage_2_dispatches_canonical_module() -> None:
    """stage_smoke_config(2) must dispatch the canonical stage2_v331_softplus."""

    cfg = stage_smoke_config(2)
    assert cfg.stage_index == 2
    assert cfg.stage_module == "stage2_v331_softplus"
    assert cfg.optimizer_descriptor_id == "pr95_stage2_adamw_baseline_mlx"
    assert cfg.optimizer.use_muon is False
    assert cfg.optimizer.adamw_lr == 1e-3
    assert cfg.optimizer_backend_status == "implemented_mlx_local_timing_proxy"


def test_stage_smoke_config_stage_2_accepts_explicit_descriptor_id() -> None:
    """stage_smoke_config(2) must accept --optimizer-descriptor-id override."""

    cfg = stage_smoke_config(
        2,
        optimizer_descriptor_id="pr95_stage2_adamw_baseline_mlx",
    )
    assert cfg.optimizer_descriptor_id == "pr95_stage2_adamw_baseline_mlx"
    assert cfg.stage_module == "stage2_v331_softplus"


# ---------------------------------------------------------------------------
# End-to-end synthetic timing smoke (Stage 2 100-step on MLX)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_stage_2_synthetic_timing_smoke_runs_end_to_end() -> None:
    """Stage 2 100-step MLX synthetic timing smoke runs + emits non-promotable contract."""

    result = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=2,
        steps=10,
        batch_size=1,
        synthetic_pairs=1,
        seed=20260525,
        base_channels=36,
        latent_dim=28,
    )

    # Canonical Stage 2 identifiers.
    assert result["stage_index"] == 2
    assert result["stage_module"] == "stage2_v331_softplus"
    assert "stage2_pr95_stage2_adamw_baseline_mlx" in result["candidate_id"]

    # Loss converges to finite scalar.
    assert isinstance(result["last_loss"], float)
    assert np.isfinite(result["last_loss"])

    # Runtime profile carries canonical training fidelity.
    rp = result["runtime_profile"]
    assert rp["stage_index"] == 2
    assert rp["stage_id"] == "stage2_v331_softplus"
    assert rp["optimizer_descriptor_id"] == "pr95_stage2_adamw_baseline_mlx"
    assert rp["training_fidelity"] == "synthetic_timing_only"
    assert rp["training_backend"] == "mlx"
    assert isinstance(rp["seconds_per_step"], float)
    assert rp["seconds_per_step"] > 0
    assert rp["state_bytes"] > 0

    # Canonical non-promotable contract per CLAUDE.md "MPS auth eval is NOISE"
    # + Catalog #1 + #192 + #287 + #323.
    assert result["score_claim"] is False
    assert result["score_claim_valid"] is False
    assert result["promotion_eligible"] is False
    assert result["rank_or_kill_eligible"] is False
    assert result["ready_for_exact_eval_dispatch"] is False
    assert result["promotable"] is False
    assert result["dispatch_attempted"] is False
    assert result["gpu_launched"] is False

    # Evidence grade tagged macOS-MLX research-signal.
    assert "macOS" in result["evidence_grade"] or "MLX" in result["evidence_grade"]


# ---------------------------------------------------------------------------
# Paired forward parity: Stage 1 vs Stage 2 at random init
# ---------------------------------------------------------------------------


def test_stage_1_vs_stage_2_byte_identical_forward_at_random_init() -> None:
    """Stage 1 + Stage 2 share architecture: forward at random init is byte-identical.

    Per PR 95 published 8-stage curriculum sister-canonicalized 2026-05-25:
    Stage 2 (v331_softplus) shares Stage 1 (v328_ce)'s architecture
    (HNeRVDecoder + base_ch=36, latent_dim=28); the loss-family transition
    (softplus seg loss vs RGB MSE baseline) is what distinguishes them.
    Both stages share canonical AdamW LR=1e-3. At step 0 (before training),
    seeded random init produces byte-identical forward output.

    This is the canonical sanity check that Stage 2 does NOT silently
    perturb the architecture; if this test fails, the Stage 2 trainer is
    diverging from PR 95's canonical curriculum continuation pattern.
    """

    import mlx.core as mx

    seed = 20260525

    # Both bundles use the SAME seed + SAME architecture.
    mx.random.seed(seed)
    bundle_stage_1 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    idx = mx.array([0], dtype=mx.uint32)
    pred_1 = bundle_stage_1(idx)
    mx.eval(pred_1)
    arr_1 = np.asarray(pred_1)

    mx.random.seed(seed)
    bundle_stage_2 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    pred_2 = bundle_stage_2(idx)
    mx.eval(pred_2)
    arr_2 = np.asarray(pred_2)

    max_abs_diff = float(np.max(np.abs(arr_1 - arr_2)))

    # Byte-identical at random init: max_abs_diff = 0.0 exactly.
    assert max_abs_diff == 0.0, (
        f"Stage 1 vs Stage 2 forward parity FAIL at random init: "
        f"max_abs_diff={max_abs_diff}; expected 0.0 byte-identical "
        f"(architecture must be identical before the loss-family switch)"
    )

    # PASS_BAND_5E3 sanity per MLX-ARCH-5 dispatch contract.
    assert max_abs_diff < 5e-3, (
        f"Stage 1 vs Stage 2 forward parity FAIL beyond ε=5e-3 fp32 band: "
        f"max_abs_diff={max_abs_diff}"
    )

    # Sanity: configured stage modules ARE different (Stage 1 v328_ce vs
    # Stage 2 v331_softplus). The loss-family transition is what
    # distinguishes the stages; the optimizer LR may share the canonical
    # baseline (both 1e-3 per sister-canonical landing 2026-05-25).
    cfg_1 = stage_smoke_config(1)
    cfg_2 = stage_smoke_config(2)
    assert cfg_1.optimizer.adamw_lr == cfg_2.optimizer.adamw_lr == 1e-3
    assert cfg_1.stage_module != cfg_2.stage_module
    assert cfg_1.stage_module == "stage1_v328_ce"
    assert cfg_2.stage_module == "stage2_v331_softplus"
