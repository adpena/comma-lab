# SPDX-License-Identifier: MIT
"""Tests for PR 95 Stage 3 v332_smooth MLX curriculum build.

Sister of `test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py` (Stage 2
softplus refinement) and `test_optimizer_scheduler_registry.py` (Stage
1/2/5/8 descriptor invariants). Covers the canonical PR 95 8-stage curriculum
Stage 3 extension. Stage 3 is `stage3_v332_smooth` with
`smooth_disagreement_seg_loss(tau=0.3)` and AdamW LR 1e-4 (FRESH cosine, not
continuing Stage 2's LR ladder) per the recovered public PR95 source
(`.omx/research/pr95_8stage_curriculum_forensic_20260513.md` +
`.omx/research/pr95_curriculum_recovery_20260513_codex.md`).

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the
docstring-overstatement trap)": all assertions in this module are
[macOS-MLX research-signal] non-promotable; no contest-axis score claims.

Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA": Stage 3 MLX
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


def test_pr95_stage_modules_contains_stage_3_v332_smooth() -> None:
    """Stage 3 must be registered in the canonical dispatch dict."""

    assert 3 in PR95_STAGE_MODULES
    assert PR95_STAGE_MODULES[3] == "stage3_v332_smooth"


def test_pr95_stage_default_descriptor_for_stage_3() -> None:
    """Stage 3 must resolve to canonical AdamW baseline descriptor."""

    assert 3 in PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS
    assert PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[3] == (
        "pr95_stage3_adamw_baseline_mlx"
    )
    assert pr95_default_optimizer_descriptor_id(3) == (
        "pr95_stage3_adamw_baseline_mlx"
    )


def test_pr95_stage_modules_canonical_set_is_supported() -> None:
    """Canonical supported stages after Stage 3 landing: at minimum {1, 2, 3, 5, 8}.

    Asserted as superset-of so future Stage 4/6/7 additions do not require
    mutating this Stage 3 test per Catalog #110/#113 HISTORICAL_PROVENANCE
    APPEND-ONLY discipline.
    """

    assert {1, 2, 3, 5, 8}.issubset(set(PR95_STAGE_MODULES))
    assert {1, 2, 3, 5, 8}.issubset(set(PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS))


def test_unsupported_stage_raises_value_error_with_canonical_supported_list() -> None:
    """Unsupported stage raises ValueError mentioning canonical supported set.

    Soft-error pattern uses `sorted(PR95_STAGE_MODULES)` dynamically so
    future stage 6/7 additions do not require updating hardcoded strings.
    Stage 4 was the formerly-unsupported value tested here; per the PR 95
    Stage 4 v332_qat MLX curriculum BUILD landing 2026-05-25 it is now
    supported, so this test exercises Stage 6 (next unsupported) instead.
    Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline preserved.
    """

    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        stage_smoke_config(9)
    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        pr95_default_optimizer_descriptor_id(9)


# ---------------------------------------------------------------------------
# Canonical descriptor row invariants (per Catalog #287 + #323)
# ---------------------------------------------------------------------------


def test_pr95_stage3_descriptor_in_registry() -> None:
    """Stage 3 descriptor must be registered + carry canonical contract."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage3_adamw_baseline_mlx").to_planner_candidate()

    # Optimizer config matches the recovered public PR95 Stage 3 module.
    assert row["optimizer_config"]["use_muon"] is False
    assert row["optimizer_config"]["adamw_lr"] == 1e-4
    assert row["optimizer_config"]["latent_lr_mult"] == 10.0
    # Training-config stage_loss_family declares the canonical
    # smooth_disagreement form per the recovered public PR 95 source.
    assert row["training_config"]["stage_loss_family"] == (
        "smooth_disagreement_seg_loss"
    )

    # Training config carries canonical stage routing.
    assert row["training_config"]["pr95_stage_indices"] == [3]
    assert row["training_config"]["stage_modules"] == ["stage3_v332_smooth"]
    assert row["training_config"]["backend_status"] == (
        "implemented_mlx_local_timing_proxy"
    )
    assert row["training_config"]["score_claim"] is False
    assert row["training_config"]["promotion_eligible"] is False
    assert row["training_config"]["rank_or_kill_eligible"] is False
    assert row["training_config"]["ready_for_exact_eval_dispatch"] is False
    # Stage 3 per published PR 95 curriculum: 1500 epochs canonical.
    assert row["training_config"]["stage_epochs"] == 1500
    # Stage 3 has no QAT (QAT is Stage 4).
    assert row["training_config"]["stage_uses_qat"] is False
    assert row["training_config"]["stage_uses_muon"] is False

    # Scheduler config matches stage routing.
    assert row["scheduler_config"]["stage_indices"] == [3]
    assert row["scheduler_config"]["source_pr"] == 95


def test_pr95_stage3_descriptor_passes_proxy_candidate_contract() -> None:
    """Stage 3 descriptor must satisfy proxy candidate validator."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage3_adamw_baseline_mlx").to_planner_candidate()
    assert validate_proxy_candidate(row) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected


def test_pr95_stage3_lr_is_fresh_cosine_distinct_from_stage_1_2_baseline() -> None:
    """Stage 3 LR (1e-4) is FRESH cosine, distinct from Stage 1+2 baseline (1e-3).

    Per PR 95 published 8-stage curriculum
    (`.omx/research/pr95_8stage_curriculum_forensic_20260513.md` +
    `.omx/research/pr95_curriculum_recovery_20260513_codex.md`):
    Stage 3 (v332_smooth) uses adamw_lr=1e-4 with a FRESH cosine schedule
    (1e-4 → 5e-6) — distinct from Stage 1+2's baseline LR (1e-3 per
    sister-canonical landing) AND distinct from Stage 5 (c1a_l7 at 3e-5)
    and Stage 8 (muon_finetune AdamW at 1e-5). Stage 3 is the
    intermediate bridge between Stage 2 softplus refinement and Stage 4
    QAT; its loss family (smooth_disagreement_seg_loss(tau=0.3)) is a
    sigmoid bell on negative margin, refining Stage 2's tau_softplus.
    """

    stage1 = stage_smoke_config(1)
    stage2 = stage_smoke_config(2)
    stage3 = stage_smoke_config(3)
    stage5 = stage_smoke_config(5)
    stage8 = stage_smoke_config(8)
    # Canonical sister-landed LR ladder per the v332 smooth paradigm:
    # Stage 1 + 2 share baseline LR=1e-3; Stage 3 fresh cosine 1e-4;
    # Stage 5 + 8 use lower LRs for quantization (3e-5) and finetune (1e-5).
    assert stage1.optimizer.adamw_lr == 1e-3
    assert stage2.optimizer.adamw_lr == 1e-3
    assert stage3.optimizer.adamw_lr == 1e-4
    assert stage5.optimizer.adamw_lr == 3e-5
    assert stage8.optimizer.adamw_lr == 1e-5
    # Loss-family transition: Stage 3 uses smooth_disagreement seg loss vs
    # Stage 2's tau_softplus refinement vs Stage 1's RGB MSE baseline.
    assert stage3.stage_module == "stage3_v332_smooth"
    assert stage2.stage_module == "stage2_v331_softplus"
    assert stage1.stage_module == "stage1_v328_ce"


# ---------------------------------------------------------------------------
# stage_smoke_config dispatch invariants
# ---------------------------------------------------------------------------


def test_stage_smoke_config_stage_3_dispatches_canonical_module() -> None:
    """stage_smoke_config(3) must dispatch the canonical stage3_v332_smooth."""

    cfg = stage_smoke_config(3)
    assert cfg.stage_index == 3
    assert cfg.stage_module == "stage3_v332_smooth"
    assert cfg.optimizer_descriptor_id == "pr95_stage3_adamw_baseline_mlx"
    assert cfg.optimizer.use_muon is False
    assert cfg.optimizer.adamw_lr == 1e-4
    assert cfg.optimizer_backend_status == "implemented_mlx_local_timing_proxy"


def test_stage_smoke_config_stage_3_accepts_explicit_descriptor_id() -> None:
    """stage_smoke_config(3) must accept --optimizer-descriptor-id override."""

    cfg = stage_smoke_config(
        3,
        optimizer_descriptor_id="pr95_stage3_adamw_baseline_mlx",
    )
    assert cfg.optimizer_descriptor_id == "pr95_stage3_adamw_baseline_mlx"
    assert cfg.stage_module == "stage3_v332_smooth"


# ---------------------------------------------------------------------------
# End-to-end synthetic timing smoke (Stage 3 10-step on MLX)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_stage_3_synthetic_timing_smoke_runs_end_to_end() -> None:
    """Stage 3 10-step MLX synthetic timing smoke runs + emits non-promotable contract."""

    result = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=3,
        steps=10,
        batch_size=1,
        synthetic_pairs=1,
        seed=20260525,
        base_channels=36,
        latent_dim=28,
    )

    # Canonical Stage 3 identifiers.
    assert result["stage_index"] == 3
    assert result["stage_module"] == "stage3_v332_smooth"
    assert "stage3_pr95_stage3_adamw_baseline_mlx" in result["candidate_id"]

    # Loss converges to finite scalar.
    assert isinstance(result["last_loss"], float)
    assert np.isfinite(result["last_loss"])

    # Runtime profile carries canonical training fidelity.
    rp = result["runtime_profile"]
    assert rp["stage_index"] == 3
    assert rp["stage_id"] == "stage3_v332_smooth"
    assert rp["optimizer_descriptor_id"] == "pr95_stage3_adamw_baseline_mlx"
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
# Paired forward parity: Stage 1 vs Stage 3 at random init
# ---------------------------------------------------------------------------


def test_stage_1_vs_stage_3_byte_identical_forward_at_random_init() -> None:
    """Stage 1 + Stage 3 share architecture: forward at random init is byte-identical.

    Per PR 95 published 8-stage curriculum: Stage 3 (v332_smooth) shares
    Stage 1 (v328_ce)'s architecture (HNeRVDecoder + base_ch=36,
    latent_dim=28); only the loss-family transition + fresh cosine LR
    schedule distinguish them. At step 0 (before training), seeded
    random init produces byte-identical forward output across ALL
    stages 1+2+3+5+8.

    This is the canonical sanity check that Stage 3 does NOT silently
    perturb the architecture; if this test fails, the Stage 3 trainer is
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
    bundle_stage_3 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    pred_3 = bundle_stage_3(idx)
    mx.eval(pred_3)
    arr_3 = np.asarray(pred_3)

    max_abs_diff = float(np.max(np.abs(arr_1 - arr_3)))

    # Byte-identical at random init: max_abs_diff = 0.0 exactly.
    assert max_abs_diff == 0.0, (
        f"Stage 1 vs Stage 3 forward parity FAIL at random init: "
        f"max_abs_diff={max_abs_diff}; expected 0.0 byte-identical "
        f"(architecture must be identical before the loss-family switch)"
    )

    # PASS_BAND_5E3 sanity per MLX-ARCH-5 dispatch contract.
    assert max_abs_diff < 5e-3, (
        f"Stage 1 vs Stage 3 forward parity FAIL beyond ε=5e-3 fp32 band: "
        f"max_abs_diff={max_abs_diff}"
    )

    # Sanity: configured stage modules ARE different (Stage 1 v328_ce vs
    # Stage 3 v332_smooth). The loss-family transition + LR schedule
    # transition (Stage 1 baseline 1e-3 vs Stage 3 fresh cosine 1e-4) are
    # what distinguish them.
    cfg_1 = stage_smoke_config(1)
    cfg_3 = stage_smoke_config(3)
    assert cfg_1.optimizer.adamw_lr == 1e-3
    assert cfg_3.optimizer.adamw_lr == 1e-4
    assert cfg_1.optimizer.adamw_lr != cfg_3.optimizer.adamw_lr
    assert cfg_1.stage_module != cfg_3.stage_module
    assert cfg_1.stage_module == "stage1_v328_ce"
    assert cfg_3.stage_module == "stage3_v332_smooth"
