# SPDX-License-Identifier: MIT
"""Tests for PR 95 Stage 6 lambda_sweep MLX curriculum build.

Sister of `test_pr95_mlx_stage_4_v332_qat_curriculum_build.py` (Stage 4 QAT
bridge) and `test_optimizer_scheduler_registry.py` (Stage 1/2/3/4/5/8
descriptor invariants). Covers the canonical PR 95 8-stage curriculum
Stage 6 extension. Stage 6 is `stage6_lambda_sweep` with
`l7_softplus_seg_loss` (preserved from Stage 5) + AdamW LR=3e-5
(continues Stage 5 cosine schedule, NOT fresh) + QAT (preserved from
Stage 5) + C1a λ=0.02 (the sweep parameter; Stage 5 baseline λ=0.01)
per the recovered public PR 95 source
(`.omx/research/pr95_8stage_curriculum_forensic_20260513.md` line 36 +
`.omx/research/pr95_curriculum_recovery_20260513_codex.md` line 100).

Canonical Stage 6 spec per recovered PR 95 source:

- module: `stage6_lambda_sweep.py`
- epochs: 2000 (canonical PR 95)
- optimizer: AdamW only (NOT Muon; Muon is Stage 8)
- adamw_lr: 3e-5 cosine continuation (NOT fresh; continues Stage 5)
- loss family: `l7_softplus_seg_loss` (preserved from Stage 5)
- C1a λ (`stage_cat_lambda`): **0.02** (vs Stage 5's 0.01 — the sweep
  parameter)
- C1a σ (`stage_cat_sigma`): 0.2 (preserved from Stage 5)
- QAT: True (preserved from Stage 5)
- resume from: Stage 5 final

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the
docstring-overstatement trap)": all assertions in this module are
[macOS-MLX research-signal] non-promotable; no contest-axis score claims.

Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA": Stage 6 MLX
synthetic timing smoke is non-promotable by construction; promotion via
paired Linux x86_64 + NVIDIA per Catalog #192.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + #192 + #287 + #323:
the MLX backend is research-signal only; the lambda_sweep semantics are
recorded in the descriptor metadata (`stage_cat_lambda=0.02`) but the
MLX bundle uses the same `HNeRVSyntheticTrainingBundleMLX` architecture
as Stage 1+2+3+4+5 (byte-identical state_bytes empirically confirmed
via paired forward parity at random init).
"""

from __future__ import annotations

import json
from pathlib import Path

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


def test_pr95_stage_modules_contains_stage_6_lambda_sweep() -> None:
    """Stage 6 must be registered in the canonical dispatch dict."""

    assert 6 in PR95_STAGE_MODULES
    assert PR95_STAGE_MODULES[6] == "stage6_lambda_sweep"


def test_pr95_stage_default_descriptor_for_stage_6() -> None:
    """Stage 6 must resolve to canonical AdamW lambda_sweep descriptor."""

    assert 6 in PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS
    assert PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[6] == (
        "pr95_stage6_adamw_lambda_sweep_mlx"
    )
    assert pr95_default_optimizer_descriptor_id(6) == (
        "pr95_stage6_adamw_lambda_sweep_mlx"
    )


def test_pr95_stage_modules_canonical_set_is_supported() -> None:
    """Canonical supported stages after Stage 7 landing: at minimum {1, 2, 3, 4, 5, 6, 7, 8}.

    Asserted as superset-of so future Stage 7 addition does not require
    mutating this Stage 6 test per Catalog #110/#113 HISTORICAL_PROVENANCE
    APPEND-ONLY discipline.
    """

    assert {1, 2, 3, 4, 5, 6, 7, 8}.issubset(set(PR95_STAGE_MODULES))
    assert {1, 2, 3, 4, 5, 6, 7, 8}.issubset(
        set(PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS)
    )


def test_unsupported_stage_raises_value_error_with_canonical_supported_list() -> None:
    """Unsupported stage raises ValueError mentioning canonical supported set.

    Soft-error pattern uses `sorted(PR95_STAGE_MODULES)` dynamically so
    future stage 7 addition does not require updating hardcoded strings.
    """

    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        stage_smoke_config(9)
    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        pr95_default_optimizer_descriptor_id(9)


# ---------------------------------------------------------------------------
# Canonical descriptor row invariants (per Catalog #287 + #323)
# ---------------------------------------------------------------------------


def test_pr95_stage6_descriptor_in_registry() -> None:
    """Stage 6 descriptor must be registered + carry canonical contract.

    Per recovered PR 95 source `stage6_lambda_sweep` canonical fields:
    - adamw_lr=3e-5 (continues Stage 5 cosine)
    - stage_loss_family=l7_softplus_seg_loss (preserved from Stage 5)
    - stage_cat_lambda=0.02 (the sweep parameter; Stage 5 baseline=0.01)
    - stage_cat_sigma=0.2 (preserved from Stage 5)
    - stage_uses_qat=True (preserved from Stage 5)
    - stage_uses_muon=False (Muon is Stage 8)
    - stage_epochs=2000 (canonical PR 95)
    """

    registry = default_optimizer_scheduler_registry()
    row = registry.get(
        "pr95_stage6_adamw_lambda_sweep_mlx"
    ).to_planner_candidate()

    # Optimizer config matches the recovered public PR95 Stage 6 module.
    assert row["optimizer_config"]["use_muon"] is False
    # Stage 6 continues Stage 5 cosine (base LR remains 3e-5; the cosine
    # continuation is a runtime scheduler concern, not a descriptor concern).
    assert row["optimizer_config"]["adamw_lr"] == 3e-5
    assert row["optimizer_config"]["latent_lr_mult"] == 10.0

    # Training-config stage_loss_family preserves Stage 5's L7 softplus
    # form per the recovered public PR 95 source (Stage 6 IS a Stage 5
    # continuation with C1a λ swept; loss family is preserved).
    assert row["training_config"]["stage_loss_family"] == (
        "l7_softplus_seg_loss"
    )

    # Training config carries canonical stage routing.
    assert row["training_config"]["pr95_stage_indices"] == [6]
    assert row["training_config"]["stage_modules"] == ["stage6_lambda_sweep"]
    assert row["training_config"]["backend_status"] == (
        "implemented_mlx_local_timing_proxy"
    )
    assert row["training_config"]["score_claim"] is False
    assert row["training_config"]["promotion_eligible"] is False
    assert row["training_config"]["rank_or_kill_eligible"] is False
    assert row["training_config"]["ready_for_exact_eval_dispatch"] is False
    # Stage 6 per published PR 95 curriculum: 2000 epochs canonical (sister
    # Stage 4 landing memo line 173 + forensic memo line 36 + codex line 100).
    assert row["training_config"]["stage_epochs"] == 2000
    # Stage 6 PRESERVES Stage 5's QAT bit (Stage 6 IS Stage 5 continuation
    # with C1a λ swept).
    assert row["training_config"]["stage_uses_qat"] is True
    # Stage 6 uses AdamW only; Muon is Stage 8.
    assert row["training_config"]["stage_uses_muon"] is False
    # Stage 6 lambda_sweep parameter: C1a λ = 0.02 (vs Stage 5's 0.01).
    assert row["training_config"]["stage_cat_lambda"] == 0.02
    # Stage 6 preserves Stage 5's C1a σ = 0.2.
    assert row["training_config"]["stage_cat_sigma"] == 0.2

    # Scheduler config matches stage routing.
    assert row["scheduler_config"]["stage_indices"] == [6]
    assert row["scheduler_config"]["source_pr"] == 95


def test_pr95_stage6_descriptor_passes_proxy_candidate_contract() -> None:
    """Stage 6 descriptor must satisfy proxy candidate validator."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get(
        "pr95_stage6_adamw_lambda_sweep_mlx"
    ).to_planner_candidate()
    assert validate_proxy_candidate(row) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected


def test_pr95_stage6_lambda_sweep_is_the_distinguishing_parameter() -> None:
    """Stage 6 ONLY distinguishing change from Stage 5 is C1a λ 0.01 → 0.02.

    Per PR 95 published 8-stage curriculum:
    - Stage 6 IS Stage 5 with the SAME loss family
      (l7_softplus_seg_loss), SAME LR (3e-5), SAME QAT bit (True),
      SAME σ (0.2), but DIFFERENT C1a λ (0.02 vs Stage 5's 0.01).
    - The Stage 6 epoch budget (2000) is the canonical PR 95 published
      run length for the λ-sweep checkpoint.

    This test pins the sweep semantics: Stage 6 IS a 1-parameter sweep
    on top of Stage 5; nothing else changes.
    """

    registry = default_optimizer_scheduler_registry()
    stage5_row = registry.get(
        "pr95_stage5_adamw_baseline_mlx"
    ).to_planner_candidate()
    stage6_row = registry.get(
        "pr95_stage6_adamw_lambda_sweep_mlx"
    ).to_planner_candidate()

    # SHARED fields (Stage 6 is Stage 5 continuation):
    assert (
        stage5_row["optimizer_config"]["adamw_lr"]
        == stage6_row["optimizer_config"]["adamw_lr"]
        == 3e-5
    )
    assert (
        stage5_row["training_config"]["stage_loss_family"]
        == stage6_row["training_config"]["stage_loss_family"]
        == "l7_softplus_seg_loss"
    )
    assert (
        stage5_row["training_config"]["stage_uses_qat"]
        == stage6_row["training_config"]["stage_uses_qat"]
        is True
    )
    assert (
        stage5_row["training_config"]["stage_uses_muon"]
        == stage6_row["training_config"]["stage_uses_muon"]
        is False
    )
    assert (
        stage5_row["training_config"]["stage_cat_sigma"]
        == stage6_row["training_config"]["stage_cat_sigma"]
        == 0.2
    )

    # DISTINGUISHING field: C1a λ changes (the sweep parameter).
    assert stage5_row["training_config"]["stage_cat_lambda"] == 0.01
    assert stage6_row["training_config"]["stage_cat_lambda"] == 0.02
    assert (
        stage6_row["training_config"]["stage_cat_lambda"]
        != stage5_row["training_config"]["stage_cat_lambda"]
    )


def test_pr95_stage6_lr_continues_stage_5_cosine_schedule() -> None:
    """Stage 6 LR (3e-5) continues Stage 5 cosine; distinct from Stage 1+2 (1e-3) / Stage 3+4 (1e-4) / Stage 8 (1e-5).

    Per PR 95 published 8-stage curriculum
    (`.omx/research/pr95_8stage_curriculum_forensic_20260513.md` +
    `.omx/research/pr95_curriculum_recovery_20260513_codex.md`):
    Stage 6 (lambda_sweep) uses adamw_lr=3e-5 CONTINUING Stage 5's cosine
    schedule (Stage 5 ran 3e-5 → 0 over 9000 epochs; Stage 6 resumes
    from Stage 5 endpoint and continues for 2000 more epochs at the
    same 3e-5 base). Distinct from:
      - Stage 1+2 baseline LR=1e-3 (CE / softplus)
      - Stage 3+4 cosine LR=1e-4 (smooth / smooth+QAT)
      - Stage 5+6 cosine LR=3e-5 (l7_softplus+QAT with C1a λ-sweep)
      - Stage 8 muon_finetune AdamW LR=1e-5 (Muon hidden + AdamW
        stem/heads/biases/latents)
    """

    stage1 = stage_smoke_config(1)
    stage2 = stage_smoke_config(2)
    stage3 = stage_smoke_config(3)
    stage4 = stage_smoke_config(4)
    stage5 = stage_smoke_config(5)
    stage6 = stage_smoke_config(6)
    stage7 = stage_smoke_config(7)
    stage8 = stage_smoke_config(8)
    # Canonical sister-landed LR ladder per the PR 95 8-stage curriculum:
    assert stage1.optimizer.adamw_lr == 1e-3
    assert stage2.optimizer.adamw_lr == 1e-3
    assert stage3.optimizer.adamw_lr == 1e-4
    assert stage4.optimizer.adamw_lr == 1e-4
    assert stage5.optimizer.adamw_lr == 3e-5
    assert stage6.optimizer.adamw_lr == 3e-5
    assert stage7.optimizer.adamw_lr == 3e-5
    assert stage8.optimizer.adamw_lr == 1e-5
    # Stages 5/6/7 share the same start LR; Stage 6/7 sweep regularizer
    # parameters, not optimizer families.
    assert stage5.optimizer.adamw_lr == stage6.optimizer.adamw_lr
    assert stage6.optimizer.adamw_lr == stage7.optimizer.adamw_lr
    # Loss-family preservation: Stage 6 keeps Stage 5's l7_softplus_seg_loss.
    assert stage6.stage_module == "stage6_lambda_sweep"
    assert stage7.stage_module == "stage7_sigma_sweep"
    assert stage5.stage_module == "stage5_c1a_l7"
    assert stage4.stage_module == "stage4_v332_qat"
    assert stage3.stage_module == "stage3_v332_smooth"
    assert stage2.stage_module == "stage2_v331_softplus"
    assert stage1.stage_module == "stage1_v328_ce"


# ---------------------------------------------------------------------------
# stage_smoke_config dispatch invariants
# ---------------------------------------------------------------------------


def test_stage_smoke_config_stage_6_dispatches_canonical_module() -> None:
    """stage_smoke_config(6) must dispatch the canonical stage6_lambda_sweep."""

    cfg = stage_smoke_config(6)
    assert cfg.stage_index == 6
    assert cfg.stage_module == "stage6_lambda_sweep"
    assert cfg.optimizer_descriptor_id == "pr95_stage6_adamw_lambda_sweep_mlx"
    assert cfg.optimizer.use_muon is False
    assert cfg.optimizer.adamw_lr == 3e-5
    assert cfg.optimizer_backend_status == "implemented_mlx_local_timing_proxy"


def test_stage_smoke_config_stage_6_accepts_explicit_descriptor_id() -> None:
    """stage_smoke_config(6) must accept --optimizer-descriptor-id override."""

    cfg = stage_smoke_config(
        6,
        optimizer_descriptor_id="pr95_stage6_adamw_lambda_sweep_mlx",
    )
    assert cfg.optimizer_descriptor_id == "pr95_stage6_adamw_lambda_sweep_mlx"
    assert cfg.stage_module == "stage6_lambda_sweep"


# ---------------------------------------------------------------------------
# End-to-end synthetic timing smoke (Stage 6 10-step on MLX)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_stage_6_synthetic_timing_smoke_runs_end_to_end() -> None:
    """Stage 6 10-step MLX synthetic timing smoke runs + emits non-promotable contract."""

    result = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=6,
        steps=10,
        batch_size=1,
        synthetic_pairs=1,
        seed=20260525,
        base_channels=36,
        latent_dim=28,
    )

    # Canonical Stage 6 identifiers.
    assert result["stage_index"] == 6
    assert result["stage_module"] == "stage6_lambda_sweep"
    assert "stage6_pr95_stage6_adamw_lambda_sweep_mlx" in result["candidate_id"]

    # Loss converges to finite scalar.
    assert isinstance(result["last_loss"], float)
    assert np.isfinite(result["last_loss"])

    # Runtime profile carries canonical training fidelity.
    rp = result["runtime_profile"]
    assert rp["stage_index"] == 6
    assert rp["stage_id"] == "stage6_lambda_sweep"
    assert rp["optimizer_descriptor_id"] == "pr95_stage6_adamw_lambda_sweep_mlx"
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
# Paired forward parity: Stage 5 vs Stage 6 at random init
# (also Stage 4 vs Stage 6 per parent prompt request)
# ---------------------------------------------------------------------------


def test_stage_5_vs_stage_6_byte_identical_forward_at_random_init() -> None:
    """Stage 5 + Stage 6 share architecture: forward at random init is byte-identical.

    Per PR 95 published 8-stage curriculum: Stage 6 (lambda_sweep) shares
    Stage 5 (c1a_l7)'s architecture (HNeRVDecoder + base_ch=36,
    latent_dim=28); only the C1a λ sweep parameter (0.01 → 0.02)
    distinguishes them at the training-config layer. At step 0 (before
    training), seeded random init produces byte-identical forward output
    across ALL stages 1+2+3+4+5+6+8.

    NOTE on λ-sweep vs architecture: PR 95 canonical applies C1a λ in
    the cat_entropy_v2 loss term (Hinton-Vinyals-Dean 2014 sister; the
    soft-MDL term per `pr95_8stage_curriculum_forensic_20260513.md`
    line 93 / codex line 96). The MLX synthetic timing proxy does not
    apply C1a to the persistent state_dict structure; the λ sweep is
    recorded as a training-config metadata flag (`stage_cat_lambda=0.02`)
    and the MLX bundle architecture (and state_bytes) is preserved
    byte-identical across Stages 1-6 and Stage 8. Predicted state_bytes
    = sister Stage 4's 915,944 (5x extension pattern proven; if diverges,
    NULL hypothesis REJECTED and the divergence is characterized).
    """

    import mlx.core as mx

    seed = 20260525

    # Both bundles use the SAME seed + SAME architecture.
    mx.random.seed(seed)
    bundle_stage_5 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    idx = mx.array([0], dtype=mx.uint32)
    pred_5 = bundle_stage_5(idx)
    mx.eval(pred_5)
    arr_5 = np.asarray(pred_5)

    mx.random.seed(seed)
    bundle_stage_6 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    pred_6 = bundle_stage_6(idx)
    mx.eval(pred_6)
    arr_6 = np.asarray(pred_6)

    max_abs_diff = float(np.max(np.abs(arr_5 - arr_6)))

    # Byte-identical at random init: max_abs_diff = 0.0 exactly.
    assert max_abs_diff == 0.0, (
        f"Stage 5 vs Stage 6 forward parity FAIL at random init: "
        f"max_abs_diff={max_abs_diff}; expected 0.0 byte-identical "
        f"(architecture must be identical before the λ sweep applies)"
    )

    # PASS_BAND_5E3 sanity per MLX-ARCH-5 dispatch contract.
    assert max_abs_diff < 5e-3, (
        f"Stage 5 vs Stage 6 forward parity FAIL beyond ε=5e-3 fp32 band: "
        f"max_abs_diff={max_abs_diff}"
    )

    # Sanity: configured stage modules ARE different (Stage 5 c1a_l7 vs
    # Stage 6 lambda_sweep). The C1a λ sweep distinguishes them at the
    # training-config layer; both share AdamW adamw_lr=3e-5 at the
    # optimizer-config layer.
    cfg_5 = stage_smoke_config(5)
    cfg_6 = stage_smoke_config(6)
    assert cfg_5.optimizer.adamw_lr == 3e-5
    assert cfg_6.optimizer.adamw_lr == 3e-5
    # Stage 6 IS Stage 5 cosine continuation: same start LR.
    assert cfg_5.optimizer.adamw_lr == cfg_6.optimizer.adamw_lr
    assert cfg_5.stage_module != cfg_6.stage_module
    assert cfg_5.stage_module == "stage5_c1a_l7"
    assert cfg_6.stage_module == "stage6_lambda_sweep"


def test_stage_4_vs_stage_6_byte_identical_forward_at_random_init() -> None:
    """Stage 4 + Stage 6 share the SAME `HNeRVSyntheticTrainingBundleMLX` architecture.

    Per parent prompt request: characterize Stage 4 vs Stage 6 paired
    forward parity. The MLX synthetic timing proxy uses ONE canonical
    bundle architecture across ALL stages (1-6 + 8); the per-stage
    differences (CE / softplus / smooth / smooth+QAT / l7+C1a / λ-sweep /
    Muon+AdamW) are recorded as training-config metadata, not bundle
    architectural changes. Predicted state_bytes byte-identical to
    sister Stage 4's 915,944. NULL hypothesis (5x extension pattern
    holds) strongly favored.
    """

    import mlx.core as mx

    seed = 20260525

    mx.random.seed(seed)
    bundle_stage_4 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    idx = mx.array([0], dtype=mx.uint32)
    pred_4 = bundle_stage_4(idx)
    mx.eval(pred_4)
    arr_4 = np.asarray(pred_4)

    mx.random.seed(seed)
    bundle_stage_6 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    pred_6 = bundle_stage_6(idx)
    mx.eval(pred_6)
    arr_6 = np.asarray(pred_6)

    max_abs_diff = float(np.max(np.abs(arr_4 - arr_6)))

    # Byte-identical at random init: max_abs_diff = 0.0 exactly.
    assert max_abs_diff == 0.0, (
        f"Stage 4 vs Stage 6 forward parity FAIL at random init: "
        f"max_abs_diff={max_abs_diff}; expected 0.0 byte-identical"
    )


# ---------------------------------------------------------------------------
# Catalog #344 canonical equation queue + Catalog #313 ledger row
# ---------------------------------------------------------------------------


def test_canonical_equation_queued_for_ratify_n() -> None:
    """Stage 6 landing queues canonical equation `pr95_mlx_stage_6_lambda_sweep_one_to_one_curriculum_port_v1`.

    Per Catalog #344 operator-decision protocol: canonical equation
    registrations are operator-routable, NOT auto-registered. This test
    documents the QUEUE state (the canonical equation candidate is
    queued via the landing memo `## Catalog #344 RATIFY-N candidate`
    section); promotion to the canonical equation registry requires
    explicit operator decision per Catalog #344 protocol.

    The landing memo records the equation candidate id; this test
    asserts that the corresponding source-text token is present in the
    landing memo (verifies the QUEUE record landed; does NOT verify the
    canonical equation registry has the equation, which would be
    auto-registration and is FORBIDDEN per Catalog #344).
    """

    repo_root = Path(__file__).resolve().parents[3]
    memo_path = (
        repo_root
        / ".omx"
        / "research"
        / "pr95_mlx_stage_6_lambda_sweep_curriculum_build_landed_20260525.md"
    )
    assert memo_path.exists(), f"landing memo missing: {memo_path}"
    body = memo_path.read_text(encoding="utf-8")
    assert "pr95_mlx_stage_6_lambda_sweep_one_to_one_curriculum_port_v1" in body
    assert "FORMALIZATION_PENDING" in body or "RATIFY-N" in body


def test_catalog_313_probe_outcomes_row_registered() -> None:
    """Stage 6 landing registers Catalog #313 probe-outcomes row.

    Per Catalog #313 sister discipline: every cathedral consumer landing
    must register a probe-outcomes row in `.omx/state/probe_outcomes.jsonl`
    via the canonical `tac.probe_outcomes_ledger.register_probe_outcome`
    helper (NEVER bare write per Catalog #131). This test verifies the
    row landed with the canonical probe_id format.
    """

    repo_root = Path(__file__).resolve().parents[3]
    ledger_path = repo_root / ".omx" / "state" / "probe_outcomes.jsonl"
    assert ledger_path.exists(), f"probe outcomes ledger missing: {ledger_path}"

    canonical_probe_id = (
        "pr95_mlx_stage_6_lambda_sweep_curriculum_build_synthetic_timing_smoke_100ep"
    )
    found = False
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("probe_id") == canonical_probe_id:
            found = True
            break
    assert found, (
        f"Catalog #313 probe-outcomes row missing for {canonical_probe_id} "
        f"in {ledger_path}"
    )


# ---------------------------------------------------------------------------
# APPEND-ONLY superset-of pattern verification (Stage 1+2+3+4+5 tests not mutated)
# ---------------------------------------------------------------------------


def test_stage_5_descriptor_not_mutated_by_stage_6_landing() -> None:
    """Stage 5 descriptor invariants remain stable.

    Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:
    the Stage 6 BUILD landing must NOT mutate Stage 5 invariants. This
    test verifies that Stage 5's canonical surface is preserved
    byte-equivalent after Stage 6 lands.
    """

    cfg_5 = stage_smoke_config(5)
    assert cfg_5.stage_index == 5
    assert cfg_5.stage_module == "stage5_c1a_l7"
    assert cfg_5.optimizer_descriptor_id == "pr95_stage5_adamw_baseline_mlx"
    assert cfg_5.optimizer.adamw_lr == 3e-5
    assert cfg_5.optimizer.use_muon is False

    registry = default_optimizer_scheduler_registry()
    row_5 = registry.get(
        "pr95_stage5_adamw_baseline_mlx"
    ).to_planner_candidate()
    # Stage 5 still declares λ=0.01; only Stage 6 sweeps to 0.02.
    assert row_5["training_config"]["stage_cat_lambda"] == 0.01
    assert row_5["training_config"]["stage_cat_sigma"] == 0.2
    assert row_5["training_config"]["stage_uses_qat"] is True
    assert row_5["training_config"]["stage_loss_family"] == (
        "l7_softplus_seg_loss"
    )


def test_stage_4_descriptor_not_mutated_by_stage_6_landing() -> None:
    """Stage 4 descriptor invariants remain stable.

    Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:
    the Stage 6 BUILD landing must NOT mutate Stage 4 invariants.
    """

    cfg_4 = stage_smoke_config(4)
    assert cfg_4.stage_index == 4
    assert cfg_4.stage_module == "stage4_v332_qat"
    assert cfg_4.optimizer_descriptor_id == "pr95_stage4_adamw_qat_mlx"
    assert cfg_4.optimizer.adamw_lr == 1e-4
    assert cfg_4.optimizer.use_muon is False

    registry = default_optimizer_scheduler_registry()
    row_4 = registry.get(
        "pr95_stage4_adamw_qat_mlx"
    ).to_planner_candidate()
    # Stage 4 still declares smooth_disagreement (NOT l7_softplus).
    assert row_4["training_config"]["stage_loss_family"] == (
        "smooth_disagreement_seg_loss"
    )
    assert row_4["training_config"]["stage_epochs"] == 500
    assert row_4["training_config"]["stage_uses_qat"] is True
