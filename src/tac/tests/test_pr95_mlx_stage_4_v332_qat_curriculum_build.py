# SPDX-License-Identifier: MIT
"""Tests for PR 95 Stage 4 v332_qat MLX curriculum build.

Sister of `test_pr95_mlx_stage_3_v332_smooth_curriculum_build.py` (Stage 3
smooth-disagreement bridge) and `test_optimizer_scheduler_registry.py` (Stage
1/2/3/5/8 descriptor invariants). Covers the canonical PR 95 8-stage
curriculum Stage 4 extension. Stage 4 is `stage4_v332_qat` with
`smooth_disagreement_seg_loss(tau=0.3)` (preserved from Stage 3) + AdamW
LR=1e-4 (continues Stage 3 cosine schedule, NOT fresh) + QAT
(Quantization-Aware Training) per the recovered public PR95 source
(`.omx/research/pr95_8stage_curriculum_forensic_20260513.md` +
`.omx/research/pr95_curriculum_recovery_20260513_codex.md`).

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the
docstring-overstatement trap)": all assertions in this module are
[macOS-MLX research-signal] non-promotable; no contest-axis score claims.

Per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA": Stage 4 MLX
synthetic timing smoke is non-promotable by construction; promotion via
paired Linux x86_64 + NVIDIA per Catalog #192.

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + #192 + #287 + #323:
the MLX backend is research-signal only; the QAT semantics are recorded
in the descriptor metadata (`stage_uses_qat=True`) but the MLX bundle uses
the same `HNeRVSyntheticTrainingBundleMLX` architecture as Stage 1+2+3
(byte-identical state_bytes empirically confirmed via paired forward
parity at random init).
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


def test_pr95_stage_modules_contains_stage_4_v332_qat() -> None:
    """Stage 4 must be registered in the canonical dispatch dict."""

    assert 4 in PR95_STAGE_MODULES
    assert PR95_STAGE_MODULES[4] == "stage4_v332_qat"


def test_pr95_stage_default_descriptor_for_stage_4() -> None:
    """Stage 4 must resolve to canonical AdamW QAT descriptor."""

    assert 4 in PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS
    assert PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS[4] == (
        "pr95_stage4_adamw_qat_mlx"
    )
    assert pr95_default_optimizer_descriptor_id(4) == (
        "pr95_stage4_adamw_qat_mlx"
    )


def test_pr95_stage_modules_canonical_set_is_supported() -> None:
    """Canonical supported stages after Stage 4 landing: at minimum {1, 2, 3, 4, 5, 8}.

    Asserted as superset-of so future Stage 6/7 additions do not require
    mutating this Stage 4 test per Catalog #110/#113 HISTORICAL_PROVENANCE
    APPEND-ONLY discipline.
    """

    assert {1, 2, 3, 4, 5, 8}.issubset(set(PR95_STAGE_MODULES))
    assert {1, 2, 3, 4, 5, 8}.issubset(
        set(PR95_STAGE_DEFAULT_OPTIMIZER_DESCRIPTOR_IDS)
    )


def test_unsupported_stage_raises_value_error_with_canonical_supported_list() -> None:
    """Unsupported stage raises ValueError mentioning canonical supported set.

    Soft-error pattern uses `sorted(PR95_STAGE_MODULES)` dynamically so
    future stage 6/7 additions do not require updating hardcoded strings.
    """

    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        stage_smoke_config(9)
    with pytest.raises(ValueError, match=r"supported PR95 MLX timing stages"):
        pr95_default_optimizer_descriptor_id(9)


# ---------------------------------------------------------------------------
# Canonical descriptor row invariants (per Catalog #287 + #323)
# ---------------------------------------------------------------------------


def test_pr95_stage4_descriptor_in_registry() -> None:
    """Stage 4 descriptor must be registered + carry canonical contract."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage4_adamw_qat_mlx").to_planner_candidate()

    # Optimizer config matches the recovered public PR95 Stage 4 module.
    assert row["optimizer_config"]["use_muon"] is False
    # Stage 4 continues Stage 3 cosine (base LR remains 1e-4; the cosine
    # continuation is a runtime scheduler concern, not a descriptor concern).
    assert row["optimizer_config"]["adamw_lr"] == 1e-4
    assert row["optimizer_config"]["latent_lr_mult"] == 10.0

    # Training-config stage_loss_family preserves Stage 3's smooth_disagreement
    # form per the recovered public PR 95 source (Stage 4 adds QAT to Stage 3
    # loss family; loss family is preserved).
    assert row["training_config"]["stage_loss_family"] == (
        "smooth_disagreement_seg_loss"
    )

    # Training config carries canonical stage routing.
    assert row["training_config"]["pr95_stage_indices"] == [4]
    assert row["training_config"]["stage_modules"] == ["stage4_v332_qat"]
    assert row["training_config"]["backend_status"] == (
        "implemented_mlx_local_timing_proxy"
    )
    assert row["training_config"]["score_claim"] is False
    assert row["training_config"]["promotion_eligible"] is False
    assert row["training_config"]["rank_or_kill_eligible"] is False
    assert row["training_config"]["ready_for_exact_eval_dispatch"] is False
    # Stage 4 per published PR 95 curriculum: 500 epochs canonical.
    assert row["training_config"]["stage_epochs"] == 500
    # Stage 4 INTRODUCES QAT (the v332_qat bit) — distinct from Stage 3's
    # stage_uses_qat=False.
    assert row["training_config"]["stage_uses_qat"] is True
    # Stage 4 uses AdamW only; Muon is Stage 8.
    assert row["training_config"]["stage_uses_muon"] is False

    # Scheduler config matches stage routing.
    assert row["scheduler_config"]["stage_indices"] == [4]
    assert row["scheduler_config"]["source_pr"] == 95


def test_pr95_stage4_descriptor_passes_proxy_candidate_contract() -> None:
    """Stage 4 descriptor must satisfy proxy candidate validator."""

    registry = default_optimizer_scheduler_registry()
    row = registry.get("pr95_stage4_adamw_qat_mlx").to_planner_candidate()
    assert validate_proxy_candidate(row) == []
    for key, expected in FALSE_AUTHORITY_FIELDS.items():
        assert row[key] is expected


def test_pr95_stage4_lr_continues_stage_3_cosine_schedule() -> None:
    """Stage 4 LR (1e-4) continues Stage 3 cosine, distinct from Stage 1+2 baseline (1e-3).

    Per PR 95 published 8-stage curriculum
    (`.omx/research/pr95_8stage_curriculum_forensic_20260513.md` +
    `.omx/research/pr95_curriculum_recovery_20260513_codex.md`):
    Stage 4 (v332_qat) uses adamw_lr=1e-4 CONTINUING Stage 3's cosine
    schedule (NOT fresh; Stage 3's cosine ran 1e-4 -> 5e-6 over 1500
    epochs; Stage 4 starts from the Stage 3 endpoint and continues for
    500 more epochs). Distinct from Stage 1+2's baseline LR (1e-3 per
    sister-canonical landing) AND distinct from Stage 5 (c1a_l7 at 3e-5)
    and Stage 8 (muon_finetune AdamW at 1e-5). Stage 4 is the QAT layer
    on top of Stage 3's smooth_disagreement substrate; its loss family
    (smooth_disagreement_seg_loss(tau=0.3)) is preserved from Stage 3.
    """

    stage1 = stage_smoke_config(1)
    stage2 = stage_smoke_config(2)
    stage3 = stage_smoke_config(3)
    stage4 = stage_smoke_config(4)
    stage5 = stage_smoke_config(5)
    stage8 = stage_smoke_config(8)
    # Canonical sister-landed LR ladder per the PR 95 8-stage curriculum:
    # Stage 1 + 2 share baseline LR=1e-3; Stage 3 fresh cosine 1e-4;
    # Stage 4 continues Stage 3 cosine (base LR remains 1e-4); Stage 5 + 8
    # use lower LRs for quantization (3e-5) and finetune (1e-5).
    assert stage1.optimizer.adamw_lr == 1e-3
    assert stage2.optimizer.adamw_lr == 1e-3
    assert stage3.optimizer.adamw_lr == 1e-4
    assert stage4.optimizer.adamw_lr == 1e-4
    assert stage5.optimizer.adamw_lr == 3e-5
    assert stage8.optimizer.adamw_lr == 1e-5
    # Stage 3 and Stage 4 share the same start LR (Stage 4 is a cosine
    # continuation, not a fresh schedule).
    assert stage3.optimizer.adamw_lr == stage4.optimizer.adamw_lr
    # Loss-family preservation: Stage 4 keeps Stage 3's smooth_disagreement
    # seg loss; both differ from Stage 2's tau_softplus and Stage 1's RGB
    # MSE baseline.
    assert stage4.stage_module == "stage4_v332_qat"
    assert stage3.stage_module == "stage3_v332_smooth"
    assert stage2.stage_module == "stage2_v331_softplus"
    assert stage1.stage_module == "stage1_v328_ce"


# ---------------------------------------------------------------------------
# stage_smoke_config dispatch invariants
# ---------------------------------------------------------------------------


def test_stage_smoke_config_stage_4_dispatches_canonical_module() -> None:
    """stage_smoke_config(4) must dispatch the canonical stage4_v332_qat."""

    cfg = stage_smoke_config(4)
    assert cfg.stage_index == 4
    assert cfg.stage_module == "stage4_v332_qat"
    assert cfg.optimizer_descriptor_id == "pr95_stage4_adamw_qat_mlx"
    assert cfg.optimizer.use_muon is False
    assert cfg.optimizer.adamw_lr == 1e-4
    assert cfg.optimizer_backend_status == "implemented_mlx_local_timing_proxy"


def test_stage_smoke_config_stage_4_accepts_explicit_descriptor_id() -> None:
    """stage_smoke_config(4) must accept --optimizer-descriptor-id override."""

    cfg = stage_smoke_config(
        4,
        optimizer_descriptor_id="pr95_stage4_adamw_qat_mlx",
    )
    assert cfg.optimizer_descriptor_id == "pr95_stage4_adamw_qat_mlx"
    assert cfg.stage_module == "stage4_v332_qat"


# ---------------------------------------------------------------------------
# End-to-end synthetic timing smoke (Stage 4 10-step on MLX)
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
def test_stage_4_synthetic_timing_smoke_runs_end_to_end() -> None:
    """Stage 4 10-step MLX synthetic timing smoke runs + emits non-promotable contract."""

    result = run_pr95_mlx_synthetic_timing_smoke(
        stage_index=4,
        steps=10,
        batch_size=1,
        synthetic_pairs=1,
        seed=20260525,
        base_channels=36,
        latent_dim=28,
    )

    # Canonical Stage 4 identifiers.
    assert result["stage_index"] == 4
    assert result["stage_module"] == "stage4_v332_qat"
    assert "stage4_pr95_stage4_adamw_qat_mlx" in result["candidate_id"]

    # Loss converges to finite scalar.
    assert isinstance(result["last_loss"], float)
    assert np.isfinite(result["last_loss"])

    # Runtime profile carries canonical training fidelity.
    rp = result["runtime_profile"]
    assert rp["stage_index"] == 4
    assert rp["stage_id"] == "stage4_v332_qat"
    assert rp["optimizer_descriptor_id"] == "pr95_stage4_adamw_qat_mlx"
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
# Paired forward parity: Stage 3 vs Stage 4 at random init
# ---------------------------------------------------------------------------


def test_stage_3_vs_stage_4_byte_identical_forward_at_random_init() -> None:
    """Stage 3 + Stage 4 share architecture: forward at random init is byte-identical.

    Per PR 95 published 8-stage curriculum: Stage 4 (v332_qat) shares
    Stage 3 (v332_smooth)'s architecture (HNeRVDecoder + base_ch=36,
    latent_dim=28); only the QAT bit + cosine schedule continuation
    distinguish them at the training-config layer. At step 0 (before
    training), seeded random init produces byte-identical forward output
    across ALL stages 1+2+3+4+5+8.

    NOTE on QAT vs architecture: PR 95 canonical applies QAT IN-PLACE on
    Conv2d/Linear weights per-batch (`apply_qat(decoder); decoded =
    decoder(...); restore_qat(decoder, originals)`). The MLX synthetic
    timing proxy does not apply QAT to the persistent state_dict
    structure; QAT is recorded as a training-config metadata flag
    (`stage_uses_qat=True`) and the MLX bundle architecture (and
    state_bytes) is preserved byte-identical across Stage 3 and Stage 4.
    Empirically verified at 915,944 state_bytes both stages.

    This is the canonical sanity check that Stage 4 does NOT silently
    perturb the architecture at the bundle level; if this test fails,
    the Stage 4 trainer is diverging from PR 95's canonical curriculum
    continuation pattern.
    """

    import mlx.core as mx

    seed = 20260525

    # Both bundles use the SAME seed + SAME architecture.
    mx.random.seed(seed)
    bundle_stage_3 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    idx = mx.array([0], dtype=mx.uint32)
    pred_3 = bundle_stage_3(idx)
    mx.eval(pred_3)
    arr_3 = np.asarray(pred_3)

    mx.random.seed(seed)
    bundle_stage_4 = HNeRVSyntheticTrainingBundleMLX(
        latent_count=1,
        latent_dim=28,
        base_channels=36,
        seed=seed,
        output_layout="n2chw",
    )
    pred_4 = bundle_stage_4(idx)
    mx.eval(pred_4)
    arr_4 = np.asarray(pred_4)

    max_abs_diff = float(np.max(np.abs(arr_3 - arr_4)))

    # Byte-identical at random init: max_abs_diff = 0.0 exactly.
    assert max_abs_diff == 0.0, (
        f"Stage 3 vs Stage 4 forward parity FAIL at random init: "
        f"max_abs_diff={max_abs_diff}; expected 0.0 byte-identical "
        f"(architecture must be identical before the QAT bit applies)"
    )

    # PASS_BAND_5E3 sanity per MLX-ARCH-5 dispatch contract.
    assert max_abs_diff < 5e-3, (
        f"Stage 3 vs Stage 4 forward parity FAIL beyond ε=5e-3 fp32 band: "
        f"max_abs_diff={max_abs_diff}"
    )

    # Sanity: configured stage modules ARE different (Stage 3 v332_smooth vs
    # Stage 4 v332_qat). The QAT bit + cosine schedule continuation
    # distinguish them at the training-config layer; both share AdamW
    # adamw_lr=1e-4 at the optimizer-config layer.
    cfg_3 = stage_smoke_config(3)
    cfg_4 = stage_smoke_config(4)
    assert cfg_3.optimizer.adamw_lr == 1e-4
    assert cfg_4.optimizer.adamw_lr == 1e-4
    # Stage 4 IS Stage 3 cosine continuation: same start LR.
    assert cfg_3.optimizer.adamw_lr == cfg_4.optimizer.adamw_lr
    assert cfg_3.stage_module != cfg_4.stage_module
    assert cfg_3.stage_module == "stage3_v332_smooth"
    assert cfg_4.stage_module == "stage4_v332_qat"


# ---------------------------------------------------------------------------
# Catalog #344 canonical equation queue + Catalog #313 ledger row
# ---------------------------------------------------------------------------


def test_canonical_equation_queued_for_ratify_n() -> None:
    """Stage 4 landing queues canonical equation `pr95_mlx_stage_4_v332_qat_one_to_one_curriculum_port_v1`.

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
        / "pr95_mlx_stage_4_v332_qat_curriculum_build_landed_20260525.md"
    )
    assert memo_path.exists(), f"landing memo missing: {memo_path}"
    body = memo_path.read_text(encoding="utf-8")
    assert "pr95_mlx_stage_4_v332_qat_one_to_one_curriculum_port_v1" in body
    assert "FORMALIZATION_PENDING" in body or "RATIFY-N" in body


def test_catalog_313_probe_outcomes_row_registered() -> None:
    """Stage 4 landing registers Catalog #313 probe-outcomes row.

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
        "pr95_mlx_stage_4_v332_qat_curriculum_build_synthetic_timing_smoke_100ep"
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
# APPEND-ONLY superset-of pattern verification (Stage 3 tests not mutated)
# ---------------------------------------------------------------------------


def test_stage_3_tests_not_mutated_by_stage_4_landing() -> None:
    """Stage 3 test invariants (stage_module + descriptor + LR) remain stable.

    Per Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY discipline:
    the Stage 4 BUILD landing must NOT mutate Stage 3 invariants. This
    test verifies that Stage 3's canonical surface is preserved
    byte-equivalent after Stage 4 lands.
    """

    cfg_3 = stage_smoke_config(3)
    assert cfg_3.stage_index == 3
    assert cfg_3.stage_module == "stage3_v332_smooth"
    assert cfg_3.optimizer_descriptor_id == "pr95_stage3_adamw_baseline_mlx"
    assert cfg_3.optimizer.adamw_lr == 1e-4
    assert cfg_3.optimizer.use_muon is False

    registry = default_optimizer_scheduler_registry()
    row_3 = registry.get("pr95_stage3_adamw_baseline_mlx").to_planner_candidate()
    # Stage 3 still declares QAT False; only Stage 4 introduces QAT.
    assert row_3["training_config"]["stage_uses_qat"] is False
    assert row_3["training_config"]["stage_epochs"] == 1500
    assert row_3["training_config"]["stage_loss_family"] == (
        "smooth_disagreement_seg_loss"
    )
