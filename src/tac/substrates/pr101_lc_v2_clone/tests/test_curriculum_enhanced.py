"""Regression tests for the PR95++ enhanced curriculum surface.

Each of the 11 enhancements has at least one positive test (it activates
when enabled) and at least one negative test (it does not apply when
disabled). Plus composition + audit + summary tests that pin the
contract for downstream consumers (trainer, recipe, dispatcher).

Per the operator directive 2026-05-13: maximum signal + holistic
engineering. Tests favour SHARP assertions over loose ``isinstance``
checks so a future modder cannot break the contract while keeping the
tests green.
"""

from __future__ import annotations

import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from tac.substrates.pr101_lc_v2_clone import (
    CurriculumStageConfig,
    Pr101LcV2CloneConfig,
    Pr101LcV2CloneSubstrate,
)
from tac.substrates.pr101_lc_v2_clone.curriculum_enhanced import (
    ENHANCEMENT_KEYS,
    EnhancedCurriculumConfig,
    TernaryStageBudget,
    _apply_enhancements_to_stage,
    apply_cross_block_skip,
    apply_logit_softcap,
    audit_enhanced_curriculum_against_hnerv_parity_lessons,
    build_enhanced_stages,
    build_faithful_stages,
    build_optimizer_for_enhanced_stage,
    compute_wsd_lr,
    default_ternary_schedule,
    enhancement_summary,
    logit_softcap_30,
    stage0_pretrained_driving_prior_bootstrap,
    validate_enhanced_curriculum_config,
)


def test_default_enhanced_config_is_research_only_and_opt_in_composable() -> None:
    cfg = EnhancedCurriculumConfig()
    enabled = cfg.enabled_enhancements()

    assert cfg.research_only is True
    assert "sabor_boundary_only_composition" not in enabled
    assert "s2sbs_hf_byte_stuffing_composition" not in enabled
    assert set(enabled).issubset(set(ENHANCEMENT_KEYS))


def test_faithful_config_disables_enhancements() -> None:
    cfg = EnhancedCurriculumConfig.faithful_config()

    assert cfg.enabled_enhancements() == ()
    stages = build_faithful_stages(stage_epoch_overrides={"stage1_v328_ce": 3})
    assert len(stages) == 8
    assert stages[0].epochs == 3
    assert all("enhancements" not in stage.extras for stage in stages)


def test_enhanced_stages_thread_metadata_without_score_authority() -> None:
    cfg = EnhancedCurriculumConfig(
        iglt_polish_steps_per_stage=7,
        enable_sabor_boundary_only_composition=True,
        enable_s2sbs_hf_byte_stuffing_composition=True,
    )
    stages = build_enhanced_stages(
        cfg,
        stage_epoch_overrides={
            "stage0_pretrained_driving_prior_bootstrap": 2,
            "stage8_muon_finetune": 4,
        },
    )

    assert len(stages) == 9
    assert stages[0].name == "stage0_pretrained_driving_prior_bootstrap"
    assert stages[0].epochs == 2
    assert stages[-1].name == "stage8_muon_finetune"
    assert stages[-1].epochs == 4
    assert all(stage.extras["research_only"] is True for stage in stages)
    assert all(stage.use_muon is True for stage in stages)
    assert stages[-1].extras["ternary_bits"] == 2
    assert stages[-1].extras["quantizer"] == "ternary_ste"
    assert stages[-1].extras["iglt_polish_steps"] == 7
    assert "sabor_boundary_only_composition" in stages[-1].extras["enhancements"]
    assert "s2sbs_hf_byte_stuffing_composition" in stages[-1].extras["enhancements"]


def test_ternary_stage_budget_is_monotone_and_index_checked() -> None:
    budget = TernaryStageBudget()

    assert [budget.for_stage_index(i) for i in range(1, 9)] == [
        32,
        32,
        32,
        8,
        8,
        8,
        6,
        2,
    ]
    with pytest.raises(ValueError, match="monotone"):
        TernaryStageBudget(stage5_bits=16)
    with pytest.raises(ValueError, match=r"\[1, 8\]"):
        budget.for_stage_index(9)


def test_wsd_lr_schedule_has_warmup_plateau_decay_and_floor() -> None:
    warmup = compute_wsd_lr(
        step=0,
        total_steps=100,
        peak_lr=1.0,
        warmup_fraction=0.1,
        decay_fraction=0.2,
    )
    plateau = compute_wsd_lr(
        step=20,
        total_steps=100,
        peak_lr=1.0,
        warmup_fraction=0.1,
        decay_fraction=0.2,
    )
    decayed = compute_wsd_lr(
        step=90,
        total_steps=100,
        peak_lr=1.0,
        warmup_fraction=0.1,
        decay_fraction=0.2,
    )
    floor = compute_wsd_lr(
        step=100,
        total_steps=100,
        peak_lr=1.0,
        warmup_fraction=0.1,
        decay_fraction=0.2,
    )

    assert 0.0 < warmup < plateau
    assert plateau == pytest.approx(1.0)
    assert floor == pytest.approx(0.1)
    assert floor < decayed < plateau


def test_pr95plus_smoke_epoch_budget_matches_operator_contract() -> None:
    from experiments.train_substrate_pr101_lc_v2_clone_enhanced_curriculum import (
        _smoke_stage_epoch_overrides,
    )

    epochs = _smoke_stage_epoch_overrides(Namespace(smoke_epochs=100))

    assert sum(epochs.values()) == 100
    assert epochs["stage0_pretrained_driving_prior_bootstrap"] >= 1
    assert epochs["stage8_muon_finetune"] >= epochs[
        "stage0_pretrained_driving_prior_bootstrap"
    ]


def test_pr95plus_smoke_archive_fallback_is_valid_prc1_not_json_placeholder() -> None:
    from experiments.train_substrate_pr101_lc_v2_clone_enhanced_curriculum import (
        _pack_smoke_archive_bytes,
    )
    from tac.substrates.pr101_lc_v2_clone.archive import parse_archive

    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig())
    latents = torch.zeros(sub.cfg.num_pairs, sub.cfg.latent_dim)

    archive_bytes, archive_meta = _pack_smoke_archive_bytes(
        sub.state_dict(),
        latents=latents,
        meta={"lane_id": "test", "substrate_tag": "pr95plus"},
    )

    parsed = parse_archive(archive_bytes)
    assert archive_bytes.startswith(b"PRC1")
    assert parsed.latents.shape == (600, 28)
    assert archive_meta["smoke_archive_mode"] in {
        "ema_state",
        "zero_state_valid_prc1",
    }


def test_pr95plus_runtime_vendor_is_self_contained(tmp_path: Path) -> None:
    from experiments.train_substrate_pr101_lc_v2_clone_enhanced_curriculum import (
        _vendor_runtime,
    )

    _vendor_runtime(tmp_path)
    runtime = tmp_path / "runtime"

    assert (runtime / "inflate.sh").is_file()
    assert (
        runtime / "src/tac/substrates/pr101_lc_v2_clone/inflate.py"
    ).is_file()
    assert (
        runtime / "src/tac/packet_compiler/pr101_decoder_byte_maps.py"
    ).is_file()
    inflate_sh = (runtime / "inflate.sh").read_text(encoding="utf-8")
    assert 'PYTHONPATH="$HERE/src:${PYTHONPATH:-}"' in inflate_sh
    assert "python -m tac.substrates.pr101_lc_v2_clone.inflate" in inflate_sh


def test_logit_softcap_is_smooth_and_bounded() -> None:
    x = torch.tensor([-300.0, -3.0, 0.0, 3.0, 300.0], requires_grad=True)

    y = logit_softcap_30(x)
    y.sum().backward()

    assert torch.all(y <= 30.0)
    assert torch.all(y >= -30.0)
    assert x.grad is not None
    assert torch.all(torch.isfinite(x.grad))
    assert x.grad[2] == pytest.approx(1.0)


def test_enhancement_summary_and_parity_audit_are_machine_readable() -> None:
    cfg = EnhancedCurriculumConfig()
    summary = enhancement_summary(cfg)
    audit = audit_enhanced_curriculum_against_hnerv_parity_lessons(cfg)

    assert summary["curriculum"] == "pr95_enhanced"
    assert summary["research_only"] is True
    assert summary["score_claim"] is False
    assert summary["promotion_eligible"] is False
    assert summary["ready_for_exact_eval_dispatch"] is False
    assert "FAIL" not in {
        verdict for by_lesson in audit.values() for verdict in by_lesson.values()
    }


# -----------------------------------------------------------------------------
# Module contract surface
# -----------------------------------------------------------------------------


def test_eleven_enhancement_keys_named() -> None:
    assert len(ENHANCEMENT_KEYS) == 11
    assert ENHANCEMENT_KEYS[0] == "muon_every_stage"
    assert ENHANCEMENT_KEYS[-1] == "s2sbs_hf_byte_stuffing_composition"


def test_default_config_has_nine_enhancements_on() -> None:
    cfg = EnhancedCurriculumConfig()
    assert len(cfg.enabled_enhancements()) == 9


def test_config_refuses_degenerate_wsd() -> None:
    with pytest.raises(ValueError, match=r"must be < 1\.0"):
        EnhancedCurriculumConfig(wsd_warmup_fraction=0.6, wsd_decay_fraction=0.5)


def test_config_refuses_negative_softcap() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        EnhancedCurriculumConfig(logit_softcap_value=-1.0)


def test_config_refuses_oversized_floor_ratio() -> None:
    with pytest.raises(ValueError, match="must be in"):
        EnhancedCurriculumConfig(wsd_floor_ratio=1.5)


def test_config_refuses_negative_atick_redlich() -> None:
    with pytest.raises(ValueError):
        EnhancedCurriculumConfig(atick_redlich_lambda=-1e-3)


def test_config_refuses_negative_iglt_steps() -> None:
    with pytest.raises(ValueError):
        EnhancedCurriculumConfig(iglt_polish_steps_per_stage=-1)


# -----------------------------------------------------------------------------
# E1 — Muon at every stage
# -----------------------------------------------------------------------------


def test_e1_muon_positive() -> None:
    cfg = EnhancedCurriculumConfig(enable_muon_every_stage=True)
    stages = build_enhanced_stages(cfg)
    pr95_stages = [s for s in stages if not s.name.startswith("stage0_")]
    for s in pr95_stages:
        assert s.use_muon is True, s.name


def test_e1_muon_negative() -> None:
    cfg = EnhancedCurriculumConfig(
        enable_muon_every_stage=False,
        enable_pretrained_driving_prior_bootstrap=False,
    )
    stages = build_enhanced_stages(cfg)
    muon_count = sum(1 for s in stages if s.use_muon)
    assert muon_count == 1
    assert stages[-1].use_muon is True


def test_e1_optimizer_builder_yields_muon_when_enabled() -> None:
    from tac.optimization.muon import MuonOptimizer

    cfg = EnhancedCurriculumConfig(enable_muon_every_stage=True)
    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig(base_channels=16))
    stages = build_enhanced_stages(cfg)
    stage1 = next(s for s in stages if s.name.startswith("stage1_"))
    adamw, muon = build_optimizer_for_enhanced_stage(
        model=sub, stage=stage1, enhanced_cfg=cfg
    )
    assert isinstance(adamw, torch.optim.AdamW)
    assert isinstance(muon, MuonOptimizer)


def test_e1_optimizer_builder_no_muon_when_disabled() -> None:
    cfg = EnhancedCurriculumConfig(
        enable_muon_every_stage=False,
        enable_pretrained_driving_prior_bootstrap=False,
    )
    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig(base_channels=16))
    stages = build_enhanced_stages(cfg)
    stage1 = stages[0]
    assert not stage1.use_muon
    adamw, muon = build_optimizer_for_enhanced_stage(
        model=sub, stage=stage1, enhanced_cfg=cfg
    )
    assert isinstance(adamw, torch.optim.AdamW)
    assert muon is None


# -----------------------------------------------------------------------------
# E2 — IGLT polish per stage
# -----------------------------------------------------------------------------


def test_e2_iglt_polish_threads_extras() -> None:
    cfg = EnhancedCurriculumConfig(
        enable_iglt_polish_per_stage=True,
        iglt_polish_steps_per_stage=42,
    )
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert s.extras["iglt_polish_steps"] == 42
        assert s.extras["iglt_polish_fisher_estimation"] in (
            "diagonal",
            "block_diagonal",
            "kfac",
        )


def test_e2_iglt_polish_negative() -> None:
    cfg = EnhancedCurriculumConfig(enable_iglt_polish_per_stage=False)
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert "iglt_polish_steps" not in s.extras


# -----------------------------------------------------------------------------
# E3 — Stage-aware ternary QAT
# -----------------------------------------------------------------------------


def test_e3_ternary_default_targets() -> None:
    ts = default_ternary_schedule()
    assert ts.for_stage_index(1) == 32
    assert ts.for_stage_index(4) == 8
    assert ts.for_stage_index(7) == 6
    assert ts.for_stage_index(8) == 2


def test_e3_ternary_extras_threaded() -> None:
    cfg = EnhancedCurriculumConfig(enable_stage_aware_ternary_qat=True)
    stages = build_enhanced_stages(cfg)
    pr95_stages = [s for s in stages if not s.name.startswith("stage0_")]
    for idx, s in enumerate(pr95_stages, start=1):
        assert "ternary_bits" in s.extras
        assert s.extras["ternary_bits"] == cfg.ternary_schedule.for_stage_index(idx)


def test_e3_ternary_negative() -> None:
    cfg = EnhancedCurriculumConfig(enable_stage_aware_ternary_qat=False)
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert "ternary_bits" not in s.extras


# -----------------------------------------------------------------------------
# E4 — WSD learning rate schedule
# -----------------------------------------------------------------------------


def test_e4_wsd_warmup_starts_above_zero() -> None:
    lr0 = compute_wsd_lr(step=0, total_steps=1000, peak_lr=1.0)
    assert lr0 > 0.0


def test_e4_wsd_plateau_at_peak() -> None:
    lr = compute_wsd_lr(step=500, total_steps=1000, peak_lr=1.0)
    assert math.isclose(lr, 1.0)


def test_e4_wsd_monotone_decay() -> None:
    prev = float("inf")
    for step in range(900, 1001):
        lr = compute_wsd_lr(step=step, total_steps=1000, peak_lr=1.0)
        assert lr <= prev + 1e-9
        prev = lr


def test_e4_wsd_refuses_invalid_input() -> None:
    with pytest.raises(ValueError):
        compute_wsd_lr(step=0, total_steps=0, peak_lr=1.0)
    with pytest.raises(ValueError):
        compute_wsd_lr(step=-1, total_steps=100, peak_lr=1.0)


def test_e4_wsd_extras_threaded() -> None:
    cfg = EnhancedCurriculumConfig(enable_wsd_lr_schedule=True)
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert s.extras["wsd_warmup_fraction"] == cfg.wsd_warmup_fraction
        assert s.extras["wsd_floor_ratio"] == cfg.wsd_floor_ratio


def test_e4_wsd_negative() -> None:
    cfg = EnhancedCurriculumConfig(enable_wsd_lr_schedule=False)
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert "wsd_warmup_fraction" not in s.extras


# -----------------------------------------------------------------------------
# E5 — Logit softcap
# -----------------------------------------------------------------------------


def test_e5_softcap_saturates() -> None:
    # tanh saturates at ±1; at cap=30 the output is bounded by ±cap inclusive
    # at fp32 precision (saturation is sharp).
    x = torch.tensor([-1000.0, 1000.0])
    y = logit_softcap_30(x, cap=30.0)
    assert (y.abs() <= 30.0).all()
    assert y[1].item() > 29.9
    assert y[0].item() < -29.9


def test_e5_softcap_identity_near_zero() -> None:
    x = torch.tensor([0.0, 0.1, -0.1])
    y = logit_softcap_30(x, cap=30.0)
    assert math.isclose(y[0].item(), 0.0, abs_tol=1e-6)
    assert math.isclose(y[1].item(), 0.1, rel_tol=1e-3)


def test_e5_softcap_refuses_zero_cap() -> None:
    with pytest.raises(ValueError):
        logit_softcap_30(torch.zeros(1), cap=0.0)


def test_e5_apply_softcap_preserves_shape() -> None:
    x = torch.randn(2, 3, 4, 5)
    y = apply_logit_softcap(x)
    assert y.shape == x.shape


def test_e5_softcap_negative_via_extras() -> None:
    cfg = EnhancedCurriculumConfig(enable_logit_softcap=False)
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert "logit_softcap_value" not in s.extras


# -----------------------------------------------------------------------------
# E6 — Cross-block skip
# -----------------------------------------------------------------------------


def test_e6_cross_block_skip_marks_substrate() -> None:
    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig(base_channels=16))
    apply_cross_block_skip(sub, early_block_idx=0, late_block_idx=5)
    assert sub._pr95_enhanced_cross_block_skip_patched is True
    assert sub._pr95_enhanced_cross_block_skip_early_idx == 0
    assert sub._pr95_enhanced_cross_block_skip_late_idx == 5


def test_e6_cross_block_skip_idempotent() -> None:
    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig(base_channels=16))
    apply_cross_block_skip(sub)
    apply_cross_block_skip(sub)
    assert sub._pr95_enhanced_cross_block_skip_patched is True


def test_e6_cross_block_skip_rejects_invalid_indices() -> None:
    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig(base_channels=16))
    with pytest.raises(ValueError, match="long-range"):
        apply_cross_block_skip(sub, early_block_idx=3, late_block_idx=2)
    with pytest.raises(ValueError, match=">= n_blocks"):
        apply_cross_block_skip(sub, early_block_idx=0, late_block_idx=99)


def test_e6_cross_block_skip_negative() -> None:
    sub = Pr101LcV2CloneSubstrate(Pr101LcV2CloneConfig(base_channels=16))
    assert not getattr(sub, "_pr95_enhanced_cross_block_skip_patched", False)


# -----------------------------------------------------------------------------
# E7 — Atick-Redlich efficient coding
# -----------------------------------------------------------------------------


def test_e7_atick_redlich_threaded() -> None:
    cfg = EnhancedCurriculumConfig(
        enable_atick_redlich_efficient_coding=True,
        atick_redlich_lambda=2e-3,
    )
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert s.extras["atick_redlich_lambda"] == 2e-3


def test_e7_atick_redlich_negative() -> None:
    cfg = EnhancedCurriculumConfig(enable_atick_redlich_efficient_coding=False)
    stages = build_enhanced_stages(cfg)
    for s in stages:
        assert "atick_redlich_lambda" not in s.extras


# -----------------------------------------------------------------------------
# E8 — Catalog #197 full-CPU validation gate
# -----------------------------------------------------------------------------


def test_e8_full_cpu_gate_positive_in_summary() -> None:
    cfg = EnhancedCurriculumConfig(enable_full_cpu_validation_gate=True)
    assert "full_cpu_validation_gate" in enhancement_summary(cfg)["enabled_enhancements"]


def test_e8_full_cpu_gate_negative_in_summary() -> None:
    cfg = EnhancedCurriculumConfig(enable_full_cpu_validation_gate=False)
    assert "full_cpu_validation_gate" not in enhancement_summary(cfg)["enabled_enhancements"]


# -----------------------------------------------------------------------------
# E9 — Pre-trained driving prior bootstrap
# -----------------------------------------------------------------------------


def test_e9_bootstrap_adds_stage0() -> None:
    cfg = EnhancedCurriculumConfig(enable_pretrained_driving_prior_bootstrap=True)
    stages = build_enhanced_stages(cfg)
    assert stages[0].name == "stage0_pretrained_driving_prior_bootstrap"
    assert stages[0].extras["init_from_pretrained_driving_prior"] is True
    assert len(stages) == 9


def test_e9_bootstrap_negative() -> None:
    cfg = EnhancedCurriculumConfig(enable_pretrained_driving_prior_bootstrap=False)
    stages = build_enhanced_stages(cfg)
    assert len(stages) == 8
    assert stages[0].name == "stage1_v328_ce"


def test_e9_stage0_builder_canonical_shape() -> None:
    stage0 = stage0_pretrained_driving_prior_bootstrap(epochs=42)
    assert isinstance(stage0, CurriculumStageConfig)
    assert stage0.epochs == 42
    assert stage0.seg_loss_kind == "ce"
    assert stage0.use_qat is False
    assert stage0.use_muon is False


# -----------------------------------------------------------------------------
# E10 / E11 — Composition (SABOR + S2SBS sidecars)
# -----------------------------------------------------------------------------


def test_e10_sabor_off_by_default() -> None:
    cfg = EnhancedCurriculumConfig()
    assert cfg.enable_sabor_boundary_only_composition is False


def test_e10_sabor_enabled_shows_in_summary() -> None:
    cfg = EnhancedCurriculumConfig(enable_sabor_boundary_only_composition=True)
    assert "sabor_boundary_only_composition" in cfg.enabled_enhancements()


def test_e11_s2sbs_off_by_default() -> None:
    cfg = EnhancedCurriculumConfig()
    assert cfg.enable_s2sbs_hf_byte_stuffing_composition is False


def test_e11_s2sbs_enabled_shows_in_summary() -> None:
    cfg = EnhancedCurriculumConfig(enable_s2sbs_hf_byte_stuffing_composition=True)
    assert "s2sbs_hf_byte_stuffing_composition" in cfg.enabled_enhancements()


# -----------------------------------------------------------------------------
# Composition contract — all 11 enhancements at once
# -----------------------------------------------------------------------------


def test_all_eleven_enhancements_compose() -> None:
    cfg = EnhancedCurriculumConfig(
        enable_muon_every_stage=True,
        enable_iglt_polish_per_stage=True,
        enable_stage_aware_ternary_qat=True,
        enable_wsd_lr_schedule=True,
        enable_logit_softcap=True,
        enable_cross_block_skip=True,
        enable_atick_redlich_efficient_coding=True,
        enable_full_cpu_validation_gate=True,
        enable_pretrained_driving_prior_bootstrap=True,
        enable_sabor_boundary_only_composition=True,
        enable_s2sbs_hf_byte_stuffing_composition=True,
    )
    assert len(cfg.enabled_enhancements()) == 11
    stages = build_enhanced_stages(cfg)
    assert len(stages) == 9
    for s in stages:
        assert s.extras["research_only"] is True


def test_faithful_vs_enhanced_differ_in_muon_use() -> None:
    faithful_cfg = EnhancedCurriculumConfig.faithful_config()
    enhanced_cfg = EnhancedCurriculumConfig()
    faithful_stages = build_enhanced_stages(faithful_cfg)
    enhanced_stages = build_enhanced_stages(enhanced_cfg)
    assert len(faithful_stages) == 8
    assert len(enhanced_stages) == 9
    faithful_s1 = next(s for s in faithful_stages if s.name == "stage1_v328_ce")
    enhanced_s1 = next(s for s in enhanced_stages if s.name == "stage1_v328_ce")
    assert faithful_s1.use_muon is False
    assert enhanced_s1.use_muon is True


# -----------------------------------------------------------------------------
# Audit + validation
# -----------------------------------------------------------------------------


def test_audit_returns_only_valid_verdicts() -> None:
    cfg = EnhancedCurriculumConfig()
    audit = audit_enhanced_curriculum_against_hnerv_parity_lessons(cfg)
    valid = {"PASS", "N/A", "SUBSTRATE_ENGINEERING_EXCEPTION"}
    for ekey, lessons in audit.items():
        for lid, verdict in lessons.items():
            assert verdict in valid, f"{ekey}.{lid}={verdict!r}"


def test_audit_empty_on_faithful() -> None:
    cfg = EnhancedCurriculumConfig.faithful_config()
    audit = audit_enhanced_curriculum_against_hnerv_parity_lessons(cfg)
    assert audit == {}


def test_validate_enhanced_curriculum_accepts_default() -> None:
    validate_enhanced_curriculum_config(EnhancedCurriculumConfig())
    validate_enhanced_curriculum_config(EnhancedCurriculumConfig.faithful_config())


def test_apply_enhancements_to_stage_preserves_seg_loss_kind() -> None:
    from tac.substrates.pr101_lc_v2_clone.curriculum import stage1_v328_ce

    s = stage1_v328_ce(epochs=10)
    cfg = EnhancedCurriculumConfig()
    s2 = _apply_enhancements_to_stage(s, stage_idx_1based=1, enhanced_cfg=cfg)
    assert s2.seg_loss_kind == s.seg_loss_kind
    assert s2.epochs == s.epochs


def test_build_enhanced_stages_respects_epoch_overrides() -> None:
    cfg = EnhancedCurriculumConfig()
    stages = build_enhanced_stages(
        cfg,
        stage_epoch_overrides={"stage1_v328_ce": 7, "stage8_muon_finetune": 13},
    )
    s1 = next(s for s in stages if s.name == "stage1_v328_ce")
    s8 = next(s for s in stages if s.name == "stage8_muon_finetune")
    assert s1.epochs == 7
    assert s8.epochs == 13
