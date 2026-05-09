"""Tests for Phase 3 joint scorer-renderer-codec scaffold.

These tests cover the SCAFFOLD ONLY — they do not exercise the future
trainer (which is gated on Phase 2 anchor + aaf68f37 verdict + operator
approval per the council deliberation memo).
"""
from __future__ import annotations

import pytest

from tac.phase3 import (
    PHASE3_PREDICTED_BAND_TAG,
    PHASE3_PROVENANCE,
    PHASE3_VERSION,
    JointScorerRendererCodecConfig,
    JointScorerRendererCodecScaffold,
    Phase3DispatchGate,
    Phase3DispatchGateError,
    phase3_distillation_gap_estimate,
    phase3_lagrangian_form,
)
from tac.phase3.inflate import (
    PHASE3_INFLATE_DESIGN_ONLY_NOT_IMPL_MESSAGE,
    phase3_inflate_design_only,
    phase3_inflate_loc_budget,
)


# --------------------------------------------------------------------------
# Provenance / version tests
# --------------------------------------------------------------------------


def test_version_is_scaffold_design_only_marker():
    assert "scaffold" in PHASE3_VERSION.lower()
    assert "design-only" in PHASE3_VERSION.lower()


def test_predicted_band_tag_is_predicted_not_empirical():
    assert PHASE3_PREDICTED_BAND_TAG.startswith("[predicted;")
    assert "Phase 3 council" in PHASE3_PREDICTED_BAND_TAG
    assert "conditional on Phase 2 landing 0.142" in PHASE3_PREDICTED_BAND_TAG


def test_provenance_compliance_tags_include_critical_claude_md_rules():
    tags = PHASE3_PROVENANCE["compliance_tags"]
    assert "ema_0p997_snapshot_restore" in tags
    assert "eval_roundtrip_true" in tags
    assert "no_mps_authoritative" in tags
    assert "scorer_at_eval_frozen_contest" in tags
    assert "score_tag_predicted_only" in tags


def test_provenance_dispatch_readiness_default_blocks_all_dispatch():
    dr = PHASE3_PROVENANCE["dispatch_readiness"]
    assert dr["DISPATCH_READY"] is False
    assert dr["REQUIRES_OPERATOR_APPROVAL"] is True
    assert dr["GATED_ON_PHASE2_ANCHOR"] is True
    assert dr["GATED_ON_AAF68FC7_VERDICT"] is True


def test_provenance_archive_grammar_fields_present():
    fields = PHASE3_PROVENANCE["archive_grammar_fields"]
    for required in (
        "representation_name",
        "target_modes",
        "source_artifact",
        "archive_builder",
        "inflate_consumer",
        "runtime_manifest",
        "changed_payload_paths",
        "old_new_sha256s",
    ):
        assert required in fields, f"missing archive-grammar field: {required}"


def test_provenance_six_hook_wireins_present():
    hooks = PHASE3_PROVENANCE["six_hook_wireins"]
    for required in (
        "sensitivity_map",
        "pareto",
        "bit_allocator",
        "cathedral_autopilot",
        "continual_learning",
        "probe_disambiguator",
    ):
        assert required in hooks, f"missing 6-hook wire-in: {required}"


# --------------------------------------------------------------------------
# Config validation tests
# --------------------------------------------------------------------------


def test_config_defaults_are_sane():
    cfg = JointScorerRendererCodecConfig()
    assert cfg.rate_target_bytes == 100_000.0
    assert cfg.distillation_temperature == 2.0  # Hinton T=2.0 canonical
    assert cfg.distillation_gap_target == 0.03  # 3% per Hinton 2014 §3
    assert cfg.use_t19_adaptive_rho is True
    assert cfg.cross_paradigm_substrate_sources == ("A1",)


def test_config_rejects_non_positive_rate():
    with pytest.raises(ValueError, match="rate_target_bytes"):
        JointScorerRendererCodecConfig(rate_target_bytes=0)
    with pytest.raises(ValueError, match="rate_target_bytes"):
        JointScorerRendererCodecConfig(rate_target_bytes=-1)


def test_config_rejects_non_positive_distillation_temperature():
    with pytest.raises(ValueError, match="distillation_temperature"):
        JointScorerRendererCodecConfig(distillation_temperature=0)


def test_config_rejects_distillation_gap_outside_unit_interval():
    with pytest.raises(ValueError, match="distillation_gap_target"):
        JointScorerRendererCodecConfig(distillation_gap_target=0)
    with pytest.raises(ValueError, match="distillation_gap_target"):
        JointScorerRendererCodecConfig(distillation_gap_target=1)
    with pytest.raises(ValueError, match="distillation_gap_target"):
        JointScorerRendererCodecConfig(distillation_gap_target=-0.1)


def test_config_rejects_invalid_rho_band():
    with pytest.raises(ValueError, match="rho_min"):
        JointScorerRendererCodecConfig(rho_min=0)
    with pytest.raises(ValueError, match="rho_min"):
        JointScorerRendererCodecConfig(rho_min=10, rho_max=5)


def test_config_rejects_empty_substrate_sources():
    with pytest.raises(ValueError, match="cross_paradigm_substrate_sources"):
        JointScorerRendererCodecConfig(cross_paradigm_substrate_sources=("",))


def test_config_accepts_cross_paradigm_multi_substrate():
    cfg = JointScorerRendererCodecConfig(
        cross_paradigm_substrate_sources=("A1", "PR100", "PR101", "PR103")
    )
    assert cfg.cross_paradigm_substrate_sources == ("A1", "PR100", "PR101", "PR103")


# --------------------------------------------------------------------------
# Dispatch gate tests
# --------------------------------------------------------------------------


def test_gate_default_blocks_dispatch():
    with pytest.raises(Phase3DispatchGateError, match="Phase 2 anchor not verified"):
        Phase3DispatchGate().check()


def test_gate_blocks_when_phase2_anchor_too_high():
    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.150,  # > 0.142 target
        phase2_anchor_evidence_path="experiments/results/phase2_dispatch_xyz/eval.json",
        distillation_gap_estimate=0.025,
        distillation_gap_evidence_path="experiments/results/phase3_distill_eval.json",
        operator_approved_gpu_budget_usd=800.0,
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path="aaf_review.md",
        phase3_council_deliberation_path="phase3_council.md",
    )
    with pytest.raises(Phase3DispatchGateError, match="0.142"):
        gate.check()


def test_gate_blocks_when_distillation_gap_too_wide():
    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.140,
        phase2_anchor_evidence_path="ev.json",
        distillation_gap_estimate=0.050,  # > 3% Hinton target
        distillation_gap_evidence_path="distill.json",
        operator_approved_gpu_budget_usd=800.0,
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path="aaf.md",
        phase3_council_deliberation_path="council.md",
    )
    with pytest.raises(Phase3DispatchGateError, match="3%"):
        gate.check()


def test_gate_blocks_when_gpu_budget_outside_band():
    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.140,
        phase2_anchor_evidence_path="ev.json",
        distillation_gap_estimate=0.025,
        distillation_gap_evidence_path="distill.json",
        operator_approved_gpu_budget_usd=2000.0,  # > $1200 cap
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path="aaf.md",
        phase3_council_deliberation_path="council.md",
    )
    with pytest.raises(Phase3DispatchGateError, match=r"\$1200"):
        gate.check()


def test_gate_blocks_when_aaf_verdict_not_clean():
    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.140,
        phase2_anchor_evidence_path="ev.json",
        distillation_gap_estimate=0.025,
        distillation_gap_evidence_path="distill.json",
        operator_approved_gpu_budget_usd=800.0,
        aaf68f37_verdict_clean=False,  # not clean
        aaf68f37_verdict_evidence_path="aaf.md",
        phase3_council_deliberation_path="council.md",
    )
    with pytest.raises(Phase3DispatchGateError, match="aaf68f37"):
        gate.check()


def test_gate_passes_when_all_preconditions_clear():
    gate = Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=0.140,
        phase2_anchor_evidence_path="experiments/results/phase2_xyz/eval.json",
        distillation_gap_estimate=0.025,
        distillation_gap_evidence_path="experiments/results/phase3_distill/eval.json",
        operator_approved_gpu_budget_usd=800.0,
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path="aaf_clean.md",
        phase3_council_deliberation_path="phase3_dispatch_council.md",
    )
    # Should not raise
    gate.check()


# --------------------------------------------------------------------------
# Scaffold tests
# --------------------------------------------------------------------------


def test_scaffold_can_instantiate_for_unit_testing():
    """Scaffold construction does NOT enforce gate (so tests can exercise API)."""
    cfg = JointScorerRendererCodecConfig()
    gate = Phase3DispatchGate()  # default — would block dispatch
    scaffold = JointScorerRendererCodecScaffold(config=cfg, gate=gate)
    assert scaffold.config == cfg
    assert scaffold.gate == gate


def test_scaffold_emit_build_manifest_stub_has_required_fields():
    cfg = JointScorerRendererCodecConfig()
    gate = Phase3DispatchGate()
    scaffold = JointScorerRendererCodecScaffold(config=cfg, gate=gate)
    stub = scaffold.emit_build_manifest_stub()

    for required in (
        "phase",
        "lane_id",
        "config",
        "lagrangian_form",
        "distillation_gap_estimate_plan",
        "council_memo_path",
        "dispatch_status",
        "predicted_score_band",
        "dispatch_ready",
        "requires_operator_approval",
        "build_manifest_schema_version",
    ):
        assert required in stub, f"build manifest stub missing field: {required}"

    assert stub["dispatch_ready"] is False
    assert stub["requires_operator_approval"] is True
    assert stub["lane_id"] == "lane_phase3_joint_scorer_renderer_codec"
    assert "predicted; Phase 3 council" in stub["predicted_score_band"]


# --------------------------------------------------------------------------
# Lagrangian form tests
# --------------------------------------------------------------------------


def test_phase3_lagrangian_form_returns_string_dict():
    form = phase3_lagrangian_form()
    assert isinstance(form, dict)
    for required in ("name", "form", "theorems_invoked", "compliance_tags"):
        assert required in form
        assert isinstance(form[required], str)


def test_phase3_lagrangian_form_invokes_canonical_theorems():
    form = phase3_lagrangian_form()
    text = form["theorems_invoked"]
    assert "Tishby 1999" in text
    assert "Berger 1971" in text
    assert "Hinton 2014" in text
    assert "Boyd 2011" in text
    assert "Ballé 2018" in text


# --------------------------------------------------------------------------
# Distillation-gap-estimate plan tests
# --------------------------------------------------------------------------


def test_distillation_gap_estimate_plan_includes_method_and_target():
    cfg = JointScorerRendererCodecConfig()
    plan = phase3_distillation_gap_estimate(cfg)
    assert "Hinton" in plan["method"]
    assert plan["temperature"] == 2.0
    assert plan["target_gap"] == 0.03
    assert "measurement_protocol" in plan
    assert "evidence_artifact_path_template" in plan
    assert "blocker_class_if_exceeded" in plan


def test_distillation_gap_estimate_plan_evidence_path_avoids_tmp():
    cfg = JointScorerRendererCodecConfig()
    plan = phase3_distillation_gap_estimate(cfg)
    assert "/tmp/" not in plan["evidence_artifact_path_template"]


# --------------------------------------------------------------------------
# Inflate scaffold tests
# --------------------------------------------------------------------------


def test_inflate_design_only_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="DESIGN-ONLY"):
        phase3_inflate_design_only(
            "experiments/results/test_archive.zip",
            "experiments/results/test_output/",
        )


def test_inflate_refuses_mps_device():
    with pytest.raises(ValueError, match="MPS"):
        phase3_inflate_design_only(
            "experiments/results/x.zip",
            "experiments/results/y/",
            device="mps",
        )


def test_inflate_refuses_tmp_paths():
    with pytest.raises(ValueError, match="/tmp"):
        phase3_inflate_design_only(
            "/tmp/foo.zip",
            "experiments/results/y/",
        )
    with pytest.raises(ValueError, match="/tmp"):
        phase3_inflate_design_only(
            "experiments/results/x.zip",
            "/tmp/output/",
        )


def test_inflate_loc_budget_under_200():
    """CLAUDE.md HNeRV parity discipline L4: inflate.py ≤ 200 LOC."""
    loc = phase3_inflate_loc_budget()
    assert 0 < loc <= 200, f"inflate.py LOC = {loc}, must be ≤ 200"


def test_inflate_not_impl_message_references_council_memo():
    """The NotImplementedError message must reference the council memo so
    operators discover the dispatch decision tree.
    """
    assert (
        "fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md"
        in PHASE3_INFLATE_DESIGN_ONLY_NOT_IMPL_MESSAGE
    )
