# SPDX-License-Identifier: MIT
"""Adversarial tests for the frozen-scorer gradient replacement contract."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import tac.witness_dsl.scorer_gradient_policy as policy_module
from tac.boundary_math.segnet_gradient_replacement import (
    array_content_sha256,
    evaluate_teacher_step,
)
from tac.witness_dsl.scorer_gradient_policy import (
    ProviderCostateEvaluation,
    ProviderCustody,
    ScorerGradientPolicy,
    ScorerObjectiveContext,
    TeacherGradientObservation,
    compile_scorer_gradient_policy,
)

SCORER_SHA = "a" * 64
PREPROCESS_SHA = "b" * 64
RECEIVER_R_SHA = "c" * 64
GT_TARGETS_SHA = "d" * 64


def _context(**overrides) -> ScorerObjectiveContext:
    values = {
        "scorer_sha256": SCORER_SHA,
        "preprocess_sha256": PREPROCESS_SHA,
        "receiver_r_sha256": RECEIVER_R_SHA,
        "gt_targets_sha256": GT_TARGETS_SHA,
        "pair_index": 17,
        "loss_name": "tau_softplus_margin_ce",
        "loss_parameters": {"tau": 0.2, "ce_weight": 1.0},
        "stage_name": "seg_margin",
        "stage_parameters": {"epoch": 12, "roundtrip_ste": True},
    }
    values.update(overrides)
    return ScorerObjectiveContext(**values)


def _custody(tmp_path: Path, *, kind: str = "checkpoint") -> ProviderCustody:
    artifact = tmp_path / f"provider.{kind}"
    artifact.write_bytes(b"frozen provider bytes\n")
    return ProviderCustody(
        kind=kind,
        path=str(artifact),
        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
        size_bytes=artifact.stat().st_size,
    )


def _policy(tmp_path: Path, *, mode: str = "periodic_costate", **overrides):
    context = overrides.pop("objective_context", _context())
    values = {
        "mode": mode,
        "refresh_interval_steps": 4,
        "max_staleness_steps": 3,
        "scorer_fingerprint": context.scorer_sha256,
        "objective_context": context,
        "objective_context_fingerprint": context.fingerprint(),
        "min_costate_cosine": 0.99,
        "max_costate_relative_l2": 0.05,
        "min_costate_norm_ratio": 0.95,
        "max_costate_norm_ratio": 1.05,
        "max_teacher_loss_regret": 0.11,
        "provider_custody": _custody(
            tmp_path, kind="cache" if mode == "trusted_jacobian_cache" else "checkpoint"
        ),
    }
    if mode == "trusted_jacobian_cache":
        values["max_frame_relative_l2"] = 0.02
    values.update(overrides)
    return ScorerGradientPolicy(**values)


def _bound_step_check(
    *,
    objective_fingerprint: str,
    anchor_frame: np.ndarray,
    custody_sha: str,
    evaluated_at_step: int,
    candidate_loss: float = 0.8,
):
    return evaluate_teacher_step(
        current_loss=1.0,
        candidate_loss=candidate_loss,
        reference_loss=0.7,
        objective_context_fingerprint=objective_fingerprint,
        anchor_frame=anchor_frame,
        candidate_frame=np.full((2, 2), 0.8),
        reference_frame=np.full((2, 2), 0.7),
        provider_custody_sha256=custody_sha,
        evaluated_at_step=evaluated_at_step,
    )


def _evidence(
    compiled,
    *,
    teacher_costate=None,
    provider_anchor_costate=None,
    current_costate=None,
    anchor_frame=None,
    current_frame=None,
    measured_at_step: int = 0,
    current_step: int = 1,
):
    policy = compiled.source
    assert policy.provider_custody is not None
    assert policy.objective_context_fingerprint is not None
    anchor = (
        np.ones((2, 2), dtype=np.float64)
        if anchor_frame is None
        else np.asarray(anchor_frame)
    )
    current = anchor * 1.01 if current_frame is None else np.asarray(current_frame)
    teacher = (
        np.array([[1.0, -2.0], [0.5, 3.0]])
        if teacher_costate is None
        else np.asarray(teacher_costate)
    )
    provider_anchor = (
        teacher.copy()
        if provider_anchor_costate is None
        else np.asarray(provider_anchor_costate)
    )
    current_value = (
        teacher.copy() if current_costate is None else np.asarray(current_costate)
    )
    anchor_sha = array_content_sha256(anchor)
    current_sha = array_content_sha256(current)
    anchor_evaluation = ProviderCostateEvaluation(
        costate=provider_anchor,
        frame_sha256=anchor_sha,
        objective_context_fingerprint=policy.objective_context_fingerprint,
        provider_custody_sha256=policy.provider_custody.sha256,
        evaluated_at_step=measured_at_step,
    )
    observation = TeacherGradientObservation(
        teacher_costate_at_anchor=teacher,
        provider_costate_at_anchor=anchor_evaluation,
        anchor_frame=anchor,
        anchor_frame_sha256=anchor_sha,
        measured_at_step=measured_at_step,
        objective_context_fingerprint=policy.objective_context_fingerprint,
        scorer_fingerprint=policy.scorer_fingerprint or "",
        teacher_step_check=_bound_step_check(
            objective_fingerprint=policy.objective_context_fingerprint,
            anchor_frame=anchor,
            custody_sha=policy.provider_custody.sha256,
            evaluated_at_step=measured_at_step,
        ),
    )
    current_evaluation = ProviderCostateEvaluation(
        costate=current_value,
        frame_sha256=current_sha,
        objective_context_fingerprint=policy.objective_context_fingerprint,
        provider_custody_sha256=policy.provider_custody.sha256,
        evaluated_at_step=current_step,
    )
    return {
        "current_provider_evaluation": current_evaluation,
        "teacher_observation": observation,
        "current_frame": current,
        "current_frame_sha256": current_sha,
        "current_objective_context_fingerprint": policy.objective_context_fingerprint,
        "current_step": current_step,
    }


def test_full_teacher_is_explicit_and_replacement_fields_are_not_dormant() -> None:
    compiled = ScorerGradientPolicy(mode="full_teacher").compile()
    decision = compiled.decide(
        current_provider_evaluation=None,
        teacher_observation=None,
        current_frame=None,
        current_frame_sha256=None,
        current_objective_context_fingerprint=None,
        current_step=0,
    )
    assert decision.admitted
    assert decision.selected_mode == "full_teacher"
    with pytest.raises(ValidationError, match="dormant replacement fields"):
        ScorerGradientPolicy(mode="full_teacher", refresh_interval_steps=1)


def test_replacement_policy_requires_complete_objective_context(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="requires explicit fields"):
        ScorerGradientPolicy(mode="periodic_costate")
    context = _context()
    with pytest.raises(ValidationError, match="does not match"):
        _policy(tmp_path, objective_context=context, objective_context_fingerprint="e" * 64)


def test_compile_full_hashes_and_nonrefresh_decisions_only_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_hash = policy_module._sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return real_hash(path)

    monkeypatch.setattr(policy_module, "_sha256_file", counted)
    compiled = _policy(tmp_path).compile()
    assert calls == 1
    decision = compiled.decide(**_evidence(compiled, measured_at_step=0, current_step=1))
    assert decision.admitted
    assert calls == 1, "non-refresh decision must use stat, not full SHA"
    refresh = compiled.decide(
        **_evidence(
            compiled,
            measured_at_step=2,
            current_step=2,
            current_frame=np.ones((2, 2), dtype=np.float64),
        )
    )
    assert refresh.admitted
    assert calls == 2, "teacher refresh must re-verify the full provider SHA"


def test_refresh_refuses_current_costate_not_validated_at_teacher_anchor(
    tmp_path: Path,
) -> None:
    compiled = _policy(tmp_path).compile()
    wrong_sign = -np.array([[1.0, -2.0], [0.5, 3.0]])
    decision = compiled.decide(
        **_evidence(
            compiled,
            current_costate=wrong_sign,
            current_frame=np.ones((2, 2), dtype=np.float64),
            measured_at_step=2,
            current_step=2,
        )
    )
    assert decision.fallback_to_full_teacher
    assert any(
        "refresh current provider costate content differs" in reason
        for reason in decision.reasons
    )


def test_refresh_refuses_anchor_current_cross_frame_evidence(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    decision = compiled.decide(
        **_evidence(compiled, measured_at_step=2, current_step=2)
    )
    assert decision.fallback_to_full_teacher
    assert any("refresh anchor frame" in reason for reason in decision.reasons)


@pytest.mark.parametrize("parameter_field", ["loss_parameters", "stage_parameters"])
def test_objective_parameter_mutation_after_compile_is_refused(
    tmp_path: Path, parameter_field: str
) -> None:
    policy = _policy(tmp_path)
    compiled = policy.compile()
    assert policy.objective_context is not None
    parameters = getattr(policy.objective_context, parameter_field)
    parameters["post_compile_mutation"] = 1.0
    decision = compiled.decide(**_evidence(compiled))
    assert decision.fallback_to_full_teacher
    assert any("mutated after compilation" in reason for reason in decision.reasons)


def test_provider_mutation_after_compile_falls_back_without_rehash(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    assert compiled.source.provider_custody is not None
    path = Path(compiled.source.provider_custody.path)
    original = path.read_bytes()
    path.write_bytes(b"X" * len(original))
    decision = compiled.decide(**_evidence(compiled))
    assert decision.fallback_to_full_teacher
    assert any("stat fingerprint changed" in reason for reason in decision.reasons)


def test_exact_global_candidate_is_admitted_and_wrong_costate_falls_back(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    admitted = compiled.decide(**_evidence(compiled))
    assert admitted.admitted
    assert admitted.global_costate_metrics is not None
    assert admitted.global_costate_metrics.cosine_similarity == pytest.approx(1.0)

    wrong = compiled.decide(
        **_evidence(compiled, provider_anchor_costate=np.array([[-1.0, 2.0], [-0.5, -3.0]]))
    )
    assert wrong.fallback_to_full_teacher
    assert any("global costate cosine" in reason for reason in wrong.reasons)


def test_mask_exact_offmask_reversed_cannot_launder_global_failure(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    teacher = np.array([[1.0, 2.0], [3.0, 4.0]])
    candidate = np.array([[1.0, 2.0], [-3.0, -4.0]])
    annulus = np.array([[True, True], [False, False]])
    decision = compiled.decide(
        **_evidence(
            compiled,
            teacher_costate=teacher,
            provider_anchor_costate=candidate,
            current_costate=candidate,
        ),
        annulus_mask=annulus,
    )
    assert decision.fallback_to_full_teacher
    assert decision.annulus_costate_metrics is not None
    assert decision.annulus_costate_metrics.cosine_similarity == pytest.approx(1.0)
    assert decision.global_costate_metrics is not None
    assert decision.global_costate_metrics.cosine_similarity < 0.99
    assert any("global costate" in reason for reason in decision.reasons)


def test_cross_objective_evidence_is_refused(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    evidence = _evidence(compiled)
    other_context = _context(pair_index=18)
    other_fingerprint = other_context.fingerprint()
    evidence["current_objective_context_fingerprint"] = other_fingerprint
    evidence["current_provider_evaluation"] = replace(
        evidence["current_provider_evaluation"],
        objective_context_fingerprint=other_fingerprint,
    )
    decision = compiled.decide(**evidence)
    assert decision.fallback_to_full_teacher
    assert any("current objective/context" in reason for reason in decision.reasons)


def test_cross_frame_current_costate_is_refused(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    evidence = _evidence(compiled)
    anchor_hash = evidence["teacher_observation"].anchor_frame_sha256
    evidence["current_provider_evaluation"] = replace(
        evidence["current_provider_evaluation"], frame_sha256=anchor_hash
    )
    decision = compiled.decide(**evidence)
    assert decision.fallback_to_full_teacher
    assert any("current provider evaluation frame hash" in reason for reason in decision.reasons)


def test_replayed_provider_or_teacher_check_is_refused(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    evidence = _evidence(compiled)
    evidence["current_provider_evaluation"] = replace(
        evidence["current_provider_evaluation"], evaluated_at_step=0
    )
    replayed_provider = compiled.decide(**evidence)
    assert replayed_provider.fallback_to_full_teacher
    assert any("replayed step" in reason for reason in replayed_provider.reasons)

    evidence = _evidence(compiled)
    observation = evidence["teacher_observation"]
    evidence["teacher_observation"] = replace(
        observation,
        provider_costate_at_anchor=replace(
            observation.provider_costate_at_anchor, evaluated_at_step=99
        ),
    )
    replayed_anchor_provider = compiled.decide(**evidence)
    assert replayed_anchor_provider.fallback_to_full_teacher
    assert any(
        "provider anchor evaluation replayed step" in reason
        for reason in replayed_anchor_provider.reasons
    )

    evidence = _evidence(compiled)
    observation = evidence["teacher_observation"]
    evidence["teacher_observation"] = replace(
        observation,
        teacher_step_check=replace(observation.teacher_step_check, evaluated_at_step=99),
    )
    replayed_check = compiled.decide(**evidence)
    assert replayed_check.fallback_to_full_teacher
    assert any("step check was replayed" in reason for reason in replayed_check.reasons)


@pytest.mark.parametrize("binding", ["objective", "anchor", "custody", "candidate"])
def test_cross_bound_teacher_step_check_is_refused(tmp_path: Path, binding: str) -> None:
    compiled = _policy(tmp_path).compile()
    evidence = _evidence(compiled)
    observation = evidence["teacher_observation"]
    changes = {
        "objective": {"objective_context_fingerprint": "e" * 64},
        "anchor": {"anchor_frame_sha256": "e" * 64},
        "custody": {"provider_custody_sha256": "e" * 64},
        "candidate": {"candidate_frame_sha256": "e" * 64},
    }[binding]
    evidence["teacher_observation"] = replace(
        observation,
        teacher_step_check=replace(observation.teacher_step_check, **changes),
    )
    decision = compiled.decide(**evidence)
    assert decision.fallback_to_full_teacher
    assert any("teacher step check" in reason for reason in decision.reasons)


def test_nonfinite_or_shape_mismatched_current_costate_is_refused(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    nonfinite = compiled.decide(
        **_evidence(compiled, current_costate=np.full((2, 2), np.nan))
    )
    assert nonfinite.fallback_to_full_teacher
    assert any("nonfinite" in reason for reason in nonfinite.reasons)
    wrong_shape = compiled.decide(
        **_evidence(compiled, current_costate=np.ones((4,)))
    )
    assert wrong_shape.fallback_to_full_teacher
    assert any("shape mismatch" in reason for reason in wrong_shape.reasons)


def test_stale_or_changed_teacher_anchor_is_refused(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    stale = compiled.decide(**_evidence(compiled, measured_at_step=0, current_step=4))
    assert stale.fallback_to_full_teacher
    assert any("refresh is due" in reason for reason in stale.reasons)
    evidence = _evidence(compiled)
    observation = evidence["teacher_observation"]
    evidence["teacher_observation"] = replace(
        observation, scorer_fingerprint="e" * 64
    )
    changed = compiled.decide(**evidence)
    assert changed.fallback_to_full_teacher
    assert any("scorer fingerprint" in reason for reason in changed.reasons)


def test_teacher_step_regret_gate_is_required(tmp_path: Path) -> None:
    compiled = _policy(tmp_path).compile()
    evidence = _evidence(compiled)
    observation = evidence["teacher_observation"]
    assert compiled.source.provider_custody is not None
    evidence["teacher_observation"] = replace(
        observation,
        teacher_step_check=_bound_step_check(
            objective_fingerprint=observation.objective_context_fingerprint,
            anchor_frame=observation.anchor_frame,
            custody_sha=compiled.source.provider_custody.sha256,
            evaluated_at_step=observation.measured_at_step,
            candidate_loss=0.99,
        ),
    )
    decision = compiled.decide(**evidence)
    assert decision.fallback_to_full_teacher
    assert any("one-step check failed" in reason for reason in decision.reasons)


def test_trusted_cache_additionally_requires_frame_trust_radius(tmp_path: Path) -> None:
    compiled = _policy(tmp_path, mode="trusted_jacobian_cache").compile()
    assert compiled.decide(**_evidence(compiled)).admitted
    outside = compiled.decide(
        **_evidence(compiled, current_frame=np.ones((2, 2)) * 1.03)
    )
    assert outside.fallback_to_full_teacher
    assert any("left cache trust radius" in reason for reason in outside.reasons)


def test_mapping_compile_rejects_invented_fields(tmp_path: Path) -> None:
    payload = _policy(tmp_path).model_dump()
    payload["invented_trainer_flag"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        compile_scorer_gradient_policy(payload)
