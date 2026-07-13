# SPDX-License-Identifier: MIT
"""Fail-closed tests for the mechanism-owning INSTANT provider."""

from __future__ import annotations

import contextlib
import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

import tac.boundary_math.instant_projected_adjoint as instant
import tac.boundary_math.segnet_gradient_replacement as replacement
from tac.boundary_math.instant_projected_adjoint import (
    InstantMechanismProof,
    InstantProjectedAdjointProvider,
    InstantProviderCostateEvaluation,
    calibrate_adaptive_projector_numpy,
    save_calibration,
)
from tac.boundary_math.segnet_gradient_replacement import array_content_sha256, evaluate_teacher_step
from tac.witness_dsl.scorer_gradient_policy import (
    InstantAdmissionEconomics,
    ProviderCostateEvaluation,
    ProviderCustody,
    ScorerGradientPolicy,
    ScorerObjectiveContext,
    TeacherGradientObservation,
)


class _TinyFrozenScorer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pointwise = torch.nn.Conv2d(1, 2, 1)
        self.spatial = torch.nn.Conv2d(2, 2, 3, padding=1, bias=False)
        with torch.no_grad():
            self.pointwise.weight.copy_(torch.tensor([[[[0.75]]], [[[-0.5]]]]))
            self.pointwise.bias.copy_(torch.tensor([0.1, -0.2]))
            self.spatial.weight.copy_(torch.linspace(-0.2, 0.3, self.spatial.weight.numel()).reshape_as(self.spatial.weight))
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.spatial(torch.tanh(self.pointwise(value)))


def _objective(logits: torch.Tensor, context: ScorerObjectiveContext) -> torch.Tensor:
    assert context.loss_name == "mean_squared_probe_relaxation"
    return logits.square().mean()


@dataclass(frozen=True)
class _Fixture:
    scorer: _TinyFrozenScorer
    context: ScorerObjectiveContext
    custody: ProviderCustody
    provider: InstantProjectedAdjointProvider
    policy: ScorerGradientPolicy
    manifest_path: Path
    frame: torch.Tensor


def _write_manifest(tmp_path: Path, scorer: _TinyFrozenScorer, *, wrong_identity: bool = False) -> Path:
    cotangents = np.array(
        [
            [[[[1.0, 0.2], [-0.3, 0.7]], [[0.4, -0.5], [0.9, 0.1]]]],
            [[[[0.8, -0.4], [0.6, 0.3]], [[-0.2, 0.7], [0.5, -0.6]]]],
        ],
        dtype=np.float64,
    )
    calibration = calibrate_adaptive_projector_numpy(cotangents, energy_target=0.95, oversampling=0)
    calibration_path = tmp_path / "pointwise.npz"
    save_calibration(calibration_path, calibration)
    identity = InstantProjectedAdjointProvider.eligible_layer_identities(scorer)["pointwise"]
    manifest = {
        "schema": "tac.instant_projected_adjoint_provider.v1",
        "layers": {
            "pointwise": {
                "calibration_path": calibration_path.name,
                "calibration_sha256": hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
                "module_identity_sha256": ("0" * 64 if wrong_identity else identity),
            }
        },
    }
    path = tmp_path / "instant_provider_manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return path


def _fixture(tmp_path: Path) -> _Fixture:
    scorer = _TinyFrozenScorer()
    scorer_sha = InstantProjectedAdjointProvider.scorer_state_sha256(scorer)
    context = ScorerObjectiveContext(
        scorer_sha256=scorer_sha,
        preprocess_sha256="b" * 64,
        receiver_r_sha256="c" * 64,
        gt_targets_sha256="d" * 64,
        pair_index=0,
        loss_name="mean_squared_probe_relaxation",
        loss_parameters={"reduction": "mean", "stage_native_objective": False},
        stage_name="tau_softplus",
        stage_parameters={"epoch": 899, "probe_relaxation": "MSE"},
    )
    manifest_path = _write_manifest(tmp_path, scorer)
    custody = ProviderCustody(
        kind="cache",
        path=str(manifest_path),
        sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        size_bytes=manifest_path.stat().st_size,
    )
    provider = InstantProjectedAdjointProvider(
        scorer=scorer,
        objective=_objective,
        objective_context=context,
        expected_objective_context_fingerprint=context.fingerprint(),
        provider_manifest_path=manifest_path,
        expected_provider_manifest_sha256=custody.sha256,
        expected_provider_manifest_size_bytes=custody.size_bytes,
    )
    policy = ScorerGradientPolicy(
        mode="instant_projected_adjoint",
        refresh_interval_steps=4,
        max_staleness_steps=3,
        scorer_fingerprint=context.scorer_sha256,
        objective_context=context,
        objective_context_fingerprint=context.fingerprint(),
        provider_custody=custody,
        instant_admission_economics=InstantAdmissionEconomics(
            exact_seconds=1.0,
            approximate_seconds=0.1,
            projected_candidate_validation_seconds=0.1,
        ),
    )
    frame = torch.tensor([[[[0.2, 0.7], [-0.3, 0.5]]]], dtype=torch.float32)
    return _Fixture(scorer, context, custody, provider, policy, manifest_path, frame)


def _evidence(fixture: _Fixture, *, candidate_loss: float = 0.8) -> dict:
    evaluation = fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)
    teacher = evaluation.costate * 1.01
    step_check = evaluate_teacher_step(
        current_loss=1.0,
        candidate_loss=candidate_loss,
        reference_loss=0.7,
        objective_context_fingerprint=fixture.context.fingerprint(),
        anchor_frame=fixture.frame,
        candidate_frame=fixture.frame * 0.8,
        reference_frame=fixture.frame * 0.7,
        provider_custody_sha256=fixture.custody.sha256,
        evaluated_at_step=0,
    )
    observation = TeacherGradientObservation(
        teacher_costate_at_anchor=teacher,
        provider_costate_at_anchor=evaluation,
        anchor_frame=fixture.frame,
        anchor_frame_sha256=array_content_sha256(fixture.frame),
        measured_at_step=0,
        objective_context_fingerprint=fixture.context.fingerprint(),
        scorer_fingerprint=fixture.context.scorer_sha256,
        teacher_step_check=step_check,
        renderer_gradient_cosine=0.5,
        renderer_gradient_cosine_floor=1.0e-12,
    )
    return {
        "current_provider_evaluation": evaluation,
        "teacher_observation": observation,
        "current_frame": fixture.frame,
        "current_frame_sha256": array_content_sha256(fixture.frame),
        "current_objective_context_fingerprint": fixture.context.fingerprint(),
        "current_step": 0,
    }


def test_provider_derives_costate_and_digest_from_owned_execution(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evaluation = fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=7)
    assert isinstance(evaluation, InstantProviderCostateEvaluation)
    assert evaluation.frame_sha256 == array_content_sha256(fixture.frame)
    assert evaluation.provider_custody_sha256 == fixture.custody.sha256
    assert evaluation.mechanism_proof is not None
    assert evaluation.derivation_digest == evaluation.mechanism_proof.derivation_digest
    assert evaluation.derivation_digest == evaluation.mechanism_proof.recompute_derivation_digest()
    assert evaluation.mechanism_proof.exact_forward_equal
    assert evaluation.mechanism_proof.backward_calls == (("pointwise", 1),)
    assert torch.isfinite(evaluation.costate).all()


def test_public_provider_has_no_raw_costate_binding_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    assert not hasattr(replacement, "instant_projected_adjoint_provider_evaluation")
    with pytest.raises(TypeError, match="costate"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0, costate=torch.ones_like(fixture.frame))


def test_provider_refuses_broken_exact_forward(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)

    @contextlib.contextmanager
    def broken_context(bank, proofs):
        del bank, proofs
        original = fixture.scorer.pointwise.forward

        def broken(value):
            return original(value) + 1.0

        fixture.scorer.pointwise.forward = broken
        try:
            yield
        finally:
            fixture.scorer.pointwise.forward = original

    monkeypatch.setattr(fixture.provider, "_projected_context", broken_context)
    with pytest.raises(ValueError, match="execution surface has instance overrides"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)


def test_provider_refuses_missing_mechanism_proof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)

    @contextlib.contextmanager
    def proofless_context(bank, proofs):
        del bank, proofs
        yield

    monkeypatch.setattr(fixture.provider, "_projected_context", proofless_context)
    with pytest.raises(ValueError, match="execution surface has instance overrides"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)


def test_provider_refuses_dense_primitive_with_forged_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    def dense_with_forged_counters(value, weight, bias, calibration, *, proof=None):
        if proof is not None:
            proof.backward_calls += 1
            proof.channel_axis_calls += 1
        return torch.nn.functional.conv2d(value, weight, bias)

    monkeypatch.setattr(instant, "instant_pointwise_conv2d", dense_with_forged_counters)
    with pytest.raises(ValueError, match="projected primitive identity changed"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)


def test_exact_forward_and_projection_proof_guards_have_independent_canaries() -> None:
    exact = torch.tensor([1.0])
    instant._require_exact_forward_equal(exact.clone(), exact)
    with pytest.raises(ValueError, match="changed the exact scorer forward"):
        instant._require_exact_forward_equal(exact + 1.0, exact)

    valid = instant.ProjectionProof(backward_calls=1, channel_axis_calls=1)
    instant._require_projection_proofs({"pointwise": valid})
    with pytest.raises(ValueError, match="mechanism proof missing"):
        instant._require_projection_proofs({"pointwise": instant.ProjectionProof()})


def test_provider_refuses_dense_context_override_with_forged_counters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)

    @contextlib.contextmanager
    def dense_context(bank, proofs):
        del bank
        original = fixture.scorer.pointwise.forward
        proof = proofs["pointwise"]

        def dense(value):
            proof.backward_calls += 1
            proof.channel_axis_calls += 1
            return original(value)

        fixture.scorer.pointwise.forward = dense
        try:
            yield
        finally:
            fixture.scorer.pointwise.forward = original

    monkeypatch.setattr(fixture.provider, "_projected_context", dense_context)
    with pytest.raises(ValueError, match="execution surface has instance overrides"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)


def test_provider_refuses_other_instance_execution_method_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    monkeypatch.setattr(fixture.provider, "_load_verified_bank", lambda: ({}, {}, "0" * 64))
    with pytest.raises(ValueError, match="_load_verified_bank"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)


def test_provider_refuses_wrong_bank_and_mutated_custody(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    wrong_dir = tmp_path / "wrong"
    wrong_dir.mkdir()
    wrong_manifest = _write_manifest(wrong_dir, fixture.scorer, wrong_identity=True)
    with pytest.raises(ValueError, match="eligible-layer identity mismatch"):
        InstantProjectedAdjointProvider(
            scorer=fixture.scorer,
            objective=_objective,
            objective_context=fixture.context,
            expected_objective_context_fingerprint=fixture.context.fingerprint(),
            provider_manifest_path=wrong_manifest,
            expected_provider_manifest_sha256=hashlib.sha256(wrong_manifest.read_bytes()).hexdigest(),
            expected_provider_manifest_size_bytes=wrong_manifest.stat().st_size,
        )
    fixture.manifest_path.write_bytes(b"X" * fixture.custody.size_bytes)
    with pytest.raises(ValueError, match="manifest SHA-256"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)


def test_provider_refuses_scorer_objective_frame_and_step_drift(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with torch.no_grad():
        fixture.scorer.pointwise.weight.add_(1.0)
    with pytest.raises(ValueError, match="scorer state changed"):
        fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=0)
    for bad_step in (True, -1, 1.5):
        with pytest.raises(ValueError, match="evaluated_at_step"):
            fixture.provider.evaluate(frame=fixture.frame, evaluated_at_step=bad_step)
    with pytest.raises(ValueError, match="finite"):
        fixture.provider.evaluate(frame=torch.full_like(fixture.frame, float("nan")), evaluated_at_step=0)


def test_policy_admits_only_internally_derived_current_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence(fixture)
    decision = fixture.policy.compile().decide(**evidence)
    assert decision.admitted
    assert decision.selected_mode == "instant_projected_adjoint"
    evaluation = evidence["current_provider_evaluation"]
    tampered = replace(evaluation, costate=torch.full_like(fixture.frame, float("inf")))
    decision = fixture.policy.compile().decide(**dict(evidence, current_provider_evaluation=tampered))
    assert decision.fallback_to_full_teacher
    assert any("nonfinite" in reason for reason in decision.reasons)


def test_policy_refuses_generic_or_digest_tampered_instant_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence(fixture)
    evaluation = evidence["current_provider_evaluation"]
    generic = ProviderCostateEvaluation(
        costate=evaluation.costate,
        frame_sha256=evaluation.frame_sha256,
        objective_context_fingerprint=evaluation.objective_context_fingerprint,
        provider_custody_sha256=evaluation.provider_custody_sha256,
        evaluated_at_step=evaluation.evaluated_at_step,
    )
    decision = fixture.policy.compile().decide(
        **dict(evidence, current_provider_evaluation=generic)
    )
    assert decision.fallback_to_full_teacher
    assert any("mechanism-owning provider" in reason for reason in decision.reasons)

    tampered = replace(evaluation, costate=evaluation.costate * 0.9)
    decision = fixture.policy.compile().decide(
        **dict(evidence, current_provider_evaluation=tampered)
    )
    assert decision.fallback_to_full_teacher
    assert any("proof costate SHA-256 mismatch" in reason for reason in decision.reasons)


def test_policy_refuses_self_consistent_forged_specialized_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence(fixture)
    legitimate = evidence["current_provider_evaluation"]
    assert legitimate.mechanism_proof is not None
    arbitrary_costate = torch.full_like(fixture.frame, 123.0)
    arbitrary_sha = array_content_sha256(arbitrary_costate)
    forged_proof = replace(legitimate.mechanism_proof, costate_sha256=arbitrary_sha)
    forged_proof = replace(
        forged_proof,
        derivation_digest=forged_proof.recompute_derivation_digest(),
    )
    forged = replace(
        legitimate,
        costate=arbitrary_costate,
        derivation_digest=forged_proof.derivation_digest,
        mechanism_proof=forged_proof,
    )
    observation = evidence["teacher_observation"]
    decision = fixture.policy.compile().decide(
        **dict(
            evidence,
            current_provider_evaluation=forged,
            teacher_observation=replace(observation, provider_costate_at_anchor=forged),
        )
    )
    assert decision.fallback_to_full_teacher
    assert any("capability does not bind" in reason for reason in decision.reasons)

    with pytest.raises(TypeError, match="provider-issued only"):
        type(legitimate.execution_capability)()


def test_policy_refuses_reconstructed_evidence_without_capability(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence(fixture)
    legitimate = evidence["current_provider_evaluation"]
    reconstructed = InstantProviderCostateEvaluation(
        costate=legitimate.costate,
        frame_sha256=legitimate.frame_sha256,
        objective_context_fingerprint=legitimate.objective_context_fingerprint,
        provider_custody_sha256=legitimate.provider_custody_sha256,
        evaluated_at_step=legitimate.evaluated_at_step,
        derivation_digest=legitimate.derivation_digest,
        mechanism_proof=InstantMechanismProof(**vars(legitimate.mechanism_proof)),
    )
    decision = fixture.policy.compile().decide(
        **dict(evidence, current_provider_evaluation=reconstructed)
    )
    assert decision.fallback_to_full_teacher
    assert any("execution capability is missing" in reason for reason in decision.reasons)


def test_policy_refuses_in_place_mutation_of_issued_costate(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence(fixture)
    evaluation = evidence["current_provider_evaluation"]
    evaluation.costate.add_(1.0)
    decision = fixture.policy.compile().decide(**evidence)
    assert decision.fallback_to_full_teacher
    assert any("capability content mismatch" in reason for reason in decision.reasons)


def test_policy_refuses_proofless_instant_anchor_evidence(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    evidence = _evidence(fixture)
    observation = evidence["teacher_observation"]
    proofless = replace(observation.provider_costate_at_anchor, mechanism_proof=None)
    decision = fixture.policy.compile().decide(
        **dict(
            evidence,
            teacher_observation=replace(
                observation,
                provider_costate_at_anchor=proofless,
            ),
        )
    )
    assert decision.fallback_to_full_teacher
    assert any("mechanism proof is missing" in reason for reason in decision.reasons)


def test_policy_still_rejects_universal_thresholds_and_wrong_custody(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    values = fixture.policy.model_dump()
    values["min_costate_cosine"] = 0.9
    with pytest.raises(ValidationError, match="rejects universal"):
        ScorerGradientPolicy(**values)
    values.pop("min_costate_cosine")
    values["provider_custody"] = fixture.custody.model_copy(update={"kind": "checkpoint"}).model_dump()
    with pytest.raises(ValidationError, match=r"provider_custody\.kind"):
        ScorerGradientPolicy(**values)


def test_policy_refuses_missing_or_decisive_no_go_instant_economics(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    values = fixture.policy.model_dump()
    values.pop("instant_admission_economics")
    with pytest.raises(ValidationError, match="requires instant_admission_economics"):
        ScorerGradientPolicy(**values)

    values = fixture.policy.model_dump()
    values["instant_admission_economics"] = InstantAdmissionEconomics(
        exact_seconds=1.0,
        approximate_seconds=0.5,
        projected_candidate_validation_seconds=5.0,
    ).model_dump()
    with pytest.raises(ValidationError, match="decisive NO-GO"):
        ScorerGradientPolicy(**values)
