"""Tests for typed distillation-stage chain metadata.

This covers the substrate-level stage-sequence schema in
``tac.composition.distillation``.  Tests for the Hinton 2014 KL-loss
primitive (``DistillationChain`` with param-count levels) live in
``test_distillation_chain.py``.
"""

from __future__ import annotations

import pytest

from tac.composition.distillation import (
    DistillationChainError,
    DistillationStage,
    DistillationStageChain,
    build_distillation_stage_chain,
)
from tac.composition.registry import ScoreAxis
from tac.kl_config import RoundtripContract


def _stage(
    stage_id: str,
    teacher: str,
    student: str,
    *,
    axes: tuple[ScoreAxis | str, ...] = (ScoreAxis.SEG,),
) -> DistillationStage:
    return DistillationStage(
        stage_id=stage_id,
        teacher_substrate_id=teacher,
        student_substrate_id=student,
        loss_family="segnet_aux_kl",
        scorer_axes=axes,
        evidence_refs=("artifact://unit",),
    )


def test_stage_preserves_eval_roundtrip_and_axis_metadata() -> None:
    stage = _stage("s1", "teacher", "student", axes=("seg", "pose"))
    payload = stage.to_dict()
    assert payload["eval_roundtrip"] is True
    assert payload["roundtrip_contract"]["same_as_scorer_input"] is True
    assert payload["scorer_axes"] == ["seg", "pose"]
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_stage_fail_closes_on_bad_axis_and_score_authority() -> None:
    with pytest.raises(DistillationChainError):
        _stage("s1", "teacher", "student", axes=("cuda_score",))
    with pytest.raises(DistillationChainError):
        DistillationStage(
            stage_id="s1",
            teacher_substrate_id="teacher",
            student_substrate_id="student",
            loss_family="segnet_aux_kl",
            scorer_axes=(ScoreAxis.SEG,),
            score_claim=True,
        )


def test_non_research_stage_requires_promotion_safe_roundtrip() -> None:
    bad_roundtrip = RoundtripContract(eval_roundtrip=False)
    with pytest.raises(DistillationChainError):
        DistillationStage(
            stage_id="s1",
            teacher_substrate_id="teacher",
            student_substrate_id="student",
            loss_family="segnet_aux_kl",
            scorer_axes=(ScoreAxis.SEG,),
            roundtrip_contract=bad_roundtrip,
        )
    research = DistillationStage(
        stage_id="s1",
        teacher_substrate_id="teacher",
        student_substrate_id="student",
        loss_family="segnet_aux_kl",
        scorer_axes=(ScoreAxis.SEG,),
        roundtrip_contract=bad_roundtrip,
        research_only=True,
    )
    assert research.to_dict()["research_only"] is True
    assert research.to_dict()["eval_roundtrip"] is False


def test_chain_requires_contiguous_teacher_student_stages() -> None:
    s1 = _stage("s1", "teacher", "mid")
    s2 = _stage("s2", "mid", "student", axes=(ScoreAxis.POSE,))
    chain = build_distillation_stage_chain("chain", (s1, s2))
    assert chain.teacher_substrate_id == "teacher"
    assert chain.student_substrate_id == "student"
    assert chain.scorer_axes == (ScoreAxis.SEG, ScoreAxis.POSE)
    assert chain.eval_roundtrip is True
    with pytest.raises(DistillationChainError):
        build_distillation_stage_chain("bad", (s1, _stage("s2", "other", "student")))


def test_chain_serialization_is_deterministic_and_no_score_claim() -> None:
    s1 = _stage("s1", "teacher", "student")
    c1 = build_distillation_stage_chain("chain", (s1,), metadata={"owner": "unit"})
    c2 = build_distillation_stage_chain("chain", (s1,), metadata={"owner": "unit"})
    assert c1.to_json() == c2.to_json()
    assert c1.sha256() == c2.sha256()
    payload = c1.to_dict()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["metadata"]["owner"] == "unit"


def test_chain_rejects_duplicate_stage_ids_and_empty_chain() -> None:
    s1 = _stage("s1", "teacher", "mid")
    s2 = _stage("s1", "mid", "student")
    with pytest.raises(DistillationChainError):
        DistillationStageChain(chain_id="chain", stages=(s1, s2))
    with pytest.raises(DistillationChainError):
        DistillationStageChain(chain_id="empty", stages=())
