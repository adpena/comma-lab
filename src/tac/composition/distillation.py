# SPDX-License-Identifier: MIT
"""Typed distillation-STAGE chain metadata for composition substrates.

The chain model here records teacher-to-student SUBSTRATE stages without
turning proxy or distillation evidence into score authority.  It preserves
eval-roundtrip and scorer-axis contracts so trainers, planners, and ledgers
can agree about what was distilled before an exact archive evaluation
exists.

NAMING DISTINCTION (legacy cleanup 2026-05-13):
This module's :class:`DistillationStageChain` is the **typed-metadata**
surface (substrate-level stage sequence).  The Hinton 2014 KL-loss
distillation primitive (param-count-level student chain + ``T^2`` scaled
loss) is :class:`tac.composition.distillation_chain.DistillationChain`.
The two used to collide on the name ``DistillationChain``; renamed here
to extinct the collision per CLAUDE.md "Beauty, simplicity, and developer
experience — non-negotiable" duplicate-deletion discipline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tac.composition.frontier_primitives import canonical_json_bytes, metadata_sha256
from tac.composition.registry import (
    PLANNING_ONLY,
    PROMOTION_ELIGIBLE,
    READY_FOR_EXACT_EVAL_DISPATCH,
    SCORE_CLAIM,
    ScoreAxis,
)
from tac.kl_config import RoundtripContract

DISTILLATION_CHAIN_SCHEMA_VERSION = "tac_composition_distillation_chain_v1"


class DistillationChainError(ValueError):
    """Raised when distillation-chain metadata is malformed."""


def _clean_metadata(metadata: Mapping[str, Any] | Sequence[tuple[str, Any]] | None) -> tuple[tuple[str, str], ...]:
    if metadata is None:
        return ()
    if isinstance(metadata, Mapping):
        return tuple(sorted((str(k), str(v)) for k, v in metadata.items()))
    return tuple((str(k), str(v)) for k, v in metadata)


def _normalize_axes(axes: Sequence[ScoreAxis | str]) -> tuple[ScoreAxis, ...]:
    if not axes:
        raise DistillationChainError("scorer_axes cannot be empty")
    out: list[ScoreAxis] = []
    for axis in axes:
        try:
            out.append(axis if isinstance(axis, ScoreAxis) else ScoreAxis(str(axis)))
        except ValueError as exc:
            raise DistillationChainError(f"unknown scorer axis {axis!r}") from exc
    return tuple(out)


@dataclass(frozen=True, slots=True)
class DistillationStage:
    """One teacher-to-student substrate distillation stage.

    ``research_only=True`` allows incomplete roundtrip metadata to be recorded
    for forensic work.  Production-capable stages fail closed unless the
    roundtrip contract is promotion-safe.
    """

    stage_id: str
    teacher_substrate_id: str
    student_substrate_id: str
    loss_family: str
    scorer_axes: tuple[ScoreAxis | str, ...]
    roundtrip_contract: RoundtripContract = field(default_factory=RoundtripContract)
    teacher_artifact_id: str = ""
    student_artifact_id: str = ""
    evidence_refs: tuple[str, ...] = ()
    research_only: bool = False
    score_claim: bool = SCORE_CLAIM
    promotion_eligible: bool = PROMOTION_ELIGIBLE
    ready_for_exact_eval_dispatch: bool = READY_FOR_EXACT_EVAL_DISPATCH
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.stage_id:
            raise DistillationChainError("stage_id cannot be empty")
        if not self.teacher_substrate_id:
            raise DistillationChainError("teacher_substrate_id cannot be empty")
        if not self.student_substrate_id:
            raise DistillationChainError("student_substrate_id cannot be empty")
        if self.teacher_substrate_id == self.student_substrate_id:
            raise DistillationChainError("teacher and student substrates must differ")
        if not self.loss_family:
            raise DistillationChainError("loss_family cannot be empty")
        axes = _normalize_axes(self.scorer_axes)
        object.__setattr__(self, "scorer_axes", axes)
        object.__setattr__(
            self,
            "metadata",
            _clean_metadata(self.metadata),
        )
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise DistillationChainError(
                "distillation stages cannot claim score or dispatch readiness"
            )
        if not self.roundtrip_contract.promotion_safe and not self.research_only:
            raise DistillationChainError(
                "non-research distillation stages require promotion-safe "
                "eval_roundtrip metadata"
            )

    @property
    def eval_roundtrip(self) -> bool:
        return self.roundtrip_contract.eval_roundtrip

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_refs": list(self.evidence_refs),
            "eval_roundtrip": self.roundtrip_contract.eval_roundtrip,
            "loss_family": self.loss_family,
            "metadata": dict(self.metadata),
            "planning_only": PLANNING_ONLY,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "research_only": self.research_only,
            "roundtrip_contract": self.roundtrip_contract.to_provenance(),
            "schema_version": DISTILLATION_CHAIN_SCHEMA_VERSION,
            "score_claim": self.score_claim,
            "scorer_axes": [axis.value for axis in self.scorer_axes],
            "stage_id": self.stage_id,
            "student_artifact_id": self.student_artifact_id,
            "student_substrate_id": self.student_substrate_id,
            "teacher_artifact_id": self.teacher_artifact_id,
            "teacher_substrate_id": self.teacher_substrate_id,
        }

    def sha256(self) -> str:
        return metadata_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class DistillationStageChain:
    """Linear teacher-to-student substrate distillation-stage chain.

    Typed-metadata surface; not to be confused with
    :class:`tac.composition.distillation_chain.DistillationChain` (the
    Hinton 2014 KL-loss + param-count chain primitive).
    """

    chain_id: str
    stages: tuple[DistillationStage, ...]
    metadata: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    score_claim: bool = SCORE_CLAIM
    promotion_eligible: bool = PROMOTION_ELIGIBLE
    ready_for_exact_eval_dispatch: bool = READY_FOR_EXACT_EVAL_DISPATCH

    def __post_init__(self) -> None:
        if not self.chain_id:
            raise DistillationChainError("chain_id cannot be empty")
        if not self.stages:
            raise DistillationChainError("distillation chain must contain stages")
        if self.score_claim or self.promotion_eligible or self.ready_for_exact_eval_dispatch:
            raise DistillationChainError(
                "distillation chains cannot claim score or dispatch readiness"
            )
        object.__setattr__(
            self,
            "metadata",
            _clean_metadata(self.metadata),
        )
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise DistillationChainError(f"duplicate stage_ids: {stage_ids}")
        for idx, (left, right) in enumerate(
            zip(self.stages, self.stages[1:], strict=False)
        ):
            if left.student_substrate_id != right.teacher_substrate_id:
                raise DistillationChainError(
                    "distillation stages must be contiguous: "
                    f"stage {idx} student={left.student_substrate_id!r} "
                    f"!= next teacher={right.teacher_substrate_id!r}"
                )

    @property
    def teacher_substrate_id(self) -> str:
        return self.stages[0].teacher_substrate_id

    @property
    def student_substrate_id(self) -> str:
        return self.stages[-1].student_substrate_id

    @property
    def scorer_axes(self) -> tuple[ScoreAxis, ...]:
        axes: list[ScoreAxis] = []
        for stage in self.stages:
            for axis in stage.scorer_axes:
                if axis not in axes:
                    axes.append(axis)
        return tuple(axes)

    @property
    def eval_roundtrip(self) -> bool:
        return all(stage.eval_roundtrip for stage in self.stages)

    def append(self, stage: DistillationStage) -> DistillationStageChain:
        """Return a new chain with ``stage`` appended and validated."""

        return DistillationStageChain(
            chain_id=self.chain_id,
            stages=(*self.stages, stage),
            metadata=self.metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "chain_id": self.chain_id,
            "eval_roundtrip": self.eval_roundtrip,
            "kind": "teacher_student_distillation_chain",
            "metadata": dict(self.metadata),
            "planning_only": PLANNING_ONLY,
            "promotion_eligible": self.promotion_eligible,
            "ready_for_exact_eval_dispatch": self.ready_for_exact_eval_dispatch,
            "schema_version": DISTILLATION_CHAIN_SCHEMA_VERSION,
            "score_claim": self.score_claim,
            "scorer_axes": [axis.value for axis in self.scorer_axes],
            "stages": [stage.to_dict() for stage in self.stages],
            "student_substrate_id": self.student_substrate_id,
            "teacher_substrate_id": self.teacher_substrate_id,
        }
        payload["metadata_sha256"] = metadata_sha256(payload)
        return payload

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")

    def sha256(self) -> str:
        return metadata_sha256(self.to_dict())


def build_distillation_stage_chain(
    chain_id: str,
    stages: Sequence[DistillationStage],
    *,
    metadata: Mapping[str, Any] | Sequence[tuple[str, Any]] | None = None,
) -> DistillationStageChain:
    """Construct and validate a linear distillation-stage chain."""

    return DistillationStageChain(
        chain_id=chain_id,
        stages=tuple(stages),
        metadata=_clean_metadata(metadata),
    )


__all__ = [
    "DISTILLATION_CHAIN_SCHEMA_VERSION",
    "DistillationChainError",
    "DistillationStage",
    "DistillationStageChain",
    "build_distillation_stage_chain",
]
