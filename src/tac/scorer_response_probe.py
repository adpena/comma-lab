# SPDX-License-Identifier: MIT
"""Score-response probe for substrate distinguishing features.

This module operationalizes the Rule #6 lesson from the 2026-05-17
substrate-design meta-assumption review: byte liveness is not score response.
Given a baseline exact-eval artifact and a candidate exact-eval artifact, it
computes contest-formula component deltas and classifies whether the candidate
actually moved the scorer-visible terms or merely changed bytes/rate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from tac.exact_eval_custody import (
    ExactEvalEvidenceValidation,
    contest_score,
    validate_exact_eval_evidence,
)


VERDICT_SCORER_RESPONSE_POSITIVE = "SCORER_RESPONSE_POSITIVE"
VERDICT_SCORER_RESPONSE_PRESENT_RATE_NEGATIVE = "SCORER_RESPONSE_PRESENT_RATE_NEGATIVE"
VERDICT_RATE_ONLY_IMPROVEMENT = "RATE_ONLY_IMPROVEMENT"
VERDICT_NO_MEASURABLE_RESPONSE = "NO_MEASURABLE_RESPONSE"
VERDICT_SCORE_REGRESSION = "SCORE_REGRESSION"
VERDICT_BLOCKED_CUSTODY = "BLOCKED_CUSTODY"
VERDICT_BLOCKED_CONTROL_MISMATCH = "BLOCKED_CONTROL_MISMATCH"

VALID_PROBE_MODES = frozenset({"ablation", "candidate"})


@dataclass(frozen=True)
class ScoreResponseEvidence:
    """Normalized score evidence and contest-formula component terms."""

    label: str
    axis: str
    archive_sha256: str
    runtime_tree_sha256: str
    n_samples: int
    archive_bytes: int
    seg_dist: float
    pose_dist: float
    score: float
    seg_term: float
    pose_term: float
    rate_term: float


@dataclass(frozen=True)
class ScoreResponseReport:
    """Baseline-vs-candidate score-response classification."""

    verdict: str
    blockers: tuple[str, ...]
    annotations: tuple[str, ...]
    baseline: ScoreResponseEvidence | None
    candidate: ScoreResponseEvidence | None
    total_delta: float | None
    seg_term_delta: float | None
    pose_term_delta: float | None
    rate_term_delta: float | None
    scorer_term_delta: float | None
    min_total_improvement: float
    min_scorer_term_improvement: float
    mode: str

    def to_json_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable report."""

        def evidence_to_dict(evidence: ScoreResponseEvidence | None) -> dict[str, Any] | None:
            if evidence is None:
                return None
            return {
                "label": evidence.label,
                "axis": evidence.axis,
                "archive_sha256": evidence.archive_sha256,
                "runtime_tree_sha256": evidence.runtime_tree_sha256,
                "n_samples": evidence.n_samples,
                "archive_bytes": evidence.archive_bytes,
                "seg_dist": evidence.seg_dist,
                "pose_dist": evidence.pose_dist,
                "score": evidence.score,
                "seg_term": evidence.seg_term,
                "pose_term": evidence.pose_term,
                "rate_term": evidence.rate_term,
            }

        return {
            "schema": "substrate_score_response_probe_v1",
            "verdict": self.verdict,
            "blockers": list(self.blockers),
            "annotations": list(self.annotations),
            "mode": self.mode,
            "thresholds": {
                "min_total_improvement": self.min_total_improvement,
                "min_scorer_term_improvement": self.min_scorer_term_improvement,
            },
            "baseline": evidence_to_dict(self.baseline),
            "candidate": evidence_to_dict(self.candidate),
            "deltas": {
                "total_delta": self.total_delta,
                "seg_term_delta": self.seg_term_delta,
                "pose_term_delta": self.pose_term_delta,
                "rate_term_delta": self.rate_term_delta,
                "scorer_term_delta": self.scorer_term_delta,
            },
        }


def _finite_threshold(value: float, *, name: str) -> float:
    if not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    out = float(value)
    if out < 0.0 or out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite non-negative number")
    return out


def _evidence_from_validation(
    *,
    label: str,
    mapping: Mapping[str, Any],
    validation: ExactEvalEvidenceValidation,
) -> ScoreResponseEvidence:
    assert validation.archive_bytes is not None
    assert validation.seg_dist is not None
    assert validation.pose_dist is not None
    assert validation.score is not None
    assert validation.n_samples is not None
    archive_bytes = validation.archive_bytes
    seg_dist = validation.seg_dist
    pose_dist = validation.pose_dist
    score = validation.score
    return ScoreResponseEvidence(
        label=label,
        axis=str(mapping.get("axis") or "").strip(),
        archive_sha256=validation.archive_sha256,
        runtime_tree_sha256=validation.runtime_tree_sha256,
        n_samples=validation.n_samples,
        archive_bytes=archive_bytes,
        seg_dist=seg_dist,
        pose_dist=pose_dist,
        score=score,
        seg_term=100.0 * seg_dist,
        pose_term=(10.0 * pose_dist) ** 0.5,
        rate_term=25.0 * archive_bytes / 37_545_489,
    )


def validate_score_response_evidence(
    mapping: Mapping[str, Any],
    *,
    label: str,
    expected_axis: str | None = None,
    strict_exact_custody: bool = True,
) -> tuple[ScoreResponseEvidence | None, tuple[str, ...], tuple[str, ...]]:
    """Validate and normalize one exact-eval evidence mapping."""

    validation = validate_exact_eval_evidence(
        mapping,
        expected_axis=expected_axis,
        require_hardware=strict_exact_custody,
        require_auth_eval_command=strict_exact_custody,
        require_log_path=strict_exact_custody,
        require_devices=strict_exact_custody,
        require_inflated_outputs_manifest=False,
        require_raw_output_aggregate_sha256=False,
        annotation_prefix=label,
    )
    if validation.blockers:
        blockers = tuple(f"{label}:{blocker}" for blocker in validation.blockers)
        return None, blockers, validation.annotations
    evidence = _evidence_from_validation(label=label, mapping=mapping, validation=validation)
    recomputed = contest_score(
        seg_dist=evidence.seg_dist,
        pose_dist=evidence.pose_dist,
        archive_bytes=evidence.archive_bytes,
    )
    if abs(recomputed - evidence.score) > 1e-9:
        return None, (f"{label}:score_formula_mismatch",), validation.annotations
    return evidence, (), validation.annotations


def compare_score_response(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expected_axis: str | None = None,
    mode: str = "ablation",
    min_total_improvement: float = 0.001,
    min_scorer_term_improvement: float = 0.0005,
    strict_exact_custody: bool = True,
) -> ScoreResponseReport:
    """Compare two evidence rows and classify scorer-visible response.

    Negative deltas are improvements because they reduce the contest score.
    In ``ablation`` mode, baseline and candidate must share axis, runtime tree,
    and sample count so the distinguishing feature is the tested variable.
    ``candidate`` mode still requires axis/sample matching but allows runtime
    tree differences for across-substrate scouting reports.
    """

    if mode not in VALID_PROBE_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_PROBE_MODES)}")
    min_total = _finite_threshold(min_total_improvement, name="min_total_improvement")
    min_scorer = _finite_threshold(
        min_scorer_term_improvement,
        name="min_scorer_term_improvement",
    )

    baseline_evidence, baseline_blockers, baseline_annotations = validate_score_response_evidence(
        baseline,
        label="baseline",
        expected_axis=expected_axis,
        strict_exact_custody=strict_exact_custody,
    )
    candidate_evidence, candidate_blockers, candidate_annotations = (
        validate_score_response_evidence(
            candidate,
            label="candidate",
            expected_axis=expected_axis,
            strict_exact_custody=strict_exact_custody,
        )
    )
    blockers = tuple(baseline_blockers + candidate_blockers)
    annotations = tuple(baseline_annotations + candidate_annotations)
    if blockers:
        return ScoreResponseReport(
            verdict=VERDICT_BLOCKED_CUSTODY,
            blockers=blockers,
            annotations=annotations,
            baseline=baseline_evidence,
            candidate=candidate_evidence,
            total_delta=None,
            seg_term_delta=None,
            pose_term_delta=None,
            rate_term_delta=None,
            scorer_term_delta=None,
            min_total_improvement=min_total,
            min_scorer_term_improvement=min_scorer,
            mode=mode,
        )

    assert baseline_evidence is not None
    assert candidate_evidence is not None
    control_blockers: list[str] = []
    if baseline_evidence.axis != candidate_evidence.axis:
        control_blockers.append("axis_mismatch")
    if baseline_evidence.n_samples != candidate_evidence.n_samples:
        control_blockers.append("sample_count_mismatch")
    if mode == "ablation" and (
        baseline_evidence.runtime_tree_sha256 != candidate_evidence.runtime_tree_sha256
    ):
        control_blockers.append("runtime_tree_mismatch")
    if control_blockers:
        return ScoreResponseReport(
            verdict=VERDICT_BLOCKED_CONTROL_MISMATCH,
            blockers=tuple(control_blockers),
            annotations=annotations,
            baseline=baseline_evidence,
            candidate=candidate_evidence,
            total_delta=None,
            seg_term_delta=None,
            pose_term_delta=None,
            rate_term_delta=None,
            scorer_term_delta=None,
            min_total_improvement=min_total,
            min_scorer_term_improvement=min_scorer,
            mode=mode,
        )

    total_delta = candidate_evidence.score - baseline_evidence.score
    seg_delta = candidate_evidence.seg_term - baseline_evidence.seg_term
    pose_delta = candidate_evidence.pose_term - baseline_evidence.pose_term
    rate_delta = candidate_evidence.rate_term - baseline_evidence.rate_term
    scorer_delta = seg_delta + pose_delta

    if scorer_delta <= -min_scorer and total_delta <= -min_total:
        verdict = VERDICT_SCORER_RESPONSE_POSITIVE
    elif scorer_delta <= -min_scorer:
        verdict = VERDICT_SCORER_RESPONSE_PRESENT_RATE_NEGATIVE
    elif total_delta <= -min_total:
        verdict = VERDICT_RATE_ONLY_IMPROVEMENT
    elif total_delta >= min_total or scorer_delta >= min_scorer:
        verdict = VERDICT_SCORE_REGRESSION
    else:
        verdict = VERDICT_NO_MEASURABLE_RESPONSE

    return ScoreResponseReport(
        verdict=verdict,
        blockers=(),
        annotations=annotations,
        baseline=baseline_evidence,
        candidate=candidate_evidence,
        total_delta=total_delta,
        seg_term_delta=seg_delta,
        pose_term_delta=pose_delta,
        rate_term_delta=rate_delta,
        scorer_term_delta=scorer_delta,
        min_total_improvement=min_total,
        min_scorer_term_improvement=min_scorer,
        mode=mode,
    )


__all__ = [
    "ScoreResponseEvidence",
    "ScoreResponseReport",
    "VERDICT_BLOCKED_CONTROL_MISMATCH",
    "VERDICT_BLOCKED_CUSTODY",
    "VERDICT_NO_MEASURABLE_RESPONSE",
    "VERDICT_RATE_ONLY_IMPROVEMENT",
    "VERDICT_SCORER_RESPONSE_POSITIVE",
    "VERDICT_SCORER_RESPONSE_PRESENT_RATE_NEGATIVE",
    "VERDICT_SCORE_REGRESSION",
    "compare_score_response",
    "validate_score_response_evidence",
]
