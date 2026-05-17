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
    extract_archive_sha256,
    extract_observed_runtime_tree_sha256,
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

_COMMON_AXIS_ALIASES = {
    "contest_cpu": "contest_cpu",
    "contest_cuda": "contest_cuda",
    "cpu_advisory": "macos_cpu_advisory",
    "macos_cpu": "macos_cpu_advisory",
    "macos_cpu_advisory": "macos_cpu_advisory",
    "mps": "mps_advisory",
    "mps_advisory": "mps_advisory",
}


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
    max_ablation_archive_bytes_delta: int
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
                "max_ablation_archive_bytes_delta": self.max_ablation_archive_bytes_delta,
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


def _non_negative_int(value: int, *, name: str) -> int:
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _first_value(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _nested_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _normalize_axis_label(value: object) -> str:
    """Normalize common exact-eval axis spellings without promoting advisories."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    unbracketed = raw.strip("[]").strip()
    lowered = (
        unbracketed.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("__", "_")
    )
    return _COMMON_AXIS_ALIASES.get(lowered, unbracketed)


def _command_flag_value(command: object, flag: str) -> str:
    parts = (
        [str(part) for part in command]
        if isinstance(command, list)
        else str(command).split()
    )
    for idx, part in enumerate(parts[:-1]):
        if part == flag:
            return parts[idx + 1].strip()
    prefix = f"{flag}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :].strip()
    return ""


def normalize_score_response_mapping(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize common contest-auth-eval schemas into flat custody fields.

    The repo has several historical score artifacts. Newer exact-dispatch
    rows already use ``axis``/``score``/``seg_dist``/``pose_dist``. Older
    ``experiments/contest_auth_eval.py`` artifacts use ``score_axis``,
    ``canonical_score``, ``avg_segnet_dist``, ``avg_posenet_dist`` and nested
    ``provenance`` fields. This helper preserves any explicit flat field while
    filling gaps from the canonical richer schema.
    """

    out = dict(mapping)
    provenance = _nested_mapping(mapping, "provenance")
    custody = _nested_mapping(mapping, "custody")
    runtime_custody = _nested_mapping(mapping, "runtime_custody")
    score_recomputation = _nested_mapping(mapping, "score_recomputation")
    runtime_manifest = _nested_mapping(provenance, "inflate_runtime_manifest")
    inflated_manifest = _nested_mapping(provenance, "inflated_output_manifest")
    command = _first_value(out, "auth_eval_command", "command") or custody.get("command")

    out["axis"] = _normalize_axis_label(
        _first_value(out, "axis", "score_axis", "evidence_axis")
    )
    out["archive_sha256"] = _first_value(
        out,
        "archive_sha256",
        "candidate_archive_sha256",
        "exact_archive_sha256",
    ) or extract_archive_sha256(custody) or extract_archive_sha256(provenance)
    out["runtime_tree_sha256"] = _first_value(
        out,
        "runtime_tree_sha256",
        "observed_runtime_tree_sha256",
        "inflate_runtime_tree_sha256",
    ) or extract_observed_runtime_tree_sha256(runtime_custody) or extract_observed_runtime_tree_sha256(
        provenance
    )
    out["n_samples"] = _first_value(out, "n_samples", "sample_count") or custody.get("n_samples")
    out["archive_bytes"] = _first_value(
        out,
        "archive_bytes",
        "archive_size_bytes",
    ) or score_recomputation.get("archive_bytes") or custody.get("archive_bytes") or provenance.get(
        "archive_size_bytes"
    )
    out["seg_dist"] = _first_value(
        out,
        "seg_dist",
        "avg_segnet_dist",
        "segmentation_distortion",
    ) or score_recomputation.get("avg_segnet_dist")
    out["pose_dist"] = _first_value(
        out,
        "pose_dist",
        "avg_posenet_dist",
        "pose_distortion",
    ) or score_recomputation.get("avg_posenet_dist")
    out["score"] = _first_value(
        out,
        "score",
        "canonical_score",
        "score_recomputed_from_components",
        "score_recomputed",
        "recomputed_score",
    ) or score_recomputation.get("recomputed_score") or score_recomputation.get(
        "reported_score"
    )
    out["hardware"] = _first_value(out, "hardware", "hardware_summary") or " ".join(
        str(part)
        for part in (
            custody.get("gpu_model"),
            custody.get("device"),
            provenance.get("platform_system"),
            provenance.get("platform_machine"),
            provenance.get("gpu_model"),
            provenance.get("device"),
        )
        if part
    )
    out["inflate_device"] = _first_value(
        out,
        "inflate_device",
        "inflate_device_policy",
    ) or _command_flag_value(command, "--inflate-device") or provenance.get(
        "inflate_device_policy"
    ) or provenance.get("device") or custody.get("device")
    out["eval_device"] = _first_value(out, "eval_device", "device") or provenance.get(
        "device"
    ) or custody.get("device")
    out["auth_eval_command"] = command or provenance.get("sys_argv")
    out["log_path"] = _first_value(out, "log_path", "report_path") or out.get("report_path")
    if isinstance(runtime_custody, Mapping):
        out["inflated_outputs_manifest_sha256"] = _first_value(
            out,
            "inflated_outputs_manifest_sha256",
            "inflated_output_manifest_sha256",
        ) or runtime_custody.get("inflated_output_manifest_sha256")
        out["raw_output_aggregate_sha256"] = _first_value(
            out,
            "raw_output_aggregate_sha256",
            "inflated_output_aggregate_sha256",
        ) or runtime_custody.get("inflated_output_aggregate_sha256")
    if isinstance(inflated_manifest, Mapping):
        out["inflated_outputs_manifest_path"] = _first_value(
            out,
            "inflated_outputs_manifest_path",
            "inflated_output_manifest_path",
        ) or inflated_manifest.get("path")
        out["inflated_outputs_manifest_sha256"] = _first_value(
            out,
            "inflated_outputs_manifest_sha256",
            "inflated_output_manifest_sha256",
        ) or inflated_manifest.get("sha256")
        payload = _nested_mapping(inflated_manifest, "payload")
        out["raw_output_aggregate_sha256"] = _first_value(
            out,
            "raw_output_aggregate_sha256",
            "inflated_output_aggregate_sha256",
        ) or payload.get("aggregate_sha256")
    if isinstance(runtime_manifest, Mapping) and not out.get("runtime_tree_sha256"):
        out["runtime_tree_sha256"] = runtime_manifest.get("runtime_tree_sha256")
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

    normalized = normalize_score_response_mapping(mapping)
    if not str(normalized.get("axis") or "").strip():
        return None, (f"{label}:axis_missing",), (f"{label}_axis_missing",)
    validation = validate_exact_eval_evidence(
        normalized,
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
    evidence = _evidence_from_validation(label=label, mapping=normalized, validation=validation)
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
    max_ablation_archive_bytes_delta: int = 0,
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
    max_ablation_bytes = _non_negative_int(
        max_ablation_archive_bytes_delta,
        name="max_ablation_archive_bytes_delta",
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
            max_ablation_archive_bytes_delta=max_ablation_bytes,
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
    if mode == "ablation":
        archive_bytes_delta = abs(
            candidate_evidence.archive_bytes - baseline_evidence.archive_bytes
        )
        if archive_bytes_delta > max_ablation_bytes:
            control_blockers.append(
                "archive_bytes_mismatch:"
                f"{archive_bytes_delta}>{max_ablation_bytes}"
            )
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
            max_ablation_archive_bytes_delta=max_ablation_bytes,
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
        max_ablation_archive_bytes_delta=max_ablation_bytes,
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
    "normalize_score_response_mapping",
    "validate_score_response_evidence",
]
