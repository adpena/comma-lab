# SPDX-License-Identifier: MIT
"""CUDA component-risk gate for HDM8 postfilter selector candidates."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA = (
    "hdm8_selector_cuda_component_risk_gate_v1"
)
DEFAULT_HDM8_CUDA_REFERENCE_RESULT = Path(
    "experiments/results/modal_auth_eval/"
    "hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/"
    "contest_auth_eval.json"
)
CUDA_PREFIX_AXES = frozenset(
    {
        "modal-t4-cuda-proxy-prefix",
        "local-cuda-proxy-prefix",
        "cuda-proxy-prefix",
    }
)
PASSING_STATUSES = frozenset(
    {
        "passed_cuda_prefix_component_check",
        "passed_exact_cuda_component_check",
    }
)
DEFAULT_MAX_POSE_DELTA = 0.0
DEFAULT_MAX_SEG_DELTA = 0.0
DEFAULT_MAX_SCORE_DELTA = 0.0
DEFAULT_MIN_CUDA_PREFIX_PAIRS = 24


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve(path: Path | str | None, *, repo_root: Path) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_from_auth_eval(payload: Mapping[str, Any]) -> float | None:
    for key in (
        "score_recomputed_from_components",
        "canonical_score",
        "final_score",
        "score",
    ):
        parsed = _finite_float(payload.get(key))
        if parsed is not None:
            return parsed
    return None


def _auth_eval_anchor(path: Path, *, repo_root: Path) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    if not path.is_file():
        return None, [f"auth_eval_anchor_missing:{_repo_rel(path, repo_root)}"]
    try:
        payload = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"auth_eval_anchor_json_invalid:{type(exc).__name__}"]
    if not isinstance(payload, Mapping):
        return None, ["auth_eval_anchor_not_object"]

    score = _score_from_auth_eval(payload)
    pose = _finite_float(payload.get("avg_posenet_dist"))
    seg = _finite_float(payload.get("avg_segnet_dist"))
    if score is None:
        blockers.append("auth_eval_anchor_score_missing")
    if pose is None:
        blockers.append("auth_eval_anchor_avg_posenet_dist_missing")
    if seg is None:
        blockers.append("auth_eval_anchor_avg_segnet_dist_missing")
    axis = payload.get("score_axis")
    evidence_grade = payload.get("evidence_grade")
    exact_cuda = (
        axis == "contest_cuda"
        and evidence_grade == "contest-CUDA"
        and payload.get("exact_cuda_eval_complete") is True
    )
    if not exact_cuda:
        blockers.append("auth_eval_anchor_not_contest_cuda_exact")

    provenance = payload.get("provenance")
    archive_sha256 = (
        provenance.get("archive_sha256") if isinstance(provenance, Mapping) else None
    )
    archive_bytes = (
        provenance.get("archive_size_bytes") if isinstance(provenance, Mapping) else None
    )
    return (
        {
            "path": _repo_rel(path, repo_root),
            "score": score,
            "avg_posenet_dist": pose,
            "avg_segnet_dist": seg,
            "score_axis": axis,
            "evidence_grade": evidence_grade,
            "exact_cuda_eval_complete": payload.get("exact_cuda_eval_complete"),
            "n_samples": payload.get("n_samples"),
            "archive_sha256": archive_sha256,
            "archive_bytes": archive_bytes,
        },
        blockers,
    )


def _component_deltas(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, float]:
    return {
        "score_delta": float(candidate["score"]) - float(reference["score"]),
        "pose_delta": float(candidate["avg_posenet_dist"])
        - float(reference["avg_posenet_dist"]),
        "seg_delta": float(candidate["avg_segnet_dist"])
        - float(reference["avg_segnet_dist"]),
    }


def _threshold_blockers(
    deltas: Mapping[str, float],
    *,
    max_pose_delta: float,
    max_seg_delta: float,
    max_score_delta: float,
) -> list[str]:
    blockers: list[str] = []
    if float(deltas["pose_delta"]) > max_pose_delta:
        blockers.append(
            "posenet_delta_exceeds_threshold:"
            f"{float(deltas['pose_delta']):.12g}>{max_pose_delta:.12g}"
        )
    if float(deltas["seg_delta"]) > max_seg_delta:
        blockers.append(
            "segnet_delta_exceeds_threshold:"
            f"{float(deltas['seg_delta']):.12g}>{max_seg_delta:.12g}"
        )
    if float(deltas["score_delta"]) > max_score_delta:
        blockers.append(
            "score_delta_exceeds_threshold:"
            f"{float(deltas['score_delta']):.12g}>{max_score_delta:.12g}"
        )
    return blockers


def _local_proxy_axis(axis: object) -> bool:
    text = str(axis or "").lower()
    return "mps" in text or text.startswith("local-cpu") or text == "cpu"


def _cuda_prefix_axis(axis: object) -> bool:
    text = str(axis or "").lower()
    return text in CUDA_PREFIX_AXES or ("cuda" in text and "proxy-prefix" in text)


def build_hdm8_selector_cuda_component_gate(
    *,
    proxy: Mapping[str, Any] | None,
    candidate_archive_sha256: str,
    candidate_archive_bytes: int,
    repo_root: Path,
    reference_result_path: Path | str | None = DEFAULT_HDM8_CUDA_REFERENCE_RESULT,
    candidate_exact_cuda_result_path: Path | str | None = None,
    max_pose_delta: float = DEFAULT_MAX_POSE_DELTA,
    max_seg_delta: float = DEFAULT_MAX_SEG_DELTA,
    max_score_delta: float = DEFAULT_MAX_SCORE_DELTA,
    min_cuda_prefix_pairs: int = DEFAULT_MIN_CUDA_PREFIX_PAIRS,
) -> dict[str, Any]:
    """Build a fail-closed selector gate from exact-CUDA or CUDA-prefix evidence."""

    repo_root = Path(repo_root)
    gate: dict[str, Any] = {
        "schema": HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA,
        "required": True,
        "passed": False,
        "status": "blocked",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "candidate_archive_sha256": candidate_archive_sha256,
        "candidate_archive_bytes": int(candidate_archive_bytes),
        "thresholds": {
            "max_pose_delta": float(max_pose_delta),
            "max_seg_delta": float(max_seg_delta),
            "max_score_delta": float(max_score_delta),
            "min_cuda_prefix_pairs": int(min_cuda_prefix_pairs),
        },
        "blockers": [],
    }
    reference_path = _resolve(reference_result_path, repo_root=repo_root)
    if reference_path is not None:
        gate["reference_exact_cuda_result_path"] = _repo_rel(reference_path, repo_root)
        reference, reference_blockers = _auth_eval_anchor(reference_path, repo_root=repo_root)
        if reference is not None:
            gate["reference"] = reference
        gate["blockers"].extend(f"reference_{item}" for item in reference_blockers)
    else:
        reference = None
        gate["blockers"].append("reference_exact_cuda_result_path_missing")

    candidate_path = _resolve(candidate_exact_cuda_result_path, repo_root=repo_root)
    if candidate_path is not None:
        gate["candidate_exact_cuda_result_path"] = _repo_rel(candidate_path, repo_root)
        candidate, candidate_blockers = _auth_eval_anchor(candidate_path, repo_root=repo_root)
        if candidate is not None:
            gate["candidate"] = candidate
            if (
                isinstance(candidate.get("archive_sha256"), str)
                and candidate["archive_sha256"] != candidate_archive_sha256
            ):
                candidate_blockers.append("candidate_exact_cuda_archive_sha_mismatch")
        gate["blockers"].extend(f"candidate_{item}" for item in candidate_blockers)
        if reference is not None and candidate is not None and not gate["blockers"]:
            deltas = _component_deltas(candidate, reference)
            gate["component_deltas"] = deltas
            gate["blockers"].extend(
                _threshold_blockers(
                    deltas,
                    max_pose_delta=max_pose_delta,
                    max_seg_delta=max_seg_delta,
                    max_score_delta=max_score_delta,
                )
            )
            if not gate["blockers"]:
                gate.update(
                    {
                        "passed": True,
                        "status": "passed_exact_cuda_component_check",
                        "ready_for_exact_eval_dispatch": True,
                        "evidence_axis": "contest_cuda",
                    }
                )
        return gate

    proxy = proxy or {}
    axis = proxy.get("axis")
    gate["proxy_axis"] = axis
    if _local_proxy_axis(axis):
        gate["blockers"].append("mps_or_local_proxy_axis_requires_cuda_component_probe")
        gate["status"] = "blocked_mps_or_local_proxy_only"
        return gate
    if not _cuda_prefix_axis(axis):
        gate["blockers"].append("cuda_prefix_or_exact_cuda_component_evidence_missing")
        gate["status"] = "blocked_cuda_component_evidence_missing"
        return gate

    n_pairs = _positive_int(proxy.get("n_pairs"))
    if n_pairs is None or n_pairs < min_cuda_prefix_pairs:
        gate["blockers"].append(
            "cuda_prefix_pairs_below_minimum:"
            f"{n_pairs or 0}<{int(min_cuda_prefix_pairs)}"
        )
    baseline_pose = _finite_float(proxy.get("baseline_avg_posenet_dist"))
    candidate_pose = _finite_float(proxy.get("avg_posenet_dist"))
    baseline_seg = _finite_float(proxy.get("baseline_avg_segnet_dist"))
    candidate_seg = _finite_float(proxy.get("avg_segnet_dist"))
    score_delta = _finite_float(
        proxy.get("delta_vs_none_charged", proxy.get("delta_vs_none"))
    )
    missing = [
        name
        for name, value in (
            ("baseline_avg_posenet_dist", baseline_pose),
            ("avg_posenet_dist", candidate_pose),
            ("baseline_avg_segnet_dist", baseline_seg),
            ("avg_segnet_dist", candidate_seg),
            ("delta_vs_none_charged_or_delta_vs_none", score_delta),
        )
        if value is None
    ]
    if missing:
        gate["blockers"].append("cuda_prefix_component_fields_missing:" + ",".join(missing))
        return gate

    deltas = {
        "score_delta": float(score_delta),
        "pose_delta": float(candidate_pose) - float(baseline_pose),
        "seg_delta": float(candidate_seg) - float(baseline_seg),
    }
    gate.update(
        {
            "evidence_axis": axis,
            "cuda_prefix": {
                "n_pairs": n_pairs,
                "baseline_avg_posenet_dist": baseline_pose,
                "avg_posenet_dist": candidate_pose,
                "baseline_avg_segnet_dist": baseline_seg,
                "avg_segnet_dist": candidate_seg,
                "score_delta": score_delta,
            },
            "component_deltas": deltas,
        }
    )
    gate["blockers"].extend(
        _threshold_blockers(
            deltas,
            max_pose_delta=max_pose_delta,
            max_seg_delta=max_seg_delta,
            max_score_delta=max_score_delta,
        )
    )
    if not gate["blockers"]:
        gate.update(
            {
                "passed": True,
                "status": "passed_cuda_prefix_component_check",
                "ready_for_exact_eval_dispatch": True,
            }
        )
    return gate


def hdm8_selector_cuda_gate_required(
    row: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None = None,
) -> bool:
    """Return true when a row/manifest is an HDM8 selector requiring the gate."""

    for payload in (row, manifest):
        if not isinstance(payload, Mapping):
            continue
        if payload.get("cuda_component_risk_gate_required") is True:
            return True
        if payload.get("requires_hdm8_selector_cuda_component_gate") is True:
            return True
        if (
            payload.get("schema") == "hdm8_film_grain_sidecar_packet_manifest_v1"
            and payload.get("postfilter_mode") == "selector"
        ):
            return True
        if (
            payload.get("schema") == "hdm8_film_grain_sidecar_archive_manifest_v1"
            and payload.get("selector_packed_in_archive") is True
        ):
            return True
    return False


def _gate_from_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    for key in (
        "cuda_component_risk_gate",
        "hdm8_selector_cuda_component_risk_gate",
    ):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    policy = payload.get("cuda_transfer_policy")
    if isinstance(policy, Mapping):
        value = policy.get("cuda_component_risk_gate")
        if isinstance(value, Mapping):
            return value
    return None


def validate_hdm8_selector_cuda_component_gate(
    gate: Mapping[str, Any] | None,
    *,
    expected_archive_sha256: str | None = None,
) -> list[str]:
    """Return fail-closed blockers for a required selector gate."""

    if not isinstance(gate, Mapping):
        return ["hdm8_selector_cuda_component_gate_missing"]
    blockers: list[str] = []
    if gate.get("schema") != HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA:
        blockers.append("hdm8_selector_cuda_component_gate_schema_invalid")
    if gate.get("required") is not True:
        blockers.append("hdm8_selector_cuda_component_gate_not_required_true")
    if gate.get("score_claim") is not False:
        blockers.append("hdm8_selector_cuda_component_gate_score_claim_not_false")
    if gate.get("promotion_eligible") is not False:
        blockers.append("hdm8_selector_cuda_component_gate_promotion_eligible_not_false")
    if gate.get("passed") is not True:
        blockers.append("hdm8_selector_cuda_component_gate_not_passed")
    if gate.get("ready_for_exact_eval_dispatch") is not True:
        blockers.append("hdm8_selector_cuda_component_gate_not_dispatch_ready")
    if gate.get("status") not in PASSING_STATUSES:
        blockers.append(
            "hdm8_selector_cuda_component_gate_status_not_passing:"
            f"{gate.get('status')}"
        )
    if _local_proxy_axis(gate.get("proxy_axis")) or _local_proxy_axis(gate.get("evidence_axis")):
        blockers.append("hdm8_selector_cuda_component_gate_mps_or_local_axis")
    gate_blockers = gate.get("blockers")
    if isinstance(gate_blockers, list) and gate_blockers:
        blockers.append(
            "hdm8_selector_cuda_component_gate_has_blockers:"
            + ",".join(str(item) for item in gate_blockers[:6])
        )
    if expected_archive_sha256 is not None:
        candidate_sha = gate.get("candidate_archive_sha256")
        if candidate_sha != expected_archive_sha256:
            blockers.append(
                "hdm8_selector_cuda_component_gate_archive_sha_mismatch:"
                f"{candidate_sha}!={expected_archive_sha256}"
            )
    return blockers


def validate_hdm8_selector_cuda_gate_context(
    row: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None = None,
    *,
    expected_archive_sha256: str | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate a row/manifest context and return blockers plus gate facts."""

    if not hdm8_selector_cuda_gate_required(row, manifest):
        return [], {}
    gate = _gate_from_payload(manifest) or _gate_from_payload(row)
    blockers = validate_hdm8_selector_cuda_component_gate(
        gate,
        expected_archive_sha256=expected_archive_sha256,
    )
    facts: dict[str, Any] = {
        "hdm8_selector_cuda_component_gate_required": True,
        "hdm8_selector_cuda_component_gate_status": gate.get("status")
        if isinstance(gate, Mapping)
        else None,
        "hdm8_selector_cuda_component_gate_passed": gate.get("passed")
        if isinstance(gate, Mapping)
        else None,
        "hdm8_selector_cuda_component_gate_evidence_axis": gate.get("evidence_axis")
        if isinstance(gate, Mapping)
        else None,
    }
    if isinstance(gate, Mapping):
        facts["hdm8_selector_cuda_component_gate"] = dict(gate)
    return blockers, facts


__all__ = [
    "DEFAULT_HDM8_CUDA_REFERENCE_RESULT",
    "DEFAULT_MAX_POSE_DELTA",
    "DEFAULT_MAX_SCORE_DELTA",
    "DEFAULT_MAX_SEG_DELTA",
    "DEFAULT_MIN_CUDA_PREFIX_PAIRS",
    "HDM8_SELECTOR_CUDA_COMPONENT_GATE_SCHEMA",
    "build_hdm8_selector_cuda_component_gate",
    "hdm8_selector_cuda_gate_required",
    "validate_hdm8_selector_cuda_component_gate",
    "validate_hdm8_selector_cuda_gate_context",
]
