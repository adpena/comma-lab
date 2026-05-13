"""Shared canonical parser for contest_auth_eval-style JSON artifacts."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuthEvalRecord:
    score: float | None
    archive_bytes: int | None
    archive_sha256: str | None
    avg_segnet_dist: float | None
    avg_posenet_dist: float | None
    rate_unscaled: float | None
    samples: int | None
    device: str
    gpu_t4_match: bool
    promotion_eligible: bool
    score_claim_valid: bool
    evidence_grade: str
    score_axis: str
    cpu_leaderboard_reproduction_eligible: bool
    rank_or_kill_eligible: bool
    hardware_compliance_blocker: str | None


def _get(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = payload
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    return value


def inflated_output_manifest_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Return a compact summary of ``contest_auth_eval`` raw-output custody.

    This is deliberately separate from :class:`AuthEvalRecord`: same
    archive/runtime score artifacts can have different inflated frames when the
    runtime branches on CPU vs CUDA. The summary lets paired-axis tools compare
    decoded-frame custody without making it a score-promotion requirement.
    """

    if not isinstance(payload, dict):
        return None
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    candidates = (
        payload.get("inflated_output_manifest"),
        payload.get("inflated_outputs_manifest"),
        provenance.get("inflated_output_manifest"),
        provenance.get("inflated_outputs_manifest"),
    )
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        nested_payload = candidate.get("payload")
        manifest = nested_payload if isinstance(nested_payload, dict) else candidate
        aggregate_sha256 = manifest.get("aggregate_sha256")
        if not isinstance(aggregate_sha256, str) or not aggregate_sha256:
            continue
        summary = {
            "aggregate_sha256": aggregate_sha256,
            "raw_file_count": _int(manifest.get("raw_file_count")),
            "total_bytes": _int(manifest.get("total_bytes")),
            "manifest_path": candidate.get("path"),
            "manifest_sha256": candidate.get("sha256"),
        }
        files = _inflated_output_files_summary(manifest.get("files"))
        if files:
            summary["files"] = files
        return summary
    return None


def _inflated_output_files_summary(files: Any) -> list[dict[str, Any]]:
    if not isinstance(files, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            continue
        sha = item.get("sha256")
        if not isinstance(sha, str) or not sha:
            continue
        row: dict[str, Any] = {
            "sha256": sha,
            "bytes": _int(item.get("bytes")),
        }
        for key in ("video_name", "relative_path"):
            value = item.get(key)
            if isinstance(value, str) and value:
                row[key] = value
        exists = item.get("exists")
        if isinstance(exists, bool):
            row["exists"] = exists
        rows.append(row)
    return rows


def runtime_tree_sha256(payload: dict[str, Any]) -> str | None:
    """Return the preferred runtime content/tree SHA from auth-eval payloads."""

    if not isinstance(payload, dict):
        return None
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    manifest = payload.get("inflate_runtime_manifest")
    provenance_manifest = provenance.get("inflate_runtime_manifest")
    if not isinstance(manifest, dict):
        manifest = {}
    if not isinstance(provenance_manifest, dict):
        provenance_manifest = {}
    candidates = (
        payload.get("runtime_content_tree_sha256"),
        payload.get("inflate_runtime_content_tree_sha256"),
        provenance.get("runtime_content_tree_sha256"),
        provenance_manifest.get("runtime_content_tree_sha256"),
        manifest.get("runtime_content_tree_sha256"),
        payload.get("runtime_tree_sha256"),
        payload.get("inflate_runtime_tree_sha256"),
        provenance.get("runtime_tree_sha256"),
        provenance_manifest.get("runtime_tree_sha256"),
        manifest.get("runtime_tree_sha256"),
    )
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return None


def _float(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _strict_bool(value: Any) -> bool:
    return value is True


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _float(value)
        if parsed is not None:
            return parsed
    return None


def _first_int(*values: Any) -> int | None:
    for value in values:
        parsed = _int(value)
        if parsed is not None:
            return parsed
    return None


def _parse_report_text(value: Any) -> dict[str, float | int] | None:
    if not isinstance(value, str) or "Evaluation results over" not in value:
        return None

    def _grab_float(pattern: str) -> float | None:
        match = re.search(pattern, value)
        return float(match.group(1).replace(",", "")) if match else None

    pose = _grab_float(r"Average PoseNet Distortion:\s*([0-9.eE+,-]+)")
    seg = _grab_float(r"Average SegNet Distortion:\s*([0-9.eE+,-]+)")
    rate = _grab_float(r"Compression Rate:\s*([0-9.eE+,-]+)")
    final = _grab_float(r"Final score:.*=\s*([0-9.eE+,-]+)")
    sample_match = re.search(r"Evaluation results over (\d+) samples", value)
    samples = int(sample_match.group(1)) if sample_match else None
    out: dict[str, float | int] = {}
    if pose is not None:
        out["avg_posenet_dist"] = pose
    if seg is not None:
        out["avg_segnet_dist"] = seg
    if rate is not None:
        out["rate_unscaled"] = rate
    if final is not None:
        out["final_score"] = final
    if samples is not None:
        out["n_samples"] = samples
    if pose is not None and seg is not None and rate is not None:
        out["canonical_score"] = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * rate
    return out or None


def _platform_is_linux_x86_64(provenance: dict[str, Any], payload: dict[str, Any]) -> bool:
    system = str(
        provenance.get("platform_system")
        or payload.get("platform_system")
        or ""
    )
    machine = str(
        provenance.get("platform_machine")
        or payload.get("platform_machine")
        or ""
    ).lower()
    hardware = str(provenance.get("hardware") or payload.get("hardware") or "").lower()
    return (
        system == "Linux" and machine in {"x86_64", "amd64"}
    ) or "linux x86_64" in hardware or "github-actions-ubuntu-latest-x86_64" in hardware


def _platform_is_macos(provenance: dict[str, Any], payload: dict[str, Any]) -> bool:
    system = str(
        provenance.get("platform_system")
        or payload.get("platform_system")
        or ""
    )
    hardware = str(provenance.get("hardware") or payload.get("hardware") or "").lower()
    return system == "Darwin" or "macos" in hardware or "apple silicon" in hardware


def _device_norm(device: str) -> str:
    return str(device or "").lower()


def _is_contest_cuda_axis(*, device: str, samples: int | None, gpu_t4_match: bool) -> bool:
    return _device_norm(device) == "cuda" and samples == 600 and gpu_t4_match


def _is_contest_cpu_axis(
    *,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    device: str,
    samples: int | None,
) -> bool:
    return (
        _device_norm(device) == "cpu"
        and samples == 600
        and _platform_is_linux_x86_64(provenance, payload)
    )


def _declared_contest_axis(payload: dict[str, Any], axis: str) -> bool:
    wanted = axis.replace("_", "-")
    explicit = str(payload.get("score_axis") or payload.get("device_axis") or "").lower()
    if explicit.replace("_", "-") == wanted:
        return True
    grade_text = " ".join(
        str(value)
        for value in (
            payload.get("evidence_grade"),
            payload.get("lane_tag"),
            payload.get("evidence_semantics"),
        )
        if value
    ).lower()
    return wanted in grade_text or axis in grade_text


def _non_contest_axis_for_device(device: str) -> str:
    device_kind = _device_norm(device)
    if device_kind == "cpu":
        return "cpu_advisory"
    return device_kind or "unknown"


def _score_axis(
    *,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    device: str,
    samples: int | None,
    gpu_t4_match: bool,
) -> str:
    explicit = str(payload.get("score_axis") or payload.get("device_axis") or "")
    explicit_norm = explicit.lower().replace("-", "_")
    contest_cuda = _is_contest_cuda_axis(
        device=device,
        samples=samples,
        gpu_t4_match=gpu_t4_match,
    )
    contest_cpu = _is_contest_cpu_axis(
        payload=payload,
        provenance=provenance,
        device=device,
        samples=samples,
    )
    if explicit_norm == "contest_cuda" or _declared_contest_axis(payload, "contest_cuda"):
        if contest_cuda:
            return "contest_cuda"
        if contest_cpu:
            return "contest_cpu"
        return _non_contest_axis_for_device(device)
    if explicit_norm == "contest_cpu" or _declared_contest_axis(payload, "contest_cpu"):
        if contest_cpu:
            return "contest_cpu"
        if contest_cuda:
            return "contest_cuda"
        return _non_contest_axis_for_device(device)
    if explicit and explicit_norm not in {"cpu", "cuda", "mps"}:
        return explicit
    if contest_cuda:
        return "contest_cuda"
    if contest_cpu:
        return "contest_cpu"
    if _device_norm(device) == "cpu":
        return "cpu_advisory"
    return _device_norm(device) or "unknown"


def _hardware_compliance_blocker(
    *,
    payload: dict[str, Any],
    provenance: dict[str, Any],
    device: str,
    samples: int | None,
    gpu_t4_match: bool,
    axis: str,
) -> str | None:
    blocker = payload.get("hardware_compliance_blocker")
    if blocker is not None:
        return str(blocker)
    if axis in {"contest_cuda", "contest_cpu"}:
        return None
    device_kind = _device_norm(device)
    if device_kind == "cpu" and (
        _declared_contest_axis(payload, "contest_cpu")
        or samples == 600
        or payload.get("score_axis") == "cpu_advisory"
    ):
        if samples != 600:
            return "contest_cpu_requires_600_samples_linux_x86_64"
        if not _platform_is_linux_x86_64(provenance, payload):
            return "contest_cpu_requires_linux_x86_64"
    if device_kind == "cuda" and (
        _declared_contest_axis(payload, "contest_cuda") or samples == 600 or gpu_t4_match
    ):
        if samples != 600:
            return "contest_cuda_requires_600_samples_t4"
        if not gpu_t4_match:
            return "contest_cuda_requires_t4"
    return None


def parse_auth_eval_payload(payload: dict[str, Any]) -> AuthEvalRecord | None:
    """Parse a score JSON using canonical auth-eval fields first.

    ``final_score`` is a display-rounded upstream report field in many
    artifacts. Ranking tools must prefer ``canonical_score`` and
    ``score_recomputed_from_components`` when present.
    """
    if not isinstance(payload, dict):
        return None
    raw_provenance = payload.get("provenance")
    provenance_present = isinstance(raw_provenance, dict)
    provenance = raw_provenance if provenance_present else {}
    report = _parse_report_text(payload.get("report_text")) or {}
    top_level_seg = _first_float(
        payload.get("avg_segnet_dist"),
        payload.get("seg_dist_avg"),
        _get(payload, "components", "segnet_avg"),
        payload.get("segnet_distortion"),
        payload.get("segnet"),
    )
    top_level_pose = _first_float(
        payload.get("avg_posenet_dist"),
        payload.get("pose_dist_avg"),
        _get(payload, "components", "posenet_avg"),
        payload.get("posenet_distortion"),
        payload.get("posenet"),
        payload.get("pose_distortion"),
    )
    top_level_rate = _first_float(
        payload.get("rate_unscaled"),
        payload.get("rate"),
        payload.get("compression_rate"),
        _get(payload, "components", "rate"),
    )
    top_level_components_complete = (
        top_level_seg is not None
        and top_level_pose is not None
        and top_level_rate is not None
    )
    score_recomputed = _float(payload.get("score_recomputed_from_components"))
    canonical_score_recomputed = _float(payload.get("canonical_score_recomputed"))

    score = _first_float(
        score_recomputed
        if top_level_components_complete or (score_recomputed not in {None, 0.0})
        else None,
        report.get("canonical_score"),
        payload.get("canonical_score"),
        canonical_score_recomputed
        if top_level_components_complete or (canonical_score_recomputed not in {None, 0.0})
        else None,
        payload.get("score"),
        payload.get("total_score"),
        payload.get("final_score"),
        report.get("final_score"),
        _get(payload, "result", "final_score"),
    )
    if score is None:
        return None

    archive_bytes = _first_int(
        payload.get("archive_size_bytes"),
        payload.get("archive_bytes"),
        payload.get("archive_size"),
        payload.get("bytes"),
    )
    device_value = provenance.get("device") or payload.get("device")
    device = str(device_value or "")
    samples = _first_int(payload.get("n_samples"), payload.get("samples"), report.get("n_samples"))
    gpu_t4_raw = (
        provenance.get("gpu_t4_match")
        if provenance_present
        else payload.get("gpu_t4_match")
    )
    gpu_t4_match = _strict_bool(gpu_t4_raw)
    promotion_eligible = _strict_bool(payload.get("promotion_eligible"))
    score_claim_valid = _strict_bool(payload.get("score_claim_valid"))
    promotion_blockers = payload.get("promotion_blockers")
    rank_or_kill_blockers = payload.get("rank_or_kill_blockers")
    has_promotion_blockers = isinstance(promotion_blockers, list) and bool(promotion_blockers)
    has_rank_or_kill_blockers = isinstance(rank_or_kill_blockers, list) and bool(rank_or_kill_blockers)
    axis = _score_axis(
        payload=payload,
        provenance=provenance,
        device=device,
        samples=samples,
        gpu_t4_match=gpu_t4_match,
    )
    if axis != "contest_cuda":
        promotion_eligible = False
        score_claim_valid = False
    if has_promotion_blockers:
        promotion_eligible = False
    cpu_leaderboard_reproduction_eligible = (
        _strict_bool(payload.get("cpu_leaderboard_reproduction_eligible"))
        if "cpu_leaderboard_reproduction_eligible" in payload
        else axis == "contest_cpu"
    ) and axis == "contest_cpu"
    rank_or_kill_eligible = (
        _strict_bool(payload.get("rank_or_kill_eligible"))
        if "rank_or_kill_eligible" in payload
        else False
    ) and promotion_eligible and axis == "contest_cuda"
    if has_rank_or_kill_blockers:
        rank_or_kill_eligible = False
    evidence_grade = str(payload.get("evidence_grade") or "")
    evidence_grade_norm = evidence_grade.lower()
    if axis == "contest_cuda":
        evidence_grade = evidence_grade or "A++"
    elif axis == "contest_cpu":
        evidence_grade = evidence_grade or "contest-CPU"
    elif axis == "cpu_advisory":
        if not evidence_grade or "contest-cpu" in evidence_grade_norm or "contest_cpu" in evidence_grade_norm:
            evidence_grade = "macOS-CPU advisory" if _platform_is_macos(provenance, payload) else "CPU advisory"
    elif _device_norm(device) == "cuda" and (
        not evidence_grade
        or "contest-cuda" in evidence_grade_norm
        or "contest_cuda" in evidence_grade_norm
        or evidence_grade_norm in {"a", "a++"}
    ):
        evidence_grade = "A"
    elif not evidence_grade:
        evidence_grade = "invalid"
    hardware_compliance_blocker = _hardware_compliance_blocker(
        payload=payload,
        provenance=provenance,
        device=device,
        samples=samples,
        gpu_t4_match=gpu_t4_match,
        axis=axis,
    )

    return AuthEvalRecord(
        score=score,
        archive_bytes=archive_bytes,
        archive_sha256=str(provenance.get("archive_sha256") or payload.get("archive_sha256") or payload.get("sha256") or "") or None,
        avg_segnet_dist=_first_float(top_level_seg, report.get("avg_segnet_dist")),
        avg_posenet_dist=_first_float(top_level_pose, report.get("avg_posenet_dist")),
        rate_unscaled=_first_float(top_level_rate, report.get("rate_unscaled")),
        samples=samples,
        device=device,
        gpu_t4_match=gpu_t4_match,
        promotion_eligible=promotion_eligible,
        score_claim_valid=score_claim_valid,
        evidence_grade=evidence_grade,
        score_axis=axis,
        cpu_leaderboard_reproduction_eligible=cpu_leaderboard_reproduction_eligible,
        rank_or_kill_eligible=rank_or_kill_eligible,
        hardware_compliance_blocker=hardware_compliance_blocker,
    )
