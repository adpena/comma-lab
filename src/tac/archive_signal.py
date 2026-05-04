"""Offline archive-byte and component-trace signal ranking.

This module is intentionally diagnostic-only.  It reads existing JSON artifacts
and ZIP byte profiles, but it does not inflate archives, load scorers, claim
scores, or dispatch GPU work.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.archive_byte_profile import CONTEST_ORIGINAL_BYTES, profile_archive

SCHEMA = "archive_signal_table_v1"
TOOL = "src/tac/archive_signal.py"
EVIDENCE_GRADE = "diagnostic_observability"
RATE_LAMBDA = 25.0 / CONTEST_ORIGINAL_BYTES
DEFAULT_FRONTIER_LABEL = "PR79/S2"
DEFAULT_FRONTIER_SCORE = 0.31453355357318635


class ArchiveSignalError(ValueError):
    """Raised when an observability artifact is malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def load_json(path: Path | str) -> dict[str, Any]:
    json_path = Path(path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArchiveSignalError(f"{json_path} must contain a JSON object")
    payload["_source_path"] = str(json_path)
    payload["_source_sha256"] = sha256_file(json_path)
    return payload


def contest_rate_term(byte_count: int | float) -> float:
    return float(byte_count) * RATE_LAMBDA


def _finite_float(value: Any, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ArchiveSignalError(f"{field} must be finite, got {value!r}")
    return out


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _score_from_components(*, avg_pose: float, avg_seg: float, archive_bytes: int) -> dict[str, float]:
    pose = math.sqrt(10.0 * avg_pose)
    seg = 100.0 * avg_seg
    rate = contest_rate_term(archive_bytes)
    return {
        "score_pose_contribution": pose,
        "score_seg_contribution": seg,
        "score_rate_contribution": rate,
        "score_recomputed_from_components": pose + seg + rate,
    }


def baseline_from_exact_eval(
    path: Path | str,
    *,
    label: str = DEFAULT_FRONTIER_LABEL,
    fallback_score: float = DEFAULT_FRONTIER_SCORE,
) -> dict[str, Any]:
    payload = load_json(path)
    archive_bytes = int(payload["archive_size_bytes"])
    avg_pose = _finite_float(payload["avg_posenet_dist"], field="avg_posenet_dist")
    avg_seg = _finite_float(payload["avg_segnet_dist"], field="avg_segnet_dist")
    score = payload.get("score_recomputed_from_components", payload.get("canonical_score"))
    if score is None:
        score = fallback_score
    components = _score_from_components(
        avg_pose=avg_pose,
        avg_seg=avg_seg,
        archive_bytes=archive_bytes,
    )
    return {
        "label": label,
        "source_json": payload["_source_path"],
        "source_json_sha256": payload["_source_sha256"],
        "archive_bytes": archive_bytes,
        "n_samples": int(payload.get("n_samples", 0)),
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": avg_seg,
        "score_recomputed_from_components": _finite_float(
            score,
            field="score_recomputed_from_components",
        ),
        "score_pose_contribution": components["score_pose_contribution"],
        "score_seg_contribution": components["score_seg_contribution"],
        "score_rate_contribution": components["score_rate_contribution"],
        "canonical_score_source": payload.get("canonical_score_source"),
        "archive_sha256": _dig(payload, "provenance", "archive_sha256")
        or _dig(payload, "provenance", "archive", "sha256"),
    }


def synthetic_baseline(
    *,
    archive_bytes: int,
    score: float = DEFAULT_FRONTIER_SCORE,
    label: str = DEFAULT_FRONTIER_LABEL,
) -> dict[str, Any]:
    return {
        "label": label,
        "source_json": None,
        "source_json_sha256": None,
        "archive_bytes": int(archive_bytes),
        "n_samples": None,
        "avg_posenet_dist": None,
        "avg_segnet_dist": None,
        "score_recomputed_from_components": float(score),
        "score_pose_contribution": None,
        "score_seg_contribution": None,
        "score_rate_contribution": contest_rate_term(archive_bytes),
        "canonical_score_source": "operator_supplied_frontier_score",
        "archive_sha256": None,
    }


def _dig(mapping: Mapping[str, Any], *path: str) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def _rate_delta_record(candidate_bytes: int, baseline: Mapping[str, Any]) -> dict[str, Any]:
    baseline_bytes = int(baseline["archive_bytes"])
    delta = int(candidate_bytes) - baseline_bytes
    rate_delta = contest_rate_term(delta)
    return {
        "baseline_label": baseline["label"],
        "baseline_archive_bytes": baseline_bytes,
        "delta_bytes_vs_baseline_archive": delta,
        "rate_score_delta_vs_baseline_archive": rate_delta,
        "component_worsening_budget_before_equal_total": max(0.0, -rate_delta),
        "component_improvement_required_to_equal_total": max(0.0, rate_delta),
    }


def _rate_potential(byte_count: int | None) -> float | None:
    if byte_count is None:
        return None
    return contest_rate_term(max(0, int(byte_count)))


def _stream_class(name: str) -> str:
    low = name.lower()
    if "mask" in low or "qma9" in low:
        return "mask"
    if "pose" in low or "qp1" in low or "p1d1" in low:
        return "pose"
    if "renderer" in low or "model" in low or "qh0" in low or "qzs3" in low:
        return "renderer_model"
    if "action" in low or "router" in low or "post" in low or "shift" in low or "frac" in low:
        return "action_control"
    if "zip" in low or "header" in low or "container" in low:
        return "container"
    return "unknown"


def _signal_row(
    *,
    source_label: str,
    source_kind: str,
    name: str,
    byte_count: int | None,
    evidence_grade: str,
    row_kind: str = "stream",
    codec: str | None = None,
    source_json: str | None = None,
    archive_bytes: int | None = None,
    baseline: Mapping[str, Any] | None = None,
    notes: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "row_kind": row_kind,
        "source_label": source_label,
        "source_kind": source_kind,
        "name": name,
        "stream_class": _stream_class(name),
        "byte_count": None if byte_count is None else int(byte_count),
        "rate_potential_if_fully_removed": _rate_potential(byte_count),
        "codec": codec,
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "source_json": source_json,
        "notes": list(notes),
    }
    if archive_bytes is not None:
        row["source_archive_bytes"] = int(archive_bytes)
        if baseline is not None:
            row.update(_rate_delta_record(int(archive_bytes), baseline))
    if extra:
        row.update(dict(extra))
    row["priority_score"] = _priority_score(row)
    return row


def _priority_score(row: Mapping[str, Any]) -> float:
    rate_potential = row.get("rate_potential_if_fully_removed")
    if row.get("row_kind") == "stream":
        stream_specific = [
            row.get("stream_component_worsening_budget_before_equal_rate"),
            row.get("stream_component_improvement_required_to_equal_rate"),
            rate_potential,
        ]
        finite_stream = [
            float(value)
            for value in stream_specific
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        return max(finite_stream) if finite_stream else 0.0

    if row.get("row_kind") == "atom":
        finite_atom = [
            float(value)
            for value in [row.get("component_score_contribution"), rate_potential]
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        return max(finite_atom) if finite_atom else 0.0

    score_delta = row.get("component_score_delta_vs_baseline")
    if row.get("row_kind") == "component_trace_summary" and isinstance(score_delta, (int, float)):
        return abs(float(score_delta))

    candidates = [
        row.get("component_worsening_budget_before_equal_total"),
        row.get("component_score_contribution"),
        score_delta,
        rate_potential,
    ]
    finite = [float(value) for value in candidates if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return max(finite) if finite else 0.0


def rows_from_archive_profile(path_or_profile: Path | str | Mapping[str, Any], baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    if isinstance(path_or_profile, Mapping):
        payload = dict(path_or_profile)
        source_json = payload.get("_source_path")
    else:
        path = Path(path_or_profile)
        if path.suffix.lower() == ".zip":
            payload = profile_archive(path)
            source_json = str(path)
        else:
            payload = load_json(path)
            source_json = str(path)

    archives = payload.get("archives") if isinstance(payload.get("archives"), list) else [payload]
    rows: list[dict[str, Any]] = []
    for archive in archives:
        if not isinstance(archive, Mapping) or archive.get("valid_profile") is False:
            continue
        label = str(archive.get("archive_name") or Path(str(archive.get("archive_path", "archive"))).name)
        archive_bytes = _optional_int(archive.get("total_bytes"))
        for member in archive.get("members", []):
            if not isinstance(member, Mapping) or member.get("is_dir"):
                continue
            rows.append(
                _signal_row(
                    source_label=label,
                    source_kind="archive_byte_profile",
                    name=str(member["name"]),
                    byte_count=int(member["compressed_size"]),
                    codec=str(member.get("method")),
                    evidence_grade=str(archive.get("evidence_grade", "byte_profile_only")),
                    source_json=source_json,
                    archive_bytes=archive_bytes,
                    baseline=baseline,
                    extra={
                        "uncompressed_size": int(member.get("uncompressed_size", 0)),
                        "member_sha256": member.get("sha256"),
                        "member_rate_term": member.get("rate_term"),
                    },
                )
            )
    return rows


def rows_from_pr81_profile(path: Path | str, baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = load_json(path)
    archive = payload.get("archive") or {}
    archive_bytes = int(archive["bytes"])
    deltas = {
        (row.get("component"), row.get("reference")): row
        for row in payload.get("component_byte_deltas", [])
        if isinstance(row, Mapping)
    }
    name_to_component = {
        "range_mask.qma9": "mask_or_range_mask",
        "split_model_reordered.br_bundle": "model_or_renderer",
        "optimized_poses.qp1.br": "pose",
        "router_actions.3bit": "router_or_actions",
    }
    rows: list[dict[str, Any]] = []
    for segment in _dig(payload, "payload_split", "segments") or []:
        if not isinstance(segment, Mapping):
            continue
        component = name_to_component.get(str(segment["name"]))
        ref_delta = deltas.get((component, "PR79_S2")) or deltas.get((component, DEFAULT_FRONTIER_LABEL))
        extra: dict[str, Any] = {
            "segment_offset": int(segment.get("offset", 0)),
            "segment_sha256": segment.get("sha256"),
            "reference_component": component,
        }
        if isinstance(ref_delta, Mapping) and ref_delta.get("available"):
            delta_bytes = int(ref_delta["delta_bytes_pr81_minus_reference"])
            rate_delta = contest_rate_term(delta_bytes)
            extra.update(
                {
                    "delta_bytes_vs_reference_stream": delta_bytes,
                    "reference_stream_bytes": int(ref_delta["reference_bytes"]),
                    "rate_score_delta_vs_reference_stream": rate_delta,
                    "stream_component_worsening_budget_before_equal_rate": max(0.0, -rate_delta),
                    "stream_component_improvement_required_to_equal_rate": max(0.0, rate_delta),
                }
            )
        rows.append(
            _signal_row(
                source_label="PR81",
                source_kind="public_static_profile",
                name=str(segment["name"]),
                byte_count=int(segment["bytes"]),
                codec=str(segment.get("codec")),
                evidence_grade=str(payload.get("evidence_grade", "external/planning_only")),
                source_json=payload["_source_path"],
                archive_bytes=archive_bytes,
                baseline=baseline,
                notes=["Static public profile. Exact replay/eval required before score use."],
                extra=extra,
            )
        )
    return rows


def rows_from_pr82_profile(path: Path | str, baseline: Mapping[str, Any]) -> list[dict[str, Any]]:
    payload = load_json(path)
    archive_bytes = int(_dig(payload, "zip_container", "archive_bytes"))
    evidence = str(payload.get("evidence_grade", "empirical_static_archive_profile"))
    rows: list[dict[str, Any]] = []
    for segment in _dig(payload, "compact_bundle", "segments") or []:
        if not isinstance(segment, Mapping):
            continue
        rows.append(
            _signal_row(
                source_label="PR82",
                source_kind="public_static_profile",
                name=str(segment["name"]),
                byte_count=int(segment["encoded_bytes"]),
                codec="brotli" if segment.get("brotli_decodable") else "encoded",
                evidence_grade=evidence,
                source_json=payload["_source_path"],
                archive_bytes=archive_bytes,
                baseline=baseline,
                notes=["Static public profile. Exact replay/eval required before score use."],
                extra={
                    "encoded_offset": int(segment.get("encoded_offset", 0)),
                    "decoded_bytes": segment.get("decoded_bytes"),
                    "decoded_magic8_ascii": segment.get("decoded_magic8_ascii"),
                    "segment_sha256": segment.get("encoded_sha256"),
                },
            )
        )
    for record in _dig(payload, "anatomy", "model_qh0", "top_records_by_bytes") or []:
        if not isinstance(record, Mapping):
            continue
        rows.append(
            _signal_row(
                source_label="PR82",
                source_kind="public_static_model_anatomy",
                row_kind="atom",
                name=f"model_qh0:{record.get('name')}",
                byte_count=int(record.get("record_bytes", 0)),
                codec=str(record.get("quantization")),
                evidence_grade=evidence,
                source_json=payload["_source_path"],
                archive_bytes=archive_bytes,
                baseline=baseline,
                notes=["Layer byte anatomy only; not a standalone transfer proof."],
                extra={"shape": record.get("shape")},
            )
        )
    return rows


def component_rows_from_trace(path: Path | str, baseline: Mapping[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    payload = load_json(path)
    if payload.get("score_claim") is not False:
        raise ArchiveSignalError(f"{path} is not a diagnostic component trace")
    archive_bytes = int(payload["archive_size_bytes"])
    trace_score = _finite_float(
        payload.get("score_recomputed_from_components"),
        field="score_recomputed_from_components",
    )
    score_delta = trace_score - float(baseline["score_recomputed_from_components"])
    summary = _signal_row(
        source_label=Path(str(path)).parent.name,
        source_kind="component_trace",
        row_kind="component_trace_summary",
        name="trace_score_delta_vs_baseline",
        byte_count=archive_bytes,
        codec=None,
        evidence_grade=str(payload.get("evidence_grade", "diagnostic_component_trace")),
        source_json=payload["_source_path"],
        archive_bytes=archive_bytes,
        baseline=baseline,
        notes=["Diagnostic component trace; only actionable after cross-check against exact eval for identical bytes."],
        extra={
            "component_score_delta_vs_baseline": score_delta,
            "trace_score_recomputed_from_components": trace_score,
            "trace_avg_posenet_dist": payload.get("avg_posenet_dist"),
            "trace_avg_segnet_dist": payload.get("avg_segnet_dist"),
            "contest_auth_eval_cross_check": payload.get("contest_auth_eval_cross_check"),
        },
    )
    rows = [summary]
    samples = payload.get("top_combined_samples") or []
    if not samples and isinstance(payload.get("samples"), list):
        samples = sorted(
            payload["samples"],
            key=lambda row: float(row.get("score_combined_contribution_first_order", 0.0)),
            reverse=True,
        )
    for sample in samples[: max(0, top_k)]:
        if not isinstance(sample, Mapping):
            continue
        contribution = _finite_float(
            sample.get("score_combined_contribution_first_order", 0.0),
            field="score_combined_contribution_first_order",
        )
        rows.append(
            _signal_row(
                source_label=Path(str(path)).parent.name,
                source_kind="component_trace",
                row_kind="component_atom",
                name=f"pair_{int(sample['pair_index']):04d}",
                byte_count=None,
                codec=None,
                evidence_grade=str(payload.get("evidence_grade", "diagnostic_component_trace")),
                source_json=payload["_source_path"],
                archive_bytes=archive_bytes,
                baseline=baseline,
                notes=["Rate-equivalent pair budget is a first-order ranking signal, not an exact additive proof."],
                extra={
                    "pair_index": int(sample["pair_index"]),
                    "frame_indices": sample.get("frame_indices"),
                    "video_name": sample.get("video_name"),
                    "posenet_dist": sample.get("posenet_dist"),
                    "segnet_dist": sample.get("segnet_dist"),
                    "component_score_contribution": contribution,
                    "rate_equivalent_break_even_bytes": contribution / RATE_LAMBDA,
                    "score_pose_contribution_first_order": sample.get("score_pose_contribution_first_order"),
                    "score_seg_contribution_exact": sample.get("score_seg_contribution_exact"),
                },
            )
        )
    return rows


def expand_globs(patterns: Iterable[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for item in sorted(glob.glob(pattern, recursive=True)):
            path = Path(item)
            if path.is_file() and path not in seen:
                out.append(path)
                seen.add(path)
    return out


def build_signal_table(
    *,
    baseline_exact_json: Path | str | None,
    baseline_archive_bytes: int | None = None,
    baseline_score: float = DEFAULT_FRONTIER_SCORE,
    archive_paths: Sequence[Path | str] = (),
    archive_profile_jsons: Sequence[Path | str] = (),
    pr81_profile_json: Path | str | None = None,
    pr82_profile_json: Path | str | None = None,
    component_trace_jsons: Sequence[Path | str] = (),
    component_trace_globs: Sequence[str] = (),
    top_k: int = 12,
) -> dict[str, Any]:
    if baseline_exact_json is not None and Path(baseline_exact_json).exists():
        baseline = baseline_from_exact_eval(baseline_exact_json)
    elif baseline_archive_bytes is not None:
        baseline = synthetic_baseline(archive_bytes=baseline_archive_bytes, score=baseline_score)
    else:
        raise ArchiveSignalError("baseline_exact_json or baseline_archive_bytes is required")

    stream_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    def add_source(path: Path | str, kind: str) -> None:
        source_path = Path(path)
        sources.append(
            {
                "kind": kind,
                "path": str(source_path),
                "exists": source_path.exists(),
                "sha256": sha256_file(source_path) if source_path.is_file() else None,
            }
        )

    for archive in archive_paths:
        add_source(archive, "archive_zip")
        stream_rows.extend(rows_from_archive_profile(archive, baseline))
    for profile_json in archive_profile_jsons:
        add_source(profile_json, "archive_profile_json")
        stream_rows.extend(rows_from_archive_profile(profile_json, baseline))
    if pr81_profile_json is not None and Path(pr81_profile_json).exists():
        add_source(pr81_profile_json, "pr81_static_profile")
        stream_rows.extend(rows_from_pr81_profile(pr81_profile_json, baseline))
    if pr82_profile_json is not None and Path(pr82_profile_json).exists():
        add_source(pr82_profile_json, "pr82_static_profile")
        stream_rows.extend(rows_from_pr82_profile(pr82_profile_json, baseline))

    trace_paths = [Path(path) for path in component_trace_jsons if Path(path).exists()]
    trace_paths.extend(expand_globs(component_trace_globs))
    seen_trace_paths: set[Path] = set()
    for trace_path in trace_paths:
        if trace_path in seen_trace_paths:
            continue
        seen_trace_paths.add(trace_path)
        add_source(trace_path, "component_trace_json")
        component_rows.extend(component_rows_from_trace(trace_path, baseline, top_k=top_k))

    ranked_rows = sorted(
        stream_rows + component_rows,
        key=lambda row: (
            -float(row.get("priority_score", 0.0)),
            str(row.get("source_label")),
            str(row.get("name")),
        ),
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "evidence_grade": EVIDENCE_GRADE,
        "rate_formula": f"25 * bytes / {CONTEST_ORIGINAL_BYTES}",
        "promotion_policy": (
            "This table is offline planning signal. It may guide the next dispatch, "
            "but exact CUDA auth eval on identical archive bytes is required for any score claim."
        ),
        "baseline": baseline,
        "sources": sources,
        "stream_signal_rows": sorted(
            stream_rows,
            key=lambda row: (-float(row.get("priority_score", 0.0)), str(row.get("source_label")), str(row.get("name"))),
        ),
        "component_signal_rows": sorted(
            component_rows,
            key=lambda row: (-float(row.get("priority_score", 0.0)), str(row.get("source_label")), str(row.get("name"))),
        ),
        "ranked_signal_rows": ranked_rows,
        "top_dispatch_guidance": ranked_rows[: max(0, top_k)],
    }


def render_markdown(table: Mapping[str, Any], *, top_k: int = 16) -> str:
    baseline = table["baseline"]
    lines = [
        "# Replay Observability Signal",
        "",
        f"- schema: `{table['schema']}`",
        f"- evidence_grade: `{table['evidence_grade']}`",
        f"- score_claim: `{table['score_claim']}`",
        f"- dispatch_performed: `{table['dispatch_performed']}`",
        f"- baseline: `{baseline['label']}` bytes=`{baseline['archive_bytes']}` score=`{baseline['score_recomputed_from_components']}`",
        "",
        table["promotion_policy"],
        "",
        "## Top Signals",
        "",
        "| rank | source | kind | name | bytes | priority | rate delta vs baseline | guidance |",
        "|---:|---|---|---|---:|---:|---:|---|",
    ]
    for idx, row in enumerate(table.get("ranked_signal_rows", [])[:top_k], start=1):
        guidance = "exact CUDA eval required before score use"
        if row.get("row_kind") == "component_atom":
            guidance = "repair/allocate if a charged atom can buy this pair budget"
        elif row.get("component_worsening_budget_before_equal_total", 0.0) > 0:
            guidance = "byte saving can absorb this much component worsening"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("source_label")),
                    str(row.get("row_kind")),
                    str(row.get("name")),
                    "" if row.get("byte_count") is None else str(row.get("byte_count")),
                    f"{float(row.get('priority_score', 0.0)):.9f}",
                    f"{float(row.get('rate_score_delta_vs_baseline_archive', 0.0)):.9f}",
                    guidance,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Source Count",
            "",
            f"- sources: `{len(table.get('sources', []))}`",
            f"- stream rows: `{len(table.get('stream_signal_rows', []))}`",
            f"- component rows: `{len(table.get('component_signal_rows', []))}`",
        ]
    )
    return "\n".join(lines) + "\n"


def write_signal_outputs(
    table: Mapping[str, Any],
    *,
    json_out: Path | None = None,
    markdown_out: Path | None = None,
    markdown_top_k: int = 16,
) -> None:
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_bytes(json_bytes(table))
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(render_markdown(table, top_k=markdown_top_k), encoding="utf-8")
