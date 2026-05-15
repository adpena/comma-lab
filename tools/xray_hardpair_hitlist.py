#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a deterministic hard-pair hitlist from XRay diagnostic artifacts.

Inputs are diagnostic-only artifacts produced by tools such as
``tools/xray_pair_component_errors.py`` and
``tools/xray_paired_cpu_cuda_axis_delta.py``. This utility does not score,
dispatch, promote, retire, or rank candidates for submission. It only turns
per-pair component tails plus paired CPU/CUDA axis deltas into an operator
hitlist for selector, film-grain, foveation, and repair work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "xray_hardpair_hitlist_v1"
TOOL = "tools/xray_hardpair_hitlist.py"
DEFAULT_ORIGINAL_UNCOMPRESSED_SIZE_BYTES = 37_545_489
FALSE_AUTHORITY = {
    "research_only": True,
    "score_claim": False,
    "dispatch_attempted": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


@dataclass(frozen=True)
class PairObservation:
    pair_idx: int
    source_path: str
    source_label: str
    device: str | None
    evidence_grade: str | None
    pose_score_contribution: float
    seg_score_contribution: float
    component_score_no_rate: float
    frame0_l1: float | None
    frame1_l1: float | None
    raw: dict[str, Any]


@dataclass(frozen=True)
class AxisContext:
    source_path: str
    source_kind: str
    classification: str | None
    dominant_component: str | None
    score_delta_byte_equivalent: float | None
    contest_cpu_byte_gap: float | None
    contest_cuda_byte_gap: float | None
    raw_output_aggregate_match: bool | None
    score_delta_cuda_minus_cpu: float | None
    seg_delta: float | None
    pose_delta: float | None
    rate_delta: float | None


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load JSON {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[Any]:
    rows: list[Any] = []
    try:
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSONL: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"could not load JSONL {path}: {exc}") from exc
    return rows


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value.replace(",", ""))
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def _score_terms(row: dict[str, Any]) -> tuple[float, float, float]:
    pose_term = _as_float(row.get("pose_score_contribution"))
    seg_term = _as_float(row.get("seg_score_contribution"))
    pose_dist = _as_float(row.get("pose_dist"))
    seg_dist = _as_float(row.get("seg_dist"))
    if pose_term is None and pose_dist is not None:
        pose_term = math.sqrt(max(0.0, 10.0 * pose_dist))
    if seg_term is None and seg_dist is not None:
        seg_term = 100.0 * seg_dist
    pose = pose_term or 0.0
    seg = seg_term or 0.0
    component = _as_float(row.get("component_score_no_rate"))
    if component is None:
        component = pose + seg
    return pose, seg, component


def load_pair_xray(path: Path) -> list[PairObservation]:
    suffix = path.suffix.lower()
    payload = _load_jsonl(path) if suffix == ".jsonl" else _load_json(path)
    if isinstance(payload, dict):
        source_label = str(payload.get("label") or path.stem)
        device = payload.get("device")
        evidence_grade = payload.get("evidence_grade")
        row_payload = payload.get("rows")
    elif isinstance(payload, list):
        source_label = path.stem
        device = None
        evidence_grade = None
        row_payload = payload
    else:
        raise ValueError(f"{path} must contain a JSON object or JSONL row list")
    if not isinstance(row_payload, list):
        raise ValueError(f"{path} is missing a rows list from pair XRay output")

    observations: list[PairObservation] = []
    for idx, item in enumerate(row_payload):
        if not isinstance(item, dict):
            raise ValueError(f"{path} rows[{idx}] must be a JSON object")
        pair_idx = item.get("pair_idx", item.get("pair_index"))
        if not isinstance(pair_idx, int):
            raise ValueError(f"{path} rows[{idx}] missing integer pair_idx")
        pose, seg, component = _score_terms(item)
        observations.append(
            PairObservation(
                pair_idx=pair_idx,
                source_path=str(path),
                source_label=source_label,
                device=str(device) if device is not None else None,
                evidence_grade=str(evidence_grade) if evidence_grade is not None else None,
                pose_score_contribution=pose,
                seg_score_contribution=seg,
                component_score_no_rate=component,
                frame0_l1=_as_float(item.get("frame0_l1")),
                frame1_l1=_as_float(item.get("frame1_l1")),
                raw=item,
            )
        )
    return observations


def _byte_gap(target_gaps: Any, axis: str) -> float | None:
    if not isinstance(target_gaps, dict):
        return None
    row = target_gaps.get(axis)
    if isinstance(row, dict):
        return _as_float(row.get("byte_gap_if_components_unchanged"))
    return None


def _raw_output_match(payload: dict[str, Any]) -> bool | None:
    raw = payload.get("raw_output_comparison")
    if isinstance(raw, dict) and isinstance(raw.get("aggregate_sha256_match"), bool):
        return raw["aggregate_sha256_match"]
    paired = payload.get("paired_axis_xray")
    if isinstance(paired, dict) and isinstance(paired.get("raw_output_aggregate_sha256_match"), bool):
        return paired["raw_output_aggregate_sha256_match"]
    return None


def _axis_context_from_json(path: Path, payload: dict[str, Any]) -> AxisContext | None:
    if isinstance(payload.get("paired_axis_xray"), dict):
        paired = payload["paired_axis_xray"]
        return AxisContext(
            source_path=str(path),
            source_kind="selector_cuda_transfer_calibration",
            classification=str(paired.get("classification")) if paired.get("classification") is not None else None,
            dominant_component=str(paired.get("dominant_score_delta_component"))
            if paired.get("dominant_score_delta_component") is not None
            else None,
            score_delta_byte_equivalent=_as_float(paired.get("score_delta_byte_equivalent")),
            contest_cpu_byte_gap=_byte_gap(paired.get("target_gaps"), "contest_cpu"),
            contest_cuda_byte_gap=_byte_gap(paired.get("target_gaps"), "contest_cuda"),
            raw_output_aggregate_match=paired.get("raw_output_aggregate_sha256_match")
            if isinstance(paired.get("raw_output_aggregate_sha256_match"), bool)
            else None,
            score_delta_cuda_minus_cpu=_as_float(paired.get("cuda_minus_cpu_score_delta")),
            seg_delta=None,
            pose_delta=None,
            rate_delta=None,
        )

    components = payload.get("components")
    if isinstance(components, dict) and isinstance(components.get("delta_cuda_minus_cpu"), dict):
        delta = components["delta_cuda_minus_cpu"]
        return AxisContext(
            source_path=str(path),
            source_kind="paired_cpu_cuda_axis_delta",
            classification=str(payload.get("classification")) if payload.get("classification") is not None else None,
            dominant_component=str(components.get("dominant_score_delta_component"))
            if components.get("dominant_score_delta_component") is not None
            else None,
            score_delta_byte_equivalent=_as_float(components.get("score_delta_byte_equivalent")),
            contest_cpu_byte_gap=_byte_gap(payload.get("target_gaps"), "contest_cpu"),
            contest_cuda_byte_gap=_byte_gap(payload.get("target_gaps"), "contest_cuda"),
            raw_output_aggregate_match=_raw_output_match(payload),
            score_delta_cuda_minus_cpu=_as_float(delta.get("score_delta_cuda_minus_cpu")),
            seg_delta=_as_float(delta.get("seg_score_contribution_delta")),
            pose_delta=_as_float(delta.get("pose_score_contribution_delta")),
            rate_delta=_as_float(delta.get("rate_score_contribution_delta")),
        )

    if isinstance(payload.get("components"), dict) and isinstance(payload.get("deltas"), dict):
        components = payload["components"]
        deltas = payload["deltas"]
        seg = _as_float(components.get("seg_contribution"))
        pose = _as_float(components.get("pose_contribution"))
        dominant = None
        if seg is not None or pose is not None:
            dominant = "pose" if (pose or 0.0) >= (seg or 0.0) else "seg"
        return AxisContext(
            source_path=str(path),
            source_kind="contest_result_review_packet",
            classification=str(payload.get("classification")) if payload.get("classification") is not None else None,
            dominant_component=dominant,
            score_delta_byte_equivalent=_as_float(deltas.get("remaining_bytes_equivalent_at_unchanged_components")),
            contest_cpu_byte_gap=_as_float(deltas.get("remaining_bytes_equivalent_at_unchanged_components"))
            if payload.get("score_axis") == "contest_cpu"
            else None,
            contest_cuda_byte_gap=_as_float(deltas.get("remaining_bytes_equivalent_at_unchanged_components"))
            if payload.get("score_axis") == "contest_cuda"
            else None,
            raw_output_aggregate_match=None,
            score_delta_cuda_minus_cpu=_as_float(deltas.get("cpu_cuda_gap")),
            seg_delta=None,
            pose_delta=None,
            rate_delta=None,
        )
    return None


_NUM = r"([-+]?(?:(?:\d+(?:,\d{3})*)|(?:\d*\.\d+))(?:[eE][-+]?\d+)?)"


def _regex_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return _as_float(match.group(1))


def _regex_word(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().lower().replace("posenet", "pose").replace("segnet", "seg")


def _markdown_target_gap(text: str, axis: str) -> float | None:
    section = re.search(
        r"## Target gaps(?P<body>.*?)(?:\n## |\Z)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    body = section.group("body") if section else text
    axis_pattern = re.escape(axis)
    for line in body.splitlines():
        if re.search(rf"\|\s*{axis_pattern}\s*\|", line, flags=re.IGNORECASE):
            numbers = re.findall(_NUM, line)
            return _as_float(numbers[-1]) if numbers else None
    return None


def _axis_context_from_markdown(path: Path, text: str) -> AxisContext:
    dominant = _regex_word(r"dominant(?: score-delta)? component:\s*`?([A-Za-z0-9_-]+)`?", text)
    if dominant is None and "PoseNet contribution" in text:
        dominant = "pose"
    byte_equiv = _regex_float(r"score-delta byte equivalent:\s*`?" + _NUM, text)
    if byte_equiv is None:
        byte_equiv = _regex_float(r"byte-equivalent CUDA gap[^:\n]*:\s*`?" + _NUM, text)
    cuda_gap = _markdown_target_gap(text, "contest-CUDA")
    cpu_gap = _markdown_target_gap(text, "contest-CPU")
    return AxisContext(
        source_path=str(path),
        source_kind="paired_axis_markdown",
        classification=_regex_word(r"(?:verdict|classification):\s*`?([A-Za-z0-9_-]+)`?", text),
        dominant_component=dominant,
        score_delta_byte_equivalent=byte_equiv,
        contest_cpu_byte_gap=cpu_gap,
        contest_cuda_byte_gap=cuda_gap,
        raw_output_aggregate_match=None if "aggregate match" not in text.lower() else "`true`" in text.lower(),
        score_delta_cuda_minus_cpu=_regex_float(r"total(?: CUDA-minus-CPU)? score delta:\s*`?" + _NUM, text),
        seg_delta=_regex_float(r"seg contribution delta:\s*`?" + _NUM, text),
        pose_delta=_regex_float(r"pose contribution delta:\s*`?" + _NUM, text),
        rate_delta=_regex_float(r"rate contribution delta:\s*`?" + _NUM, text),
    )


def load_axis_context(path: Path) -> AxisContext:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        payload = _load_json(path) if suffix == ".json" else None
        if not isinstance(payload, dict):
            raise ValueError(f"{path} must contain a JSON object for paired-axis context")
        context = _axis_context_from_json(path, payload)
        if context is None:
            raise ValueError(f"{path} is not a recognized paired-axis or result-review artifact")
        return context
    if suffix in {".md", ".markdown", ".txt"}:
        try:
            return _axis_context_from_markdown(path, path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ValueError(f"could not load Markdown {path}: {exc}") from exc
    raise ValueError(f"{path} must be JSON, JSONL, or Markdown")


def _select_primary_context(contexts: list[AxisContext]) -> AxisContext | None:
    if not contexts:
        return None

    def key(ctx: AxisContext) -> tuple[float, float, str]:
        candidates = [
            abs(value)
            for value in (
                ctx.score_delta_byte_equivalent,
                ctx.contest_cuda_byte_gap,
                ctx.contest_cpu_byte_gap,
            )
            if value is not None
        ]
        byte_mass = max(candidates) if candidates else 0.0
        cuda_bias = 1.0 if ctx.contest_cuda_byte_gap is not None or "cuda" in (ctx.classification or "") else 0.0
        return (byte_mass, cuda_bias, ctx.source_path)

    return sorted(contexts, key=key, reverse=True)[0]


def _dominant_pair_component(pose: float, seg: float, primary: AxisContext | None) -> str:
    if primary and primary.dominant_component == "rate":
        return "rate"
    return "pose" if pose >= seg else "seg"


def _axis_weights(primary: AxisContext | None) -> tuple[float, float, str]:
    if primary and primary.dominant_component == "pose":
        return 2.0, 1.0, "axis_dominant_pose"
    if primary and primary.dominant_component == "seg":
        return 1.0, 2.0, "axis_dominant_seg"
    if primary and primary.dominant_component == "rate":
        return 0.5, 0.5, "axis_dominant_rate_deprioritize_components"
    return 1.0, 1.0, "component_tail"


def _component_byte_equivalent(component_score: float, original_uncompressed_size_bytes: int) -> float:
    return component_score * float(original_uncompressed_size_bytes) / 25.0


def _suggest_tags(
    *,
    pair_dominant: str,
    primary: AxisContext | None,
    observations: list[PairObservation],
) -> list[str]:
    devices = {obs.device for obs in observations if obs.device}
    evidence = {obs.evidence_grade for obs in observations if obs.evidence_grade}
    tags: set[str] = {"hardpair_tail"}
    context_text = " ".join(
        text
        for text in (
            primary.classification if primary else None,
            primary.source_kind if primary else None,
        )
        if text
    ).lower()

    if "cuda" in context_text or (primary and primary.contest_cuda_byte_gap is not None):
        if primary and primary.dominant_component == "pose":
            tags.add("cuda_pose_repair")
            tags.add("foveation_pose_repair")
        if primary and primary.dominant_component == "seg":
            tags.add("cuda_seg_repair")
        if pair_dominant == "pose":
            tags.add("cuda_pose_repair")
            tags.add("foveation_pose_repair")
        elif pair_dominant == "seg":
            tags.add("cuda_seg_repair")
        tags.add("cuda_in_loop_required")

    if "cpu" in devices or any("CPU" in grade for grade in evidence):
        if pair_dominant == "seg":
            tags.add("cpu_leaderboard_seg_repair")
        elif pair_dominant == "pose":
            tags.add("cpu_leaderboard_pose_repair")

    if primary and primary.raw_output_aggregate_match is False:
        tags.add("cpu_cuda_raw_output_mismatch")
    if primary and primary.dominant_component in {"pose", "seg"}:
        tags.add("discard_byte_only")
    if primary and primary.dominant_component == "rate":
        tags.add("rate_repack_only")
    if any((obs.frame0_l1 or 0.0) > 0.0 or (obs.frame1_l1 or 0.0) > 0.0 for obs in observations):
        tags.add("film_grain_selector_review")
    return sorted(tags)


def build_hitlist(
    *,
    pair_observations: list[PairObservation],
    axis_contexts: list[AxisContext],
    label: str,
    top_k: int | None,
    original_uncompressed_size_bytes: int = DEFAULT_ORIGINAL_UNCOMPRESSED_SIZE_BYTES,
) -> dict[str, Any]:
    if not pair_observations:
        raise ValueError("at least one pair XRay row is required")
    primary = _select_primary_context(axis_contexts)
    pose_weight, seg_weight, priority_basis = _axis_weights(primary)

    grouped: dict[int, list[PairObservation]] = {}
    for obs in pair_observations:
        grouped.setdefault(obs.pair_idx, []).append(obs)

    rows: list[dict[str, Any]] = []
    for pair_idx in sorted(grouped):
        observations = sorted(grouped[pair_idx], key=lambda obs: (obs.source_path, obs.source_label))
        pose = max(obs.pose_score_contribution for obs in observations)
        seg = max(obs.seg_score_contribution for obs in observations)
        component = max(obs.component_score_no_rate for obs in observations)
        priority = pose_weight * pose + seg_weight * seg
        pair_dominant = _dominant_pair_component(pose, seg, primary)
        axis_gap = None
        if primary:
            axis_gap = (
                primary.contest_cuda_byte_gap
                if primary.contest_cuda_byte_gap is not None
                else primary.score_delta_byte_equivalent
            )
        rows.append(
            {
                "pair_idx": pair_idx,
                "priority": priority,
                "priority_basis": priority_basis,
                "dominant_component": pair_dominant,
                "pose_score_contribution": pose,
                "seg_score_contribution": seg,
                "component_score_no_rate": component,
                "byte_equivalent_component_mass": _component_byte_equivalent(
                    component,
                    original_uncompressed_size_bytes,
                ),
                "axis_byte_equivalent_gap": axis_gap,
                "axis_dominant_component": primary.dominant_component if primary else None,
                "suggested_lane_tags": _suggest_tags(
                    pair_dominant=pair_dominant,
                    primary=primary,
                    observations=observations,
                ),
                "sources": [
                    {
                        "path": obs.source_path,
                        "label": obs.source_label,
                        "device": obs.device,
                        "evidence_grade": obs.evidence_grade,
                    }
                    for obs in observations
                ],
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["priority"]),
            -float(row["component_score_no_rate"]),
            int(row["pair_idx"]),
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["priority_rank"] = rank

    limited_rows = rows if top_k is None or top_k <= 0 else rows[:top_k]
    input_hash = hashlib.sha256(
        json.dumps(
            {
                "pairs": [
                    {
                        "pair_idx": obs.pair_idx,
                        "source_path": obs.source_path,
                        "raw": obs.raw,
                    }
                    for obs in sorted(
                        pair_observations,
                        key=lambda obs: (obs.source_path, obs.pair_idx, obs.source_label),
                    )
                ],
                "axis": [
                    {
                        "source_path": ctx.source_path,
                        "source_kind": ctx.source_kind,
                        "classification": ctx.classification,
                        "dominant_component": ctx.dominant_component,
                        "score_delta_byte_equivalent": ctx.score_delta_byte_equivalent,
                        "contest_cpu_byte_gap": ctx.contest_cpu_byte_gap,
                        "contest_cuda_byte_gap": ctx.contest_cuda_byte_gap,
                        "raw_output_aggregate_match": ctx.raw_output_aggregate_match,
                        "score_delta_cuda_minus_cpu": ctx.score_delta_cuda_minus_cpu,
                        "seg_delta": ctx.seg_delta,
                        "pose_delta": ctx.pose_delta,
                        "rate_delta": ctx.rate_delta,
                    }
                    for ctx in sorted(axis_contexts, key=lambda ctx: ctx.source_path)
                ],
                "label": label,
                "original_uncompressed_size_bytes": original_uncompressed_size_bytes,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "from_state_hash": input_hash,
        "label": label,
        "n_input_pair_rows": len(pair_observations),
        "n_unique_pairs": len(grouped),
        "top_k": top_k,
        "original_uncompressed_size_bytes": original_uncompressed_size_bytes,
        "authority": {
            **FALSE_AUTHORITY,
            "notes": [
                "diagnostic_only_from_xray_artifacts",
                "not_a_score_claim",
                "not_promotion_or_dispatch_authority",
            ],
        },
        "primary_axis_context": None
        if primary is None
        else {
            "source_path": primary.source_path,
            "source_kind": primary.source_kind,
            "classification": primary.classification,
            "dominant_component": primary.dominant_component,
            "score_delta_byte_equivalent": primary.score_delta_byte_equivalent,
            "contest_cpu_byte_gap": primary.contest_cpu_byte_gap,
            "contest_cuda_byte_gap": primary.contest_cuda_byte_gap,
            "raw_output_aggregate_match": primary.raw_output_aggregate_match,
            "score_delta_cuda_minus_cpu": primary.score_delta_cuda_minus_cpu,
            "seg_delta": primary.seg_delta,
            "pose_delta": primary.pose_delta,
            "rate_delta": primary.rate_delta,
        },
        "axis_contexts": [
            {
                "source_path": ctx.source_path,
                "source_kind": ctx.source_kind,
                "classification": ctx.classification,
                "dominant_component": ctx.dominant_component,
                "score_delta_byte_equivalent": ctx.score_delta_byte_equivalent,
                "contest_cpu_byte_gap": ctx.contest_cpu_byte_gap,
                "contest_cuda_byte_gap": ctx.contest_cuda_byte_gap,
                "raw_output_aggregate_match": ctx.raw_output_aggregate_match,
                "score_delta_cuda_minus_cpu": ctx.score_delta_cuda_minus_cpu,
                "seg_delta": ctx.seg_delta,
                "pose_delta": ctx.pose_delta,
                "rate_delta": ctx.rate_delta,
            }
            for ctx in sorted(axis_contexts, key=lambda ctx: ctx.source_path)
        ],
        "hitlist": limited_rows,
        "all_pair_count_before_top_k": len(rows),
    }


def render_markdown(report: dict[str, Any]) -> str:
    authority = report["authority"]
    lines = [
        "# XRay Hard-Pair Hitlist",
        "",
        f"- schema: `{report['schema']}`",
        f"- label: `{report['label']}`",
        f"- from_state_hash: `{report['from_state_hash']}`",
        f"- n_unique_pairs: `{report['n_unique_pairs']}`",
        f"- score_claim: `{str(authority['score_claim']).lower()}`",
        f"- promotion_eligible: `{str(authority['promotion_eligible']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(authority['ready_for_exact_eval_dispatch']).lower()}`",
        "",
    ]
    primary = report.get("primary_axis_context")
    if primary:
        lines.extend(
            [
                "## Primary Axis Context",
                "",
                f"- source: `{primary['source_path']}`",
                f"- classification: `{primary['classification']}`",
                f"- dominant_component: `{primary['dominant_component']}`",
                f"- score_delta_byte_equivalent: `{primary['score_delta_byte_equivalent']}`",
                f"- contest_cuda_byte_gap: `{primary['contest_cuda_byte_gap']}`",
                f"- raw_output_aggregate_match: `{primary['raw_output_aggregate_match']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Hitlist",
            "",
            "| rank | pair | priority | dominant | axis byte gap | tags |",
            "|---:|---:|---:|---|---:|---|",
        ]
    )
    for row in report["hitlist"]:
        tags = ", ".join(row["suggested_lane_tags"])
        axis_gap = "" if row["axis_byte_equivalent_gap"] is None else f"{row['axis_byte_equivalent_gap']:.1f}"
        lines.append(
            f"| {row['priority_rank']} | {row['pair_idx']} | {row['priority']:.12f} | "
            f"{row['dominant_component']} | {axis_gap} | {tags} |"
        )
    lines.extend(
        [
            "",
            "_Tag_: `[diagnostic: xray hard-pair hitlist]`. This report is not a score claim, "
            "not promotion evidence, and not dispatch authority.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic diagnostic hard-pair hitlist from pair-component "
            "XRay JSON plus paired CPU/CUDA axis-delta JSON or Markdown artifacts."
        )
    )
    parser.add_argument(
        "--pair-xray-json",
        action="append",
        type=Path,
        default=[],
        help="Path to pair_component_xray.json or JSONL rows. Repeatable.",
    )
    parser.add_argument(
        "--paired-axis-artifact",
        "--axis-delta",
        action="append",
        type=Path,
        default=[],
        help="Paired CPU/CUDA delta artifact in JSON or Markdown. Repeatable.",
    )
    parser.add_argument("--label", default="xray_hardpair_hitlist")
    parser.add_argument("--top-k", type=int, default=50, help="Rows to emit; <=0 emits all rows.")
    parser.add_argument(
        "--original-uncompressed-size-bytes",
        type=int,
        default=DEFAULT_ORIGINAL_UNCOMPRESSED_SIZE_BYTES,
    )
    parser.add_argument("--output-dir", type=Path, help="Write hardpair_hitlist.json/md and rebuild_command.txt.")
    return parser


def _rebuild_command(args: argparse.Namespace) -> str:
    parts = [".venv/bin/python tools/xray_hardpair_hitlist.py"]
    for path in args.pair_xray_json:
        parts.append(f"--pair-xray-json {shlex.quote(str(path))}")
    for path in args.paired_axis_artifact:
        parts.append(f"--paired-axis-artifact {shlex.quote(str(path))}")
    parts.append(f"--label {shlex.quote(args.label)}")
    parts.append(f"--top-k {args.top_k}")
    parts.append(f"--original-uncompressed-size-bytes {args.original_uncompressed_size_bytes}")
    if args.output_dir:
        parts.append(f"--output-dir {shlex.quote(str(args.output_dir))}")
    return " \\\n  ".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.pair_xray_json:
        print("ERROR: at least one --pair-xray-json is required", file=sys.stderr)
        return 2
    try:
        pair_rows = [
            obs
            for path in args.pair_xray_json
            for obs in load_pair_xray(path)
        ]
        axis_contexts = [load_axis_context(path) for path in args.paired_axis_artifact]
        report = build_hitlist(
            pair_observations=pair_rows,
            axis_contexts=axis_contexts,
            label=args.label,
            top_k=args.top_k,
            original_uncompressed_size_bytes=args.original_uncompressed_size_bytes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = args.output_dir / "hardpair_hitlist.json"
        md_path = args.output_dir / "hardpair_hitlist.md"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(render_markdown(report), encoding="utf-8")
        (args.output_dir / "rebuild_command.txt").write_text(_rebuild_command(args), encoding="utf-8")
        print(f"[xray-hardpair-hitlist] wrote {json_path}")
        print(f"[xray-hardpair-hitlist] wrote {md_path}")
        return 0

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
