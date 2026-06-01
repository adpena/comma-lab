# SPDX-License-Identifier: MIT
"""Fail-closed adjudication for SNeRV rate-sweep rows.

The SNeRV LF-store/HF-generate carrier is promising only if its receiver-visible
packet is charged honestly. In particular, L-inf per-coefficient step maps are
scorer-derived content; until the receiver can regenerate them deterministically,
they are payload bytes, not a free oracle side channel.

This module turns advisory sweep JSON into planner-safe rows:

* ``beats_frontier_rate_only`` means bytes are below a reference archive byte
  count. It is never a score, rank, kill, or promotion claim.
* legacy rows with no charged step-map evidence are blocked as undercharged.
* low-rate rows that destroy pose are separated from distortion-promising rows
  whose blocker is step-map/rate overhead.

All outputs remain ``[macOS-CPU advisory]`` and false-authority by construction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SCHEMA = "snerv_rate_adjudication.v1"
AXIS_TAG = "[macOS-CPU advisory]"
DEFAULT_PR101_FRONTIER_BYTES = 178_493

FALSE_AUTHORITY_BLOCKERS = (
    "frontier_comparison_is_rate_only_not_score_authority",
    "score_axis_is_macos_cpu_advisory",
    "full_600_pair_receiver_replay_missing",
    "paired_contest_cpu_cuda_auth_eval_missing",
)


class SnervRateAdjudicationError(ValueError):
    """Raised when an advisory row cannot be adjudicated safely."""


@dataclass(frozen=True)
class SnervRateAdjudicatedRow:
    """Planner-safe interpretation of one SNeRV advisory/sweep row."""

    source_index: int
    levels: int | None
    bits_per_coeff: float | None
    archive_bytes_charged: int
    receiver_archive_packet_bytes: int | None
    receiver_archive_header_bytes: int | None
    receiver_archive_sha256: str | None
    receiver_archive_replay_verified: bool
    lf_payload_bytes: int | None
    linf_steps_payload_bytes: int | None
    linf_steps_payload_codec: str | None
    linf_steps_coder_mode: str | None
    linf_steps_coder_bins: int | None
    linf_steps_coder_groups: tuple[dict[str, Any], ...]
    linf_steps_fp32_lzma_baseline_bytes: int | None
    linf_steps_payload_vs_fp32_baseline_ratio: float | None
    step_map_accounting: str
    step_map_overhead_bytes: int | None
    pr101_frontier_bytes: int
    beats_frontier_rate_only: bool
    d_seg_linf: float | None
    d_pose_linf: float | None
    score_linf_advisory: float | None
    score_l2_advisory: float | None
    classification: str
    blockers: tuple[str, ...]
    axis_tag: str = AXIS_TAG
    score_claim: bool = False
    frontier_score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False

    def as_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def iter_snerv_candidate_rows(payload: Any) -> list[dict[str, Any]]:
    """Return all row-like dicts from legacy sweeps or structured smoke reports."""

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = []
        for key in ("four_pair_comparable_rows", "rows", "adjudicated_rows"):
            value = payload.get(key)
            if isinstance(value, list):
                rows.extend(value)
        if _looks_like_row(payload):
            rows.append(payload)
    else:
        raise SnervRateAdjudicationError(
            f"expected JSON list/object, got {type(payload).__name__}"
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SnervRateAdjudicationError(
                f"expected row dict, got {type(row).__name__}"
            )
        if _looks_like_row(row):
            out.append(row)
    if not out:
        raise SnervRateAdjudicationError("no SNeRV advisory rows found")
    return out


def adjudicate_snerv_row(
    row: dict[str, Any],
    *,
    source_index: int,
    pr101_frontier_bytes: int = DEFAULT_PR101_FRONTIER_BYTES,
    pose_preservation_ceiling: float = 0.10,
    seg_preservation_ceiling: float = 0.02,
) -> SnervRateAdjudicatedRow:
    """Convert one advisory row into a false-authority planner-safe row."""

    cfg = row.get("config") if isinstance(row.get("config"), dict) else {}
    levels = _optional_int(row.get("levels", cfg.get("levels")))
    bits_per_coeff = _optional_float(
        row.get("bits", row.get("bits_per_coeff", cfg.get("target_bits_per_coeff")))
    )
    archive_bytes = _required_int(
        row.get("archive_bytes_total", row.get("archive_bytes")),
        "archive_bytes_total/archive_bytes",
    )
    receiver_archive_packet_bytes = _optional_int(row.get("receiver_archive_packet_bytes"))
    receiver_archive_header_bytes = _optional_int(row.get("receiver_archive_header_bytes"))
    receiver_archive_sha256 = _optional_str(row.get("receiver_archive_sha256"))
    receiver_archive_replay_verified = row.get("receiver_archive_replay_verified") is True
    receiver_archive_parser_closed = (
        receiver_archive_packet_bytes is not None
        and receiver_archive_packet_bytes == archive_bytes
        and bool(receiver_archive_sha256)
        and receiver_archive_replay_verified
    )
    lf_payload_bytes = _optional_int(row.get("lf_payload_bytes", row.get("lf_bytes")))
    decoder_bytes = _optional_int(row.get("decoder_bytes")) or 0
    metadata_bytes = _optional_int(row.get("metadata_bytes")) or 0
    linf_steps_payload_bytes = _optional_int(row.get("linf_steps_payload_bytes"))
    linf_steps_payload_codec = _optional_str(row.get("linf_steps_payload_codec"))
    linf_steps_coder_mode = _optional_str(row.get("linf_steps_coder_mode"))
    linf_steps_coder_bins = _optional_int(row.get("linf_steps_coder_bins"))
    linf_steps_coder_groups = _optional_groups(row.get("linf_steps_coder_groups"))
    linf_steps_fp32_lzma_baseline_bytes = _optional_int(
        row.get("linf_steps_fp32_lzma_baseline_bytes")
    )
    step_map_charged_flag = bool(row.get("step_map_charged"))
    if linf_steps_payload_bytes is None and step_map_charged_flag and lf_payload_bytes is not None:
        linf_steps_payload_bytes = max(
            archive_bytes - lf_payload_bytes - decoder_bytes - metadata_bytes,
            0,
        )

    if linf_steps_payload_bytes is None:
        step_map_accounting = "missing_or_legacy_undercharged"
        step_overhead = None
    else:
        step_map_accounting = "charged_receiver_visible_payload"
        step_overhead = linf_steps_payload_bytes
    step_payload_ratio = None
    if (
        linf_steps_payload_bytes is not None
        and linf_steps_fp32_lzma_baseline_bytes is not None
        and linf_steps_fp32_lzma_baseline_bytes > 0
    ):
        step_payload_ratio = (
            float(linf_steps_payload_bytes) / float(linf_steps_fp32_lzma_baseline_bytes)
        )

    d_seg = _optional_float(row.get("d_seg_mean_linf", row.get("d_seg_linf")))
    d_pose = _optional_float(row.get("d_pose_mean_linf", row.get("d_pose_linf")))
    score_linf = _optional_float(row.get("score_linf"))
    score_l2 = _optional_float(row.get("score_l2"))
    beats_rate = archive_bytes < pr101_frontier_bytes

    blockers = list(FALSE_AUTHORITY_BLOCKERS)
    if step_map_accounting == "missing_or_legacy_undercharged":
        blockers.insert(0, "linf_step_map_payload_missing_or_legacy_undercharged")
        classification = "legacy_undercharged_requires_step_map_replay"
    else:
        if receiver_archive_parser_closed:
            blockers.insert(0, "not_packaged_as_contest_archive_zip")
        elif linf_steps_payload_codec:
            blockers.insert(
                0,
                "contest_receiver_archive_parser_not_yet_wired_to_compact_step_map_packet",
            )
        else:
            blockers.insert(0, "receiver_runtime_does_not_yet_parse_linf_step_maps")
        distortion_promising = _within(d_pose, pose_preservation_ceiling) and _within(
            d_seg, seg_preservation_ceiling
        )
        if beats_rate and not distortion_promising:
            classification = "rate_below_frontier_pose_or_seg_destroyed"
        elif beats_rate and distortion_promising:
            classification = "rate_promising_runtime_unclosed"
        elif distortion_promising:
            classification = "distortion_promising_step_map_rate_blocked"
        else:
            classification = "not_frontier_score_comparable"

    return SnervRateAdjudicatedRow(
        source_index=source_index,
        levels=levels,
        bits_per_coeff=bits_per_coeff,
        archive_bytes_charged=archive_bytes,
        receiver_archive_packet_bytes=receiver_archive_packet_bytes,
        receiver_archive_header_bytes=receiver_archive_header_bytes,
        receiver_archive_sha256=receiver_archive_sha256,
        receiver_archive_replay_verified=receiver_archive_replay_verified,
        lf_payload_bytes=lf_payload_bytes,
        linf_steps_payload_bytes=linf_steps_payload_bytes,
        linf_steps_payload_codec=linf_steps_payload_codec,
        linf_steps_coder_mode=linf_steps_coder_mode,
        linf_steps_coder_bins=linf_steps_coder_bins,
        linf_steps_coder_groups=linf_steps_coder_groups,
        linf_steps_fp32_lzma_baseline_bytes=linf_steps_fp32_lzma_baseline_bytes,
        linf_steps_payload_vs_fp32_baseline_ratio=step_payload_ratio,
        step_map_accounting=step_map_accounting,
        step_map_overhead_bytes=step_overhead,
        pr101_frontier_bytes=pr101_frontier_bytes,
        beats_frontier_rate_only=beats_rate,
        d_seg_linf=d_seg,
        d_pose_linf=d_pose,
        score_linf_advisory=score_linf,
        score_l2_advisory=score_l2,
        classification=classification,
        blockers=tuple(blockers),
    )


def build_snerv_rate_adjudication_payload(
    payload: Any,
    *,
    source_path: str | None = None,
    pr101_frontier_bytes: int = DEFAULT_PR101_FRONTIER_BYTES,
    pose_preservation_ceiling: float = 0.10,
    seg_preservation_ceiling: float = 0.02,
) -> dict[str, Any]:
    """Build a machine-readable false-authority adjudication report."""

    rows = [
        adjudicate_snerv_row(
            row,
            source_index=i,
            pr101_frontier_bytes=pr101_frontier_bytes,
            pose_preservation_ceiling=pose_preservation_ceiling,
            seg_preservation_ceiling=seg_preservation_ceiling,
        )
        for i, row in enumerate(iter_snerv_candidate_rows(payload))
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.classification] = counts.get(row.classification, 0) + 1

    exact_blockers = sorted({b for row in rows for b in row.blockers})
    best_distortion = min(
        rows,
        key=lambda r: (
            r.d_pose_linf is None,
            float("inf") if r.d_pose_linf is None else r.d_pose_linf,
            r.archive_bytes_charged,
        ),
    )
    best_rate = min(rows, key=lambda r: r.archive_bytes_charged)
    if any(
        row.step_map_accounting == "missing_or_legacy_undercharged"
        or not row.receiver_archive_replay_verified
        for row in rows
    ):
        actionable_next = "compact_step_map_packet_and_snAR1_receiver_replay_closure"
    elif any(row.classification == "rate_below_frontier_pose_or_seg_destroyed" for row in rows):
        actionable_next = "score_aware_stepmap_waterfill_and_decoder_fit_before_packaging"
    else:
        actionable_next = "contest_archive_zip_packaging_full600_and_paired_cpu_cuda_replay"
    return {
        "schema": SCHEMA,
        "source_path": source_path,
        "axis_tag": AXIS_TAG,
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pr101_frontier_bytes": pr101_frontier_bytes,
        "pose_preservation_ceiling": pose_preservation_ceiling,
        "seg_preservation_ceiling": seg_preservation_ceiling,
        "rows": [row.as_jsonable() for row in rows],
        "summary": {
            "row_count": len(rows),
            "classification_counts": counts,
            "best_rate_source_index": best_rate.source_index,
            "best_rate_classification": best_rate.classification,
            "best_distortion_source_index": best_distortion.source_index,
            "best_distortion_classification": best_distortion.classification,
            "any_frontier_score_claim": False,
            "actionable_next_code_move": actionable_next,
        },
        "exact_readiness_refusal": {
            "ready": False,
            "reason": (
                "SNeRV rows are advisory-only and need contest archive.zip packaging, "
                "full-600 replay where absent, and paired contest CPU/CUDA eval before "
                "any score, rank, or promotion use."
            ),
            "blockers": exact_blockers,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        },
    }


def _looks_like_row(row: dict[str, Any]) -> bool:
    return "archive_bytes_total" in row or "archive_bytes" in row


def _required_int(value: Any, name: str) -> int:
    out = _optional_int(value)
    if out is None:
        raise SnervRateAdjudicationError(f"missing required integer field: {name}")
    return out


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value


def _optional_groups(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(dict(group) for group in value if isinstance(group, dict))


def _within(value: float | None, ceiling: float) -> bool:
    return value is not None and value <= ceiling
