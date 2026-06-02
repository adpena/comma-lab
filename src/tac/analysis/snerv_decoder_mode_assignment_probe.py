# SPDX-License-Identifier: MIT
"""Fail-closed SNeRV decoder mixed-mode assignment probe.

The mixed decoder payload grammar can protect some HF kernels with fp16/int8,
push many to int4/int2, and drop others to zero. This module compares explicit
receiver-decoded mode plans through the existing SNeRV advisory path without
turning the local advisory score into promotion authority.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tac.substrates.snerv_inverse_steg_carrier.advisory import run_snerv_advisory

SCHEMA = "snerv_decoder_mode_assignment_probe.v1"
AXIS_TAG = "[macOS-CPU advisory]"
MIXED_DECODER_CODEC = "mixed_magnitude_symmetric"
DECODER_SUBBANDS_PER_LEVEL = 3
MODE_ALIASES = {
    "0": "zero",
    "none": "zero",
    "drop": "zero",
    "z": "zero",
    "zero": "zero",
    "2": "int2",
    "i2": "int2",
    "int2": "int2",
    "4": "int4",
    "i4": "int4",
    "int4": "int4",
    "8": "int8",
    "i8": "int8",
    "int8": "int8",
    "16": "fp16",
    "f16": "fp16",
    "float16": "fp16",
    "fp16": "fp16",
}
HEURISTIC_MODE_PLAN_NAMES = {
    "",
    "auto",
    "heuristic",
    "magnitude",
    "magnitude_heuristic",
    "mixed_magnitude_heuristic",
}
FAIL_CLOSED_BLOCKERS = (
    "macos_cpu_advisory_only",
    "paired_contest_cpu_cuda_auth_eval_missing",
    "not_packaged_as_contest_archive_zip",
)


class SnervDecoderModeProbeError(ValueError):
    """Raised when a decoder mode plan cannot be probed safely."""


def parse_mode_plan(
    raw: str,
    *,
    levels: int,
) -> tuple[str, tuple[str, ...] | None]:
    """Parse one mode plan into a stable label and explicit kernel modes.

    ``None`` modes means the archive grammar's magnitude heuristic is used.
    Explicit mode plans must provide one mode per ``level x subband`` kernel.
    """

    expected = _expected_mode_count(levels)
    text = str(raw or "").strip()
    normalized_text = text.lower().replace("-", "_").replace(" ", "_")
    if normalized_text in HEURISTIC_MODE_PLAN_NAMES:
        return "magnitude_heuristic", None

    modes = tuple(
        _normalize_mode_token(chunk)
        for chunk in text.split(",")
        if chunk.strip()
    )
    if len(modes) != expected:
        raise SnervDecoderModeProbeError(
            f"decoder mode plan {text!r} has {len(modes)} modes; "
            f"expected {expected} for levels={levels}"
        )
    histogram = _mode_histogram(modes)
    label = "explicit_" + "_".join(
        f"{mode}{count}" for mode, count in histogram.items() if count
    )
    return label, modes


def run_snerv_decoder_mode_assignment_probe(
    *,
    mode_plans: Sequence[str],
    n_pairs: int = 1,
    levels: int = 1,
    bits_per_coeff: float = 2.0,
    wavelet: str = "db2",
    pair_stride: int = 1,
    start_pair: int = 0,
    pr101_frontier_bytes: int = 178_493,
    upstream_dir: str = "upstream",
    video_path: str = "upstream/videos/0.mkv",
    step_map_coder_bins: int = 4,
    step_map_coder_mode: str = "uniform",
    step_map_adaptive_bin_choices: Sequence[int] = (128, 16, 4),
    step_map_constant_importance_quantile: float | None = None,
    step_map_waterfill_bits_per_coeff: float = 4.0,
    hf_decoder_fit_mode: str = "least_squares",
    hf_decoder_saliency_gain: float = 1.0,
    hf_decoder_saliency_component: str = "combined",
    receiver_packet_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a local advisory-only race over decoder mode assignments."""

    plans = tuple(mode_plans)
    if not plans:
        plans = ("magnitude_heuristic",)
    candidates = []
    parsed_plans = [
        parse_mode_plan(plan, levels=levels)
        for plan in plans
    ]
    packet_dir = (
        Path(receiver_packet_dir).expanduser().resolve(strict=False)
        if receiver_packet_dir is not None
        else None
    )
    if packet_dir is not None:
        packet_dir.mkdir(parents=True, exist_ok=True)
    for label, modes in parsed_plans:
        result = run_snerv_advisory(
            n_pairs=n_pairs,
            levels=levels,
            wavelet=wavelet,
            target_bits_per_coeff=bits_per_coeff,
            pair_stride=pair_stride,
            start_pair=start_pair,
            pr101_frontier_bytes=pr101_frontier_bytes,
            video_path=video_path,
            upstream_dir=upstream_dir,
            step_map_coder_bins=step_map_coder_bins,
            step_map_coder_mode=step_map_coder_mode,
            step_map_adaptive_bin_choices=tuple(
                int(v) for v in step_map_adaptive_bin_choices
            ),
            step_map_constant_importance_quantile=(
                step_map_constant_importance_quantile
            ),
            step_map_waterfill_bits_per_coeff=step_map_waterfill_bits_per_coeff,
            hf_decoder_fit_mode=hf_decoder_fit_mode,
            hf_decoder_saliency_gain=hf_decoder_saliency_gain,
            hf_decoder_saliency_component=hf_decoder_saliency_component,
            decoder_payload_codec=MIXED_DECODER_CODEC,
            decoder_payload_mixed_modes=modes,
        )
        candidates.append(
            candidate_summary_from_result(
                label,
                modes,
                result,
                receiver_packet_dir=packet_dir,
                candidate_index=len(candidates),
            )
        )

    replay_verified = [
        candidate
        for candidate in candidates
        if candidate.get("receiver_archive_replay_verified") is True
    ]
    best = min(
        replay_verified,
        key=lambda candidate: float(candidate.get("score_linf", float("inf"))),
        default=None,
    )
    blockers = list(FAIL_CLOSED_BLOCKERS)
    if int(n_pairs) != 600:
        blockers.insert(1, "full_600_pair_receiver_replay_missing")
    if not replay_verified:
        blockers.insert(0, "no_receiver_replay_verified_candidate")
    for candidate in candidates:
        for blocker in candidate.get("blockers", ()):
            if blocker not in blockers:
                blockers.append(str(blocker))

    return {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "n_pairs": int(n_pairs),
        "levels": int(levels),
        "wavelet": wavelet,
        "bits_per_coeff": float(bits_per_coeff),
        "mode_plan_count": len(candidates),
        "decoder_payload_codec": MIXED_DECODER_CODEC,
        "score_claim": False,
        "frontier_score_claim": False,
        "rank_or_kill_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_or_full_video_launched": False,
        "best_plan_label": best.get("label") if best else None,
        "best_plan_score_linf_advisory": (
            float(best["score_linf"]) if best and best.get("score_linf") is not None else None
        ),
        "candidates": candidates,
        "blockers": blockers,
        "notes": (
            "Advisory-only local decoder mode assignment probe. Every candidate "
            "uses receiver-decoded mixed decoder bytes; no exact eval, full-600 "
            "promotion, or rank/kill authority is granted by this artifact."
        ),
    }


def candidate_summary_from_result(
    label: str,
    modes: tuple[str, ...] | None,
    result: Any,
    *,
    receiver_packet_dir: Path | None = None,
    candidate_index: int = 0,
) -> dict[str, Any]:
    """Extract the planner-relevant fields from one advisory result."""

    payload = result.as_jsonable()
    header = payload.get("decoder_payload_header")
    if not isinstance(header, dict):
        header = {}
    packet = payload.get("receiver_archive_packet")
    packet_sha256 = packet.get("sha256") if isinstance(packet, dict) else None
    blockers = tuple(str(v) for v in payload.get("archive_byte_closure_blockers", ()))
    packet_export, packet_blockers = _write_receiver_packet_if_requested(
        result,
        label=label,
        packet_sha256=str(packet_sha256 or ""),
        receiver_packet_dir=receiver_packet_dir,
        candidate_index=candidate_index,
    )
    return {
        "label": str(label),
        "modes": list(modes) if modes is not None else None,
        "mode_assignment_source": header.get("mode_assignment_source"),
        "mode_histogram": dict(header.get("mode_histogram", {})),
        "decoder_payload_bytes": _optional_int(header.get("payload_bytes")),
        "decoder_bytes": _optional_int(payload.get("decoder_bytes")),
        "receiver_archive_packet_bytes": _optional_int(
            payload.get("receiver_archive_packet_bytes")
        ),
        "receiver_archive_packet_sha256": packet_sha256,
        "receiver_archive_packet_export": packet_export,
        "receiver_archive_packet_path": packet_export.get("path"),
        "receiver_archive_packet_file_sha256": packet_export.get("sha256"),
        "receiver_archive_packet_is_contest_archive_zip": False,
        "contest_archive_zip_path": None,
        "candidate_archive_path": None,
        "archive_path": None,
        "archive_bytes_total": _optional_int(payload.get("archive_bytes_total")),
        "receiver_archive_replay_verified": (
            payload.get("receiver_archive_replay_verified") is True
        ),
        "receiver_archive_replay_error": payload.get("receiver_archive_replay_error"),
        "d_seg_mean_linf": _optional_float(payload.get("d_seg_mean_linf")),
        "d_pose_mean_linf": _optional_float(payload.get("d_pose_mean_linf")),
        "score_linf": _optional_float(payload.get("score_linf")),
        "d_seg_mean_l2": _optional_float(payload.get("d_seg_mean_l2")),
        "d_pose_mean_l2": _optional_float(payload.get("d_pose_mean_l2")),
        "score_l2": _optional_float(payload.get("score_l2")),
        "rate_term": _optional_float(payload.get("rate_term")),
        "blockers": [*blockers, *packet_blockers],
        "axis_tag": AXIS_TAG,
        "score_claim": False,
        "frontier_score_claim": False,
        "rank_or_kill_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _write_receiver_packet_if_requested(
    result: Any,
    *,
    label: str,
    packet_sha256: str,
    receiver_packet_dir: Path | None,
    candidate_index: int,
) -> tuple[dict[str, Any], list[str]]:
    if receiver_packet_dir is None:
        return {}, []
    packet_bytes = getattr(result, "receiver_archive_packet", None)
    if not isinstance(packet_bytes, bytes):
        return {}, ["receiver_packet_export_requested_but_raw_packet_missing"]
    safe_label = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in str(label)
    ).strip("_") or "candidate"
    out = receiver_packet_dir / f"{candidate_index:04d}_{safe_label}.snar"
    out.write_bytes(packet_bytes)
    file_sha256 = _sha256_file(out)
    export = {
        "schema": "snerv_receiver_packet_export.v1",
        "kind": "snerv_receiver_packet_snar1_not_contest_archive_zip",
        "path": out.as_posix(),
        "bytes": int(out.stat().st_size),
        "sha256": file_sha256,
        "expected_sha256": packet_sha256 or None,
        "contest_archive_zip": False,
    }
    if packet_sha256 and file_sha256 != packet_sha256:
        return export, ["receiver_packet_export_sha256_mismatch"]
    return export, []


def _expected_mode_count(levels: int) -> int:
    value = int(levels)
    if value < 1:
        raise SnervDecoderModeProbeError("levels must be >= 1")
    return value * DECODER_SUBBANDS_PER_LEVEL


def _normalize_mode_token(raw: str) -> str:
    token = str(raw or "").strip().lower().replace("-", "")
    if token not in MODE_ALIASES:
        raise SnervDecoderModeProbeError(
            f"unknown decoder mode {raw!r}; expected one of zero,int2,int4,int8,fp16"
        )
    return MODE_ALIASES[token]


def _mode_histogram(modes: Sequence[str]) -> dict[str, int]:
    histogram = dict.fromkeys(("zero", "int2", "int4", "int8", "fp16"), 0)
    for mode in modes:
        histogram[mode] += 1
    return histogram


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
