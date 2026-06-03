# SPDX-License-Identifier: MIT
"""Lossless full-packet SNeRV LF payload recoding.

The LF codec sweep proves that alternative integer-stream grammars can encode
the same LF planes.  This module closes the next custody gap: swap only the
``lf_payload`` section inside a real SNAR1 packet, then prove the receiver sees
the same decoded LF state while every other section stays byte-identical.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import numpy as np

from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    DecodedSnervArchive,
    SnervArchiveError,
    encode_lf_quant_payload,
    inspect_lf_quant_payload_header,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SnervFrameCode,
    decode_frame,
    dequantize_lf,
)

SCHEMA = "snerv_lf_payload_archive_recode.v1"
AXIS_TAG = "[receiver-proof:false-authority]"
DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
UNCHANGED_SECTIONS = ("metadata_payload", "decoder_payload", "step_map_packet")


class SnervLfPayloadArchiveRecodeError(ValueError):
    """Raised when a SNeRV archive LF recode is invalid."""


def build_snerv_lf_payload_archive_recode(
    source_packet: bytes,
    *,
    mode: str,
    source_packet_path: str | None = None,
    frame_proof_max_output_bytes: int = DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES,
    force_frame_proof: bool = False,
) -> tuple[dict[str, Any], bytes]:
    """Return a losslessly recoded SNAR1 packet plus proof report.

    The candidate packet is not a score or promotion surface.  It is a
    receiver-custody artifact that proves the selected LF codec can live inside
    the full packet grammar without mutating the decoded signal.
    """

    source_blob = bytes(source_packet)
    if not source_blob:
        raise SnervLfPayloadArchiveRecodeError("source_packet must be non-empty")
    if not str(mode).strip():
        raise SnervLfPayloadArchiveRecodeError("mode must be non-empty")

    source = unpack_snerv_archive(source_blob)
    source_lf_planes = source.decode_lf_quant_planes()
    candidate_lf_payload = encode_lf_quant_payload(source_lf_planes, codec=mode)
    candidate_packet = pack_snerv_archive(
        metadata_payload=source.sections["metadata_payload"],
        lf_payload=candidate_lf_payload,
        decoder_payload=source.sections["decoder_payload"],
        step_map_packet=source.sections["step_map_packet"],
        metadata=source.metadata,
    )
    candidate = unpack_snerv_archive(candidate_packet.packet)
    candidate_lf_planes = candidate.decode_lf_quant_planes()

    lf_exact, lf_hash_rows = _lf_plane_equality(source_lf_planes, candidate_lf_planes)
    unchanged = {
        section: (
            source.sections[section] == candidate.sections[section]
            and _sha256(source.sections[section])
            == _sha256(candidate.sections[section])
        )
        for section in UNCHANGED_SECTIONS
    }
    frame_proof = _streaming_frame_equality_proof(
        source,
        candidate,
        max_output_bytes=int(frame_proof_max_output_bytes),
        force=bool(force_frame_proof),
    )
    frame_status = str(frame_proof["status"])
    receiver_contract = bool(
        lf_exact
        and all(unchanged.values())
        and frame_status not in {"failed", "error"}
    )
    source_lf_bytes = len(source.sections["lf_payload"])
    candidate_lf_bytes = len(candidate.sections["lf_payload"])
    packet_byte_delta = int(candidate_packet.total_bytes - len(source_blob))
    lf_byte_delta = int(candidate_lf_bytes - source_lf_bytes)
    blockers = _blockers(
        receiver_contract=receiver_contract,
        frame_status=frame_status,
        unchanged=unchanged,
    )
    report = {
        "schema": SCHEMA,
        "axis_tag": AXIS_TAG,
        "family": "snerv",
        "operation": "lossless_lf_payload_recode_inside_snar1_packet",
        "mode": str(mode),
        "source_packet": {
            "path": source_packet_path,
            "bytes": len(source_blob),
            "sha256": _sha256(source_blob),
            "decoded_packet_sha256": source.packet_sha256,
        },
        "candidate_packet": {
            "bytes": candidate_packet.total_bytes,
            "sha256": _sha256(candidate_packet.packet),
            "decoded_packet_sha256": candidate.packet_sha256,
            "header_bytes": candidate_packet.header_bytes,
        },
        "packet_byte_delta": packet_byte_delta,
        "packet_rate_score_delta": float(packet_byte_delta * CONTEST_BYTE_PRICE_SCORE),
        "lf_payload": {
            "source_bytes": source_lf_bytes,
            "candidate_bytes": candidate_lf_bytes,
            "byte_delta": lf_byte_delta,
            "rate_score_delta": float(lf_byte_delta * CONTEST_BYTE_PRICE_SCORE),
            "source_sha256": _sha256(source.sections["lf_payload"]),
            "candidate_sha256": _sha256(candidate.sections["lf_payload"]),
            "source_header": inspect_lf_quant_payload_header(
                source.sections["lf_payload"]
            ),
            "candidate_header": inspect_lf_quant_payload_header(
                candidate.sections["lf_payload"]
            ),
        },
        "section_bytes": {
            "source": {k: len(v) for k, v in source.sections.items()},
            "candidate": dict(candidate_packet.section_bytes),
        },
        "section_sha256": {
            "source": {k: _sha256(v) for k, v in source.sections.items()},
            "candidate": dict(candidate_packet.section_sha256),
        },
        "unchanged_sections_exact": unchanged,
        "lf_plane_count": len(source_lf_planes),
        "lf_planes_exact_equal": lf_exact,
        "lf_plane_hash_rows": lf_hash_rows,
        "receiver_frame_equality_proof": frame_proof,
        "receiver_contract_satisfied": receiver_contract,
        "runtime_consumption_proof_ready": receiver_contract,
        "score_claim": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    return report, candidate_packet.packet


def render_snerv_lf_payload_archive_recode_markdown(
    report: Mapping[str, Any],
) -> str:
    """Render a compact operator-readable recode report."""

    source = report.get("source_packet", {}) if isinstance(report, Mapping) else {}
    candidate = report.get("candidate_packet", {}) if isinstance(report, Mapping) else {}
    lf = report.get("lf_payload", {}) if isinstance(report, Mapping) else {}
    frame = (
        report.get("receiver_frame_equality_proof", {})
        if isinstance(report, Mapping)
        else {}
    )
    blockers = [str(v) for v in report.get("blockers", [])] if isinstance(report, Mapping) else []
    lines = [
        "# SNeRV LF Payload Archive Recode",
        "",
        f"- schema: `{report.get('schema')}`",
        f"- axis: `{report.get('axis_tag')}`",
        f"- mode: `{report.get('mode')}`",
        f"- source packet: `{source.get('bytes')}` bytes `{source.get('sha256')}`",
        f"- candidate packet: `{candidate.get('bytes')}` bytes `{candidate.get('sha256')}`",
        f"- packet byte delta: `{report.get('packet_byte_delta')}`",
        f"- LF byte delta: `{lf.get('byte_delta')}`",
        f"- LF planes exact: `{report.get('lf_planes_exact_equal')}`",
        f"- unchanged sections exact: `{report.get('unchanged_sections_exact')}`",
        f"- receiver frame proof: `{frame.get('status')}`",
        f"- receiver contract satisfied: `{report.get('receiver_contract_satisfied')}`",
        "",
        "## Blockers",
    ]
    lines.extend(f"- `{blocker}`" for blocker in blockers)
    return "\n".join(lines) + "\n"


def _lf_plane_equality(
    source: list[np.ndarray],
    candidate: list[np.ndarray],
) -> tuple[bool, list[dict[str, Any]]]:
    if len(source) != len(candidate):
        return False, [
            {
                "plane_index": -1,
                "source_count": len(source),
                "candidate_count": len(candidate),
                "exact_equal": False,
            }
        ]
    rows = []
    all_equal = True
    for idx, (a, b) in enumerate(zip(source, candidate, strict=True)):
        a_arr = np.asarray(a, dtype="<i8")
        b_arr = np.asarray(b, dtype="<i8")
        exact = bool(a_arr.shape == b_arr.shape and np.array_equal(a_arr, b_arr))
        all_equal = all_equal and exact
        rows.append(
            {
                "plane_index": idx,
                "shape": [int(v) for v in a_arr.shape],
                "source_sha256": _sha256(a_arr.tobytes()),
                "candidate_sha256": _sha256(b_arr.tobytes()),
                "exact_equal": exact,
            }
        )
    return all_equal, rows


def _streaming_frame_equality_proof(
    source: DecodedSnervArchive,
    candidate: DecodedSnervArchive,
    *,
    max_output_bytes: int,
    force: bool,
) -> dict[str, Any]:
    try:
        estimate = _estimated_receiver_output_bytes(source)
        if not force and estimate is not None and estimate > int(max_output_bytes):
            return {
                "status": "skipped_by_output_byte_guard",
                "estimated_output_bytes": estimate,
                "max_output_bytes": int(max_output_bytes),
                "exactness_basis": (
                    "lf_planes_exact_and_metadata_decoder_step_map_sections_unchanged"
                ),
            }
        source_hash, candidate_hash, compared, max_abs = _stream_frame_hash_compare(
            source,
            candidate,
        )
        exact = source_hash == candidate_hash and max_abs == 0.0
        return {
            "status": "proven_exact" if exact else "failed",
            "estimated_output_bytes": estimate,
            "max_output_bytes": int(max_output_bytes),
            "compared_plane_count": compared,
            "source_frame_sha256": source_hash,
            "candidate_frame_sha256": candidate_hash,
            "max_abs_diff": max_abs,
            "exact_equal": exact,
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "estimated_output_bytes": _estimated_receiver_output_bytes(source),
            "max_output_bytes": int(max_output_bytes),
        }


def _stream_frame_hash_compare(
    source: DecodedSnervArchive,
    candidate: DecodedSnervArchive,
) -> tuple[str, str, int, float]:
    source_parts = _receiver_decode_parts(source)
    candidate_parts = _receiver_decode_parts(candidate)
    if source_parts["orig_hw"] != candidate_parts["orig_hw"]:
        raise SnervArchiveError("candidate orig_hw differs from source")
    if source_parts["levels"] != candidate_parts["levels"]:
        raise SnervArchiveError("candidate levels differ from source")
    if source_parts["wavelet"] != candidate_parts["wavelet"]:
        raise SnervArchiveError("candidate wavelet differs from source")

    source_hash = hashlib.sha256()
    candidate_hash = hashlib.sha256()
    compared = 0
    max_abs = 0.0
    for idx, (source_code, candidate_code) in enumerate(
        zip(source_parts["codes"], candidate_parts["codes"], strict=True)
    ):
        source_lf_sequence = None
        candidate_lf_sequence = None
        sequence_index = None
        if source_parts["temporal_group_count"] > 1:
            group = idx % source_parts["temporal_group_count"]
            source_lf_sequence = source_parts["decoded_lfs"][group :: source_parts["temporal_group_count"]]
            candidate_lf_sequence = candidate_parts["decoded_lfs"][group :: candidate_parts["temporal_group_count"]]
            sequence_index = idx // source_parts["temporal_group_count"]
        source_frame = np.clip(
            decode_frame(
                source_code,
                source_parts["decoder"],
                lf_sequence=source_lf_sequence,
                sequence_index=sequence_index,
            ),
            0.0,
            255.0,
        ).astype("<f4", copy=False)
        candidate_frame = np.clip(
            decode_frame(
                candidate_code,
                candidate_parts["decoder"],
                lf_sequence=candidate_lf_sequence,
                sequence_index=sequence_index,
            ),
            0.0,
            255.0,
        ).astype("<f4", copy=False)
        if source_frame.shape != candidate_frame.shape:
            raise SnervArchiveError(
                f"candidate frame {idx} shape {candidate_frame.shape} != source {source_frame.shape}"
            )
        diff = float(np.max(np.abs(source_frame - candidate_frame)))
        max_abs = max(max_abs, diff)
        source_hash.update(source_frame.tobytes())
        candidate_hash.update(candidate_frame.tobytes())
        compared += 1
    return source_hash.hexdigest(), candidate_hash.hexdigest(), compared, max_abs


def _receiver_decode_parts(decoded: DecodedSnervArchive) -> dict[str, Any]:
    metadata = decoded.metadata
    levels = _metadata_int(metadata, "levels", minimum=1)
    wavelet = _metadata_str(metadata, "wavelet")
    orig_hw = _metadata_hw(metadata)
    lf_planes = decoded.decode_lf_quant_planes()
    zeros = decoded.decode_lf_zero_points()
    step_maps = decoded.decode_step_maps()
    decoder = decoded.decode_decoder()
    if not (len(lf_planes) == len(zeros) == len(step_maps)):
        raise SnervArchiveError("receiver replay state count mismatch")
    codes: list[SnervFrameCode] = []
    decoded_lfs: list[np.ndarray] = []
    for idx, (q, zero, steps) in enumerate(zip(lf_planes, zeros, step_maps, strict=True)):
        if q.shape != steps.shape:
            raise SnervArchiveError(
                f"receiver replay plane {idx} LF shape {q.shape} != step shape {steps.shape}"
            )
        code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=tuple(int(v) for v in q.shape),
            levels=levels,
            wavelet=wavelet,
            orig_hw=orig_hw,
            per_element_steps=steps,
        )
        codes.append(code)
        decoded_lfs.append(dequantize_lf(q, 1.0, float(zero), per_element_steps=steps))

    temporal_group_count = 1
    if int(decoder.model_size.temporal_context) > 0:
        temporal_group_count = _metadata_int(metadata, "channels", default=1, minimum=1)
    return {
        "metadata": metadata,
        "levels": levels,
        "wavelet": wavelet,
        "orig_hw": orig_hw,
        "decoder": decoder,
        "codes": codes,
        "decoded_lfs": decoded_lfs,
        "temporal_group_count": temporal_group_count,
    }


def _estimated_receiver_output_bytes(decoded: DecodedSnervArchive) -> int | None:
    try:
        h, w = _metadata_hw(decoded.metadata)
        n_pairs = _metadata_int(decoded.metadata, "n_pairs", default=0, minimum=0)
        frames_per_pair = _metadata_int(
            decoded.metadata,
            "frames_per_pair",
            default=0,
            minimum=0,
        )
        channels = _metadata_int(decoded.metadata, "channels", default=0, minimum=0)
        if n_pairs and frames_per_pair and channels:
            plane_count = n_pairs * frames_per_pair * channels
        else:
            plane_count = len(decoded.decode_lf_quant_planes())
        return int(plane_count * h * w * np.dtype("<f4").itemsize)
    except Exception:
        return None


def _blockers(
    *,
    receiver_contract: bool,
    frame_status: str,
    unchanged: Mapping[str, bool],
) -> list[str]:
    blockers = []
    if not receiver_contract:
        blockers.append("snerv_lf_payload_archive_recode_receiver_contract_failed")
    if not all(unchanged.values()):
        blockers.append("snerv_lf_payload_archive_recode_mutated_non_lf_section")
    if frame_status == "skipped_by_output_byte_guard":
        blockers.append("receiver_frame_streaming_proof_skipped_by_output_byte_guard")
    if frame_status in {"failed", "error"}:
        blockers.append("receiver_frame_equality_proof_failed")
    blockers.extend(
        [
            "not_packaged_as_contest_archive_zip",
            "full_video_scorer_replay_missing",
            "paired_contest_cpu_cuda_auth_eval_missing",
        ]
    )
    return _ordered_unique(blockers)


def _metadata_int(
    metadata: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
) -> int:
    if key not in metadata:
        if default is None:
            raise SnervArchiveError(f"SNeRV archive metadata missing {key!r}")
        value = int(default)
    else:
        value = int(metadata[key])
    if minimum is not None and value < minimum:
        raise SnervArchiveError(f"SNeRV archive metadata {key!r} below {minimum}")
    return value


def _metadata_str(metadata: Mapping[str, Any], key: str) -> str:
    value = str(metadata.get(key) or "")
    if not value:
        raise SnervArchiveError(f"SNeRV archive metadata missing {key!r}")
    return value


def _metadata_hw(metadata: Mapping[str, Any]) -> tuple[int, int]:
    value = metadata.get("orig_hw") or metadata.get("carrier_hw")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise SnervArchiveError("SNeRV archive metadata missing orig_hw")
    return (int(value[0]), int(value[1]))


def _ordered_unique(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = str(item)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(bytes(blob)).hexdigest()


__all__ = [
    "AXIS_TAG",
    "DEFAULT_FRAME_PROOF_MAX_OUTPUT_BYTES",
    "SCHEMA",
    "SnervLfPayloadArchiveRecodeError",
    "build_snerv_lf_payload_archive_recode",
    "render_snerv_lf_payload_archive_recode_markdown",
]
