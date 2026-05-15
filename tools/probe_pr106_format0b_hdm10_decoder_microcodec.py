#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Planning-only HDM10 decoder microcodec probe for PR106 format 0x0B.

This tool reads the active PR106 PacketIR format-0x0B archive, extracts the
charged HDM9 decoder tail, and tests deterministic byte recodes that can prove
lossless raw-decoder equivalence. It emits no archive and makes no score claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import math
import sys
from pathlib import Path
from typing import Any

import brotli  # type: ignore[import-not-found]

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.hnerv_decoder_recode import (  # noqa: E402
    decode_hdm8_q_brotli_recipe_elided_fixture,
)
from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_HDM9_HLM2_DECODER_MAGIC,
    PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES,
    PR106_HDM9_HLM3_LATENT_MAGIC,
    PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES,
    PR106_HDM9_Q_BROTLI_CHUNK_BYTES,
    PR106_HDM9_SCALE_COUNT,
    PR106_HDM9_SCALE_HIGH_MASK_BYTES,
    PR106_HDM9_SCALE_LOW3_BYTES,
    PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES,
    PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED,
    decode_hdm9_decoder_to_hdm8_payload,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
)
from tac.repo_io import json_text, repo_relative, sha256_bytes  # noqa: E402
from tac.tool_manifest import attach_tool_run_manifest  # noqa: E402

CONTEST_ORIGINAL_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_ORIGINAL_BYTES
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/"
    / "candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip"
)
DEFAULT_JSON_OUT = (
    REPO_ROOT
    / ".omx/research/pr106_format0b_hdm10_decoder_microcodec_probe_20260515_codex.json"
)
DEFAULT_MD_OUT = (
    REPO_ROOT
    / ".omx/research/pr106_format0b_hdm10_decoder_microcodec_probe_20260515_codex.md"
)


class ProbeError(ValueError):
    """Raised when the HDM10 planning probe cannot classify an input."""


@dataclasses.dataclass(frozen=True)
class DecoderTail:
    """Parsed active HDM9 decoder section inside format 0x0B."""

    decoder_payload: bytes
    tail: bytes
    q_payload: bytes
    scale_low3: bytes
    scale_high_mask: bytes


def combination_rank(indices: tuple[int, ...], *, n: int, k: int) -> int:
    """Return lexicographic rank for a sorted k-combination from range(n)."""

    if len(indices) != k:
        raise ProbeError(f"expected {k} indices; got {len(indices)}")
    if tuple(sorted(indices)) != indices:
        raise ProbeError("combination indices must be sorted")
    if indices and (indices[0] < 0 or indices[-1] >= n):
        raise ProbeError(f"combination indices must be in [0, {n})")
    rank = 0
    previous = -1
    remaining = k
    for value in indices:
        for candidate in range(previous + 1, value):
            rank += math.comb(n - candidate - 1, remaining - 1)
        previous = value
        remaining -= 1
    return rank


def combination_unrank(rank: int, *, n: int, k: int) -> tuple[int, ...]:
    """Inverse of :func:`combination_rank`."""

    total = math.comb(n, k)
    if not (0 <= rank < total):
        raise ProbeError(f"combination rank {rank} outside [0, {total})")
    out: list[int] = []
    start = 0
    remaining = k
    for _ in range(k):
        for candidate in range(start, n):
            count = math.comb(n - candidate - 1, remaining - 1)
            if rank >= count:
                rank -= count
                continue
            out.append(candidate)
            start = candidate + 1
            remaining -= 1
            break
    return tuple(out)


def combination_rank_byte_width(*, n: int, k: int) -> int:
    """Return minimal whole bytes needed for a fixed-cardinality rank."""

    total = math.comb(n, k)
    max_rank = max(total - 1, 0)
    return max(1, (max_rank.bit_length() + 7) // 8)


def high_mask_positions(mask: bytes, *, scale_count: int = PR106_HDM9_SCALE_COUNT) -> tuple[int, ...]:
    """Return set-bit positions from the HDM9 scale high-byte mask."""

    expected = PR106_HDM9_SCALE_HIGH_MASK_BYTES
    if len(mask) != expected:
        raise ProbeError(f"HDM9 scale high mask must be {expected} bytes; got {len(mask)}")
    padding_bits = len(mask) * 8 - scale_count
    if padding_bits:
        padding_mask = ((1 << padding_bits) - 1) << (8 - padding_bits)
        if mask[-1] & padding_mask:
            raise ProbeError("HDM9 scale high mask padding bits must be zero")
    return tuple(
        index
        for index in range(scale_count)
        if (mask[index // 8] >> (index % 8)) & 1
    )


def mask_from_positions(
    positions: tuple[int, ...],
    *,
    scale_count: int = PR106_HDM9_SCALE_COUNT,
) -> bytes:
    """Pack scale high-byte positions back to the HDM9 mask layout."""

    out = bytearray(PR106_HDM9_SCALE_HIGH_MASK_BYTES)
    for index in positions:
        if not (0 <= index < scale_count):
            raise ProbeError(f"scale high-mask index {index} outside [0, {scale_count})")
        out[index // 8] |= 1 << (index % 8)
    return bytes(out)


def extract_format0b_decoder_tail(member_payload: bytes) -> tuple[DecoderTail, dict[str, Any]]:
    """Parse a format-0x0B member and extract charged decoder-tail structure."""

    packet = parse_pr106_sidecar_packet(member_payload)
    expected_format = (
        PR106_SIDECAR_FORMAT_PR101_HDM9_HLM3_MAGICLESS_FIXED_META_NOOP_RANK_ELIDED
    )
    if packet.format_id != expected_format:
        raise ProbeError(
            f"expected format_id=0x{expected_format:02X}; got 0x{packet.format_id:02X}"
        )
    if len(packet.sidecar_payload) != PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES:
        raise ProbeError("format0B sidecar payload length mismatch")
    if len(packet.pr106_bytes) < 4 or packet.pr106_bytes[0] != 0xFF:
        raise ProbeError("format0B reconstructed PR106 bytes missing headerless marker")
    decoder_len = int.from_bytes(packet.pr106_bytes[1:4], "little")
    if decoder_len != PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES:
        raise ProbeError(
            "format0B decoder length mismatch: "
            f"got {decoder_len}, expected {PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES}"
        )
    decoder = packet.pr106_bytes[4 : 4 + decoder_len]
    if len(decoder) != decoder_len or not decoder.startswith(PR106_HDM9_HLM2_DECODER_MAGIC):
        raise ProbeError("format0B decoder payload is not HDM9")
    latents = packet.pr106_bytes[4 + decoder_len :]
    if len(latents) != PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES:
        raise ProbeError("format0B latent payload length mismatch")
    if not latents.startswith(PR106_HDM9_HLM3_LATENT_MAGIC):
        raise ProbeError("format0B latent payload is not HLM3")

    q_payload_len = sum(PR106_HDM9_Q_BROTLI_CHUNK_BYTES)
    expected_tail_len = q_payload_len + PR106_HDM9_SCALE_LOW3_BYTES + PR106_HDM9_SCALE_HIGH_MASK_BYTES
    tail = decoder[len(PR106_HDM9_HLM2_DECODER_MAGIC) :]
    if len(tail) != expected_tail_len:
        raise ProbeError(f"format0B decoder tail length mismatch: got {len(tail)}")
    q_payload = tail[:q_payload_len]
    scale_tail = tail[q_payload_len:]
    scale_low3 = scale_tail[:PR106_HDM9_SCALE_LOW3_BYTES]
    scale_high_mask = scale_tail[PR106_HDM9_SCALE_LOW3_BYTES:]
    layout = {
        "format_id": "0x0B",
        "reconstructed_pr106_bytes": len(packet.pr106_bytes),
        "reconstructed_pr106_sha256": sha256_bytes(packet.pr106_bytes),
        "sidecar_payload_bytes": len(packet.sidecar_payload),
        "sidecar_payload_sha256": sha256_bytes(packet.sidecar_payload),
        "member_decoder_tail_offset": 0,
        "member_decoder_tail_bytes": len(tail),
        "member_latent_tail_offset": len(tail),
        "member_latent_tail_bytes": (
            PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES - len(PR106_HDM9_HLM3_LATENT_MAGIC)
        ),
        "member_sidecar_offset": len(member_payload) - len(packet.sidecar_payload),
        "member_sidecar_bytes": len(packet.sidecar_payload),
    }
    return (
        DecoderTail(
            decoder_payload=decoder,
            tail=tail,
            q_payload=q_payload,
            scale_low3=scale_low3,
            scale_high_mask=scale_high_mask,
        ),
        layout,
    )


def raw_decoder_sha256_from_hdm9_payload(hdm9_decoder_payload: bytes) -> str:
    """Decode HDM9 through the HDM8 oracle and return raw decoder SHA-256."""

    hdm8_payload = decode_hdm9_decoder_to_hdm8_payload(hdm9_decoder_payload)
    raw = decode_hdm8_q_brotli_recipe_elided_fixture(hdm8_payload).to_raw()
    return sha256_bytes(raw)


def _q_chunk_rows(q_payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    for index, chunk_len in enumerate(PR106_HDM9_Q_BROTLI_CHUNK_BYTES):
        chunk = q_payload[cursor : cursor + chunk_len]
        cursor += chunk_len
        try:
            raw = brotli.decompress(chunk)
            raw_bytes = len(raw)
            raw_sha = sha256_bytes(raw)
            decodable = True
        except brotli.error:
            raw_bytes = None
            raw_sha = None
            decodable = False
        rows.append(
            {
                "index": index,
                "charged_bytes": len(chunk),
                "sha256": sha256_bytes(chunk),
                "brotli_decodable": decodable,
                "decoded_raw_bytes": raw_bytes,
                "decoded_raw_sha256": raw_sha,
            }
        )
    if cursor != len(q_payload):
        raise ProbeError("HDM9 q-payload constants do not consume the q payload")
    return rows


def _candidate_row(
    *,
    name: str,
    candidate_tail: bytes,
    reconstructed_tail: bytes,
    source_tail: DecoderTail,
    source_raw_sha256: str,
    runtime_supported_now: bool,
    candidate_runtime_decoder_implemented: bool,
    blockers: list[str],
    notes: list[str],
) -> dict[str, Any]:
    decoder_payload = PR106_HDM9_HLM2_DECODER_MAGIC + reconstructed_tail
    candidate_raw_sha = raw_decoder_sha256_from_hdm9_payload(decoder_payload)
    lossless = candidate_raw_sha == source_raw_sha256
    charged_delta = len(candidate_tail) - len(source_tail.tail)
    return {
        "name": name,
        "score_claim": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_archive_emitted": False,
        "runtime_supported_now": runtime_supported_now,
        "candidate_runtime_decoder_implemented": candidate_runtime_decoder_implemented,
        "lossless_raw_decoder_equivalence": lossless,
        "decoded_raw_sha256": candidate_raw_sha,
        "source_raw_sha256": source_raw_sha256,
        "candidate_charged_decoder_tail_bytes": len(candidate_tail),
        "current_charged_decoder_tail_bytes": len(source_tail.tail),
        "charged_decoder_byte_delta_vs_current": charged_delta,
        "charged_decoder_bytes_drop": charged_delta < 0,
        "rate_score_delta_if_components_equal": charged_delta * RATE_SCORE_PER_BYTE,
        "candidate_tail_sha256": sha256_bytes(candidate_tail),
        "reconstructed_hdm9_decoder_sha256": sha256_bytes(decoder_payload),
        "dispatch_blockers": blockers,
        "notes": notes,
    }


def build_hdm10_candidate_rows(source_tail: DecoderTail, source_raw_sha256: str) -> list[dict[str, Any]]:
    """Return deterministic HDM10 planning candidates for the HDM9 decoder tail."""

    positions = high_mask_positions(source_tail.scale_high_mask)
    popcount = len(positions)
    rank = combination_rank(positions, n=PR106_HDM9_SCALE_COUNT, k=popcount)
    rank_width = combination_rank_byte_width(n=PR106_HDM9_SCALE_COUNT, k=popcount)
    rank_blob = rank.to_bytes(rank_width, "little")
    decoded_positions = combination_unrank(rank, n=PR106_HDM9_SCALE_COUNT, k=popcount)
    if decoded_positions != positions:
        raise ProbeError("HDM10 combinadic rank failed position roundtrip")
    reconstructed_mask = mask_from_positions(decoded_positions)
    if reconstructed_mask != source_tail.scale_high_mask:
        raise ProbeError("HDM10 combinadic rank failed mask roundtrip")

    rows = [
        _candidate_row(
            name="hdm9_current_tail_control",
            candidate_tail=source_tail.tail,
            reconstructed_tail=source_tail.tail,
            source_tail=source_tail,
            source_raw_sha256=source_raw_sha256,
            runtime_supported_now=True,
            candidate_runtime_decoder_implemented=True,
            blockers=[
                "control_candidate_no_charged_decoder_byte_drop",
                "no_candidate_archive_emitted",
            ],
            notes=["current active HDM9 decoder tail"],
        ),
        _candidate_row(
            name="hdm10_generic_popcount_plus_combinadic_scale_mask_rank",
            candidate_tail=source_tail.q_payload
            + source_tail.scale_low3
            + bytes([popcount])
            + rank_blob,
            reconstructed_tail=source_tail.q_payload + source_tail.scale_low3 + reconstructed_mask,
            source_tail=source_tail,
            source_raw_sha256=source_raw_sha256,
            runtime_supported_now=False,
            candidate_runtime_decoder_implemented=False,
            blockers=[
                "no_charged_decoder_byte_drop",
                "requires_new_decoder_microcodec_runtime",
                "no_candidate_archive_emitted",
                "full_frame_inflate_output_parity_missing",
            ],
            notes=[
                "stores popcount as charged byte",
                f"combinadic_rank_width_bytes={rank_width}",
            ],
        ),
        _candidate_row(
            name="hdm10_fixed_popcount_combinadic_scale_mask_rank",
            candidate_tail=source_tail.q_payload + source_tail.scale_low3 + rank_blob,
            reconstructed_tail=source_tail.q_payload + source_tail.scale_low3 + reconstructed_mask,
            source_tail=source_tail,
            source_raw_sha256=source_raw_sha256,
            runtime_supported_now=False,
            candidate_runtime_decoder_implemented=False,
            blockers=[
                "requires_new_decoder_microcodec_runtime",
                "fixed_popcount_is_payload_specific_runtime_constant",
                "no_candidate_archive_emitted",
                "full_frame_inflate_output_parity_missing",
                "exact_cuda_dispatch_forbidden_in_this_task",
            ],
            notes=[
                f"runtime would fix scale_high_popcount={popcount}",
                f"combinadic_rank_width_bytes={rank_width}",
            ],
        ),
        _candidate_row(
            name="hdm10_fixed_scale_mask_runtime_constant",
            candidate_tail=source_tail.q_payload + source_tail.scale_low3,
            reconstructed_tail=source_tail.q_payload
            + source_tail.scale_low3
            + source_tail.scale_high_mask,
            source_tail=source_tail,
            source_raw_sha256=source_raw_sha256,
            runtime_supported_now=False,
            candidate_runtime_decoder_implemented=False,
            blockers=[
                "would_move_exact_scale_mask_payload_bytes_into_runtime_source",
                "requires_new_decoder_microcodec_runtime",
                "no_candidate_archive_emitted",
                "full_frame_inflate_output_parity_missing",
                "exact_cuda_dispatch_forbidden_in_this_task",
            ],
            notes=[
                "byte-positive but contest-compliance-risky",
                f"runtime_constant_scale_high_mask={source_tail.scale_high_mask.hex()}",
            ],
        ),
    ]
    for row in rows:
        if not row["lossless_raw_decoder_equivalence"]:
            raise ProbeError(f"{row['name']} failed raw decoder equivalence")
    return sorted(
        rows,
        key=lambda row: (
            int(row["charged_decoder_byte_delta_vs_current"]),
            str(row["name"]),
        ),
    )


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive = [row for row in rows if int(row["charged_decoder_byte_delta_vs_current"]) < 0]
    positive_without_exact_mask = [
        row
        for row in positive
        if row["name"] != "hdm10_fixed_scale_mask_runtime_constant"
    ]
    contest_safe_positive = [
        row
        for row in positive
        if not any(
            "runtime_constant" in blocker
            or "payload_specific" in blocker
            or "runtime_source" in blocker
            or "would_move" in blocker
            for blocker in row["dispatch_blockers"]
        )
    ]
    if not positive:
        verdict = "fail_closed_no_charged_decoder_byte_drop"
        recommended_next = (
            "Preserve this as a negative HDM10 planning signal and keep decoder "
            "work on chunk payload/context transforms."
        )
    else:
        verdict = "byte_positive_hdm10_planning_candidates_require_runtime_and_custody"
        recommended_next = (
            "Do not dispatch. The only byte-positive rows require a new runtime "
            "decoder and payload-specific constants; route through a separate "
            "PacketIR design review before any archive builder exists."
        )
    best = positive[0] if positive else None
    best_without_exact_mask = positive_without_exact_mask[0] if positive_without_exact_mask else None
    return {
        "verdict": verdict,
        "score_claim": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "charged_decoder_byte_positive_candidate_count": len(positive),
        "contest_safe_byte_positive_candidate_count": len(contest_safe_positive),
        "best_byte_positive_candidate": None if best is None else best["name"],
        "best_byte_positive_delta_bytes": None
        if best is None
        else int(best["charged_decoder_byte_delta_vs_current"]),
        "best_without_exact_mask_constant_candidate": None
        if best_without_exact_mask is None
        else best_without_exact_mask["name"],
        "best_without_exact_mask_constant_delta_bytes": None
        if best_without_exact_mask is None
        else int(best_without_exact_mask["charged_decoder_byte_delta_vs_current"]),
        "dispatch_blockers": [
            "planning_probe_only_no_archive_emitted",
            "no_hdm10_runtime_decoder_implemented",
            "candidate_archive_manifest_missing",
            "full_frame_inflate_output_parity_missing",
            "score_claim_false_until_byte_closed_archive_runtime_exists",
            "ready_for_exact_eval_dispatch_false_by_task_scope",
        ],
        "recommended_next": recommended_next,
    }


def build_probe(source_archive: Path) -> dict[str, Any]:
    """Build a planning-only HDM10 decoder microcodec probe manifest."""

    archive_bytes = source_archive.read_bytes()
    member = read_single_stored_member_archive(archive_bytes)
    tail, member_layout = extract_format0b_decoder_tail(member.payload)
    source_raw_sha = raw_decoder_sha256_from_hdm9_payload(tail.decoder_payload)
    rows = build_hdm10_candidate_rows(tail, source_raw_sha)
    positions = high_mask_positions(tail.scale_high_mask)
    q_rows = _q_chunk_rows(tail.q_payload)
    return {
        "schema": "pr106_format0b_hdm10_decoder_microcodec_probe_v1",
        "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "tool": "tools/probe_pr106_format0b_hdm10_decoder_microcodec.py",
        "planning_only": True,
        "research_only": True,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_archive_preflight": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "source": {
            "archive_path": repo_relative(source_archive, REPO_ROOT),
            "archive_bytes": source_archive.stat().st_size,
            "archive_sha256": sha256_bytes(archive_bytes),
            "member_name": member.name,
            "member_bytes": len(member.payload),
            "member_sha256": sha256_bytes(member.payload),
            "format_id": "0x0B",
        },
        "member_layout": member_layout,
        "hdm9_decoder": {
            "decoder_payload_bytes": len(tail.decoder_payload),
            "decoder_payload_sha256": sha256_bytes(tail.decoder_payload),
            "charged_decoder_tail_bytes": len(tail.tail),
            "charged_decoder_tail_sha256": sha256_bytes(tail.tail),
            "q_payload_bytes": len(tail.q_payload),
            "q_payload_sha256": sha256_bytes(tail.q_payload),
            "q_chunk_rows": q_rows,
            "scale_tail_bytes": len(tail.scale_low3) + len(tail.scale_high_mask),
            "scale_low3_bytes": len(tail.scale_low3),
            "scale_low3_sha256": sha256_bytes(tail.scale_low3),
            "scale_high_mask_bytes": len(tail.scale_high_mask),
            "scale_high_mask_hex": tail.scale_high_mask.hex(),
            "scale_high_mask_popcount": len(positions),
            "scale_high_mask_positions": list(positions),
            "raw_decoder_sha256": source_raw_sha,
        },
        "candidates": rows,
        "decision": _decision(rows),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render a compact operator-facing report."""

    source = payload["source"]
    decoder = payload["hdm9_decoder"]
    decision = payload["decision"]
    lines = [
        "# PR106 Format 0x0B HDM10 Decoder Microcodec Probe",
        "",
        f"- score_claim: `{str(payload['score_claim']).lower()}`",
        f"- dispatch_attempted: `{str(payload['dispatch_attempted']).lower()}`",
        f"- ready_for_exact_eval_dispatch: `{str(payload['ready_for_exact_eval_dispatch']).lower()}`",
        f"- verdict: `{decision['verdict']}`",
        "",
        "## Source",
        "",
        f"- archive: `{source['archive_path']}`",
        f"- archive bytes: `{source['archive_bytes']}`",
        f"- archive sha256: `{source['archive_sha256']}`",
        f"- member: `{source['member_name']}`",
        "",
        "## HDM9 Decoder Tail",
        "",
        f"- charged decoder tail: `{decoder['charged_decoder_tail_bytes']}` bytes",
        f"- q payload: `{decoder['q_payload_bytes']}` bytes",
        f"- scale low3: `{decoder['scale_low3_bytes']}` bytes",
        f"- scale high mask: `{decoder['scale_high_mask_bytes']}` bytes",
        f"- scale high mask hex: `{decoder['scale_high_mask_hex']}`",
        f"- scale high mask popcount: `{decoder['scale_high_mask_popcount']}`",
        "",
        "## Candidate Rows",
        "",
        "| candidate | raw equivalence | charged tail bytes | delta | runtime now | blockers |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["candidates"]:
        blockers = ", ".join(f"`{blocker}`" for blocker in row["dispatch_blockers"])
        lines.append(
            "| `{name}` | `{equiv}` | {bytes_} | {delta} | `{runtime}` | {blockers} |".format(
                name=row["name"],
                equiv=str(row["lossless_raw_decoder_equivalence"]).lower(),
                bytes_=row["candidate_charged_decoder_tail_bytes"],
                delta=row["charged_decoder_byte_delta_vs_current"],
                runtime=str(row["runtime_supported_now"]).lower(),
                blockers=blockers,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- byte-positive candidate count: `{decision['charged_decoder_byte_positive_candidate_count']}`",
            f"- contest-safe byte-positive candidate count: `{decision['contest_safe_byte_positive_candidate_count']}`",
            f"- best byte-positive candidate: `{decision['best_byte_positive_candidate']}`",
            f"- best byte-positive delta bytes: `{decision['best_byte_positive_delta_bytes']}`",
            "",
            "## Blockers",
            "",
        ]
    )
    lines.extend(f"- `{blocker}`" for blocker in decision["dispatch_blockers"])
    lines.extend(["", "## Recommendation", "", decision["recommended_next"], ""])
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    try:
        payload = build_probe(args.source_archive)
    except (OSError, ProbeError, ValueError) as exc:
        print(f"FATAL: PR106 format-0x0B HDM10 decoder probe failed: {exc}", file=sys.stderr)
        return 2
    payload = attach_tool_run_manifest(
        payload,
        tool=Path(__file__).relative_to(REPO_ROOT).as_posix(),
        argv=raw_argv,
        input_paths=[args.source_archive],
        repo_root=REPO_ROOT,
        output_path=args.json_out,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json_text(payload), encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
