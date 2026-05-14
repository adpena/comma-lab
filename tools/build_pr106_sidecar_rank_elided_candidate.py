#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a no-score PR106/R2 sidecar rank-elided prototype archive.

The landed PR106/R2 PR101 sidecar grammar carries one byte of Huffman
length-rank payload plus one metadata byte recording that width. For the fixed
PR106 schema, the delta alphabet has four symbols and the minimum code length is
two bits, so the only Kraft-tight length vector is the uniform 2-bit code. A
successor runtime can infer that rank instead of storing it.

This tool emits a deterministic format-0x04 prototype packet:

``magic | 0x04 | pr106_len(u32le) | pr106_bytes | rank_elided_payload | meta5``

where ``meta5 = noop_count(u16le) | dim_bytes(u16le) | noop_rank_bytes(u8)``.
The sidecar length prefix is also elided because the ZIP member payload length
delimits the packet. This is a byte-level candidate generator and proof artifact
only: it does not claim score movement, and exact eval remains blocked until
runtime decode/apply and same-runtime parity/custody proofs are attached.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import struct
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.packet_compiler.pr106_sidecar_packet import (  # noqa: E402
    PR106_PR101_RANKED_SCHEMA,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    PR106_SIDECAR_MAGIC,
    decode_pr101_ranked_sidecar_payload_to_dim_delta,
    decode_pr106_sidecar_packet_dim_delta,
    emit_single_stored_member_archive,
    encode_pr101_ranked_sidecar_payload,
    parse_pr106_sidecar_packet,
    read_single_stored_member_archive,
    sha256_hex,
)

TOOL = "tools/build_pr106_sidecar_rank_elided_candidate.py"
SCHEMA = "pr106_sidecar_rank_elided_format04_candidate_v1"
FORMAT_ID_RANK_ELIDED = PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED
ARCHIVE_BYTES_DENOMINATOR = 37_545_489


class RankElidedSidecar(NamedTuple):
    """Format-0x04 sidecar bytes plus enough data to re-expand for proof."""

    source_pr101_payload: bytes
    source_pr101_framing_meta: bytes
    rank_elided_payload: bytes
    rank_elided_meta: bytes
    elided_length_rank_blob: bytes
    noop_count: int
    dim_bytes: int
    source_rank_bytes: int
    noop_rank_bytes: int

    @property
    def source_charged_bytes(self) -> int:
        return len(self.source_pr101_payload) + len(self.source_pr101_framing_meta)

    @property
    def rank_elided_charged_bytes(self) -> int:
        return len(self.rank_elided_payload) + len(self.rank_elided_meta)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _section_row(
    name: str,
    *,
    offset: int,
    payload: bytes,
    score_affecting: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "offset": offset,
        "offset_start": offset,
        "bytes": len(payload),
        "byte_count": len(payload),
        "end_offset": offset + len(payload),
        "offset_end_exclusive": offset + len(payload),
        "sha256": sha256_hex(payload),
        "score_affecting": score_affecting,
    }


def _rank_elided_expected_payload_bytes(
    *,
    noop_count: int,
    dim_bytes: int,
    noop_rank_bytes: int,
) -> int:
    if not (0 <= int(noop_count) <= int(PR106_PR101_RANKED_SCHEMA.n_pairs)):
        raise ValueError(
            f"noop_count must be in [0, {PR106_PR101_RANKED_SCHEMA.n_pairs}]"
        )
    n_valid = int(PR106_PR101_RANKED_SCHEMA.n_pairs) - int(noop_count)
    expected_huff_bytes = (
        n_valid * int(PR106_PR101_RANKED_SCHEMA.huff_min_len) + 7
    ) // 8
    return int(dim_bytes) + expected_huff_bytes + int(noop_rank_bytes)


def build_rank_elided_sidecar(
    dims: np.ndarray,
    deltas: np.ndarray,
) -> RankElidedSidecar:
    """Return a format-0x04 sidecar obtained from the canonical PR101 grammar."""

    payload, framing_meta = encode_pr101_ranked_sidecar_payload(
        dims,
        deltas,
        schema=PR106_PR101_RANKED_SCHEMA,
    )
    if len(framing_meta) != 6:
        raise ValueError(f"expected six PR101 framing bytes; got {len(framing_meta)}")
    noop_count, dim_bytes, rank_bytes, noop_rank_bytes = struct.unpack(
        "<HHBB",
        framing_meta,
    )
    if rank_bytes != 1:
        raise ValueError(
            "PR106 rank-elided candidate requires a one-byte length rank; "
            f"got {rank_bytes}"
        )
    if len(payload) < dim_bytes + rank_bytes + noop_rank_bytes:
        raise ValueError("PR101 sidecar payload is too short for recorded widths")
    length_rank_blob = payload[dim_bytes : dim_bytes + rank_bytes]
    if length_rank_blob != b"\x00":
        raise ValueError(
            "PR106 rank-elided candidate requires the fixed uniform Huffman "
            f"length-rank blob b'\\x00'; got {length_rank_blob.hex()}"
        )

    rank_elided_payload = payload[:dim_bytes] + payload[dim_bytes + rank_bytes :]
    rank_elided_meta = struct.pack("<HHB", noop_count, dim_bytes, noop_rank_bytes)
    return RankElidedSidecar(
        source_pr101_payload=payload,
        source_pr101_framing_meta=framing_meta,
        rank_elided_payload=rank_elided_payload,
        rank_elided_meta=rank_elided_meta,
        elided_length_rank_blob=length_rank_blob,
        noop_count=int(noop_count),
        dim_bytes=int(dim_bytes),
        source_rank_bytes=int(rank_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
    )


def reexpand_rank_elided_sidecar(
    rank_elided_payload: bytes,
    rank_elided_meta: bytes,
) -> tuple[bytes, bytes]:
    """Reconstruct the equivalent format-0x02 PR101 payload and six-byte meta."""

    if len(rank_elided_meta) != 5:
        raise ValueError(
            f"format-0x04 rank-elided meta must be 5 bytes; got {len(rank_elided_meta)}"
    )
    noop_count, dim_bytes, noop_rank_bytes = struct.unpack("<HHB", rank_elided_meta)
    expected_payload_bytes = _rank_elided_expected_payload_bytes(
        noop_count=int(noop_count),
        dim_bytes=int(dim_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
    )
    if len(rank_elided_payload) != expected_payload_bytes:
        raise ValueError(
            "rank-elided payload length mismatch: "
            f"got {len(rank_elided_payload)} bytes; expected {expected_payload_bytes}"
        )
    source_payload = (
        rank_elided_payload[:dim_bytes]
        + b"\x00"
        + rank_elided_payload[dim_bytes:]
    )
    source_meta = struct.pack("<HHBB", noop_count, dim_bytes, 1, noop_rank_bytes)
    return source_payload, source_meta


def decode_rank_elided_sidecar_payload(
    rank_elided_payload: bytes,
    rank_elided_meta: bytes,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode format-0x04 sidecar bytes by re-expanding the fixed rank."""

    payload, framing_meta = reexpand_rank_elided_sidecar(
        rank_elided_payload,
        rank_elided_meta,
    )
    return decode_pr101_ranked_sidecar_payload_to_dim_delta(
        payload,
        framing_meta,
        schema=PR106_PR101_RANKED_SCHEMA,
    )


def emit_rank_elided_packet(
    *,
    pr106_bytes: bytes,
    rank_elided_payload: bytes,
    rank_elided_meta: bytes,
) -> bytes:
    """Emit a format-0x04 prototype PR106 sidecar packet."""

    if len(pr106_bytes) > 0xFFFF_FFFF:
        raise ValueError("PR106 inner payload too large for u32 length field")
    if len(rank_elided_meta) != 5:
        raise ValueError(
            f"format-0x04 rank-elided meta must be 5 bytes; got {len(rank_elided_meta)}"
        )
    return b"".join(
        [
            bytes([PR106_SIDECAR_MAGIC, FORMAT_ID_RANK_ELIDED]),
            struct.pack("<I", len(pr106_bytes)),
            pr106_bytes,
            rank_elided_payload,
            rank_elided_meta,
        ]
    )


def parse_rank_elided_packet(payload: bytes) -> dict[str, Any]:
    """Parse the local format-0x04 packet for static proof checks."""

    if len(payload) < 2 + 4 + 5:
        raise ValueError(f"format-0x04 payload too short: {len(payload)} bytes")
    if payload[0] != PR106_SIDECAR_MAGIC:
        raise ValueError(
            f"sidecar magic mismatch: got 0x{payload[0]:02X}, "
            f"expected 0x{PR106_SIDECAR_MAGIC:02X}"
        )
    if payload[1] != FORMAT_ID_RANK_ELIDED:
        raise ValueError(
            f"format mismatch: got 0x{payload[1]:02X}, expected 0x{FORMAT_ID_RANK_ELIDED:02X}"
        )
    pr106_len = struct.unpack_from("<I", payload, 2)[0]
    pr106_start = 6
    pr106_end = pr106_start + pr106_len
    if pr106_end + 5 > len(payload):
        raise ValueError(
            f"format-0x04 packet truncated: pr106_len={pr106_len} total={len(payload)}"
        )
    rank_elided_payload = payload[pr106_end:-5]
    rank_elided_meta = payload[-5:]
    noop_count, dim_bytes, noop_rank_bytes = struct.unpack("<HHB", rank_elided_meta)
    expected_payload_bytes = _rank_elided_expected_payload_bytes(
        noop_count=int(noop_count),
        dim_bytes=int(dim_bytes),
        noop_rank_bytes=int(noop_rank_bytes),
    )
    if len(rank_elided_payload) != expected_payload_bytes:
        raise ValueError(
            "format-0x04 rank-elided payload length mismatch: "
            f"got {len(rank_elided_payload)} bytes; expected {expected_payload_bytes}"
        )
    return {
        "format_id": FORMAT_ID_RANK_ELIDED,
        "pr106_bytes": payload[pr106_start:pr106_end],
        "rank_elided_payload": rank_elided_payload,
        "rank_elided_meta": rank_elided_meta,
    }


def rank_elided_consumed_byte_proof(
    *,
    pr106_bytes: bytes,
    rank_elided_payload: bytes,
    rank_elided_meta: bytes,
) -> dict[str, Any]:
    """Return parser-level byte accounting for the format-0x04 prototype."""

    sections: list[dict[str, Any]] = []
    offset = 0

    def add(name: str, payload: bytes, *, score_affecting: bool) -> None:
        nonlocal offset
        sections.append(
            _section_row(
                name,
                offset=offset,
                payload=payload,
                score_affecting=score_affecting,
            )
        )
        offset += len(payload)

    add("magic", bytes([PR106_SIDECAR_MAGIC]), score_affecting=False)
    add("format_id", bytes([FORMAT_ID_RANK_ELIDED]), score_affecting=False)
    add("pr106_len_le_u32", struct.pack("<I", len(pr106_bytes)), score_affecting=False)
    add("pr106_payload", pr106_bytes, score_affecting=True)
    add("rank_elided_sidecar_payload", rank_elided_payload, score_affecting=True)
    add("rank_elided_meta", rank_elided_meta, score_affecting=True)

    emitted = emit_rank_elided_packet(
        pr106_bytes=pr106_bytes,
        rank_elided_payload=rank_elided_payload,
        rank_elided_meta=rank_elided_meta,
    )
    parsed = parse_rank_elided_packet(emitted)
    accounted = sum(int(row["bytes"]) for row in sections)
    cursor = 0
    gaps: list[dict[str, int]] = []
    for row in sections:
        row_offset = int(row["offset"])
        if row_offset != cursor:
            gaps.append({"expected_offset": cursor, "observed_offset": row_offset})
        cursor = int(row["end_offset"])

    return {
        "schema": "pr106_sidecar_rank_elided_consumed_byte_proof_v1",
        "proof_scope": "packet_ir_parser_accounting_not_runtime_inflate_consumption",
        "runtime_consumption_claim": False,
        "format_id": f"0x{FORMAT_ID_RANK_ELIDED:02X}",
        "emitted_payload_bytes": len(emitted),
        "emitted_payload_sha256": sha256_hex(emitted),
        "accounted_payload_bytes": accounted,
        "all_payload_bytes_accounted": accounted == len(emitted) and not gaps,
        "unconsumed_trailing_bytes": max(0, len(emitted) - accounted),
        "section_gaps": gaps,
        "parsed_reemit_identity": parsed["pr106_bytes"] == pr106_bytes
        and parsed["rank_elided_payload"] == rank_elided_payload
        and parsed["rank_elided_meta"] == rank_elided_meta,
        "score_affecting_section_names": [
            str(row["name"]) for row in sections if bool(row["score_affecting"])
        ],
        "sections": sections,
    }


def build_rank_elided_candidate_report(
    *,
    source_archive: Path,
    output_dir: Path,
    source_label: str = "pr106_r2_packetir_source",
    expected_source_sha256: str | None = None,
    expected_member_name: str | None = None,
    candidate_archive_name: str = "pr106_sidecar_rank_elided_format04_candidate.zip",
    write_markdown: bool = True,
) -> dict[str, Any]:
    """Build the prototype archive and write a JSON proof manifest."""

    source_archive = Path(source_archive)
    output_dir = Path(output_dir)
    source_archive_bytes = source_archive.read_bytes()
    source_archive_sha = sha256_hex(source_archive_bytes)
    if expected_source_sha256 is not None and source_archive_sha != expected_source_sha256:
        raise ValueError(
            f"source SHA mismatch for {source_archive}: got {source_archive_sha}, "
            f"expected {expected_source_sha256}"
        )
    source_member = read_single_stored_member_archive(
        source_archive_bytes,
        expected_member_name=expected_member_name,
    )
    source_packet = parse_pr106_sidecar_packet(source_member.payload)
    source_dims, source_deltas = decode_pr106_sidecar_packet_dim_delta(source_packet)
    rank_elided = build_rank_elided_sidecar(source_dims, source_deltas)
    decoded_dims, decoded_deltas = decode_rank_elided_sidecar_payload(
        rank_elided.rank_elided_payload,
        rank_elided.rank_elided_meta,
    )
    semantic_equivalence = bool(
        np.array_equal(decoded_dims, source_dims)
        and np.array_equal(decoded_deltas, source_deltas)
    )
    if not semantic_equivalence:
        raise ValueError("format-0x04 rank-elided sidecar failed semantic equivalence")

    source_pr101_same_as_packet = (
        source_packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR
        and source_packet.sidecar_payload == rank_elided.source_pr101_payload
        and source_packet.framing_meta == rank_elided.source_pr101_framing_meta
    )
    candidate_payload = emit_rank_elided_packet(
        pr106_bytes=source_packet.pr106_bytes,
        rank_elided_payload=rank_elided.rank_elided_payload,
        rank_elided_meta=rank_elided.rank_elided_meta,
    )
    candidate_member = replace(source_member, payload=candidate_payload)
    candidate_archive_bytes = emit_single_stored_member_archive(candidate_member)

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_archive_path = output_dir / candidate_archive_name
    candidate_archive_path.write_bytes(candidate_archive_bytes)
    manifest_path = output_dir / "rank_elided_candidate_manifest.json"
    markdown_path = output_dir / "rank_elided_candidate.md"

    candidate_archive_sha = sha256_hex(candidate_archive_bytes)
    candidate_payload_sha = sha256_hex(candidate_payload)
    source_payload_charged = len(source_packet.sidecar_payload) + (
        0 if source_packet.framing_meta is None else len(source_packet.framing_meta)
    )
    candidate_archive_byte_delta = len(candidate_archive_bytes) - len(source_archive_bytes)
    candidate_payload_byte_delta = len(candidate_payload) - len(source_member.payload)
    charged_sidecar_byte_delta = (
        rank_elided.rank_elided_charged_bytes - source_payload_charged
    )

    blockers = [
        "runtime_decoder_missing_for_format_0x04",
        "runtime_decode_apply_proof_required_for_new_candidate_archive",
        "existing_scored_runtime_does_not_consume_format_0x04",
        "full_frame_same_runtime_parity_or_same_runtime_auth_eval_missing",
        "exact_cuda_auth_eval_missing",
    ]
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "source_label": source_label,
        "evidence_axis": "packet-ir-parser-local-no-score",
        "research_only": True,
        "score_claim": False,
        "contest_axis_claim": False,
        "promotion_eligible": False,
        "dispatch_attempted": False,
        "source_archive_sha256": source_archive_sha,
        "source_archive_bytes": len(source_archive_bytes),
        "candidate_archive_sha256": candidate_archive_sha,
        "candidate_archive_bytes": len(candidate_archive_bytes),
        "candidate_archive_byte_delta": candidate_archive_byte_delta,
        "archive_build_blockers": [],
        "candidate_diff_audit": {
            "blockers": [],
            "total_byte_delta": candidate_archive_byte_delta,
        },
        "byte_closed_archive_emitted": True,
        "byte_closed_for_existing_runtime": False,
        "packet_ir_decoder_implemented": True,
        "existing_scored_runtime_decoder_implemented": False,
        "ready_for_exact_eval_dispatch": False,
        "exact_next_dispatch_command": None,
        "dispatch_blockers": blockers,
        "source_archive": {
            "path": source_archive.as_posix(),
            "bytes": len(source_archive_bytes),
            "sha256": source_archive_sha,
            "expected_sha256": expected_source_sha256,
            "expected_sha256_matches": (
                None if expected_source_sha256 is None else source_archive_sha == expected_source_sha256
            ),
        },
        "source_member": {
            "name": source_member.name,
            "payload_bytes": len(source_member.payload),
            "payload_sha256": sha256_hex(source_member.payload),
        },
        "source_packet": {
            "format_id": f"0x{source_packet.format_id:02X}",
            "pr106_bytes": len(source_packet.pr106_bytes),
            "pr106_sha256": sha256_hex(source_packet.pr106_bytes),
            "sidecar_payload_bytes": len(source_packet.sidecar_payload),
            "sidecar_payload_sha256": sha256_hex(source_packet.sidecar_payload),
            "framing_meta_bytes": 0
            if source_packet.framing_meta is None
            else len(source_packet.framing_meta),
            "framing_meta_sha256": None
            if source_packet.framing_meta is None
            else sha256_hex(source_packet.framing_meta),
            "charged_sidecar_bytes": source_payload_charged,
            "source_pr101_canonical_payload_matches_packet": source_pr101_same_as_packet,
        },
        "candidate_archive": {
            "path": candidate_archive_path.as_posix(),
            "bytes": len(candidate_archive_bytes),
            "sha256": candidate_archive_sha,
            "byte_delta_vs_source_archive": candidate_archive_byte_delta,
            "rate_term_delta_if_components_equal": (
                25.0 * candidate_archive_byte_delta / ARCHIVE_BYTES_DENOMINATOR
            ),
        },
        "candidate_member": {
            "name": candidate_member.name,
            "payload_bytes": len(candidate_payload),
            "payload_sha256": candidate_payload_sha,
            "payload_byte_delta_vs_source_member": candidate_payload_byte_delta,
        },
        "candidate_packet": {
            "format_id": f"0x{FORMAT_ID_RANK_ELIDED:02X}",
            "schema": "rank_elided_pr101_no_sidecar_len_v1",
            "packet_ir_decoder_implemented": True,
            "runtime_decoder_implemented": False,
            "runtime_decoder_scope": "packet_ir_local_parser_only_not_existing_scored_runtime",
            "pr106_bytes": len(source_packet.pr106_bytes),
            "pr106_sha256": sha256_hex(source_packet.pr106_bytes),
            "rank_elided_sidecar_payload_bytes": len(rank_elided.rank_elided_payload),
            "rank_elided_sidecar_payload_sha256": sha256_hex(
                rank_elided.rank_elided_payload
            ),
            "rank_elided_meta_bytes": len(rank_elided.rank_elided_meta),
            "rank_elided_meta_sha256": sha256_hex(rank_elided.rank_elided_meta),
            "charged_sidecar_bytes": rank_elided.rank_elided_charged_bytes,
            "charged_sidecar_byte_delta_vs_source_packet": charged_sidecar_byte_delta,
            "packet_payload_byte_delta_vs_source_member": candidate_payload_byte_delta,
        },
        "rank_elision": {
            "fixed_huffman_length_vector_reason": (
                "PR106/R2 delta alphabet has four symbols and huff_min_len=2; "
                "four 2-bit codewords exactly exhaust Kraft, so length rank is fixed"
            ),
            "elided_length_rank_blob_bytes": len(rank_elided.elided_length_rank_blob),
            "elided_length_rank_blob_sha256": sha256_hex(
                rank_elided.elided_length_rank_blob
            ),
            "elided_rank_bytes_meta_field_bytes": 1,
            "elided_sidecar_len_prefix_bytes": 2,
            "total_packet_payload_savings_bytes": -candidate_payload_byte_delta,
        },
        "semantic_equivalence": {
            "rank_elided_decodes_to_source_dim_delta": semantic_equivalence,
            "dim_sha256": sha256_hex(source_dims.tobytes()),
            "delta_q_sha256": sha256_hex(source_deltas.tobytes()),
            "n_pairs": int(source_dims.size),
            "n_non_noop": int(np.count_nonzero(source_dims != PR106_PR101_RANKED_SCHEMA.no_op_sentinel)),
        },
        "packet_ir_consumed_byte_proof": rank_elided_consumed_byte_proof(
            pr106_bytes=source_packet.pr106_bytes,
            rank_elided_payload=rank_elided.rank_elided_payload,
            rank_elided_meta=rank_elided.rank_elided_meta,
        ),
        "required_next_proof": (
            "prove runtime decode/apply consumption, prove same-runtime "
            "full-frame parity, then claim lane and run exact CUDA auth eval"
        ),
    }

    manifest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    report["manifest_path"] = manifest_path.as_posix()
    if write_markdown:
        markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
        report["markdown_path"] = markdown_path.as_posix()
        manifest_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
            encoding="utf-8",
        )
    return report


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render a short operator-facing no-score report."""

    source = report["source_archive"]
    candidate = report["candidate_archive"]
    packet = report["candidate_packet"]
    blockers = "\n".join(f"- `{item}`" for item in report["dispatch_blockers"])
    return "\n".join(
        [
            "# PR106/R2 Rank-Elided Sidecar Candidate",
            "",
            "This is a packet-IR local candidate artifact only. It is not a score claim,",
            "not byte-closed for the existing scored runtime, not promotion-eligible,",
            "and not ready for exact-eval dispatch.",
            "",
            "## Byte Accounting",
            "",
            "| field | value |",
            "| --- | ---: |",
            f"| source archive bytes | {source['bytes']} |",
            f"| candidate archive bytes | {candidate['bytes']} |",
            f"| archive byte delta | {candidate['byte_delta_vs_source_archive']} |",
            f"| candidate format | {packet['format_id']} |",
            f"| candidate charged sidecar bytes | {packet['charged_sidecar_bytes']} |",
            f"| charged sidecar delta | {packet['charged_sidecar_byte_delta_vs_source_packet']} |",
            "",
            "## Dispatch Status",
            "",
            f"- score_claim: `{report['score_claim']}`",
            f"- byte_closed_for_existing_runtime: `{report['byte_closed_for_existing_runtime']}`",
            f"- existing_scored_runtime_decoder_implemented: `{report['existing_scored_runtime_decoder_implemented']}`",
            f"- ready_for_exact_eval_dispatch: `{report['ready_for_exact_eval_dispatch']}`",
            "- exact_next_dispatch_command: `None`",
            "",
            "## Blockers",
            "",
            blockers,
            "",
            "## Artifacts",
            "",
            f"- candidate archive: `{candidate['path']}`",
            f"- candidate sha256: `{candidate['sha256']}`",
            f"- manifest: `{report.get('manifest_path', '')}`",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-label", default="pr106_r2_packetir_source")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-member-name")
    parser.add_argument(
        "--candidate-archive-name",
        default="pr106_sidecar_rank_elided_format04_candidate.zip",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Write only the JSON manifest and candidate archive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_rank_elided_candidate_report(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        source_label=args.source_label,
        expected_source_sha256=args.expected_source_sha256,
        expected_member_name=args.expected_member_name,
        candidate_archive_name=args.candidate_archive_name,
        write_markdown=not args.no_markdown,
    )
    print(
        "wrote rank-elided no-score candidate "
        f"{report['candidate_archive']['path']} "
        f"delta={report['candidate_archive']['byte_delta_vs_source_archive']} "
        f"ready_for_exact_eval_dispatch={report['ready_for_exact_eval_dispatch']}"
    )


if __name__ == "__main__":
    main()
