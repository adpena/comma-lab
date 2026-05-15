#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile byte layout for public HNeRV frontier archives.

This is a deconstruction tool, not a scorer. It reads a contest archive,
extracts the single charged payload member, and emits deterministic section
bytes, SHA-256, and entropy so public-submission ideas can feed repacking and
Lagrangian allocation work without relying on prose notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from zipfile import ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.packet_compiler.pr106_sidecar_packet import (
    PR106_HDM9_HLM2_DECODER_MAGIC,
    PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES,
    PR106_HDM9_HLM3_LATENT_MAGIC,
    PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES,
    PR106_PR101_FIXED_META_PAYLOAD_BYTES,
    PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES,
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
    PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED,
    PR106_SIDECAR_MAGIC,
)

PR101_DECODER_BLOB_LEN = 162_164
PR101_LATENT_BLOB_LEN = 15_387

PR103_SCA_LEN = 56
PR103_BR_LEN = 7_097
PR103_HIST_LEN = 895
PR103_MERGED_AC_LEN = 153_856
PR103_LATENT_META_LEN = 112
PR103_LO_LEN = 15_537
PR103_HI_HIST_LEN = 15
PR106_HDM9_HLM3_MAGICLESS_FIXED_PAYLOAD_KIND = (
    "pr106_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided"
)


def pr106_hdm9_hlm3_magicless_payload_bytes() -> int:
    return (
        PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES
        - len(PR106_HDM9_HLM2_DECODER_MAGIC)
        + PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES
        - len(PR106_HDM9_HLM3_LATENT_MAGIC)
        + PR106_PR101_FIXED_META_NOOP_RANK_ELIDED_PAYLOAD_BYTES
    )


@dataclass(frozen=True)
class SectionProfile:
    name: str
    start: int
    end: int
    bytes: int
    sha256: str
    entropy_bits_per_byte: float


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def entropy_bits_per_byte(data: bytes) -> float:
    if not data:
        return 0.0
    n = len(data)
    counts = Counter(data)
    return -sum((count / n) * math.log2(count / n) for count in counts.values())


def section(name: str, blob: bytes, start: int, end: int) -> SectionProfile:
    if start < 0 or end < start or end > len(blob):
        raise ValueError(f"bad section {name}: start={start} end={end} len={len(blob)}")
    data = blob[start:end]
    return SectionProfile(
        name=name,
        start=start,
        end=end,
        bytes=len(data),
        sha256=sha256_bytes(data),
        entropy_bits_per_byte=round(entropy_bits_per_byte(data), 6),
    )


def extract_single_member(archive: Path) -> tuple[str, bytes, int, str]:
    archive_blob = archive.read_bytes()
    with ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"{archive} has {len(infos)} file members; expected exactly one")
        member = infos[0]
        payload = zf.read(member.filename)
    return member.filename, payload, len(archive_blob), sha256_bytes(archive_blob)


def infer_profile_kind(kind: str, archive: Path, member_name: str, payload: bytes) -> str:
    if kind != "auto":
        return kind
    text = f"{archive} {member_name}".lower()
    if payload[:1] == bytes([PR106_SIDECAR_MAGIC]):
        return "pr106_sidecar_wrapper"
    if len(payload) == pr106_hdm9_hlm3_magicless_payload_bytes():
        return PR106_HDM9_HLM3_MAGICLESS_FIXED_PAYLOAD_KIND
    if "pr101" in text or "hnerv_ft_microcodec" in text:
        return "pr101_microcodec"
    if "pr103" in text or "hnerv_lc_ac" in text:
        return "pr103_lc_ac"
    if payload[:1] == b"\xff" and len(payload) >= 4:
        dec_len = int.from_bytes(payload[1:4], "little")
        if 0 < dec_len < len(payload) - 4:
            return "ff_packed_brotli_hnerv"
    if len(payload) == PR101_DECODER_BLOB_LEN + PR101_LATENT_BLOB_LEN + 607:
        return "pr101_microcodec"
    pr103_min = (
        PR103_SCA_LEN
        + PR103_BR_LEN
        + PR103_HIST_LEN
        + PR103_MERGED_AC_LEN
        + PR103_LATENT_META_LEN
        + PR103_LO_LEN
        + PR103_HI_HIST_LEN
    )
    if len(payload) >= pr103_min:
        return "pr103_lc_ac"
    return "unknown_single_payload"


def profile_payload(kind: str, payload: bytes) -> list[SectionProfile]:
    sections: list[SectionProfile] = []
    if kind == "pr106_sidecar_wrapper":
        return profile_pr106_sidecar_wrapper(payload)
    if kind == PR106_HDM9_HLM3_MAGICLESS_FIXED_PAYLOAD_KIND:
        return profile_pr106_hdm9_hlm3_magicless_fixed_payload(payload)
    if kind == "pr101_microcodec":
        decoder_end = PR101_DECODER_BLOB_LEN
        latent_end = decoder_end + PR101_LATENT_BLOB_LEN
        return [
            section("decoder_compact_brotli_streams", payload, 0, decoder_end),
            section("latents_raw_lzma_delta_u8", payload, decoder_end, latent_end),
            section("sidecar_dim_delta_huffman_enum", payload, latent_end, len(payload)),
        ]
    if kind == "pr103_lc_ac":
        cursor = 0
        for name, size in [
            ("scales_fp16", PR103_SCA_LEN),
            ("non_ac_weights_brotli", PR103_BR_LEN),
            ("ac_histograms_brotli", PR103_HIST_LEN),
            ("merged_range_coded_weights_and_hi_latents", PR103_MERGED_AC_LEN),
            ("latent_min_scale_fp16", PR103_LATENT_META_LEN),
            ("latent_low_bytes_brotli", PR103_LO_LEN),
            ("latent_hi_histogram_brotli", PR103_HI_HIST_LEN),
        ]:
            sections.append(section(name, payload, cursor, cursor + size))
            cursor += size
        sections.append(section("sidecar_corrections_brotli", payload, cursor, len(payload)))
        return sections
    if kind == "ff_packed_brotli_hnerv":
        dec_len = int.from_bytes(payload[1:4], "little")
        return [
            section("packed_header_ff_len24", payload, 0, 4),
            section("decoder_packed_brotli", payload, 4, 4 + dec_len),
            section("latents_and_sidecar_brotli", payload, 4 + dec_len, len(payload)),
        ]
    return [section("opaque_single_payload", payload, 0, len(payload))]


def profile_pr106_hdm9_hlm3_magicless_fixed_payload(payload: bytes) -> list[SectionProfile]:
    expected = pr106_hdm9_hlm3_magicless_payload_bytes()
    if len(payload) != expected:
        raise ValueError(
            "PR106 HDM9/HLM3 magicless payload length mismatch: "
            f"got {len(payload)} bytes; expected {expected}"
        )
    decoder_tail_bytes = PR106_HDM9_HLM2_DECODER_PAYLOAD_BYTES - len(
        PR106_HDM9_HLM2_DECODER_MAGIC
    )
    latent_tail_bytes = PR106_HDM9_HLM3_LATENT_PAYLOAD_BYTES - len(
        PR106_HDM9_HLM3_LATENT_MAGIC
    )
    latent_start = decoder_tail_bytes
    sidecar_start = latent_start + latent_tail_bytes
    return [
        section(
            "inner_decoder_packed_brotli_hdm9_magicless_tail",
            payload,
            0,
            latent_start,
        ),
        section(
            "inner_latents_and_sidecar_brotli_hlm3_magicless_tail",
            payload,
            latent_start,
            sidecar_start,
        ),
        section(
            "sidecar_payload_pr101_fixed_meta_noop_rank_elided",
            payload,
            sidecar_start,
            len(payload),
        ),
    ]


def profile_pr106_sidecar_wrapper(payload: bytes) -> list[SectionProfile]:
    if len(payload) < 8:
        raise ValueError(f"PR106 sidecar wrapper too short: {len(payload)}")
    if payload[0] != PR106_SIDECAR_MAGIC:
        raise ValueError(f"PR106 sidecar wrapper magic mismatch: 0x{payload[0]:02x}")
    format_id = payload[1]
    pos = 2
    if format_id == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED:
        if len(payload) < pos + PR106_PR101_FIXED_META_PAYLOAD_BYTES:
            raise ValueError("PR106 implicit-len sidecar payload truncated")
        inner_start = pos
        inner_end = len(payload) - PR106_PR101_FIXED_META_PAYLOAD_BYTES
        header_name = "pr106_sidecar_header_fe_fmt"
        header_end = 2
    else:
        (inner_len,) = struct.unpack_from("<I", payload, pos)
        pos += 4
        inner_start = pos
        inner_end = inner_start + inner_len
        header_name = "pr106_sidecar_header_fe_fmt_len_u32"
        header_end = 6
    if inner_end > len(payload):
        raise ValueError(f"PR106 sidecar inner payload truncated: {inner_end} > {len(payload)}")
    inner = payload[inner_start:inner_end]
    if len(inner) < 4 or inner[0] != 0xFF:
        raise ValueError("PR106 sidecar inner payload is not ff-packed HNeRV")
    dec_len = int.from_bytes(inner[1:4], "little")
    inner_decoder_start = inner_start + 4
    inner_decoder_end = inner_decoder_start + dec_len
    if inner_decoder_end > inner_end:
        raise ValueError("PR106 sidecar inner decoder section exceeds inner payload")
    rows = [
        section(header_name, payload, 0, header_end),
        section("inner_packed_header_ff_len24", payload, inner_start, inner_start + 4),
        section("inner_decoder_packed_brotli", payload, inner_decoder_start, inner_decoder_end),
        section("inner_latents_and_sidecar_brotli", payload, inner_decoder_end, inner_end),
    ]

    pos = inner_end
    if format_id in (PR106_SIDECAR_FORMAT_BROTLI, PR106_SIDECAR_FORMAT_PR101_GRAMMAR):
        if pos + 2 > len(payload):
            raise ValueError("PR106 sidecar wrapper missing sidecar length")
        (sidecar_len,) = struct.unpack_from("<H", payload, pos)
        sidecar_len_start = pos
        pos += 2
        sidecar_start = pos
        sidecar_end = sidecar_start + sidecar_len
        if sidecar_end > len(payload):
            raise ValueError("PR106 sidecar wrapper sidecar payload truncated")
        rows.append(section("sidecar_len_u16", payload, sidecar_len_start, sidecar_start))
    elif format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        if len(payload) - pos < 5:
            raise ValueError("PR106 rank-elided sidecar truncated before framing meta")
        sidecar_start = pos
        sidecar_end = len(payload) - 5
    elif format_id in (
        PR106_SIDECAR_FORMAT_PR101_FIXED_META_RANK_ELIDED,
        PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED,
    ):
        sidecar_start = pos
        sidecar_end = len(payload)
    else:
        raise ValueError(f"unsupported PR106 sidecar format_id=0x{format_id:02x}")

    if format_id == PR106_SIDECAR_FORMAT_BROTLI:
        if sidecar_end != len(payload):
            raise ValueError("PR106 brotli sidecar wrapper has trailing bytes")
        rows.append(section("sidecar_payload_brotli_dim_delta", payload, sidecar_start, sidecar_end))
        return rows

    if format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        rows.append(section("sidecar_payload_pr101_ranked_no_op", payload, sidecar_start, sidecar_end))
        framing_start = sidecar_end
        framing_end = framing_start + 6
        if framing_end != len(payload):
            raise ValueError("PR106 PR101-grammar sidecar wrapper framing/trailing mismatch")
        rows.append(section("sidecar_framing_meta_pr101", payload, framing_start, framing_end))
        return rows

    if format_id == PR106_SIDECAR_FORMAT_PR101_RANK_ELIDED:
        rows.append(section("sidecar_payload_pr101_rank_elided", payload, sidecar_start, sidecar_end))
        rows.append(section("sidecar_framing_meta_pr101_rank_elided", payload, sidecar_end, len(payload)))
        return rows

    sidecar_name = (
        "sidecar_payload_pr101_implicit_len_fixed_meta_rank_elided"
        if format_id == PR106_SIDECAR_FORMAT_PR101_IMPLICIT_LEN_FIXED_META_RANK_ELIDED
        else "sidecar_payload_pr101_fixed_meta_rank_elided"
    )
    rows.append(section(sidecar_name, payload, sidecar_start, sidecar_end))
    return rows


def build_record(archive: Path, kind: str) -> dict[str, object]:
    member_name, payload, archive_bytes, archive_sha = extract_single_member(archive)
    resolved_kind = infer_profile_kind(kind, archive, member_name, payload)
    sections = profile_payload(resolved_kind, payload)
    return {
        "archive": str(archive),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "member_name": member_name,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "kind": resolved_kind,
        "zip_overhead_bytes": archive_bytes - len(payload),
        "sections": [asdict(item) for item in sections],
        "evidence_grade": "forensic_byte_profile",
        "score_claim": False,
    }


def render_markdown(record: dict[str, object]) -> str:
    lines = [
        f"# HNeRV Frontier Payload Profile: {record['archive']}",
        "",
        f"- archive_bytes: `{record['archive_bytes']}`",
        f"- archive_sha256: `{record['archive_sha256']}`",
        f"- zip_member: `{record['member_name']}`",
        f"- member_bytes: `{record['member_bytes']}`",
        f"- member_sha256: `{record['member_sha256']}`",
        f"- inferred_kind: `{record['kind']}`",
        f"- zip_overhead_bytes: `{record['zip_overhead_bytes']}`",
        "",
        "| section | start | end | bytes | entropy b/B | sha256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in record["sections"]:
        lines.append(
            "| {name} | {start} | {end} | {bytes} | {entropy_bits_per_byte:.6f} | `{sha256}` |".format(
                **item
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--kind", default="auto")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    args = parser.parse_args()

    records = [build_record(path, args.kind) for path in args.archives]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text("\n".join(render_markdown(record) for record in records))
    if not args.json_out and not args.md_out:
        print(json.dumps(records, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
