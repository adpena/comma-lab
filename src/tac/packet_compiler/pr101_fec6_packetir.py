# SPDX-License-Identifier: MIT
"""PR101/FEC6 FP11 PacketIR parser and identity proof.

This module proves byte-level custody for the PR101 frame-exploit selector
packet:

``FP11 | source_len:u32le | source_pr101_payload | selector_len:u16le | FEC6``

It is intentionally read-only and scorer-free.  A passing proof means only that
the archive member can be parsed into explicit PacketIR sections and re-emitted
byte-for-byte.  It is not a runtime, frame-parity, score, promotion, or dispatch
claim.
"""

from __future__ import annotations

import hashlib
import io
import re
import struct
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

FP11_MAGIC = b"FP11"
FEC6_MAGIC = b"FEC6"
PR101_FEC6_DEFAULT_MEMBER_NAME = "x"
PR101_FEC6_IDENTITY_PROOF_SCHEMA = "pr101_fec6_packetir_identity_proof_v1"
PR101_FEC6_PACKETIR_SECTION_HASH_DOMAIN = (
    "pr101_fec6_packetir_emitted_member_payload_section_bytes_v1"
)

FEC6_FIXED_K16_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)
FEC6_FIXED_K16_CODE_BITS = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE = {
    bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PR101FEC6PacketIRError(ValueError):
    """Raised when PR101/FEC6 PacketIR bytes are malformed."""


@dataclass(frozen=True)
class StoredZipMember:
    """Single stored ZIP member plus metadata required for byte-identical emit."""

    name: str
    payload: bytes
    date_time: tuple[int, int, int, int, int, int]
    external_attr: int
    create_system: int
    flag_bits: int
    comment: bytes
    extra: bytes
    archive_comment: bytes = b""


@dataclass(frozen=True)
class PacketIRSection:
    """One byte range in the PR101/FEC6 archive member payload."""

    name: str
    offset: int
    length: int
    payload: bytes
    value: int | str | None = None
    role: str = "packet_section"

    @property
    def end_offset(self) -> int:
        return self.offset + self.length

    def to_manifest(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "offset": self.offset,
            "end_offset": self.end_offset,
            "length": self.length,
            "sha256": sha256_hex(self.payload),
            "hash_domain": PR101_FEC6_PACKETIR_SECTION_HASH_DOMAIN,
        }
        if self.value is not None:
            row["value"] = self.value
        return row


@dataclass(frozen=True)
class PR101FEC6PacketIR:
    """Parsed FP11/FEC6 member payload with exact source bytes preserved."""

    fp11_magic: bytes
    source_len_u32le: bytes
    source_pr101_payload: bytes
    selector_len_u16le: bytes
    selector_fec6_payload: bytes
    selector_codes: tuple[int, ...]
    selector_code_bits_total: int

    @property
    def source_len(self) -> int:
        return int.from_bytes(self.source_len_u32le, "little")

    @property
    def selector_len(self) -> int:
        return int.from_bytes(self.selector_len_u16le, "little")

    @property
    def n_pairs(self) -> int:
        return int.from_bytes(self.selector_fec6_payload[4:6], "little")

    @property
    def selector_bitstream(self) -> bytes:
        return self.selector_fec6_payload[6:]

    @property
    def payload_bytes(self) -> int:
        return len(emit_pr101_fec6_packetir_member(self))

    @property
    def sections(self) -> tuple[PacketIRSection, ...]:
        source_start = 8
        source_end = source_start + len(self.source_pr101_payload)
        selector_len_offset = source_end
        selector_start = selector_len_offset + 2
        selector_end = selector_start + len(self.selector_fec6_payload)
        return (
            PacketIRSection("fp11_magic", 0, 4, self.fp11_magic, value="FP11"),
            PacketIRSection(
                "source_len_u32le",
                4,
                4,
                self.source_len_u32le,
                value=self.source_len,
                role="length_field",
            ),
            PacketIRSection(
                "source_pr101_payload",
                source_start,
                len(self.source_pr101_payload),
                self.source_pr101_payload,
                role="source_payload",
            ),
            PacketIRSection(
                "selector_len_u16le",
                selector_len_offset,
                2,
                self.selector_len_u16le,
                value=self.selector_len,
                role="length_field",
            ),
            PacketIRSection(
                "selector_fec6_payload",
                selector_start,
                len(self.selector_fec6_payload),
                self.selector_fec6_payload,
                role="selector_payload",
            ),
            PacketIRSection(
                "selector_fec6_magic",
                selector_start,
                4,
                self.selector_fec6_payload[:4],
                value="FEC6",
                role="selector_header",
            ),
            PacketIRSection(
                "selector_fec6_n_pairs_u16le",
                selector_start + 4,
                2,
                self.selector_fec6_payload[4:6],
                value=self.n_pairs,
                role="selector_header",
            ),
            PacketIRSection(
                "selector_fec6_fixed_huffman_bitstream",
                selector_start + 6,
                len(self.selector_bitstream),
                self.selector_bitstream,
                role="selector_bitstream",
            ),
            PacketIRSection(
                "packet_member_payload",
                0,
                selector_end,
                emit_pr101_fec6_packetir_member(self),
                role="member_payload",
            ),
        )


def sha256_hex(data: bytes) -> str:
    """Return the SHA-256 hex digest of in-memory bytes."""

    return hashlib.sha256(data).hexdigest()


def canonical_expected_sha256(expected_sha256: str | None) -> tuple[str | None, bool | None]:
    """Normalize an optional expected SHA-256 and report whether it is well formed."""

    if expected_sha256 is None:
        return None, None
    canonical = expected_sha256.strip().lower()
    return canonical, bool(_SHA256_RE.fullmatch(canonical))


def read_single_stored_fec6_member_archive(
    archive_bytes: bytes,
    *,
    expected_member_name: str = PR101_FEC6_DEFAULT_MEMBER_NAME,
) -> StoredZipMember:
    """Read the single stored PR101/FEC6 ZIP member and preserve ZIP metadata."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise PR101FEC6PacketIRError(f"expected one ZIP member; got {len(infos)}")
        info = infos[0]
        if info.filename != expected_member_name:
            raise PR101FEC6PacketIRError(
                f"expected ZIP member {expected_member_name!r}; got {info.filename!r}"
            )
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise PR101FEC6PacketIRError(f"unsafe ZIP member name: {info.filename!r}")
        if info.compress_type != zipfile.ZIP_STORED:
            raise PR101FEC6PacketIRError(
                f"expected stored ZIP member; got method={info.compress_type}"
            )
        return StoredZipMember(
            name=info.filename,
            payload=zf.read(info.filename),
            date_time=info.date_time,
            external_attr=info.external_attr,
            create_system=info.create_system,
            flag_bits=info.flag_bits,
            comment=info.comment,
            extra=info.extra,
            archive_comment=zf.comment,
        )


def emit_single_stored_member_archive(member: StoredZipMember) -> bytes:
    """Emit a single-member stored ZIP preserving source ZIP metadata."""

    out = io.BytesIO()
    info = zipfile.ZipInfo(member.name, date_time=member.date_time)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = member.external_attr
    info.create_system = member.create_system
    info.flag_bits = member.flag_bits
    info.comment = member.comment
    info.extra = member.extra
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(info, member.payload)
        zf.comment = member.archive_comment
    return out.getvalue()


def decode_fec6_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> tuple[tuple[int, ...], int]:
    """Decode FEC6 fixed-Huffman selector codes and return ``(codes, used_bits)``."""

    if n_pairs < 0:
        raise PR101FEC6PacketIRError(f"n_pairs must be non-negative; got {n_pairs}")
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    max_code_bits = max(len(bits) for bits in FEC6_FIXED_K16_CODE_BITS)
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise PR101FEC6PacketIRError("FEC6 selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC6_FIXED_K16_DECODE.get(prefix)
        if code is not None:
            codes.append(int(code))
            prefix = ""
            continue
        if len(prefix) > max_code_bits:
            raise PR101FEC6PacketIRError("FEC6 selector contains invalid prefix code")
    if prefix:
        raise PR101FEC6PacketIRError("FEC6 selector ended mid-symbol")
    expected_index_bytes = (bit_pos + 7) // 8
    if len(payload) != expected_index_bytes:
        raise PR101FEC6PacketIRError(
            "FEC6 selector has trailing zero bytes: "
            f"index_bytes={len(payload)} expected={expected_index_bytes}"
        )
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise PR101FEC6PacketIRError("FEC6 selector has non-zero padding bits")
    return tuple(codes), bit_pos


def parse_pr101_fec6_packetir_member(member_payload: bytes) -> PR101FEC6PacketIR:
    """Parse an FP11/FEC6 member payload into exact PacketIR sections."""

    if len(member_payload) < 10:
        raise PR101FEC6PacketIRError("PR101/FEC6 FP11 member truncated before header")
    fp11_magic = member_payload[:4]
    if fp11_magic != FP11_MAGIC:
        raise PR101FEC6PacketIRError(f"FP11 magic mismatch: {fp11_magic!r}")
    source_len_u32le = member_payload[4:8]
    source_len = struct.unpack("<I", source_len_u32le)[0]
    source_start = 8
    source_end = source_start + source_len
    if source_end > len(member_payload):
        raise PR101FEC6PacketIRError("FP11 member truncated in source_pr101_payload")
    selector_len_offset = source_end
    if selector_len_offset + 2 > len(member_payload):
        raise PR101FEC6PacketIRError("FP11 member truncated before selector_len_u16le")
    selector_len_u16le = member_payload[selector_len_offset : selector_len_offset + 2]
    selector_len = struct.unpack("<H", selector_len_u16le)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end > len(member_payload):
        raise PR101FEC6PacketIRError("FP11 member truncated in selector_fec6_payload")
    if selector_end != len(member_payload):
        raise PR101FEC6PacketIRError(
            f"FP11 member has trailing bytes: selector_end={selector_end} total={len(member_payload)}"
        )
    source_pr101_payload = member_payload[source_start:source_end]
    selector_fec6_payload = member_payload[selector_start:selector_end]
    if len(selector_fec6_payload) < 6:
        raise PR101FEC6PacketIRError("FEC6 selector payload truncated before header")
    if selector_fec6_payload[:4] != FEC6_MAGIC:
        raise PR101FEC6PacketIRError(
            f"FEC6 selector magic mismatch: {selector_fec6_payload[:4]!r}"
        )
    n_pairs = struct.unpack("<H", selector_fec6_payload[4:6])[0]
    selector_codes, used_bits = decode_fec6_fixed_huffman_codes(
        selector_fec6_payload[6:],
        n_pairs=n_pairs,
    )
    code_bits_total = sum(len(FEC6_FIXED_K16_CODE_BITS[code]) for code in selector_codes)
    if code_bits_total != used_bits:
        raise PR101FEC6PacketIRError(
            f"FEC6 selector bit accounting mismatch: code_bits={code_bits_total} used_bits={used_bits}"
        )
    return PR101FEC6PacketIR(
        fp11_magic=fp11_magic,
        source_len_u32le=source_len_u32le,
        source_pr101_payload=source_pr101_payload,
        selector_len_u16le=selector_len_u16le,
        selector_fec6_payload=selector_fec6_payload,
        selector_codes=selector_codes,
        selector_code_bits_total=code_bits_total,
    )


def emit_pr101_fec6_packetir_member(packet: PR101FEC6PacketIR) -> bytes:
    """Re-emit a parsed PR101/FEC6 member payload from exact PacketIR sections."""

    return (
        packet.fp11_magic
        + packet.source_len_u32le
        + packet.source_pr101_payload
        + packet.selector_len_u16le
        + packet.selector_fec6_payload
    )


def pr101_fec6_packetir_manifest(packet: PR101FEC6PacketIR) -> dict[str, Any]:
    """Return a non-promotable PacketIR manifest for parsed PR101/FEC6 bytes."""

    primary_sections = [
        row for row in packet.sections if row.name in {
            "fp11_magic",
            "source_len_u32le",
            "source_pr101_payload",
            "selector_len_u16le",
            "selector_fec6_payload",
        }
    ]
    accounted = sum(section.length for section in primary_sections)
    return {
        "schema": "pr101_fec6_packetir_manifest_v1",
        "format": "FP11/FEC6",
        "source_len": packet.source_len,
        "selector_len": packet.selector_len,
        "member_payload_bytes": packet.payload_bytes,
        "member_payload_sha256": sha256_hex(emit_pr101_fec6_packetir_member(packet)),
        "primary_section_names": [section.name for section in primary_sections],
        "all_member_bytes_accounted": accounted == packet.payload_bytes,
        "accounted_member_payload_bytes": accounted,
        "sections": [section.to_manifest() for section in packet.sections],
        "selector": {
            "magic": "FEC6",
            "n_pairs": packet.n_pairs,
            "palette_size": len(FEC6_FIXED_K16_MODE_IDS),
            "palette_mode_ids": list(FEC6_FIXED_K16_MODE_IDS),
            "selector_index_bytes": len(packet.selector_bitstream),
            "selector_code_bits_total": packet.selector_code_bits_total,
            "zero_padding_bits": len(packet.selector_bitstream) * 8
            - packet.selector_code_bits_total,
            "first_32_codes": list(packet.selector_codes[:32]),
        },
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def prove_pr101_fec6_packetir_identity(
    *,
    archive_path: Path,
    expected_member_name: str = PR101_FEC6_DEFAULT_MEMBER_NAME,
    expected_archive_sha256: str | None = None,
) -> dict[str, Any]:
    """Prove parse/emit identity for one PR101/FEC6 archive.

    The proof is deliberately local and read-only.  It neither imports the
    submission runtime nor evaluates frames.  A passing result is parser custody
    only and remains non-promotable by construction.
    """

    archive_path = Path(archive_path)
    archive_bytes = archive_path.read_bytes()
    archive_sha = sha256_hex(archive_bytes)
    expected_archive_sha, expected_archive_sha_well_formed = canonical_expected_sha256(
        expected_archive_sha256
    )
    member = read_single_stored_fec6_member_archive(
        archive_bytes,
        expected_member_name=expected_member_name,
    )
    packet = parse_pr101_fec6_packetir_member(member.payload)
    emitted_payload = emit_pr101_fec6_packetir_member(packet)
    emitted_archive = emit_single_stored_member_archive(replace(member, payload=emitted_payload))
    packet_manifest = pr101_fec6_packetir_manifest(packet)

    blockers: list[str] = []
    if expected_archive_sha_well_formed is False:
        blockers.append("expected_archive_sha256_malformed")
    if (
        expected_archive_sha_well_formed is True
        and expected_archive_sha is not None
        and archive_sha != expected_archive_sha
    ):
        blockers.append("expected_archive_sha256_mismatch")
    if emitted_payload != member.payload:
        blockers.append("packetir_member_parse_emit_not_identity")
    if emitted_archive != archive_bytes:
        blockers.append("single_member_zip_parse_emit_not_identity")
    if packet_manifest["all_member_bytes_accounted"] is not True:
        blockers.append("packetir_section_accounting_failed")

    member_reemit_identity = emitted_payload == member.payload
    archive_reemit_identity = emitted_archive == archive_bytes
    reemit_identity = member_reemit_identity and archive_reemit_identity
    identity_passed = not blockers
    sections = packet_manifest["sections"]
    return {
        "schema": PR101_FEC6_IDENTITY_PROOF_SCHEMA,
        "proof_scope": "packetir_parse_emit_identity_not_runtime_inflate_not_score",
        "archive_path": archive_path.as_posix(),
        "archive_sha256": archive_sha,
        "member_name": member.name,
        "member_bytes": len(member.payload),
        "member_sha256": sha256_hex(member.payload),
        "sections": sections,
        "archive": {
            "path": archive_path.as_posix(),
            "bytes": len(archive_bytes),
            "sha256": archive_sha,
            "zip_comment_bytes": len(member.archive_comment),
            "zip_comment_sha256": sha256_hex(member.archive_comment),
            "expected_sha256": expected_archive_sha,
            "expected_sha256_well_formed": expected_archive_sha_well_formed,
            "expected_sha256_matches": (
                None
                if expected_archive_sha is None or expected_archive_sha_well_formed is False
                else archive_sha == expected_archive_sha
            ),
        },
        "member": {
            "name": member.name,
            "expected_name": expected_member_name,
            "expected_name_matches": member.name == expected_member_name,
            "bytes": len(member.payload),
            "sha256": sha256_hex(member.payload),
        },
        "packet": packet_manifest,
        "emitted_member": {
            "bytes": len(emitted_payload),
            "sha256": sha256_hex(emitted_payload),
            "byte_identical_to_source_member": member_reemit_identity,
        },
        "emitted_archive": {
            "bytes": len(emitted_archive),
            "sha256": sha256_hex(emitted_archive),
            "byte_identical_to_source_archive": archive_reemit_identity,
        },
        "byte_exact_identity": {
            "source_archive_bytes": len(archive_bytes),
            "source_archive_sha256": archive_sha,
            "source_member_name": member.name,
            "source_member_bytes": len(member.payload),
            "source_member_sha256": sha256_hex(member.payload),
            "emitted_member_bytes": len(emitted_payload),
            "emitted_member_sha256": sha256_hex(emitted_payload),
            "emitted_archive_bytes": len(emitted_archive),
            "emitted_archive_sha256": sha256_hex(emitted_archive),
            "member_byte_identical": member_reemit_identity,
            "archive_byte_identical": archive_reemit_identity,
            "expected_archive_sha256": expected_archive_sha,
            "expected_archive_sha256_matches": (
                None
                if expected_archive_sha is None or expected_archive_sha_well_formed is False
                else archive_sha == expected_archive_sha
            ),
        },
        "reemit_identity": reemit_identity,
        "member_reemit_identity": member_reemit_identity,
        "archive_reemit_identity": archive_reemit_identity,
        "packet_ir_identity_passed": identity_passed,
        "blockers": blockers,
        "proof_not_score": True,
        "evidence_axis": "packet-ir-parser-local-no-score",
        "runtime_consumption_claim": False,
        "full_frame_inflate_output_parity_claim": False,
        "contest_axis_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "required_next_proof": (
            "runtime selector decode/apply proof, full-frame same-runtime parity, "
            "and exact auth eval with axis labels before score language"
        ),
    }


def render_pr101_fec6_packetir_identity_markdown(proof: dict[str, Any]) -> str:
    """Render a compact operator-facing markdown summary for an identity proof."""

    lines = [
        "# PR101/FEC6 PacketIR Identity Proof",
        "",
        f"- Schema: `{proof.get('schema')}`",
        f"- Archive: `{proof.get('archive_path')}`",
        f"- Archive SHA-256: `{proof.get('archive_sha256')}`",
        f"- Member: `{proof.get('member_name')}`",
        f"- Member bytes: `{proof.get('member_bytes')}`",
        f"- Member SHA-256: `{proof.get('member_sha256')}`",
        f"- Re-emit identity: `{proof.get('reemit_identity')}`",
        f"- PacketIR identity passed: `{proof.get('packet_ir_identity_passed')}`",
        f"- Score claim: `{proof.get('score_claim')}`",
        f"- Promotion eligible: `{proof.get('promotion_eligible')}`",
        f"- Ready for exact eval dispatch: `{proof.get('ready_for_exact_eval_dispatch')}`",
        "",
        "## Blockers",
        "",
    ]
    blockers = proof.get("blockers") or []
    if blockers:
        lines.extend(f"- `{blocker}`" for blocker in blockers)
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Sections",
            "",
            "| name | offset | end | length | sha256 |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for section in proof.get("sections", []):
        lines.append(
            "| {name} | {offset} | {end} | {length} | `{sha}` |".format(
                name=section["name"],
                offset=section["offset"],
                end=section["end_offset"],
                length=section["length"],
                sha=section["sha256"],
            )
        )
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "FEC6_FIXED_K16_CODE_BITS",
    "FEC6_FIXED_K16_MODE_IDS",
    "FEC6_MAGIC",
    "FP11_MAGIC",
    "PR101_FEC6_DEFAULT_MEMBER_NAME",
    "PR101_FEC6_IDENTITY_PROOF_SCHEMA",
    "PR101FEC6PacketIR",
    "PR101FEC6PacketIRError",
    "PacketIRSection",
    "StoredZipMember",
    "decode_fec6_fixed_huffman_codes",
    "emit_pr101_fec6_packetir_member",
    "emit_single_stored_member_archive",
    "parse_pr101_fec6_packetir_member",
    "pr101_fec6_packetir_manifest",
    "prove_pr101_fec6_packetir_identity",
    "read_single_stored_fec6_member_archive",
    "render_pr101_fec6_packetir_identity_markdown",
    "sha256_hex",
]
