"""PR106 sidecar PacketIR parser and identity emitter.

This module owns the byte-level wrapper used by PR106 latent-sidecar variants:

``magic(0xFE) | format_id | pr106_len(u32le) | pr106_bytes | sidecar...``

It is intentionally scorer-free and torch-free. The compiler layer can use it
to prove that a transform consumes and re-emits the same bytes before a new
grammar or sidecar mutation is allowed near exact eval.
"""

from __future__ import annotations

import hashlib
import io
import struct
import zipfile
from dataclasses import dataclass

PR106_SIDECAR_MAGIC = 0xFE
PR106_SIDECAR_FORMAT_BROTLI = 0x01
PR106_SIDECAR_FORMAT_PR101_GRAMMAR = 0x02
PR106_SUPPORTED_SIDECAR_FORMATS = (
    PR106_SIDECAR_FORMAT_BROTLI,
    PR106_SIDECAR_FORMAT_PR101_GRAMMAR,
)
PR106_DEFAULT_MEMBER_NAME = "0.bin"


@dataclass(frozen=True)
class StoredZipMember:
    """Single stored ZIP member plus enough metadata for byte-identical emit."""

    name: str
    payload: bytes
    date_time: tuple[int, int, int, int, int, int]
    external_attr: int
    create_system: int
    flag_bits: int
    comment: bytes
    extra: bytes


@dataclass(frozen=True)
class PR106SidecarPacket:
    """Typed PR106 sidecar wrapper.

    ``framing_meta`` is ``None`` for format ``0x01``. For format ``0x02`` it is
    the six-byte PR101 grammar footer:
    ``noop_count(u16le) | dim_bytes(u16le) | rank_bytes(u8) | noop_rank_bytes(u8)``.
    """

    format_id: int
    pr106_bytes: bytes
    sidecar_payload: bytes
    framing_meta: bytes | None = None

    @property
    def sidecar_kind(self) -> str:
        if self.format_id == PR106_SIDECAR_FORMAT_BROTLI:
            return "brotli_dim_delta"
        if self.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
            return "pr101_ranked_no_op"
        return f"unknown_0x{self.format_id:02x}"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_single_stored_member_archive(
    archive_bytes: bytes,
    *,
    expected_member_name: str = PR106_DEFAULT_MEMBER_NAME,
) -> StoredZipMember:
    """Return the only stored member from a PR106-style archive ZIP."""

    with zipfile.ZipFile(io.BytesIO(archive_bytes), "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"expected one ZIP member; got {len(infos)}")
        info = infos[0]
        if info.filename != expected_member_name:
            raise ValueError(
                f"expected ZIP member {expected_member_name!r}; got {info.filename!r}"
            )
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"expected stored ZIP member; got method={info.compress_type}")
        return StoredZipMember(
            name=info.filename,
            payload=zf.read(info.filename),
            date_time=info.date_time,
            external_attr=info.external_attr,
            create_system=info.create_system,
            flag_bits=info.flag_bits,
            comment=info.comment,
            extra=info.extra,
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
    return out.getvalue()


def parse_pr106_sidecar_packet(
    payload: bytes,
    *,
    supported_formats: tuple[int, ...] = PR106_SUPPORTED_SIDECAR_FORMATS,
) -> PR106SidecarPacket:
    """Parse PR106 sidecar ``0.bin`` bytes into typed sections."""

    if len(payload) < 8:
        raise ValueError(f"PR106 sidecar payload too short: {len(payload)} bytes")
    if payload[0] != PR106_SIDECAR_MAGIC:
        raise ValueError(
            f"sidecar magic mismatch: got 0x{payload[0]:02X}, "
            f"expected 0x{PR106_SIDECAR_MAGIC:02X}"
        )
    format_id = payload[1]
    if format_id not in supported_formats:
        expected = ", ".join(f"0x{value:02X}" for value in supported_formats)
        raise ValueError(
            f"unsupported PR106 sidecar format_id=0x{format_id:02X}; expected {expected}"
        )
    pos = 2
    (pr106_len,) = struct.unpack_from("<I", payload, pos)
    pos += 4
    end_pr106 = pos + pr106_len
    if end_pr106 > len(payload):
        raise ValueError(
            f"PR106 inner payload truncated: pr106_len={pr106_len} total={len(payload)}"
        )
    pr106_bytes = payload[pos:end_pr106]
    pos = end_pr106

    if format_id == PR106_SIDECAR_FORMAT_BROTLI:
        if pos + 2 > len(payload):
            raise ValueError("brotli sidecar truncated before sidecar_len")
        (sidecar_len,) = struct.unpack_from("<H", payload, pos)
        pos += 2
        end_sidecar = pos + sidecar_len
        if end_sidecar > len(payload):
            raise ValueError(
                f"brotli sidecar truncated: sidecar_len={sidecar_len} total={len(payload)}"
            )
        sidecar = payload[pos:end_sidecar]
        if end_sidecar != len(payload):
            raise ValueError(
                f"brotli sidecar trailing bytes: pos={end_sidecar} total={len(payload)}"
            )
        return PR106SidecarPacket(
            format_id=format_id,
            pr106_bytes=pr106_bytes,
            sidecar_payload=sidecar,
            framing_meta=None,
        )

    if pos + 2 > len(payload):
        raise ValueError("PR101 grammar sidecar truncated before payload_len")
    (sidecar_len,) = struct.unpack_from("<H", payload, pos)
    pos += 2
    end_sidecar = pos + sidecar_len
    if end_sidecar > len(payload):
        raise ValueError(
            f"PR101 grammar sidecar truncated: payload_len={sidecar_len} total={len(payload)}"
        )
    sidecar = payload[pos:end_sidecar]
    pos = end_sidecar
    if pos + 6 > len(payload):
        raise ValueError("PR101 grammar sidecar truncated before framing_meta")
    framing_meta = payload[pos : pos + 6]
    pos += 6
    if pos != len(payload):
        raise ValueError(
            f"PR101 grammar sidecar trailing bytes: pos={pos} total={len(payload)}"
        )
    return PR106SidecarPacket(
        format_id=format_id,
        pr106_bytes=pr106_bytes,
        sidecar_payload=sidecar,
        framing_meta=framing_meta,
    )


def emit_pr106_sidecar_packet(packet: PR106SidecarPacket) -> bytes:
    """Emit ``0.bin`` bytes from a parsed PR106 sidecar packet."""

    if packet.format_id not in PR106_SUPPORTED_SIDECAR_FORMATS:
        raise ValueError(f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}")
    if len(packet.pr106_bytes) > 0xFFFF_FFFF:
        raise ValueError("PR106 inner payload too large for u32 length field")
    if len(packet.sidecar_payload) > 0xFFFF:
        raise ValueError("sidecar payload too large for u16 length field")
    if packet.format_id == PR106_SIDECAR_FORMAT_BROTLI and packet.framing_meta is not None:
        raise ValueError("format_id=0x01 must not carry framing_meta")
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x02 requires framing_meta")
        if len(packet.framing_meta) != 6:
            raise ValueError(
                f"format_id=0x02 framing_meta must be 6 bytes; got {len(packet.framing_meta)}"
            )

    out = io.BytesIO()
    out.write(bytes([PR106_SIDECAR_MAGIC, packet.format_id]))
    out.write(struct.pack("<I", len(packet.pr106_bytes)))
    out.write(packet.pr106_bytes)
    out.write(struct.pack("<H", len(packet.sidecar_payload)))
    out.write(packet.sidecar_payload)
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        assert packet.framing_meta is not None
        out.write(packet.framing_meta)
    return out.getvalue()


def _consumed_section_row(
    name: str,
    *,
    offset: int,
    payload: bytes,
    score_affecting: bool,
) -> dict[str, object]:
    return {
        "name": name,
        "offset": offset,
        "bytes": len(payload),
        "end_offset": offset + len(payload),
        "sha256": sha256_hex(payload),
        "score_affecting": score_affecting,
    }


def pr106_sidecar_consumed_byte_proof(
    packet: PR106SidecarPacket,
) -> dict[str, object]:
    """Return parser-level proof that every emitted payload byte is accounted for.

    This is intentionally narrower than a runtime-inflate proof. It proves the
    PacketIR grammar consumes the emitted ``0.bin`` payload without gaps or
    trailing bytes; exact runtime consumption still needs an inflate mutation
    smoke or same-runtime eval artifact.
    """

    if packet.format_id not in PR106_SUPPORTED_SIDECAR_FORMATS:
        raise ValueError(
            f"unsupported PR106 sidecar format_id=0x{packet.format_id:02X}"
        )

    sections: list[dict[str, object]] = []
    offset = 0

    def add(name: str, payload: bytes, *, score_affecting: bool) -> None:
        nonlocal offset
        sections.append(
            _consumed_section_row(
                name,
                offset=offset,
                payload=payload,
                score_affecting=score_affecting,
            )
        )
        offset += len(payload)

    add("magic", bytes([PR106_SIDECAR_MAGIC]), score_affecting=False)
    add("format_id", bytes([packet.format_id]), score_affecting=False)
    add(
        "pr106_len_le_u32",
        struct.pack("<I", len(packet.pr106_bytes)),
        score_affecting=False,
    )
    add("pr106_payload", packet.pr106_bytes, score_affecting=True)
    add(
        "sidecar_len_le_u16",
        struct.pack("<H", len(packet.sidecar_payload)),
        score_affecting=False,
    )
    add("sidecar_payload", packet.sidecar_payload, score_affecting=True)
    if packet.format_id == PR106_SIDECAR_FORMAT_PR101_GRAMMAR:
        if packet.framing_meta is None:
            raise ValueError("format_id=0x02 requires framing_meta")
        add("framing_meta", packet.framing_meta, score_affecting=True)

    emitted = emit_pr106_sidecar_packet(packet)
    accounted = sum(int(row["bytes"]) for row in sections)
    cursor = 0
    gaps: list[dict[str, object]] = []
    for row in sections:
        row_offset = int(row["offset"])
        if row_offset != cursor:
            gaps.append({"expected_offset": cursor, "observed_offset": row_offset})
        cursor = int(row["end_offset"])

    return {
        "schema": "pr106_sidecar_packet_ir_consumed_byte_proof_v1",
        "proof_scope": "packet_ir_parser_accounting_not_runtime_inflate_consumption",
        "runtime_consumption_claim": False,
        "emitted_payload_bytes": len(emitted),
        "emitted_payload_sha256": sha256_hex(emitted),
        "accounted_payload_bytes": accounted,
        "all_payload_bytes_accounted": accounted == len(emitted) and not gaps,
        "unconsumed_trailing_bytes": max(0, len(emitted) - accounted),
        "section_gaps": gaps,
        "score_affecting_section_names": [
            str(row["name"]) for row in sections if bool(row["score_affecting"])
        ],
        "sections": sections,
    }


def pr106_sidecar_manifest(
    packet: PR106SidecarPacket,
    *,
    archive_sha256: str | None = None,
) -> dict[str, object]:
    """Return a small machine-checkable identity manifest for review artifacts."""

    emitted = emit_pr106_sidecar_packet(packet)
    manifest: dict[str, object] = {
        "schema": "pr106_sidecar_packet_ir_manifest_v1",
        "format_id": f"0x{packet.format_id:02X}",
        "sidecar_kind": packet.sidecar_kind,
        "pr106_bytes": len(packet.pr106_bytes),
        "pr106_sha256": sha256_hex(packet.pr106_bytes),
        "sidecar_bytes": len(packet.sidecar_payload),
        "sidecar_sha256": sha256_hex(packet.sidecar_payload),
        "framing_meta_bytes": 0 if packet.framing_meta is None else len(packet.framing_meta),
        "framing_meta_sha256": None
        if packet.framing_meta is None
        else sha256_hex(packet.framing_meta),
        "emitted_payload_bytes": len(emitted),
        "emitted_payload_sha256": sha256_hex(emitted),
        "packet_ir_consumed_byte_proof": pr106_sidecar_consumed_byte_proof(packet),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    if archive_sha256 is not None:
        manifest["archive_sha256"] = archive_sha256
    return manifest
