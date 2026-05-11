"""Custom binary container format — RESEARCH-ONLY.

Status: ``score_claim=false``, ``promotion_eligible=false``,
``ready_for_exact_eval_dispatch=false``. This module designs and prototypes
a non-ZIP container format optimised for the contest's specific archive
pattern (typically one member, no per-member CRC redundancy, no central
directory). It is NOT wired into the contest runtime; the contest
``inflate.sh`` continues to expect a standard ``archive.zip``.

The Rust sister implementation
(``runtime-rs/crates/tac-packet-compiler/src/custom_binary_container/``)
produces byte-identical output for the same input. The two implementations
form a parity pair — any encoder change must land in BOTH simultaneously
and a golden vector regenerated.

Wire format::

    magic              : 4 bytes  = b"TACP"
    version            : u8       = 0x01
    flags              : u8       = bit 0: 0 = uncompressed body
    n_records          : u16 LE
    per record:
      name_len         : u16 LE
      name             : name_len bytes (UTF-8)
      body_len         : u32 LE
      body             : body_len bytes
    trailer:
      container_sha256 : 32 bytes (SHA-256 over everything above)

For the contest's one-record pattern this saves ~64 bytes vs ZIP overhead.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from typing import List, Sequence

TACP_MAGIC = b"TACP"
TACP_VERSION = 0x01


@dataclass(frozen=True)
class TacpRecord:
    """One named record inside a TACP container.

    Attributes:
        name: UTF-8 record name. Length must fit in u16; capped at 255 bytes
            (mirrors a ZIP-friendly archive-name cap).
        body: Raw record body bytes.
    """

    name: str
    body: bytes


def encode_container(records: Sequence[TacpRecord]) -> bytes:
    """Encode a list of records into a TACP container.

    Returns the serialised TACP bytes. Raises ``ValueError`` on impossible
    inputs (name too long, body too long, record count overflow).
    """
    if len(records) > 0xFFFF:
        raise ValueError(
            f"TACP supports at most {0xFFFF} records; got {len(records)}"
        )
    body_parts: List[bytes] = [
        TACP_MAGIC,
        struct.pack("<BBH", TACP_VERSION, 0, len(records)),
    ]
    for r in records:
        name_bytes = r.name.encode("utf-8")
        if len(name_bytes) > 255:
            raise ValueError(
                f"record name length {len(name_bytes)} exceeds the 255-byte "
                f"ZIP-friendly cap (name={r.name!r})"
            )
        if len(r.body) > 0xFFFF_FFFF:
            raise ValueError(
                f"record body length {len(r.body)} exceeds u32::MAX"
            )
        body_parts.append(struct.pack("<H", len(name_bytes)))
        body_parts.append(name_bytes)
        body_parts.append(struct.pack("<I", len(r.body)))
        body_parts.append(r.body)
    body = b"".join(body_parts)
    trailer = hashlib.sha256(body).digest()
    return body + trailer


def decode_container(blob: bytes) -> List[TacpRecord]:
    """Decode a TACP container blob back into the original record list.

    Raises ``ValueError`` on any corruption (bad magic, bad version,
    truncated record, trailer SHA-256 mismatch).
    """
    if len(blob) < 4 + 1 + 1 + 2 + 32:
        raise ValueError(f"TACP blob too short: {len(blob)} bytes (minimum 40)")
    if blob[:4] != TACP_MAGIC:
        raise ValueError(f"bad TACP magic: {blob[:4]!r}")
    version = blob[4]
    if version != TACP_VERSION:
        raise ValueError(
            f"unsupported TACP version {version} (we only support {TACP_VERSION})"
        )
    flags = blob[5]
    if flags != 0:
        raise ValueError(
            f"unsupported TACP flags {flags:#04x} (sub-payload mode reserved)"
        )
    trailer_start = len(blob) - 32
    body = blob[:trailer_start]
    expected = blob[trailer_start:]
    actual = hashlib.sha256(body).digest()
    if actual != expected:
        raise ValueError("TACP trailer SHA-256 mismatch (container corrupt)")
    (n_records,) = struct.unpack("<H", blob[6:8])
    records: List[TacpRecord] = []
    pos = 8
    for r_idx in range(n_records):
        if pos + 2 > trailer_start:
            raise ValueError(f"TACP truncated reading name_len at record {r_idx}")
        (name_len,) = struct.unpack("<H", blob[pos : pos + 2])
        pos += 2
        if pos + name_len > trailer_start:
            raise ValueError(f"TACP truncated reading name at record {r_idx}")
        try:
            name = blob[pos : pos + name_len].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"TACP record {r_idx} name is not valid UTF-8: {exc}"
            ) from exc
        pos += name_len
        if pos + 4 > trailer_start:
            raise ValueError(f"TACP truncated reading body_len at record {r_idx}")
        (body_len,) = struct.unpack("<I", blob[pos : pos + 4])
        pos += 4
        if pos + body_len > trailer_start:
            raise ValueError(f"TACP truncated reading body at record {r_idx}")
        records.append(TacpRecord(name=name, body=bytes(blob[pos : pos + body_len])))
        pos += body_len
    if pos != trailer_start:
        raise ValueError(
            f"TACP has {trailer_start - pos} bytes of slack after records (corrupt)"
        )
    return records


def section_savings_vs_zip(record_names: Sequence[str]) -> tuple[int, int, int]:
    """Byte-savings model for TACP vs ZIP framing.

    Returns ``(zip_overhead, tacp_overhead, savings)`` in bytes.

    The model counts only the **container framing overhead**; record body
    bytes are identical between ZIP and TACP at our compression layer.

    ZIP per-record cost (PKWARE APPNOTE 4.5)::

      30 bytes local file header
      + name_len bytes
      + 46 bytes central directory record
      + name_len bytes (duplicated in CDR)

    EOCD adds 22 bytes once per archive.

    TACP per-record cost::

      2 bytes name_len + name_len bytes + 4 bytes body_len

    Header is 8 bytes (magic + version + flags + n_records); trailer is
    32 bytes (SHA-256). Total = 40 + sum(6 + name_len) + 0.
    """
    n = len(record_names)
    total_name_len = sum(len(s) for s in record_names)
    zip_overhead = 22 + 76 * n + 2 * total_name_len
    tacp_overhead = 40 + 6 * n + total_name_len
    savings = zip_overhead - tacp_overhead
    return zip_overhead, tacp_overhead, savings
