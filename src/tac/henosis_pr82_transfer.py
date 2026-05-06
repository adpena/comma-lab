"""PR82/Henosis atom-transfer helpers.

The helpers in this module are intentionally scorer-free. They parse the
public PR82 compact bundle, expose deterministic per-pair activity summaries,
and build randmulti-row deltas suitable for transplant into the PR85 family.

REHYDRATED 2026-05-05 from .recovery_spec.json (preserved at
.recovery_quarantine_20260505T004735Z/src/tac/henosis_pr82_transfer.recovery_spec.json).
Spec source: bytecode disassembly of compiled .pyc; whitespace + inline comments lost.

PARTIAL REHYDRATION: error class, dataclasses, and trivial helpers
reconstructed exactly. The randmulti row decode/encode loops are deferred to
``NotImplementedError`` (pycdc cannot fully decompile their nested closures).
There are no live consumers in the current codebase.
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_QUARANTINE_SPEC = (
    ".recovery_quarantine_20260505T004735Z/src/tac/henosis_pr82_transfer.recovery_spec.json"
)


class HenosisPr82TransferError(ValueError):
    """Raised when a PR82 atom transfer input is malformed or unsupported."""

    pass


@dataclass(frozen=True)
class Pr82ReplayContract:
    """Public PR82 replay contract: bytes + sha256 + member layout."""

    archive_path: Path
    expected_bytes: int
    expected_sha256: str
    expected_members: tuple[str, ...]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pr82Bundle:
    """Parsed PR82 bundle: byte payloads keyed by member name."""

    contract: Pr82ReplayContract
    members: Mapping[str, bytes]
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pr82RandmultiGroup:
    """One PR82 randmulti group: row indices + sparse values."""

    group_id: int
    n_frames: int
    row_count: int
    nonzero_entries: int
    payload: bytes
    extra: Mapping[str, Any] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def brotli_decompress_segment(data: bytes, name: str) -> bytes:
    """Decompress a brotli-compressed PR82 segment, fail loud on errors."""
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - env dependent
        raise HenosisPr82TransferError(
            "PR82 transfer requires the brotli package"
        ) from exc
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise HenosisPr82TransferError(
            f"PR82 segment {name!r} is not brotli-decodable"
        ) from exc


def _read_vlq(data: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while cursor < len(data):
        byte = data[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            raise HenosisPr82TransferError(
                "PR82 VLQ stream is truncated or overlong"
            )
    raise HenosisPr82TransferError("PR82 VLQ stream is truncated or overlong")


def _write_vlq(value: int) -> bytes:
    if value < 0:
        raise HenosisPr82TransferError(
            f"PR82 VLQ requires non-negative value, got {value}"
        )
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _rehydration_failure(symbol: str) -> NotImplementedError:
    return NotImplementedError(
        f"rehydration incomplete: {symbol} contains nested closures pycdc "
        f"cannot decompile; original bytecode preserved in {_QUARANTINE_SPEC}"
    )


def parse_replay_contract(path: Path | str) -> Pr82ReplayContract:
    raise _rehydration_failure("parse_replay_contract")


def parse_pr82_bundle(
    archive_path: Path | str, contract: Pr82ReplayContract
) -> Pr82Bundle:
    raise _rehydration_failure("parse_pr82_bundle")


def _vlq_indices_values(
    data: bytes, cursor: int, count: int
) -> tuple[list[int], list[int], int]:
    raise _rehydration_failure("_vlq_indices_values")


def randmulti_semantic_label(
    group_id: int, height: int, width: int
) -> str:
    raise _rehydration_failure("randmulti_semantic_label")


def randmulti_group_qps1_nm2_compatible(group: Pr82RandmultiGroup) -> bool:
    raise _rehydration_failure("randmulti_group_qps1_nm2_compatible")


def _decode_randmulti_rows(
    data: bytes, cursor: int, n_rows: int
) -> tuple[list[Any], int]:
    raise _rehydration_failure("_decode_randmulti_rows")


def _encode_randmulti_rows(rows: Sequence[Any]) -> bytes:
    out = bytearray()
    for row in rows:
        values = list(row)
        if len(values) != 600:
            raise HenosisPr82TransferError(
                f"PR82 randmulti row must contain 600 frame choices, got {len(values)}"
            )
        active = [(idx, int(value)) for idx, value in enumerate(values) if int(value) != 0]
        if len(active) < 255:
            out.append(len(active))
        else:
            if len(active) > 0xFFFF:
                raise HenosisPr82TransferError(
                    f"PR82 randmulti sparse row has too many active entries: {len(active)}"
                )
            out.append(255)
            out.extend(struct.pack("<H", len(active)))
        prev = -1
        for idx, _value in active:
            out.extend(_write_vlq(idx - prev - 1))
            prev = idx
        for _idx, value in active:
            if value < 0 or value > 255:
                raise HenosisPr82TransferError(
                    f"PR82 randmulti choice must fit uint8, got {value}"
                )
            out.append(value)
    return bytes(out)


def decode_randmulti_groups(
    payload: bytes, schedule: Sequence[Any]
) -> list[Pr82RandmultiGroup]:
    raise _rehydration_failure("decode_randmulti_groups")


def encode_randmulti_qrm1(
    groups: Sequence[Pr82RandmultiGroup], *, quality: int = 11
) -> bytes:
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - env dependent
        raise HenosisPr82TransferError(
            "PR82 transfer requires the brotli package"
        ) from exc
    if quality < 0 or quality > 11:
        raise HenosisPr82TransferError(
            f"QRM1 Brotli quality must be in [0, 11], got {quality}"
        )
    if len(groups) > 0xFFFF:
        raise HenosisPr82TransferError(
            f"QRM1 supports at most 65535 groups, got {len(groups)}"
        )
    body = bytearray(b"QRM1")
    body.extend(struct.pack("<H", len(groups)))
    seen: set[int] = set()
    for group in groups:
        group_id = int(group.group_id)
        if group_id < 0 or group_id > 0xFFFF:
            raise HenosisPr82TransferError(
                f"QRM1 group_id must fit uint16, got {group.group_id!r}"
            )
        if group_id in seen:
            raise HenosisPr82TransferError(
                f"QRM1 duplicate randmulti group id: {group_id}"
            )
        seen.add(group_id)
        if group.n_frames != 600:
            raise HenosisPr82TransferError(
                f"QRM1 randmulti groups must target 600 frame pairs, got {group.n_frames}"
            )
        if group.row_count < 0:
            raise HenosisPr82TransferError(
                f"QRM1 row_count must be non-negative, got {group.row_count}"
            )
        body.extend(struct.pack("<H", group_id))
        body.extend(bytes(group.payload))
    return brotli.compress(bytes(body), quality=quality)


def decode_randmulti_qrm1(
    encoded: bytes, schedule: Sequence[Any]
) -> list[Pr82RandmultiGroup]:
    raise _rehydration_failure("decode_randmulti_qrm1")
