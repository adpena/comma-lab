"""Wire helpers for the DDM HP3 HPAC and resumable-token candidates.

``HP31`` is an exact, reversible storage chart over a canonical ``IHS1``
payload.  It replaces the 600x8 frame embedding with first-row plus modulo-256
temporal residuals.  ``HPT1`` frames independent chronological Range streams so
the real receiver can checkpoint after every 24-frame stage.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

HPAC_MAGIC = b"HP31"
TOKEN_MAGIC = b"HPT1"
MONOLITHIC_MAGIC = b"HPM1"
HPAC_VERSION = 1
TOKEN_VERSION = 1
FRAME_COUNT = 600
FRAME_DIM = 8
TOKEN_CHUNK_FRAMES = 24
_HPAC_HEADER = struct.Struct("<4sBBII")
_TOKEN_HEADER = struct.Struct("<4sBBHH")
_MONOLITHIC_HEADER = struct.Struct("<4sIQQ")


@dataclass(frozen=True)
class TokenChunkEnvelope:
    frame_count: int
    chunk_frames: int
    chunks: tuple[bytes, ...]


@dataclass(frozen=True)
class MonolithicCheckpointEnvelope:
    position: int
    state: tuple[int, int]
    range_payload: bytes


def factor_frame_embedding(ihs1: bytes, frame_offset: int) -> bytes:
    """Return an exact HP31 temporal-residual chart over ``ihs1``."""

    if not ihs1.startswith(b"IHS1"):
        raise ValueError("HP31 source is not an IHS1 payload")
    byte_count = FRAME_COUNT * FRAME_DIM
    if frame_offset < 4 or frame_offset + byte_count > len(ihs1):
        raise ValueError("HP31 frame-embedding offset is outside the payload")
    transformed = bytearray(ihs1)
    source = np.frombuffer(ihs1[frame_offset : frame_offset + byte_count], dtype=np.uint8).reshape(
        FRAME_COUNT, FRAME_DIM
    )
    residual = np.empty_like(source)
    residual[0] = source[0]
    residual[1:] = source[1:] - source[:-1]
    transformed[frame_offset : frame_offset + byte_count] = residual.tobytes()
    header = _HPAC_HEADER.pack(HPAC_MAGIC, HPAC_VERSION, 1, len(ihs1), frame_offset)
    return header + bytes(transformed)


def restore_ihs1(payload: bytes) -> bytes:
    """Restore canonical IHS1 bytes from IHS1 or the HP31 exact chart."""

    if payload.startswith(b"IHS1"):
        return payload
    if len(payload) < _HPAC_HEADER.size:
        raise ValueError("HP31 payload is truncated before its header")
    magic, version, transform, original_bytes, frame_offset = _HPAC_HEADER.unpack_from(payload)
    if magic != HPAC_MAGIC or version != HPAC_VERSION or transform != 1:
        raise ValueError("unsupported HP31 header")
    transformed = payload[_HPAC_HEADER.size :]
    if len(transformed) != original_bytes or not transformed.startswith(b"IHS1"):
        raise ValueError("HP31 transformed body is not the declared IHS1 payload")
    byte_count = FRAME_COUNT * FRAME_DIM
    if frame_offset < 4 or frame_offset + byte_count > len(transformed):
        raise ValueError("HP31 frame-embedding offset is outside the payload")
    restored = bytearray(transformed)
    residual = np.frombuffer(transformed[frame_offset : frame_offset + byte_count], dtype=np.uint8).reshape(
        FRAME_COUNT, FRAME_DIM
    )
    source = np.add.accumulate(residual, axis=0, dtype=np.uint8)
    restored[frame_offset : frame_offset + byte_count] = source.tobytes()
    output = bytes(restored)
    if not output.startswith(b"IHS1"):
        raise ValueError("HP31 inverse did not restore IHS1")
    return output


def pack_token_chunks(
    chunks: tuple[bytes, ...],
    *,
    frame_count: int = FRAME_COUNT,
    chunk_frames: int = TOKEN_CHUNK_FRAMES,
) -> bytes:
    """Frame retained independent Range streams without discarding any stream."""

    expected = (frame_count + chunk_frames - 1) // chunk_frames
    if len(chunks) != expected or not chunks:
        raise ValueError(f"HPT1 expected {expected} chunks, received {len(chunks)}")
    if any(not chunk or len(chunk) % 4 for chunk in chunks):
        raise ValueError("every HPT1 Range chunk must be nonempty uint32 words")
    header = _TOKEN_HEADER.pack(
        TOKEN_MAGIC,
        TOKEN_VERSION,
        0,
        frame_count,
        chunk_frames,
    )
    lengths = struct.pack(f"<{len(chunks)}I", *(len(chunk) for chunk in chunks))
    return header + lengths + b"".join(chunks)


def unpack_token_chunks(payload: bytes) -> TokenChunkEnvelope:
    """Parse HPT1 with exact consumption and fixed n600 stage geometry."""

    if len(payload) < _TOKEN_HEADER.size:
        raise ValueError("HPT1 payload is truncated before its header")
    magic, version, flags, frame_count, chunk_frames = _TOKEN_HEADER.unpack_from(payload)
    if magic != TOKEN_MAGIC or version != TOKEN_VERSION or flags != 0:
        raise ValueError("unsupported HPT1 header")
    if frame_count != FRAME_COUNT or not 1 <= chunk_frames <= 120:
        raise ValueError("HPT1 geometry differs from the declared n600/<=120-frame form")
    count = (frame_count + chunk_frames - 1) // chunk_frames
    table_end = _TOKEN_HEADER.size + 4 * count
    if len(payload) < table_end:
        raise ValueError("HPT1 payload is truncated in the length table")
    lengths = struct.unpack_from(f"<{count}I", payload, _TOKEN_HEADER.size)
    chunks: list[bytes] = []
    offset = table_end
    for length in lengths:
        if length == 0 or length % 4 or offset + length > len(payload):
            raise ValueError("HPT1 contains an invalid Range chunk length")
        chunks.append(payload[offset : offset + length])
        offset += length
    if offset != len(payload):
        raise ValueError("HPT1 payload has trailing bytes")
    return TokenChunkEnvelope(frame_count, chunk_frames, tuple(chunks))


def pack_monolithic_checkpoint(
    range_payload: bytes,
    *,
    position: int,
    state: tuple[int, int],
) -> bytes:
    """Prefix one counted 300-frame Range seek checkpoint to a retained stream."""

    if not range_payload or len(range_payload) % 4:
        raise ValueError("HPM1 Range payload must be nonempty uint32 words")
    if position < 0 or position > len(range_payload) // 4:
        raise ValueError("HPM1 checkpoint position is outside the Range payload")
    if len(state) != 2 or any(value < 0 or value >= 1 << 64 for value in state):
        raise ValueError("HPM1 checkpoint state is not two uint64 words")
    return _MONOLITHIC_HEADER.pack(MONOLITHIC_MAGIC, position, state[0], state[1]) + range_payload


def unpack_monolithic_checkpoint(payload: bytes) -> MonolithicCheckpointEnvelope:
    """Parse an HPM1 stream with exact fixed-header consumption."""

    if len(payload) <= _MONOLITHIC_HEADER.size:
        raise ValueError("HPM1 payload is truncated")
    magic, position, state_low, state_high = _MONOLITHIC_HEADER.unpack_from(payload)
    range_payload = payload[_MONOLITHIC_HEADER.size :]
    if magic != MONOLITHIC_MAGIC or len(range_payload) % 4:
        raise ValueError("invalid HPM1 payload")
    if position > len(range_payload) // 4:
        raise ValueError("HPM1 checkpoint position is outside the Range payload")
    return MonolithicCheckpointEnvelope(position, (state_low, state_high), range_payload)
