# SPDX-License-Identifier: MIT
"""Decode-only runtime for HPRC packets.

HPRC V0 is a receiver-contract scaffold, not a trained renderer. The runtime
streams a deterministic RGB video from the archive-contained decoder/latent/
selector/residual sections so archive-bound candidates can be custody-proven
before the MLX trainer starts emitting real weights.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from tac.substrates.hprc.archive import (
    HprcPacket,
    HprcSectionKind,
    parse_hprc_packet,
)
from tac.substrates.hprc.learned_receiver import (
    build_compact_preview_digest,
    decode_compact_receiver_packet,
    is_compact_receiver_packet,
    write_compact_receiver_raw,
)

CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
CONTEST_FRAME_COUNT = 1200
CONTEST_RAW_BYTES = CAMERA_H * CAMERA_W * CHANNELS * CONTEST_FRAME_COUNT

HPRC_PIXEL_DRIVING_SECTIONS: frozenset[HprcSectionKind] = frozenset(
    {
        HprcSectionKind.DECODER_QW,
        HprcSectionKind.LATENTS_RC,
        HprcSectionKind.CODEBOOKS_Q,
        HprcSectionKind.SELECTORS_RC,
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.RDO_PLAN,
        HprcSectionKind.RECEIVER_STATE,
    }
)
HPRC_METADATA_ONLY_SECTIONS: frozenset[HprcSectionKind] = frozenset(
    {HprcSectionKind.MANIFEST_JSON}
)


def _packet_from_bytes(packet_bytes: bytes | bytearray | memoryview) -> HprcPacket:
    return parse_hprc_packet(packet_bytes)


def hprc_pixel_driver_digest(packet: HprcPacket) -> bytes:
    """Return the digest of archive-contained sections that drive pixels."""

    h = hashlib.sha256()
    h.update(b"hprc_pixel_driver_v0")
    h.update(json.dumps(packet.config.as_dict(), sort_keys=True).encode("utf-8"))
    for section in packet.sections:
        if section.kind not in HPRC_PIXEL_DRIVING_SECTIONS:
            continue
        h.update(int(section.kind).to_bytes(2, "little"))
        h.update(len(section.payload).to_bytes(8, "little"))
        h.update(hashlib.sha256(section.payload).digest())
    return h.digest()


def _frame_rgb_from_digest(
    digest: bytes,
    frame_index: int,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Render one deterministic RGB frame from packet digest material."""

    seed = np.frombuffer(digest, dtype=np.uint8)
    y = np.arange(height, dtype=np.uint16)[:, None]
    x = np.arange(width, dtype=np.uint16)[None, :]
    base = (x + 3 * y + frame_index) & 0xFF
    frame = np.empty((height, width, CHANNELS), dtype=np.uint8)
    frame[:, :, 0] = (base + int(seed[0])) & 0xFF
    frame[:, :, 1] = ((2 * x + y + frame_index * 3 + int(seed[7])) & 0xFF).astype(
        np.uint8
    )
    frame[:, :, 2] = ((x + 2 * y + frame_index * 5 + int(seed[13])) & 0xFF).astype(
        np.uint8
    )
    return frame


def hprc_preview_digest(
    packet_bytes: bytes | bytearray | memoryview,
    *,
    frame_indices: Iterable[int] = (0, 1, 599, 1199),
    height: int = 32,
    width: int = 32,
) -> str:
    """Return a small deterministic digest of receiver pixels."""

    packet = _packet_from_bytes(packet_bytes)
    if is_compact_receiver_packet(packet):
        compact_frame_indices = tuple(
            idx for idx in frame_indices if 0 <= idx < packet.config.frames
        ) or (0,)
        return build_compact_preview_digest(
            decode_compact_receiver_packet(packet),
            frame_indices=compact_frame_indices,
            height=height,
            width=width,
        )
    pixel_digest = hprc_pixel_driver_digest(packet)
    h = hashlib.sha256()
    for frame_index in frame_indices:
        if frame_index < 0 or frame_index >= CONTEST_FRAME_COUNT:
            raise ValueError(f"frame_index outside contest video range: {frame_index}")
        h.update(
            _frame_rgb_from_digest(
                pixel_digest,
                frame_index,
                height=height,
                width=width,
            ).tobytes()
        )
    return h.hexdigest()


def inflate_one_video(
    archive_bytes: bytes | bytearray | memoryview,
    output_path: str | Path,
    *,
    device: str = "cpu",
) -> None:
    """Write the deterministic HPRC RGB byte stream to ``output_path``."""

    if device not in {"cpu", "auto"}:
        raise RuntimeError(f"HPRC V0 inflate supports cpu/auto only, got {device!r}")
    packet = _packet_from_bytes(archive_bytes)
    if is_compact_receiver_packet(packet):
        write_compact_receiver_raw(
            packet,
            output_path,
            height=CAMERA_H,
            width=CAMERA_W,
        )
        return
    pixel_digest = hprc_pixel_driver_digest(packet)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for frame_index in range(CONTEST_FRAME_COUNT):
            f.write(
                _frame_rgb_from_digest(
                    pixel_digest,
                    frame_index,
                    height=CAMERA_H,
                    width=CAMERA_W,
                ).tobytes()
            )


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CHANNELS",
    "CONTEST_FRAME_COUNT",
    "CONTEST_RAW_BYTES",
    "HPRC_METADATA_ONLY_SECTIONS",
    "HPRC_PIXEL_DRIVING_SECTIONS",
    "hprc_pixel_driver_digest",
    "hprc_preview_digest",
    "inflate_one_video",
]
