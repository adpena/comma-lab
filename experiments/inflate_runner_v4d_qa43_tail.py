# SPDX-License-Identifier: MIT
"""v4d receiver variant with a pair-local QA43 int4 warp-tail section.

This receiver extends, rather than relabels, the existing v4d two-plane warp.
It consumes ``state/qa43_warp_tail.r7`` and applies six signed int4 deltas to
the decoded v4d warp-pose vector before the existing two-plane, photometric,
rolling-shutter, uint8 realization.  It is therefore a ``warp-tail`` program
in the same actuator pool as PFS1/TT1.  It is not the distinct free-frame0
counterfactual.

Tail framing (all bytes consumed):

``<8s magic><B codec><75s pair_map><I r7_length><32s semantic_sha><r7_frame>``

The R7 lattice is ``[active_pair, 1, 6, 1]`` uint4 with signed value
``code - 8``.  An empty map has an empty R7 frame.  The six generic receiver
quanta are the already-registered v4d warp finite-difference steps; no
video-derived table is hidden in this free receiver.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Final

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ddm_r7_token_coder import (
    decode_token_codes,
    encode_token_codes,
    frame_accounting,
)
from inflate_runner_v4d import Decoder as V4DDecoder

PAIR_COUNT: Final = 600
PAIR_MAP_BYTES: Final = 75
COEFFICIENT_RANK: Final = 6
TAIL_MEMBER: Final = "state/qa43_warp_tail.r7"
TAIL_SCHEMA: Final = "ddm_qa43_v4d_warp_tail.v1"
TAIL_MAGIC: Final = b"Q43WTL01"
TAIL_HEADER: Final = struct.Struct("<8sB75sI32s")
CODEC_IDS: Final = {"smevr": 1, "brotli11": 2}
ID_CODECS: Final = {value: key for key, value in CODEC_IDS.items()}
V4D_WARP_QUANTA: Final = (0.08, 0.004, 0.004, 0.0015, 0.0015, 0.004)


class QA43TailReceiverError(ValueError):
    """The candidate tail or manifest did not close."""


def _active_pairs(pair_map: bytes) -> list[int]:
    if len(pair_map) != PAIR_MAP_BYTES:
        raise QA43TailReceiverError("QA43 pair map is not exactly 75 bytes")
    bits = np.unpackbits(
        np.frombuffer(pair_map, dtype=np.uint8),
        bitorder="big",
    )
    return [int(pair) for pair in np.flatnonzero(bits)]


def _semantic_sha(pair_map: bytes, frame: bytes) -> bytes:
    return hashlib.sha256(pair_map + frame).digest()


def encode_tail(
    coefficients: np.ndarray,
    pair_map: bytes,
    *,
    codec: str,
) -> bytes:
    """Encode the exact canonical tail consumed by :class:`Decoder`."""

    active = _active_pairs(pair_map)
    values = np.asarray(coefficients)
    if (
        values.dtype.kind != "i"
        or values.shape != (len(active), COEFFICIENT_RANK)
        or np.any(values < -7)
        or np.any(values > 7)
    ):
        raise QA43TailReceiverError("QA43 coefficients differ from signed int4 lattice")
    if codec not in CODEC_IDS:
        raise QA43TailReceiverError("QA43 tail codec differs")
    if active:
        codes = (values.astype(np.int16) + 8).astype(np.uint8)
        lattice = codes.reshape(len(active), 1, COEFFICIENT_RANK, 1)
        frame = encode_token_codes(lattice, levels=16, codec=codec)
    else:
        frame = b""
    return (
        TAIL_HEADER.pack(
            TAIL_MAGIC,
            CODEC_IDS[codec],
            pair_map,
            len(frame),
            _semantic_sha(pair_map, frame),
        )
        + frame
    )


def parse_tail(payload: bytes) -> tuple[str, bytes, np.ndarray]:
    """Parse, canonically re-encode, and expand one full n600 tail field."""

    if len(payload) < TAIL_HEADER.size:
        raise QA43TailReceiverError("QA43 tail is truncated")
    magic, codec_id, pair_map, frame_length, digest = TAIL_HEADER.unpack_from(payload)
    codec = ID_CODECS.get(codec_id)
    if magic != TAIL_MAGIC or codec is None:
        raise QA43TailReceiverError("QA43 tail magic or codec differs")
    if len(payload) != TAIL_HEADER.size + frame_length:
        raise QA43TailReceiverError("QA43 tail has truncation or inert trailers")
    frame = payload[TAIL_HEADER.size:]
    if _semantic_sha(pair_map, frame) != digest:
        raise QA43TailReceiverError("QA43 tail semantic SHA-256 differs")
    active = _active_pairs(pair_map)
    if active:
        accounting = frame_accounting(frame)
        if accounting.codec != codec:
            raise QA43TailReceiverError("QA43 outer and R7 codecs differ")
        codes = decode_token_codes(frame)
        if codes.shape != (len(active), 1, COEFFICIENT_RANK, 1):
            raise QA43TailReceiverError("QA43 R7 lattice shape differs")
        signed = codes[:, 0, :, 0].astype(np.int16) - 8
        if np.any(signed < -7) or np.any(signed > 7):
            raise QA43TailReceiverError("QA43 decoded coefficient exceeds signed int4")
        if encode_tail(signed, pair_map, codec=codec) != payload:
            raise QA43TailReceiverError("QA43 tail is not canonical")
    else:
        if frame:
            raise QA43TailReceiverError("QA43 empty map carries a nonempty R7 frame")
        signed = np.empty((0, COEFFICIENT_RANK), dtype=np.int16)

    expanded = np.zeros((PAIR_COUNT, COEFFICIENT_RANK), dtype=np.int16)
    if active:
        expanded[np.asarray(active, dtype=np.int64)] = signed
    return codec, pair_map, expanded


class Decoder(V4DDecoder):
    """Exact v4d decoder with the consumed QA43 warp-tail applied."""

    def __init__(self, archive_dir: Path) -> None:
        super().__init__(archive_dir)
        manifest = json.loads((archive_dir / "manifest.json").read_text())
        payload = (archive_dir / TAIL_MEMBER).read_bytes()
        self._install_tail(manifest, payload)

    @classmethod
    def from_parent_decoder(
        cls,
        parent: V4DDecoder,
        manifest: dict[str, object],
        payload: bytes,
    ) -> Decoder:
        """Apply the shipped tail path to an already-parsed exact v4d parent."""

        decoder = cls.__new__(cls)
        decoder.__dict__ = parent.__dict__.copy()
        decoder._install_tail(manifest, payload)
        return decoder

    def _install_tail(
        self,
        manifest: dict[str, object],
        payload: bytes,
    ) -> None:
        codec, pair_map, coefficients = parse_tail(payload)
        if self.n_pairs != PAIR_COUNT:
            raise QA43TailReceiverError("QA43 receiver requires exactly n600")
        if manifest.get("qa43_tail_schema") != TAIL_SCHEMA:
            raise QA43TailReceiverError("QA43 manifest schema differs")
        if manifest.get("qa43_tail_codec") != codec:
            raise QA43TailReceiverError("QA43 manifest codec differs")
        if manifest.get("qa43_tail_sha256") != hashlib.sha256(payload).hexdigest():
            raise QA43TailReceiverError("QA43 manifest tail SHA-256 differs")
        if manifest.get("qa43_pair_map_sha256") != hashlib.sha256(pair_map).hexdigest():
            raise QA43TailReceiverError("QA43 manifest pair-map SHA-256 differs")
        if tuple(manifest.get("qa43_warp_quanta", ())) != V4D_WARP_QUANTA:
            raise QA43TailReceiverError("QA43 manifest warp quanta differ")
        self.qa43_tail_codec = codec
        self.qa43_pair_map = pair_map
        self.qa43_coefficients = coefficients
        self.p_best = self.p_best.copy()
        self.p_best += coefficients.astype(np.float64) * np.asarray(
            V4D_WARP_QUANTA,
            dtype=np.float64,
        )


def main() -> None:
    archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
    names = [row.strip() for row in names_file.read_text().splitlines() if row.strip()]
    if names != ["0.mkv"]:
        raise SystemExit("this receiver serves exactly the custodied 0.mkv")
    decoder = Decoder(archive_dir)
    name = PurePosixPath(names[0])
    if name.is_absolute() or ".." in name.parts:
        raise SystemExit("unsafe video name")
    target = output_dir / name.with_suffix(".raw")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as sink:
        for pair in range(decoder.n_pairs):
            frame1 = decoder.f1(pair)
            frame0 = decoder.f0(pair, frame1)
            sink.write(frame0.tobytes())
            sink.write(frame1.tobytes())


if __name__ == "__main__":
    main()
