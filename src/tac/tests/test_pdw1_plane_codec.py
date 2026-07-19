"""Tests for the strict PDW1P plane codec (labels+fills → scorer plane)."""

from __future__ import annotations

import numpy as np
import pytest

from tac.codec.pdw1_plane_codec import (
    MAGIC,
    Pdw1PlaneCodecError,
    Pdw1PlanePayload,
    decode_pdw1p,
    encode_pdw1p,
    expand_scorer_plane,
)


def _payload(n_pairs: int = 2, height: int = 8, width: int = 12, k: int = 5):
    rng = np.random.default_rng(11)
    labels = rng.integers(0, k, size=(n_pairs, height, width), dtype=np.uint8)
    fills = rng.integers(0, 256, size=(n_pairs, k, 3), dtype=np.uint8)
    return Pdw1PlanePayload(labels=labels, fills=fills)


def test_roundtrip_byte_identity() -> None:
    payload = _payload()
    blob = encode_pdw1p(payload)
    decoded = decode_pdw1p(blob)
    assert np.array_equal(decoded.labels, payload.labels)
    assert np.array_equal(decoded.fills, payload.fills)
    assert encode_pdw1p(decoded) == blob


def test_expand_scorer_plane_paints_fills() -> None:
    payload = _payload(n_pairs=1)
    plane = expand_scorer_plane(payload, 0)
    assert plane.shape == (8, 12, 3)
    assert plane.dtype == np.uint8
    labels = np.asarray(payload.labels)[0]
    fills = np.asarray(payload.fills)[0]
    for y in range(8):
        for x in range(12):
            assert np.array_equal(plane[y, x], fills[labels[y, x]])


def test_expand_refuses_out_of_range_pair() -> None:
    payload = _payload(n_pairs=1)
    with pytest.raises(Pdw1PlaneCodecError, match="out of range"):
        expand_scorer_plane(payload, 1)


def test_decode_refuses_bad_magic_version_and_trailer() -> None:
    blob = encode_pdw1p(_payload())
    with pytest.raises(Pdw1PlaneCodecError, match="magic"):
        decode_pdw1p(b"XXXXXXXX" + blob[len(MAGIC) :])
    bad_version = bytearray(blob)
    bad_version[8] = 99
    with pytest.raises(Pdw1PlaneCodecError, match="version"):
        decode_pdw1p(bytes(bad_version))
    with pytest.raises(Pdw1PlaneCodecError, match="trailing"):
        decode_pdw1p(blob + b"\x00")


def test_decode_refuses_truncation() -> None:
    blob = encode_pdw1p(_payload())
    with pytest.raises(Pdw1PlaneCodecError):
        decode_pdw1p(blob[:-3])
    with pytest.raises(Pdw1PlaneCodecError, match="header"):
        decode_pdw1p(blob[:4])


def test_decode_refuses_label_values_out_of_class_range() -> None:
    labels = np.full((1, 4, 4), 3, dtype=np.uint8)
    fills = np.zeros((1, 4, 3), dtype=np.uint8)
    blob = encode_pdw1p(Pdw1PlanePayload(labels=labels, fills=fills))
    # Shrink the declared class count below the max label: header K at byte 16
    # (layout <8sHHHHB: magic 0..7, version 8..9, n_pairs 10..11, height 12..13,
    # width 14..15, K 16).
    tampered = bytearray(blob)
    assert tampered[16] == 4
    tampered[16] = 3
    # K participates in fills length so the parse fails structurally either
    # on label range or on framing — both are refusals.
    with pytest.raises(Pdw1PlaneCodecError):
        decode_pdw1p(bytes(tampered))


def test_payload_constructor_refuses_bad_shapes() -> None:
    with pytest.raises(Pdw1PlaneCodecError, match="uint8"):
        Pdw1PlanePayload(
            labels=np.zeros((1, 4, 4), dtype=np.int64),
            fills=np.zeros((1, 5, 3), dtype=np.uint8),
        )
    with pytest.raises(Pdw1PlaneCodecError, match="pair count"):
        Pdw1PlanePayload(
            labels=np.zeros((2, 4, 4), dtype=np.uint8),
            fills=np.zeros((1, 5, 3), dtype=np.uint8),
        )
    with pytest.raises(Pdw1PlaneCodecError, match="n_classes"):
        Pdw1PlanePayload(
            labels=np.zeros((1, 4, 4), dtype=np.uint8),
            fills=np.zeros((1, 1, 3), dtype=np.uint8),
        )
    bad = np.zeros((1, 4, 4), dtype=np.uint8)
    bad[0, 0, 0] = 7
    with pytest.raises(Pdw1PlaneCodecError, match=">= n_classes"):
        Pdw1PlanePayload(labels=bad, fills=np.zeros((1, 5, 3), dtype=np.uint8))
