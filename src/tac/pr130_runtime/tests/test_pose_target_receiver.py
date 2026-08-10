from __future__ import annotations

import hashlib
import importlib
import lzma
import os
import struct
import subprocess
import sys
from pathlib import Path

import brotli
import numpy as np
import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "fx1_runtime_tree"
if str(RUNTIME) not in sys.path:
    sys.path.insert(0, str(RUNTIME))

carrier_codec = importlib.import_module("carrier_codec")
receiver = importlib.import_module("pose_target_receiver")


def _target_packet(codes: np.ndarray) -> bytes:
    chunks = [
        receiver.PZ2_HEADER.pack(
            receiver.PZ2_MAGIC,
            1,
            receiver.PZ2_DIRECT,
            receiver.N,
            receiver.TARGET_DIMS,
        )
    ]
    for dimension in range(receiver.TARGET_DIMS):
        raw = np.asarray(codes[:, dimension], dtype="<u2").tobytes()
        compressed = brotli.compress(raw, quality=11)
        chunks.append(receiver.PZ2_STREAM_HEADER.pack(8, -1.0, 0.125, len(compressed)))
        chunks.append(compressed)
    return b"".join(chunks)


def _encoded_coefficients(absolute: np.ndarray) -> np.ndarray:
    unsigned = np.asarray(absolute, dtype=np.int64) & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_unsigned = (unsigned - previous) & 0xFFF
    delta = np.where(delta_unsigned >= 0x800, delta_unsigned - 0x1000, delta_unsigned)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def _fixture() -> tuple[
    bytes,
    bytes,
    receiver.Predictor,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(20260809)
    target_codes = rng.integers(
        0, 256, size=(receiver.N, receiver.TARGET_DIMS), dtype=np.uint16
    )
    target_packet = _target_packet(target_codes)
    basis_scales = np.linspace(0.01, 0.12, receiver.CARRIER_DIMS, dtype=np.float32)
    basis_codes = rng.integers(
        -15,
        16,
        size=receiver.BASIS_COUNT,
        dtype=np.int8,
    )
    coefficient_scales = np.linspace(
        0.02, 0.13, receiver.CARRIER_DIMS, dtype=np.float32
    )
    absolute = rng.integers(
        -2048,
        2048,
        size=(receiver.N, receiver.CARRIER_DIMS),
        dtype=np.int32,
    )
    compact = carrier_codec.encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        _encoded_coefficients(absolute),
    )
    _, basis_bits, coefficient_bits = carrier_codec.HEADER.unpack_from(compact)
    cursor = carrier_codec.HEADER.size
    scale_bytes = receiver.CARRIER_DIMS * 4
    saved_basis_scales = compact[cursor : cursor + scale_bytes]
    cursor += scale_bytes
    cursor += scale_bytes  # coefficient scales
    lengths = compact[cursor : cursor + carrier_codec.ALPHABET_SIZE]
    cursor += carrier_codec.ALPHABET_SIZE
    cursor += receiver.CARRIER_DIMS  # Rice parameters
    basis_payload_bytes = (basis_bits + 7) // 8
    basis_payload = compact[cursor : cursor + basis_payload_bytes]
    cursor += basis_payload_bytes
    assert len(compact) - cursor == (coefficient_bits + 7) // 8
    basis_component = (
        struct.pack("<I", basis_bits)
        + saved_basis_scales
        + lengths
        + basis_payload
    )
    predictor = receiver.Predictor(
        feature_mode=receiver.FEATURE_TARGET,
        shift=8,
        feature_offsets=np.zeros(receiver.TARGET_DIMS, dtype=np.int32),
        output_offsets=np.zeros(receiver.CARRIER_DIMS, dtype=np.int32),
        weights=np.arange(
            receiver.TARGET_DIMS * receiver.CARRIER_DIMS, dtype=np.int32
        ).reshape(receiver.TARGET_DIMS, receiver.CARRIER_DIMS),
    )
    expected_basis = (
        basis_codes.reshape(receiver.BASIS_SHAPE).astype(np.float32)
        * basis_scales[:, None, None, None]
    )
    expected_coefficients = absolute.astype(np.float32) * coefficient_scales[None]
    return (
        basis_component,
        target_packet,
        predictor,
        coefficient_scales,
        absolute,
        expected_basis,
        expected_coefficients,
    )


def test_pz3r_round_trip_is_exact_and_deterministic() -> None:
    (
        basis_component,
        target_packet,
        predictor,
        coefficient_scales,
        absolute,
        expected_basis,
        expected_coefficients,
    ) = _fixture()
    kwargs = {
        "basis_component": basis_component,
        "target_packet": target_packet,
        "predictor": predictor,
        "coefficient_scales": coefficient_scales,
        "absolute_coefficients": absolute,
    }
    first = receiver.encode_pose_target_carrier(**kwargs)
    second = receiver.encode_pose_target_carrier(**kwargs)
    assert first == second
    basis, coefficients = receiver.decode_pose_target_carrier(first)
    np.testing.assert_array_equal(basis, expected_basis)
    np.testing.assert_array_equal(coefficients, expected_coefficients)


def test_target_packet_is_causal_and_tampering_fails_closed() -> None:
    (
        basis_component,
        target_packet,
        predictor,
        coefficient_scales,
        absolute,
        _,
        _,
    ) = _fixture()
    carrier = receiver.encode_pose_target_carrier(
        basis_component=basis_component,
        target_packet=target_packet,
        predictor=predictor,
        coefficient_scales=coefficient_scales,
        absolute_coefficients=absolute,
    )
    target_codes, _ = receiver.decode_pz2_packet(target_packet)
    mutated_codes = target_codes.copy()
    mutated_codes[0, 0] ^= 1
    mutated_packet = _target_packet(mutated_codes)
    assert not np.array_equal(
        receiver.predict_coefficients(target_codes, predictor),
        receiver.predict_coefficients(mutated_codes, predictor),
    )

    fields = list(receiver.HEADER.unpack_from(carrier))
    basis_bytes, target_bytes, model_bytes, residual_bytes = fields[5:9]
    start = receiver.HEADER.size
    basis = carrier[start : start + basis_bytes]
    model_start = start + basis_bytes + target_bytes
    model = carrier[model_start : model_start + model_bytes]
    residual = carrier[model_start + model_bytes :]
    fields[6] = len(mutated_packet)
    tampered = receiver.HEADER.pack(*fields) + basis + mutated_packet + model + residual
    assert len(residual) == residual_bytes
    with pytest.raises(ValueError, match="coefficient hash mismatch"):
        receiver.decode_pose_target_carrier(tampered)


def test_corruption_and_unsupported_packet_fail_closed() -> None:
    fixture = _fixture()
    carrier = receiver.encode_pose_target_carrier(
        basis_component=fixture[0],
        target_packet=fixture[1],
        predictor=fixture[2],
        coefficient_scales=fixture[3],
        absolute_coefficients=fixture[4],
    )
    corrupted = bytearray(carrier)
    corrupted[-1] ^= 1
    with pytest.raises(ValueError):
        receiver.decode_pose_target_carrier(corrupted)

    packet = bytearray(fixture[1])
    packet[9] = 1  # delta1 is deliberately unsupported by this receiver version.
    with pytest.raises(ValueError, match="unsupported PZ2"):
        receiver.decode_pz2_packet(packet)

    digest = hashlib.sha256(carrier).hexdigest()
    assert digest == hashlib.sha256(bytes(carrier)).hexdigest()


def test_legacy_outer_bundle_selects_brotli_for_pz3r(tmp_path: Path) -> None:
    semantic = b"s"
    carrier = b"PZ3R"
    hpac = b"h"
    raw = struct.pack("<II", len(semantic), len(carrier)) + semantic + carrier + hpac
    models = lzma.compress(raw, format=lzma.FORMAT_XZ)
    data_dir = tmp_path / "archive"
    data_dir.mkdir()
    (data_dir / "p").write_bytes(struct.pack("<I", len(models)) + models + b"tokens")
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHON": sys.executable,
            "PR130_DEPENDENCY_SELECTION_ONLY": "1",
        }
    )
    result = subprocess.run(
        [
            str(RUNTIME / "inflate.sh"),
            str(data_dir),
            "0",
            str(data_dir / "out.raw"),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.stdout.strip() == (
        "PR130_DEPENDENCY_SELECTION model_codec=legacy_lzma needs_brotli=1"
    )
