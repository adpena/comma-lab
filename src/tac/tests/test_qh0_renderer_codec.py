# SPDX-License-Identifier: MIT
from __future__ import annotations

import zipfile
import json
import struct
import zlib
from pathlib import Path

import pytest
import torch

from tac.pr85_bundle import expand_pr85_bundle_to_runtime_members
from tac.qh0_renderer_codec import (
    QH0CodecError,
    QH1_HEADER_STRUCT,
    QH1_SCHEMA,
    decode_qh0_state_dict,
    load_qh0,
    reconstruct_qh1_payload,
)
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer


REPO = Path(__file__).resolve().parents[3]


def _real_pr85_qh0_payload() -> bytes:
    archive = REPO / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
    if not archive.is_file():
        pytest.skip("public PR85 intake archive is not present")
    with zipfile.ZipFile(archive, "r") as zf:
        raw = zf.read("x")
    return expand_pr85_bundle_to_runtime_members(raw).members["renderer.bin"]


def test_real_pr85_qh0_decodes_to_jointframegenerator_state_dict() -> None:
    payload = _real_pr85_qh0_payload()
    expected = build_quantizr_faithful_renderer().state_dict()

    state, report = decode_qh0_state_dict(payload, device="cpu")

    assert report.magic == "QH0"
    assert report.payload_bytes == len(payload)
    assert report.consumed_bytes == len(payload)
    assert report.tensor_count == len(expected)
    assert report.q_fp4_tensor_count > 0
    assert report.fp16_tensor_count > 0
    assert report.int8_dense_tensor_count > 0
    assert set(state) == set(expected)
    for key, expected_tensor in expected.items():
        assert state[key].shape == expected_tensor.shape, key
        assert state[key].dtype == torch.float32, key
        assert torch.isfinite(state[key]).all(), key


def test_real_pr85_qh0_loads_strict_model() -> None:
    payload = _real_pr85_qh0_payload()

    model = load_qh0(payload, device="cpu")

    assert model.num_classes == 5
    assert model.pose_dim == 6
    assert model.cond_dim == 48
    assert sum(p.numel() for p in model.parameters()) > 80_000


def test_qh0_decoder_fails_closed_on_bad_magic_and_trailing_bytes() -> None:
    payload = _real_pr85_qh0_payload()

    with pytest.raises(QH0CodecError, match="unsupported QH0 renderer magic"):
        decode_qh0_state_dict(b"BAD" + payload[3:], device="cpu")
    with pytest.raises(QH0CodecError, match="trailing bytes"):
        decode_qh0_state_dict(payload + b"x", device="cpu")


def _qh1_for_slice(source: bytes, offset: int, nbytes: int) -> bytes:
    base = bytearray(source)
    original = bytes(base[offset : offset + nbytes])
    base[offset : offset + nbytes] = b"\x00" * nbytes
    base_encoded = zlib.compress(bytes(base), 9)
    patch_encoded = zlib.compress(original, 9)
    header = {
        "schema": QH1_SCHEMA,
        "source_bytes": len(source),
        "source_sha256": __import__("hashlib").sha256(source).hexdigest(),
        "base_codec": "zlib",
        "base_encoded_bytes": len(base_encoded),
        "base_encoded_sha256": __import__("hashlib").sha256(base_encoded).hexdigest(),
        "base_decoded_sha256": __import__("hashlib").sha256(bytes(base)).hexdigest(),
        "records": [
            {
                "offset": offset,
                "nbytes": nbytes,
                "codec": "zlib",
                "encoded_bytes": len(patch_encoded),
                "encoded_sha256": __import__("hashlib").sha256(patch_encoded).hexdigest(),
                "decoded_sha256": __import__("hashlib").sha256(original).hexdigest(),
            }
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    return b"QH1" + QH1_HEADER_STRUCT.pack(len(header_bytes)) + header_bytes + base_encoded + patch_encoded


def test_qh1_reconstructs_exact_qh0_and_decodes_same_state() -> None:
    payload = _real_pr85_qh0_payload()
    qh1 = _qh1_for_slice(payload, 48830, 4609)

    assert reconstruct_qh1_payload(qh1) == payload

    qh0_state, qh0_report = decode_qh0_state_dict(payload, device="cpu")
    qh1_state, qh1_report = decode_qh0_state_dict(qh1, device="cpu")
    assert qh1_report.magic == qh0_report.magic == "QH0"
    assert qh1_report.payload_bytes == qh0_report.payload_bytes == len(payload)
    assert set(qh1_state) == set(qh0_state)
    for key in qh0_state:
        assert torch.equal(qh1_state[key], qh0_state[key]), key


def test_qh1_rejects_overlapping_records() -> None:
    payload = _real_pr85_qh0_payload()
    base_encoded = zlib.compress(payload, 9)
    patch_a = zlib.compress(payload[10:20], 9)
    patch_b = zlib.compress(payload[15:25], 9)
    import hashlib

    header = {
        "schema": QH1_SCHEMA,
        "source_bytes": len(payload),
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "base_codec": "zlib",
        "base_encoded_bytes": len(base_encoded),
        "base_encoded_sha256": hashlib.sha256(base_encoded).hexdigest(),
        "base_decoded_sha256": hashlib.sha256(payload).hexdigest(),
        "records": [
            {
                "offset": 10,
                "nbytes": 10,
                "codec": "zlib",
                "encoded_bytes": len(patch_a),
                "encoded_sha256": hashlib.sha256(patch_a).hexdigest(),
                "decoded_sha256": hashlib.sha256(payload[10:20]).hexdigest(),
            },
            {
                "offset": 15,
                "nbytes": 10,
                "codec": "zlib",
                "encoded_bytes": len(patch_b),
                "encoded_sha256": hashlib.sha256(patch_b).hexdigest(),
                "decoded_sha256": hashlib.sha256(payload[15:25]).hexdigest(),
            },
        ],
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    qh1 = b"QH1" + QH1_HEADER_STRUCT.pack(len(header_bytes)) + header_bytes + base_encoded + patch_a + patch_b
    with pytest.raises(QH0CodecError, match="overlapping target slice"):
        reconstruct_qh1_payload(qh1)


def test_preflight_recognizes_qh0_as_runtime_renderer(tmp_path: Path) -> None:
    from tac.preflight import preflight_check

    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"QH0" + b"\x00" * 16)

    assert preflight_check(renderer_path=renderer, verbose=False) == []


def test_preflight_recognizes_qh1_as_runtime_renderer(tmp_path: Path) -> None:
    from tac.preflight import preflight_check

    renderer = tmp_path / "renderer.bin"
    renderer.write_bytes(b"QH1" + struct.pack("<I", 2) + b"{}")

    assert preflight_check(renderer_path=renderer, verbose=False) == []
