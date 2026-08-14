from __future__ import annotations

import lzma
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_rx1_rate_representation_attack as rx1
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_deterministic_zip_round_trip_is_structural_floor(tmp_path: Path) -> None:
    member = b"retained-real-payload"
    first = rx1.deterministic_zip(member)
    second = rx1.deterministic_zip(member)
    assert first == second
    assert len(first) == len(member) + 100
    path = tmp_path / "archive.zip"
    path.write_bytes(first)
    assert rx1.read_stored_member(path) == member


def test_retention_inventory_hashes_every_retained_file(tmp_path: Path) -> None:
    rx1.atomic_bytes(tmp_path / "retained" / "z.bin", b"z")
    rx1.atomic_bytes(tmp_path / "retained" / "nested" / "a.bin", b"abc")
    inventory = rx1.retention_inventory(tmp_path)
    assert inventory["file_count"] == 2
    assert inventory["total_bytes"] == 4
    assert [Path(row["path"]).name for row in inventory["files"]] == ["a.bin", "z.bin"]
    assert all(len(row["sha256"]) == 64 for row in inventory["files"])


def test_rx1_xz_container_round_trip() -> None:
    ihs1 = b"IHS1" + bytes(range(64))
    compressed = lzma.compress(ihs1, format=lzma.FORMAT_XZ)
    model = rx1.pack_rx1_model(
        compressed,
        b"semantic",
        b"carrier",
        codec_id=rx1.RX1_CODEC_XZ,
        table_mode=rx1.RX1_TABLE_ON,
    )
    parsed = rx1.unpack_rx1_model(model, brotli_binary="brotli")
    assert parsed["hpac"] == ihs1
    assert parsed["semantic_stream"] == b"semantic"
    assert parsed["carrier_stream"] == b"carrier"
    assert parsed["table_mode"] == rx1.RX1_TABLE_ON


@pytest.mark.parametrize("mutation", ("truncated", "trailing", "reserved"))
def test_rx1_container_rejects_noncanonical_framing(mutation: str) -> None:
    compressed = lzma.compress(b"IHS1-real", format=lzma.FORMAT_XZ)
    model = rx1.pack_rx1_model(
        compressed,
        b"semantic",
        b"carrier",
        codec_id=rx1.RX1_CODEC_XZ,
        table_mode=rx1.RX1_TABLE_OFF,
    )
    if mutation == "truncated":
        candidate = model[:-1]
    elif mutation == "trailing":
        candidate = model + b"x"
    else:
        candidate = bytearray(model)
        candidate[7] = 1
        candidate = bytes(candidate)
    with pytest.raises(ValueError):
        rx1.unpack_rx1_model(candidate, brotli_binary="brotli")


def test_neutral_residual_preserves_positive_scale_and_zeros_codes() -> None:
    compact = np.float16(0.25).tobytes() + bytes(range(94))
    neutral = rx1._neutral_residual(compact)
    assert len(neutral) == 96
    assert neutral[:2] == compact[:2]
    assert neutral[2:] == bytes(94)


def test_spatial_frame_is_exact_inverse_of_group_concatenation() -> None:
    permutation = np.arange(rx1.EVENTS_PER_FRAME, dtype=np.int64)[::-1]
    groups = [permutation[:100_000], permutation[100_000:]]
    spatial = (np.arange(rx1.EVENTS_PER_FRAME, dtype=np.uint32) % 5).astype(np.uint8)
    events = np.concatenate([spatial.reshape(-1)[positions] for positions in groups])
    restored = rx1.spatial_frame(events, groups)
    assert np.array_equal(restored.reshape(-1), spatial)


def test_probability_conversion_is_normalized_and_order_preserving() -> None:
    codes = np.asarray([[0, 8, -8, 16, -16], [4, 4, 4, 4, 4]], dtype=np.int16)
    probabilities = rx1.cp.probability_from_codes(codes, 8)
    assert probabilities.shape == codes.shape
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert int(probabilities[0].argmax()) == int(codes[0].argmax())
    assert np.allclose(probabilities[1], 0.2)


def test_rc64_snapshot_header_has_stable_exact_state_fields() -> None:
    body = b"retained-prefix"
    state = rx1.RC64_STATE_HEADER.pack(b"R6S1", 1, 2, 3, 4, 5, len(body)) + body
    magic, low, high, pending, partial, partial_bits, size = rx1.RC64_STATE_HEADER.unpack_from(state)
    assert (magic, low, high, pending, partial, partial_bits, size) == (
        b"R6S1",
        1,
        2,
        3,
        4,
        5,
        len(body),
    )
    assert state[rx1.RC64_STATE_HEADER.size :] == body


def test_rx1_python_files_pass_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=Path.cwd(),
        strict=False,
        roots=(
            "experiments/ddm_rx1_rate_representation_attack.py",
            "experiments/tests/test_ddm_rx1_rate_representation_attack.py",
        ),
    )
    assert findings == []
