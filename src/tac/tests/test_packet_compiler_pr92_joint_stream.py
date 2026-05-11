"""Tests for the PR92 ``qzs3_range_joint_r258`` packet-compiler primitives.

Covers RMC1 composite framing, RSA1 + RSB1 side-action grammars, golden
vector pinning, and representative failure modes.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler import (
    MAGIC_RMC1,
    MAGIC_RSA1,
    MAGIC_RSB1,
    RMC1Composite,
    RSA1Side,
    RSB1Side,
    encode_router_actions,
    pack_rmc1_composite,
    pack_rsa1_side,
    pack_rsb1_side,
    unpack_rmc1_composite,
    unpack_rsa1_side,
    unpack_rsb1_side,
)

GOLDEN_DIR = (
    Path(__file__).resolve().parent.parent / "packet_compiler" / "golden_vectors"
)


# ── Magic constants ─────────────────────────────────────────────────────────


def test_pr92_magic_constants_match_source_pinning() -> None:
    assert MAGIC_RMC1 == b"RMC1"
    assert MAGIC_RSA1 == b"RSA1"
    assert MAGIC_RSB1 == b"RSB1"


# ── RMC1 composite round-trip ───────────────────────────────────────────────


def test_pr92_rmc1_round_trips_two_streams() -> None:
    seg = b"\x01\x02\x03\xff"
    side = b"\xa0\xa1\xa2"
    composite = pack_rmc1_composite(seg, side)
    parsed = unpack_rmc1_composite(composite.payload)
    assert isinstance(parsed, RMC1Composite)
    assert parsed.seg_bytes == seg
    assert parsed.side_bytes == side


def test_pr92_rmc1_round_trips_empty_components() -> None:
    parsed = unpack_rmc1_composite(pack_rmc1_composite(b"", b"").payload)
    assert parsed.seg_bytes == b""
    assert parsed.side_bytes == b""


def test_pr92_rmc1_round_trips_one_empty_component() -> None:
    parsed = unpack_rmc1_composite(pack_rmc1_composite(b"seg", b"").payload)
    assert parsed.seg_bytes == b"seg"
    assert parsed.side_bytes == b""

    parsed = unpack_rmc1_composite(pack_rmc1_composite(b"", b"side").payload)
    assert parsed.seg_bytes == b""
    assert parsed.side_bytes == b"side"


def test_pr92_rmc1_payload_starts_with_magic() -> None:
    composite = pack_rmc1_composite(b"x", b"y")
    assert composite.payload[:4] == MAGIC_RMC1


def test_pr92_rmc1_decoder_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="RMC1 magic"):
        unpack_rmc1_composite(b"XXXX" + struct.pack("<II", 0, 0))


def test_pr92_rmc1_decoder_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated"):
        unpack_rmc1_composite(MAGIC_RMC1 + b"\x00\x00")


def test_pr92_rmc1_decoder_rejects_length_mismatch() -> None:
    payload = MAGIC_RMC1 + struct.pack("<II", 10, 0) + b"only-3"
    with pytest.raises(ValueError, match="length mismatch"):
        unpack_rmc1_composite(payload)


# ── RSA1 side-action round-trip ─────────────────────────────────────────────


def test_pr92_rsa1_round_trips_standard_3bit() -> None:
    rng = np.random.default_rng(seed=20260511)
    actions = rng.integers(0, 8, size=600, dtype=np.uint8)
    body = encode_router_actions(actions, bits=3)
    side = pack_rsa1_side(count=600, action_bits=3, table_id=2, body=body)
    parsed = unpack_rsa1_side(side.payload)
    assert isinstance(parsed, RSA1Side)
    assert parsed.count == 600
    assert parsed.action_bits == 3
    assert parsed.table_id == 2
    # Body is preserved at least up to the required length.
    required = (600 * 3 + 7) // 8
    assert parsed.body[:required] == body[:required]


def test_pr92_rsa1_payload_starts_with_magic() -> None:
    body = encode_router_actions(np.zeros(8, dtype=np.uint8), bits=3)
    side = pack_rsa1_side(count=8, action_bits=3, table_id=0, body=body)
    assert side.payload[:4] == MAGIC_RSA1


def test_pr92_rsa1_rejects_out_of_range_count() -> None:
    with pytest.raises(ValueError, match="count"):
        pack_rsa1_side(count=-1, action_bits=3, table_id=0, body=b"\x00")
    with pytest.raises(ValueError, match="count"):
        pack_rsa1_side(count=70000, action_bits=3, table_id=0, body=b"\x00")


def test_pr92_rsa1_rejects_invalid_action_bits() -> None:
    with pytest.raises(ValueError, match="action_bits"):
        pack_rsa1_side(count=8, action_bits=0, table_id=0, body=b"\x00")
    with pytest.raises(ValueError, match="action_bits"):
        pack_rsa1_side(count=8, action_bits=9, table_id=0, body=b"\x00")


def test_pr92_rsa1_rejects_out_of_range_table_id() -> None:
    with pytest.raises(ValueError, match="table_id"):
        pack_rsa1_side(count=8, action_bits=3, table_id=256, body=b"\x00\x00\x00")


def test_pr92_rsa1_rejects_undersized_body() -> None:
    with pytest.raises(ValueError, match="body too small"):
        pack_rsa1_side(count=600, action_bits=3, table_id=0, body=b"\x00")


def test_pr92_rsa1_decoder_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="RSA1 magic"):
        unpack_rsa1_side(b"XXXX" + b"\x00" * 8)


def test_pr92_rsa1_decoder_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated"):
        unpack_rsa1_side(MAGIC_RSA1 + b"\x00\x00")


def test_pr92_rsa1_decoder_rejects_unsupported_action_bits() -> None:
    # Hand-craft a payload with action_bits=12 (invalid).
    payload = MAGIC_RSA1 + struct.pack("<HBB", 0, 12, 0)
    with pytest.raises(ValueError, match="unsupported RSA1"):
        unpack_rsa1_side(payload)


# ── RSB1 side-action round-trip ─────────────────────────────────────────────


def test_pr92_rsb1_round_trips_brotli_compressed() -> None:
    rng = np.random.default_rng(seed=20260511)
    actions = rng.integers(0, 256, size=300, dtype=np.uint8)
    side = pack_rsb1_side(actions=actions, table_id=7)
    parsed = unpack_rsb1_side(side.payload)
    assert isinstance(parsed, RSB1Side)
    assert parsed.count == 300
    assert parsed.table_id == 7
    assert parsed.body_bytes == actions.tobytes()


def test_pr92_rsb1_round_trips_empty() -> None:
    side = pack_rsb1_side(actions=np.zeros(0, dtype=np.uint8), table_id=0)
    parsed = unpack_rsb1_side(side.payload)
    assert parsed.count == 0
    assert parsed.body_bytes == b""


def test_pr92_rsb1_payload_starts_with_magic() -> None:
    side = pack_rsb1_side(actions=np.zeros(4, dtype=np.uint8), table_id=0)
    assert side.payload[:4] == MAGIC_RSB1


def test_pr92_rsb1_rejects_out_of_range_count() -> None:
    huge = np.zeros(70000, dtype=np.uint8)
    with pytest.raises(ValueError, match="count"):
        pack_rsb1_side(actions=huge)


def test_pr92_rsb1_rejects_out_of_range_table_id() -> None:
    with pytest.raises(ValueError, match="table_id"):
        pack_rsb1_side(actions=np.zeros(4, dtype=np.uint8), table_id=256)


def test_pr92_rsb1_decoder_rejects_wrong_magic() -> None:
    with pytest.raises(ValueError, match="RSB1 magic"):
        unpack_rsb1_side(b"XXXX" + b"\x00" * 8)


def test_pr92_rsb1_decoder_rejects_truncated_header() -> None:
    with pytest.raises(ValueError, match="truncated"):
        unpack_rsb1_side(MAGIC_RSB1 + b"\x00\x00\x00")


# ── Frozen dataclass invariants ─────────────────────────────────────────────


def test_pr92_dataclasses_are_frozen() -> None:
    composite = pack_rmc1_composite(b"a", b"b")
    with pytest.raises((AttributeError, TypeError)):
        composite.seg_bytes = b"x"  # type: ignore[misc]

    side = pack_rsa1_side(count=8, action_bits=3, table_id=0, body=b"\x00" * 3)
    with pytest.raises((AttributeError, TypeError)):
        side.count = 999  # type: ignore[misc]


# ── Golden vectors ──────────────────────────────────────────────────────────


class TestPR92GoldenVectors:
    def test_rmc_joint_stream_golden_vector(self) -> None:
        # Build a deterministic RMC1-wrapping-RSA1-side composite. This
        # composes the two primitives together so the golden vector pins
        # the joint behavior + the inner side-action framing.
        rng = np.random.default_rng(seed=20260511)
        seg_bytes = bytes(rng.integers(0, 256, size=128, dtype=np.uint8).tolist())
        actions = rng.integers(0, 8, size=120, dtype=np.uint8)
        body = encode_router_actions(actions, bits=3)
        side = pack_rsa1_side(count=120, action_bits=3, table_id=2, body=body)
        composite = pack_rmc1_composite(seg_bytes, side.payload)
        digest = hashlib.sha256(composite.payload).hexdigest()
        golden = GOLDEN_DIR / "pr92_rmc_joint_stream_v1.json"
        if golden.exists():
            data = json.loads(golden.read_text(encoding="utf-8"))
            assert data["sha256"] == digest, (
                "PR92 RMC1 byte stream changed; "
                "delete + regenerate vector if intentional"
            )
        else:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            golden.write_text(
                json.dumps(
                    {
                        "action_bits": 3,
                        "action_count": 120,
                        "payload_len": len(composite.payload),
                        "schema": "pr92_rmc_joint_stream.v1",
                        "seed": 20260511,
                        "seg_bytes_len": len(seg_bytes),
                        "sha256": digest,
                        "table_id": 2,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
