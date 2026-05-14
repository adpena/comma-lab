# SPDX-License-Identifier: MIT
"""Tests for CPL2 wire format int-key preservation (ORCH-SYNC Bug 2).

Background
----------
WIRE-DECODER (commit 669b5b5f) discovered that ``CodecPipeline``'s CPL1
wire format coerces ``effective_byte_maps`` int keys to strings via
``json.dumps``. The PR101 decoder's ``decode_decoder_compact`` then does
``idx in effective_byte_maps`` for an INT idx, misses the string key,
falls through to ``DECODER_BYTE_MAPS`` defaults, and produces 100%
sign-flips on negzig tensors (e.g. ``blocks.5.bias``).

CPL2 (default 2026-05-08) preserves int keys via a sentinel envelope:
``{"__intkey__": true, "items": [[k, v], ...]}``. CPL1 retained for
backwards compat (forensic only — operators must opt in with
``version=1``).

Coverage
--------
1. CPL1 still coerces int keys to strings (regression baseline).
2. CPL2 round-trips a dict[int, str] with EXACT int keys.
3. CPL2 round-trips nested int-keyed dicts.
4. CPL2 detects mixed int+str key dicts and raises ValueError.
5. CPL2 rejects the sentinel string ``__intkey__`` as a literal key.
6. The negzig sign-flip case: ``effective_byte_maps={5: "negzig", ...}``
   round-trips through CPL2 with int keys preserved, so the PR101 decoder
   correctly applies negzig to tensor index 5 (no sign-flip).
7. CPL1 magic is still accepted by ``decode`` (auto-detection).
8. CPL2 magic is the default emitted by ``encode``.
9. Empty op_state round-trips through CPL2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from tac.codec_pipeline import (
    CodecPipeline,
    EncodeResult,
    ValidationReport,
    _cpl2_recursively_decode,
    _cpl2_recursively_encode,
    _decode_op_state_from_json_bytes,
    _encode_op_state_to_json_bytes,
)


# ---------------------------------------------------------------------------
# Stub op for low-level wire-format coverage (no torch dependency).
# ---------------------------------------------------------------------------


@dataclass
class _IntKeyOp:
    """Stub op that records a dict[int, str] in op_state.

    Encodes a deterministic blob; decode rehydrates and asserts the int
    keys round-trip correctly. Used to exercise the CPL2 wire format
    without dragging in PR101 / PR103 codec stacks.
    """
    name: str = "intkey_stub"
    keys_to_emit: dict[int, str] | None = None

    def encode(
        self,
        state_dict: dict[str, Any],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        op_state = {"int_keyed_map": dict(self.keys_to_emit or {})}
        return EncodeResult(
            blob=b"stub-blob",
            bytes_in=0,
            bytes_out=len(b"stub-blob"),
            op_name=self.name,
            op_state=op_state,
        )

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        return {"_replayed_op_state": op_state}

    def validate(
        self,
        state_dict: dict[str, Any],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        return ValidationReport(passed=True, op_name=self.name, findings=[])


# ---------------------------------------------------------------------------
# Test 1: CPL1 baseline — int keys are coerced to strings (regression).
# ---------------------------------------------------------------------------


def test_cpl1_coerces_int_keys_to_strings_regression() -> None:
    """Document the legacy CPL1 bug: int dict keys become string keys.

    Future agents must NOT regress CPL2 back to CPL1 default; this test
    pins the failure mode CPL2 fixes.
    """
    op_state = {"effective_byte_maps": {0: "default", 5: "negzig"}}
    state_b = _encode_op_state_to_json_bytes(op_state, version=1)
    decoded = _decode_op_state_from_json_bytes(state_b, version=1)
    # CPL1 returns string keys (the bug).
    assert "effective_byte_maps" in decoded
    assert "0" in decoded["effective_byte_maps"]
    assert "5" in decoded["effective_byte_maps"]
    # Int keys are NOT preserved under v1.
    assert 0 not in decoded["effective_byte_maps"]
    assert 5 not in decoded["effective_byte_maps"]


# ---------------------------------------------------------------------------
# Test 2: CPL2 preserves dict[int, str] exactly.
# ---------------------------------------------------------------------------


def test_cpl2_roundtrips_int_keys_exactly() -> None:
    op_state = {"effective_byte_maps": {0: "default", 5: "negzig", 9: "off"}}
    state_b = _encode_op_state_to_json_bytes(op_state, version=2)
    decoded = _decode_op_state_from_json_bytes(state_b, version=2)
    assert decoded == op_state
    assert decoded["effective_byte_maps"][5] == "negzig"
    assert isinstance(list(decoded["effective_byte_maps"].keys())[0], int)


# ---------------------------------------------------------------------------
# Test 3: CPL2 nested int-keyed dicts round-trip.
# ---------------------------------------------------------------------------


def test_cpl2_roundtrips_nested_int_keys() -> None:
    op_state = {
        "outer": {
            7: {1: "alpha", 2: "beta"},
            8: {3: "gamma"},
        },
        "metadata": "foo",
    }
    state_b = _encode_op_state_to_json_bytes(op_state, version=2)
    decoded = _decode_op_state_from_json_bytes(state_b, version=2)
    assert decoded == op_state
    assert decoded["outer"][7][1] == "alpha"


# ---------------------------------------------------------------------------
# Test 4: CPL2 rejects mixed int+str keys.
# ---------------------------------------------------------------------------


def test_cpl2_rejects_mixed_int_str_keys() -> None:
    op_state = {"mixed": {0: "a", "name": "b"}}
    with pytest.raises(ValueError, match="mixes int and str keys"):
        _encode_op_state_to_json_bytes(op_state, version=2)


# ---------------------------------------------------------------------------
# Test 5: CPL2 rejects literal __intkey__ string keys (sentinel collision).
# ---------------------------------------------------------------------------


def test_cpl2_rejects_intkey_sentinel_collision() -> None:
    op_state = {"bad": {"__intkey__": True, "items": [[0, "a"]]}}
    with pytest.raises(ValueError, match="reserved string key"):
        _encode_op_state_to_json_bytes(op_state, version=2)


# ---------------------------------------------------------------------------
# Test 6: full pipeline negzig sign-flip case round-trips correctly.
# ---------------------------------------------------------------------------


def test_pipeline_intkey_op_state_roundtrips_via_cpl2() -> None:
    """The exact failure mode WIRE-DECODER discovered: emit op_state with
    ``{5: "negzig"}``, encode + decode through the pipeline, confirm the
    decoded op_state preserves int 5 (not str "5").
    """
    op = _IntKeyOp(keys_to_emit={0: "default", 5: "negzig", 9: "off"})
    pipe = CodecPipeline([op])
    blob, manifest = pipe.encode({}, skip_validate=True)
    # Default version is CPL2.
    assert blob[:4] == b"CPL2"

    decoded_state, replayed = pipe.decode(blob)
    op_state_back = decoded_state["_replayed_op_state"]
    assert op_state_back == {
        "int_keyed_map": {0: "default", 5: "negzig", 9: "off"}
    }
    # The exact key the negzig sign-flip bug needed: int 5, not "5".
    assert 5 in op_state_back["int_keyed_map"]
    assert op_state_back["int_keyed_map"][5] == "negzig"


# ---------------------------------------------------------------------------
# Test 7: CPL1 magic still decodes (forensic backwards compat).
# ---------------------------------------------------------------------------


def test_cpl1_legacy_magic_still_decodes() -> None:
    """Encode with version=1 (CPL1 magic), decode without specifying — the
    pipeline auto-detects magic and routes to v1 JSON decoding.
    """
    # Use a stub op whose decode does NOT depend on int keys (so CPL1's
    # string-key coercion is invisible to the test).
    op = _IntKeyOp(keys_to_emit={0: "default"})
    pipe = CodecPipeline([op])
    blob, _ = pipe.encode({}, skip_validate=True, version=1)
    assert blob[:4] == b"CPL1"
    decoded_state, replayed = pipe.decode(blob)
    op_state_back = decoded_state["_replayed_op_state"]
    # CPL1 string-key coercion is preserved (regression evidence).
    assert "0" in op_state_back["int_keyed_map"]


# ---------------------------------------------------------------------------
# Test 8: CPL2 is the DEFAULT emitted by encode.
# ---------------------------------------------------------------------------


def test_cpl2_is_default_encode_version() -> None:
    op = _IntKeyOp(keys_to_emit={})
    pipe = CodecPipeline([op])
    assert pipe.DEFAULT_VERSION == 2
    blob, _ = pipe.encode({}, skip_validate=True)
    assert blob[:4] == b"CPL2"


# ---------------------------------------------------------------------------
# Test 9: empty op_state still round-trips through CPL2.
# ---------------------------------------------------------------------------


def test_cpl2_empty_op_state_roundtrips() -> None:
    state_b = _encode_op_state_to_json_bytes({}, version=2)
    decoded = _decode_op_state_from_json_bytes(state_b, version=2)
    assert decoded == {}


# ---------------------------------------------------------------------------
# Test 10: low-level helpers exposed for unit-test of the recursion
# ---------------------------------------------------------------------------


def test_cpl2_recursive_encode_decode_helpers() -> None:
    payload = {
        "scalar": 42,
        "list": [1, 2, {"nested_int_keys": {0: "x", 1: "y"}}],
        "deep": {"a": {"b": {3: "c", 4: "d"}}},
    }
    encoded = _cpl2_recursively_encode(payload)
    # Round-trip through json+recursive-decode to mimic on-wire path.
    serialized = json.dumps(encoded, sort_keys=True)
    raw = json.loads(serialized)
    decoded = _cpl2_recursively_decode(raw)
    assert decoded == payload
    assert decoded["list"][2]["nested_int_keys"][0] == "x"
    assert decoded["deep"]["a"]["b"][3] == "c"
