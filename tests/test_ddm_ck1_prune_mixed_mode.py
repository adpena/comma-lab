"""DDM-CK1: SM3R MODE_ROW_PRUNE_MIXED (mode 6) packer/receiver contract.

Mode 6 composes the two admitted semantic edits on one body: keep01's
``MODE_ROW_PRUNE`` row geometry plus an SD1M-style per-tensor bit-depth table
so tensors outside the prune selection can store below four bits.

These tests exercise the CODEC MECHANISM on a deterministic structured state.
They make no score, rate, or d_seg claim and are not an empirical anchor: the
real-input proof (the exact rr4 semantic state, the exact archive container,
and parse-back through the staged shipping receiver) lives in the DDM-CK1
generation receipt, not here.  The one real-input fact this file pins is the
byte-freeze in ``test_mode_five_payload_bytes_are_frozen`` — MODE_ROW_PRUNE's
payload must not move, because keep01 is the live pointer.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

import ddm_mp2_semantic_receiver as mp2  # noqa: E402
import ddm_sd1_semantic_rd_curve as sd1  # noqa: E402
import ddm_sm3_semantic_representation as sm3  # noqa: E402

# Rank>=2 tensor order mirrors the shipped SD1M v1 template order.
_WEIGHT_SHAPES = (
    ("token_embed.weight", (8, 5)),
    ("frame_embed.weight", (9, 6)),
    ("coord_mix.weight", (7, 4)),
    ("blocks.0.dw.weight", (6, 3)),
    ("blocks.0.pw.weight", (8, 4)),
    ("blocks.0.film.weight", (10, 5)),
    ("blocks.1.dw.weight", (6, 3)),
    ("blocks.1.pw.weight", (8, 4)),
    ("blocks.1.film.weight", (10, 5)),
    ("blocks.2.dw.weight", (6, 3)),
    ("blocks.2.pw.weight", (8, 4)),
    ("blocks.2.film.weight", (10, 5)),
    ("blocks.3.dw.weight", (6, 3)),
    ("blocks.3.pw.weight", (8, 4)),
    ("blocks.3.film.weight", (10, 5)),
    ("head.weight", (5, 4)),
)

# MODE_ROW_PRUNE payload SHA-256 for `structured_state()` at keep_percent=50,
# recorded from the pre-landing source (sm3 sha 6da0aa13...) and re-asserted
# after.  A change here means MODE_ROW_PRUNE's bytes moved, which would move
# the live keep01 pointer's archive.
_FROZEN_MODE_FIVE_SHA256 = "b9f7e1311d6f9c77dd9e65e92abc73416dd0cd21607e9ab8eba25e18c64e4861"

_KEEP_PERCENT = 50
_MIXED_DEPTHS = {"frame_embed.weight": 3, "blocks.0.film.weight": 3}


def _deterministic(count: int, salt: int) -> torch.Tensor:
    """Reproducible values from integer arithmetic only (no RNG, no libm)."""

    index = torch.arange(count, dtype=torch.float64)
    values = ((index * 37.0 + salt * 11.0) % 97.0) / 97.0 - 0.5
    return (values * (1.0 + salt % 3)).to(torch.float32)


def structured_state() -> OrderedDict[str, torch.Tensor]:
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for salt, (name, shape) in enumerate(_WEIGHT_SHAPES):
        rows, columns = shape
        state[name] = _deterministic(rows * columns, salt + 1).reshape(rows, columns)
        state[name.replace(".weight", ".bias")] = _deterministic(rows, salt + 41)
    return state


def _mode_five(state):
    return sm3.pack_prune_candidate(state, _KEEP_PERCENT)


def _mode_six(state, depths=None):
    return sm3.pack_prune_mixed_candidate(state, _KEEP_PERCENT, depths or {})


def _max_deviation(left, right) -> float:
    return max(float((left[name] - right[name]).abs().max()) for name in left)


# --------------------------------------------------------------------------
# Structure of the fixture and the selections it must cover
# --------------------------------------------------------------------------


def test_quantized_order_covers_prune_and_mixed_selections():
    state = structured_state()
    names = sd1.quantized_names(state)
    assert names == [name for name, _ in _WEIGHT_SHAPES]
    assert sm3.PRUNE_NAMES.issubset(names)
    assert sm3.PRUNE_MIXED_Q3_NAMES.issubset(names)
    # The composed edit's value depends on the two selections being disjoint.
    assert not sm3.PRUNE_NAMES & sm3.PRUNE_MIXED_Q3_NAMES


def test_mixed_q3_names_are_the_non_overlapping_s2_legs():
    assert sm3.PRUNE_MIXED_Q3_NAMES == {"frame_embed.weight", "blocks.0.film.weight"}


# --------------------------------------------------------------------------
# MODE_ROW_PRUNE must not move
# --------------------------------------------------------------------------


def test_mode_five_payload_bytes_are_frozen():
    payload, _, _ = _mode_five(structured_state())
    assert hashlib.sha256(payload).hexdigest() == _FROZEN_MODE_FIVE_SHA256


def test_mode_five_roundtrip_is_still_exact():
    state = structured_state()
    payload, expected, _ = _mode_five(state)
    sm3.assert_state_equal(expected, sm3.unpack_candidate(payload, state))
    assert payload[5] == sm3.MODE_ROW_PRUNE


def test_mode_five_still_decodes_through_the_shipping_receiver():
    state = structured_state()
    payload, expected, _ = _mode_five(state)
    decoded = mp2.unpack_variant_semantic_or_none(payload, state)
    assert decoded is not None
    assert _max_deviation(decoded, expected) == 0.0


# --------------------------------------------------------------------------
# Mode 6 wire format
# --------------------------------------------------------------------------


def test_mode_six_header_fields():
    state = structured_state()
    payload, _, details = _mode_six(state, _MIXED_DEPTHS)
    assert payload[:4] == sm3.MAGIC
    assert payload[4] == sm3.VERSION
    assert payload[5] == sm3.MODE_ROW_PRUNE_MIXED == 6
    assert payload[6] == _KEEP_PERCENT
    assert payload[7] == 0
    mask = struct.unpack_from("<H", payload, 8)[0]
    assert mask == details["selection_mask"]
    assert mask == sm3.mask_for_names(sd1.quantized_names(state), sm3.PRUNE_NAMES)


def test_empty_depth_override_is_mode_five_plus_the_depth_table():
    state = structured_state()
    five, _, _ = _mode_five(state)
    six, _, _ = _mode_six(state)
    table_bytes = (len(sd1.quantized_names(state)) + 1) // 2
    assert len(six) - len(five) == table_bytes == 8
    assert six[:5] == five[:5]
    assert six[6:10] == five[6:10]
    assert six[10 : 10 + table_bytes] == sd1._pack_depth_nibbles([4] * len(sd1.quantized_names(state)))
    assert six[10 + table_bytes :] == five[10:]


def test_details_report_the_non_default_allocation_only():
    _, _, details = _mode_six(structured_state(), _MIXED_DEPTHS)
    assert details["non_default_bit_allocation"] == _MIXED_DEPTHS
    assert set(details["bit_allocation"]) == set(sd1.quantized_names(structured_state()))
    assert details["keep_percent"] == _KEEP_PERCENT


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------


def test_mode_six_roundtrip_through_the_packer_is_exact():
    state = structured_state()
    payload, expected, _ = _mode_six(state, _MIXED_DEPTHS)
    sm3.assert_state_equal(expected, sm3.unpack_candidate(payload, state))


def test_mode_six_roundtrip_through_the_shipping_receiver_is_exact():
    state = structured_state()
    payload, expected, _ = _mode_six(state, _MIXED_DEPTHS)
    decoded = mp2.unpack_variant_semantic_or_none(payload, state)
    assert decoded is not None
    assert list(decoded) == list(expected)
    assert _max_deviation(decoded, expected) == 0.0


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 7, 8])
def test_packer_and_receiver_agree_at_every_depth(bits):
    state = structured_state()
    depths = {name: bits for name in sd1.quantized_names(state)}
    payload, expected, _ = _mode_six(state, depths)
    assert _max_deviation(mp2.unpack_variant_semantic_or_none(payload, state), expected) == 0.0
    sm3.assert_state_equal(expected, sm3.unpack_candidate(payload, state))


def test_mixed_depth_shrinks_the_payload_below_mode_five():
    state = structured_state()
    five, _, _ = _mode_five(state)
    six, _, _ = _mode_six(state, _MIXED_DEPTHS)
    assert len(six) < len(five)


def test_depth_applies_to_the_pruned_survivor_rows():
    state = structured_state()
    wide, _, _ = _mode_six(state, {})
    narrow, expected, _ = _mode_six(state, {name: 2 for name in sm3.PRUNE_NAMES})
    assert len(narrow) < len(wide)
    assert _max_deviation(mp2.unpack_variant_semantic_or_none(narrow, state), expected) == 0.0


def test_pruned_rows_outside_the_kept_set_decode_to_zero():
    state = structured_state()
    _, expected, details = _mode_six(state, _MIXED_DEPTHS)
    for name, keep in details["kept_rows"].items():
        rows = expected[name].reshape(state[name].shape[0], -1)
        nonzero_rows = int((rows.abs().sum(dim=1) > 0).sum())
        assert nonzero_rows <= keep


# --------------------------------------------------------------------------
# Depth resolution
# --------------------------------------------------------------------------


def test_resolve_mixed_depths_defaults_to_four():
    names = sd1.quantized_names(structured_state())
    resolved = sm3.resolve_mixed_depths(names, {"head.weight": 3})
    assert list(resolved) == names
    assert resolved["head.weight"] == 3
    assert all(resolved[name] == 4 for name in names if name != "head.weight")


def test_resolve_mixed_depths_rejects_unknown_name():
    names = sd1.quantized_names(structured_state())
    with pytest.raises(ValueError, match="absent from the tensor order"):
        sm3.resolve_mixed_depths(names, {"no.such.weight": 3})


@pytest.mark.parametrize("bits", [0, 1, 9, 16])
def test_resolve_mixed_depths_rejects_out_of_range_depth(bits):
    names = sd1.quantized_names(structured_state())
    with pytest.raises(ValueError, match=r"must be in \[2, 8\]"):
        sm3.resolve_mixed_depths(names, {"head.weight": bits})


# --------------------------------------------------------------------------
# Fail-closed paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize("keep_percent", [0, 100, 255])
def test_pack_rejects_invalid_keep_percent(keep_percent):
    with pytest.raises(ValueError, match="keep percentage"):
        sm3.pack_prune_mixed_candidate(structured_state(), keep_percent, {})


def test_unpack_rejects_a_wrong_mode_byte():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    corrupted = payload[:5] + bytes([sm3.MODE_ROW_PRUNE]) + payload[6:]
    with pytest.raises(ValueError):
        sm3.unpack_prune_mixed_candidate(corrupted, state)


def test_unpack_rejects_a_nonzero_reserved_byte():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    corrupted = payload[:7] + bytes([1]) + payload[8:]
    with pytest.raises(ValueError, match="mixed-prune header"):
        sm3.unpack_prune_mixed_candidate(corrupted, state)


def test_unpack_rejects_a_mask_that_does_not_match_the_template():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    corrupted = payload[:8] + struct.pack("<H", 0) + payload[10:]
    with pytest.raises(ValueError, match="selection mask differs"):
        sm3.unpack_prune_mixed_candidate(corrupted, state)


def test_unpack_rejects_a_truncated_depth_table():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    with pytest.raises(ValueError, match="truncated mixed-prune depth allocation"):
        sm3.unpack_prune_mixed_candidate(payload[:12], state)


def test_unpack_rejects_trailing_bytes():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    with pytest.raises(ValueError, match="trailing bytes"):
        sm3.unpack_prune_mixed_candidate(payload + b"\x00", state)


def test_receiver_rejects_an_unknown_sm3r_mode():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    corrupted = payload[:5] + bytes([9]) + payload[6:]
    with pytest.raises(mp2.MP2SemanticFormatError, match="unsupported SM3R mode 9"):
        mp2.unpack_variant_semantic_or_none(corrupted, state)


def test_receiver_rejects_trailing_bytes():
    state = structured_state()
    payload, _, _ = _mode_six(state, _MIXED_DEPTHS)
    with pytest.raises(mp2.MP2SemanticFormatError, match="trailing bytes"):
        mp2.unpack_variant_semantic_or_none(payload + b"\x00", state)


def test_receiver_returns_none_for_untagged_payloads():
    assert mp2.unpack_variant_semantic_or_none(b"HV1 legacy q4 bytes", structured_state()) is None


# --------------------------------------------------------------------------
# The refactor the mode-6 landing performed must not disturb SD1M
# --------------------------------------------------------------------------


def test_sd1m_still_decodes_after_the_shared_nibble_helper_refactor():
    state = structured_state()
    names = sd1.quantized_names(state)
    allocation = OrderedDict((name, 3 if name in sm3.PRUNE_MIXED_Q3_NAMES else 4) for name in names)
    payload, expected = sd1.pack_semantic_state(state, allocation, legacy_int4=False)
    decoded = mp2.unpack_variant_semantic_or_none(payload, state)
    assert decoded is not None
    assert _max_deviation(decoded, expected) == 0.0


def test_sd1m_rejects_an_invalid_depth_nibble():
    state = structured_state()
    names = sd1.quantized_names(state)
    payload, _ = sd1.pack_semantic_state(
        state, OrderedDict((name, 4) for name in names), legacy_int4=False
    )
    corrupted = payload[:6] + bytes([0x01]) + payload[7:]
    with pytest.raises(mp2.MP2SemanticFormatError, match="invalid SD1M bit depth"):
        mp2.unpack_variant_semantic_or_none(corrupted, state)


def test_standard_q4_helpers_delegate_to_the_general_form():
    value = structured_state()["head.weight"]
    direct, restored_direct = sm3.standard_q4_payload("head.weight", value)
    general, restored_general = sm3.standard_qn_payload("head.weight", value, 4)
    assert direct == general
    assert torch.equal(restored_direct, restored_general)
