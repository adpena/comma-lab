"""Executable identity controls for the ddm_up3 carrier splice.

These are not "does it run" tests.  Each one is a falsifier of a specific claim the
byte-close rests on, and the two headline ones -- ``test_identity_control_*`` and
``test_roundtrip_*`` -- are exactly the controls ``ddm_up2``'s borrowed tool failed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO / "experiments" / "ddm_up3_carrier_splice.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("ddm_up3_carrier_splice", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


up3 = _load_module()

_RUNTIME = up3.DEFAULT_RUNTIME
_HAVE_BODY = (_RUNTIME / "archive.zip").is_file()
_HAVE_SOLVED = up3.DEFAULT_SOLVED.is_file()

needs_body = pytest.mark.skipif(
    not _HAVE_BODY, reason="pointer body is on the SSD custody tier"
)
needs_solved = pytest.mark.skipif(
    not _HAVE_SOLVED, reason="ddm_up2 solved codes are on the SSD custody tier"
)


# --------------------------------------------------------------------------
# Pure layout algebra -- runs anywhere, no custody volume needed.
# --------------------------------------------------------------------------


def test_packed_metadata_layout_reproduces_the_receivers_own_offsets():
    """Derived offsets must equal the literals in ``_restore_packed_cap1_metadata``."""
    layout = up3.packed_metadata_layout()
    base = up3.PACKED_METADATA_OFFSET
    assert base == 102
    assert base + layout["factor_base"][0] == 102
    assert base + layout["factors"][0] == 103
    assert base + layout["biases"][0] == 114
    assert base + layout["lengths"][0] == 123
    # The offset ddm_up2 reported as a body-specific literal.  It is the receiver's.
    assert base + layout["k_base"][0] == 139
    assert base + layout["ks"][0] == 140
    assert base + layout["ks"][1] == 142
    assert up3.PACKED_METADATA_BYTES == 40
    assert up3.CANONICAL_METADATA_BYTES - up3.PACKED_METADATA_BYTES == 40


@needs_body
@pytest.mark.parametrize(
    ("count", "bits"), [(12, 7), (12, 6), (32, 4), (12, 1), (12, 8), (5, 3)]
)
def test_pack_unsigned_inverts_the_receivers_unpack(count, bits):
    ra, _cr, _ar1, _cp = up3._import_runtime(_RUNTIME)
    rng = np.random.default_rng(20260819)
    values = rng.integers(0, 1 << bits, size=count)
    packed = up3._pack_unsigned(values.tolist(), count, bits)
    assert len(packed) == up3._field_bytes(count, bits)
    assert np.array_equal(ra._unpack_unsigned(packed, count, bits), values)


def test_pack_unsigned_refuses_a_value_that_escapes_the_field():
    with pytest.raises(up3.Up3Error):
        up3._pack_unsigned([2], 1, 1)


@pytest.mark.parametrize("length", [0, 1, 2, 3, 7, 8, 22187, 22186])
def test_ck2_interleave_roundtrips_at_every_parity(length):
    ra, _cr, _ar1, _cp = (
        up3._import_runtime(_RUNTIME) if _HAVE_BODY else (None, None, None, None)
    )
    if ra is None:
        pytest.skip("receiver not available")
    body = np.random.default_rng(7).integers(0, 256, size=length).astype(np.uint8).tobytes()
    assert ra._ck2_uninterleave_planes(up3._ck2_interleave_planes(body)) == body


# --------------------------------------------------------------------------
# Against the shipped body.
# --------------------------------------------------------------------------


@needs_body
def test_parsed_body_matches_the_pointer_receipt():
    body = up3.parse_shipped_body(_RUNTIME)
    assert body.archive_sha256 == up3.POINTER_ARCHIVE_SHA256
    assert len(body.archive_bytes) == up3.POINTER_ARCHIVE_BYTES
    assert body.ck2_carrier is True
    assert body.residual_bits == 78_065
    assert len(body.rice_payload) == 9_759
    assert body.ks.tolist() == [9, 9, 9, 8, 8, 9, 9, 9, 9, 9, 9, 9]
    assert int(body.ks.min()) == 8  # the true k_base, not 177
    assert body.codes.shape == (up3.N_PAIRS, up3.CARRIER_DIM)


@needs_body
def test_forward_packer_reproduces_the_shipped_packed_metadata():
    body = up3.parse_shipped_body(_RUNTIME)
    packed = up3.pack_cap1_metadata(
        factors=body.factors, biases=body.biases, lengths=body.lengths, ks=body.ks
    )
    assert packed == body.packed_metadata


@needs_body
def test_forward_ar1_inverts_restore_ar1_bias_on_the_shipped_codes():
    """The encoder-side AR(1) prediction must reproduce the stored residuals exactly."""
    body = up3.parse_shipped_body(_RUNTIME)
    _ra, cr, _ar1, _cp = up3._import_runtime(_RUNTIME)
    stored = cr._unzigzag(
        cr._rice_decode(
            body.ks.astype(np.int64).reshape(up3.CARRIER_DIM, 1),
            body.rice_payload,
            body.residual_bits,
            up3.N_PAIRS,
            up3.CARRIER_DIM,
        )
    )
    forward = up3.forward_ar1_bias(body.codes, body.factors, body.biases, _RUNTIME)
    assert np.array_equal(forward.astype(np.int64), stored.astype(np.int64))
    # And the decode direction returns the codes we started from.
    recovered = up3.decode_codes(
        body.rice_payload,
        residual_bits=body.residual_bits,
        ks=body.ks,
        factors=body.factors,
        biases=body.biases,
        runtime_dir=_RUNTIME,
    )
    assert np.array_equal(recovered, body.codes)


@needs_body
def test_forward_ar1_inverts_restore_on_random_lattice_points():
    """The inverse is exact for arbitrary int12 codes, not just the shipped ones."""
    body = up3.parse_shipped_body(_RUNTIME)
    _ra, _cr, _ar1, cp = up3._import_runtime(_RUNTIME)
    rng = np.random.default_rng(20260819)
    codes = rng.integers(-2048, 2048, size=(up3.N_PAIRS, up3.CARRIER_DIM)).astype(np.int32)
    residuals = up3.forward_ar1_bias(codes, body.factors, body.biases, _RUNTIME)
    model = cp.Ar1BiasModel(
        body.factors.astype(np.int16), body.biases.astype(np.int16)
    )
    assert np.array_equal(cp.restore_ar1_bias(residuals, model), codes)


@needs_body
def test_encoding_the_shipped_codes_reproduces_the_shipped_rice_payload():
    body = up3.parse_shipped_body(_RUNTIME)
    ks, payload, bits = up3.encode_codes(
        body.codes, factors=body.factors, biases=body.biases, runtime_dir=_RUNTIME
    )
    assert payload == body.rice_payload
    assert bits == body.residual_bits
    assert ks.tolist() == body.ks.tolist()


@needs_body
def test_identity_control_shipped_codes_rebuild_the_pointer_archive():
    """THE control ddm_up2's borrowed tool failed: shipped codes -> shipped bytes."""
    report = up3.control_identity(_RUNTIME)
    assert report["byte_identical"] is True
    assert report["observed_sha256"] == up3.POINTER_ARCHIVE_SHA256
    assert report["observed_bytes"] == up3.POINTER_ARCHIVE_BYTES
    assert report["packed_metadata_identical"] is True
    assert report["rice_payload_identical"] is True


@needs_body
@needs_solved
def test_roundtrip_solved_codes_survive_the_written_bytes_exactly():
    codes = np.load(up3.DEFAULT_SOLVED).astype(np.int32)
    report = up3.control_roundtrip(codes, _RUNTIME, container_search=True)
    assert report["codes_exact"] is True
    assert report["max_abs_code_delta"] == 0
    assert report["rice_bits"] == 78_072
    assert report["parsed_back_rice_payload_bytes"] == 9_759


@needs_body
@needs_solved
def test_container_search_reaches_zero_byte_delta_without_breaking_identity():
    codes = np.load(up3.DEFAULT_SOLVED).astype(np.int32)
    searched = up3.control_roundtrip(codes, _RUNTIME, container_search=True)
    plain = up3.control_roundtrip(codes, _RUNTIME, container_search=False)
    assert searched["delta_archive_bytes"] == 0
    assert plain["delta_archive_bytes"] == 48
    assert plain["container"]["identical_to_shipped_shape"] is True
    # The search must never cost bytes relative to the shipped shape.
    assert searched["delta_archive_bytes"] <= plain["delta_archive_bytes"]
    # And the shipped shape must still win its tie, so identity stays exact.
    assert up3.control_identity(_RUNTIME)["byte_identical"] is True


@needs_body
@needs_solved
def test_double_compile_is_deterministic():
    codes = np.load(up3.DEFAULT_SOLVED).astype(np.int32)
    report = up3.control_determinism(codes, _RUNTIME, container_search=True)
    assert report["identical"] is True
    assert report["first_sha256"] == report["second_sha256"]


@needs_body
def test_builder_fails_closed_when_rice_ks_escape_the_packed_window():
    """The packed k field is 1 bit over a u8 base; a wider span must REFUSE."""
    body = up3.parse_shipped_body(_RUNTIME)
    codes = body.codes.copy()
    # Blow up one dimension's residual magnitude so its Rice k leaves the window.
    codes[1::2, 0] = 2047
    codes[0::2, 0] = -2048
    with pytest.raises(up3.Up3Error, match="span more than the packed 1-bit field"):
        up3.build_archive(body, codes, runtime_dir=_RUNTIME)


@needs_body
def test_builder_refuses_codes_outside_the_int12_lattice():
    body = up3.parse_shipped_body(_RUNTIME)
    codes = body.codes.copy().astype(np.int32)
    codes[0, 0] = 5000
    with pytest.raises(up3.Up3Error, match="signed-int12"):
        up3.build_archive(body, codes, runtime_dir=_RUNTIME)


@needs_body
def test_builder_self_verification_refuses_bytes_it_cannot_parse_back():
    """build_archive must never return bytes it has not proven decode to ``codes``."""
    import dataclasses

    body = up3.parse_shipped_body(_RUNTIME)
    # Corrupt the AR(1) model the metadata will advertise, so the written section no
    # longer describes the payload the builder encoded.
    broken = dataclasses.replace(
        body, biases=(body.biases.astype(np.int16) + 1).astype(np.int8)
    )
    with pytest.raises(up3.Up3Error, match="does not parse back"):
        up3.build_archive(broken, body.codes, runtime_dir=_RUNTIME)
    # verify=False is the escape hatch, and it must still produce bytes.
    unchecked = up3.build_archive(
        broken, body.codes, runtime_dir=_RUNTIME, verify=False
    )
    assert unchecked["archive_size"] > 0


@needs_body
def test_builder_refuses_a_wrong_shaped_code_matrix():
    body = up3.parse_shipped_body(_RUNTIME)
    with pytest.raises(up3.Up3Error, match="codes must be"):
        up3.build_archive(body, np.zeros((10, 12), dtype=np.int32), runtime_dir=_RUNTIME)


# --------------------------------------------------------------------------
# The gate's score arithmetic (pure; no custody volume needed).
# --------------------------------------------------------------------------


def _load_gate():
    spec = importlib.util.spec_from_file_location(
        "ddm_up3_byteclose_gate", REPO / "experiments" / "ddm_up3_byteclose_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@needs_body
def test_gate_reproduces_the_pointer_score_from_its_legs():
    """The score function must rebuild the T4 receipt's S from its three legs."""
    gate = _load_gate()
    rebuilt = gate.score_from(7.77e-06, 0.00030309, 176_420)
    assert rebuilt == pytest.approx(gate.POINTER_SCORE, abs=1e-15)
    assert gate.rate_term(176_420) == pytest.approx(0.11747083650981346, abs=1e-15)


@needs_body
def test_gate_net_delta_clears_the_admit_bar_at_the_measured_pose():
    """The candidate's measured d_pose must clear the bar with rate and seg held."""
    gate = _load_gate()
    base = gate.score_from(7.77e-06, 0.00030309, 176_420)
    candidate = gate.score_from(7.649246787072966e-06, 0.00030309, 176_420)
    net = candidate - base
    assert net < gate.ADMIT_BAR
    assert net == pytest.approx(-6.876309991788766e-05, rel=1e-9)


@needs_body
def test_gate_rate_term_charges_a_byte_delta():
    """A +48 B container would cost real score; the arithmetic must show it."""
    gate = _load_gate()
    assert gate.rate_term(176_468) - gate.rate_term(176_420) == pytest.approx(
        25 * 48 / 37_545_489, rel=1e-12
    )
