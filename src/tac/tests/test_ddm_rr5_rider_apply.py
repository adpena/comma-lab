"""Tests for the ddm_rr5 CPR1 lossless rider.

The suite is split deliberately:

* the CODER tests are hermetic -- they run anywhere and cover the round-trip,
  the table reconstruction, the bitfield inverses and every refusal path;
* the BODY tests need the retained pointer body on the SSD tier and are skipped
  (never silently passed) when it is absent.

Two of these are the controls the charter names by hand: a decode-identity
control and a corrupted-input refusal.
"""

from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac import rr5_arith_basis as rider  # noqa: E402

POINTER_RUNTIME = Path("/Volumes/APDataStore/pact/ddm_up3/candidate_runtime")
POINTER_ARCHIVE = POINTER_RUNTIME / "archive.zip"
POINTER_SHA = "7ce46fd7a845d5987903a0d85a56581961eb7716a55c38a7361e3b5ecae94b5f"

requires_body = pytest.mark.skipif(
    not POINTER_ARCHIVE.exists(),
    reason="retained pointer body is not mounted on this host",
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "ddm_rr5_rider_apply", REPO / "tools/ddm_rr5_rider_apply.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["ddm_rr5_rider_apply"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def carrier_body(tool):
    if not POINTER_ARCHIVE.exists():
        pytest.skip("retained pointer body is not mounted")
    container = tool.parse_container(
        POINTER_ARCHIVE, POINTER_RUNTIME, expect_sha256=POINTER_SHA
    )
    return container.carrier_body


# --------------------------------------------------------------------------- #
# coder: hermetic
# --------------------------------------------------------------------------- #
def test_arith_round_trip_is_exact_on_a_synthetic_field():
    rng = np.random.default_rng(20260819)
    symbols = rng.integers(0, rider.BASIS_ALPHABET, size=rider.BASIS_SYMBOLS)
    payload, bits = rider.encode_basis_arith(symbols)
    assert np.array_equal(rider.decode_basis_arith(payload, bits), symbols)


def test_arith_round_trip_survives_a_degenerate_single_symbol_field():
    symbols = np.zeros(rider.BASIS_SYMBOLS, dtype=np.int64)
    symbols[0] = 1  # two distinct symbols is the codec's stated minimum
    payload, bits = rider.encode_basis_arith(symbols)
    assert np.array_equal(rider.decode_basis_arith(payload, bits), symbols)


def test_arith_beats_the_incumbent_on_a_skewed_per_atom_field():
    """The context is the atom index, so per-atom skew is what it must exploit."""
    rng = np.random.default_rng(7)
    blocks = []
    for atom in range(rider.CARRIER_DIM):
        weights = np.exp(-np.abs(np.arange(rider.BASIS_ALPHABET) - atom) / 1.5)
        blocks.append(
            rng.choice(
                rider.BASIS_ALPHABET,
                size=rider.BASIS_SYMBOLS // rider.CARRIER_DIM,
                p=weights / weights.sum(),
            )
        )
    symbols = np.concatenate(blocks)
    payload, _ = rider.encode_basis_arith(symbols)
    lengths = rider.huffman_lengths_from_histogram(
        np.bincount(symbols, minlength=rider.BASIS_ALPHABET)
    )
    huffman, _ = rider.huffman_encode(symbols, lengths)
    assert len(payload) < len(huffman)


def test_encode_refuses_a_wrong_symbol_count():
    with pytest.raises(rider.RiderError, match="27,648"):
        rider.encode_basis_arith(np.zeros(10, dtype=np.int64))


def test_encode_refuses_a_symbol_outside_the_five_bit_alphabet():
    symbols = np.zeros(rider.BASIS_SYMBOLS, dtype=np.int64)
    symbols[5] = rider.BASIS_ALPHABET
    with pytest.raises(rider.RiderError, match="alphabet"):
        rider.encode_basis_arith(symbols)


def test_huffman_round_trip_is_exact():
    rng = np.random.default_rng(11)
    symbols = rng.integers(0, 8, size=4096)
    lengths = rider.huffman_lengths_from_histogram(
        np.bincount(symbols, minlength=rider.BASIS_ALPHABET)
    )
    payload, bits = rider.huffman_encode(symbols, lengths)
    assert np.array_equal(
        rider.huffman_decode(lengths, payload, bits, symbols.size), symbols
    )


def test_canonical_codes_refuse_an_oversubscribed_table():
    lengths = np.zeros(rider.BASIS_ALPHABET, dtype=np.uint8)
    lengths[:4] = 1  # four symbols cannot share length-1 codes
    with pytest.raises(rider.RiderError, match="oversubscribed"):
        rider.canonical_codes(lengths)


def test_canonical_codes_refuse_an_incomplete_table():
    lengths = np.zeros(rider.BASIS_ALPHABET, dtype=np.uint8)
    lengths[0] = 1
    lengths[1] = 2  # Kraft sum < 1
    with pytest.raises(rider.RiderError, match="incomplete"):
        rider.canonical_codes(lengths)


def test_huffman_lengths_reject_a_degenerate_alphabet():
    histogram = np.zeros(rider.BASIS_ALPHABET, dtype=np.int64)
    histogram[3] = 100
    with pytest.raises(rider.RiderError, match="at least two symbols"):
        rider.huffman_lengths_from_histogram(histogram)


@pytest.mark.parametrize(("count", "bits"), [(32, 4), (12, 7), (12, 6), (12, 1)])
def test_pack_and_unpack_unsigned_are_inverse(count, bits):
    rng = np.random.default_rng(count * bits)
    values = rng.integers(0, 1 << bits, size=count)
    assert np.array_equal(
        rider.unpack_unsigned(rider.pack_unsigned(values, bits), count, bits), values
    )


def test_pack_unsigned_refuses_an_overflowing_value():
    with pytest.raises(rider.RiderError, match="4-bit"):
        rider.pack_unsigned(np.array([16]), 4)


def test_packed_lengths_round_trip_through_the_metadata_block():
    rng = np.random.default_rng(3)
    metadata = bytearray(rng.integers(0, 256, size=rider.PACKED_METADATA_BYTES).tolist())
    lengths = rng.integers(0, 16, size=rider.BASIS_ALPHABET)
    updated = rider.with_packed_lengths(bytes(metadata), lengths)
    assert len(updated) == rider.PACKED_METADATA_BYTES
    assert np.array_equal(rider.packed_lengths(bytes(updated)), lengths)


def test_with_packed_lengths_touches_only_the_lengths_span():
    metadata = bytes(range(rider.PACKED_METADATA_BYTES))
    updated = rider.with_packed_lengths(
        metadata, np.zeros(rider.BASIS_ALPHABET, dtype=np.int64)
    )
    lo, hi = rider.PACKED_LENGTHS_SPAN
    start, end = lo - rider.PACKED_METADATA_OFFSET, hi - rider.PACKED_METADATA_OFFSET
    assert bytes(updated[:start]) == metadata[:start]
    assert bytes(updated[end:]) == metadata[end:]


def test_split_carrier_body_refuses_a_truncated_body():
    with pytest.raises(rider.RiderError, match="shorter than"):
        rider.split_carrier_body(b"\x00" * 10)


def test_split_carrier_body_refuses_zero_bit_counts():
    with pytest.raises(rider.RiderError, match="nonzero"):
        rider.split_carrier_body(b"\x00" * 400)


# --------------------------------------------------------------------------- #
# body: needs the retained pointer archive
# --------------------------------------------------------------------------- #
@requires_body
def test_split_and_assemble_carrier_body_are_inverse(carrier_body):
    assert rider.assemble_carrier_body(rider.split_carrier_body(carrier_body)) == (
        carrier_body
    )


@requires_body
def test_shipped_huffman_table_reconstructs_exactly_from_the_histogram(carrier_body):
    """The 16 B packed table may only be dropped because it is derivable."""
    fields = rider.split_carrier_body(carrier_body)
    lengths = rider.packed_lengths(bytes(fields["metadata"]))
    symbols = rider.huffman_decode(
        lengths, bytes(fields["basis"]), int(fields["basis_bits"]), rider.BASIS_SYMBOLS
    )
    rebuilt = rider.huffman_lengths_from_histogram(
        np.bincount(symbols, minlength=rider.BASIS_ALPHABET)
    )
    assert np.array_equal(rebuilt, lengths)
    replay, replay_bits = rider.huffman_encode(symbols, rebuilt)
    assert replay == bytes(fields["basis"])
    assert replay_bits == int(fields["basis_bits"])


@requires_body
def test_rider_decode_identity_restores_the_shipped_body_byte_for_byte(carrier_body):
    """THE decode-identity control: the rider is lossless or this fails."""
    applied = rider.apply_rider_to_carrier_body(carrier_body)
    assert rider.restore_carrier_body(bytes(applied["body"])) == carrier_body


@requires_body
def test_rider_actually_shrinks_the_basis_stream(carrier_body):
    applied = rider.apply_rider_to_carrier_body(carrier_body)
    assert applied["rider_basis_bytes"] < applied["shipped_basis_bytes"]
    assert applied["table_dropped"] is True


@requires_body
def test_rider_is_deterministic(carrier_body):
    first = rider.apply_rider_to_carrier_body(carrier_body)["body"]
    second = rider.apply_rider_to_carrier_body(carrier_body)["body"]
    assert bytes(first) == bytes(second)


@requires_body
def test_apply_refuses_an_archive_whose_sha_does_not_match(tool, tmp_path):
    """Corrupted-input refusal: one flipped byte must stop the tool."""
    corrupted = tmp_path / "corrupt.zip"
    payload = bytearray(POINTER_ARCHIVE.read_bytes())
    payload[-1] ^= 0xFF
    corrupted.write_bytes(bytes(payload))
    with pytest.raises(tool.RiderApplyError, match="sha256"):
        tool.parse_container(corrupted, POINTER_RUNTIME, expect_sha256=POINTER_SHA)


@requires_body
def test_apply_refuses_a_member_that_is_not_an_rx1m_container(tool, tmp_path):
    bogus = tmp_path / "bogus.zip"
    with zipfile.ZipFile(bogus, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("p", b"NOTRX1M" + b"\x00" * 4096)
    with pytest.raises(tool.RiderApplyError, match="RX1M"):
        tool.parse_container(bogus, POINTER_RUNTIME, expect_sha256=None)


@requires_body
def test_apply_refuses_an_archive_with_the_wrong_member_name(tool, tmp_path):
    bogus = tmp_path / "wrongname.zip"
    with zipfile.ZipFile(bogus, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("q", b"\x00" * 64)
    with pytest.raises(tool.RiderApplyError, match="member p"):
        tool.parse_container(bogus, POINTER_RUNTIME, expect_sha256=None)


@requires_body
def test_identity_control_reproduces_the_input_archive_exactly(tool):
    """Without this the measured delta could be a container artefact."""
    container = tool.parse_container(
        POINTER_ARCHIVE, POINTER_RUNTIME, expect_sha256=POINTER_SHA
    )
    identity = tool.identity_control(container)
    assert identity["byte_identical"] is True
    assert identity["archive_sha256"] == POINTER_SHA


@requires_body
def test_ck2_interleave_is_the_exact_inverse_of_the_receiver_uninterleave(tool):
    container = tool.parse_container(
        POINTER_ARCHIVE, POINTER_RUNTIME, expect_sha256=POINTER_SHA
    )
    ra = container.residual_archive
    body = container.carrier_body
    assert ra._ck2_uninterleave_planes(tool._ck2_interleave(body)) == body


@requires_body
def test_container_search_space_is_the_sealed_up3_space(tool):
    from tac.win_families.container_optimizer import UP3_DECLARED_OPTIONS

    assert len(tool.CONTAINER_OPTIONS) == len(UP3_DECLARED_OPTIONS)
    assert tool.CONTAINER_OPTIONS[0] == (
        UP3_DECLARED_OPTIONS[0].interleave,
        UP3_DECLARED_OPTIONS[0].brotli_quality,
        UP3_DECLARED_OPTIONS[0].brotli_lgwin,
    )
    assert len(tool.CONTAINER_SPACE_SEAL) == 64


@requires_body
def test_compressed_models_is_inert_in_the_shipped_runtime(tool):
    """The one field excluded from decode identity must have no consumer."""
    report = tool.assert_field_is_inert(POINTER_RUNTIME, ("compressed_models",))
    assert report["all_inert"] is True
    assert report["per_field"]["compressed_models"]["read_sites"] == []


@requires_body
def test_inertness_check_detects_a_planted_read_site(tool, tmp_path):
    """The inertness gate must be able to FAIL, or it proves nothing."""
    tree = tmp_path / "tree"
    (tree / "runtime").mkdir(parents=True)
    (tree / "runtime" / "consumer.py").write_text(
        "def use(parts):\n    return parts.compressed_models\n"
    )
    report = tool.assert_field_is_inert(tree, ("compressed_models",))
    assert report["all_inert"] is False


@requires_body
def test_rider_receipt_records_every_control_and_the_realized_delta(tool, tmp_path):
    receipt = tool.apply_rider(
        POINTER_ARCHIVE,
        POINTER_RUNTIME,
        tmp_path / "out",
        expect_sha256=POINTER_SHA,
        full=False,
    )
    assert receipt["controls"]["C3_receiver_decode_identity"] == "PASS"
    assert receipt["realized"]["archive_delta_bytes"] > 0
    assert receipt["score_claim"] is False
    assert receipt["output"]["archive_bytes"] < receipt["input"]["archive_bytes"]
    delta = receipt["realized"]["archive_delta_bytes"]
    assert receipt["realized"]["delta_S"] == pytest.approx(-delta * tool.S_PER_BYTE)


@requires_body
def test_rider_runtime_tree_carries_a_byte_identical_coder_module(tool, tmp_path):
    receipt = tool.apply_rider(
        POINTER_ARCHIVE,
        POINTER_RUNTIME,
        tmp_path / "out",
        expect_sha256=POINTER_SHA,
        full=False,
    )
    emitted = Path(receipt["output"]["runtime_dir"]) / "runtime/rr5_arith_basis.py"
    source = REPO / "src/tac/rr5_arith_basis.py"
    assert emitted.read_bytes() == source.read_bytes()


@requires_body
def test_rider_archive_pins_its_own_sha_in_the_emitted_inflate(tool, tmp_path):
    receipt = tool.apply_rider(
        POINTER_ARCHIVE,
        POINTER_RUNTIME,
        tmp_path / "out",
        expect_sha256=POINTER_SHA,
        full=False,
    )
    inflate = (Path(receipt["output"]["runtime_dir"]) / "inflate.py").read_text()
    assert receipt["output"]["archive_sha256"] in inflate
    assert str(receipt["output"]["archive_bytes"]) in inflate.replace("_", "")


@requires_body
def test_unpatched_receiver_refuses_the_rider_archive(tool, tmp_path):
    """The reserved bit must be FAIL-CLOSED against an old receiver.

    An unpatched tree masks reserved bits above 0x07, so it must REFUSE rather
    than parse the rider body as if the basis were still Huffman-coded.
    """
    receipt = tool.apply_rider(
        POINTER_ARCHIVE,
        POINTER_RUNTIME,
        tmp_path / "out",
        expect_sha256=POINTER_SHA,
        full=False,
    )
    rider_archive = Path(receipt["output"]["archive_path"])
    ra = tool._load_receiver(POINTER_RUNTIME)  # the ORIGINAL, unpatched receiver
    with pytest.raises(Exception):  # noqa: B017 - any refusal is the pass
        ra.read_residual_archive(rider_archive)


@requires_body
def test_patched_receiver_carries_the_rider_branch_and_widened_mask(tool, tmp_path):
    receipt = tool.apply_rider(
        POINTER_ARCHIVE,
        POINTER_RUNTIME,
        tmp_path / "out",
        expect_sha256=POINTER_SHA,
        full=False,
    )
    text = (
        Path(receipt["output"]["runtime_dir"]) / "runtime/residual_archive.py"
    ).read_text()
    assert "SZ1_RESERVED_KNOWN_BITS = 0x0F" in text
    assert "RR5_RESERVED_ARITH_BASIS = 0x08" in text
    assert "restore_carrier_body(carrier_body)" in text


@requires_body
def test_emit_rider_runtime_refuses_a_tree_whose_anchor_is_missing(tool, tmp_path):
    """The receiver patch must fail closed if the upstream file moves under it."""
    tree = tmp_path / "tree"
    (tree / "runtime").mkdir(parents=True)
    (tree / "runtime" / "residual_archive.py").write_text("# no anchors here\n")
    (tree / "inflate.py").write_text('ARCHIVE_SHA256 = "x"\nARCHIVE_BYTES = 1\n')
    with pytest.raises(tool.RiderApplyError, match="anchor"):
        tool.emit_rider_runtime(tree, tmp_path / "dest", b"payload")


def test_assemble_refuses_a_wrong_length_scales_block():
    fields = {
        "basis_bits": 8,
        "residual_bits": 8,
        "scales": b"\x00" * 10,
        "metadata": bytes(rider.PACKED_METADATA_BYTES),
        "basis": b"\x00",
        "rice": b"\x00",
        "body_tail": b"",
    }
    with pytest.raises(rider.RiderError, match="96 bytes"):
        rider.assemble_carrier_body(fields)


def test_assemble_refuses_a_bit_count_that_overflows_the_u24_field():
    fields = {
        "basis_bits": 1 << 24,
        "residual_bits": 8,
        "scales": bytes(rider.SCALES_BYTES),
        "metadata": bytes(rider.PACKED_METADATA_BYTES),
        "basis": bytes((1 << 24) // 8),
        "rice": b"\x00",
        "body_tail": b"",
    }
    with pytest.raises(rider.RiderError, match="u24"):
        rider.assemble_carrier_body(fields)
