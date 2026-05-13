"""Smoke tests for the 6 modules rehydrated 2026-05-05.

Each rehydrated module ships its public API surface (constants, error classes,
dataclasses, function signatures). Some modules still defer bytecode-damaged
functions to ``NotImplementedError`` because pycdc could not fully decompile
intricate closures, generators, or masked-conv mixins. This test file pins the
expected public API, validates the fully reconstructed ``pr85_bundle`` runtime
roundtrips, and keeps the now-live ``endgame_archive_decision`` CLI fail-closed
on malformed archive input.

Recovery spec source:
``.recovery_quarantine_20260505T004735Z/src/tac/<module>.recovery_spec.json``.
"""
from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Section 1: import-resolves tests (every required symbol must be importable)
# ---------------------------------------------------------------------------


REQUIRED_SYMBOLS = {
    "tac.pr85_bundle": [
        "Pr85BundleError",
        "Pr85Bundle",
        "Pr85RuntimeExpansion",
        "Pr85SegmentContract",
        "SEGMENT_ORDER",
        "HEADER_V5_SEGMENTS",
        "HEADER_EXPLICIT30_SEGMENTS",
        "FIXED_V5_LENGTHS",
        "PR85_HEADERLESS_RANDMULTI_SPECS",
        "QPOST_MAGIC",
        "HPM1_MAGIC",
        "HPM1_HEADER_BYTES",
        "QPOST_STREAM_NAMES",
        "decode_rmb1_randmulti_payload",
        "decode_pr85_randmulti_to_headerless_rows",
        "compare_pr85_randmulti_decoded_rows",
        "pack_pr85_bundle",
        "parse_pr85_bundle",
        "parse_hpm1_mask_segment",
        "validate_pr85_member_name",
        "infer_pr85_segment_contract",
        "transcode_pr85_randmulti_to_qrm1",
        "build_pr85_qpost_bin",
        "decode_pr85_p1d1_pose_to_fp16",
        "expand_pr85_bundle_to_runtime_members",
    ],
    "tac.archive_byte_profile": [
        "ArchiveByteProfileError",
        "SCHEMA",
        "ARCHIVE_SCHEMA",
        "TOOL",
        "EVIDENCE_GRADE",
        "CONTEST_ORIGINAL_BYTES",
        "RATE_TERM_COEFFICIENT",
        "contest_rate_term",
        "main",
    ],
    "tac.stbm1br_mask_codec": [
        "STBM1BR_MAGIC",
        "QTBM_MAGICS",
        "N_CLASSES",
        "N_SYM",
        "DEFAULT_SHAPE",
        "STBM1BRError",
        "STBM1BRMetadata",
        "sha256_bytes",
        "parse_stbm1br_metadata",
        "metadata_as_dict",
        "decode_stbm1br_mask_segment",
        "decode_stbm1br_mask_file",
        "_RangeDecoder",
    ],
    "tac.endgame_archive_decision": [
        "EndgameArchiveDecisionError",
        "SCHEMA",
        "TOOL",
        "EVIDENCE_GRADE",
        "STBM1BR_MAGIC",
        "RMB1_MAGIC",
        "RSB1_MAGIC",
        "QMA9_HEADER_BYTES",
        "main",
        "build_arg_parser",
        "build_endgame_decision_profile",
    ],
    "tac.pr86_hpac_codec": [
        "Pr86HpacReplayError",
        "Pr86ArchiveContract",
        "Pr86ArchiveBundle",
        "HpacProbabilityVariant",
        "HPAC_PROBABILITY_VARIANTS",
        "DEFAULT_PR86_ARCHIVE",
        "DEFAULT_HPAC_PROBABILITY_VARIANT",
        "EXPECTED_PR86_ARCHIVE_BYTES",
        "EXPECTED_PR86_ARCHIVE_SHA256",
        "EXPECTED_PR86_TOKENS_SHA256",
        "EXPECTED_PR86_MEMBERS",
        "EXPECTED_PR86_MEMBER_BYTES",
        "RECORDED_PR86_DEPENDENCIES",
        "NUM_CLASSES",
        "supported_hpac_probability_variant_names",
        "resolve_hpac_probability_variant",
        "default_source_artifact_paths",
        "sha256_bytes",
        "sha256_path",
        "repo_rel",
    ],
    "tac.pr91_hpm1_codec": [
        "Pr91Hpm1Error",
        "Hpm1MaskPayload",
        "DEFAULT_PR91_ARCHIVE",
        "DEFAULT_PR91_INTAKE_DIR",
        "EXPECTED_PR91_ARCHIVE_BYTES",
        "EXPECTED_PR91_ARCHIVE_SHA256",
        "EXPECTED_PR91_HPM1_MASK_BYTES",
        "EXPECTED_PR91_HPM1_MASK_SHA256",
        "EXPECTED_PR91_HPM1_TOKENS_SHA256",
        "EXPECTED_PR91_HPM1_HPAC_SHA256",
        "PR91_HPM1_CONTEXT_MODES",
        "DEFAULT_PR91_HPM1_CONTEXT_WINDOWS",
        "split_hpm1_mask_segment",
        "extract_pr91_hpm1_payload",
        "run_pr91_hpm1_probability_variant_matrix",
        "run_pr91_hpm1_preflight",
        "validate_hpm1_static_contract",
    ],
    "tac.quantizr_torch_fp4_codec": [
        "TORCH_FP4_FORMAT",
        "PROTECTED_MODULES",
        "is_torch_fp4_payload",
        "encode_torch_fp4_payload",
        "encode_torch_fp4_state_dict",
        "decode_torch_fp4_payload",
        "load_torch_fp4_payload",
        "load_torch_fp4_bytes",
    ],
    "tac.qh0_renderer_codec": [
        "QH0_MAGIC",
        "QM0_MAGIC",
        "QH1_MAGIC",
        "QH0_SUPPORTED_MAGICS",
        "QH0CodecError",
        "QH0DecodeReport",
        "FP4_POS_LEVELS",
        "unpack_nibbles",
        "reconstruct_qh1_payload",
        "decode_qh0_state_dict",
        "load_qh0",
    ],
    "tac.qh0_record_serializer": [
        "QH0Record",
        "QH0RecordSet",
        "QH0SerializedVariant",
        "QH0SerializerError",
        "sha256_bytes",
        "unsplit_even_odd_bytes",
        "split_even_odd_bytes",
        "unpack_hilo_fp4_bytes",
        "pack_hilo_fp4_bytes",
        "parse_qh0_record_set",
        "serialize_records",
        "build_serialized_variants",
    ],
    "tac.public_frontier_intake": [
        "PublicFrontierIntakeError",
        "SCHEMA",
        "TOOL",
        "EVIDENCE_GRADE",
        "main",
        "build_arg_parser",
        "profile_public_frontier_archive",
    ],
    "tac.henosis_pr82_transfer": [
        "HenosisPr82TransferError",
        "Pr82ReplayContract",
        "Pr82Bundle",
        "Pr82RandmultiGroup",
        "sha256_bytes",
        "sha256_path",
        "brotli_decompress_segment",
        "parse_replay_contract",
        "parse_pr82_bundle",
        "decode_randmulti_groups",
        "encode_randmulti_qrm1",
        "decode_randmulti_qrm1",
    ],
}


@pytest.mark.parametrize("module_name,required_symbols", list(REQUIRED_SYMBOLS.items()))
def test_module_exposes_required_symbols(module_name: str, required_symbols: list[str]) -> None:
    """Every rehydrated module must expose its full public API surface."""
    module = importlib.import_module(module_name)
    missing = [sym for sym in required_symbols if not hasattr(module, sym)]
    assert not missing, f"{module_name} missing symbols: {missing}"


# ---------------------------------------------------------------------------
# Section 2: pr85_bundle full-reconstruction smoke tests (the one
# fully-reconstructed module).
# ---------------------------------------------------------------------------


def test_pr85_bundle_segment_order_constants() -> None:
    from tac.pr85_bundle import (
        HEADER_EXPLICIT30_SEGMENTS,
        HEADER_V5_SEGMENTS,
        SEGMENT_ORDER,
        FIXED_V5_LENGTHS,
        QPOST_STREAM_NAMES,
    )

    assert SEGMENT_ORDER == (
        "mask",
        "model",
        "pose",
        "post",
        "shift",
        "frac",
        "frac2",
        "frac3",
        "bias",
        "region",
        "randmulti",
    )
    assert HEADER_V5_SEGMENTS == SEGMENT_ORDER[:8]
    assert HEADER_EXPLICIT30_SEGMENTS == SEGMENT_ORDER[:10]
    assert FIXED_V5_LENGTHS == {"bias": 223, "region": 273}
    assert QPOST_STREAM_NAMES == (
        "post",
        "shift",
        "frac",
        "frac2",
        "frac3",
        "bias",
        "region",
        "randmulti",
    )


def test_pr85_bundle_v5_pack_parse_roundtrip() -> None:
    from tac.pr85_bundle import (
        SEGMENT_ORDER,
        pack_pr85_bundle,
        parse_pr85_bundle,
    )

    segs = {n: bytes(range(8)) for n in SEGMENT_ORDER}
    segs["bias"] = b"\x00" * 223
    segs["region"] = b"\x00" * 273
    segs["mask"] = b"QMA9" + b"\x00" * 100
    segs["randmulti"] = b"\x01" + b"\x00" * 8
    raw = pack_pr85_bundle(segs, header_mode="v5")
    bundle = parse_pr85_bundle(raw)
    assert bundle.format == "pr85_v5_8byte_lengths"
    assert bundle.header_bytes == 24
    for name in SEGMENT_ORDER:
        assert bytes(bundle.segments[name]) == segs[name], f"roundtrip mismatch: {name}"


def test_pr85_bundle_v5_rejects_wrong_fixed_length() -> None:
    from tac.pr85_bundle import (
        SEGMENT_ORDER,
        Pr85BundleError,
        pack_pr85_bundle,
    )

    segs = {n: bytes(range(8)) for n in SEGMENT_ORDER}
    segs["bias"] = b"\x00" * 222  # Wrong! should be 223
    segs["region"] = b"\x00" * 273
    with pytest.raises(Pr85BundleError, match="v5 fixed-length segment"):
        pack_pr85_bundle(segs, header_mode="v5")


def test_pr85_bundle_pack_rejects_unknown_header_mode() -> None:
    from tac.pr85_bundle import (
        SEGMENT_ORDER,
        Pr85BundleError,
        pack_pr85_bundle,
    )

    segs = {n: bytes(range(8)) for n in SEGMENT_ORDER}
    segs["bias"] = b"\x00" * 223
    segs["region"] = b"\x00" * 273
    with pytest.raises(Pr85BundleError, match="unknown PR85 header_mode"):
        pack_pr85_bundle(segs, header_mode="invalid_mode")


def test_pr85_bundle_pack_rejects_missing_segment() -> None:
    from tac.pr85_bundle import Pr85BundleError, pack_pr85_bundle

    with pytest.raises(Pr85BundleError, match="missing PR85 segment"):
        pack_pr85_bundle({"mask": b"QMA9"}, header_mode="v5")


def test_validate_pr85_member_name() -> None:
    from tac.pr85_bundle import Pr85BundleError, validate_pr85_member_name

    assert validate_pr85_member_name("x") == "x"
    for bad in ("y", "../x", "/x", "x/y", ""):
        with pytest.raises(Pr85BundleError):
            validate_pr85_member_name(bad)


def test_u24le_helpers() -> None:
    from tac.pr85_bundle import _pack_u24le, _u24le, Pr85BundleError

    for value in (0, 1, 255, 256, 65535, 65536, 16777215):
        packed = _pack_u24le(value)
        assert len(packed) == 3
        assert _u24le(packed, 0) == value
    with pytest.raises(Pr85BundleError):
        _pack_u24le(-1)
    with pytest.raises(Pr85BundleError):
        _pack_u24le(16777216)


# ---------------------------------------------------------------------------
# Section 3: smoke tests that confirm deferred functions FAIL LOUD (not silent).
# ---------------------------------------------------------------------------


def test_stbm1br_decode_raises_not_implemented() -> None:
    """Per acceptance criteria: deferred functions must fail loud, not silent."""
    from tac.stbm1br_mask_codec import (
        STBM1BR_MAGIC,
        decode_stbm1br_mask_segment,
    )

    payload = STBM1BR_MAGIC + b"fakebytes"
    with pytest.raises(NotImplementedError, match="rehydration incomplete"):
        decode_stbm1br_mask_segment(payload)


def test_pr86_hpac_encode_raises_not_implemented() -> None:
    from tac.pr86_hpac_codec import encode_tokens_hpac

    with pytest.raises(NotImplementedError, match="rehydration incomplete"):
        encode_tokens_hpac()


def test_pr91_hpm1_probability_matrix_fails_closed_on_missing_archive(tmp_path) -> None:
    from tac.pr91_hpm1_codec import Pr91Hpm1Error, run_pr91_hpm1_probability_variant_matrix

    with pytest.raises(Pr91Hpm1Error, match="archive_missing"):
        run_pr91_hpm1_probability_variant_matrix(tmp_path / "fake.zip")


def test_endgame_archive_decision_main_fails_closed_on_invalid_zip(tmp_path) -> None:
    from tac.endgame_archive_decision import main

    fake = tmp_path / "ref.zip"
    fake.write_bytes(b"fake")
    with pytest.raises(SystemExit):
        main(["--reference", str(fake)])


# ---------------------------------------------------------------------------
# Section 4: HPAC probability variant registry (real reconstruction).
# ---------------------------------------------------------------------------


def test_hpac_probability_variants_registry() -> None:
    from tac.pr86_hpac_codec import (
        DEFAULT_HPAC_PROBABILITY_VARIANT,
        Pr86HpacReplayError,
        resolve_hpac_probability_variant,
        supported_hpac_probability_variant_names,
    )

    names = supported_hpac_probability_variant_names()
    assert isinstance(names, tuple) and len(names) == 4
    assert DEFAULT_HPAC_PROBABILITY_VARIANT in names

    for name in names:
        variant = resolve_hpac_probability_variant(name)
        assert variant.name == name
        assert variant.probability_dtype in ("float32", "float64")

    # Reject unknown variants
    with pytest.raises(Pr86HpacReplayError):
        resolve_hpac_probability_variant("bogus_variant")


# ---------------------------------------------------------------------------
# Section 5: contest rate term math is exact.
# ---------------------------------------------------------------------------


def test_contest_rate_term_exact() -> None:
    from tac.archive_byte_profile import (
        CONTEST_ORIGINAL_BYTES,
        contest_rate_term,
    )

    # Exact contest formula: 25 * bytes / 37545489
    assert CONTEST_ORIGINAL_BYTES == 37545489
    assert contest_rate_term(0) == 0.0
    assert contest_rate_term(CONTEST_ORIGINAL_BYTES) == 25.0
    assert contest_rate_term(686635) == pytest.approx(
        25 * 686635 / 37545489, rel=1e-9
    )


# ---------------------------------------------------------------------------
# Section 6: STBM1BR metadata parser smoke (does not require rust bridge).
# ---------------------------------------------------------------------------


def test_stbm1br_parse_metadata_rejects_bad_magic() -> None:
    from tac.stbm1br_mask_codec import STBM1BRError, parse_stbm1br_metadata

    with pytest.raises(STBM1BRError, match="does not start with magic"):
        parse_stbm1br_metadata(b"NOTMAGIC" + b"\x00" * 100)


# ---------------------------------------------------------------------------
# Section 7: PR91 HPM1 mask segment splitter (real reconstruction).
# ---------------------------------------------------------------------------


def test_quantizr_torch_fp4_payload_detection() -> None:
    from tac.quantizr_torch_fp4_codec import (
        TORCH_FP4_FORMAT,
        is_torch_fp4_payload,
    )

    assert is_torch_fp4_payload(
        {
            "__format__": f"{TORCH_FP4_FORMAT}_v1",
            "quantized": {},
            "dense_fp16": {},
        }
    )
    assert not is_torch_fp4_payload(
        {"__format__": "other_format", "quantized": {}, "dense_fp16": {}}
    )
    assert not is_torch_fp4_payload({"foo": "bar"})
    assert not is_torch_fp4_payload([])
    assert not is_torch_fp4_payload(None)
    assert not is_torch_fp4_payload(b"raw bytes")


def test_qh0_unpack_nibbles_roundtrip() -> None:
    import torch

    from tac.qh0_renderer_codec import unpack_nibbles

    packed = torch.tensor([0x12, 0x34, 0xAB, 0xCD], dtype=torch.uint8)
    unpacked = unpack_nibbles(packed, count=8)
    assert unpacked.tolist() == [0x1, 0x2, 0x3, 0x4, 0xA, 0xB, 0xC, 0xD]
    # Truncate to fewer
    unpacked = unpack_nibbles(packed, count=3)
    assert unpacked.tolist() == [0x1, 0x2, 0x3]


def test_qh0_record_serializer_byte_split_roundtrip() -> None:
    from tac.qh0_record_serializer import (
        pack_hilo_fp4_bytes,
        split_even_odd_bytes,
        unpack_hilo_fp4_bytes,
        unsplit_even_odd_bytes,
    )

    data = bytes(range(64))
    # Even/odd split is its own inverse only after split→unsplit
    assert unsplit_even_odd_bytes(split_even_odd_bytes(data)) == data
    # Hi/lo nibble split is its own inverse
    fp4 = bytes(range(0x10, 0x50))
    assert unpack_hilo_fp4_bytes(pack_hilo_fp4_bytes(fp4), len(fp4)) == fp4


def test_henosis_pr82_vlq_roundtrip() -> None:
    from tac.henosis_pr82_transfer import _read_vlq, _write_vlq

    for value in (0, 1, 127, 128, 255, 16383, 16384, 1 << 32):
        encoded = _write_vlq(value)
        decoded, cursor = _read_vlq(encoded, 0)
        assert decoded == value, f"VLQ roundtrip failed for {value}"
        assert cursor == len(encoded)


def test_split_hpm1_mask_segment_structure() -> None:
    """Build a synthetic HPM1 segment and verify the split honors the header."""
    import struct

    from tac.pr85_bundle import HPM1_HEADER_BYTES, HPM1_MAGIC
    from tac.pr91_hpm1_codec import Hpm1MaskPayload, split_hpm1_mask_segment

    # 11 uint32 fields after the 4-byte magic
    header_fields = (600, 384, 512, 5, 0, 1, 0, 0, 100, 200, 4)
    header = HPM1_MAGIC + struct.pack("<11I", *header_fields)
    assert len(header) == HPM1_HEADER_BYTES, f"got {len(header)} != {HPM1_HEADER_BYTES}"
    tokens = bytes(range(100))
    hpac = bytes(range(200))
    segment = header + tokens + hpac

    payload = split_hpm1_mask_segment(segment)
    assert isinstance(payload, Hpm1MaskPayload)
    assert payload.n_frames == 600
    assert payload.height == 384
    assert payload.width == 512
    assert payload.tokens_len == 100
    assert payload.hpac_len == 200
    assert payload.tokens == tokens
    assert payload.hpac == hpac
    config = payload.config()
    assert config["n_frames"] == 600
    assert "tokens" not in config  # config only returns header fields
