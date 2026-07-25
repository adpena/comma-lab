# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import tac.optimization.arith_selfcomp_rate_coders as coders
from tools.measure_arith_selfcomp_rate_coders import MeasurementError, _settled_seg_secant_curve


@pytest.fixture
def spatial_i8() -> np.ndarray:
    return np.array(
        [
            [
                [[0, 1], [1, -1], [2, -2], [3, -3]],
                [[0, 1], [1, -1], [4, -4], [127, -128]],
            ]
        ],
        dtype=np.int8,
    )


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64])
def test_exact_roundtrip_signed_extremes_zeros_and_determinism(dtype: np.dtype) -> None:
    info = np.iinfo(dtype)
    array = np.array([0, 1, -1, info.max, info.min, 0], dtype=dtype).reshape(1, 2, 3, 1)
    pairs = (
        (coders.encode_iid_arithmetic, coders.decode_iid_arithmetic),
        (coders.encode_spatial_context_arithmetic, coders.decode_spatial_context_arithmetic),
        (coders.encode_zigzag_rle_arithmetic, coders.decode_zigzag_rle_arithmetic),
    )
    for encode, decode in pairs:
        first = encode(array)
        assert first == encode(array)
        restored = decode(first)
        assert restored.dtype == array.dtype
        np.testing.assert_array_equal(restored, array)
    raw = coders.serialize_signed_array(array)
    assert raw == coders.serialize_signed_array(array)
    np.testing.assert_array_equal(coders.deserialize_signed_array(raw), array)


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        (coders.encode_iid_arithmetic, coders.decode_iid_arithmetic),
        (coders.encode_spatial_context_arithmetic, coders.decode_spatial_context_arithmetic),
        (coders.encode_zigzag_rle_arithmetic, coders.decode_zigzag_rle_arithmetic),
    ],
)
def test_repository_frames_reject_truncation_and_every_trailer(
    spatial_i8: np.ndarray,
    encode,
    decode,
) -> None:
    payload = encode(spatial_i8)
    with pytest.raises(coders.RateCoderError):
        decode(payload[:-1])
    with pytest.raises(coders.RateCoderError):
        decode(payload + b"x")


@pytest.mark.parametrize(
    ("encode", "decode"),
    [
        (coders.encode_g4_decoder_context, coders.decode_g4_decoder_context),
        (coders.encode_willems_ctw, coders.decode_willems_ctw),
        (coders.encode_bellard_class_mixing, coders.decode_bellard_class_mixing),
    ],
)
@pytest.mark.parametrize("raw", [b"", b"abracadabra" * 10, bytes(range(256))])
def test_decoder_derived_context_coders_are_exact_deterministic_and_strict(
    encode,
    decode,
    raw: bytes,
) -> None:
    frame = encode(raw)
    assert encode(raw) == frame
    assert decode(frame) == raw
    accounting = coders.byte_context_frame_accounting(frame)
    assert accounting["model_parameter_bytes"] == 0
    assert accounting["framed_bytes"] == accounting["header_bytes"] + accounting["coded_payload_bytes"]
    for corrupted in (frame[:-1], frame + b"x"):
        with pytest.raises(coders.RateCoderError):
            decode(corrupted)


def test_raw_and_lzma_reject_truncation_and_trailer(spatial_i8: np.ndarray) -> None:
    raw_frame = coders.serialize_signed_array(spatial_i8)
    for corrupted in (raw_frame[:-1], raw_frame + b"x"):
        with pytest.raises(coders.RateCoderError):
            coders.deserialize_signed_array(corrupted)
    compressed = coders.encode_lzma(raw_frame)
    for corrupted in (compressed[:-1], compressed + b"x"):
        with pytest.raises(coders.RateCoderError):
            coders.decode_lzma(corrupted)


def test_spatial_contract_fails_closed_for_non_hwc_shapes() -> None:
    flat = np.arange(8, dtype=np.int8)
    matrix = flat.reshape(2, 4)
    for value in (flat, matrix):
        with pytest.raises(coders.RateCoderError, match=r"\[\.\.\., H, W, C\]"):
            coders.encode_spatial_context_arithmetic(value)
        with pytest.raises(coders.RateCoderError, match=r"\[\.\.\., H, W, C\]"):
            coders.measure_signed_array_ladder(value)


def test_true_left_up_same_channel_context_and_full_byte_accounting(spatial_i8: np.ndarray) -> None:
    contexts = coders._spatial_context_ids(spatial_i8)
    # First pixel has zero/missing left and upper neighbours in both channels.
    assert tuple(int(value) for value in contexts[0, 0, 0]) == (0, 0)
    # At x=1, channel 0 sees left=0; channel 1 sees left=+1.  A flattened
    # predecessor model would see channel 1's prior scalar for channel 0 and
    # cannot produce this same-channel distinction.
    assert int(contexts[0, 0, 1, 0]) == 0
    assert int(contexts[0, 0, 1, 1]) != 0
    # At y=1,x=0 the context comes from the upper same-channel value.
    assert int(contexts[0, 1, 0, 0]) == 0
    assert int(contexts[0, 1, 0, 1]) != 0

    ladder = coders.measure_signed_array_ladder(spatial_i8)
    iid = ladder["repository_iid_arithmetic"]
    spatial = ladder["repository_spatial_context_arithmetic"]
    for row in (iid, spatial):
        assert row["parseback_exact"] is True
        assert row["framed_bytes"] == row["framing_bytes"] + row["model_table_bytes"] + row["payload_bytes"]
        assert row["model_table_bytes"] > 0
    assert spatial["model_table_bytes"] > iid["model_table_bytes"]
    assert "left/up-same-channel" in spatial["dependency_identity"]


def test_explicit_int8_iid_vs_context_comparison(spatial_i8: np.ndarray) -> None:
    comparison = coders.measure_int8_coder_comparison(spatial_i8)
    assert comparison["authority_label"] == "MEASURED_LOCAL_EXACT_INT8_FRAMES"
    assert comparison["iid"]["codec"] == "repository_iid_arithmetic"
    assert comparison["spatial_context"]["codec"] == "repository_spatial_context_arithmetic"
    assert comparison["context_over_iid_ratio"] == pytest.approx(
        comparison["spatial_context"]["framed_bytes"] / comparison["iid"]["framed_bytes"]
    )
    with pytest.raises(coders.RateCoderError, match="int8"):
        coders.measure_int8_coder_comparison(spatial_i8.astype(np.int16))


def test_optional_constriction_is_lazy_fail_closed_and_named_as_decoder_dependency(
    monkeypatch: pytest.MonkeyPatch, spatial_i8: np.ndarray
) -> None:
    monkeypatch.setattr(coders, "_constriction_module", lambda: None)
    monkeypatch.setattr(coders.importlib.metadata, "version", lambda _name: "test")
    with pytest.raises(coders.RateCoderError, match="constriction"):
        coders.encode_spatial_context_constriction(spatial_i8)
    row = coders.measure_signed_array_ladder(spatial_i8)["constriction_spatial_context_arithmetic"]
    assert row["available"] is False
    assert row["decoder_dependency_required"] is True
    assert "not-repository-RangeDecoder" in row["dependency_identity"]


def test_optional_zstd_is_lazy_and_reported_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coders, "_zstd_module", lambda: None)
    monkeypatch.setattr(coders.shutil, "which", lambda _name: None)
    with pytest.raises(coders.RateCoderError, match="zstandard"):
        coders.encode_zstd_19(b"payload")
    row = coders.measure_byte_ladder(b"payload" * 4)["zstd_19"]
    assert row["available"] is False
    assert row["framed_bytes"] == 0
    assert row["parseback_exact"] is False


def test_installed_constriction_roundtrip_and_strict_framing(spatial_i8: np.ndarray) -> None:
    if coders._constriction_module() is None:
        pytest.skip("optional constriction is not installed")
    payload = coders.encode_spatial_context_constriction(spatial_i8)
    np.testing.assert_array_equal(coders.decode_spatial_context_constriction(payload), spatial_i8)
    for corrupted in (payload[:-1], payload + b"x"):
        with pytest.raises(coders.RateCoderError):
            coders.decode_spatial_context_constriction(corrupted)


@pytest.mark.parametrize(
    ("module_getter", "encode", "decode"),
    [
        (coders._brotli_module, coders.encode_brotli_q11, coders.decode_brotli_q11),
        (coders._zstd_module, coders.encode_zstd_19, coders.decode_zstd_19),
    ],
)
def test_installed_optional_byte_coders_reject_truncation_and_trailer(module_getter, encode, decode) -> None:
    if module_getter() is None:
        pytest.skip("optional byte coder is not installed")
    payload = encode(b"deterministic optional codec payload" * 8)
    for corrupted in (payload[:-1], payload + b"x"):
        with pytest.raises(coders.RateCoderError):
            decode(corrupted)


def test_block_fp_accounts_qint_exponent_and_header_bytes() -> None:
    value = np.array([[0.0, 0.25, -0.75], [1.0, -2.0, 0.125]], dtype=np.float32)
    row = coders.measure_block_fp(value, block_size=1, clip_threshold=0.5)
    accounting = row["byte_accounting"]
    assert accounting["sum_matches_framed_bytes"] is True
    assert row["framed_bytes"] == accounting["qint_bytes"] + accounting["exponent_bytes"] + accounting["header_bytes"]
    assert accounting["qint_bytes"] == value.size
    assert accounting["exponent_bytes"] == value.shape[0] * 4
    assert row["packed_byte_coders"][row["best_packed_byte_coder"]]["parseback_exact"] is True
    assert row["matched_realized_dseg"] == "OWED_N_GE_24_UNLESS_ACTUAL_RESULTS_SUPPLIED"
    assert row["sensitivity_allocator_composition"] == "UNMEASURED_NO_ALLOCATION_APPLIED"


def test_authority_labels_keep_pdw2_and_score_fail_closed() -> None:
    labels = coders.authority_labels()
    assert labels["pdw1"] == "MEASURED_REDERIVED_PDW1_EXACTLY_338_BYTES"
    assert labels["pdw2"] == "DERIVED_ONLY_NO_STRICT_ENCODER"
    assert labels["score"] == "UNMEASURED_NO_SCORER_OR_CONTEST_AXIS"


def test_settled_curve_import_keeps_unmeasured_coders_blocked(tmp_path) -> None:
    source_a = tmp_path / "a.json"
    source_b = tmp_path / "b.json"
    source_a.write_text("{}\n")
    source_b.write_text("{}\n")
    references = [
        {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        for source in (source_a, source_b)
    ]
    curve = tmp_path / "curve.json"
    curve.write_text(
        json.dumps(
            {
                "schema": "seg_secant_rd_curve_composed.v1",
                "unique_pair_count": 24,
                "source_receipts": references,
                "measured_points": [
                    {
                        "point_id": "p",
                        "family": "precision_truncation",
                        "pair_count": 24,
                        "d_seg": 0.01,
                        "d_pose": 0.0,
                        "brotli_q11_bytes_per_pair": 10.0,
                        "zstd_19_bytes_per_pair": 12.0,
                    }
                ],
                "waterfill": {"status": "MEASURED_SECANT_KKT_CANDIDATE"},
            }
        )
    )
    result = _settled_seg_secant_curve(curve)
    assert result is not None
    assert result["points"][0]["best_complete_measured_coder"] == "brotli_q11"
    assert result["points"][0]["measured_existing_coders"]["brotli_q11"]["total_bytes_all_streams"] == 240
    assert "EXACT_REDERIVATION" in result["points"][0]["unmeasured_requested_coders"]["lzma_xz_preset9"]
    source_a.write_text("changed\n")
    with pytest.raises(MeasurementError, match="hash mismatch"):
        _settled_seg_secant_curve(curve)
