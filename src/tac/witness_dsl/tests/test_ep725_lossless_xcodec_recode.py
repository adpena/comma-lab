from __future__ import annotations

import copy
import io
import json
import struct
import zipfile

import brotli
import numpy as np
import pytest

from tac.witness_dsl.ep725_lossless_xcodec_recode import (
    Ep725LosslessXCodecError,
    inspect_source_zip,
    parse_ep725_lvls1,
    search_ep725_lossless_xcodec,
)


def _pack(*blocks: bytes) -> bytes:
    return b"LVLS1\x00" + b"".join(struct.pack("<I", len(block)) + block for block in blocks)


def _fixture_member() -> bytes:
    manifest = {
        "format_version": 1,
        "n_pairs": 600,
        "code_shape": [1200, 32],
        "base_param_order": ["matrix", "bias"],
        "base_shapes": {"matrix": [3, 4], "bias": [5]},
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode("ascii")
    matrix = np.arange(12, dtype=np.int8).reshape(3, 4)
    bias = np.arange(-2, 3, dtype=np.int8)
    base = brotli.compress(matrix.tobytes() + bias.tobytes(), quality=11)
    code = np.tile(np.arange(32, dtype=np.int8), (1200, 1))
    return _pack(manifest_bytes, base, brotli.compress(code.tobytes(), quality=11), b"")


def _fixture_archive(member: bytes) -> bytes:
    output = io.BytesIO()
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.create_version = 20
    info.extract_version = 20
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(output, mode="w") as archive:
        archive.writestr(copy.copy(info), member, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    return output.getvalue()


def test_exact_search_roundtrips_the_full_quantized_state() -> None:
    source_archive = _fixture_archive(_fixture_member())
    result = search_ep725_lossless_xcodec(source_archive, deflate_levels=(6,))

    assert result.transformed_points_measured == 4
    assert result.decoded_state_sha256 == result.selected_decoded_state_sha256
    receipt = result.structural_receipt()
    assert receipt["proof"]["full_quantized_state_equal"] is True
    assert receipt["proof"]["base_arrays_equal"] is True
    assert receipt["proof"]["code_array_equal"] is True
    assert receipt["truth"]["candidate_claim"] is False
    assert len(result.selected.archive_bytes) <= len(source_archive)


def test_selected_transformed_member_parses_without_source_form() -> None:
    source_archive = _fixture_archive(_fixture_member())
    result = search_ep725_lossless_xcodec(source_archive, deflate_levels=(6,))
    parsed = parse_ep725_lvls1(
        result.selected.member_bytes,
        require_source_form=not result.selected.transformed,
    )
    assert parsed.code_quantized.shape == (1200, 32)


def test_source_sha_and_trailing_member_bytes_fail_closed() -> None:
    member = _fixture_member()
    archive = _fixture_archive(member)
    with pytest.raises(Ep725LosslessXCodecError, match="archive SHA-256"):
        inspect_source_zip(archive, expected_archive_sha256="0" * 64, deflate_levels=(6,))
    with pytest.raises(Ep725LosslessXCodecError, match="unconsumed trailing"):
        parse_ep725_lvls1(member + b"foreign", require_source_form=True)


def test_noncanonical_manifest_and_existing_source_xcodec_fail_closed() -> None:
    parsed = parse_ep725_lvls1(_fixture_member(), require_source_form=True)
    pretty = json.dumps(parsed.manifest, indent=2).encode("ascii")
    malformed = _pack(pretty, parsed.base_brotli, parsed.code_brotli, parsed.pose_bytes)
    with pytest.raises(Ep725LosslessXCodecError, match="canonical compact JSON"):
        parse_ep725_lvls1(malformed, require_source_form=True)

    manifest = dict(parsed.manifest)
    manifest["xcodec"] = {"p": [], "c": 0}
    existing = _pack(
        json.dumps(manifest, separators=(",", ":")).encode("ascii"),
        parsed.base_brotli,
        parsed.code_brotli,
        parsed.pose_bytes,
    )
    with pytest.raises(Ep725LosslessXCodecError, match="must not already"):
        parse_ep725_lvls1(existing, require_source_form=True)


def test_invalid_xcodec_indices_fail_closed() -> None:
    parsed = parse_ep725_lvls1(_fixture_member(), require_source_form=True)
    manifest = dict(parsed.manifest)
    manifest["xcodec"] = {"p": [99], "c": 0}
    malformed = _pack(
        json.dumps(manifest, separators=(",", ":")).encode("ascii"),
        parsed.base_brotli,
        parsed.code_brotli,
        parsed.pose_bytes,
    )
    with pytest.raises(Ep725LosslessXCodecError, match="out of range"):
        parse_ep725_lvls1(malformed, require_source_form=False)
