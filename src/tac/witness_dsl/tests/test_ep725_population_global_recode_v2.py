from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.witness_dsl.ep725_lossless_xcodec_recode import inspect_source_zip, parse_ep725_lvls1
from tac.witness_dsl.ep725_population_global_recode_v2 import (
    G20_LAYOUT_CODEC,
    G20_OUTER_PROFILE,
    G20_POPULATION_TRANSFORM,
    MAGIC,
    POPULATION_TRANSFORMS,
    SOURCE_ARCHIVE_SHA256,
    SOURCE_MEMBER_SHA256,
    Ep725PopulationGlobalRecodeError,
    LayoutCodecV2,
    OuterProfileV2,
    PopulationTransformV2,
    RecodeConfigV2,
    build_complete_archive,
    decode_population,
    encode_population,
    encode_population_global_member,
    parse_population_global_member,
)

SOURCE = Path("/Volumes/VertigoDataTier/pact/yhat_rd_ladder_20260719/prepare/full_n600_packet/archive.zip")


def _real_source():
    if not SOURCE.is_file():
        pytest.skip("frozen ep725 n600 source SSD is not connected")
    archive = SOURCE.read_bytes()
    profile = inspect_source_zip(
        archive,
        expected_archive_sha256=SOURCE_ARCHIVE_SHA256,
        expected_member_sha256=SOURCE_MEMBER_SHA256,
    )
    return profile, parse_ep725_lvls1(profile.member_bytes, require_source_form=True)


def test_complete_population_transform_menu_is_exact_on_real_n600_state() -> None:
    _, parsed = _real_source()
    assert parsed.code_quantized.shape == (1200, 32)
    assert len(POPULATION_TRANSFORMS) > 500
    for transform in POPULATION_TRANSFORMS:
        encoded = encode_population(parsed.code_quantized, transform)
        decoded = decode_population(encoded, transform)
        assert decoded.dtype == np.int8
        assert np.array_equal(decoded, parsed.code_quantized), transform


@pytest.mark.parametrize(
    "transform",
    [
        G20_POPULATION_TRANSFORM,
        PopulationTransformV2("delta", 4, 3, 8, True),
        PopulationTransformV2("xor", 5, 7, 64, False),
    ],
)
@pytest.mark.parametrize(
    "layout",
    [
        G20_LAYOUT_CODEC,
        LayoutCodecV2("joint", "brotli10", "raw"),
        LayoutCodecV2("separate", "lzma9", "bz2"),
    ],
)
@pytest.mark.parametrize(
    "outer",
    [G20_OUTER_PROFILE, OuterProfileV2("store", None), OuterProfileV2("deflate", 1)],
)
def test_real_n600_member_and_complete_archive_roundtrip(
    transform: PopulationTransformV2,
    layout: LayoutCodecV2,
    outer: OuterProfileV2,
) -> None:
    source, parsed = _real_source()
    config = RecodeConfigV2(
        transpose_mask=41,
        zigzag_mask=257,
        population_transform=transform,
        layout_codec=layout,
        outer_profile=outer,
    )
    point = build_complete_archive(source, parsed, config)
    selected = parse_population_global_member(point.member_bytes)
    assert selected.config == config
    assert selected.pose_bytes == parsed.pose_bytes
    assert np.array_equal(selected.code_quantized, parsed.code_quantized)
    assert all(np.array_equal(selected.base_quantized[name], parsed.base_quantized[name]) for name in parsed.base_order)
    assert encode_population_global_member(parsed, config) == point.member_bytes
    with zipfile.ZipFile(io.BytesIO(point.archive_bytes)) as archive:
        assert archive.namelist() == ["0.bin"]
        assert archive.read("0.bin") == point.member_bytes
        assert archive.testzip() is None


def test_parser_refuses_trailing_bytes_and_noncanonical_manifest() -> None:
    _, parsed = _real_source()
    config = RecodeConfigV2(
        transpose_mask=0,
        zigzag_mask=0,
        population_transform=G20_POPULATION_TRANSFORM,
        layout_codec=G20_LAYOUT_CODEC,
        outer_profile=G20_OUTER_PROFILE,
    )
    member = encode_population_global_member(parsed, config)
    with pytest.raises(Ep725PopulationGlobalRecodeError, match="trailing"):
        parse_population_global_member(member + b"x")

    manifest_size = int.from_bytes(member[len(MAGIC) : len(MAGIC) + 4], "little")
    manifest_start = len(MAGIC) + 4
    manifest = member[manifest_start : manifest_start + manifest_size]
    assert manifest.startswith(b"{")
    corrupted = (
        member[: len(MAGIC)]
        + (manifest_size + 1).to_bytes(4, "little")
        + b" "
        + manifest
        + member[manifest_start + manifest_size :]
    )
    with pytest.raises(Ep725PopulationGlobalRecodeError, match="canonical"):
        parse_population_global_member(corrupted)


def test_transform_wires_fail_closed() -> None:
    with pytest.raises(Ep725PopulationGlobalRecodeError):
        PopulationTransformV2.from_wire([0, 0, 1, 7, 0])
    with pytest.raises(Ep725PopulationGlobalRecodeError):
        PopulationTransformV2.from_wire([2, 0, 0, 0, 0])
    with pytest.raises(Ep725PopulationGlobalRecodeError):
        LayoutCodecV2.from_wire([1, 0, 3])
    with pytest.raises(Ep725PopulationGlobalRecodeError):
        OuterProfileV2.from_wire([0, 9])
