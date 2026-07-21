# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import numpy as np
import pytest

from tac.boundary_math.analytic_lane_render_band import rasterize_lane_coverage_range_dependent
from tac.boundary_math.lane_sdf_component import LaneLine
from tac.lossless.range_coder import encode_static_symbols
from tac.optimization.s4_archive_composer import (
    S4ArchiveError,
    SectionBytes,
    build_payload_manifest,
    canonical_json_bytes,
    deterministic_archive,
    parse_sections,
    serialize_sections,
)

REPO = Path(__file__).resolve().parents[3]


def _sections(component_payload: bytes = b"component") -> list[SectionBytes]:
    rows = [
        SectionBytes("seed.ppcs", b"seed", "raw", 4),
        SectionBytes("base.pbase3", b"base", "mixed", 8),
        SectionBytes("causal.pcr3", b"", "raw", 0),
        SectionBytes("events.pce3", b"events", "lzma1_raw_1MiB", 12),
        SectionBytes("components.pcomp3", component_payload, "zlib9", 16),
    ]
    manifest = build_payload_manifest(rows, source_commit="1" * 40)
    manifest_bytes = canonical_json_bytes(manifest)
    return [SectionBytes("manifest.json", manifest_bytes, "raw", len(manifest_bytes)), *rows]


def _inflate_module():
    path = REPO / "submissions/s4_archive_composer/inflate.py"
    spec = importlib.util.spec_from_file_location("s4_standalone_inflate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_monolith_roundtrip_is_canonical_hash_bound_and_hot_swappable() -> None:
    first = serialize_sections(_sections())
    assert serialize_sections(parse_sections(first)) == first
    second = serialize_sections(_sections(b"replacement-component-bytes"))
    assert first != second
    assert parse_sections(second)[-1].payload == b"replacement-component-bytes"
    assert [row.name for row in parse_sections(second)] == [row.name for row in _sections()]


def test_monolith_rejects_drift_trailing_and_unregistered_codec() -> None:
    payload = serialize_sections(_sections())
    drift = bytearray(payload)
    drift[-33] ^= 1
    with pytest.raises(S4ArchiveError, match="digest"):
        parse_sections(bytes(drift))
    with pytest.raises(S4ArchiveError, match=r"digest|trailing"):
        parse_sections(payload + b"x")
    with pytest.raises(S4ArchiveError, match="codec"):
        SectionBytes("seed.ppcs", b"seed", "brotli_q11", 4)


def test_deterministic_archive_has_exactly_one_zero_bin_member(tmp_path: Path) -> None:
    payload = serialize_sections(_sections())
    left = deterministic_archive(tmp_path / "left.zip", payload)
    right = deterministic_archive(tmp_path / "right.zip", payload)
    assert left["sha256"] == right["sha256"]
    assert hashlib.sha256((tmp_path / "left.zip").read_bytes()).hexdigest() == left["sha256"]
    with zipfile.ZipFile(tmp_path / "left.zip") as archive:
        assert archive.namelist() == ["0.bin"]
        assert archive.read("0.bin") == payload


def test_standalone_source_has_no_tac_import_and_strict_parser_matches_repo() -> None:
    runtime = REPO / "submissions/s4_archive_composer/inflate.py"
    source = runtime.read_text()
    assert "import tac" not in source
    assert "from tac" not in source
    payload = serialize_sections(_sections())
    standalone = _inflate_module()
    parsed = standalone.parse_s4(payload)
    assert parsed["components.pcomp3"][0] == b"component"
    assert parsed["base.pbase3"][1] == "mixed"


def test_factor2_support_fill_is_exact_for_integer_scorer_plane() -> None:
    standalone = _inflate_module()
    rows = standalone.support_indices(standalone.CAMERA_H, standalone.SCORER_H)
    cols = standalone.support_indices(standalone.CAMERA_W, standalone.SCORER_W)
    labels = (np.indices((standalone.SCORER_H, standalone.SCORER_W)).sum(axis=0) % 5).astype(np.uint8)
    palette = np.asarray(
        ((153, 255, 51), (51, 255, 204), (0, 153, 0), (102, 204, 51), (0, 255, 153)),
        dtype=np.uint8,
    )
    frame = standalone.realize(labels, rows, cols, palette)
    target = palette[labels]
    for row_offset in range(2):
        for col_offset in range(2):
            assert np.array_equal(
                frame[rows[:, row_offset, None], cols[None, :, col_offset], :],
                target,
            )


def test_standalone_range_decoder_is_exact_557_twin() -> None:
    standalone = _inflate_module()
    frequencies = [13, 5, 2, 17, 1]
    symbols = bytes((index * 7 + index // 3) % len(frequencies) for index in range(257))
    encoded = encode_static_symbols(symbols, frequencies=frequencies)
    assert standalone.range_decode_static(encoded, frequencies, len(symbols)) == symbols


def test_range_codec_is_registered_for_future_section_hot_swaps() -> None:
    assert SectionBytes("events.pce3", b"range-envelope", "range_static_v1", 1024).codec == (
        "range_static_v1"
    )
    assert SectionBytes("components.pcomp3", b"range-envelope", "range_static_v1", 2048).codec == (
        "range_static_v1"
    )


def test_standalone_lane_raster_matches_native_counted_camera_model() -> None:
    standalone = _inflate_module()
    vector = np.asarray(
        [0.0, 0.0, 0.018, -0.8, 0.0, 1.75, 9.0, 1.5, 0.55, 5.0, 80.0],
        dtype=np.float64,
    )
    header = {
        "v_h": 174.0,
        "cx": None,
        "softness": 1.0,
        "dash_gate": True,
        "dash_forward_max_m": 55.0,
    }
    camera = {"height_m": 1.2, "fx_scorer": 400.3, "fy_scorer": 399.5}
    line = LaneLine(
        centerline_coeffs=vector[:4],
        halfwidth_coeffs=vector[4:6],
        dash_period_m=float(vector[6]),
        dash_phase_m=float(vector[7]),
        dash_duty=float(vector[8]),
        forward_range=(float(vector[9]), float(vector[10])),
    )
    native = rasterize_lane_coverage_range_dependent(
        [line],
        h=standalone.SCORER_H,
        w=standalone.SCORER_W,
        softness=header["softness"],
        dash_gate=header["dash_gate"],
        dash_forward_max_m=header["dash_forward_max_m"],
        v_h=header["v_h"],
        cx=header["cx"],
    )
    assert np.array_equal(standalone.lane_mask([vector], header, camera), native >= 0.5)


def test_manifest_rejects_section_registry_drift() -> None:
    rows = _sections()
    manifest = __import__("json").loads(rows[0].payload)
    manifest["section_registry"][-1]["encoded_bytes"] += 1
    rows[0] = SectionBytes("manifest.json", canonical_json_bytes(manifest), "raw", len(rows[0].payload))
    with pytest.raises(S4ArchiveError, match="custody mismatch"):
        serialize_sections(rows)
