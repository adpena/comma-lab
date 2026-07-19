# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tools.measure_yhat_rd_ladder import (
    RUNG_ORDER,
    SCHEMA_CHUNK,
    LadderError,
    _scientific_stage,
    compose_chunks,
    compress_payload,
    parse_chunk_container,
    parse_plane_description,
    quantize_u8_plane,
    serialize_chunk_container,
    serialize_plane_description,
)


def _plane_i32() -> np.ndarray:
    values = np.arange(384 * 512 * 3, dtype=np.int32)
    return values.reshape(384, 512, 3)


def test_i32_descriptor_roundtrip_is_complete_and_hashed() -> None:
    source = _plane_i32()
    payload = serialize_plane_description(source, encoding="i32_numerator", denominator=786432)
    decoded, header = parse_plane_description(payload)
    assert np.array_equal(decoded, source)
    assert header["encoding"] == "i32_numerator"
    assert header["denominator"] == 786432
    assert header["payload_bytes"] == len(payload)


def test_u8_descriptor_and_chunk_container_roundtrip() -> None:
    plane = np.arange(384 * 512 * 3, dtype=np.uint32).reshape(384, 512, 3).astype(np.uint8)
    first = serialize_plane_description(plane, encoding="u8_plane", denominator=786432, quant_levels=64)
    second = serialize_plane_description(255 - plane, encoding="u8_plane", denominator=786432, quant_levels=64)
    container = serialize_chunk_container(((7, first), (19, second)))
    parsed = parse_chunk_container(container)
    assert [pair for pair, _ in parsed] == [7, 19]
    assert parsed[0][1] == first
    assert parsed[1][1] == second


def test_descriptor_refuses_tamper_and_container_refuses_duplicate_pair() -> None:
    payload = bytearray(serialize_plane_description(_plane_i32(), encoding="i32_numerator", denominator=786432))
    payload[-1] ^= 1
    with pytest.raises(LadderError, match="hash custody"):
        parse_plane_description(bytes(payload))
    good = serialize_plane_description(_plane_i32(), encoding="i32_numerator", denominator=786432)
    with pytest.raises(LadderError, match="unique"):
        serialize_chunk_container(((1, good), (1, good)))


def test_quantization_is_deterministic_bounded_and_on_declared_grid() -> None:
    target = np.linspace(-5.0, 260.0, 384 * 512 * 3).reshape(384, 512, 3)
    q = quantize_u8_plane(target, 16)
    assert q.dtype == np.uint8
    assert q.min() == 0
    assert q.max() == 255
    assert len(np.unique(q)) == 16
    assert np.array_equal(q, quantize_u8_plane(target, 16))


def test_actual_codec_roundtrip_reports_both_coders() -> None:
    payload = serialize_chunk_container(
        ((3, serialize_plane_description(_plane_i32(), encoding="i32_numerator", denominator=786432)),)
    )
    report = compress_payload(payload)
    assert report["raw_bytes"] == len(payload)
    assert report["brotli_q11_bytes"] > 0
    assert report["zstd_19_bytes"] > 0
    assert report["lossless_parseback"] is True


def _chunk(path: Path, pairs: list[int], witness_hash: str = "a" * 64) -> None:
    rung_rows = {
        rung: {
            "description": {"encoding": "u8_plane", "quant_levels": 16},
            "distortionnet": {"d_seg": 0.1, "d_pose": 0.2},
            "shared_plane_error_vs_source": {"mean_abs": 1.0, "rmse": 2.0},
            "lattice": {
                "exact_blocks": 1,
                "heuristic_blocks": 0,
                "budget_blocks": 0,
                "proven_affine_infeasible_blocks": 0,
                "target_repair_cells": 0,
                "nonzero_realized_numerator_error_cells": 0,
            },
            "runtime_seconds": {"lattice_solve": 0.5},
        }
        for rung in RUNG_ORDER
    }
    doc = {
        "schema": SCHEMA_CHUNK,
        "witness_prepare": {
            "sha256": witness_hash,
            "full_archive": {
                "archive_zip_bytes": 1200,
                "rate_term_actual": 25 * 1200 / 37_545_489,
            },
        },
        "pairs": [
            {
                "pair_id": pair,
                "byteclosed_witness_direct_distortion": {"d_seg": 0.1, "d_pose": 0.2},
                "rungs": rung_rows,
            }
            for pair in pairs
        ],
        "chunk_rate": {rung: {"brotli_q11_bytes": 100, "zstd_19_bytes": 110} for rung in RUNG_ORDER},
    }
    path.write_text(json.dumps(doc))


def test_compose_requires_disjoint_n24_and_emits_table_and_csv(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    _chunk(first, list(range(12)))
    _chunk(second, list(range(12, 24)))
    output, csv_path = tmp_path / "table.json", tmp_path / "table.csv"
    table = compose_chunks(argparse.Namespace(receipts=[first, second], output=output, csv=csv_path))
    assert table["pair_count"] == 24
    assert [row["rung"] for row in table["rows"]] == list(RUNG_ORDER)
    assert output.is_file()
    assert csv_path.is_file()


def test_compose_refuses_overlap_and_witness_mismatch(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    _chunk(first, list(range(12)))
    _chunk(second, list(range(11, 23)))
    with pytest.raises(LadderError, match="overlap"):
        compose_chunks(
            argparse.Namespace(
                receipts=[first, second],
                output=tmp_path / "x.json",
                csv=tmp_path / "x.csv",
            )
        )
    _chunk(second, list(range(12, 24)), witness_hash="b" * 64)
    with pytest.raises(LadderError, match="do not share"):
        compose_chunks(
            argparse.Namespace(
                receipts=[first, second],
                output=tmp_path / "y.json",
                csv=tmp_path / "y.csv",
            )
        )


def test_resume_scientific_projection_ignores_runtime_without_mutation() -> None:
    first = {
        "pair_id": 3,
        "pair_runtime_seconds": 10.0,
        "rungs": {"r": {"value": 7, "runtime_seconds": {"solve": 2.0}}},
    }
    second = {
        "pair_id": 3,
        "pair_runtime_seconds": 99.0,
        "rungs": {"r": {"value": 7, "runtime_seconds": {"solve": 88.0}}},
    }
    assert _scientific_stage(first) == _scientific_stage(second)
    assert first["rungs"]["r"]["runtime_seconds"] == {"solve": 2.0}
    second["rungs"]["r"]["value"] = 8
    assert _scientific_stage(first) != _scientific_stage(second)
