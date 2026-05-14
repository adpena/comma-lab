# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_tool() -> Any:
    path = REPO / "tools" / "probe_precoarsening_entropy_coders.py"
    spec = importlib.util.spec_from_file_location(
        "probe_precoarsening_entropy_coders_under_test", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _record(module: Any, idx: int, q: list[int], scale: bytes = b"sc") -> Any:
    return module.TensorStream(
        schema_index=idx,
        name=f"tensor_{idx}",
        shape=(len(q),),
        q_i8=np.array(q, dtype=np.int8),
        scale_bytes=scale,
        source_symbol_bytes=bytes(value + 127 for value in q),
    )


def test_coder_packet_overhead_charges_tables_lengths_and_scales() -> None:
    tool = _load_tool()

    overhead = tool.coder_packet_overhead_bytes(n_tensors=2, scale_bytes=6)

    assert overhead["fixed_packet_header_bytes"] == 8
    assert overhead["model_header_bytes"] == 2 * 255 * 2
    assert overhead["stream_length_table_bytes"] == 2 * 4
    assert overhead["scale_bytes"] == 6
    assert overhead["total_packet_overhead_bytes"] == 8 + (2 * 255 * 2) + 8 + 6


def test_static_model_proxy_is_no_score_and_full_headered() -> None:
    tool = _load_tool()
    records = (
        _record(tool, 0, [-1, 0, 0, 1], b"aa"),
        _record(tool, 1, [1, 1, 1, 0], b"bbbb"),
    )

    proxy = tool.build_static_model_proxy(records)

    assert proxy["score_claim"] is False
    assert proxy["evidence_label"] == "CPU/proxy_no_bitstream"
    assert proxy["payload_floor_bytes"] > 0
    assert proxy["model_header_bytes"] == 2 * 255 * 2
    assert proxy["stream_length_table_bytes"] == 2 * 4
    assert proxy["scale_bytes"] == 6
    expected_payload_bytes = sum(
        row["smoothed_entropy_payload_bytes"] for row in proxy["per_tensor"]
    )
    assert proxy["payload_floor_bytes"] == expected_payload_bytes
    assert proxy["payload_floor_byte_alignment"] == "sum_per_tensor_ceil_bits_div_8"
    assert proxy["total_estimated_bytes"] == expected_payload_bytes + proxy["total_packet_overhead_bytes"]
    assert "proxy_floor_not_an_actual_bitstream" in proxy["limitations"]


def test_reactivation_review_never_promotes_to_dispatch() -> None:
    tool = _load_tool()

    blocked = tool.reactivation_review(
        source_decoder_section_bytes=100,
        brotli_q11_source_layout_bytes=100,
        brotli_q11_canonical_bytes=105,
        proxy_total_bytes=120,
        constriction_total_bytes=110,
    )
    assert blocked["score_claim"] is False
    assert blocked["ready_for_exact_eval_dispatch"] is False
    assert blocked["verdict"] == "measured_precoarsening_static_config_retired"

    prototype = tool.reactivation_review(
        source_decoder_section_bytes=100,
        brotli_q11_source_layout_bytes=100,
        brotli_q11_canonical_bytes=105,
        proxy_total_bytes=120,
        constriction_total_bytes=90,
    )
    assert prototype["score_claim"] is False
    assert prototype["ready_for_exact_eval_dispatch"] is False
    assert prototype["verdict"] == "reactivate_runtime_prototype_only_no_score_claim"


def test_canonical_precoarsening_stream_uses_signed7_shift_and_scale_bytes() -> None:
    tool = _load_tool()
    records = (_record(tool, 0, [-127, 0, 127], b"xy"),)

    payload = tool._canonical_precoarsening_bytes(records)

    assert payload == bytes([0, 127, 254]) + b"xy"


def test_real_pr101_pr106_parser_outputs_are_locked_if_custody_archives_exist() -> None:
    tool = _load_tool()
    pr101 = (
        REPO
        / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
    )
    pr106 = (
        REPO
        / "experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip"
    )
    if not pr101.is_file() or not pr106.is_file():
        pytest.skip("public PR101/PR106 custody archives are not present in this checkout")

    manifest = tool.build_probe_manifest(pr101, pr106)
    by_label = {target["label"]: target for target in manifest["targets"]}

    pr101_row = by_label["PR101 hnerv_ft_microcodec"]
    assert pr101_row["archive_sha256"] == (
        "b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e"
    )
    assert pr101_row["decoder_section_sha256"] == (
        "836d1876bffd74f77f30e387a3b4cac1dbb25929cc4d348830d36cfa2a6d48a6"
    )
    assert pr101_row["q_symbol_count"] == 228_958
    assert pr101_row["source_layout_brotli_q11"]["matches_source_decoder_section"] is True
    assert pr101_row["reactivation_review"]["verdict"] == "measured_precoarsening_static_config_retired"

    pr106_row = by_label["PR106 belt_and_suspenders"]
    assert pr106_row["archive_sha256"] == (
        "3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58"
    )
    assert pr106_row["decoder_section_sha256"] == (
        "654999f81f0552fb7568e6977e73aa329661c10c79a6ab6cddc3171302352004"
    )
    assert pr106_row["q_symbol_count"] == 228_958
    assert pr106_row["source_layout_brotli_q11"]["matches_source_decoder_section"] is True
    assert pr106_row["reactivation_review"]["canonical_i8_delta_vs_reference_bytes"] == -52
    assert pr106_row["reactivation_review"]["verdict"] == "measured_precoarsening_static_config_retired"
