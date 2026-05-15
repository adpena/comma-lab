# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "probe_pr106_format0b_hdm10_decoder_microcodec.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "probe_pr106_format0b_hdm10_decoder_microcodec_under_test",
        SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def test_combinadic_mask_rank_roundtrip_for_active_hdm9_mask() -> None:
    mask = bytes.fromhex("81001c05")
    positions = mod.high_mask_positions(mask)

    rank = mod.combination_rank(
        positions,
        n=mod.PR106_HDM9_SCALE_COUNT,
        k=len(positions),
    )

    assert positions == (0, 7, 18, 19, 20, 24, 26)
    assert mod.combination_rank_byte_width(
        n=mod.PR106_HDM9_SCALE_COUNT,
        k=len(positions),
    ) == 3
    assert (
        mod.combination_unrank(
            rank,
            n=mod.PR106_HDM9_SCALE_COUNT,
            k=len(positions),
        )
        == positions
    )
    assert mod.mask_from_positions(positions) == mask


def test_mask_padding_bits_are_rejected() -> None:
    with pytest.raises(mod.ProbeError):
        mod.high_mask_positions(bytes.fromhex("000000f0"))


def test_live_format0b_hdm10_probe_if_artifact_available() -> None:
    if not mod.DEFAULT_SOURCE_ARCHIVE.exists():
        pytest.skip("format-0x0B artifact is not checked into the repository")

    payload = mod.build_probe(mod.DEFAULT_SOURCE_ARCHIVE)

    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["source"]["format_id"] == "0x0B"
    assert payload["hdm9_decoder"]["charged_decoder_tail_bytes"] == 169_946
    assert payload["hdm9_decoder"]["scale_high_mask_hex"] == "81001c05"
    assert payload["decision"]["charged_decoder_byte_positive_candidate_count"] == 2
    assert payload["decision"]["contest_safe_byte_positive_candidate_count"] == 0
    assert (
        payload["decision"]["best_without_exact_mask_constant_candidate"]
        == "hdm10_fixed_popcount_combinadic_scale_mask_rank"
    )
    rows = {row["name"]: row for row in payload["candidates"]}
    assert rows["hdm10_fixed_popcount_combinadic_scale_mask_rank"][
        "lossless_raw_decoder_equivalence"
    ]
    assert (
        rows["hdm10_fixed_popcount_combinadic_scale_mask_rank"][
            "charged_decoder_byte_delta_vs_current"
        ]
        == -1
    )
    assert (
        rows["hdm10_fixed_scale_mask_runtime_constant"][
            "charged_decoder_byte_delta_vs_current"
        ]
        == -4
    )
    assert "no_hdm10_runtime_decoder_implemented" in payload["decision"]["dispatch_blockers"]
