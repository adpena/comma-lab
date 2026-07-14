# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_tool() -> Any:
    path = Path(__file__).resolve().parents[3] / "tools/probe_margin_adaptive_mixed_precision_n600.py"
    spec = importlib.util.spec_from_file_location("probe_margin_adaptive_mixed_precision_n600", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _waterfill_row(bits: float) -> dict[str, Any]:
    return {
        "certified_pixels": 1,
        "selected_bits_sum": bits,
        "margin_bands": [
            {
                "margin_gt": None,
                "margin_le": None,
                "certified_pixels": 1 if index == 0 else 0,
                "selected_bits_sum": bits if index == 0 else 0.0,
            }
            for index in range(8)
        ],
    }


def _profile_row(*, flips: int, hybrid: int, digest: str) -> dict[str, Any]:
    return {
        "flips": flips,
        "pixels": 1,
        "flip_fraction": float(flips),
        "strict_interval_certified_pixels": hybrid,
        "strict_interval_or_frozen_tie_rule_certified_pixels": hybrid,
        "max_abs_logit_error": 0.0,
        "metal_seconds": 0.25,
        "candidate_argmax_sha256": digest,
    }


def test_summary_freezes_design_minimum_without_validation_reselection() -> None:
    tool = _load_tool()
    rows = []
    for pair_index in range(600):
        validation_flip = int(pair_index == 264)
        rows.append(
            {
                "pair_index": pair_index,
                "split": "design" if pair_index < 264 else "second_validation",
                "pixels": 1,
                "reference_seconds": 1.0,
                "reference_argmax_sha256": f"ref{pair_index}",
                "profiles": {
                    "cap8": _profile_row(
                        flips=validation_flip,
                        hybrid=1 - validation_flip,
                        digest=f"a{pair_index}",
                    ),
                    "cap12": _profile_row(
                        flips=0,
                        hybrid=1,
                        digest=f"b{pair_index}",
                    ),
                },
                "strict_waterfill": _waterfill_row(8.0),
                "exact_observed_waterfill": _waterfill_row(8.0),
            }
        )
    receipt = {
        "contract": {
            "pair_start": 0,
            "pair_stop": 600,
            "n_processes": 1,
        },
        "profiles": {
            "cap8": {
                "mac_weighted_average_bits": 8.0,
                "mac_weighted_average_storage_bits": 8.0,
            },
            "cap12": {
                "mac_weighted_average_bits": 12.0,
                "mac_weighted_average_storage_bits": 16.0,
            },
        },
        "rows": rows,
        "trials": [],
    }
    summary = tool._summarize(receipt)
    assert summary["design_selected_profile"] == "cap8"
    assert summary["design_minimum_average_bits_profile"] == "cap8"
    assert summary["diagnostic_full_corpus_minimum_profile"] == "cap12"
    assert summary["selected_second_validation_exact"] is False
    assert summary["native_margin_adaptive_candidate_admitted"] is False
    assert summary["verdict"] == "NO_ADMITTED_MARGIN_ADAPTIVE_NATIVE_PROFILE_IN_LADDER"


def test_profile_aggregate_keeps_interval_and_exact_gates_separate() -> None:
    tool = _load_tool()
    rows = [
        {
            "pair_index": 0,
            "split": "design",
            "profiles": {
                "cap8": _profile_row(flips=0, hybrid=0, digest="candidate")
            },
        }
    ]
    aggregate = tool._profile_aggregate(rows, profile="cap8", split="full")
    assert aggregate["argmax_exact_gate"] is True
    assert aggregate["source_corpus_zero_flip_certificate_gate"] is False
    assert aggregate["strict_interval_certified_fraction"] == 0.0
