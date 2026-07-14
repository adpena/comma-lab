# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from probe_weight_l1_class_pair_tie_snap_scorer_forward_n600 import (  # noqa: E402
    _summary,
    _validate_design_receipt,
)


def _base_row(pair_index: int) -> dict[str, object]:
    return {
        "pair_index": pair_index,
        "split": "design" if pair_index < 264 else "second_validation",
        "flips": 1 if pair_index == 11 else 0,
        "pixels": 4,
        "flip_fraction": 0.25 if pair_index == 11 else 0.0,
        "candidate_argmax_sha256": f"plain-{pair_index}",
        "reference_argmax_sha256": f"reference-{pair_index}",
        "max_abs_logit_error": 1.0e-5,
        "sum_squared_logit_error": 1.0e-10,
        "logit_elements": 24,
        "uncertified_pixels": int(pair_index == 11),
        "baseline_margin_min": 0.0,
        "baseline_margin_pair_quantiles": {
            "q0": 0.0,
            "q0.001": 0.0,
            "q0.01": 0.0,
            "q0.05": 0.0,
            "q0.5": 1.0,
        },
    }


def _decision_row(pair_index: int, *, flips: int = 0) -> dict[str, object]:
    return {
        "pair_index": pair_index,
        "split": "design" if pair_index < 264 else "second_validation",
        "flips": flips,
        "pixels": 4,
        "flip_fraction": flips / 4,
        "snapped_pixels": int(pair_index == 11),
        "candidate_margin_min": 1.0e-6,
        "reference_margin_min": 0.0,
        "candidate_argmax_sha256": f"candidate-{pair_index}-{flips}",
        "reference_argmax_sha256": f"reference-{pair_index}",
    }


def _cache_row(pair_index: int) -> dict[str, object]:
    return {
        "pair_index": pair_index,
        "one_thread_argmax_sha256": f"reference-{pair_index}",
        "cached_argmax_sha256": f"cache-{pair_index}",
        "argmax_mismatch_pixels": 0,
        "margin_max_abs_delta": 0.0,
    }


def test_summary_keeps_second_validation_load_bearing() -> None:
    receipt = {
        "contract": {"pair_start": 0, "pair_count": 600},
        "base_rows": [_base_row(pair_index) for pair_index in range(600)],
        "decision_rows": [_decision_row(pair_index) for pair_index in range(600)],
        "cache_custody": {
            "segnet_rows": [_cache_row(pair_index) for pair_index in range(600)]
        },
    }
    exact = _summary(receipt)
    assert exact["design_exact"] is True
    assert exact["second_validation_exact"] is True
    assert exact["argmax_exact_admitted"] is True
    receipt["decision_rows"][587] = _decision_row(587, flips=1)
    failed = _summary(receipt)
    assert failed["design_exact"] is True
    assert failed["second_validation_exact"] is False
    assert failed["argmax_exact_admitted"] is False


def test_design_receipt_requires_target_and_global_snap_conflicts(tmp_path: Path) -> None:
    payload = {
        "schema": "weight_l1_tie_conflict_diagnostic.v1",
        "contract": {
            "design_pairs": [0, 263],
            "second_validation_pairs": [264, 599],
            "second_validation_not_used_to_design": True,
            "epsilon": 2.0**-19,
        },
        "custody": {
            "weight_receipt_sha256": "a" * 64,
            "tie_receipt_sha256_at_freeze": "b" * 64,
        },
        "rows": [
            {
                "pair_index": 11,
                "pixels": [
                    {
                        "candidate_top2_classes": [4, 0],
                        "reference_class": 0,
                        "candidate_plain_class": 4,
                        "candidate_tie_snap_class": 0,
                        "candidate_winner_runner_margin": 2.0**-20,
                    }
                ],
            },
            *[
                {
                    "pair_index": pair_index,
                    "pixels": [
                        {
                            "candidate_top2_classes": [1, 0],
                            "reference_class": 1,
                            "candidate_plain_class": 1,
                            "candidate_tie_snap_class": 0,
                        }
                    ],
                }
                for pair_index in (195, 263)
            ],
        ],
    }
    path = tmp_path / "design.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_design_receipt(path)["schema"].endswith(".v1")
    payload["contract"]["second_validation_not_used_to_design"] = False  # type: ignore[index]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="contract differs"):
        _validate_design_receipt(path)
