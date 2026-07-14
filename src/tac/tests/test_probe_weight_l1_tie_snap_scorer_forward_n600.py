# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[3] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from probe_weight_l1_tie_snap_scorer_forward_n600 import (  # noqa: E402
    _aggregate_decisions,
    _summary,
    _validate_weight_predecessor,
)


def _decision_row(pair_index: int, *, split: str, flips: int) -> dict[str, object]:
    return {
        "pair_index": pair_index,
        "split": split,
        "flips": flips,
        "pixels": 4,
        "flip_fraction": flips / 4,
        "snapped_pixels": 1,
        "candidate_margin_min": 1.0e-6,
        "reference_margin_min": 0.0,
        "candidate_argmax_sha256": f"candidate-{pair_index}",
        "reference_argmax_sha256": f"reference-{pair_index}",
    }


def test_decision_aggregate_preserves_split_and_digest_custody() -> None:
    rows = [
        _decision_row(0, split="calibration", flips=0),
        _decision_row(120, split="heldout", flips=1),
    ]
    calibration = _aggregate_decisions(rows, split="calibration")
    heldout = _aggregate_decisions(rows, split="heldout")
    full = _aggregate_decisions(rows, split="full")
    assert calibration["argmax_exact_gate"] is True
    assert heldout["argmax_exact_gate"] is False
    assert full["flips"] == 1
    assert full["pairs_with_snaps"] == 2
    assert len(str(full["argmax_corpus_sha256"])) == 64


def test_summary_never_reselects_epsilon_on_heldout_rows() -> None:
    base_rows = [{"pair_index": pair_index} for pair_index in range(600)]
    selected_rows = [
        _decision_row(
            pair_index,
            split="calibration" if pair_index < 120 else "heldout",
            flips=1 if pair_index == 120 else 0,
        )
        for pair_index in range(600)
    ]
    wider_rows = [
        _decision_row(
            pair_index,
            split="calibration" if pair_index < 120 else "heldout",
            flips=0,
        )
        for pair_index in range(600)
    ]
    receipt = {
        "contract": {"pair_start": 0, "pair_count": 600},
        "base_rows": base_rows,
        "arms": {
            "epsilon_2m19": {
                "epsilon": 2.0**-19,
                "decision_rows": selected_rows,
            },
            "epsilon_2m18": {
                "epsilon": 2.0**-18,
                "decision_rows": wider_rows,
            },
        },
        "cache_custody": {"segnet_rows": []},
    }
    summary = _summary(receipt)
    assert summary["minimum_calibration_exact_arm"] == "epsilon_2m19"
    assert summary["selected_heldout_exact"] is False
    assert summary["arms"]["epsilon_2m18"]["heldout"]["argmax_exact_gate"] is True
    assert summary["argmax_exact_admitted"] is False


def test_weight_predecessor_requires_full_negative_and_qdq_custody(
    tmp_path: Path,
) -> None:
    payload = {
        "schema": "weight_l1_int64_fixedpoint_scorer_n600.v1",
        "completed": True,
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": False,
            "candidate": {"full": {"flips": 1}},
        },
        "model_manifest": {
            "converted_conv2d_count": 125,
            "maximum_bits": 31,
            "assignment_rule": (
                "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            ),
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
        },
        "custody": {
            "qdq_precursor_sha256": "a" * 64,
            "qdq_precursor_fingerprint": "b" * 64,
        },
    }
    path = tmp_path / "weight.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _validate_weight_predecessor(path)["completed"] is True
    payload["summary"]["argmax_exact_admitted"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="already exact"):
        _validate_weight_predecessor(path)
