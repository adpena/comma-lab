from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.throughput_authority_anchors_20260714 import (
    build_full_r_anchor,
    build_qdq_anchor,
)


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _qdq() -> dict[str, object]:
    indices_hash = "a" * 64
    return {
        "schema": "fixedpoint_scorer_forward_n600.v2",
        "contract": {
            "native_integer_speed_claim": False,
            "activation_scale_mode": "fixed_calibration",
            "calibration_split": [0, 120],
            "heldout_split": [120, 600],
            "accumulation": "QDQ fp32",
        },
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "minimum_argmax_exact_arm": "w22a22",
            "minimum_training_tolerance_arm": "w18a18",
            "rung2_verdict": "ARGMAX_FIXEDPOINT_FEASIBLE",
            "verdict_scope": "n600 instance",
            "cache_custody": {
                "status": "MEASURED",
                "pairs": 600,
                "unique_pair_indices": 600,
                "observed_pair_indices_sha256": indices_hash,
                "expected_pair_indices_sha256": indices_hash,
            },
            "arms": {
                "w22a22": {
                    "status": "MEASURED",
                    "argmax_exact_admitted": True,
                    "segnet": {
                        "full": {
                            "aggregate_flip_fraction": 0.0,
                            "worst_pair_flip_fraction": 0.0,
                            "uncertified_pixels": 0,
                            "argmax_corpus_sha256": "b" * 64,
                        }
                    },
                }
            },
        },
    }


def test_qdq_anchor_requires_exact_n600_custody(tmp_path: Path) -> None:
    path = tmp_path / "qdq.json"
    payload = _qdq()
    _write(path, payload)
    anchor = build_qdq_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    assert anchor.empirical_output["minimum_argmax_exact_arm"] == "w22a22"
    assert anchor.inputs["activation_scale_mode"] == "fixed_calibration"
    payload["summary"]["cache_custody"]["pairs"] = 599  # type: ignore[index]
    with pytest.raises(ValueError, match=r"0\.\.599"):
        build_qdq_anchor(path, payload, repo=tmp_path)


def test_full_r_anchor_requires_complete_real_corpus(tmp_path: Path) -> None:
    path = tmp_path / "full_r.json"
    payload = {
        "schema": "pythagorean_exact_arithmetic_full_r_n600.v2",
        "contract": {"q_weight_bits": 15, "state_bits_by_boundary": [7, 5]},
        "summary": {
            "complete": True,
            "authority": {
                "coverage_exact": True,
                "frames": 1200,
                "within_derived_bound": True,
            },
            "fixed_q15_int32_atomic": {
                "cross_process_identical": True,
                "exact_numpy_int_corpus_parity": True,
            },
            "float_atomic": {"cross_process_identical": False},
            "overall_verdict": "REAL-L70-LEVER-FULL-R-N600",
            "verdict_scope": "n600 instance",
        },
    }
    _write(path, payload)
    anchor = build_full_r_anchor(path, payload, repo=tmp_path)
    assert anchor.residual == 0.0
    payload["summary"]["authority"]["frames"] = 1199  # type: ignore[index]
    with pytest.raises(ValueError, match="custody"):
        build_full_r_anchor(path, payload, repo=tmp_path)
