# SPDX-License-Identifier: MIT
"""CPU-only contract tests for the Task #494 custom-Metal host gate."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "fixedpoint_authority_host_gate",
    ROOT / "tools/bench_fixedpoint_authority_kernels.py",
)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_int64_precursor_can_unlock_nonexact_qdq_arm(tmp_path: Path) -> None:
    qdq_path = tmp_path / "qdq.json"
    control_rows = [
        {"pair_index": pair_index, "reference_argmax_sha256": "c" * 64}
        for pair_index in range(600)
    ]
    _write_json(
        qdq_path,
        {
            "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
            "contract": {"activation_scale_mode": "dynamic_exact_absmax"},
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "minimum_argmax_exact_arm": None,
                "arms": {
                    "w26a26": {
                        "argmax_exact_admitted": False,
                        "training_tolerance_admitted": True,
                    }
                },
            },
            "calibration": {"segnet_operator_absmax": {"only.layer": 1.0}},
            "arms": {"fp32_control": {"segnet_rows": control_rows}},
        },
    )
    precursor_path = tmp_path / "exact.json"
    _write_json(
        precursor_path,
        {
            "schema": "exact_int64_fixedpoint_scorer_n600.v1",
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "argmax_exact_admitted": True,
                "bits": 26,
            },
            "model_manifest": {
                "bits": 26,
                "converted_conv2d_count": 125,
                "accumulation": "exact_signed_int64",
            },
            "custody": {"qdq_precursor_sha256": _sha256(qdq_path)},
        },
    )
    with pytest.raises(ValueError, match="no exact-argmax-admitted"):
        GATE._load_calibration(qdq_path, bits=26)
    calibration, _, mode, bits, precursor = GATE._load_calibration(
        qdq_path,
        bits=26,
        integer_precursor_path=precursor_path,
    )
    assert calibration == {"only.layer": 1.0}
    assert mode == "dynamic_exact_absmax"
    assert bits == 26
    assert precursor is not None


def test_mixed_int64_precursor_can_unlock_geometry_safe_metal_arm(tmp_path: Path) -> None:
    qdq_path = tmp_path / "qdq.json"
    control_rows = [
        {"pair_index": pair_index, "reference_argmax_sha256": "c" * 64}
        for pair_index in range(600)
    ]
    _write_json(
        qdq_path,
        {
            "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
            "contract": {"activation_scale_mode": "dynamic_exact_absmax"},
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "minimum_argmax_exact_arm": None,
                "arms": {
                    "w26a26": {
                        "argmax_exact_admitted": False,
                        "training_tolerance_admitted": True,
                    }
                },
            },
            "calibration": {"segnet_operator_absmax": {"only.layer": 1.0}},
            "arms": {"fp32_control": {"segnet_rows": control_rows}},
        },
    )
    precursor_path = tmp_path / "mixed.json"
    _write_json(
        precursor_path,
        {
            "schema": "mixed_int64_fixedpoint_scorer_n600.v1",
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "argmax_exact_admitted": True,
                "minimum_bits": 26,
                "maximum_bits": 30,
            },
            "model_manifest": {
                "minimum_bits": 26,
                "maximum_bits": 30,
                "converted_conv2d_count": 125,
                "accumulation": "exact_signed_int64",
                "assignment_rule": "largest_geometry_safe_bits_with_signed_int64_static_bound",
            },
            "custody": {"qdq_precursor_sha256": _sha256(qdq_path)},
        },
    )
    calibration, _, mode, bits, precursor = GATE._load_calibration(
        qdq_path,
        bits=26,
        integer_precursor_path=precursor_path,
    )
    assert calibration == {"only.layer": 1.0}
    assert mode == "dynamic_exact_absmax"
    assert bits == 26
    assert precursor is not None
    assert precursor["schema"] == "mixed_int64_fixedpoint_scorer_n600.v1"


def test_weight_l1_precursor_requires_label_free_static_bound(tmp_path: Path) -> None:
    qdq_path = tmp_path / "qdq.json"
    control_rows = [
        {"pair_index": pair_index, "reference_argmax_sha256": "c" * 64}
        for pair_index in range(600)
    ]
    _write_json(
        qdq_path,
        {
            "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
            "contract": {"activation_scale_mode": "dynamic_exact_absmax"},
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "minimum_argmax_exact_arm": None,
                "arms": {
                    "w26a26": {
                        "argmax_exact_admitted": False,
                        "training_tolerance_admitted": True,
                    }
                },
            },
            "calibration": {"segnet_operator_absmax": {"only.layer": 1.0}},
            "arms": {"fp32_control": {"segnet_rows": control_rows}},
        },
    )
    precursor_path = tmp_path / "weight_l1.json"
    payload = {
        "schema": "weight_l1_int64_fixedpoint_scorer_n600.v1",
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "minimum_bits": 26,
            "maximum_bits": 31,
        },
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 31,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound",
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
        },
        "custody": {"qdq_precursor_sha256": _sha256(qdq_path)},
    }
    _write_json(precursor_path, payload)
    _, _, _, _, precursor = GATE._load_calibration(
        qdq_path,
        bits=26,
        integer_precursor_path=precursor_path,
    )
    assert precursor is not None
    payload["model_manifest"]["label_or_frame_dependent"] = True  # type: ignore[index]
    _write_json(precursor_path, payload)
    with pytest.raises(ValueError, match="manifest differs"):
        GATE._load_calibration(
            qdq_path,
            bits=26,
            integer_precursor_path=precursor_path,
        )


def test_tie_snap_precursor_requires_calibration_and_heldout_exactness(
    tmp_path: Path,
) -> None:
    qdq_path = tmp_path / "qdq.json"
    control_rows = [
        {"pair_index": pair_index, "reference_argmax_sha256": "c" * 64}
        for pair_index in range(600)
    ]
    _write_json(
        qdq_path,
        {
            "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
            "contract": {"activation_scale_mode": "dynamic_exact_absmax"},
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "minimum_argmax_exact_arm": None,
                "arms": {
                    "w26a26": {
                        "argmax_exact_admitted": False,
                        "training_tolerance_admitted": True,
                    }
                },
            },
            "calibration": {"segnet_operator_absmax": {"only.layer": 1.0}},
            "arms": {"fp32_control": {"segnet_rows": control_rows}},
        },
    )
    precursor_path = tmp_path / "tie_snap.json"
    payload = {
        "schema": "weight_l1_tie_snap_scorer_n600.v1",
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "minimum_calibration_exact_arm": "epsilon_2m19",
            "minimum_calibration_exact_epsilon": 2.0**-19,
            "selected_heldout_exact": True,
            "selected_full_exact": True,
        },
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 31,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": (
                "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            ),
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
        },
        "contract": {
            "decision_rule": "lowest class index within epsilon of candidate maximum",
            "epsilon_selection": (
                "minimum calibration-exact epsilon; no heldout reselection"
            ),
        },
        "custody": {"qdq_precursor_sha256": _sha256(qdq_path)},
    }
    _write_json(precursor_path, payload)
    _, _, _, _, precursor = GATE._load_calibration(
        qdq_path,
        bits=26,
        integer_precursor_path=precursor_path,
    )
    assert precursor is not None
    assert GATE._selected_tie_snap_epsilon(precursor) == 2.0**-19
    payload["summary"]["selected_heldout_exact"] = False  # type: ignore[index]
    _write_json(precursor_path, payload)
    with pytest.raises(ValueError, match="has not admitted"):
        GATE._load_calibration(
            qdq_path,
            bits=26,
            integer_precursor_path=precursor_path,
        )


def test_class_pair_tie_snap_requires_frozen_disjoint_validation(tmp_path: Path) -> None:
    qdq_path = tmp_path / "qdq.json"
    control_rows = [
        {"pair_index": pair_index, "reference_argmax_sha256": "c" * 64}
        for pair_index in range(600)
    ]
    _write_json(
        qdq_path,
        {
            "schema": "dynamic_fixedpoint_scorer_forward_n600.v1",
            "contract": {"activation_scale_mode": "dynamic_exact_absmax"},
            "summary": {
                "status": "MEASURED",
                "full_real_n600": True,
                "minimum_argmax_exact_arm": None,
                "arms": {
                    "w26a26": {
                        "argmax_exact_admitted": False,
                        "training_tolerance_admitted": True,
                    }
                },
            },
            "calibration": {"segnet_operator_absmax": {"only.layer": 1.0}},
            "arms": {"fp32_control": {"segnet_rows": control_rows}},
        },
    )
    precursor_path = tmp_path / "class_pair.json"
    payload = {
        "schema": "weight_l1_class_pair_tie_snap_scorer_n600.v1",
        "summary": {
            "status": "MEASURED",
            "full_real_n600": True,
            "argmax_exact_admitted": True,
            "design_exact": True,
            "second_validation_exact": True,
        },
        "model_manifest": {
            "minimum_bits": 26,
            "maximum_bits": 31,
            "converted_conv2d_count": 125,
            "accumulation": "exact_signed_int64",
            "assignment_rule": (
                "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            ),
            "bound_kind": "activation_qmax_times_max_output_quantized_weight_l1",
            "label_or_frame_dependent": False,
        },
        "contract": {
            "design_split": [0, 264],
            "second_validation_split": [264, 600],
            "epsilon": 2.0**-19,
            "candidate_winner_class": 4,
            "candidate_runner_class": 0,
            "replacement_class": 0,
            "rule_frozen_before_second_validation_access": True,
            "second_validation_reselection": False,
            "runtime_label_or_frame_dependent": False,
        },
        "custody": {"qdq_precursor_sha256": _sha256(qdq_path)},
    }
    _write_json(precursor_path, payload)
    _, _, _, _, precursor = GATE._load_calibration(
        qdq_path,
        bits=26,
        integer_precursor_path=precursor_path,
    )
    assert precursor is not None
    assert GATE._selected_tie_snap_rule(precursor) == {
        "kind": "ordered_class_pair",
        "epsilon": 2.0**-19,
        "winner_class": 4,
        "runner_class": 0,
        "replacement_class": 0,
    }
    payload["summary"]["second_validation_exact"] = False  # type: ignore[index]
    _write_json(precursor_path, payload)
    with pytest.raises(ValueError, match="has not admitted"):
        GATE._load_calibration(
            qdq_path,
            bits=26,
            integer_precursor_path=precursor_path,
        )


def test_full_corpus_exact_argmax_is_not_blocked_by_conservative_interval() -> None:
    fidelity = {
        "status": "MEASURED",
        "fidelity": True,
        "pair_start": 0,
        "pair_count": 600,
        "argmax_corpus_sha256": "a" * 64,
        "flips": 0,
        "aggregate_flip_fraction": 0.0,
        "worst_pair_flip_fraction": 0.0,
        "certificate": {"uncertified_pixels": 1},
        "timing": {"cpu_to_metal_speedup_x": 2.0},
    }
    replica = {
        "status": "MEASURED",
        "fidelity": False,
        "pair_start": 0,
        "pair_count": 600,
        "argmax_corpus_sha256": "a" * 64,
    }
    receipt = {
        "contract": {
            "n_processes": 2,
            "pair_start": 0,
            "pair_count": 600,
            "bits": 26,
            "activation_scale_mode": "dynamic_exact_absmax",
        },
        "trials": [fidelity, replica],
    }
    summary = GATE._summary(receipt)
    assert summary["argmax_exact"] is True
    assert summary["cross_process_argmax_identical"] is True
    assert summary["strict_interval_certified"] is False
    assert summary["strict_interval_certificate_required_for_admission"] is False
    assert summary["admitted_candidate_authority_filter"] is True
