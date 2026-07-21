from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "measure_per_stratum_recursive_fractal_optimal.py"
SPEC = importlib.util.spec_from_file_location("per_stratum_measure", MODULE_PATH)
assert SPEC and SPEC.loader
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


def test_cap_arithmetic_and_exact_archive_ratios() -> None:
    result = measure.cap_accounting()
    assert result["rate_component_exact"] == "3865000/37545489"
    assert result["remaining_sub_0_15_distortion_budget_exact"] == ("35336467/750909780")
    assert result["rate_component"] == pytest.approx(25 * 154_600 / 37_545_489)
    assert measure._archive_accounting(90_566)["cap_ratio_exact"] == "45283/77300"
    assert measure._archive_accounting(451_191)["cap_ratio"] == pytest.approx(2.9184411384217337)


def test_vp_sensitivity_uses_measured_p50_and_labels_causal_null() -> None:
    result = measure.vp_sensitivity(
        road_p50_pixels=39.02261829831544,
        lane_p50_pixels=47.119248012565066,
    )
    assert result["nominal_vanishing_point"]["v"] == pytest.approx(174.0)
    displacement = result["threshold_corner_displacement_pixels"]
    assert displacement["horizontal"] == pytest.approx(13.978784027546617)
    assert displacement["vertical"] == pytest.approx(28.10903444450031)
    assert displacement["euclidean"] == pytest.approx(31.393060065735824)
    road = result["minimum_one_axis_angular_equivalents"]["Road"]
    lane = result["minimum_one_axis_angular_equivalents"]["Lane"]
    assert road["pitch_delta_deg"] == pytest.approx(5.54345, abs=1e-5)
    assert road["yaw_delta_deg"] == pytest.approx(5.56780, abs=1e-5)
    assert lane["pitch_delta_deg"] == pytest.approx(6.67811, abs=1e-5)
    assert lane["yaw_delta_deg"] == pytest.approx(6.71338, abs=1e-5)
    assert result["calibration_explained_fraction"] is None
    assert result["genuine_geometry_fraction"] is None
    assert result["verdict"] == "UNIDENTIFIABLE_FROM_CURRENT_CUSTODY"


def test_generic_and_real_scorer_k_identity_homography_canaries() -> None:
    result = measure.scorer_k_identity_canary(
        lambda homography: {
            "R": homography,
            "t": (0.0, 0.0, 0.0),
            "pose": (0.0,) * 6,
        }
    )
    assert result["verdict"] == "PASS_IDENTITY_CANARY_ONLY"
    assert result["defaults_used"] is False
    with pytest.raises(measure.CustodyError):
        measure.scorer_k_identity_canary(
            lambda homography: {
                "R": homography,
                "t": (0.0, 0.1, 0.0),
                "pose": (0.0,) * 6,
            }
        )
    real = measure.calibrated_geometry_identity_canary()
    assert real["api"].startswith("CalibratedGeometry.homography_to_pose")
    assert real["max_pose_error"] == 0.0


def _stage(*, frame: int, nonzero_rotation: bool = False) -> dict[str, object]:
    rotation_term = 0.25 if nonzero_rotation else 0.0
    pose = [
        [1.0, -rotation_term, 0.0, float(frame)],
        [rotation_term, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    return {
        "frame": frame,
        "absolute_f1_pose": pose,
        "calibrated_cross_xi": [0.0, 0.0, 0.0, 0.0, 0.0, rotation_term],
        "calibrated_within_xi": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_rotation_audit_reads_4x4_and_translation_first_twists() -> None:
    identity = measure.audit_rotation_stages([_stage(frame=index) for index in range(600)])
    assert identity["rotation_observation_verdict"] == "IDENTITY_ONLY_STORED_ROTATION"
    assert identity["max_absolute_translation_magnitude"] == 599.0
    assert identity["nonzero_rotation_transition_vectors"] == 0
    nonzero = measure.audit_rotation_stages([_stage(frame=index, nonzero_rotation=index == 17) for index in range(600)])
    assert nonzero["nonidentity_absolute_rotation_frames"] == 1
    assert nonzero["nonzero_cross_rotation_transitions"] == 1
    assert nonzero["max_twist_rotation_vector_magnitude"] == 0.25
    assert nonzero["rotation_observation_verdict"] == "NONZERO_STORED_ROTATION"


def test_missing_observed_homography_forces_null_attribution() -> None:
    result = measure.audit_rotation_stages([_stage(frame=index) for index in range(600)])
    assert result["observed_pixel_homography_count"] == 0
    assert result["calibration_explained_fraction"] is None
    assert result["genuine_geometry_fraction"] is None


def _real_m1_rows() -> list[dict[str, object]]:
    return [
        {"class_name": "Road", "gt_pixels": 27_407_043, "mismatch_pixels": 138_575},
        {"class_name": "Lane", "gt_pixels": 690_639, "mismatch_pixels": 149_028},
        {"class_name": "Undriv", "gt_pixels": 58_413_282, "mismatch_pixels": 52_630},
        {"class_name": "Movable", "gt_pixels": 1_460_325, "mismatch_pixels": 52_403},
        {"class_name": "MyCar", "gt_pixels": 29_993_511, "mismatch_pixels": 22_104},
    ]


def test_per_class_hard_oracle_accounting_and_treatments_are_exact() -> None:
    result = measure.per_class_accounting(_real_m1_rows(), 117_964_800)
    assert result["aggregate_gt_pixels"] == 117_964_800
    assert result["aggregate_mismatch_pixels"] == 414_740
    assert result["aggregate_d_seg"] == 0.0035157945421006942
    assert result["aggregate_d_seg_exact"] == {
        "numerator": 414_740,
        "denominator": 117_964_800,
        "reduced": "20737/5898240",
    }
    assert [row["class_name"] for row in result["rows"]] == list(measure.CLASS_NAMES)
    assert all(row["measured_unique_home_bytes"] is None for row in result["rows"])
    assert all("SETTLED_RECALL" in row["claim_kind"] for row in result["rows"])
    assert result["rows"][1]["m1_hard_oracle"]["mismatch_pixels"] == 149_028
    with pytest.raises(measure.CustodyError):
        measure.per_class_accounting([*_real_m1_rows(), _real_m1_rows()[0]], 117_964_800)


def test_current_v9_missing_surfaces_never_become_zero_byte_measurements() -> None:
    result = measure.requested_v9_row()
    assert result["total_archive_bytes"] is None
    assert all(value is None for value in result["dimension_bytes"].values())
    assert all(value is None for value in result["per_stratum_bytes"].values())
    assert result["diagnostic_is_archive"] is False
    assert result["historical_65172_byte_diagnostic"]["bytes"] == 65_172
    assert result["verdict"] == "NO_VERDICT_RECEIVER_RATE_CUSTODY"


def test_atomic_and_compact_receipt_bytes_are_deterministic(tmp_path: Path) -> None:
    payload = {
        "road_lane_ground_frame": {
            "stage_manifest": {
                "entry_count": 1,
                "sha256": "a" * 64,
                "entries": [{"path": "frame_0000.json", "sha256": "b" * 64}],
            }
        },
        "z": [1, 2],
        "a": None,
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    measure._atomic_json(first, payload)
    measure._atomic_json(second, payload)
    assert first.read_bytes() == second.read_bytes()
    compact_a = measure.compact_receipt(payload)
    compact_b = measure.compact_receipt(payload)
    assert json.dumps(compact_a, sort_keys=True) == json.dumps(compact_b, sort_keys=True)
    assert compact_a["road_lane_ground_frame"]["stage_manifest"]["entries_omitted_from_compact_receipt"] == 1


def test_storage_preflight_result_redacts_volatile_free_space(monkeypatch: pytest.MonkeyPatch) -> None:
    class Usage:
        free = 1 << 30

    monkeypatch.setattr(measure.shutil, "disk_usage", lambda _: Usage())
    output = Path("/Volumes/VertigoDataTier/pact/evidence/per_stratum_recursive_fractal_20260721/receipt.json")
    first = measure._storage_preflight(output)
    Usage.free = 2 << 30
    second = measure._storage_preflight(output)
    assert first == second
    assert "free_bytes" not in first
    assert first["observed_free_bytes_at_least_required"] is True


def test_c2_complete_sum_is_exact_to_receipt_precision() -> None:
    class_sum = sum(value for bucket_values in measure.C2_BUCKETS.values() for _, value in bucket_values)
    assert math.isclose(class_sum + 0.000100 + 0.000150, 0.01328, abs_tol=1e-15)
