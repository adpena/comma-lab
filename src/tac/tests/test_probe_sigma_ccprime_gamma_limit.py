from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO / "tools/probe_sigma_ccprime_gamma_limit.py"


def _load_tool():
    name = "_test_probe_sigma_ccprime_gamma_limit"
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = _load_tool()


def _preregistration(sigma_spec: str) -> dict:
    preregistration = {
        "schema_version": probe.PREREGISTRATION_SCHEMA,
        "experiment_id": "sigma-ab-immutable-fixture-001",
        "declared_treatment_only_diff": {
            "changed_paths": ["treatment.length_sigma_matrix"],
            "control": {"length_sigma_matrix": probe.PRESET_ALL_ONES},
            "treatment": {"length_sigma_matrix": sigma_spec},
        },
    }
    preregistration["content_address_sha256"] = probe.preregistration_content_address(preregistration)
    return preregistration


def _receipt(
    sigma_spec: str,
    *,
    preregistration: dict | None = None,
    pairs: int = 600,
    seed: str = "19",
) -> dict:
    custody = {
        "authority": {"cohort": "real-n600", "pair_count": pairs},
        "seed": seed,
        "order_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "optimizer_fingerprint": "optimizer-v1",
        "curriculum_fingerprint": "curriculum-v1",
        "init_ema_sha256": "c" * 64,
        "non_treatment_config_sha256": "config-v1",
        "data_fingerprint_sha256": "d" * 64,
    }
    trajectory = []
    for update in range(4):
        classes = {}
        for class_name in probe.CLASSES:
            decline = 0.10 if class_name in probe.RARE else 0.20
            value = 1.0 - decline * update
            classes[class_name] = {"all": value, "hard": value + 0.1, "easy": value - 0.1}
        trajectory.append({"update": update, "wall_time_seconds": float(update + 1), "d_seg_by_class": classes})
    return {
        "schema_version": probe.TRAJECTORY_SCHEMA,
        "custody": custody,
        "preregistration": preregistration or _preregistration(sigma_spec),
        "treatment": {"length_sigma_matrix": sigma_spec},
        "trajectory": trajectory,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_fitted_triangle_violation_is_blocked_and_closure_is_relaxed() -> None:
    result = probe.static_matrix_receipt(probe.PRESET_FITTED_20260707)

    assert result["surface_density_geometry"] == "scalar_pairwise_spatially_isotropic_scaled_euclidean"
    assert result["orientation_anisotropy_status"] == "ABSENT_FROM_SCALAR_SIGMA_ALONE"
    assert result["metric_admissibility"] == "BLOCKED_TRIANGLE_VIOLATION"
    violation = next(
        item for item in result["triangle_violations"]
        if item["direct_pair"] == ["Lane", "MyCar"] and item["via_class"] == "Undrivable"
    )
    assert violation["direct_sigma"] == pytest.approx(1.764344211480968)
    assert violation["via_sigma_sum"] == pytest.approx(1.7381986449045815)
    assert result["metric_closure"][1][4] == pytest.approx(1.7381986449045815)
    assert result["gamma_limit_status"] == "NOT_PROVEN_BY_THIS_STATIC_ANALYZER"
    assert result["launch_authorized"] is False


def test_fragility_is_metric_but_not_an_anisotropy_or_gamma_proof() -> None:
    result = probe.static_matrix_receipt(probe.PRESET_FRAGILITY_20260709)

    assert result["triangle_violations"] == []
    assert result["metric_admissibility"] == "METRIC_ADMISSIBLE_STATIC"
    assert result["orientation_anisotropy_status"] == "ABSENT_FROM_SCALAR_SIGMA_ALONE"
    assert result["gamma_limit_status"] == "NOT_PROVEN_BY_THIS_STATIC_ANALYZER"
    assert result["launch_authorized"] is False


def test_static_only_result_is_content_addressed_and_has_no_empirical_claim() -> None:
    result, admitted = probe.analyze(probe.PRESET_FITTED_20260707)

    assert admitted
    assert result["evidence_status"] == "DERIVED_STATIC_ONLY_NO_EMPIRICAL_CLAIM"
    assert result["launch_authorized"] is False
    address = result.pop("content_address_sha256")
    assert address == probe.canonical_sha256(result)


def test_matched_real_n600_receipts_report_convergence_gap_without_gamma_claim(tmp_path: Path) -> None:
    preregistration = _preregistration(probe.PRESET_FITTED_20260707)
    control = _receipt(probe.PRESET_ALL_ONES, preregistration=preregistration)
    treatment = _receipt(probe.PRESET_FITTED_20260707, preregistration=preregistration)
    for row in treatment["trajectory"]:
        for class_name in probe.RARE:
            for stratum in probe.STRATA:
                row["d_seg_by_class"][class_name][stratum] -= 0.06 * row["update"]
    control_path, treatment_path = tmp_path / "control.json", tmp_path / "sigma.json"
    _write(control_path, control)
    _write(treatment_path, treatment)

    result, admitted = probe.analyze(probe.PRESET_FITTED_20260707, control_path, treatment_path)

    assert admitted
    assert result["evidence_status"] == "MEASURED_FROM_SUPPLIED_REAL_N600_RECEIPTS"
    assert result["empirical_ab"]["comparison"]["instance_verdict"] == "CONVERGENCE_GAP_REDUCED"
    assert result["gamma_limit_status"] == "NOT_PROVEN_BY_THIS_ANALYZER"
    assert result["launch_authorized"] is False


@pytest.mark.parametrize(
    ("mutator", "needle"),
    [
        (lambda control, sigma: control["custody"].update({"authority": {"cohort": "fixture", "pair_count": 600}}), "real-n600"),
        (lambda control, sigma: sigma["custody"].update({"seed": "other"}), "custody mismatch"),
        (lambda control, sigma: sigma["treatment"].update({"length_sigma_matrix": "wrong"}), "length_sigma_matrix"),
    ],
)
def test_bad_empirical_receipts_fail_closed_without_claim(tmp_path: Path, mutator, needle: str) -> None:
    preregistration = _preregistration(probe.PRESET_FITTED_20260707)
    control = _receipt(probe.PRESET_ALL_ONES, preregistration=preregistration)
    treatment = _receipt(probe.PRESET_FITTED_20260707, preregistration=preregistration)
    mutator(control, treatment)
    control_path, treatment_path = tmp_path / "control.json", tmp_path / "sigma.json"
    _write(control_path, control)
    _write(treatment_path, treatment)

    result, admitted = probe.analyze(probe.PRESET_FITTED_20260707, control_path, treatment_path)

    assert not admitted
    assert result["evidence_status"] == "BLOCKED_NO_EMPIRICAL_CLAIM"
    assert result["owed_status"] == "OWED"
    assert needle in result["blocker"]
    assert result["launch_authorized"] is False


def test_preregistration_self_address_data_fingerprint_and_exact_diff_are_required(tmp_path: Path) -> None:
    preregistration = _preregistration(probe.PRESET_FITTED_20260707)
    control = _receipt(probe.PRESET_ALL_ONES, preregistration=preregistration)
    treatment = _receipt(probe.PRESET_FITTED_20260707, preregistration=preregistration)
    treatment["preregistration"]["content_address_sha256"] = "0" * 64
    control_path, treatment_path = tmp_path / "control.json", tmp_path / "sigma.json"
    _write(control_path, control)
    _write(treatment_path, treatment)

    result, admitted = probe.analyze(probe.PRESET_FITTED_20260707, control_path, treatment_path)

    assert not admitted
    assert "self-address" in result["blocker"]
    assert result["launch_authorized"] is False

    declared_extra_diff = _preregistration(probe.PRESET_FITTED_20260707)
    declared_extra_diff["declared_treatment_only_diff"]["changed_paths"].append("optimizer.lr")
    declared_extra_diff["content_address_sha256"] = probe.preregistration_content_address(declared_extra_diff)
    control = _receipt(probe.PRESET_ALL_ONES, preregistration=_preregistration(probe.PRESET_FITTED_20260707))
    treatment = _receipt(probe.PRESET_FITTED_20260707, preregistration=declared_extra_diff)
    _write(control_path, control)
    _write(treatment_path, treatment)
    result, admitted = probe.analyze(probe.PRESET_FITTED_20260707, control_path, treatment_path)
    assert not admitted
    assert "only treatment.length_sigma_matrix" in result["blocker"]


def test_data_fingerprint_mismatch_and_null_treatment_are_blocked(tmp_path: Path) -> None:
    preregistration = _preregistration(probe.PRESET_FITTED_20260707)
    control = _receipt(probe.PRESET_ALL_ONES, preregistration=preregistration)
    treatment = _receipt(probe.PRESET_FITTED_20260707, preregistration=preregistration)
    treatment["custody"]["data_fingerprint_sha256"] = "e" * 64
    control_path, treatment_path = tmp_path / "control.json", tmp_path / "sigma.json"
    _write(control_path, control)
    _write(treatment_path, treatment)

    result, admitted = probe.analyze(probe.PRESET_FITTED_20260707, control_path, treatment_path)

    assert not admitted
    assert "data fingerprint mismatch" in result["blocker"]
    assert result["launch_authorized"] is False

    result, admitted = probe.analyze(probe.PRESET_ALL_ONES, control_path, treatment_path)
    assert not admitted
    assert "distinct from all-ones" in result["blocker"]
    assert result["launch_authorized"] is False


def test_cli_writes_static_receipt_atomically(tmp_path: Path) -> None:
    output = tmp_path / "static.json"
    assert probe.main(["--sigma-spec", probe.PRESET_FITTED_20260707, "--output", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["static_sigma_analysis"]["metric_admissibility"] == "BLOCKED_TRIANGLE_VIOLATION"
    assert result["launch_authorized"] is False
