# SPDX-License-Identifier: MIT
"""No-scorer contract tests for the OSS-reconciled INSTANT harness."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "probe_instant_projected_adjoint_test",
    REPO / "tools/probe_instant_projected_adjoint.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_validation_economics_matches_operator_formula() -> None:
    observed = probe.validation_economics(
        k=4,
        exact_seconds=10.0,
        approximate_seconds=2.0,
        validate_seconds=1.0,
        fallback_seconds=0.5,
    )
    assert observed == pytest.approx(40.0 / (10.0 + 3.0 * 3.5))
    with_calibration = probe.validation_economics(
        k=4,
        exact_seconds=10.0,
        approximate_seconds=2.0,
        validate_seconds=1.0,
        fallback_seconds=0.5,
        calibration_seconds=4.0,
    )
    assert with_calibration == pytest.approx(40.0 / (4.0 + 10.0 + 3.0 * 3.5))
    with pytest.raises(ValueError, match="invalid"):
        probe.validation_economics(k=0, exact_seconds=1.0, approximate_seconds=1.0)


def test_paired_speedup_emits_median_mad_lower_bound() -> None:
    observed = probe._ratio_timing([2.0, 3.0, 4.0], [1.0, 2.0, 2.0])
    assert observed["median_seconds"] == 2.0
    assert observed["mad_seconds"] == 0.0
    assert observed["conservative_lower_bound"] == 2.0
    with pytest.raises(ValueError, match="same sample count"):
        probe._ratio_timing([1.0], [1.0, 2.0])


def test_renderer_descent_direction_gate_has_positive_and_negative_canaries() -> None:
    exact = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    positive = probe.renderer_descent_direction_gate(exact, exact)
    negative = probe.renderer_descent_direction_gate(exact, -exact)
    assert positive["passed"] is True
    assert positive["cosine_similarity"] == pytest.approx(1.0)
    assert negative["passed"] is False
    assert negative["cosine_similarity"] == pytest.approx(-1.0)


def test_oss_eligibility_is_pointwise_ungrouped_only() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1),
        torch.nn.Conv2d(4, 4, 3, padding=1),
        torch.nn.Conv2d(4, 4, 1, groups=4),
    )
    assert set(probe._eligible_convolutions(model)) == {"0"}


def test_control_laws_are_finite_and_source_custody_names_local_adapter() -> None:
    assert probe.ENERGY_TARGETS == (0.90, 0.95, 0.99)
    assert probe.OVERSAMPLING == 5
    assert probe.VALIDATION_HORIZONS == (2, 4, 8)
    assert "src/tac/boundary_math/instant_projected_adjoint.py" in probe.SOURCE_PATHS
    assert "tools/probe_yopo_first_layer_costate.py" in probe.SOURCE_PATHS


def test_output_path_contract_is_durable_results_tree() -> None:
    parser_source = (REPO / "tools/probe_instant_projected_adjoint.py").read_text()
    assert "experiments/results" in parser_source
    assert "pointer_moved\": False" in parser_source
    assert "live_trainer_touched\": False" in parser_source


def _run_binding() -> dict[str, object]:
    return {
        "schema": "instant_run_binding.v1",
        "run_manifest_sha256": "a" * 64,
        "source_custody_sha256": probe._json_sha256({"adapter.py": "e" * 64}),
        "input_custody_sha256": "c" * 64,
    }


def _arm(target: float, *, validation_seconds: float = 5.0) -> dict[str, object]:
    economics = probe.InstantAdmissionEconomics(
        exact_seconds=1.0,
        approximate_seconds=0.5,
        projected_candidate_validation_seconds=validation_seconds,
    ).to_dict()
    arm: dict[str, object] = {
        "energy_target": target,
        "oversampling": probe.OVERSAMPLING,
        "calibration": {"rank_after_oversampling": {"median": 7.0}},
        "timing": {"total": {"median_seconds": 0.5}},
        "paired_hot_step_speedup": {"conservative_lower_bound": 2.0},
        "admission_economics": economics,
        "direction": {"admission": True},
        "admitted": False,
    }
    arm["admitted"] = probe._arm_admitted(arm)
    return arm


def _stage(regime: dict[str, object]) -> dict[str, object]:
    return {
        "regime": regime,
        "renderer_parity_canary": {"max_abs": 0.0},
        "scorer_input_frame_sha256": "d" * 64,
        "dense": {"total": {"median_seconds": 1.0}},
        "arms": [_arm(target) for target in probe.ENERGY_TARGETS],
    }


def _terminal_payload() -> dict[str, object]:
    regimes = [_stage(regime) for regime in probe.REGIMES]
    return {
        "schema": probe.SCHEMA,
        "run_binding": _run_binding(),
        "source_custody": {"adapter.py": "e" * 64},
        "score_claim": False,
        "pointer_moved": False,
        "regimes": regimes,
        "verdict": {
            "verdict": "NO_GO",
            "admitted_regime_arms": [],
            "energy_targets_clearing_all_regimes": [],
        },
    }


def _set_nested(payload: object, path: tuple[object, ...], value: object) -> None:
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


def test_terminal_receipt_authentication_is_external_hash_bound_and_read_only(tmp_path) -> None:
    path = tmp_path / "measurement_receipt.json"
    payload = _terminal_payload()
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    assert probe._authenticate_terminal_receipt(
        path,
        expected_sha256=before,
        expected_run_binding=_run_binding(),
    ) == payload
    probe._derive_receipt_science(payload)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (("verdict", "verdict"), "GO"),
        (("regimes", 0, "regime", "epoch"), -1),
        (("regimes", 0, "arms", 0, "calibration", "rank_after_oversampling", "median"), 99.0),
        (("regimes", 0, "arms", 0, "timing", "total", "median_seconds"), 0.01),
        (("regimes", 0, "arms", 0, "admitted"), True),
    ],
)
def test_terminal_science_tamper_refuses_without_rewrite(
    tmp_path, field: tuple[object, ...], replacement: object
) -> None:
    path = tmp_path / "measurement_receipt.json"
    original = _terminal_payload()
    path.write_text(json.dumps(original, sort_keys=True), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    tampered = deepcopy(original)
    _set_nested(tampered, field, replacement)
    path.write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    tampered_before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="external SHA-256"):
        probe._authenticate_terminal_receipt(
            path,
            expected_sha256=expected,
            expected_run_binding=_run_binding(),
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == tampered_before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        (("payload", "stage_rows", 0, "timing_seconds"), 0.01),
        (("payload", "calibration", "metadata", "rank"), 99),
        (("payload", "calibration", "basis_sha256"), "f" * 64),
    ],
)
def test_partial_checkpoint_content_tamper_refuses_without_rewrite(
    tmp_path, field: tuple[object, ...], replacement: object
) -> None:
    path = tmp_path / "stage.json"
    identity = {"stage": "test"}
    envelope = probe.InstantRunCheckpointEnvelope(
        kind="test",
        run_binding=_run_binding(),
        identity=identity,
        payload={
            "stage_rows": [{"timing_seconds": 1.0}],
            "calibration": {"metadata": {"rank": 7}, "basis_sha256": "e" * 64},
        },
    )
    envelope.write(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    _set_nested(raw, field, replacement)
    path.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="content hash"):
        probe.InstantRunCheckpointEnvelope.load(
            path,
            expected_kind="test",
            expected_run_binding=_run_binding(),
            expected_identity=identity,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_charged_cycle_economics_refuses_fast_hot_step_canary() -> None:
    arm = _arm(0.95, validation_seconds=5.0)
    assert arm["paired_hot_step_speedup"]["conservative_lower_bound"] > 1.0
    economics = arm["admission_economics"]
    assert [row["K_steps"] for row in economics["cadences"]] == [2, 4, 8]
    assert economics["charged_terms"]["projected_candidate_validation_seconds"] == 5.0
    assert economics["admitted_cadences_K"] == []
    assert economics["decisive_no_go"] is True
    assert all(row["decisive_no_go"] for row in economics["cadences"])
    assert probe._arm_admitted(arm) is False


def test_charged_cycle_economics_rederives_reviewed_ratios() -> None:
    economics = probe.InstantAdmissionEconomics(
        exact_seconds=0.9511765830684453,
        approximate_seconds=0.8592235830146819,
        projected_candidate_validation_seconds=4.724031916004606,
    ).to_dict()
    assert [row["optimistic_upper_bound_ratio"] for row in economics["cadences"]] == pytest.approx(
        [0.29112754440460786, 0.21494370752174816, 0.19007391973448726]
    )
    assert economics["admitted_cadences_K"] == []


def test_stage_schema_rederives_admission_and_rejects_coverage_tamper() -> None:
    stage = _stage(probe.REGIMES[0])
    assert probe._validate_regime_stage_payload(
        stage, expected_regime=probe.REGIMES[0]
    ) == stage
    tampered = deepcopy(stage)
    tampered["arms"][0]["admitted"] = True
    with pytest.raises(ValueError, match="admission derivation"):
        probe._validate_regime_stage_payload(tampered, expected_regime=probe.REGIMES[0])
    missing = deepcopy(stage)
    missing["arms"].pop()
    with pytest.raises(ValueError, match="arm coverage"):
        probe._validate_regime_stage_payload(missing, expected_regime=probe.REGIMES[0])
