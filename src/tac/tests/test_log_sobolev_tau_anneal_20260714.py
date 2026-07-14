from __future__ import annotations

import math

import numpy as np
import pytest

from tac.training_curriculum.log_sobolev_tau_anneal_20260714 import (
    audit_same_telemetry_crosscheck,
    derive_de_cfl_window,
    derive_lsi_tracking_bound,
    lsi_tau_rate_numpy_fp32,
)


def test_lsi_bound_derives_c_instead_of_guessing_it() -> None:
    bound = derive_lsi_tracking_bound(
        lambda_gap_per_epoch=0.08,
        contraction_rate_over_gap=0.5,
        tracking_radius=0.02,
        path_sensitivity_per_tau=0.4,
    )
    assert bound.contraction_rate_per_epoch == pytest.approx(0.04)
    assert bound.coefficient_c == pytest.approx(0.025)
    assert bound.max_abs_tau_rate == pytest.approx(0.002)


def test_numpy_fp32_is_authority_surface() -> None:
    got = lsi_tau_rate_numpy_fp32(
        lambda_gap_per_epoch=0.08,
        contraction_rate_over_gap=0.5,
        tracking_radius=0.02,
        path_sensitivity_per_tau=0.4,
    )
    assert isinstance(got, np.float32)
    assert float(got) == pytest.approx(0.002, rel=1e-6)


def test_de_window_is_static_and_matches_two_inequalities() -> None:
    window = derive_de_cfl_window(
        optimizer_step_eta=1e-3,
        lambda_eik=0.05,
        c_a_abs=1.0,
        epsilon=0.3,
        k_max=2.0,
    )
    assert window.pi_eik == pytest.approx(1e-3 * 0.05 / (8 * 0.3**2))
    assert window.pi_bih == pytest.approx(1e-3 * 0.05 * 0.3**2 * 2.0**4)
    assert window.static_cfl_admissible
    assert not window.produces_tau_rate_bound
    assert window.epsilon_lower == pytest.approx(math.sqrt(1e-3 * 0.05 / 8))


def _complete_payload() -> dict[str, object]:
    state = {
        "state_sha256": "a" * 64,
        "epoch": 100,
        "tau": 0.8,
        "axis": "local-numpy-fp32-research",
    }
    return {
        "paper_source_status": "FULL_TEXT_VERIFIED",
        "agreement_relative_tolerance": 0.1,
        "lsi": {
            **state,
            "lambda_gap_per_epoch": 0.08,
            "contraction_rate_over_gap": 0.5,
            "tracking_radius": 0.02,
            "path_sensitivity_per_tau": 0.4,
        },
        "de": {
            **state,
            "optimizer_step_eta": 1e-3,
            "lambda_eik": 0.05,
            "c_a_abs": 1.0,
            "epsilon": 0.3,
            "k_max": 2.0,
        },
        "de_tau_rate_bound": {
            **state,
            "derivation_id": "synthetic_test_de_tau_rate_v1",
            "units": "tau_per_epoch",
            "max_abs_tau_rate": 0.0021,
        },
    }


def test_crosscheck_can_report_agreement_only_with_same_state_rate_custody() -> None:
    result = audit_same_telemetry_crosscheck(_complete_payload())
    assert result["comparison_status"] == "AGREE"
    assert result["blockers"] == []


def test_crosscheck_refuses_static_de_law_as_tau_rate() -> None:
    payload = _complete_payload()
    payload.pop("de_tau_rate_bound")
    result = audit_same_telemetry_crosscheck(payload)
    assert result["comparison_status"] == "STRUCTURAL_AGREEMENT_ONLY"
    assert any("static optimizer/viscosity CFL" in item for item in result["blockers"])


def test_crosscheck_refuses_unverified_paper_and_cross_state_rows() -> None:
    payload = _complete_payload()
    payload["paper_source_status"] = "LOCAL_SUMMARY_ONLY"
    payload["de"]["state_sha256"] = "b" * 64  # type: ignore[index]
    result = audit_same_telemetry_crosscheck(payload)
    assert result["comparison_status"] == "NO_VERDICT_SOURCE_CUSTODY"
    assert any("same state" in item for item in result["blockers"])


def test_crosscheck_refuses_absent_state_identity_instead_of_matching_nulls() -> None:
    payload = _complete_payload()
    payload["lsi"]["tau"] = None  # type: ignore[index]
    payload["de"]["tau"] = None  # type: ignore[index]
    payload["de_tau_rate_bound"]["tau"] = None  # type: ignore[index]
    result = audit_same_telemetry_crosscheck(payload)
    assert result["comparison_status"] == "NO_VERDICT_DATA_CUSTODY"
    assert any("state identity missing" in item for item in result["blockers"])


def test_crosscheck_refuses_de_rate_row_without_value() -> None:
    payload = _complete_payload()
    payload["de_tau_rate_bound"].pop("max_abs_tau_rate")  # type: ignore[union-attr]
    result = audit_same_telemetry_crosscheck(payload)
    assert result["comparison_status"] == "NO_VERDICT_DATA_CUSTODY"
    assert "DE tau-rate bound lacks max_abs_tau_rate" in result["blockers"]


@pytest.mark.parametrize("name", ["lambda_gap_per_epoch", "tracking_radius"])
def test_invalid_lsi_values_fail_closed(name: str) -> None:
    kwargs = {
        "lambda_gap_per_epoch": 0.08,
        "contraction_rate_over_gap": 0.5,
        "tracking_radius": 0.02,
        "path_sensitivity_per_tau": 0.4,
    }
    kwargs[name] = 0.0
    with pytest.raises(ValueError):
        derive_lsi_tracking_bound(**kwargs)
