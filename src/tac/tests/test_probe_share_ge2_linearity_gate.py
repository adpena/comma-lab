"""Focused tests for the fail-closed SHARE_GE2 diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("share_ge2_probe", ROOT / "tools/probe_share_ge2_linearity_gate.py")
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


def test_share_ge2_known_values_and_no_clamp() -> None:
    assert probe.share_ge2(0.5, [0.25]) == pytest.approx(0.0)
    assert probe.share_ge2(1.0, [3.0]) == pytest.approx(2.0 / 3.0)
    assert probe.share_ge2(2.0, [1.0]) == pytest.approx(-3.0)


@pytest.mark.parametrize(
    ("tau", "betas"),
    [(-1.0, [1.0]), (float("nan"), [1.0]), (0.5, []), (0.5, [0.0]), (0.5, [-1.0])],
)
def test_share_ge2_rejects_undefined_domain(tau: float, betas: list[float]) -> None:
    with pytest.raises(ValueError):
        probe.share_ge2(tau, betas)


def test_validation_economics_includes_all_companion_costs() -> None:
    assert probe.validation_economics(K=2, t_exact=10.0, t_approx=1.0, t_validate=4.0, t_fallback=5.0) == 1.0
    with pytest.raises(ValueError):
        probe.validation_economics(K=0, t_exact=1.0, t_approx=0.0, t_validate=0.0, t_fallback=0.0)


def test_mapping_canaries_expose_nonidentifiability() -> None:
    canaries = probe._formula_canaries()
    assert all(row["status"] == "PASS" for row in canaries.values())
    ambiguity = canaries["negative_undefined_beta_scalarization"]
    assert ambiguity["amplitude_excess_candidate"]["share_ge2"] != ambiguity["energy_excess_candidate"]["share_ge2"]
    operator = canaries["negative_operator_norm_composition"]
    assert operator["product_of_layer_spectral_norms"] != operator["spectral_norm_of_BA"]


def test_underexercised_k_gt_1_has_no_invented_economics_companion() -> None:
    source = probe.json.loads(probe.DEFAULT_YOPO_RECEIPT.read_text())
    for regime in ("early", "boundary"):
        k4 = next(row for row in probe._arm_economics(source["regimes"][regime]) if row["K"] == 4)
        assert k4["companion_ratio"] is None
        assert "no nonrefresh step" in k4["timing_basis"]


def test_build_receipt_fails_closed_and_rederives_sealed_counts() -> None:
    receipt = probe.build_receipt(probe.DEFAULT_YOPO_RECEIPT, own_round1=False)
    assert receipt["mapping_assessment"]["status"] == "UNVERIFIED_FAIL_CLOSED"
    assert receipt["mapping_assessment"]["o_l_segnet_forward_mapping"] == "SKIPPED"
    assert all(row["share_ge2"] is None for row in receipt["sealed_checkpoint_results"].values())
    assert all(row["tau_receipt_match"] for row in receipt["sealed_checkpoint_results"].values())
    assert all(
        row["tau_measurement_status"] == "MEASURED_DIRECT_NPZ_PARSE"
        for row in receipt["sealed_checkpoint_results"].values()
    )
    assert receipt["custody"]["all_declared_inputs_match"] is True
    assert receipt["custody"]["yopo_receipt"]["status"] == "PASS"
    assert receipt["authority"]["pointer_score_read_only"] == pytest.approx(0.1880443979880752)
    assert receipt["authority"]["pointer_sha256_before"] == receipt["authority"]["pointer_sha256_after"]
    counts = receipt["fallback_exact_teacher"]["work_counts_rederived"]
    assert counts["operational_validation_forwards_including_labels"] == 402
    assert counts["all_teacher_forward_backward_including_labels"] == 48
    assert counts["measurement_only_control_forwards_including_labels"] == 44


def test_persisted_receipt_binds_current_probe_and_unknown_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "share"
    monkeypatch.setattr(
        sys,
        "argv",
        [str(probe.Path(probe.__file__)), "--output-dir", str(output_dir)],
    )
    assert probe.main() == 0
    persisted = probe.json.loads((output_dir / "receipt.json").read_text())
    assert persisted["custody"]["probe"]["sha256"] == probe._sha256(probe.Path(probe.__file__))
    for regime in ("early", "boundary"):
        k4 = next(row for row in persisted["fallback_exact_teacher"]["by_regime"][regime] if row["K"] == 4)
        assert k4["companion_ratio"] is None
