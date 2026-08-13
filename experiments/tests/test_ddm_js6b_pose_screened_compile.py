from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_js6b_pose_screened_compile as js6b
from experiments import ddm_re1t_t4_sign_gate_worker as worker
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_q3_projector_is_rank_six_idempotent_and_null() -> None:
    constraint, projector = js6b.q3_constraint_and_projector()
    assert constraint.shape == (6, 12)
    assert projector.shape == (12, 12)
    assert np.linalg.matrix_rank(projector) == 6
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-12)
    np.testing.assert_allclose(constraint @ projector, 0.0, atol=1e-12)


def test_q3_components_reconstruct_input_and_retain_visible_component() -> None:
    rng = np.random.default_rng(837)
    delta = rng.normal(size=(4, 6, 3)).astype(np.float32)
    null, visible, diagnostics = js6b.q3_components(delta)
    assert null.dtype == np.float32
    assert visible.dtype == np.float32
    np.testing.assert_allclose(null + visible, delta, atol=2e-7)
    assert diagnostics["q3_float_constraint_max_abs"] < 1e-6
    assert 0.0 <= diagnostics["q3_pose_visible_energy_fraction"] <= 1.000001


def test_q3_components_rejects_odd_non_rgb_and_nonfinite_inputs() -> None:
    with pytest.raises(js6b.JS6BError, match="even scorer lattice"):
        js6b.q3_components(np.zeros((3, 4, 3), dtype=np.float32))
    with pytest.raises(js6b.JS6BError, match="HWC RGB"):
        js6b.q3_components(np.zeros((4, 4, 2), dtype=np.float32))
    invalid = np.zeros((4, 4, 3), dtype=np.float32)
    invalid[0, 0, 0] = np.nan
    with pytest.raises(js6b.JS6BError, match="finite float32"):
        js6b.q3_components(invalid)


def test_screen_uses_target_mass_and_measured_semantic_cell_envelope() -> None:
    result = js6b.screen_arithmetic(target_mass=33, semantic_cells=5)
    assert result["optimistic_seg_value_s"] == pytest.approx(33 * js6b.S_PER_FLIP)
    assert result["measured_pose_risk_lower_s"] == pytest.approx(5 * 5.7e-6)
    assert result["measured_pose_risk_upper_s"] == pytest.approx(5 * 3.4e-5)
    assert result["screened_net_delta_s_lower_zero_rate"] > 0.0
    assert result["admitted"] is False


def test_screen_rejects_invalid_count_geometry() -> None:
    with pytest.raises(js6b.JS6BError, match="screen counts"):
        js6b.screen_arithmetic(target_mass=-1, semantic_cells=1)
    with pytest.raises(js6b.JS6BError, match="screen counts"):
        js6b.screen_arithmetic(target_mass=1, semantic_cells=0)


def test_retain_bytes_is_resumable_but_refuses_payload_drift(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    first = js6b.retain_bytes(path, b"kept")
    assert js6b.retain_bytes(path, b"kept") == first
    with pytest.raises(js6b.JS6BError, match="refusing to replace"):
        js6b.retain_bytes(path, b"different")


def test_screen_one_retains_both_q3_payloads_and_resumes(tmp_path: Path) -> None:
    bank = tmp_path / "bank"
    proposal = bank / "proposals/p1"
    proposal.mkdir(parents=True)
    delta_path = proposal / "scorer_delta.float32.npy"
    rng = np.random.default_rng(6)
    np.save(delta_path, rng.normal(size=(3, js6b.HEIGHT, js6b.WIDTH)).astype(np.float32))
    row = {
        "proposal_id": "p1",
        "ordinal": 1,
        "pair": 42,
        "directed_edge": "Road->Lane",
        "token_site_count": 1,
        "receiver_surface": {"exact_field_target_edge_mass_on_support": 20},
        "retained_payloads": {
            "scorer_delta.float32.npy": js6b.file_record(delta_path),
        },
    }
    output = tmp_path / "output"
    result = js6b.screen_one(row, output, bank)
    assert result["disposition"] == "HELD"
    assert result["source_delta_geometry"] == "CHW"
    assert result["q3_required_after_lower_calibration"] is True
    assert result["screen"]["admitted"] is False
    for key in ("q3_null_delta", "q3_pose_visible_delta"):
        assert js6b.require_record(result[key], beneath=output).is_file()
    assert js6b.screen_one(row, output, bank) == result


def test_pose_measurement_reports_both_passes_and_repeat_noise() -> None:
    gt = np.zeros((600, 6), dtype=np.float32)
    first = np.full((600, 6), 2.0, dtype=np.float32)
    repeat = np.full((600, 6), 3.0, dtype=np.float32)
    result = worker.pose_measurement(gt, first, repeat)
    assert result["d_pose_candidate_first"] == pytest.approx(4.0)
    assert result["d_pose_candidate_repeat"] == pytest.approx(9.0)
    assert result["repeat_noise_mse"] == pytest.approx(1.0)
    assert result["pair_error_rms"].shape == (600,)
    assert result["pair_repeat_noise_rms"].shape == (600,)
    assert result["adjudicated_remotely"] is False


def test_pose_measurement_fails_closed_on_shape_or_nonfinite() -> None:
    good = np.zeros((600, 6), dtype=np.float32)
    with pytest.raises(worker.RE1TWorkerError, match="shape"):
        worker.pose_measurement(good[:-1], good, good)
    bad = good.copy()
    bad[0, 0] = np.inf
    with pytest.raises(worker.RE1TWorkerError, match="finite"):
        worker.pose_measurement(bad, good, good)


def test_screen_result_schema_never_mislabels_local_prior_as_score(tmp_path: Path) -> None:
    row = js6b.screen_arithmetic(target_mass=6, semantic_cells=2)
    receipt = {
        "axis": js6b.AXIS,
        "score_claim": False,
        "promotion_eligible": False,
        "screen": row,
    }
    path = tmp_path / "receipt.json"
    js6b.retain_json(path, receipt)
    loaded = json.loads(path.read_text())
    assert loaded["score_claim"] is False
    assert loaded["promotion_eligible"] is False
    assert "scorer-free" in loaded["axis"]


def test_new_paths_do_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=js6b.REPO,
        roots=[
            Path("experiments/ddm_js6b_pose_screened_compile.py"),
            Path("experiments/ddm_re1t_t4_sign_gate_worker.py"),
        ],
    )
    assert findings == []
