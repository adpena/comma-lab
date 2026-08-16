from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_qs1_frame0_schur_coupled_solve as qs1
from experiments import ddm_qs1_modal_t4_dual_axis as qs1_dispatch
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_cancellation_metrics_reports_energy_and_norm_fractions() -> None:
    result = qs1.cancellation_metrics(
        np.array([2.0, 0, 0, 0, 0, 0]),
        np.array([1.0, 0, 0, 0, 0, 0]),
    )
    assert result["cancellation_energy_fraction"] == pytest.approx(0.75)
    assert result["cancellation_norm_fraction"] == pytest.approx(0.5)


def test_conservative_pose_bound_dominates_exact_quadratic_delta() -> None:
    error = np.array([1.0, -2.0, 0.2, 0.0, 3.0, -0.5])
    residual = np.array([-0.1, 0.4, 0.2, -0.3, -0.6, 0.1])
    exact = (2.0 * error @ residual + residual @ residual) / (600 * 6)
    assert qs1.conservative_dpose_increase_bound(error, residual) >= exact


def test_admission_screen_prices_pose_and_rate_before_admission() -> None:
    held = qs1.admission_screen(
        seg_value_s=2e-5, residual_pose_bound_s=1.8e-5, delta_bytes=10
    )
    assert held["rate_delta_s"] == pytest.approx(10 * qs1.RATE_S_PER_BYTE)
    assert held["admitted"] is False
    survivor = qs1.admission_screen(
        seg_value_s=3e-5, residual_pose_bound_s=1e-5, delta_bytes=0
    )
    assert survivor["admitted"] is True


def test_independent_survivor_selection_keeps_best_margin_per_pair() -> None:
    rows = [
        {"proposal_id": "a", "pair": 1, "screen": {"admitted": True, "screen_margin_s": 0.1}},
        {"proposal_id": "b", "pair": 1, "screen": {"admitted": True, "screen_margin_s": 0.2}},
        {"proposal_id": "c", "pair": 2, "screen": {"admitted": False, "screen_margin_s": 1.0}},
        {"proposal_id": "d", "pair": 3, "screen": {"admitted": True, "screen_margin_s": 0.05}},
    ]
    selected = qs1._selected_independent_survivors(rows)
    assert [row["proposal_id"] for row in selected] == ["b", "d"]


def test_cp135_group_positions_are_an_exact_pixel_partition() -> None:
    positions = np.concatenate(qs1._cp135_group_positions())
    assert positions.size == 384 * 512
    np.testing.assert_array_equal(np.sort(positions), np.arange(384 * 512))


def test_strict_descent_stops_after_a_full_nonimproving_integer_pass() -> None:
    target = np.zeros(qs1.DIMENSIONS, dtype=np.int32)
    initial = target.copy()
    initial[0] = 2
    visited: list[int] = []

    def evaluate(
        candidates: tuple[np.ndarray, ...], pass_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        visited.append(pass_index)
        objectives = np.asarray([float(np.sum(item.astype(float) ** 2)) for item in candidates])
        vectors = np.repeat(objectives[:, None].astype(np.float32), 6, axis=1)
        return vectors, objectives

    final, objective, passes, _ = qs1.strict_descent(initial, 4.0, evaluate)
    np.testing.assert_array_equal(final, target)
    assert objective == 0.0
    assert passes == 3
    assert visited == [0, 1, 2]


def test_retain_npy_resumes_but_refuses_payload_drift(tmp_path: Path) -> None:
    path = tmp_path / "payload.npy"
    first = qs1.retain_npy(path, np.arange(8, dtype=np.int32))
    assert qs1.retain_npy(path, np.arange(8, dtype=np.int32)) == first
    with pytest.raises(qs1.QS1Error, match="refusing to replace"):
        qs1.retain_npy(path, np.arange(9, dtype=np.int32))


def test_output_override_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(qs1.QS1Error, match="governed SSD store"):
        qs1.run(tmp_path)


def test_dispatcher_loads_and_verifies_the_sealed_input_census(tmp_path: Path) -> None:
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    inputs: dict[str, dict[str, int | str]] = {}
    for name, payload in {
        "candidate_archive.zip": b"archive",
        "candidate_runtime.zip": b"runtime",
        "POSE_SCREEN_RESULT.json": b"{}\n",
    }.items():
        (input_root / name).write_bytes(payload)
        inputs[name] = {
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    request = {
        "schema": "ddm_qs1_t4_dual_axis_request.v1",
        "run_id": "test-run",
        "resume_from": "test-run",
        "retain_pose_vectors": True,
        "inputs": inputs,
        # A dispatchable request must carry a locally MEASURED distortion axis
        # (tac.deploy.dispatch_axis_screen).  This fixture exercises the input
        # census, so it declares the cheapest real screen; before that gate
        # landed it declared none at all and loaded anyway.
        "local_pose_delta": -1.2e-05,
        "pose_unmeasured": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    payloads, loaded = qs1_dispatch.load_sealed_inputs(
        request_path, input_root, qs1.sha256_file(request_path)
    )
    assert loaded == request
    assert payloads["candidate_archive.zip"] == b"archive"


def test_new_runner_does_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=qs1.REPO,
        roots=[Path("experiments/ddm_qs1_frame0_schur_coupled_solve.py")],
    )
    assert findings == []
