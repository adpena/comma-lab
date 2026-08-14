from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_pk4_frame0_pose_overlay_runtime as overlay
from experiments import ddm_pk4_optimal_form_frame0_pose as pk4


def test_p0j2_exact_three_rung_lengths_and_roundtrip() -> None:
    expected = {6: 43, 40: 247, 165: 997}
    for knots, byte_count in expected.items():
        controls = np.zeros((knots, overlay.DIMENSIONS), dtype=np.int32)
        controls[0, 0] = overlay.MIN_CONTROL
        controls[-1, -1] = overlay.MAX_CONTROL
        payload = overlay.encode_pose_overlay(controls)
        assert len(payload) == byte_count
        assert overlay.encoded_bytes_for_knots(knots) == byte_count
        np.testing.assert_array_equal(overlay.decode_pose_overlay(payload), controls)


def test_p0j2_rejects_alias_trailing_bytes_and_reserved_nibble() -> None:
    payload = overlay.encode_pose_overlay(np.zeros((6, 12), dtype=np.int32))
    with pytest.raises(overlay.Frame0PoseOverlayError):
        overlay.decode_pose_overlay(payload + b"\x00")
    corrupted = bytearray(payload)
    corrupted[overlay._HEADER.size] = 0xF0
    with pytest.raises(overlay.Frame0PoseOverlayError):
        overlay.decode_pose_overlay(bytes(corrupted))


def test_p0j2_expansion_is_integer_deterministic_and_int12_checked() -> None:
    controls = np.zeros((6, 12), dtype=np.int32)
    controls[0, 0] = -7
    controls[-1, 0] = 7
    first = overlay.expand_pose_controls(controls)
    second = overlay.expand_pose_controls(controls.copy())
    np.testing.assert_array_equal(first, second)
    assert first.dtype == np.int32
    assert first.shape == (600, 12)
    assert first[0, 0] == -7
    assert first[-1, 0] == 7
    base = np.zeros((600, 12), dtype=np.int32)
    np.testing.assert_array_equal(
        overlay.apply_compensation_overlay(base, overlay.encode_pose_overlay(controls)), first
    )
    base[0, 0] = -2048
    with pytest.raises(overlay.Frame0PoseOverlayError):
        overlay.apply_compensation_overlay(base, overlay.encode_pose_overlay(controls))


def test_sample_plan_is_seeded_stratified_random_and_never_prefix() -> None:
    first = pk4.sample_plan(64)
    second = pk4.sample_plan(64)
    assert first == second
    selected = first["train_pairs"] + first["holdout_pairs"]
    assert len(selected) == 64
    assert len(set(selected)) == 64
    assert set(first["train_pairs"]).isdisjoint(first["holdout_pairs"])
    assert sorted(selected) != list(range(64))
    assert first["train_denominator"] == 48
    assert first["holdout_denominator"] == 16
    for stratum in range(16):
        lo = stratum * 600 // 16
        hi = (stratum + 1) * 600 // 16
        assert sum(lo <= pair < hi for pair in selected) == 4
        assert sum(lo <= pair < hi for pair in first["holdout_pairs"]) == 1


def test_temporal_fit_consumes_only_explicit_pair_targets() -> None:
    pairs = np.asarray([0, 599], dtype=np.int16)
    targets = np.zeros((2, 12), dtype=np.int32)
    targets[0, 0] = -4
    targets[1, 0] = 4
    controls = pk4.fit_temporal_controls(
        pairs, targets, knots=2, ridge=1e-12, gain=1.0
    )
    expanded = overlay.expand_pose_controls(controls)
    assert expanded[0, 0] == -4
    assert expanded[-1, 0] == 4


def test_generalization_gate_requires_positive_two_noise_rms() -> None:
    gt = np.zeros((4, 6), dtype=np.float64)
    base = np.ones((4, 6), dtype=np.float64)
    candidate = np.full((4, 6), 0.5, dtype=np.float64)
    passed = pk4.generalization_gate(
        base, candidate, base.copy(), candidate.copy(), gt,
        lopo_modeled_pose_mse_reduction=0.1,
    )
    assert passed["holdout_denominator"] == 4
    assert passed["heldout_mean_pose_mse_reduction"] == pytest.approx(0.75)
    assert passed["pair_noise_rms_from_exact_repeat"] == 0.0
    assert passed["passed"] is True

    noisy_repeat = np.full((4, 6), 1.5, dtype=np.float64)
    failed = pk4.generalization_gate(
        base, candidate, base.copy(), noisy_repeat, gt,
        lopo_modeled_pose_mse_reduction=0.1,
    )
    assert failed["heldout_mean_pose_mse_reduction"] > 0.0
    assert failed["two_sigma_threshold"] > failed["heldout_mean_pose_mse_reduction"]
    assert failed["passed"] is False
    assert failed["disposition"] == "GATE_FAIL_NO_COMPILE"

    lopo_failed = pk4.generalization_gate(
        base, candidate, base.copy(), candidate.copy(), gt,
        lopo_modeled_pose_mse_reduction=-1e-9,
    )
    assert lopo_failed["heldout_mean_pose_mse_reduction"] > 0.0
    assert lopo_failed["lopo_positive"] is False
    assert lopo_failed["passed"] is False


def test_no_fire_orders_are_per_rung_and_payload_law_is_explicit(tmp_path: Path) -> None:
    pk4.write_no_fire_orders(tmp_path, "no Metal", "BLOCKED_BEFORE_SCORER_LAUNCH_NO_METAL")
    for label, knots in pk4.RUNG_KNOTS.items():
        path = tmp_path / "retained/rungs" / label / "SEALED_NO_FIRE_ORDER_PREFLIGHT.json"
        value = __import__("json").loads(path.read_text())
        assert value["rung"] == label
        assert value["raw_overlay_target_bytes"] == overlay.encoded_bytes_for_knots(knots)
        assert value["owner"] == "MAIN"
        assert value["consumer_store"] == str(tmp_path.resolve())
        assert value["fire_trigger"].startswith("NONE")


def test_fire_request_helper_structurally_hardcodes_pose_unknown() -> None:
    source = Path(pk4.__file__).read_text()
    module = ast.parse(source)
    function = next(
        node for node in module.body if isinstance(node, ast.FunctionDef) and node.name == "seal_t4_order"
    )
    literals = [node.value for node in ast.walk(function) if isinstance(node, ast.Constant)]
    assert "local_pose_delta" in literals
    assert 0.0 in literals
    assert "pose_unmeasured" in literals
    assert True in literals
    assert "experiments/ddm_re1t_t4_sign_gate_worker.py" in literals


def test_compile_entrypoint_rejects_failed_gate_before_materializing_bytes(tmp_path: Path) -> None:
    gate = {
        "schema": "ddm_pk4_generalization_gate.v1",
        "rung": "rung_42",
        "passed": False,
        "lopo_positive": True,
    }
    record = pk4.retain_json(tmp_path / "FAILED_GATE.json", gate)
    controls = np.zeros((6, 12), dtype=np.int32)
    with pytest.raises(pk4.PK4Error, match="not compile-eligible"):
        pk4.compile_rung(tmp_path, "rung_42", controls, None, record)  # type: ignore[arg-type]
    assert not (tmp_path / "retained/rungs/rung_42/compiled").exists()


def test_measure_checks_metal_and_ownership_before_scorer_construction() -> None:
    source = Path(pk4.__file__).read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "measure"
    )
    calls = [
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.index("probe_metal") < calls.index("validate_ownership_receipt")
    assert calls.index("validate_ownership_receipt") < calls.index("build_bank")
