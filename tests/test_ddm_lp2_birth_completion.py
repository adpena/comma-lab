from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.optimization.ddm_lp2_birth_completion import (
    DEFAULT_EPOCHS_PER_GATE,
    GATE_KEY,
    LANE_CLASS_INDEX,
    SCHEMA,
    BirthCompletionTelemetryError,
    Qa91Inventory,
    evaluate_birth_completion,
    evaluate_from_paths,
    load_qa91_inventory,
    read_lane_betti0_trend,
    verdict_to_row,
)

# --- fixtures -------------------------------------------------------------------------------------

_INVENTORY = Qa91Inventory(
    betti0_gt_lane=985,
    super_nucleus_area_frac=0.9767,
    nucleus_threshold_px=5,
    source_schema="ddm_fp1_qa91_erased_lane.v1",
)

# The real ep399 endpoint window (fp1/QA91): births STILL RISING (birth_tail_slope 8.75 comp/gate).
_REAL_ENDPOINT_WINDOW = [(359, 441), (369, 467), (379, 472), (389, 473), (399, 476)]
# A synthetic plateau window: births effectively stopped, real markings still erased.
_PLATEAU_WINDOW = [(500, 700), (510, 701), (520, 700), (530, 701), (540, 700)]


def _write_qa91(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema": "ddm_fp1_qa91_erased_lane.v1",
        "burn_endpoint_topology": {"betti0_gt_lane": 985, "betti0_realized_lane_end_ep399": 476},
        "super_nucleus_area_frac": 0.9767,
        "nucleus_threshold_px": 5,
        "score_claim": False,
    }
    payload.update(overrides)
    p = tmp_path / "qa91_erased_lane.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _topo_row(epoch: int, realized_lane: int, *, gt_lane: int = 985) -> dict:
    gt = [77, gt_lane, 38, 134, 36]
    realized = [37, realized_lane, 41, 68, 36]
    return {
        "epoch": epoch,
        "event": "a1_gate",
        "topology_per_class": {
            "betti0_gt": gt,
            "betti0_realized": realized,
            "gt_components_erased": [8, gt_lane - realized_lane, 1, 53, 0],
            "smallest_surviving_gt_component_px": [1, 89, 1, 18, 49373],
        },
    }


def _write_telemetry(tmp_path: Path, window, *, gt_lane: int = 985, extra_noise: bool = True) -> Path:
    p = tmp_path / "telemetry.jsonl"
    lines = []
    if extra_noise:
        lines.append(json.dumps({"event": "start", "epoch": 0, "note": "no topology here"}))
    for epoch, realized in window:
        lines.append(json.dumps(_topo_row(epoch, realized, gt_lane=gt_lane)))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


# --- constants / class order ----------------------------------------------------------------------


def test_lane_index_is_comma10k_canonical() -> None:
    assert LANE_CLASS_INDEX == 1  # [Road, Lane, Undrivable, Movable, MyCar]


def test_gate_key_and_schema_are_stable() -> None:
    assert GATE_KEY == "birth_completion"
    assert SCHEMA == "ddm_lp2_birth_completion_key.v1"


# --- core verdict ---------------------------------------------------------------------------------


def test_rising_endpoint_does_not_fire() -> None:
    v = evaluate_birth_completion(_REAL_ENDPOINT_WINDOW, _INVENTORY)
    assert v.fired is False
    assert v.slope_le_epsilon is False
    assert v.fit.slope_comp_per_gate > 5.0  # ~8.75 comp/gate, still rising
    assert v.above_nucleus_erasure_persists is True  # 985-476 erased, super-nucleus dominates
    assert v.erased_count == 985 - 476


def test_plateau_with_persisting_erasure_fires() -> None:
    v = evaluate_birth_completion(_PLATEAU_WINDOW, _INVENTORY)
    assert v.fired is True
    assert v.slope_le_epsilon is True
    assert abs(v.fit.slope_comp_per_gate) < v.fit.epsilon_comp_per_gate
    assert v.above_nucleus_erased_estimate >= 1


def test_plateau_without_erasure_does_not_fire() -> None:
    # realized == gt everywhere -> no erased components -> persistence False even though slope flat.
    window = [(e, 985) for e in (500, 510, 520, 530, 540)]
    v = evaluate_birth_completion(window, _INVENTORY)
    assert v.slope_le_epsilon is True
    assert v.above_nucleus_erasure_persists is False
    assert v.fired is False


def test_persistence_false_when_super_nucleus_not_dominant() -> None:
    inv = Qa91Inventory(985, super_nucleus_area_frac=0.4, nucleus_threshold_px=5, source_schema="x")
    v = evaluate_birth_completion(_PLATEAU_WINDOW, inv)
    assert v.above_nucleus_erasure_persists is False
    assert v.fired is False


# --- epsilon derivation ---------------------------------------------------------------------------


def test_epsilon_uses_quantization_floor_on_perfectly_flat_window() -> None:
    flat = [(500, 700), (510, 700), (520, 700), (530, 700), (540, 700)]
    v = evaluate_birth_completion(flat, _INVENTORY)
    assert v.fit.se_slope_ols == pytest.approx(0.0, abs=1e-12)
    assert v.fit.se_slope_quant > 0.0
    assert v.fit.se_slope == pytest.approx(v.fit.se_slope_quant)
    assert v.fit.epsilon_comp_per_gate > 0.0  # floor prevents a zero band
    assert v.fired is True


def test_epsilon_uses_ols_residual_when_window_is_noisy() -> None:
    noisy = [(500, 400), (510, 600), (520, 300), (530, 700), (540, 450)]
    v = evaluate_birth_completion(noisy, _INVENTORY)
    assert v.fit.se_slope_ols > v.fit.se_slope_quant
    assert v.fit.se_slope == pytest.approx(v.fit.se_slope_ols)


def test_epsilon_scales_with_alpha() -> None:
    v_wide = evaluate_birth_completion(_REAL_ENDPOINT_WINDOW, _INVENTORY, alpha_one_sided=0.4)
    v_tight = evaluate_birth_completion(_REAL_ENDPOINT_WINDOW, _INVENTORY, alpha_one_sided=0.01)
    # smaller alpha -> larger t_crit -> wider epsilon band
    assert v_tight.fit.epsilon_comp_per_gate > v_wide.fit.epsilon_comp_per_gate


def test_provenance_labels_epsilon_and_alpha() -> None:
    v = evaluate_birth_completion(_PLATEAU_WINDOW, _INVENTORY)
    assert "DERIVED" in v.provenance["epsilon_comp_per_gate"]
    assert "STATED-CONFIDENCE" in v.provenance["alpha_one_sided"]
    assert "DERIVED-ESTIMATE" in v.provenance["above_nucleus_erased_estimate"]


# --- determinism ----------------------------------------------------------------------------------


def test_deterministic() -> None:
    a = verdict_to_row(evaluate_birth_completion(_PLATEAU_WINDOW, _INVENTORY))
    b = verdict_to_row(evaluate_birth_completion(_PLATEAU_WINDOW, _INVENTORY))
    assert a == b


def test_window_truncates_to_last_n_gates() -> None:
    long_trend = [(e, c) for e, c in zip(range(0, 200, 10), range(20, 40), strict=False)]
    v = evaluate_birth_completion(long_trend, _INVENTORY, window_gates=5)
    assert v.fit.n_points == 5
    assert v.window_epochs == (150, 160, 170, 180, 190)


# --- fail-closed ----------------------------------------------------------------------------------


def test_insufficient_points_raises() -> None:
    with pytest.raises(BirthCompletionTelemetryError, match="insufficient telemetry"):
        evaluate_birth_completion([(0, 20), (10, 25)], _INVENTORY)


def test_window_gates_below_min_raises() -> None:
    with pytest.raises(BirthCompletionTelemetryError, match="MIN_WINDOW_POINTS"):
        evaluate_birth_completion(_PLATEAU_WINDOW, _INVENTORY, window_gates=2)


def test_duplicate_epochs_raise() -> None:
    with pytest.raises(BirthCompletionTelemetryError, match="duplicate"):
        evaluate_birth_completion([(10, 20), (10, 21), (20, 22), (30, 23)], _INVENTORY)


def test_degenerate_sxx_guard_direct() -> None:
    # Duplicate epochs are caught upstream; exercise the _ols_slope_fit S_xx guard directly.
    import numpy as np

    from tac.optimization.ddm_lp2_birth_completion import _ols_slope_fit

    with pytest.raises(BirthCompletionTelemetryError, match="S_xx"):
        _ols_slope_fit(np.array([3.9, 3.9, 3.9]), np.array([5.0, 6.0, 7.0]))


def test_negative_alpha_raises() -> None:
    with pytest.raises(BirthCompletionTelemetryError, match="alpha_one_sided"):
        evaluate_birth_completion(_PLATEAU_WINDOW, _INVENTORY, alpha_one_sided=0.9)


def test_realized_exceeds_gt_raises() -> None:
    window = [(e, 1000) for e in (500, 510, 520, 530, 540)]
    with pytest.raises(BirthCompletionTelemetryError, match="exceeds betti0_gt"):
        evaluate_birth_completion(window, _INVENTORY)


# --- inventory loader -----------------------------------------------------------------------------


def test_load_inventory_ok(tmp_path: Path) -> None:
    inv = load_qa91_inventory(_write_qa91(tmp_path))
    assert inv.betti0_gt_lane == 985
    assert inv.super_nucleus_area_frac == pytest.approx(0.9767)
    assert inv.nucleus_threshold_px == 5


def test_load_inventory_bad_schema_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"schema": "not_qa91"}), encoding="utf-8")
    with pytest.raises(BirthCompletionTelemetryError, match="schema"):
        load_qa91_inventory(p)


def test_load_inventory_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(BirthCompletionTelemetryError, match="not found"):
        load_qa91_inventory(tmp_path / "nope.json")


# --- telemetry reader -----------------------------------------------------------------------------


def test_read_trend_skips_non_topology_rows(tmp_path: Path) -> None:
    tel = _write_telemetry(tmp_path, _REAL_ENDPOINT_WINDOW)
    trend = read_lane_betti0_trend(tel, _INVENTORY)
    assert trend == _REAL_ENDPOINT_WINDOW


def test_read_trend_class_index_mismatch_raises(tmp_path: Path) -> None:
    tel = _write_telemetry(tmp_path, _REAL_ENDPOINT_WINDOW, gt_lane=111)
    with pytest.raises(BirthCompletionTelemetryError, match="class-index/schema mismatch"):
        read_lane_betti0_trend(tel, _INVENTORY)


def test_read_trend_dedupes_resume_overlap(tmp_path: Path) -> None:
    p = tmp_path / "t.jsonl"
    rows = [_topo_row(10, 20), _topo_row(20, 30), _topo_row(20, 31), _topo_row(30, 40)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    trend = read_lane_betti0_trend(p, _INVENTORY)
    assert trend == [(10, 20), (20, 31), (30, 40)]  # last-write-wins on epoch 20


def test_read_trend_empty_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.jsonl"
    p.write_text(json.dumps({"event": "x", "epoch": 1}) + "\n", encoding="utf-8")
    with pytest.raises(BirthCompletionTelemetryError, match="no topology"):
        read_lane_betti0_trend(p, _INVENTORY)


# --- end to end -----------------------------------------------------------------------------------


def test_evaluate_from_paths(tmp_path: Path) -> None:
    tel = _write_telemetry(tmp_path, _PLATEAU_WINDOW)
    inv = _write_qa91(tmp_path)
    v = evaluate_from_paths(tel, inv)
    assert v.fired is True
    assert v.epochs_per_gate == DEFAULT_EPOCHS_PER_GATE


def test_cli_returns_zero_and_writes(tmp_path: Path) -> None:
    from tools.run_ddm_lp2_birth_completion_key import main

    tel = _write_telemetry(tmp_path, _PLATEAU_WINDOW)
    inv = _write_qa91(tmp_path)
    out = tmp_path / "gate.json"
    rc = main(["--telemetry", str(tel), "--qa91-inventory", str(inv), "--output-json", str(out)])
    assert rc == 0
    row = json.loads(out.read_text(encoding="utf-8"))
    assert row["gate_key"] == "birth_completion"
    assert row["fired"] is True


def test_cli_fail_closed_returns_three(tmp_path: Path) -> None:
    from tools.run_ddm_lp2_birth_completion_key import main

    inv = _write_qa91(tmp_path)
    rc = main(["--telemetry", str(tmp_path / "missing.jsonl"), "--qa91-inventory", str(inv)])
    assert rc == 3
