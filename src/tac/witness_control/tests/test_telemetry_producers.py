from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.witness_control.telemetry_producers import (
    COMPONENT_FIELDS,
    COMPONENT_WALLCLOCK_SCHEMA,
    SPS_ENGAGEMENT_SCHEMA,
    ClipActivationAggregator,
    ComponentWallclock,
    ProducerResumeState,
    deterministic_strata,
    engagement_reasons,
    gradient_role_conflict_stats,
    ladder_birth_complete_row,
    lever_engage_row,
    live_gap_fields,
    sps_engagement_row,
    tail_cycle_endpoint_row,
    term_inert_rows,
    would_fire_row,
)


def test_component_contract_has_exact_eight_fields() -> None:
    assert COMPONENT_FIELDS == (
        "teacher_forward_s",
        "teacher_backward_s",
        "witness_forward_s",
        "witness_backward_s",
        "realized_R_s",
        "verdict_s",
        "checkpoint_io_s",
        "epoch_total_s",
    )


def test_component_schema_is_ticket_schema() -> None:
    assert COMPONENT_WALLCLOCK_SCHEMA == "witness_component_wallclock.v1"


def test_deterministic_strata_matches_registered_n600_sample() -> None:
    assert deterministic_strata(600, 4) == (75, 225, 375, 525)


def test_deterministic_strata_handles_small_population() -> None:
    assert deterministic_strata(4, 4) == (0, 1, 2, 3)


def test_deterministic_strata_rejects_empty_population() -> None:
    with pytest.raises(ValueError):
        deterministic_strata(0, 1)


def test_deterministic_strata_rejects_excess_k() -> None:
    with pytest.raises(ValueError):
        deterministic_strata(3, 4)


def test_gradient_stats_identical_roles_have_unit_cosine() -> None:
    stats = gradient_role_conflict_stats(
        {"a": np.array([1.0, 2.0])}, {"a": np.array([1.0, 2.0])})
    assert stats["global_cosine"] == pytest.approx(1.0)
    assert stats["conflict_exists_under_preregistered_rule"] is False


def test_gradient_stats_opposing_roles_trigger_registered_conflict() -> None:
    stats = gradient_role_conflict_stats(
        {"a": np.ones(10)}, {"a": -np.ones(10)})
    assert stats["global_cosine"] == pytest.approx(-1.0)
    assert stats["negative_product_weight_fraction_all"] == 1.0
    assert stats["conflict_exists_under_preregistered_rule"] is True


def test_gradient_stats_zero_role_has_null_cosine() -> None:
    stats = gradient_role_conflict_stats(
        {"a": np.zeros(2)}, {"a": np.ones(2)})
    assert stats["global_cosine"] is None
    assert stats["coactive_weight_fraction"] == 0.0


def test_gradient_stats_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="keys differ"):
        gradient_role_conflict_stats({"a": [1]}, {"b": [1]})


def test_gradient_stats_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape differs"):
        gradient_role_conflict_stats({"a": [1, 2]}, {"a": [1]})


def test_gradient_stats_matches_standalone_probe_math() -> None:
    torch = pytest.importorskip("torch")
    from tools.probe_sps_gradient_role_conflict import _gradient_stats

    pred_np = {"a": np.array([1.0, -2.0]), "b": np.array([0.5])}
    temp_np = {"a": np.array([-1.0, -1.0]), "b": np.array([0.25])}
    expected = _gradient_stats(
        {k: torch.tensor(v, dtype=torch.float64) for k, v in pred_np.items()},
        {k: torch.tensor(v, dtype=torch.float64) for k, v in temp_np.items()},
    )
    actual = gradient_role_conflict_stats(pred_np, temp_np)
    for key in (
        "global_cosine",
        "prediction_norm",
        "temporal_norm",
        "coactive_weight_fraction",
        "negative_product_weight_fraction_all",
        "negative_product_weight_fraction_coactive",
        "negative_cosine_tensor_weight_fraction",
        "conflict_exists_under_preregistered_rule",
    ):
        assert actual[key] == pytest.approx(expected[key])


def test_sps_row_carries_engagement_and_nonpromotion_scope() -> None:
    row = sps_engagement_row(
        {"a": np.ones(2)}, {"a": -np.ones(2)}, epoch=450,
        seg={"a": np.ones(2)}, pose={"a": np.zeros(2)},
        engagement="temporal_screw_engaged", reason="nominal_boundary",
        actual_event=True, pair_indices=(75, 225),
        active_temporal_terms={"temporal_screw": 0.1})
    assert row["schema"] == SPS_ENGAGEMENT_SCHEMA
    assert row["actual_event"] is True
    assert row["pair_indices"] == [75, 225]
    assert row["primary_conflict"] == "seg_vs_temporal"
    assert set(row["gradient_conflict"]) == {
        "seg_vs_temporal", "pose_vs_temporal",
        "fully_armed_seg_plus_pose_vs_temporal"}
    assert row["promotable"] is False


def test_sps_row_rejects_unknown_engagement() -> None:
    with pytest.raises(ValueError, match="unknown SPS engagement"):
        sps_engagement_row(
            {"a": [1]}, {"a": [1]}, epoch=1, engagement="other",
            reason="x", actual_event=False, pair_indices=(0,),
            active_temporal_terms={})


def test_engagement_reasons_names_nominal_boundary() -> None:
    assert engagement_reasons(450, nominal_epoch=450, window=2) == (
        "nominal_boundary",)


def test_engagement_reasons_names_signed_offsets() -> None:
    assert engagement_reasons(448, nominal_epoch=450, window=2) == (
        "nominal_boundary_offset_-2",)


def test_engagement_reasons_adds_actual_transition_once() -> None:
    assert engagement_reasons(
        450, nominal_epoch=450, window=2, actual_event=True) == (
            "actual_engagement_transition",)


def _complete_clock(path: Path) -> ComponentWallclock:
    clock = ComponentWallclock(path)
    clock.add("teacher_forward_s", 1.0)
    for name in COMPONENT_FIELDS[1:-1]:
        clock.mark_not_invoked(name)
    return clock


def test_component_row_is_complete_with_explicit_structural_zeros(tmp_path: Path) -> None:
    row = _complete_clock(tmp_path / "rows.jsonl").row(epoch=3, epoch_total_s=2.0)
    assert row["complete"] is True
    assert row["teacher_forward_s"] == 1.0
    assert row["component_calls"]["verdict_s"] == 0
    assert "verdict_s" in row["not_invoked"]


def test_component_row_does_not_forge_missing_measurements(tmp_path: Path) -> None:
    row = ComponentWallclock(tmp_path / "rows.jsonl").row(epoch=1, epoch_total_s=1.0)
    assert row["complete"] is False
    assert row["teacher_forward_s"] is None


def test_real_measurement_replaces_structural_zero_label(tmp_path: Path) -> None:
    clock = ComponentWallclock(tmp_path / "rows.jsonl")
    clock.mark_not_invoked("checkpoint_io_s")
    clock.add("checkpoint_io_s", 0.25)
    row = clock.row(epoch=1, epoch_total_s=1.0)
    assert row["checkpoint_io_s"] == 0.25
    assert "checkpoint_io_s" not in row["not_invoked"]


def test_component_timer_rejects_negative_duration(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ComponentWallclock(tmp_path / "rows.jsonl").add("verdict_s", -1.0)


def test_component_timer_rejects_unknown_component(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        ComponentWallclock(tmp_path / "rows.jsonl").add("other", 1.0)


def test_component_measure_uses_injected_monotonic_clock(tmp_path: Path) -> None:
    ticks = iter((1_000_000_000, 1_250_000_000))
    clock = ComponentWallclock(tmp_path / "rows.jsonl", clock_ns=lambda: next(ticks))
    with clock.measure("checkpoint_io_s"):
        pass
    assert clock.row(epoch=1, epoch_total_s=1.0)["checkpoint_io_s"] == 0.25


def test_component_probe_failure_is_loud_and_null(tmp_path: Path) -> None:
    clock = ComponentWallclock(tmp_path / "rows.jsonl", clock_ns=lambda: 1)

    def fail() -> None:
        raise RuntimeError("probe failed")

    assert clock.measure_probe("realized_R_s", fail) is None
    row = clock.row(epoch=1, epoch_total_s=1.0)
    assert row["realized_R_s"] is None
    assert row["errors"][0]["component"] == "realized_R_s"


def test_component_emit_uses_jsonl_store(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    expected = _complete_clock(path).emit(epoch=2, epoch_total_s=2.0)
    observed = json.loads(path.read_text().strip())
    assert observed == expected


def test_resume_state_legacy_payload_is_ignored() -> None:
    state = ProducerResumeState()
    assert state.restore_from_cfg("__dtp_", {"__legacy": 1}) is False
    assert state.sps_emitted == set()


def test_resume_state_round_trips_all_observer_latches() -> None:
    state = ProducerResumeState()
    state.mark_sps(engagement="temporal_screw_engaged", epoch=450, reason="x")
    state.mark_ladder("lane")
    state.inert_streaks["chroma_boundary"] = 2
    restored = ProducerResumeState()
    assert restored.restore_from_cfg("__dtp_", state.state_arrays("__dtp_")) is True
    assert restored.sps_emitted == state.sps_emitted
    assert restored.ladder_emitted == {"lane"}
    assert restored.inert_streaks == {"chroma_boundary": 2}


def test_resume_state_sps_latch_is_nonduplicating() -> None:
    state = ProducerResumeState()
    kwargs = {"engagement": "phase_advection_engaged", "epoch": 726, "reason": "x"}
    assert state.mark_sps(**kwargs) is True
    assert state.mark_sps(**kwargs) is False


def test_resume_state_ladder_latch_is_nonduplicating() -> None:
    state = ProducerResumeState()
    assert state.mark_ladder("lane") is True
    assert state.mark_ladder("lane") is False


def test_clip_activation_reports_global_and_per_group_rates() -> None:
    agg = ClipActivationAggregator(1.0)
    agg.observe(2.0, {"film": 0.5, "code": 2.0})
    agg.observe(0.5, {"film": 1.5, "code": 0.5})
    row = agg.row(epoch=9)
    assert row["ep"] == 9
    assert row["global"]["frac_clipped"] == 0.5
    assert row["per_group"]["film"]["norm_mean"] == 1.0


def test_clip_activation_empty_row_is_explicit() -> None:
    row = ClipActivationAggregator(1.0).row(epoch=1)
    assert row["global"] == {
        "n": 0, "frac_clipped": 0.0, "norm_mean": None, "norm_max": None}


def test_term_inert_fires_once_at_sustained_threshold() -> None:
    state = ProducerResumeState()
    assert term_inert_rows(
        {"seg": 1.0, "chroma_boundary": 0.0},
        engaged={"chroma_boundary": True}, epoch=1, state=state,
        sustained_rows=2) == []
    rows = term_inert_rows(
        {"seg": 1.0, "chroma_boundary": 0.0},
        engaged={"chroma_boundary": True}, epoch=2, state=state,
        sustained_rows=2)
    assert rows[0]["alarm"] == "term_inert"


def test_term_inert_streak_resets_when_term_binds() -> None:
    state = ProducerResumeState(inert_streaks={"x": 2})
    assert term_inert_rows(
        {"x": 1.0}, engaged={"x": True}, epoch=3, state=state) == []
    assert state.inert_streaks["x"] == 0


def test_live_gap_fields_use_ema_minus_live_sign() -> None:
    row = live_gap_fields(
        {"d_seg": 0.2, "d_pose": 1.0}, {"d_seg": 0.1, "d_pose": 1.5})
    assert row == {
        "d_seg_live": 0.1,
        "d_pose_live": 1.5,
        "d_seg_ema_minus_live": 0.1,
        "d_pose_ema_minus_live": -0.5,
    }


def test_tail_endpoint_requires_a_real_verdict() -> None:
    assert tail_cycle_endpoint_row(
        None, epoch=10, cycle=1, start_epoch=1, boundary_reason="x") is None


def test_tail_endpoint_carries_cycle_and_score() -> None:
    row = tail_cycle_endpoint_row(
        {"epoch": 9, "d_seg": 0.2, "d_pose": 1.0, "implied_S": 0.3},
        epoch=10, cycle=1, start_epoch=5, boundary_reason="restart")
    assert row["cycle_start_epoch"] == 5
    assert row["verdict_epoch"] == 9
    assert row["implied_S"] == 0.3


def test_would_fire_row_has_uniform_status() -> None:
    row = would_fire_row(
        epoch=3, lever="muon", metric="remaining_meat", value=0.0,
        threshold=1e-4, dwell=8, sensor_data_epoch=2,
        event_mode=True, fired=True)
    assert row["status"] == "fired"
    assert row["sensor_data_epoch"] == 2


def test_ladder_birth_complete_row_is_discrete() -> None:
    row = ladder_birth_complete_row(
        epoch=100, class_id=1, class_name="lane", final_radius=0.0)
    assert row["stage"] == "ladder_birth_complete"
    assert row["r_final"] == 0.0


@pytest.mark.parametrize("status", ["armed", "fired", "complete"])
def test_lever_engage_accepts_uniform_statuses(status: str) -> None:
    assert lever_engage_row("x", status=status, epoch=1, via="test")["status"] == status


def test_lever_engage_rejects_nonuniform_status() -> None:
    with pytest.raises(ValueError):
        lever_engage_row("x", status="held", epoch=1, via="test")


def test_dsl_telemetry_defaults_on_and_compiles_exact_flags() -> None:
    from tac.witness_dsl.curriculum_dsl import TelemetryCadence

    flags = TelemetryCadence().flags()
    assert flags["--component-wallclock-telemetry"] is True
    assert flags["--sps-engagement-telemetry"] is True
    assert flags["--sps-engagement-k-pairs"] == 4


def test_verdict_live_gap_is_named_default_off_dsl_lever() -> None:
    from tac.witness_dsl.curriculum_dsl import VerdictLiveGap

    lever = VerdictLiveGap(every=3)
    assert lever.name == "verdict_live_gap"
    assert lever.overrides == {"--verdict-live-gap-every": 3}


def test_verdict_live_gap_rejects_nonpositive_cadence() -> None:
    from tac.witness_dsl.curriculum_dsl import VerdictLiveGap

    with pytest.raises(ValueError):
        VerdictLiveGap(every=0)


def test_trainer_contains_exact_ticket_literals_and_all_three_regions() -> None:
    root = Path(__file__).resolve().parents[4]
    text = (root / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    for literal in (COMPONENT_WALLCLOCK_SCHEMA, SPS_ENGAGEMENT_SCHEMA, *COMPONENT_FIELDS):
        assert repr(literal) in text or f'"{literal}"' in text
    for region in ("threading.Thread", "muon_finisher_switch", "_record_causal_boundary"):
        assert region in text


def test_trainer_parser_exposes_default_on_observers_and_off_live_gap() -> None:
    root = Path(__file__).resolve().parents[4]
    text = (root / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert '"--component-wallclock-telemetry"' in text
    assert '"--sps-engagement-telemetry"' in text
    assert '"--verdict-live-gap-every", type=int, default=0' in text


def test_da_db_observer_is_registered_in_canonical_resume_registry() -> None:
    root = Path(__file__).resolve().parents[4]
    text = (root / "experiments/train_levelset_witness_realized_through_R_mlx.py").read_text()
    assert '_resume_registry.register(\n        "da_db_telemetry", "__dtp_"' in text
