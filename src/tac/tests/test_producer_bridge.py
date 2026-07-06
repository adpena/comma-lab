"""Tests for the #247 producer bridge — the costate controller as the ONE canonical consumer that
reads every orphaned producer (sensitivity-map / master-gradient / cathedral-autopilot) into its
SENSE, NO-FAKE (real signal OR honest available=False reason, never fabricated)."""
from __future__ import annotations

from tac.witness_control import producer_bridge as pb


def test_read_producer_signals_enumerates_all_three():
    rows = pb.read_producer_signals()
    names = {r["producer"] for r in rows}
    assert names == {
        "sensitivity_map.axis_weights",
        "master_gradient.latest_anchor",
        "cathedral_autopilot.ranker",
    }
    # every row is a well-formed ProducerSignal dict (NO-FAKE shape)
    for r in rows:
        assert set(r) == {"producer", "available", "signal", "reason", "provenance"}
        assert isinstance(r["available"], bool) and r["reason"]


def test_sensitivity_map_is_a_live_signal_with_provenance():
    rows = {r["producer"]: r for r in pb.read_producer_signals(operating_point="pr106_r2")}
    sm = rows["sensitivity_map.axis_weights"]
    assert sm["available"] is True
    assert set(sm["signal"]) == {"pose", "seg", "rate", "mixed"}
    # provenance tags propagated (CLAUDE.md evidence-tag discipline)
    assert sm["provenance"]["operating_point_tag"] and sm["provenance"]["basis"]


def test_master_gradient_no_archive_is_honestly_unavailable():
    """NO-FAKE: live training has no byte-closed archive -> available=False with the honest reason,
    NOT a fabricated anchor."""
    rows = {r["producer"]: r for r in pb.read_producer_signals(archive_sha256=None)}
    mg = rows["master_gradient.latest_anchor"]
    assert mg["available"] is False and mg["signal"] is None
    assert "no byte-closed archive" in mg["reason"]


def test_unknown_operating_point_fails_safe_not_raises():
    """A bad operating-point name must degrade to available=False, never crash the SENSE read."""
    rows = {r["producer"]: r for r in pb.read_producer_signals(operating_point="does_not_exist")}
    sm = rows["sensitivity_map.axis_weights"]
    assert sm["available"] is False and sm["signal"] is None and "unavailable" in sm["reason"]


def test_cathedral_autopilot_surfaced_with_composition_note():
    rows = {r["producer"]: r for r in pb.read_producer_signals()}
    ca = rows["cathedral_autopilot.ranker"]
    # import-verified available; the composition (duty_to_measure -> ranker) is named in the reason
    assert ca["available"] is True
    assert "duty_to_measure" in ca["reason"]


def test_duty_to_measure_as_candidates_shape_and_no_fabricated_score():
    cands = pb.duty_to_measure_as_candidates()
    assert isinstance(cands, list)
    for c in cands:
        assert set(c) == {"candidate_lever", "activation_state", "note"}
        assert "candidate_lever" in c and "predicted_dS" not in c  # NO fabricated ΔS


def test_shadow_report_surfaces_producer_signals():
    """The ONE controller's ShadowReport now carries the de-orphaned producer signals + persists them
    in the JSONL row."""
    from tac.witness_control.shadow_controller import RunInputs, build_shadow_report
    rep = build_shadow_report(RunInputs(run_dir="/tmp/prod_smoke", verdicts=[], stage_rows={}, flags={}))
    ps = rep.producer_signals
    assert {r["producer"] for r in ps} == {
        "sensitivity_map.axis_weights", "master_gradient.latest_anchor", "cathedral_autopilot.ranker"}
    assert "producer_signals" in rep.to_row()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
