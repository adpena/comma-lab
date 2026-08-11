# SPDX-License-Identifier: MIT
"""Focused tests for the pure DDM endgame policy."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
MODULE_PATH = REPO / "src" / "tac" / "witness_control" / "ddm_endgame_policy.py"
SPEC = importlib.util.spec_from_file_location("_test_ddm_endgame_policy_leaf", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = POLICY
SPEC.loader.exec_module(POLICY)

TOOL_PATH = REPO / "tools" / "derive_ddm_endgame_policy.py"
TOOL_SPEC = importlib.util.spec_from_file_location("_test_derive_ddm_endgame_policy", TOOL_PATH)
assert TOOL_SPEC is not None and TOOL_SPEC.loader is not None
TOOL = importlib.util.module_from_spec(TOOL_SPEC)
sys.modules[TOOL_SPEC.name] = TOOL
TOOL_SPEC.loader.exec_module(TOOL)

ActionKind = POLICY.ActionKind
ActionQuote = POLICY.ActionQuote
AdvisorySignals = POLICY.AdvisorySignals
Decision = POLICY.Decision
DecisionAction = POLICY.DecisionAction
CompleteScoreDominanceReceipt = POLICY.CompleteScoreDominanceReceipt
EndgamePolicyError = POLICY.EndgamePolicyError
OperatingPoint = POLICY.OperatingPoint
StopVerdict = POLICY.StopVerdict
TrajectoryRegime = POLICY.TrajectoryRegime
build_endgame_arithmetic_receipt = POLICY.build_endgame_arithmetic_receipt
contest_score = POLICY.contest_score
decide_endgame_policy = POLICY.decide_endgame_policy
strict_integer_byte_ceiling = POLICY.strict_integer_byte_ceiling
derive_request = TOOL.derive_request
derive_stop_verdict = POLICY.derive_stop_verdict

CHECKPOINT_SHA = "1" * 64
ARCHIVE_SHA = "2" * 64
RECEIVER_SHA = "3" * 64
FOREIGN_SHA = "4" * 64


def _point(
    *,
    topology_stable: bool = True,
    transitions_pending: bool = False,
    d_seg: float = 0.002,
    d_pose: float = 0.001,
    archive_bytes: int = 100_000,
) -> OperatingPoint:
    return OperatingPoint(
        stage_id="seg_trunk_tau",
        checkpoint_sha256=CHECKPOINT_SHA,
        archive_sha256=ARCHIVE_SHA,
        receiver_sha256=RECEIVER_SHA,
        d_seg=d_seg,
        per_class_d_seg=(0.001, 0.002, 0.003, 0.004, 0.005),
        d_pose=d_pose,
        archive_bytes=archive_bytes,
        n_pairs=600,
        hard_parsed=True,
        receiver_realized=True,
        topology_stable=topology_stable,
        transitions_pending=transitions_pending,
        topology_signature="classes=5;islands=17;edges_sha256=" + "a" * 64,
        evidence_axis="[macOS-CPU frozen-scorer advisory]",
        verdict_scope="INSTANCE: unit-test operating point",
    )


def _train(
    point: OperatingPoint,
    *,
    gain_lower: float = 0.01,
    gain_upper: float = 0.02,
    with_endpoint: bool = True,
) -> ActionQuote:
    exact_gain = (gain_lower + gain_upper) / 2.0
    endpoint_d_seg = point.d_seg - exact_gain / 100.0
    assert endpoint_d_seg >= 0.0
    return ActionQuote(
        quote_id="train_window",
        kind=ActionKind.TRAIN_WINDOW,
        parent_checkpoint_sha256=point.checkpoint_sha256,
        parent_archive_sha256=point.archive_sha256,
        parent_receiver_sha256=point.receiver_sha256,
        n_pairs=point.n_pairs,
        gain_lower=gain_lower,
        gain_upper=gain_upper,
        wall_seconds_lower=100.0,
        wall_seconds_upper=100.0,
        measured=True,
        hard_parsed=True,
        receiver_realized=True,
        admissible=True,
        verdict_scope="INSTANCE: measured bounded train window",
        evidence_axis=point.evidence_axis,
        endpoint_d_seg=endpoint_d_seg if with_endpoint else None,
        endpoint_d_pose=point.d_pose if with_endpoint else None,
        endpoint_archive_bytes=point.archive_bytes if with_endpoint else None,
    )


def _finisher(
    point: OperatingPoint,
    *,
    kind: ActionKind = ActionKind.SEG_GN,
    exact_gain: float = 0.05,
    gain_lower: float | None = None,
    gain_upper: float | None = None,
) -> ActionQuote:
    endpoint_d_seg = point.d_seg - exact_gain / 100.0
    assert endpoint_d_seg >= 0.0
    realized_gain = point.score - contest_score(endpoint_d_seg, point.d_pose, point.archive_bytes)
    return ActionQuote(
        quote_id=f"{kind.value.lower()}_quote",
        kind=kind,
        parent_checkpoint_sha256=point.checkpoint_sha256,
        parent_archive_sha256=point.archive_sha256,
        parent_receiver_sha256=point.receiver_sha256,
        n_pairs=point.n_pairs,
        gain_lower=realized_gain - 1e-12 if gain_lower is None else gain_lower,
        gain_upper=realized_gain + 1e-12 if gain_upper is None else gain_upper,
        wall_seconds_lower=100.0,
        wall_seconds_upper=100.0,
        measured=True,
        hard_parsed=True,
        receiver_realized=True,
        admissible=True,
        verdict_scope="INSTANCE: measured hard parsed finisher",
        evidence_axis=point.evidence_axis,
        endpoint_d_seg=endpoint_d_seg,
        endpoint_d_pose=point.d_pose,
        endpoint_archive_bytes=point.archive_bytes,
        candidate_evaluations=1,
    )


def test_foreign_parent_quote_fails_closed() -> None:
    point = _point()
    foreign = replace(_train(point), parent_archive_sha256=FOREIGN_SHA)
    with pytest.raises(EndgamePolicyError, match="parent identity"):
        decide_endgame_policy(point, (foreign,))


def test_foreign_axis_quote_fails_closed() -> None:
    point = _point()
    foreign_axis = replace(_train(point), evidence_axis="[contest-CUDA foreign-axis]")
    with pytest.raises(EndgamePolicyError, match="evidence_axis"):
        decide_endgame_policy(point, (foreign_axis,))


def test_subset_quote_fails_closed() -> None:
    point = _point()
    subset = replace(_train(point), n_pairs=96)
    with pytest.raises(EndgamePolicyError, match="n_pairs"):
        decide_endgame_policy(point, (subset,))


def test_absent_quotes_are_honestly_measure_not_continue() -> None:
    decision = decide_endgame_policy(_point(), ())
    assert decision.action is DecisionAction.MEASURE_FINISHER_QUOTE
    assert decision.reason_codes == ("MISSING_SAME_PARENT_MEASURED_TRAINING_QUOTE",)


def test_positive_measured_training_quote_without_competitor_continues_bounded() -> None:
    point = _point()
    decision = decide_endgame_policy(point, (_train(point),))
    assert decision.action is DecisionAction.CONTINUE_BOUNDED_WINDOW
    assert decision.selected_quote_id is None


def test_training_quote_without_exact_endpoint_requires_measurement() -> None:
    point = _point()
    decision = decide_endgame_policy(point, (_train(point, with_endpoint=False),))
    assert decision.action is DecisionAction.MEASURE_FINISHER_QUOTE
    assert decision.reason_codes == ("TRAINING_QUOTE_MISSING_POSITIVE_MEASURED_HARD_REALIZED_EXACT_ENDPOINT",)


def test_operating_point_preserves_holistic_facets_and_rejects_malformed_values() -> None:
    point = _point()
    payload = point.to_payload()
    assert payload["per_class_d_seg"] == [0.001, 0.002, 0.003, 0.004, 0.005]
    assert payload["topology_signature"] == point.topology_signature
    assert OperatingPoint.from_payload(payload) == point
    with pytest.raises(EndgamePolicyError, match="canonical five-value"):
        replace(point, per_class_d_seg=(0.0, 0.0, 0.0, 0.0))
    with pytest.raises(EndgamePolicyError, match=r"per_class_d_seg\[4\]"):
        replace(point, per_class_d_seg=(0.0, 0.0, 0.0, 0.0, -1.0))
    with pytest.raises(EndgamePolicyError, match="topology_signature"):
        replace(point, topology_signature="")


def test_overlapping_gain_rate_intervals_require_measurement() -> None:
    point = _point()
    train = _train(point, gain_lower=0.01, gain_upper=0.02)
    finisher = _finisher(point, exact_gain=0.015, gain_lower=0.01, gain_upper=0.02)
    decision = decide_endgame_policy(point, (train, finisher))
    assert decision.action is DecisionAction.MEASURE_FINISHER_QUOTE
    assert decision.reason_codes == ("TRAIN_AND_FINISHER_GAIN_RATE_INTERVALS_OVERLAP",)


@pytest.mark.parametrize(
    ("topology_stable", "transitions_pending"),
    [(False, False), (True, True), (False, True)],
)
def test_gn_and_pose_handoff_require_stable_topology_and_no_transitions(
    topology_stable: bool,
    transitions_pending: bool,
) -> None:
    for kind in (ActionKind.SEG_GN, ActionKind.TERMINAL_POSE):
        point = _point(
            topology_stable=topology_stable,
            transitions_pending=transitions_pending,
        )
        decision = decide_endgame_policy(point, (_train(point), _finisher(point, kind=kind)))
        assert decision.action is DecisionAction.CONTINUE_BOUNDED_WINDOW
        assert "TOPOLOGY_STABLE" in decision.reason_codes[0]


def test_strict_gain_rate_dominance_hands_off_qdbs() -> None:
    point = _point(topology_stable=False, transitions_pending=True)
    decision = decide_endgame_policy(
        point,
        (_train(point), _finisher(point, kind=ActionKind.QDBS)),
    )
    assert decision.action is DecisionAction.HANDOFF_QDBS
    assert decision.selected_quote_id == "qdbs_quote"


def test_ncde_saddle_grokking_and_v17_metadata_cannot_actuate() -> None:
    point = _point()
    train = _train(point)
    baseline = decide_endgame_policy(point, (train,))
    loud = decide_endgame_policy(
        point,
        (train,),
        advisory_signals=AdvisorySignals(
            ncde_fire=True,
            ncde_fit_r2=0.999,
            trajectory_regime=TrajectoryRegime.FIXED_QUADRATIC_TERMINAL,
            grokking_classification="NO_GO_AS_STAGE_ADVANCE_AUTHORITY",
            v17_rho=1000.0,
            v17_radius_update="GROW_ACCEPTED_HIGH_RHO",
        ),
    )
    assert baseline.action is loud.action is DecisionAction.CONTINUE_BOUNDED_WINDOW
    assert loud.advisory_signals.to_payload()["actuation"] == "NONE"


def test_exact_nonlinear_pose_gain_controls_terminal_pose_quote() -> None:
    point = _point(d_seg=0.001, d_pose=0.001610, archive_bytes=120_000)
    endpoint_d_pose = 2.33e-5
    exact_gain = math.sqrt(10.0 * point.d_pose) - math.sqrt(10.0 * endpoint_d_pose)
    endpoint_score = contest_score(point.d_seg, endpoint_d_pose, point.archive_bytes)
    quote = ActionQuote(
        quote_id="terminal_pose_exact",
        kind=ActionKind.TERMINAL_POSE,
        parent_checkpoint_sha256=point.checkpoint_sha256,
        parent_archive_sha256=point.archive_sha256,
        parent_receiver_sha256=point.receiver_sha256,
        n_pairs=point.n_pairs,
        gain_lower=exact_gain - 1e-12,
        gain_upper=exact_gain + 1e-12,
        wall_seconds_lower=100.0,
        wall_seconds_upper=100.0,
        measured=True,
        hard_parsed=True,
        receiver_realized=True,
        admissible=True,
        verdict_scope="INSTANCE: nonlinear pose endpoint",
        evidence_axis=point.evidence_axis,
        endpoint_d_seg=point.d_seg,
        endpoint_d_pose=endpoint_d_pose,
        endpoint_archive_bytes=point.archive_bytes,
    )
    assert point.score - endpoint_score == pytest.approx(exact_gain, abs=1e-15)
    decision = decide_endgame_policy(point, (_train(point), quote))
    assert decision.action is DecisionAction.HANDOFF_TERMINAL_POSE


def test_qdbs_48_candidate_cost_quote_is_derived_and_gain_is_missing() -> None:
    quote = build_endgame_arithmetic_receipt()["qdbs_cost_quote"]
    assert quote["candidate_evaluations_max"] == 48
    assert quote["total_full_verdicts"] == 49
    assert quote["total_seconds_lower"] == 49 * 423
    assert quote["total_seconds_upper"] == 49 * 514
    assert quote["gain"] is None


def test_golden_configuration_corners_and_strict_byte_ceilings() -> None:
    receipt = build_endgame_arithmetic_receipt()
    corners = receipt["configuration_corners"]
    assert corners["tr1_optimistic_decimal_149k"]["score"] == pytest.approx(0.1441773215376773, abs=1e-15)
    assert corners["tr1_spec_mid"]["score"] == pytest.approx(0.17577269233441933, abs=1e-15)
    assert corners["tr1_banked_pose_fallback"]["score"] == pytest.approx(0.31072342498205163, abs=1e-15)
    assert receipt["current_t2"]["score_lower_bound"] == pytest.approx(1.7392661987622535, abs=1e-15)
    ceilings = receipt["strict_integer_byte_ceilings"]
    assert ceilings["dseg_3e4_dpose_2p33e5"] == {
        "official_displayed_0p172": 190334,
        "sub015": 157294,
    }
    assert ceilings["dseg_3e4_banked_r1_pose"]["sub015"] is None
    assert strict_integer_byte_ceiling(0.15, 3e-4, 0.001610) is None


def test_generated_receipt_matches_tracked_golden() -> None:
    tracked = json.loads(
        (REPO / ".omx" / "research" / "ddm_eg1_policy_arithmetic_20260728.json").read_text(encoding="utf-8")
    )
    assert tracked == build_endgame_arithmetic_receipt()


def test_decision_and_inputs_roundtrip_deterministically() -> None:
    point = _point()
    train = _train(point)
    finisher = _finisher(point, kind=ActionKind.QDBS)
    decision = decide_endgame_policy(point, (train, finisher))
    point_roundtrip = OperatingPoint.from_payload(point.to_payload())
    quote_roundtrip = ActionQuote.from_payload(finisher.to_payload())
    decision_roundtrip = Decision.from_payload(decision.to_payload())
    decision_payload = decision.to_payload()
    assert point_roundtrip == point
    assert quote_roundtrip == finisher
    assert decision_roundtrip == decision
    assert decision_payload["research_only"] is True
    assert decision_payload["score_claim"] is False
    assert decision_payload["pointer_moved"] is False
    assert json.dumps(decision_roundtrip.to_payload(), sort_keys=True) == json.dumps(
        decision.to_payload(), sort_keys=True
    )
    for key, unsafe_value in (
        ("research_only", False),
        ("score_claim", True),
        ("pointer_moved", True),
    ):
        unsafe_payload = {**decision_payload, key: unsafe_value}
        with pytest.raises(EndgamePolicyError, match=key):
            Decision.from_payload(unsafe_payload)


def test_resume_payloads_reject_invented_controls() -> None:
    point = _point()
    quote = _train(point)
    with pytest.raises(EndgamePolicyError, match="unexpected keys"):
        OperatingPoint.from_payload({**point.to_payload(), "force_handoff": True})
    with pytest.raises(EndgamePolicyError, match="unexpected keys"):
        ActionQuote.from_payload({**quote.to_payload(), "skip_receiver_check": True})
    with pytest.raises(ValueError, match="unexpected keys"):
        derive_request({"operating_point": point.to_payload(), "force_handoff": True})


# --- vehicle-neutral EG1 stop-policy interface port --------------------------


def _dominance(
    receipt_id: str,
    action: StopVerdict,
    candidate,
    *,
    parent=None,
    parent_sha: str = CHECKPOINT_SHA,
):
    return CompleteScoreDominanceReceipt.from_components(
        receipt_id=receipt_id,
        action=action,
        parent_sha256=parent_sha,
        parent_components=parent or {"distortion": 2.0, "rate": 1.0},
        candidate_components=candidate,
    )


def test_parameterized_complete_score_selects_continue():
    receipt = _dominance(
        "train", StopVerdict.CONTINUE, {"distortion": 1.5, "rate": 1.0}
    )
    assert receipt.parent_complete_score == 3.0
    assert receipt.candidate_complete_score == 2.5
    assert derive_stop_verdict((receipt,)) is StopVerdict.CONTINUE


def test_parameterized_complete_score_selects_handoff_when_best():
    receipts = (
        _dominance("train", StopVerdict.CONTINUE, {"distortion": 1.8, "rate": 1.0}),
        _dominance("finish", StopVerdict.HANDOFF, {"distortion": 1.1, "rate": 1.2}),
    )
    assert derive_stop_verdict(receipts) is StopVerdict.HANDOFF


def test_no_strict_complete_score_improvement_stops():
    receipts = (
        _dominance("equal", StopVerdict.CONTINUE, {"distortion": 2.0, "rate": 1.0}),
        _dominance("worse", StopVerdict.HANDOFF, {"distortion": 2.1, "rate": 1.0}),
    )
    assert derive_stop_verdict(receipts) is StopVerdict.STOP


def test_component_names_are_parameterized_not_contest_specific():
    receipt = _dominance(
        "generic",
        StopVerdict.HANDOFF,
        {"energy": 2.0, "latency": 3.0, "risk": 0.5},
        parent={"energy": 3.0, "latency": 3.0, "risk": 0.5},
    )
    assert derive_stop_verdict((receipt,)) is StopVerdict.HANDOFF


def test_dominance_receipts_refuse_foreign_parent():
    with pytest.raises(EndgamePolicyError, match="one parent_sha256"):
        derive_stop_verdict((
            _dominance("a", StopVerdict.CONTINUE, {"distortion": 1.0, "rate": 1.0}),
            _dominance(
                "b", StopVerdict.HANDOFF, {"distortion": 1.0, "rate": 1.0},
                parent_sha=FOREIGN_SHA,
            ),
        ))


def test_dominance_receipts_refuse_incomplete_component_join():
    with pytest.raises(EndgamePolicyError, match="same complete component names"):
        _dominance("bad", StopVerdict.CONTINUE, {"distortion": 1.0})


def test_dominance_receipts_refuse_stop_as_input_action():
    with pytest.raises(EndgamePolicyError, match="continue or handoff"):
        _dominance("bad", StopVerdict.STOP, {"distortion": 1.0, "rate": 1.0})


def test_dominance_receipts_refuse_ambiguous_equal_best_actions():
    candidate = {"distortion": 1.0, "rate": 1.0}
    with pytest.raises(EndgamePolicyError, match="equal best"):
        derive_stop_verdict((
            _dominance("a", StopVerdict.CONTINUE, candidate),
            _dominance("b", StopVerdict.HANDOFF, candidate),
        ))


def test_dominance_receipts_refuse_absent_evidence_and_duplicate_ids():
    with pytest.raises(EndgamePolicyError, match="at least one"):
        derive_stop_verdict(())
    receipt = _dominance(
        "same", StopVerdict.CONTINUE, {"distortion": 1.0, "rate": 1.0}
    )
    with pytest.raises(EndgamePolicyError, match="ids must be unique"):
        derive_stop_verdict((receipt, receipt))
