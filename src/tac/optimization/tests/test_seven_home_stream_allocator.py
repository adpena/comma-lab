# SPDX-License-Identifier: MIT
"""Behavior tests for the lawful EV2 seven-home allocator."""

from __future__ import annotations

import json
import math
from pathlib import Path

from tac.analysis.action_effect import ActionEffect
from tac.analysis.applied_action_receipt import (
    ApplicationStatus,
    StreamHomeClaim,
    build_applied_action_receipt,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.optimization.seven_home_stream_allocator import (
    SEVEN_HOME_IDS,
    ReceiptEnvelope,
    build_allocation_plan,
    derive_seven_homes,
)

REPO = Path(__file__).resolve().parents[4]
EV2_PATH = REPO / ".omx/research/ddm_ev2_per_pair_allocation_20260725T041933Z/allocation_table.json"
BASE_BYTES = 134_211
HOME_BYTES = {
    "manifest": 3_345,
    "v15_predictor_zip_outer_home": 100_099,
    "g1_movable_worldsheet_outer_home": 29_878,
    "receiver_realization_profile": 85,
    "solved_template_outer_home": 151,
    "central_directory_and_eocd": 383,
    "lane_program_seed": 270,
}


def _ev2() -> dict:
    return json.loads(EV2_PATH.read_text())


def _pointer(score: float = 0.172) -> dict:
    return {
        "effective_frontier": {
            "score": score,
            "axis": "official_leaderboard",
            "custody": "external target only",
            "evidence_grade": "[official-leaderboard display]",
            "source": "fixture-pointer",
            "source_kind": "external_public_leaderboard_target",
        }
    }


def _receipt(
    *,
    receipt_id: str,
    home_id: str,
    coder_id: str,
    delta_bytes: int,
    old_d_seg: float = 0.01,
    new_d_seg: float = 0.009,
    old_d_pose: float = 0.04,
    new_d_pose: float = 0.04,
    base_sha: str = "a" * 64,
    from_state: str = "c1:base",
    physical_edge: str | None = None,
    composed_action_ids: tuple[str, ...] = (),
    interaction: float | None = None,
    aggregate_home: bool = False,
    home_before_override: int | None = None,
    support_sha256: str | None = "d" * 64,
):
    action_id = f"action:{receipt_id}"
    effect = ActionEffect.build(
        action_id=action_id,
        family="ddm",
        action_kind="counted_receiver_edge",
        authority="inflate_torch_cpu",
        normalization_scope="full_video_exact",
        producer="seven-home-test",
        pair_ids=(0, 1),
        composed_action_ids=composed_action_ids,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=BASE_BYTES,
        new_bytes=BASE_BYTES + delta_bytes,
        base_archive_sha256=base_sha,
        archive_sha256=(receipt_id.encode().hex() + "0" * 64)[:64],
        base_state_sha256="c" * 64,
        support_sha256=support_sha256,
        interaction_or_commutator=interaction,
    )
    home_before = (
        home_before_override
        if home_before_override is not None
        else BASE_BYTES if aggregate_home else HOME_BYTES[home_id]
    )
    stream_home = StreamHomeClaim(
        stream_type=StreamType.RESIDUAL,
        layer_home=LayerHome.L1_PROGRAM,
        byte_home_id="__composed_packet__" if aggregate_home else home_id,
        coder_id=coder_id,
        coder_owner=f"owner:{coder_id}",
        receiver_consumer="fixture-receiver",
        bytes_before=home_before,
        bytes_after=home_before + delta_bytes,
    )
    status = (
        ApplicationStatus.DOWNHILL_FINITE
        if effect.delta_score_total is not None and effect.delta_score_total < 0.0
        else ApplicationStatus.UPHILL_NULL
    )
    return build_applied_action_receipt(
        receipt_id=receipt_id,
        status=status,
        action_effect=effect,
        codeword_id=f"codeword:{receipt_id}",
        application_operator_id="fixture.apply",
        application_operator_version="v1",
        physical_edge_id=physical_edge or f"edge:{receipt_id}",
        edge_from_state_id=from_state,
        edge_to_state_id=f"{from_state}+{receipt_id}",
        integer_quantum=1,
        direction=1,
        validity_radius=1.0,
        receiver_schema="fixture.receiver.v1",
        receiver_sha256="e" * 64,
        r_chain_id="uint8-r-seg-pose-v1",
        changed_uint8_count=2,
        changed_uint8_sha256="f" * 64,
        stream_home=stream_home,
        verdict_scope="INSTANCE:fixture",
        provenance_ref=f"fixture://{receipt_id}",
        bucket_id="fixture-bucket",
    )


def test_rederives_actual_seven_homes_without_promoting_162_cells() -> None:
    homes = derive_seven_homes(_ev2())

    assert tuple(row["home_id"] for row in homes) == SEVEN_HOME_IDS
    assert len(homes) == 7
    assert sum(row["counted_bytes"] for row in homes) == BASE_BYTES


def test_coder_alternatives_for_same_home_are_mutually_exclusive() -> None:
    weaker = _receipt(
        receipt_id="manifest-brotli",
        home_id="manifest",
        coder_id="BROTLI_Q11",
        delta_bytes=-100,
    )
    stronger = _receipt(
        receipt_id="manifest-kt",
        home_id="manifest",
        coder_id="BELLARD_KT",
        delta_bytes=-200,
    )

    plan = build_allocation_plan(
        ev2=_ev2(),
        pointer=_pointer(),
        envelopes=(ReceiptEnvelope(weaker), ReceiptEnvelope(stronger)),
    )

    assert len(plan["selected_home_owners"]) == 7
    selected = [row for row in plan["selected_home_owners"] if row["selection"] != "BASELINE_OWNER"]
    assert [row["receipt_id"] for row in selected] == ["manifest-kt"]
    assert any(row["receipt_id"] == "manifest-brotli" for row in plan["rejected"])


def test_pf3_and_wf7_candidates_compete_and_deltas_never_add() -> None:
    pf3 = ReceiptEnvelope(
        _receipt(
            receipt_id="pf3-coordinate",
            home_id="g1_movable_worldsheet_outer_home",
            coder_id="E4_BROTLI_Q11",
            delta_bytes=-100,
            new_d_seg=0.0088,
        ),
        opportunity_pool_id="PF3_COORDINATE",
    )
    wf7 = ReceiptEnvelope(
        _receipt(
            receipt_id="wf7-stream",
            home_id="v15_predictor_zip_outer_home",
            coder_id="CC3_MIXED",
            delta_bytes=-3422,
            new_d_seg=0.009,
            from_state="c1:wf7-base-edge",
        ),
        opportunity_pool_id="WF7_STREAM",
    )

    plan = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=(pf3, wf7))

    selected = [row for row in plan["selected_home_owners"] if row["selection"] != "BASELINE_OWNER"]
    assert len(selected) == 1
    assert plan["policy"]["pf3_coordinate_and_wf7_stream_pools_additive"] is False
    assert "CROSS_EDGE_PRICES_NOT_COMPOSED;SINGLETON_COMPARISON_ONLY" in plan[
        "interaction_or_commutator_blockers"
    ]
    assert plan["exact_score_transition"]["delta_score_total"] != (
        pf3.receipt.action_effect.delta_score_total + wf7.receipt.action_effect.delta_score_total
    )


def test_cross_base_candidate_set_fails_closed() -> None:
    left = ReceiptEnvelope(
        _receipt(
            receipt_id="left",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
            base_sha="a" * 64,
        )
    )
    right = ReceiptEnvelope(
        _receipt(
            receipt_id="right",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
            base_sha="b" * 64,
        )
    )

    plan = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=(left, right))

    assert plan["status"] == "BLOCKED_NO_VALID_APPLIED_TRANSITION"
    assert plan["selected_identity"] is None
    assert "CROSS_BASE_CANDIDATE_SET_REFUSED" in plan["interaction_or_commutator_blockers"]


def test_stream_home_base_bytes_must_match_exact_ev2_owner() -> None:
    receipt = _receipt(
        receipt_id="wrong-home-mass",
        home_id="manifest",
        coder_id="A",
        delta_bytes=-10,
        home_before_override=HOME_BYTES["manifest"] + 1,
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(ReceiptEnvelope(receipt),)
    )

    rejected = {row["receipt_id"]: row["blockers"] for row in plan["rejected"]}
    assert "STREAM_HOME_BASE_BYTES_DIFFER_FROM_EV2_OWNER" in rejected["wrong-home-mass"]


def test_shared_base_identity_with_different_score_endpoint_fails_closed() -> None:
    left = ReceiptEnvelope(
        _receipt(
            receipt_id="left-endpoint",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
            old_d_seg=0.01,
        )
    )
    right = ReceiptEnvelope(
        _receipt(
            receipt_id="right-endpoint",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
            old_d_seg=0.02,
            new_d_seg=0.019,
        )
    )

    plan = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=(left, right))

    assert plan["status"] == "BLOCKED_NO_VALID_APPLIED_TRANSITION"
    assert "BASE_SCORE_ENDPOINTS_DIFFER_FOR_SHARED_PRICE_SET" in plan[
        "interaction_or_commutator_blockers"
    ]


def test_cross_edge_composition_requires_same_from_state_and_measured_commutator() -> None:
    first = ReceiptEnvelope(
        _receipt(
            receipt_id="component-a",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
            from_state="base:A",
        )
    )
    second = ReceiptEnvelope(
        _receipt(
            receipt_id="component-b",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
            from_state="base:B",
        )
    )
    composed = ReceiptEnvelope(
        _receipt(
            receipt_id="composed",
            home_id="manifest",
            coder_id="COMPOSED",
            delta_bytes=-20,
            from_state="base:A",
            composed_action_ids=(first.receipt.action_id, second.receipt.action_id),
            interaction=None,
            aggregate_home=True,
        ),
        component_receipt_ids=(first.receipt.receipt_id, second.receipt.receipt_id),
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(first, second, composed)
    )
    rejected = {row["receipt_id"]: row["blockers"] for row in plan["rejected"]}

    assert "CROSS_EDGE_FROM_STATE_COMPOSITION_REFUSED" in rejected["composed"]
    assert "MEASURED_INTERACTION_OR_COMMUTATOR_ABSENT" in rejected["composed"]


def test_composed_transition_requires_aggregate_support_identity() -> None:
    first = ReceiptEnvelope(
        _receipt(
            receipt_id="supported-a",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
            from_state="base:A",
        )
    )
    second = ReceiptEnvelope(
        _receipt(
            receipt_id="supported-b",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
            from_state="base:A",
        )
    )
    composed = ReceiptEnvelope(
        _receipt(
            receipt_id="unsupported-composed",
            home_id="manifest",
            coder_id="COMPOSED",
            delta_bytes=-20,
            from_state="base:A",
            composed_action_ids=(first.receipt.action_id, second.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
            support_sha256=None,
        ),
        component_receipt_ids=(first.receipt.receipt_id, second.receipt.receipt_id),
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(first, second, composed)
    )
    rejected = {row["receipt_id"]: row["blockers"] for row in plan["rejected"]}

    assert "COMPOSED_SUPPORT_IDENTITY_ABSENT" in rejected["unsupported-composed"]


def test_pose_is_reconciled_with_exact_nonlinear_sqrt_term() -> None:
    receipt = _receipt(
        receipt_id="nonlinear-pose",
        home_id="solved_template_outer_home",
        coder_id="POSE_JOINT",
        delta_bytes=0,
        old_d_seg=0.01,
        new_d_seg=0.012,
        old_d_pose=0.04,
        new_d_pose=0.01,
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(ReceiptEnvelope(receipt),)
    )
    exact = 100.0 * (0.012 - 0.01) + math.sqrt(10.0 * 0.01) - math.sqrt(10.0 * 0.04)

    assert math.isclose(plan["exact_score_transition"]["delta_score_total"], exact, abs_tol=1e-15)
    assert plan["policy"]["linearized_pose_used"] is False


def test_target_is_loaded_dynamically_and_never_hardcoded() -> None:
    plan = build_allocation_plan(ev2=_ev2(), pointer=_pointer(0.14321), envelopes=())

    assert plan["dynamic_target"]["score"] == 0.14321
    assert plan["status"] == "BLOCKED_NO_VALID_APPLIED_TRANSITION"
    assert plan["unpriced_homes"] == list(SEVEN_HOME_IDS)


def test_permutation_determinism_is_byte_stable() -> None:
    rows = (
        ReceiptEnvelope(
            _receipt(
                receipt_id="z-last",
                home_id="manifest",
                coder_id="Z",
                delta_bytes=-10,
            )
        ),
        ReceiptEnvelope(
            _receipt(
                receipt_id="a-first",
                home_id="manifest",
                coder_id="A",
                delta_bytes=-20,
            )
        ),
    )

    forward = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=rows)
    reverse = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=tuple(reversed(rows)))

    assert forward == reverse
    assert forward["plan_content_sha256"] == reverse["plan_content_sha256"]
