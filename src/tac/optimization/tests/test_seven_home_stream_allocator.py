# SPDX-License-Identifier: MIT
"""Behavior tests for the lawful EV2 seven-home allocator."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

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
    SevenHomeAllocationError,
    build_allocation_plan,
    derive_seven_homes,
    envelopes_from_manifest,
)
from tools.allocate_seven_home_streams import build_from_paths

REPO = Path(__file__).resolve().parents[4]
EV2_PATH = REPO / ".omx/research/ddm_ev2_per_pair_allocation_20260725T041933Z/allocation_table.json"
POINTER_PATH = REPO / ".omx/state/canonical_frontier_pointer.json"
CC3_PATH = REPO / ".omx/research/ddm_cc3_mixed_coder_receiver_receipt_20260725.json"
E5A_PATH = (
    REPO
    / ".omx/research/ddm_e5a_midcampaign_e5_adapter_20260725"
    / "ddm_e5a_midcampaign_runtime_export_receipt.json"
)
ADAPTER_PATH = REPO / ".omx/research/applied_action_adapter_manifest_20260725.json"
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
HOME_TYPES = {
    "manifest": (StreamType.SKELETON, LayerHome.L1_PROGRAM),
    "v15_predictor_zip_outer_home": (StreamType.CONNECTION, LayerHome.L2_CHART),
    "g1_movable_worldsheet_outer_home": (StreamType.CONNECTION, LayerHome.L2_CHART),
    "receiver_realization_profile": (StreamType.SKELETON, LayerHome.L1_PROGRAM),
    "solved_template_outer_home": (StreamType.FIBER, LayerHome.L4_SCORER_FEATURE),
    "central_directory_and_eocd": (StreamType.SKELETON, LayerHome.L1_PROGRAM),
    "lane_program_seed": (StreamType.CONNECTION, LayerHome.L2_CHART),
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
    old_bytes: int = BASE_BYTES,
    authority: str = "inflate_torch_cpu",
    stream_type_override: StreamType | None = None,
    layer_home_override: LayerHome | None = None,
    effect_blockers: tuple[str, ...] = (),
    receipt_blockers: tuple[str, ...] = (),
    action_id_override: str | None = None,
):
    action_id = action_id_override or f"action:{receipt_id}"
    effect = ActionEffect.build(
        action_id=action_id,
        family="ddm",
        action_kind="counted_receiver_edge",
        authority=authority,
        normalization_scope="full_video_exact",
        producer="seven-home-test",
        pair_ids=(0, 1),
        composed_action_ids=composed_action_ids,
        old_d_seg=old_d_seg,
        new_d_seg=new_d_seg,
        old_d_pose=old_d_pose,
        new_d_pose=new_d_pose,
        old_bytes=old_bytes,
        new_bytes=old_bytes + delta_bytes,
        base_archive_sha256=base_sha,
        archive_sha256=(receipt_id.encode().hex() + "0" * 64)[:64],
        base_state_sha256="c" * 64,
        support_sha256=support_sha256,
        interaction_or_commutator=interaction,
        blockers=effect_blockers,
    )
    home_before = (
        home_before_override
        if home_before_override is not None
        else BASE_BYTES if aggregate_home else HOME_BYTES[home_id]
    )
    expected_stream_type, expected_layer_home = HOME_TYPES[home_id]
    stream_home = StreamHomeClaim(
        stream_type=stream_type_override or (
            StreamType.RESIDUAL if aggregate_home else expected_stream_type
        ),
        layer_home=layer_home_override or (
            LayerHome.L1_PROGRAM if aggregate_home else expected_layer_home
        ),
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
        blockers=receipt_blockers,
    )


def _content_hashed(payload: dict) -> dict:
    result = copy.deepcopy(payload)
    result.pop("content_sha256", None)
    result["content_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return result


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


@pytest.mark.parametrize(
    ("field", "value"),
    (("type", "RESIDUAL"), ("layer_home", "L3_residual"), ("extra", "smuggled")),
)
def test_ev2_typed_home_contract_is_exact(field: str, value: str) -> None:
    ev2 = _ev2()
    row = next(
        item for item in ev2["coarse_lawful_partition"]["rows"] if item["stream"] == "manifest"
    )
    row["typed_home"][field] = value

    with pytest.raises(SevenHomeAllocationError, match="typed"):
        derive_seven_homes(ev2)


def test_ev2_partition_and_home_structures_reject_extra_fields() -> None:
    partition_extra = _ev2()
    partition_extra["coarse_lawful_partition"]["smuggled"] = True
    with pytest.raises(SevenHomeAllocationError, match="partition schema"):
        derive_seven_homes(partition_extra)

    home_extra = _ev2()
    home_extra["coarse_lawful_partition"]["rows"][0]["smuggled"] = True
    with pytest.raises(SevenHomeAllocationError, match="row schema"):
        derive_seven_homes(home_extra)


@pytest.mark.parametrize(
    ("stream_type", "layer_home", "expected"),
    (
        (StreamType.RESIDUAL, None, "STREAM_HOME_TYPE_DIFFERS_FROM_EV2_TYPED_HOME"),
        (None, LayerHome.L3_RASTER, "STREAM_HOME_LAYER_DIFFERS_FROM_EV2_TYPED_HOME"),
    ),
)
def test_receipt_type_and_layer_must_match_ev2_home(
    stream_type: StreamType | None,
    layer_home: LayerHome | None,
    expected: str,
) -> None:
    receipt = _receipt(
        receipt_id=f"wrong-{expected}",
        home_id="manifest",
        coder_id="A",
        delta_bytes=-10,
        stream_type_override=stream_type,
        layer_home_override=layer_home,
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(ReceiptEnvelope(receipt),)
    )

    assert expected in plan["rejected"][0]["blockers"]


def test_action_effect_base_bytes_must_equal_ev2_partition() -> None:
    receipt = _receipt(
        receipt_id="wrong-archive-base",
        home_id="manifest",
        coder_id="A",
        delta_bytes=-10,
        old_bytes=BASE_BYTES + 1,
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(ReceiptEnvelope(receipt),)
    )

    assert "BASE_ARCHIVE_BYTES_DIFFER_FROM_EV2_PARTITION" in plan["rejected"][0][
        "blockers"
    ]


def test_ev2_partition_mass_cannot_be_rebased_away_from_134211() -> None:
    ev2 = _ev2()
    partition = ev2["coarse_lawful_partition"]
    lane_seed = next(row for row in partition["rows"] if row["stream"] == "lane_program_seed")
    lane_seed["counted_bytes"] += 1
    partition["counted_bytes"] += 1

    with pytest.raises(SevenHomeAllocationError, match="must equal 134211"):
        derive_seven_homes(ev2)


def test_source_receipt_and_action_effect_blockers_propagate() -> None:
    receipt = _receipt(
        receipt_id="blocked-at-source",
        home_id="manifest",
        coder_id="A",
        delta_bytes=-10,
        receipt_blockers=("receipt-custody-gap",),
        effect_blockers=("effect-parseback-gap",),
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(ReceiptEnvelope(receipt),)
    )
    blockers = plan["rejected"][0]["blockers"]

    assert "SOURCE_RECEIPT_BLOCKER:receipt-custody-gap" in blockers
    assert "SOURCE_ACTION_EFFECT_BLOCKER:effect-parseback-gap" in blockers


def test_dependency_cycle_is_rejected_without_recursion_escape() -> None:
    leaf_a = ReceiptEnvelope(
        _receipt(
            receipt_id="cycle-leaf-a",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
        )
    )
    leaf_b = ReceiptEnvelope(
        _receipt(
            receipt_id="cycle-leaf-b",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
        )
    )
    left = ReceiptEnvelope(
        _receipt(
            receipt_id="cycle-left",
            home_id="manifest",
            coder_id="A+B",
            delta_bytes=-20,
            composed_action_ids=("action:cycle-right", leaf_a.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=("cycle-right", leaf_a.receipt.receipt_id),
    )
    right = ReceiptEnvelope(
        _receipt(
            receipt_id="cycle-right",
            home_id="lane_program_seed",
            coder_id="B+A",
            delta_bytes=-20,
            composed_action_ids=("action:cycle-left", leaf_b.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=("cycle-left", leaf_b.receipt.receipt_id),
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(leaf_a, leaf_b, left, right)
    )
    rejected = {row["receipt_id"]: row["blockers"] for row in plan["rejected"]}

    assert "DEPENDENCY_CYCLE_DETECTED" in rejected["cycle-left"]
    assert "DEPENDENCY_CYCLE_DETECTED" in rejected["cycle-right"]


def test_composition_cannot_launder_an_inadmissible_dependency() -> None:
    bad_leaf = ReceiptEnvelope(
        _receipt(
            receipt_id="bad-leaf",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
            stream_type_override=StreamType.RESIDUAL,
        )
    )
    good_leaf = ReceiptEnvelope(
        _receipt(
            receipt_id="good-leaf",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
        )
    )
    parent = ReceiptEnvelope(
        _receipt(
            receipt_id="laundering-parent",
            home_id="manifest",
            coder_id="AGG",
            delta_bytes=-20,
            composed_action_ids=(bad_leaf.receipt.action_id, good_leaf.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=(bad_leaf.receipt.receipt_id, good_leaf.receipt.receipt_id),
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(bad_leaf, good_leaf, parent)
    )
    rejected = {row["receipt_id"]: row["blockers"] for row in plan["rejected"]}

    assert "STREAM_HOME_TYPE_DIFFERS_FROM_EV2_TYPED_HOME" in rejected["bad-leaf"]
    assert "DEPENDENCY_NOT_INDEPENDENTLY_ADMISSIBLE:bad-leaf" in rejected[
        "laundering-parent"
    ]


def test_nested_composition_closes_over_independently_admissible_leaf_receipts() -> None:
    leaf_a = ReceiptEnvelope(
        _receipt(
            receipt_id="nested-a",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
        )
    )
    leaf_b = ReceiptEnvelope(
        _receipt(
            receipt_id="nested-b",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
        )
    )
    child = ReceiptEnvelope(
        _receipt(
            receipt_id="nested-child",
            home_id="manifest",
            coder_id="AB",
            delta_bytes=-20,
            composed_action_ids=(leaf_a.receipt.action_id, leaf_b.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=(leaf_a.receipt.receipt_id, leaf_b.receipt.receipt_id),
    )
    leaf_c = ReceiptEnvelope(
        _receipt(
            receipt_id="nested-c",
            home_id="receiver_realization_profile",
            coder_id="C",
            delta_bytes=-10,
        )
    )
    parent = ReceiptEnvelope(
        _receipt(
            receipt_id="nested-parent",
            home_id="manifest",
            coder_id="ABC",
            delta_bytes=-30,
            composed_action_ids=(child.receipt.action_id, leaf_c.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=(child.receipt.receipt_id, leaf_c.receipt.receipt_id),
    )

    plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(leaf_a, leaf_b, child, leaf_c, parent)
    )
    selected = {
        row["home_id"]: row["receipt_id"]
        for row in plan["selected_home_owners"]
        if row["selection"] == "APPLIED_ACTION_RECEIPT"
    }

    assert plan["selected_identity"]["receipt_id"] == "nested-parent"
    assert selected == {
        "manifest": "nested-a",
        "lane_program_seed": "nested-b",
        "receiver_realization_profile": "nested-c",
    }


def test_nested_composition_cannot_reuse_one_action_under_two_receipt_ids() -> None:
    first = ReceiptEnvelope(
        _receipt(
            receipt_id="shared-action-left",
            home_id="manifest",
            coder_id="A",
            delta_bytes=-10,
            action_id_override="action:shared-leaf",
        )
    )
    auxiliary = ReceiptEnvelope(
        _receipt(
            receipt_id="shared-action-aux",
            home_id="lane_program_seed",
            coder_id="B",
            delta_bytes=-10,
        )
    )
    child = ReceiptEnvelope(
        _receipt(
            receipt_id="shared-action-child",
            home_id="manifest",
            coder_id="AB",
            delta_bytes=-20,
            composed_action_ids=(first.receipt.action_id, auxiliary.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=(first.receipt.receipt_id, auxiliary.receipt.receipt_id),
    )
    duplicate_action = ReceiptEnvelope(
        _receipt(
            receipt_id="shared-action-right",
            home_id="receiver_realization_profile",
            coder_id="C",
            delta_bytes=-10,
            action_id_override="action:shared-leaf",
        )
    )
    parent = ReceiptEnvelope(
        _receipt(
            receipt_id="shared-action-parent",
            home_id="manifest",
            coder_id="ABC",
            delta_bytes=-30,
            composed_action_ids=(child.receipt.action_id, duplicate_action.receipt.action_id),
            interaction=0.0,
            aggregate_home=True,
        ),
        component_receipt_ids=(child.receipt.receipt_id, duplicate_action.receipt.receipt_id),
    )

    plan = build_allocation_plan(
        ev2=_ev2(),
        pointer=_pointer(),
        envelopes=(first, auxiliary, child, duplicate_action, parent),
    )
    rejected = {row["receipt_id"]: row["blockers"] for row in plan["rejected"]}

    assert "NESTED_DEPENDENCY_CLOSURE_REUSES_ACTION" in rejected["shared-action-parent"]


def test_mixed_authority_axes_are_not_compared() -> None:
    cpu = ReceiptEnvelope(
        _receipt(
            receipt_id="axis-cpu",
            home_id="manifest",
            coder_id="CPU",
            delta_bytes=-10,
            authority="contest_cpu",
        )
    )
    cuda = ReceiptEnvelope(
        _receipt(
            receipt_id="axis-cuda",
            home_id="lane_program_seed",
            coder_id="CUDA",
            delta_bytes=-10,
            authority="contest_cuda",
        )
    )

    plan = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=(cpu, cuda))

    assert plan["status"] == "BLOCKED_NO_VALID_APPLIED_TRANSITION"
    assert "MIXED_AUTHORITY_AXES_REFUSED" in plan["interaction_or_commutator_blockers"]


def test_dynamic_target_comparison_is_null_for_advisory_axis_and_bool_for_contest() -> None:
    advisory = ReceiptEnvelope(
        _receipt(
            receipt_id="axis-advisory",
            home_id="manifest",
            coder_id="ADV",
            delta_bytes=-10,
            authority="inflate_torch_cpu",
        )
    )
    contest = ReceiptEnvelope(
        _receipt(
            receipt_id="axis-contest",
            home_id="manifest",
            coder_id="CPU",
            delta_bytes=-10,
            authority="contest_cpu",
        )
    )

    advisory_plan = build_allocation_plan(
        ev2=_ev2(), pointer=_pointer(), envelopes=(advisory,)
    )
    contest_plan = build_allocation_plan(ev2=_ev2(), pointer=_pointer(), envelopes=(contest,))

    assert advisory_plan["exact_score_transition"]["beats_dynamic_target"] is None
    assert advisory_plan["exact_score_transition"]["dynamic_target_comparison_eligible"] is False
    assert isinstance(contest_plan["exact_score_transition"]["beats_dynamic_target"], bool)
    assert contest_plan["exact_score_transition"]["dynamic_target_comparison_eligible"] is True

    incompatible_pointer = _pointer()
    incompatible_pointer["effective_frontier"]["source_kind"] = "local_advisory_snapshot"
    incompatible_plan = build_allocation_plan(
        ev2=_ev2(), pointer=incompatible_pointer, envelopes=(contest,)
    )
    assert incompatible_plan["dynamic_target"]["contest_compatible"] is False
    assert incompatible_plan["exact_score_transition"]["beats_dynamic_target"] is None


def test_dynamic_target_rejects_numeric_string_coercion() -> None:
    pointer = _pointer()
    pointer["effective_frontier"]["score"] = "0.172"

    with pytest.raises(SevenHomeAllocationError, match="must be finite"):
        build_allocation_plan(ev2=_ev2(), pointer=pointer, envelopes=())


def test_receipt_envelope_rejects_nonstring_metadata_coercion() -> None:
    receipt = _receipt(
        receipt_id="typed-envelope",
        home_id="manifest",
        coder_id="A",
        delta_bytes=-10,
    )

    with pytest.raises(SevenHomeAllocationError, match="opportunity_pool_id"):
        ReceiptEnvelope.from_mapping({"receipt": receipt.as_dict(), "opportunity_pool_id": 7})
    with pytest.raises(SevenHomeAllocationError, match="component_receipt_ids"):
        ReceiptEnvelope.from_mapping({"receipt": receipt.as_dict(), "component_receipt_ids": [7]})


def test_receipt_manifest_self_hash_count_and_false_authority_are_strict() -> None:
    receipt = _receipt(
        receipt_id="manifest-row",
        home_id="manifest",
        coder_id="A",
        delta_bytes=-10,
    )
    valid = _content_hashed(
        {
            "schema": "tac.seven_home_receipt_manifest.v1",
            "research_only": True,
            "promotion_eligible": False,
            "score_claim": False,
            "receipt_count": 1,
            "receipts": [receipt.as_dict()],
        }
    )
    assert len(envelopes_from_manifest(valid)) == 1

    tampered_hash = copy.deepcopy(valid)
    tampered_hash["receipts"][0]["coder_owner"] = "tampered"
    with pytest.raises(SevenHomeAllocationError, match="content_sha256"):
        envelopes_from_manifest(tampered_hash)

    wrong_count = _content_hashed({**valid, "receipt_count": 2})
    with pytest.raises(SevenHomeAllocationError, match="count"):
        envelopes_from_manifest(wrong_count)

    false_authority = _content_hashed({**valid, "promotion_eligible": True})
    with pytest.raises(SevenHomeAllocationError, match="promotion_eligible"):
        envelopes_from_manifest(false_authority)


def test_adapter_manifest_result_shape_count_and_false_authority_are_strict() -> None:
    blocker = {
        "code": "OWED",
        "source_key": "fixture.source",
        "owed_field": "fixture.field",
        "detail": "fixture detail",
    }
    blocked_result = {
        "schema": "tac.applied_action_adapter_result.v1",
        "source_kind": "PF3",
        "source_schema": "fixture.v1",
        "source_id": "fixture-id",
        "ok": False,
        "receipt": None,
        "blockers": [blocker],
        "research_only": True,
        "promotion_eligible": False,
        "score_claim": False,
    }
    valid = _content_hashed(
        {
            "schema": "tac.applied_action_adapter_manifest.v1",
            "receipt_count": 0,
            "blocked_source_count": 1,
            "results": [blocked_result],
            "research_only": True,
            "promotion_eligible": False,
            "score_claim": False,
        }
    )
    assert envelopes_from_manifest(valid) == ()

    wrong_type = _content_hashed({**valid, "receipt_count": "0"})
    with pytest.raises(SevenHomeAllocationError, match="exact integer"):
        envelopes_from_manifest(wrong_type)

    wrong_row_authority = copy.deepcopy(valid)
    wrong_row_authority["results"][0]["score_claim"] = True
    wrong_row_authority = _content_hashed(wrong_row_authority)
    with pytest.raises(SevenHomeAllocationError, match="score_claim"):
        envelopes_from_manifest(wrong_row_authority)

    invalid_shape = copy.deepcopy(valid)
    invalid_shape["results"][0]["receipt"] = {"schema": "smuggled"}
    invalid_shape = _content_hashed(invalid_shape)
    with pytest.raises(SevenHomeAllocationError, match="blockers and no receipt"):
        envelopes_from_manifest(invalid_shape)


def test_cli_records_strict_adapter_manifest_reconciliation() -> None:
    plan = build_from_paths(
        ev2_path=EV2_PATH,
        pointer_path=POINTER_PATH,
        cc3_path=CC3_PATH,
        e5a_path=E5A_PATH,
        receipt_manifest_paths=[ADAPTER_PATH],
    )

    source = plan["input_receipt_manifests"][0]
    assert source["receipt_count"] == 0
    assert source["reconciliation"] == {
        "declared_receipt_count": 0,
        "declared_blocked_source_count": 3,
        "result_count": 3,
        "parsed_receipt_count": 0,
        "counts_reconciled": True,
        "content_sha256_reconciled": True,
        "false_authority_reconciled": True,
    }
    assert plan["status"] == "BLOCKED_NO_VALID_APPLIED_TRANSITION"
