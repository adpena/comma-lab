# SPDX-License-Identifier: MIT
"""Behavior tests for the identity-bearing applied-action bridge."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tac.analysis.action_effect import ActionEffect
from tac.analysis.applied_action_receipt import (
    ApplicationStatus,
    AppliedActionReceipt,
    AppliedActionReceiptError,
    StreamHomeClaim,
    build_applied_action_receipt,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType

BASE_SHA = "a" * 64
CANDIDATE_SHA = "b" * 64
STATE_SHA = "c" * 64
RECEIVER_SHA = "d" * 64
CHANGED_SHA = "e" * 64


def _effect(*, new_d_seg: float = 0.0009, composed: tuple[str, ...] = ()) -> ActionEffect:
    return ActionEffect.build(
        action_id="edge-action-1",
        family="ddm",
        action_kind="counted_receiver_edge",
        authority="batch_local_receiver_exact",
        producer="fixture",
        pair_ids=(523,),
        composed_action_ids=composed,
        old_d_seg=0.001,
        new_d_seg=new_d_seg,
        old_d_pose=0.00002,
        new_d_pose=0.00002,
        old_bytes=1000,
        new_bytes=1010,
        base_archive_sha256=BASE_SHA,
        archive_sha256=CANDIDATE_SHA,
        base_state_sha256=STATE_SHA,
        support_sha256="f" * 64,
    )


def _stream_home() -> StreamHomeClaim:
    return StreamHomeClaim(
        stream_type=StreamType.SKELETON,
        layer_home=LayerHome.L3_RASTER,
        byte_home_id="skeleton/pair523/edge17",
        coder_id="e4_brotli_v1",
        coder_owner="ddm_e4_archive",
        receiver_consumer="j8f_counted_application",
        bytes_before=50,
        bytes_after=60,
    )


def _receipt(*, effect: ActionEffect | None = None) -> AppliedActionReceipt:
    return build_applied_action_receipt(
        receipt_id="receipt-edge-1",
        status=ApplicationStatus.DOWNHILL_FINITE,
        action_effect=effect or _effect(),
        codeword_id="dm4:proposal:523:17",
        application_operator_id="ddm_dm4_j5_counted_application.select_counted_application",
        application_operator_version="j8f-v1",
        physical_edge_id="j5:pair523:coord17:+1",
        edge_from_state_id="j5:base",
        edge_to_state_id="j5:base+coord17",
        integer_quantum=1,
        direction=1,
        validity_radius=1.0,
        receiver_schema="ddm_j5_receiver.v1",
        receiver_sha256=RECEIVER_SHA,
        r_chain_id="uint8-resize-segnet-posenet-v1",
        changed_uint8_count=90,
        changed_uint8_sha256=CHANGED_SHA,
        stream_home=_stream_home(),
        verdict_scope="INSTANCE:pair523:edge17",
        provenance_ref="fixture://j8f/receipt.json",
        bucket_id="pair523/cell17",
    )


def test_applied_action_receipt_roundtrips_all_foreign_keys() -> None:
    receipt = _receipt(effect=_effect(composed=("seg-action", "pose-action")))
    payload = json.loads(json.dumps(receipt.as_dict()))
    restored = AppliedActionReceipt.from_dict(payload)

    assert restored == receipt
    assert restored.ordered_parent_action_ids == ("seg-action", "pose-action")
    assert restored.stream_home.delta_bytes == restored.action_effect.delta_bytes == 10
    assert restored.promotion_eligible is False
    assert restored.score_claim is False
    assert restored.research_only is True


def test_applied_action_receipt_rejects_base_candidate_conflation() -> None:
    receipt = _receipt()
    with pytest.raises(AppliedActionReceiptError, match="base archive identity differs"):
        replace(receipt, base_archive_sha256="0" * 64)
    with pytest.raises(AppliedActionReceiptError, match="candidate archive identity differs"):
        replace(receipt, candidate_archive_sha256="1" * 64)


def test_applied_action_receipt_rejects_stream_byte_smearing() -> None:
    receipt = _receipt()
    wrong_home = replace(receipt.stream_home, bytes_after=61)
    with pytest.raises(AppliedActionReceiptError, match="byte delta differs"):
        replace(receipt, stream_home=wrong_home)


def test_applied_action_receipt_rejects_wrong_score_direction() -> None:
    uphill = _effect(new_d_seg=0.0011)
    with pytest.raises(AppliedActionReceiptError, match="DOWNHILL_FINITE"):
        _receipt(effect=uphill)


def test_applied_action_receipt_requires_changed_uint8_hash() -> None:
    receipt = _receipt()
    with pytest.raises(AppliedActionReceiptError, match="changed_uint8_sha256"):
        replace(receipt, changed_uint8_sha256=None)


def test_applied_action_receipt_requires_exact_ordered_parent_identity() -> None:
    receipt = _receipt(effect=_effect(composed=("A", "B")))
    with pytest.raises(AppliedActionReceiptError, match="ordered parent"):
        replace(receipt, ordered_parent_action_ids=("B", "A"))


def test_application_builder_refuses_legacy_unbound_action_effect() -> None:
    effect = replace(_effect(), base_archive_sha256=None)
    with pytest.raises(AppliedActionReceiptError, match="both base and candidate"):
        _receipt(effect=effect)
