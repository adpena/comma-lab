from __future__ import annotations

import struct

import numpy as np
import pytest

from experiments import ddm_js6_seg_representation_join as js6
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def _spec(*, sites: tuple[int, ...] = (10 * js6.W + 10, 10 * js6.W + 11)) -> js6.ComponentSpec:
    return js6.ComponentSpec(
        pair=7,
        source_class=0,
        target_class=1,
        component_id=1,
        component_pixels=2,
        token_sites=sites,
    )


def _component() -> np.ndarray:
    value = np.zeros((js6.H, js6.W), dtype=bool)
    value[10, 10:12] = True
    return value


def test_join_payload_round_trips_real_g1_and_ec1_objects() -> None:
    payload, metadata = js6.encode_join_payload(_spec(), _component())
    decoded = js6.decode_join_payload(payload)
    assert decoded["pair"] == 7
    assert decoded["family"] == "g1_worldsheet_event_edit"
    assert decoded["event_type"] == "lane_program_delta"
    assert np.count_nonzero(decoded["component"]) > 0
    assert decoded["token_sites"].tolist() == [10 * js6.W + 10, 10 * js6.W + 11]
    assert metadata["worldsheet_metadata"]["decoded_mask_errors"] == metadata[
        "worldsheet_component_symmetric_difference_pixels"
    ]


def test_join_payload_refuses_a_singleton_coordinate_event() -> None:
    spec = _spec(sites=(10 * js6.W + 10,))
    with pytest.raises(js6.JS6Error, match="coupled object"):
        js6.encode_join_payload(spec, _component())


def test_join_payload_refuses_trailing_bytes_and_outer_inner_mismatch() -> None:
    payload, _ = js6.encode_join_payload(_spec(), _component())
    with pytest.raises(js6.JS6Error, match="length/trailing"):
        js6.decode_join_payload(payload + b"x")
    values = list(js6.JOIN_HEADER.unpack_from(payload))
    values[2] = 2
    corrupted = js6.JOIN_HEADER.pack(*values) + payload[js6.JOIN_HEADER.size :]
    with pytest.raises(js6.JS6Error, match="outer header"):
        js6.decode_join_payload(corrupted)


def test_event_type_is_derived_from_directed_edge_role() -> None:
    assert js6.event_type(0, 1) == js6.ec1.EVENT_TYPE["lane_program_delta"]
    assert js6.event_type(1, 0) == js6.ec1.EVENT_TYPE["lane_program_delta"]
    assert js6.event_type(2, 3) == js6.ec1.EVENT_TYPE["island_birth"]
    assert js6.event_type(3, 2) == js6.ec1.EVENT_TYPE["island_death"]
    assert js6.event_type(2, 0) == js6.ec1.EVENT_TYPE["boundary_offset"]


def test_component_derivation_uses_nearest_current_source_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(js6, "N", 1)
    base = np.zeros((1, js6.H, js6.W), dtype=np.uint8)
    gt = base.copy()
    gt[0, 10, 10:14] = 1
    tokens = np.full_like(base, 2)
    tokens[0, 9, 10] = 0
    tokens[0, 11, 13] = 0
    rows = js6.component_specs(base, gt, tokens)
    assert len(rows) == 1
    assert rows[0].component_pixels == 4
    assert rows[0].source_class == 0
    assert rows[0].target_class == 1
    assert rows[0].token_sites == (9 * js6.W + 10, 11 * js6.W + 13)


def test_component_derivation_excludes_ec3_singletons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(js6, "N", 1)
    base = np.zeros((1, js6.H, js6.W), dtype=np.uint8)
    gt = base.copy()
    gt[0, 10, 10:12] = 1
    tokens = np.full_like(base, 2)
    tokens[0, 9, 10] = 0
    assert js6.component_specs(base, gt, tokens) == []


def test_edge_census_is_directed_and_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(js6, "N", 1)
    base = np.zeros((1, js6.H, js6.W), dtype=np.uint8)
    gt = base.copy()
    gt[0, 0, :3] = 1
    base[0, 1, :2] = 1
    census = js6.edge_census(base, gt)
    assert census[(0, 1)] == 3
    assert census[(1, 0)] == 2
    assert sum(census.values()) == 5


def test_header_layout_is_stable() -> None:
    assert js6.JOIN_HEADER.size == struct.calcsize("<8sHBBBBII")
    assert js6.JOIN_MAGIC == b"JS6JOIN1"


def test_zero_pose_break_even_integer_boundary() -> None:
    assert js6.zero_pose_break_even_max_integer_bytes(0) == -1
    assert js6.zero_pose_break_even_max_integer_bytes(52) == 66
    with pytest.raises(js6.JS6Error, match="nonnegative integer"):
        js6.zero_pose_break_even_max_integer_bytes(-1)


def test_source_passes_always_keep_payload_detector() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=js6.REPO,
        strict=False,
        roots=("experiments/ddm_js6_seg_representation_join.py",),
    )
    assert findings == []


def test_resume_refuses_stale_source_or_changed_fresh_payload(tmp_path) -> None:
    store = tmp_path / "store"
    root = store / "proposals" / "p"
    payload = b"payload"
    payload_record = js6.atomic_bytes(root / "proposal.js6p", payload)
    receipt = {
        "schema": "ddm_js6_seg_representation_proposal.v3",
        "producer_source_sha256": "source-a",
        "retained_payloads": {"proposal.js6p": payload_record},
    }
    receipt_path = root / "proposal.json"
    js6.atomic_json(receipt_path, receipt)
    assert js6.load_prior(
        receipt_path,
        store,
        expected_payload_sha256=payload_record["sha256"],
        producer_source_sha256="source-a",
    ) == receipt
    assert (
        js6.load_prior(
            receipt_path,
            store,
            expected_payload_sha256=payload_record["sha256"],
            producer_source_sha256="source-b",
        )
        is None
    )
    with pytest.raises(js6.JS6Error, match="freshly derived"):
        js6.load_prior(
            receipt_path,
            store,
            expected_payload_sha256="0" * 64,
            producer_source_sha256="source-a",
        )


def test_retention_helpers_never_overwrite_different_signal(tmp_path) -> None:
    payload_path = tmp_path / "payload.bin"
    js6.retain_bytes(payload_path, b"first")
    assert js6.retain_bytes(payload_path, b"first")["bytes"] == 5
    with pytest.raises(js6.JS6Error, match="refusing to overwrite"):
        js6.retain_bytes(payload_path, b"second")

    array_path = tmp_path / "array.npy"
    first = np.arange(8, dtype=np.float32)
    js6.retain_npy(array_path, first)
    assert js6.retain_npy(array_path, first)["bytes"] > first.nbytes
    with pytest.raises(js6.JS6Error, match="refusing to overwrite"):
        js6.retain_npy(array_path, first + 1)


def test_atomic_write_cleans_partial_and_preserves_target_on_sync_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"prior")

    def fail_sync(_fd: int) -> None:
        raise OSError("forced sync failure")

    monkeypatch.setattr(js6.os, "fsync", fail_sync)
    with pytest.raises(OSError, match="forced sync failure"):
        js6.atomic_bytes(target, b"replacement")
    assert target.read_bytes() == b"prior"
    assert list(tmp_path.glob(".*.partial")) == []


def test_prior_receipt_shortcut_is_closed_by_final_hash_audit(tmp_path) -> None:
    target = tmp_path / "payload.bin"
    record = js6.atomic_bytes(target, b"first")
    assert js6.retain_bytes(target, b"first", prior_record=record) == record
    target.write_bytes(b"other")
    assert js6.retain_bytes(target, b"first", prior_record=record) == record
    with pytest.raises(js6.JS6Error, match="retained artifact differs"):
        js6.require_record(record, beneath=tmp_path)
