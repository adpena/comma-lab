from __future__ import annotations

from pathlib import Path

import numpy as np

from experiments import ddm_ec1_event_coordinate_producer as ec1
from experiments import ddm_ec3_t4_targeted_events as ec3
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_site_selector_prefers_lower_pose_sensitivity_at_equal_target_mass() -> None:
    tokens = np.zeros((ec3.H, ec3.W), dtype=np.uint8)
    base = np.zeros_like(tokens)
    gt = np.zeros_like(tokens)
    gt[100, 100] = 1
    gt[100, 300] = 1
    sensitivity = np.ones((ec3.H, ec3.W), dtype=np.float32)
    sensitivity[:, :200] = 10.0
    option = ec3.select_token_site(tokens, base, gt, sensitivity, 0, 1, pair=9)
    assert option is not None
    assert option.token_index % ec3.W >= 200
    assert option.local_target_errors == 1


def test_balanced_assignment_has_one_pair_and_fifty_per_direction() -> None:
    pairs = list(range(ec3.STORE_EVENTS))
    options = {}
    for pair in pairs:
        for direction, (source, target) in enumerate(ec3.DIRECTIONS):
            options[(pair, direction)] = ec3.StaticOption(
                pair=pair,
                source_class=source,
                target_class=target,
                token_index=pair + direction + 1,
                local_target_errors=direction + 1,
                pose_sensitivity=0.1,
                static_priority=float(direction + 1),
            )
    assigned = ec3.assign_directions(pairs, options)
    assert len(assigned) == ec3.STORE_EVENTS
    assert len({row.pair for row in assigned}) == ec3.STORE_EVENTS
    for direction in ec3.DIRECTIONS:
        assert sum((row.source_class, row.target_class) == direction for row in assigned) == 50


def test_pair_set_preserves_current_and_g3_atlas_coverage() -> None:
    counts = np.arange(ec3.N, dtype=np.int64)
    available = np.ones((ec3.N, len(ec3.DIRECTIONS)), dtype=bool)
    g3_top64 = list(range(64))
    controls = list(range(100, 124))
    pairs, result = ec3.choose_pair_set(counts, g3_top64, controls, available)
    assert len(pairs) == ec3.STORE_EVENTS
    assert set(g3_top64).issubset(pairs)
    assert set(controls).issubset(pairs)
    assert result["current_t4_top64_covered"] == 64
    assert result["g3_top64_covered"] == 64


def test_net_s_pricing_is_the_chartered_joint_marginal() -> None:
    result = ec3.net_s_price(projected_flips=10.0, predicted_delta_d_pose=2e-9)
    expected = -100.0 * 10.0 / ec3.PIXELS + 603.0 * 2e-9
    assert result["predicted_joint_net_delta_s"] == expected


def test_pose_knn_uses_optimistic_q25_of_exact_t4_costs() -> None:
    rows = []
    for index in range(20):
        rows.append(
            {
                "proposal_id": f"p{index:02d}",
                "js4_receiver_delta_proxy": 1e-9 * (index + 1),
                "js4_site_sensitivity_mean": 1e-5 * (index + 1),
                "seed_y": index,
                "seed_x": index,
                "source_class": 0,
                "target_class": 1,
                "nonnegative_exact_pose_cost": float(index) * 1e-9,
            }
        )
    result = ec3.predict_pose_cost(
        rows,
        proxy=1e-9,
        sensitivity=1e-5,
        y=0,
        x=0,
        source_class=0,
        target_class=1,
    )
    assert result["neighbor_count"] == 16
    assert result["predicted_delta_d_pose_global_n600"] >= 0.0
    assert result["predicted_delta_d_pose_global_n600"] <= result["neighbor_pose_q50"]


def test_ec3_wire_is_unchanged_ec1_singleton() -> None:
    indices = np.asarray([12345], dtype=np.int64)
    payload = ec1.proposal_payload(517, 4, 0, indices, ec1.EVENT_TYPE["boundary_offset"])
    assert ec1.decode_proposal(payload)[:4] == (
        517,
        4,
        0,
        ec1.EVENT_TYPE["boundary_offset"],
    )
    assert np.array_equal(ec1.decode_proposal(payload)[4], indices)


def test_resume_row_restores_receipt_pointer(tmp_path: Path) -> None:
    event_store = tmp_path / "store"
    root = event_store / "proposals/p0"
    payload_record = ec3.atomic_bytes(root / "event.ec1p", b"real-payload")
    receipt_path = root / "proposal.json"
    ec3.atomic_json(
        receipt_path,
        {
            "proposal_id": "p0",
            "consumer_payloads": {"event.ec1p": payload_record},
        },
    )
    resumed = ec3.load_retained_proposal(receipt_path, event_store)
    assert resumed is not None
    assert resumed["proposal_receipt"] == ec3.file_record(receipt_path)


def test_ec3_source_has_no_scorer_or_dispatch_call() -> None:
    source = Path(ec3.__file__).read_text().lower()
    forbidden = ("score_seg(", "score_pose(", "upstream.evaluate(", ".spawn(", ".to(\"mps\")")
    assert all(value not in source for value in forbidden)


def test_ec3_passes_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=ec3.REPO,
        strict=False,
        roots=("experiments/ddm_ec3_t4_targeted_events.py",),
    )
    assert findings == []
