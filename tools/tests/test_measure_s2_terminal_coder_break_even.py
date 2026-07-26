from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

from tac.optimization.s2_partition_seed import (
    PartitionEvent,
    PartitionEventSeed,
    encode_partition_seed,
)
from tools import measure_s2_terminal_coder_break_even as terminal


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_bridge(path: Path, packet: bytes, seed: PartitionEventSeed) -> None:
    event_stream = [
        [event.pair, event.row, event.col, event.target_class, event.baseline_class]
        for event in seed.events
    ]
    payload: dict[str, object] = {
        "schema": terminal.BRIDGE_SCHEMA,
        "s2": {"packet_bytes": len(packet), "packet_sha256": hashlib.sha256(packet).hexdigest()},
        "identity": {
            "event_count": len(seed.events),
            "event_stream_sha256": hashlib.sha256(_canonical(event_stream)).hexdigest(),
            "debt_equals_r2b": True,
            "debt_equals_s2": True,
            "r2b_equals_s2": True,
            "strict_site_order": True,
            "unique_sites": True,
            "pose_rows_equal_r2b": True,
        },
        "authority_geometry": {
            "pair_count": seed.n_pairs,
            "scorer_hw": [seed.height, seed.width],
            "total_seg_sites": seed.n_pairs * seed.height * seed.width,
            "mean_d_seg": len(seed.events) / (seed.n_pairs * seed.height * seed.width),
            "baseline_mean_d_pose": 0.01,
            "s2_geometry_exact": True,
        },
        "verdict": "EXACT_C1_LIVE_TARGET_DEBT_EQUALS_R2B_INVENTORY_EQUALS_S2_PACKET",
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, PartitionEventSeed, bytes]:
    seed = PartitionEventSeed(
        n_pairs=3,
        height=5,
        width=7,
        semantic_class_ids=(0, 1, 2, 3, 4),
        events=(
            PartitionEvent(0, 0, 0, 1, 0),
            PartitionEvent(0, 4, 6, 2, 1),
            PartitionEvent(1, 2, 3, 4, 0),
            PartitionEvent(2, 1, 5, 3, 2),
        ),
    )
    packet = encode_partition_seed(seed)
    packet_path = tmp_path / "s2.bin"
    packet_path.write_bytes(packet)
    bridge_path = tmp_path / "bridge.json"
    _write_bridge(bridge_path, packet, seed)
    return packet_path, bridge_path, seed, packet


def test_strict_break_even_cap_obeys_seg_only_slice_of_joint_objective() -> None:
    cap = terminal._strict_break_even_byte_cap(event_count=17_926, pair_count=600, height=384, width=512)
    saved = terminal.SEG_WEIGHT * 17_926 / (600 * 384 * 512)
    assert cap == 22_821
    assert terminal.RATE_WEIGHT * cap / terminal.DENOMINATOR_BYTES < saved
    assert terminal.RATE_WEIGHT * (cap + 1) / terminal.DENOMINATOR_BYTES >= saved


def test_strict_break_even_cap_rejects_exact_rational_equality() -> None:
    # Regression for a float-rounding counterexample: candidate+1 lies exactly
    # on the rational break-even boundary and therefore must not be admitted.
    events = 364_274_110_928_223
    total_sites = 15_018_196_050_545_868
    cap = terminal._strict_break_even_byte_cap(
        event_count=events,
        pair_count=1,
        height=1,
        width=total_sites,
    )
    assert cap == 3_642_740
    assert 4 * terminal.DENOMINATOR_BYTES * events == (cap + 1) * total_sites


def test_extract_raw_event_stream_is_bound_to_strict_parseback(tmp_path: Path) -> None:
    packet_path, _bridge_path, seed, packet = _fixture(tmp_path)
    raw, header = terminal._extract_validated_raw_event_bytes(packet)
    assert header["event_count"] == len(seed.events)
    assert header["raw_event_bytes"] == len(raw)
    assert header["raw_event_sha256"] == hashlib.sha256(raw).hexdigest()

    mutated = bytearray(packet_path.read_bytes())
    mutated[-1] ^= 1
    with pytest.raises(Exception, match="CRC mismatch"):
        terminal._extract_validated_raw_event_bytes(bytes(mutated))


def test_measure_binds_bridge_and_reports_only_research_economics(tmp_path: Path) -> None:
    packet_path, bridge_path, seed, packet = _fixture(tmp_path)
    result = terminal.measure(packet_path=packet_path, bridge_path=bridge_path)
    expected_saved = terminal.SEG_WEIGHT * len(seed.events) / (seed.n_pairs * seed.height * seed.width)
    assert math.isclose(result["exact_economics"]["seg_score_saved_if_all_events_realized"], expected_saved)
    assert result["exact_economics"]["baseline_mean_d_pose"] == 0.01
    assert result["verdict"] == "POSE_CONDITIONAL_TERMINAL_CODER_ECONOMICS_RECEIVER_MEASUREMENT_REQUIRED"
    assert result["best_payload_only"]["strict_joint_improvement_requires_d_pose_after_below"] is not None
    assert result["inputs"]["packet"]["sha256"] == hashlib.sha256(packet).hexdigest()
    assert result["best_payload_only"]["headers_and_container_bytes_assumed"] == 0
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["pointer_moved"] is False
    receipt = result.pop("receipt_sha256")
    assert receipt == hashlib.sha256(_canonical(result)).hexdigest()


def test_measure_refuses_bridge_or_packet_mutation(tmp_path: Path) -> None:
    packet_path, bridge_path, _seed, _packet = _fixture(tmp_path)
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["identity"]["event_count"] += 1
    bridge.pop("receipt_sha256")
    bridge["receipt_sha256"] = hashlib.sha256(_canonical(bridge)).hexdigest()
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    with pytest.raises(terminal.CoderBreakEvenError, match="does not bind"):
        terminal.measure(packet_path=packet_path, bridge_path=bridge_path)


def test_measure_refuses_bridge_that_denies_exact_identity(tmp_path: Path) -> None:
    packet_path, bridge_path, _seed, _packet = _fixture(tmp_path)
    bridge = json.loads(bridge_path.read_text(encoding="utf-8"))
    bridge["identity"]["debt_equals_s2"] = False
    bridge.pop("receipt_sha256")
    bridge["receipt_sha256"] = hashlib.sha256(_canonical(bridge)).hexdigest()
    bridge_path.write_text(json.dumps(bridge), encoding="utf-8")
    with pytest.raises(terminal.CoderBreakEvenError, match="does not bind"):
        terminal.measure(packet_path=packet_path, bridge_path=bridge_path)
