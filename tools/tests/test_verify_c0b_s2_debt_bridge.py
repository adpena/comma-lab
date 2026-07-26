from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.optimization.s2_partition_seed import (
    PartitionEvent,
    PartitionEventSeed,
    decode_partition_seed,
    encode_partition_seed,
)
from tools import verify_c0b_s2_debt_bridge as bridge


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hashed(payload: dict[str, object], field: str) -> dict[str, object]:
    result = dict(payload)
    result[field] = _sha(_canonical(result))
    return result


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    pair_count = 4
    stage_pairs = 2
    height, width = 2, 3
    per_pair = [
        [[0, 1, 0, 2]],
        [],
        [[1, 2, 3, 4]],
        [[0, 0, 1, 0]],
    ]
    pose = [0.01, 0.02, 0.03, 0.04]
    rows: list[dict[str, object]] = []
    for pair_id, events in enumerate(per_pair):
        rows.append(
            {
                "pair_id": pair_id,
                "d_seg": len(events) / (height * width),
                "d_pose": pose[pair_id],
                "seg_mismatched_pixels": len(events),
                "seg_events": events,
                "seg_events_sha256": _sha(_canonical(events)),
            }
        )
    events = [[pair_id, *event] for pair_id, local in enumerate(per_pair) for event in local]
    counts = {class_id: sum(event[3] == class_id for event in events) for class_id in range(5)}
    aggregate = {
        "pair_count": pair_count,
        "mean_d_seg": len(events) / (pair_count * height * width),
        "mean_d_pose": sum(pose) / pair_count,
        "seg_event_count": len(events),
        "seg_event_stream_sha256": _sha(_canonical(events)),
        "pair_rows_sha256": _sha(_canonical(rows)),
        "per_class": {
            name: {"class_id": class_id, "errors": counts[class_id], "sites": 1, "d_seg": counts[class_id]}
            for class_id, name in enumerate(bridge.SEMANTIC_NAMES)
        },
    }
    config_sha = "c" * 64
    debt_stages = tmp_path / "debt-stages"
    debt_custody: list[dict[str, object]] = []
    for start in range(0, pair_count, stage_pairs):
        end = min(start + stage_pairs, pair_count)
        stage = _hashed(
            {
                "schema": "tac.coupled_witness_raw_debt_stage.v2",
                "config_sha256": config_sha,
                "pair_start": start,
                "pair_end_exclusive": end,
                "rows": rows[start:end],
                "stage_complete": True,
            },
            "stage_sha256",
        )
        path = debt_stages / f"pairs-{start:04d}-{end - 1:04d}.json"
        _write(path, stage)
        debt_custody.append(
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": bridge._sha256_file(path),
                "stage_sha256": stage["stage_sha256"],
                "pair_start": start,
                "pair_end_exclusive": end,
            }
        )
    candidate_raw = {"path": "/fixture/candidate.raw", "bytes": 100, "sha256": "a" * 64}
    target_raw = {"path": "/fixture/target.raw", "bytes": 100, "sha256": "b" * 64}
    debt = _hashed(
        {
            "schema": bridge.DEBT_SCHEMA,
            "config_sha256": config_sha,
            "config": {
                "pair_count": pair_count,
                "stage_pairs": stage_pairs,
                "scorer_batch_pairs": stage_pairs,
                "scorer_hw": [height, width],
                "raw": candidate_raw,
                "target_raw": target_raw,
            },
            "aggregate": aggregate,
            "pairs": rows,
            "stages": debt_custody,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
        "receipt_sha256",
    )
    debt_path = tmp_path / "debt.json"
    _write(debt_path, debt)

    r2b_stages = tmp_path / "r2b-stages"
    for start in range(0, pair_count, stage_pairs):
        end = min(start + stage_pairs, pair_count)
        flips = [[pair_id, *event, 0.125] for pair_id in range(start, end) for event in per_pair[pair_id]]
        _write(
            r2b_stages / f"batch-{start:04d}.json",
            {
                "schema": bridge.R2B_STAGE_SCHEMA,
                "pair_start": start,
                "pair_stop": end,
                "flips": flips,
                "flip_count": len(flips),
                "cache_label_mismatches": 0,
                "pose_squared_error": [[pose[pair_id]] * 6 for pair_id in range(start, end)],
            },
        )
    r2b_files = sorted(r2b_stages.glob("batch-*.json"))
    tree_sha = bridge._tree_hash(r2b_files, r2b_stages)
    r2b_receipt = tmp_path / "r2b.json"
    _write(
        r2b_receipt,
        {
            "schema": bridge.R2B_SCHEMA,
            "score_claim": False,
            "hard_oracle_batch_size": stage_pairs,
            "baseline_raw": candidate_raw,
            "target_raw": target_raw,
            "baseline": {"flip_count": len(events)},
        },
    )

    seed = PartitionEventSeed(
        n_pairs=pair_count,
        height=height,
        width=width,
        semantic_class_ids=(0, 1, 2, 3, 4),
        events=tuple(PartitionEvent(*event) for event in events),
    )
    packet = encode_partition_seed(seed)
    packet_path = tmp_path / "s2.bin"
    packet_path.write_bytes(packet)
    s2_receipt = tmp_path / "s2.json"
    _write(
        s2_receipt,
        {
            "schema": bridge.S2_SCHEMA,
            "score_claim": False,
            "n_pairs": pair_count,
            "finite_packet": {
                "packet_bytes": len(packet),
                "counted_seed_bytes": len(packet),
                "packet_sha256": _sha(packet),
                "event_count": len(events),
                "parse_back_event_identity": True,
            },
            "inventory_custody": {
                "stage_count": len(r2b_files),
                "stage_tree_sha256": tree_sha,
                "cache_label_mismatches": 0,
            },
        },
    )
    return {
        "debt": debt_path,
        "r2b_receipt": r2b_receipt,
        "r2b_stages": r2b_stages,
        "s2_packet": packet_path,
        "s2_receipt": s2_receipt,
    }


def test_bridge_requires_three_way_exact_event_identity(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    result = bridge.verify_bridge(
        debt_path=paths["debt"],
        r2b_receipt_path=paths["r2b_receipt"],
        r2b_stage_dir=paths["r2b_stages"],
        s2_packet_path=paths["s2_packet"],
        s2_receipt_path=paths["s2_receipt"],
    )
    assert result["identity"]["event_count"] == 3
    assert result["identity"]["debt_equals_r2b"] is True
    assert result["identity"]["debt_equals_s2"] is True
    assert result["identity"]["pose_rows_equal_r2b"] is True
    assert result["authority_geometry"] == {
        "pair_count": 4,
        "scorer_hw": [2, 3],
        "total_seg_sites": 24,
        "mean_d_seg": 3 / 24,
        "baseline_mean_d_pose": 0.025,
        "s2_geometry_exact": True,
    }
    assert result["score_claim"] is False
    assert len(result["receipt_sha256"]) == 64


def test_bridge_refuses_semantically_wrong_debt_schema(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    debt = json.loads(paths["debt"].read_text(encoding="utf-8"))
    debt["schema"] = "tac.coupled_witness_raw_debt.v1"
    debt.pop("receipt_sha256")
    debt = _hashed(debt, "receipt_sha256")
    _write(paths["debt"], debt)
    with pytest.raises(bridge.BridgeError, match="superseding batch-16"):
        bridge.verify_bridge(
            debt_path=paths["debt"],
            r2b_receipt_path=paths["r2b_receipt"],
            r2b_stage_dir=paths["r2b_stages"],
            s2_packet_path=paths["s2_packet"],
            s2_receipt_path=paths["s2_receipt"],
        )


def test_bridge_refuses_r2b_stage_mutation_even_if_event_count_is_unchanged(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    stage = paths["r2b_stages"] / "batch-0000.json"
    payload = json.loads(stage.read_text(encoding="utf-8"))
    payload["flips"][0][4] = 1
    _write(stage, payload)
    with pytest.raises(bridge.BridgeError, match="S2 receipt does not bind"):
        bridge.verify_bridge(
            debt_path=paths["debt"],
            r2b_receipt_path=paths["r2b_receipt"],
            r2b_stage_dir=paths["r2b_stages"],
            s2_packet_path=paths["s2_packet"],
            s2_receipt_path=paths["s2_receipt"],
        )


def test_bridge_refuses_s2_geometry_drift_with_identical_events(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    seed = decode_partition_seed(paths["s2_packet"].read_bytes())
    drifted = PartitionEventSeed(
        n_pairs=seed.n_pairs,
        height=seed.height,
        width=seed.width + 1,
        semantic_class_ids=seed.semantic_class_ids,
        events=seed.events,
    )
    packet = encode_partition_seed(drifted)
    paths["s2_packet"].write_bytes(packet)
    receipt = json.loads(paths["s2_receipt"].read_text(encoding="utf-8"))
    receipt["finite_packet"].update(
        {
            "packet_bytes": len(packet),
            "counted_seed_bytes": len(packet),
            "packet_sha256": _sha(packet),
        }
    )
    _write(paths["s2_receipt"], receipt)
    with pytest.raises(bridge.BridgeError, match="population geometry"):
        bridge.verify_bridge(
            debt_path=paths["debt"],
            r2b_receipt_path=paths["r2b_receipt"],
            r2b_stage_dir=paths["r2b_stages"],
            s2_packet_path=paths["s2_packet"],
            s2_receipt_path=paths["s2_receipt"],
        )


def test_bridge_uses_one_s2_packet_snapshot_for_identity_and_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _fixture(tmp_path)
    correct_packet = paths["s2_packet"].read_bytes()
    seed = decode_partition_seed(correct_packet)
    drifted_packet = encode_partition_seed(
        PartitionEventSeed(
            n_pairs=seed.n_pairs,
            height=seed.height,
            width=seed.width + 1,
            semantic_class_ids=seed.semantic_class_ids,
            events=seed.events,
        )
    )
    receipt = json.loads(paths["s2_receipt"].read_text(encoding="utf-8"))
    receipt["finite_packet"].update(
        {
            "packet_bytes": len(drifted_packet),
            "counted_seed_bytes": len(drifted_packet),
            "packet_sha256": _sha(drifted_packet),
        }
    )
    _write(paths["s2_receipt"], receipt)

    original_read_bytes = Path.read_bytes
    packet_calls = 0

    def swapping_read_bytes(path: Path) -> bytes:
        nonlocal packet_calls
        if path == paths["s2_packet"]:
            packet_calls += 1
            return drifted_packet if packet_calls == 1 else correct_packet
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swapping_read_bytes)
    with pytest.raises(bridge.BridgeError, match="population geometry"):
        bridge.verify_bridge(
            debt_path=paths["debt"],
            r2b_receipt_path=paths["r2b_receipt"],
            r2b_stage_dir=paths["r2b_stages"],
            s2_packet_path=paths["s2_packet"],
            s2_receipt_path=paths["s2_receipt"],
        )
    assert packet_calls == 1
