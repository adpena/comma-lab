# SPDX-License-Identifier: MIT
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.s2_partition_seed import PartitionEvent, PartitionEventSeed
from tools import measure_c0b_wseg_c1_baseline_join as join


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = np.array([[[0, 1, 2], [3, 4, 0]]], dtype=np.uint8)
    cached = target.copy()
    wseg = target.copy()
    owned = np.array([[[True, True, False], [True, False, True]]], dtype=bool)
    return target, cached, wseg, owned


def _events() -> dict[int, tuple[PartitionEvent, ...]]:
    return {0: (PartitionEvent(0, 0, 1, 1, 0),)}


def _patch_small(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(join, "SCORER_HW", (2, 3))


def test_canonical_is_key_order_stable() -> None:
    assert join._canonical({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_rejects_nan() -> None:
    with pytest.raises(join.BaselineJoinError, match="canonical-JSON"):
        join._canonical({"x": float("nan")})


def test_body_hash_round_trip() -> None:
    receipt = join._with_hash({"a": 1}, "receipt_sha256")
    join._validate_hash(receipt, "receipt_sha256")


def test_body_hash_rejects_tamper() -> None:
    receipt = join._with_hash({"a": 1}, "receipt_sha256")
    receipt["a"] = 2
    with pytest.raises(join.BaselineJoinError, match="differs"):
        join._validate_hash(receipt, "receipt_sha256")


def test_with_hash_rejects_existing_field() -> None:
    with pytest.raises(join.BaselineJoinError, match="already exists"):
        join._with_hash({"h": "x"}, "h")


def test_write_once_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    payload = {"a": 1}
    join._write_once(path, payload)
    first = path.read_bytes()
    join._write_once(path, payload)
    assert path.read_bytes() == first


def test_write_once_rejects_different_body(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    join._write_once(path, {"a": 1})
    with pytest.raises(join.BaselineJoinError, match="write-once"):
        join._write_once(path, {"a": 2})


def test_exact_file_accepts_bound_bytes(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    row = {"bytes": 3, "sha256": join._sha256_bytes(b"abc")}
    assert join._exact_file(path, row, "x")["path"] == str(path.resolve())


def test_exact_file_rejects_digest_drift(tmp_path: Path) -> None:
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    with pytest.raises(join.BaselineJoinError, match="custody differs"):
        join._exact_file(path, {"bytes": 3, "sha256": "0" * 64}, "x")


def test_events_grouped_by_pair() -> None:
    seed = PartitionEventSeed(
        n_pairs=2,
        height=2,
        width=3,
        semantic_class_ids=(0, 1, 2, 3, 4),
        events=(PartitionEvent(0, 0, 0, 1, 0), PartitionEvent(1, 1, 2, 4, 3)),
    )
    grouped = join._events_by_pair(seed)
    assert tuple(grouped) == (0, 1)
    assert grouped[1][0].target_class == 4


def test_reconstruct_baseline_replaces_only_seeded_site(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, *_ = _arrays()
    baseline, seeded, rows = join._reconstruct_baseline(target, pair_start=0, events=_events())
    assert baseline[0, 0, 1] == 0
    assert np.array_equal(baseline[~seeded], target[~seeded])
    assert rows == [[0, 0, 1, 1, 0]]


def test_reconstruct_baseline_rejects_target_class_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, *_ = _arrays()
    target[0, 0, 1] = 2
    with pytest.raises(join.BaselineJoinError, match="target class"):
        join._reconstruct_baseline(target, pair_start=0, events=_events())


def test_reconstruct_baseline_rejects_repeated_site(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, *_ = _arrays()
    repeated = {0: (_events()[0][0], _events()[0][0])}
    with pytest.raises(join.BaselineJoinError, match="repeats a site"):
        join._reconstruct_baseline(target, pair_start=0, events=repeated)


def test_matrix_rows_always_exposes_25_interfaces() -> None:
    rows = join._matrix_rows(join._zero_matrix(), left_name="baseline", right_name="target", count_name="sites")
    assert len(rows) == 25
    assert rows[0]["baseline_class"] == 0
    assert rows[-1]["target_class"] == 4


def test_measure_arrays_exact_initializer(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, cached, _wseg, owned = _arrays()
    wseg = target.copy()
    wseg[0, 0, 1] = 0
    row = join._measure_arrays(
        pair_start=0,
        target=target,
        cached_target=cached,
        wseg=wseg,
        owned=owned,
        events=_events(),
    )
    assert row["s2_events"] == 1
    assert row["seeded_expected_baseline_mismatches"] == 0
    assert row["non_event_residual"] == 0
    assert row["total_wseg_to_c1_baseline_mismatches"] == 0
    assert len(row["c1_to_target_interface_25"]) == 25


def test_measure_arrays_partitions_seeded_and_non_event_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, cached, wseg, owned = _arrays()
    wseg[0, 1, 0] = 4
    row = join._measure_arrays(
        pair_start=0,
        target=target,
        cached_target=cached,
        wseg=wseg,
        owned=owned,
        events=_events(),
    )
    assert row["seeded_expected_baseline_mismatches"] == 1
    assert row["non_event_residual"] == 1
    assert row["total_wseg_to_c1_baseline_mismatches"] == 2


def test_measure_arrays_reports_cache_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, cached, wseg, owned = _arrays()
    cached[0, 1, 2] = 1
    row = join._measure_arrays(
        pair_start=0,
        target=target,
        cached_target=cached,
        wseg=wseg,
        owned=owned,
        events=_events(),
    )
    assert row["target_cache_label_mismatches"] == 1
    assert row["pairs"][0]["target_cache_label_mismatches"] == 1


def test_measure_arrays_rejects_nonboolean_ownership(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, cached, wseg, owned = _arrays()
    with pytest.raises(join.BaselineJoinError, match="ownership"):
        join._measure_arrays(
            pair_start=0,
            target=target,
            cached_target=cached,
            wseg=wseg,
            owned=owned.astype(np.uint8),
            events=_events(),
        )


def test_measure_arrays_rejects_class_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_small(monkeypatch)
    target, cached, wseg, owned = _arrays()
    wseg[0, 0, 0] = 9
    with pytest.raises(join.BaselineJoinError, match="alphabet"):
        join._measure_arrays(
            pair_start=0,
            target=target,
            cached_target=cached,
            wseg=wseg,
            owned=owned,
            events=_events(),
        )


def _small_stage(monkeypatch: pytest.MonkeyPatch, pair: int, *, mismatch: bool = False) -> dict[str, object]:
    monkeypatch.setattr(join, "SCORER_HW", (2, 2))
    target = np.full((1, 2, 2), pair, dtype=np.uint8)
    wseg = target.copy()
    if mismatch:
        wseg[0, 0, 0] = (pair + 1) % 5
    measured = join._measure_arrays(
        pair_start=pair,
        target=target,
        cached_target=target,
        wseg=wseg,
        owned=np.ones_like(target, dtype=bool),
        events={},
    )
    return join._with_hash(
        {"schema": join.STAGE_SCHEMA, "config_sha256": "c" * 64, **measured, "stage_complete": True},
        "stage_sha256",
    )


def test_aggregate_reconciles_exact_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(join, "PAIR_COUNT", 2)
    monkeypatch.setattr(join, "STAGE_PAIRS", 1)
    stages = [_small_stage(monkeypatch, 0), _small_stage(monkeypatch, 1, mismatch=True)]
    result = join._aggregate(stages, config_sha256="c" * 64, expected_events=0)
    assert result["sites"] == 8
    assert result["total_wseg_to_c1_baseline_mismatches"] == 1
    assert len(result["pairs"]) == 2
    assert len(result["c1_to_wseg_confusion_25"]) == 25


def test_aggregate_rejects_seed_non_event_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(join, "PAIR_COUNT", 1)
    monkeypatch.setattr(join, "STAGE_PAIRS", 1)
    stage = _small_stage(monkeypatch, 0)
    stage["total_wseg_to_c1_baseline_mismatches"] = 1
    stage["stage_sha256"] = join._sha256_bytes(join._canonical({k: v for k, v in stage.items() if k != "stage_sha256"}))
    with pytest.raises(join.BaselineJoinError, match="does not reconcile"):
        join._aggregate([stage], config_sha256="c" * 64, expected_events=0)


def test_load_prefix_requires_explicit_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(join, "PAIR_COUNT", 1)
    monkeypatch.setattr(join, "STAGE_PAIRS", 1)
    stage = _small_stage(monkeypatch, 0)
    path = join._stage_path(tmp_path, 0, 1)
    join._write_once(path, stage)
    with pytest.raises(join.BaselineJoinError, match="pass --resume"):
        join._load_prefix(tmp_path, config_sha256="c" * 64, resume=False)


def test_load_prefix_accepts_valid_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(join, "PAIR_COUNT", 1)
    monkeypatch.setattr(join, "STAGE_PAIRS", 1)
    stage = _small_stage(monkeypatch, 0)
    join._write_once(join._stage_path(tmp_path, 0, 1), stage)
    assert join._load_prefix(tmp_path, config_sha256="c" * 64, resume=True)[0]["pair_start"] == 0


def test_target_argmax_rejects_wrong_camera_shape() -> None:
    with pytest.raises(join.BaselineJoinError, match="geometry/dtype"):
        join._target_argmax(object(), object(), np.zeros((1, 2, 3), dtype=np.uint8))


def test_parser_defaults_to_batch_measurement_paths() -> None:
    args = join._parser().parse_args([])
    assert isinstance(args, argparse.Namespace)
    assert args.bridge == join.DEFAULT_BRIDGE
    assert args.ws2_receipt == join.DEFAULT_WS2_RECEIPT
    assert args.cpu_threads == 4


def test_final_receipt_flags_are_nonpromotable_constants() -> None:
    source = Path(join.__file__).read_text(encoding="utf-8")
    assert '"score_claim": False' in source
    assert '"promotion_eligible": False' in source
    assert '"target_tables_forbidden_in_candidate_payload": True' in source


def test_stage_json_is_canonical_body_hashable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(join, "PAIR_COUNT", 1)
    monkeypatch.setattr(join, "STAGE_PAIRS", 1)
    stage = _small_stage(monkeypatch, 0)
    round_trip = json.loads(json.dumps(stage, allow_nan=False))
    join._validate_hash(round_trip, "stage_sha256")


def test_input_end_barrier_rehashes_every_bound_file(tmp_path: Path) -> None:
    paths: dict[str, Path] = {}
    for name in (
        "bridge",
        "debt",
        "packet",
        "target_raw",
        "cache",
        "wseg_receipt",
        "wseg_archive",
        "tool",
        "wseg_receiver_module",
        "scorer_adapter_module",
        "modules_py",
        "frame_utils_py",
        "segnet_weights",
        "posenet_weights",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path

    def row(name: str) -> dict[str, object]:
        payload = paths[name].read_bytes()
        return {"path": str(paths[name]), "bytes": len(payload), "sha256": join._sha256_bytes(payload)}

    config = {
        "bridge_custody": {
            "bridge": row("bridge"),
            "debt": row("debt"),
            "packet": row("packet"),
            "target_raw": row("target_raw"),
            "cache": row("cache"),
            "scorer_files": {
                key: row(key) for key in ("modules_py", "frame_utils_py", "segnet_weights", "posenet_weights")
            },
        },
        "wseg_custody": {"receipt": row("wseg_receipt"), "archive": row("wseg_archive")},
        "implementation_custody": {
            "tool": row("tool"),
            "wseg_receiver_module": row("wseg_receiver_module"),
            "scorer_adapter_module": row("scorer_adapter_module"),
        },
    }
    barrier = join._input_end_barrier(config)
    assert barrier["all_inputs_rehashed_after_last_stage"] is True
    assert len(barrier["files"]) == 14
    paths["packet"].write_bytes(b"drifted")
    with pytest.raises(join.BaselineJoinError, match="custody differs"):
        join._input_end_barrier(config)


def test_join_verdict_is_exactly_rederived() -> None:
    aggregate = {
        "sites": 100,
        "total_wseg_to_c1_baseline_mismatches": 2,
        "seeded_expected_baseline_mismatches": 0,
        "non_event_residual": 2,
        "target_cache_label_mismatches": 3,
    }
    verdict, row = join._join_verdict(aggregate, expected_events=7)
    assert verdict == "WSEG_MATCHES_S2_BASELINE_AT_SEEDED_SITES_BUT_DIFFERS_OUTSIDE_THE_EVENT_SET"
    assert row["seeded_site_expected_baseline_matches"] == 7
    assert row["total_wseg_to_c1_baseline_mismatch_fraction"] == 0.02
