from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import measure_v10_free_predictor_floor as scorer
from tools import score_coupled_witness_raw_debt as debt


def _contest_payload(
    *,
    raw_sha256: str = "a" * 64,
    raw_bytes: int = 20,
    archive_sha256: str = "b" * 64,
    archive_bytes: int = 10,
    archive_path: str = "/remote/archive.zip",
) -> dict[str, object]:
    files = [
        {
            "video_name": "0.mkv",
            "relative_path": "0.raw",
            "exists": True,
            "sha256": raw_sha256,
            "bytes": raw_bytes,
        }
    ]
    aggregate = debt._sha256_bytes(
        debt._canonical(
            {
                "files": [
                    {"relative_path": "0.raw", "bytes": raw_bytes, "sha256": raw_sha256},
                ]
            }
        )
    )
    seg = 0.1
    pose = 0.2
    score = 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / debt.ORIGINAL_UNCOMPRESSED_SIZE_BYTES
    return {
        "schema_version": debt.CONTEST_SCHEMA_VERSION,
        "n_samples": debt.DEFAULT_PAIR_COUNT,
        "score_axis": "contest_cpu",
        "evidence_grade": "contest-CPU",
        "lane_tag": "[contest-CPU]",
        "score_claim": True,
        "score_claim_valid": True,
        "score_claim_eligible": True,
        "cpu_leaderboard_reproduction_eligible": True,
        "evidence_semantics": "public_leaderboard_cpu_reproduction",
        "allowed_uses": ["cpu_axis_score_claim"],
        "archive_size_bytes": archive_bytes,
        "original_uncompressed_size_bytes": debt.ORIGINAL_UNCOMPRESSED_SIZE_BYTES,
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "score_recomputed_from_components": score,
        "canonical_score": score,
        "canonical_score_source": "score_recomputed_from_components",
        "provenance": {
            "schema_version": debt.CONTEST_SCHEMA_VERSION,
            "tool": "experiments/contest_auth_eval.py",
            "archive_path": archive_path,
            "archive_sha256": archive_sha256,
            "archive_size_bytes": archive_bytes,
            "device": "cpu",
            "platform_system": "Linux",
            "cuda_available": False,
            "mps_available": False,
            "inflated_output_manifest": {
                "payload": {
                    "schema": debt.CONTEST_MANIFEST_SCHEMA,
                    "raw_file_count": 1,
                    "total_bytes": raw_bytes,
                    "files": files,
                    "aggregate_sha256": aggregate,
                }
            },
        },
    }


def _target_payload(*, raw_sha256: str = "a" * 64, raw_bytes: int = 20) -> dict[str, object]:
    return {
        "schema": debt.TARGET_SCHEMA,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "source_custody": {
            "pair_count": debt.DEFAULT_PAIR_COUNT,
            "frame_count": 1200,
            "gt_cache_sha256": "c" * 64,
            "gt_cache_bytes": 123,
        },
        "candidate": {
            "inflated_raw_sha256": raw_sha256,
            "inflated_raw_bytes": raw_bytes,
            "archive_sha256": "b" * 64,
            "archive_bytes": 100,
            "d_seg": 0.0,
            "d_pose": 0.0,
        },
    }


def _class_row(errors: int = 0) -> dict[str, dict[str, object]]:
    return {
        name: {
            "class_id": index,
            "errors": errors if index == 0 else 0,
            "sites": 2 if index == 0 else 0,
            "d_seg": errors / 2 if index == 0 else None,
        }
        for index, name in enumerate(scorer.CLASS_ORDER)
    }


def test_stage_hash_and_prefix_reject_mutation(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages"
    stage = debt._stage_payload(
        config_sha256="a" * 64,
        start=0,
        end=1,
        rows=[{"pair_id": 0, "d_seg": 0.0, "d_pose": 0.0, "per_class": _class_row()}],
    )
    debt._write_once(debt._stage_path(stage_dir, 0, 1), stage)
    rows, custody = debt._load_prefix(
        stage_dir=stage_dir,
        pair_count=1,
        stage_pairs=1,
        config_sha256="a" * 64,
    )
    assert rows[0]["pair_id"] == 0
    assert custody[0]["stage_sha256"] == stage["stage_sha256"]
    mutated = json.loads((stage_dir / "pairs-0000-0000.json").read_text())
    mutated["rows"][0]["d_pose"] = 1.0
    (stage_dir / "pairs-0000-0000.json").write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match="differs from receipt body"):
        debt._load_prefix(
            stage_dir=stage_dir,
            pair_count=1,
            stage_pairs=1,
            config_sha256="a" * 64,
        )


def test_stage_schema_refuses_noninteger_boundary_even_with_valid_hash(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages"
    path = debt._stage_path(stage_dir, 0, 1)
    debt._write_once(
        path,
        debt._with_hash(
            {
                "schema": debt.STAGE_SCHEMA,
                "config_sha256": "a" * 64,
                "pair_start": "0",
                "pair_end_exclusive": 1,
                "rows": [{"pair_id": 0}],
                "stage_complete": True,
            },
            "stage_sha256",
        ),
    )
    with pytest.raises(debt.RawDebtError, match="boundaries are malformed"):
        debt._load_prefix(
            stage_dir=stage_dir,
            pair_count=1,
            stage_pairs=1,
            config_sha256="a" * 64,
        )


def test_aggregate_preserves_pair_and_class_debt() -> None:
    rows = [
        {
            "pair_id": 0,
            "d_seg": 0.25,
            "d_pose": 0.01,
            "seg_mismatched_pixels": 1,
            "seg_events": [[0, 1, 0, 1]],
            "per_class": _class_row(1),
            "cache_label_mismatches": 1,
            "cache_pose_max_abs_difference": 0.125,
        },
        {
            "pair_id": 1,
            "d_seg": 0.0,
            "d_pose": 0.03,
            "seg_mismatched_pixels": 0,
            "seg_events": [],
            "per_class": _class_row(0),
            "cache_label_mismatches": 2,
            "cache_pose_max_abs_difference": 0.25,
        },
    ]
    result = debt._aggregate_rows(rows, pair_count=2)
    assert result["mean_d_seg"] == 0.125
    assert result["mean_d_pose"] == 0.02
    assert result["per_class"]["Road"]["errors"] == 1
    assert result["per_class"]["Road"]["sites"] == 4
    assert result["seg_event_count"] == 1
    assert result["cache_label_mismatches"] == 3
    assert result["cache_pose_max_abs_difference"] == 0.25
    assert len(result["pair_rows_sha256"]) == 64


def test_contest_reference_must_bind_exact_raw(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"real archive")
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            _contest_payload(
                archive_path=str(archive),
                archive_sha256=debt._sha256_file(archive),
                archive_bytes=archive.stat().st_size,
            )
        ),
        encoding="utf-8",
    )
    result = debt._contest_reference(path, raw_sha256="a" * 64, raw_bytes=20)
    assert result["score_axis"] == "contest_cpu"
    assert result["local_archive_verified"] is True
    with pytest.raises(debt.RawDebtError, match="does not bind"):
        debt._contest_reference(path, raw_sha256="c" * 64, raw_bytes=20)


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("schema_version",), 2, "exact eligible"),
        (("n_samples",), 599, "exact eligible"),
        (("score_axis",), "contest_cuda", "exact eligible"),
        (("score_claim_valid",), False, "exact eligible"),
        (("provenance", "device"), "cuda", "exact eligible"),
        (("provenance", "platform_system"), "Darwin", "exact eligible"),
        (("provenance", "archive_size_bytes"), 11, "archive custody"),
        (("canonical_score",), 999.0, "does not recompute"),
        (
            ("provenance", "inflated_output_manifest", "payload", "aggregate_sha256"),
            "f" * 64,
            "does not bind",
        ),
    ],
)
def test_contest_reference_refuses_authority_drift(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: object,
    message: str,
) -> None:
    payload = _contest_payload()
    target: dict[str, object] = payload
    for key in keys[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[keys[-1]] = value
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match=message):
        debt._contest_reference(path, raw_sha256="a" * 64, raw_bytes=20)


def test_contest_reference_verifies_local_archive_when_present(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"real archive")
    payload = _contest_payload(
        archive_sha256=debt._sha256_file(archive),
        archive_bytes=archive.stat().st_size,
        archive_path=str(archive),
    )
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert debt._contest_reference(path, raw_sha256="a" * 64, raw_bytes=20)["local_archive_verified"] is True
    archive.write_bytes(b"drift")
    with pytest.raises(debt.RawDebtError, match="local archive bytes drifted"):
        debt._contest_reference(path, raw_sha256="a" * 64, raw_bytes=20)


def test_contest_reference_refuses_missing_explicit_local_archive(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_contest_payload()), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match="custodied local contest archive is absent"):
        debt._contest_reference(
            path,
            raw_sha256="a" * 64,
            raw_bytes=20,
            local_archive_path=tmp_path / "absent.zip",
        )


def test_contest_reference_refuses_metadata_only_authority(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(_contest_payload(archive_path="/definitely/absent/archive.zip")), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match="archive is absent"):
        debt._contest_reference(path, raw_sha256="a" * 64, raw_bytes=20)


def test_target_reference_requires_exact_zero_distortion_raw(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(json.dumps(_target_payload()), encoding="utf-8")
    result = debt._target_reference(
        path,
        raw_sha256="a" * 64,
        raw_bytes=20,
        pair_count=debt.DEFAULT_PAIR_COUNT,
    )
    assert result["d_seg"] == 0.0
    with pytest.raises(debt.RawDebtError, match="does not bind"):
        debt._target_reference(
            path,
            raw_sha256="c" * 64,
            raw_bytes=20,
            pair_count=debt.DEFAULT_PAIR_COUNT,
        )


def test_target_reference_can_be_pinned_to_exact_path_and_bytes(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(json.dumps(_target_payload()), encoding="utf-8")
    expected_sha = debt._sha256_file(path)
    result = debt._target_reference(
        path,
        raw_sha256="a" * 64,
        raw_bytes=20,
        pair_count=debt.DEFAULT_PAIR_COUNT,
        required_path=path,
        required_file_sha256=expected_sha,
    )
    assert result["sha256"] == expected_sha
    path.write_text(json.dumps({**_target_payload(), "extra": True}), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match="pinned bytes drifted"):
        debt._target_reference(
            path,
            raw_sha256="a" * 64,
            raw_bytes=20,
            pair_count=debt.DEFAULT_PAIR_COUNT,
            required_path=path,
            required_file_sha256=expected_sha,
        )


@pytest.mark.parametrize(
    ("keys", "value"),
    [
        (("schema",), "m2_live_target_selection_receipt.v2"),
        (("axis",), "[contest-CPU]"),
        (("score_claim",), True),
        (("promotion_eligible",), True),
        (("candidate", "d_seg"), 0.1),
        (("candidate", "d_pose"), "0"),
        (("candidate", "archive_sha256"), "bad"),
        (("candidate", "archive_bytes"), 0),
    ],
)
def test_target_reference_refuses_schema_semantic_or_custody_drift(
    tmp_path: Path,
    keys: tuple[str, ...],
    value: object,
) -> None:
    payload = _target_payload()
    target: dict[str, object] = payload
    for key in keys[:-1]:
        nested = target[key]
        assert isinstance(nested, dict)
        target = nested
    target[keys[-1]] = value
    path = tmp_path / "target.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match="does not bind"):
        debt._target_reference(
            path,
            raw_sha256="a" * 64,
            raw_bytes=20,
            pair_count=debt.DEFAULT_PAIR_COUNT,
        )


def _write_two_stages(stage_dir: Path, config_sha256: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    for start in range(2):
        stage = debt._stage_payload(
            config_sha256=config_sha256,
            start=start,
            end=start + 1,
            rows=[{"pair_id": start}],
        )
        debt._write_once(debt._stage_path(stage_dir, start, start + 1), stage)
    return debt._load_prefix(
        stage_dir=stage_dir,
        pair_count=2,
        stage_pairs=1,
        config_sha256=config_sha256,
    )


def test_missing_state_recovers_from_authoritative_stage_prefix(tmp_path: Path) -> None:
    config_sha = "a" * 64
    rows, stages = _write_two_stages(tmp_path / "stages", config_sha)
    state_path = tmp_path / "state.json"
    state = debt._reconcile_state(
        state_path,
        config_sha256=config_sha,
        prefix_pairs=len(rows),
        stages=stages,
    )
    assert state["completed_pairs"] == 2
    assert state["latest_stage_sha256"] == stages[-1]["stage_sha256"]
    debt._validate_hash(json.loads(state_path.read_text()), "state_sha256")


def test_lagging_state_advances_after_stage_write_crash_window(tmp_path: Path) -> None:
    config_sha = "a" * 64
    rows, stages = _write_two_stages(tmp_path / "stages", config_sha)
    state_path = tmp_path / "state.json"
    debt._atomic_json(
        state_path,
        debt._state_payload(
            config_sha256=config_sha,
            completed_pairs=1,
            latest_stage_sha256=str(stages[0]["stage_sha256"]),
        ),
    )
    state = debt._reconcile_state(
        state_path,
        config_sha256=config_sha,
        prefix_pairs=len(rows),
        stages=stages,
    )
    assert state["completed_pairs"] == 2
    assert state["latest_stage_sha256"] == stages[-1]["stage_sha256"]


def test_state_ahead_of_stage_prefix_is_refused(tmp_path: Path) -> None:
    config_sha = "a" * 64
    rows, stages = _write_two_stages(tmp_path / "stages", config_sha)
    state_path = tmp_path / "state.json"
    debt._atomic_json(
        state_path,
        debt._state_payload(config_sha256=config_sha, completed_pairs=3, latest_stage_sha256="b" * 64),
    )
    with pytest.raises(debt.RawDebtError, match="ahead"):
        debt._reconcile_state(
            state_path,
            config_sha256=config_sha,
            prefix_pairs=len(rows),
            stages=stages,
        )


def test_state_bad_latest_hash_at_declared_boundary_is_refused(tmp_path: Path) -> None:
    config_sha = "a" * 64
    rows, stages = _write_two_stages(tmp_path / "stages", config_sha)
    state_path = tmp_path / "state.json"
    debt._atomic_json(
        state_path,
        debt._state_payload(config_sha256=config_sha, completed_pairs=1, latest_stage_sha256="b" * 64),
    )
    with pytest.raises(debt.RawDebtError, match="latest_stage_sha256"):
        debt._reconcile_state(
            state_path,
            config_sha256=config_sha,
            prefix_pairs=len(rows),
            stages=stages,
        )


def test_state_body_mutation_is_refused_before_recovery(tmp_path: Path) -> None:
    config_sha = "a" * 64
    rows, stages = _write_two_stages(tmp_path / "stages", config_sha)
    state_path = tmp_path / "state.json"
    state = debt._state_payload(
        config_sha256=config_sha,
        completed_pairs=1,
        latest_stage_sha256=str(stages[0]["stage_sha256"]),
    )
    state["completed_pairs"] = 0
    debt._atomic_json(state_path, state)
    with pytest.raises(debt.RawDebtError, match="differs from receipt body"):
        debt._reconcile_state(
            state_path,
            config_sha256=config_sha,
            prefix_pairs=len(rows),
            stages=stages,
        )


def test_stale_frontier_pointer_does_not_change_launch_scientific_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "launch.json"
    monkeypatch.setattr(debt.scorer, "_effective_pointer_target", lambda: {"score": 0.2})
    first = debt._ensure_launch_manifest(path, config_sha256="a" * 64)
    monkeypatch.setattr(debt.scorer, "_effective_pointer_target", lambda: {"score": 0.1})
    resumed = debt._ensure_launch_manifest(path, config_sha256="a" * 64)
    assert resumed == first
    assert resumed["frontier_pointer_at_launch"] == {"score": 0.2}
    assert resumed["config_sha256"] == "a" * 64


def test_committed_v2_receipt_remains_valid_historical() -> None:
    path = (
        debt.REPO / ".omx/research/original_taskspace_inverse_witness_codec_20260725/"
        "c1_live_target_debt_n600_batch16.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    debt._validate_hash(receipt, "receipt_sha256")
    config = receipt["config"]
    assert debt._historical_v2_matches(
        receipt,
        receipt_path=path,
        raw_sha256=config["raw"]["sha256"],
        target_raw_sha256=config["target_raw"]["sha256"],
        cache_sha256=config["cache"]["sha256"],
        pair_count=config["pair_count"],
        stage_pairs=config["stage_pairs"],
    )


def test_fabricated_self_hashed_v2_receipt_is_not_legacy_authority(tmp_path: Path) -> None:
    config = {
        "schema": "tac.coupled_witness_raw_debt_state.v2",
        "raw": {"sha256": "a" * 64},
        "target_raw": {"sha256": "b" * 64},
        "cache": {"sha256": "c" * 64},
        "pair_count": 600,
        "stage_pairs": 16,
    }
    receipt = debt._with_hash(
        {
            "schema": debt.LEGACY_FINAL_SCHEMA,
            "config": config,
            "config_sha256": debt._sha256_bytes(debt._canonical(config)),
        },
        "receipt_sha256",
    )
    path = tmp_path / "fabricated-v2.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert not debt._historical_v2_matches(
        receipt,
        receipt_path=path,
        raw_sha256="a" * 64,
        target_raw_sha256="b" * 64,
        cache_sha256="c" * 64,
        pair_count=600,
        stage_pairs=16,
    )


def test_end_barrier_refuses_long_run_input_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows: dict[str, dict[str, object]] = {}
    for key in ("raw", "target_raw", "cache", "tool", "scorer_adapter"):
        path = tmp_path / key
        path.write_bytes(key.encode())
        rows[key] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": debt._sha256_file(path),
        }
    target_receipt = tmp_path / "target-receipt.json"
    target_receipt.write_text("{}", encoding="utf-8")
    config: dict[str, object] = {
        **rows,
        "target_reference": {
            "path": str(target_receipt.resolve()),
            "bytes": target_receipt.stat().st_size,
            "sha256": debt._sha256_file(target_receipt),
        },
        "scorer_custody": {},
    }
    monkeypatch.setattr(debt.scorer, "_validate_scorer_custody", lambda _value: None)
    barrier = debt._validate_input_end_barrier(config, contest_reference=None)
    assert barrier["scorer_custody_revalidated"] is True
    Path(str(rows["raw"]["path"])).write_bytes(b"mutated")
    with pytest.raises(debt.RawDebtError, match="scientific input drifted during scoring: raw"):
        debt._validate_input_end_barrier(config, contest_reference=None)


def test_recorded_end_barrier_must_preserve_exact_files() -> None:
    current = {
        "schema": "tac.coupled_witness_raw_debt_end_barrier.v1",
        "verified_at_utc": "later",
        "files": {"raw": {"path": "/x", "bytes": 1, "sha256": "a" * 64}},
        "scorer_custody_revalidated": True,
    }
    recorded = {**current, "verified_at_utc": "earlier"}
    debt._validate_recorded_input_end_barrier(recorded, current)
    drifted = {**recorded, "files": {"raw": {"path": "/x", "bytes": 2, "sha256": "b" * 64}}}
    with pytest.raises(debt.RawDebtError, match="barrier drifted"):
        debt._validate_recorded_input_end_barrier(drifted, current)
    with pytest.raises(debt.RawDebtError, match="lacks"):
        debt._validate_recorded_input_end_barrier(None, current)
