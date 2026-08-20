from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tac.witness_dsl.taskspace_ep725_bridge_eval import (
    PAIR_COUNT,
    RAW_BYTES,
    TaskspaceBridgeEvalError,
    build_bridge_receipt,
    coupled_score_envelope,
    read_json_evidence,
    validate_terminal_g22,
)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _pointer() -> dict[str, object]:
    return {
        "schema_version": "canonical_frontier_pointer_v1_20260519",
        "our_local_frontier_contest_cpu": {
            "score": 0.188,
            "axis": "contest_cpu",
            "archive_sha256": "a" * 64,
            "lane_id": "local",
            "hardware_substrate": "linux_x86_64_cpu",
            "measured_at_utc": "2026-07-01T00:00:00Z",
            "evidence_grade": "[contest-CPU]",
        },
        "our_local_frontier_contest_cuda": None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": {
            "best_entry": {"score": 0.172, "rank": 1, "name": "upstream"}
        },
        "upstream_leaderboard_snapshot_at_utc": "2026-07-26T00:00:00Z",
        "last_refreshed_utc": "2026-07-26T00:00:00Z",
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "refresh",
        "refresh_provenance": {},
        "effective_frontier": {"score": 0.172},
    }


def _g22(tmp_path: Path) -> Path:
    archive_raw = b"selected-archive"
    runtime_raw = b"generic-runtime"
    archive = tmp_path / "archive.zip"
    runtime = tmp_path / "inflate.py"
    archive.write_bytes(archive_raw)
    runtime.write_bytes(runtime_raw)
    raw_sha = "f" * 64
    return _write_json(
        tmp_path / "g22.json",
        {
            "schema": "tac.g22_ep725_xcodec_decode_receipt.v1",
            "archive_artifact": {
                "selected": {
                    "path": str(archive),
                    "bytes": len(archive_raw),
                    "sha256": _sha(archive_raw),
                    "counted_rate_bytes": len(archive_raw),
                    "classification": "not_a_candidate",
                    "decoded_quantized_state_sha256": "e" * 64,
                    "member": {"name": "0.bin", "bytes": 4, "sha256": "d" * 64},
                }
            },
            "generic_decoder_runtime": {
                "path": str(runtime),
                "bytes": len(runtime_raw),
                "sha256": _sha(runtime_raw),
                "contest_rate_cost_bytes": 0,
                "free_under_rule_118": True,
            },
            "decode_receipt": {
                "pair_ids": list(range(PAIR_COUNT)),
                "pair_population_exact_ordered_0_through_599": True,
                "all_chunks_immutable_and_validated": True,
                "chunk_count": 1,
                "chunk_checkpoints": [{"path": "checkpoint"}],
            },
            "output_witness": {
                "pairs_compared": PAIR_COUNT,
                "frames_compared": 1_200,
                "raw_bytes_each": RAW_BYTES,
                "camera_h": 874,
                "camera_w": 1_164,
                "dtype": "uint8",
                "all_bytes_directly_compared": True,
                "uint8_exact_equal": True,
                "source_raw_sha256": raw_sha,
                "selected_raw_sha256": raw_sha,
            },
            "cleanup": {
                "completed": True,
                "scratch_absent": True,
                "per_chunk_checkpoints_preserved": True,
            },
            "authority_pointer_status": {
                "full_n600_receiver_replay_owed": False,
                "decode_equality_invalidated": False,
                "exact_eval_invoked": False,
            },
        },
    )


def test_terminal_g22_requires_exact_full_population(tmp_path: Path) -> None:
    path = _g22(tmp_path)
    closure = validate_terminal_g22(read_json_evidence(path))
    assert closure["pair_count"] == 600
    assert closure["raw_bytes"] == RAW_BYTES
    assert closure["selected_raw_sha256"] == "f" * 64

    value = json.loads(path.read_text())
    value["decode_receipt"]["pair_ids"] = list(range(599))
    _write_json(path, value)
    with pytest.raises(TaskspaceBridgeEvalError, match=r"0\.\.599"):
        validate_terminal_g22(read_json_evidence(path))


def test_coupled_envelope_changes_all_conditional_budgets() -> None:
    low_pose = coupled_score_envelope(
        d_seg=0.0002,
        d_pose=0.00001,
        archive_bytes=80_000,
        target_score=0.172,
    )
    high_pose = coupled_score_envelope(
        d_seg=0.0002,
        d_pose=0.01,
        archive_bytes=80_000,
        target_score=0.172,
    )
    assert low_pose["conditional_boundaries"]["max_archive_bytes_at_same_distortions"] is not None
    assert high_pose["conditional_boundaries"]["max_archive_bytes_at_same_distortions"] is None
    assert high_pose["score_minus_target"] > low_pose["score_minus_target"]
    assert high_pose["independent_axis_thresholds_forbidden"] is True


def test_bridge_receipt_binds_scored_archive_and_reference_output(tmp_path: Path) -> None:
    g22 = read_json_evidence(_g22(tmp_path))
    pointer = read_json_evidence(_write_json(tmp_path / "pointer.json", _pointer()))
    scorer_path = _write_json(tmp_path / "scorer.json", {"schema": "scorer"})
    scorer = read_json_evidence(scorer_path)
    recode_raw = b"smaller-recode"
    recode = tmp_path / "recode.zip"
    recode.write_bytes(recode_raw)
    receipt = build_bridge_receipt(
        g22=g22,
        pointer=pointer,
        scorer_row={
            "pair_count": 600,
            "device": "cpu",
            "deterministic_algorithms": True,
            "raw_sha256": "f" * 64,
            "d_seg": 0.0002,
            "d_pose": 0.00001,
        },
        scorer_evidence=scorer,
        run_custody={"manifest": "bound"},
        scored_archive={
            "path": str(recode),
            "bytes": len(recode_raw),
            "sha256": _sha(recode_raw),
        },
    )
    assert receipt["artifact_identity"]["scored_archive"]["sha256"] == _sha(recode_raw)
    assert receipt["artifact_identity"]["scored_archive_matches_g22_selected"] is False
    assert receipt["artifact_identity"]["scored_output_matches_g22_reference"] is True
    assert receipt["semantic_control_identity"]["score"] == 0.172
    assert len(receipt["receipt_identity_sha256"]) == 64


def test_duplicate_json_key_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(TaskspaceBridgeEvalError, match="duplicate JSON key"):
        read_json_evidence(path)
