from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import ddm_sd2_pr130_seg_decomposition_runner as sd2


def _write_array(path: Path, value: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return sd2.artifact(path)


def _signature_fixture() -> np.ndarray:
    labels = np.zeros((1, sd2.EVAL_H, sd2.EVAL_W), dtype=np.uint8)
    labels[:, :192] = 2
    labels[:, 288:] = 4
    labels[:, 200:260, 450:] = 3
    labels[:, 220:224, 200:300] = 1
    return labels


def test_decomposition_reopens_retained_argmax_and_self_detects_order(
    tmp_path: Path,
) -> None:
    target = _signature_fixture()
    prediction = target.copy()
    prediction[0, 270, 100] = 1
    prediction[0, 191, 0] = 0

    arrays = {
        "target/segnet_argmax": _write_array(tmp_path / "target/segnet_argmax.npy", target),
        "base/segnet_argmax": _write_array(tmp_path / "base/segnet_argmax.npy", prediction),
    }
    result = sd2.decompose_retained_argmax([{"arrays": arrays}], "base")

    assert result["denominators"]["evaluated_pixels"] == sd2.EVAL_H * sd2.EVAL_W
    assert result["denominators"]["exact_mismatch_pixels"] == 2
    assert result["directed_target_rows_prediction_columns"][0][1] == 1
    assert result["directed_target_rows_prediction_columns"][2][0] == 1
    assert result["boundary_interior"]["boundary_mismatch_pixels"] == 1
    assert result["boundary_interior"]["interior_mismatch_pixels"] == 1
    assert result["class_order"]["detected_name_to_index"] == {
        "Road": 0,
        "Lane": 1,
        "Undrivable": 2,
        "Movable": 3,
        "MyCar": 4,
    }
    assert result["per_frame"][0]["error_pixels"] == 2


def test_class_self_detection_refuses_luma_order_shape() -> None:
    signatures = sd2.class_signatures([_signature_fixture()])
    wrong = copy.deepcopy(signatures)
    wrong[2]["bottom_quarter_share"] = 1.01
    with pytest.raises(ValueError, match="self-detection"):
        sd2.self_detect_class_order(wrong)


def test_queue_writer_plan_fails_closed_when_one_writer_is_missing() -> None:
    queue = json.loads(sd2.DEFAULT_QUEUE.read_text())
    retention = sd2.validate_queue(queue)
    assert retention["do_not_launch_if_any_payload_writer_is_absent"] is True

    broken = copy.deepcopy(queue)
    broken["required_retention"]["per_candidate"].remove("SegNet argmax chunks with bytes and SHA-256")
    with pytest.raises(ValueError, match="retention declarations missing"):
        sd2.validate_queue(broken)


def test_writer_preflight_executes_live_byte_array_and_range_probes(
    tmp_path: Path,
) -> None:
    retention = sd2.validate_queue(json.loads(sd2.DEFAULT_QUEUE.read_text()))
    receipt = sd2.writer_preflight(tmp_path, retention)

    assert receipt["status"] == "PASS"
    assert receipt["byte_writer_probe"]["bytes"] > 0
    assert receipt["numpy_writer_probe"]["bytes"] > 0
    assert receipt["range_writer_probe"]["bytes"] == 7
    assert Path(receipt["receipt"]["path"]).is_file()


def test_full_projection_prices_retained_logits_and_decode_contingency() -> None:
    projection = sd2.storage_projection(600, 60)
    assert projection["base_receiver_raw_bytes"] == 3_662_409_600
    assert projection["candidate_receiver_raw_bytes"] == 3_662_409_600
    assert projection["retained_segnet_logits_bytes"] == 7_077_888_000
    assert projection["retained_segnet_argmax_bytes"] == 353_894_400
    assert projection["failed_decode_attempt_contingency_bytes"] == 3_662_409_600
    assert projection["projected_final_bytes"] < 20_000_000_000


def test_decode_promotion_refuses_a_stale_archive_binding(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    (attempt / "inflated").mkdir(parents=True)
    (attempt / "inflated/0.raw").write_bytes(b"1234")
    (attempt / "subprocess_success.json").write_text(json.dumps({"returncode": 0}))
    (attempt / "command.json").write_text(
        json.dumps(
            {
                "candidate_id": sd2.BASE_ID,
                "receiver_commit": sd2.RUNTIME_COMMIT,
                "decode_device": "cpu",
                "archive": {"sha256": "stale"},
            }
        )
    )

    with pytest.raises(ValueError, match="binding differs"):
        sd2.promote_successful_attempt(
            attempt,
            tmp_path / "final/0.raw",
            4,
            candidate_id=sd2.BASE_ID,
            archive_sha256=sd2.BASE_ARCHIVE_SHA256,
            decode_device="cpu",
        )


def test_token_checkpoint_policy_follows_pinned_receiver_codec(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    sd2.materialize_runtime(runtime_dir)
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()

    range_payload = payload_dir / "range.p"
    range_payload.write_bytes(struct.pack("<I", 1) + b"m" + b"t")
    range_inspection = sd2.inspect_retained_token_codec(
        runtime_dir,
        sd2.artifact(range_payload),
    )
    range_policy = sd2.token_checkpoint_policy(
        decode_root=tmp_path / "range_decode",
        codec_inspection=range_inspection,
    )
    assert range_inspection["token_codec"] == "range"
    assert range_policy["environment"] == {}
    assert range_policy["token_cache"] is None
    assert range_policy["intra_decode_checkpointing"] == "DISABLED_RANGE_SEQUENTIAL_REPLAY"

    ans_payload = payload_dir / "ans.p"
    ans_payload.write_bytes(struct.pack("<I", (1 << 31) | 1) + b"m" + b"t")
    ans_inspection = sd2.inspect_retained_token_codec(
        runtime_dir,
        sd2.artifact(ans_payload),
    )
    ans_policy = sd2.token_checkpoint_policy(
        decode_root=tmp_path / "ans_decode",
        codec_inspection=ans_inspection,
    )
    assert ans_inspection["token_codec"] == "ans"
    assert set(ans_policy["environment"]) == {
        "PR130_TOKEN_CACHE",
        "PR130_TOKEN_RECEIPT",
    }
    assert ans_policy["token_cache"] is not None
    assert ans_policy["intra_decode_checkpointing"] == "ENABLED_ANS_STACK"


def test_range_policy_refuses_stale_ans_checkpoint_artifacts(tmp_path: Path) -> None:
    decode_root = tmp_path / "decode"
    stale = decode_root / "token_cache/tokens.progress.npz"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"retained-stale-checkpoint")
    with pytest.raises(ValueError, match="cannot be silently ignored"):
        sd2.token_checkpoint_policy(
            decode_root=decode_root,
            codec_inspection={"token_codec": "range"},
        )


def test_predecode_progress_allows_only_recorded_runner_hash_migration(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    progress_path = out_dir / "progress.json"
    configuration = {"out_dir": str(out_dir), "pair_count": 600}
    old_fingerprints = {"runner": "old", "queue": "same"}
    progress_path.write_text(
        json.dumps(
            {
                "schema": "ddm_sd2.progress.v1",
                "configuration": configuration,
                "fingerprints": old_fingerprints,
                "completed_stages": ["retention_preflight"],
                "chunks": [],
                "active_chunk": None,
            }
        )
    )
    migrated = sd2.initialize_progress(
        progress_path,
        configuration=configuration,
        fingerprints={"runner": "new", "queue": "same"},
    )
    assert migrated["fingerprints"]["runner"] == "new"
    assert migrated["migrations"][-1]["from_runner_sha256"] == "old"
    assert migrated["migrations"][-1]["admission"] == (
        "PRE_DECODE_ONLY_NO_RETAINED_SCORER_CHUNKS"
    )

    migrated["completed_stages"].append("base_real_decode")
    migrated["fingerprints"]["runner"] = "new"
    progress_path.write_text(json.dumps(migrated))
    with pytest.raises(ValueError, match="safe pre-decode"):
        sd2.initialize_progress(
            progress_path,
            configuration=configuration,
            fingerprints={"runner": "newer", "queue": "same"},
        )


def test_boundary_mask_marks_both_sides_of_four_neighbor_edge() -> None:
    labels = np.zeros((1, 3, 4), dtype=np.uint8)
    labels[:, :, 2:] = 1
    boundary = sd2.boundary_mask(labels)
    expected = np.zeros_like(boundary)
    expected[:, :, 1:3] = True
    assert np.array_equal(boundary, expected)


def test_retained_numpy_is_rehashed_before_decomposition(tmp_path: Path) -> None:
    path = tmp_path / "argmax.npy"
    record = _write_array(path, _signature_fixture())
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs on disk"):
        sd2.open_verified_npy(record)


def test_reduced_prefix_is_never_admitted_as_a_measurement() -> None:
    with pytest.raises(SystemExit):
        sd2.parse_args(
            [
                "--resume-from",
                "/Volumes/APDataStore/pact/ddm_sd2_20260810/test/progress.json",
                "--pair-count",
                "120",
            ]
        )
    args = sd2.parse_args(
        [
            "--resume-from",
            "/Volumes/APDataStore/pact/ddm_sd2_20260810/test/progress.json",
            "--pair-count",
            "120",
            "--plan-only",
        ]
    )
    assert args.plan_only is True


def test_bulk_output_is_fail_closed_to_apdatastore() -> None:
    admitted = sd2.require_bulk_store(Path("/Volumes/APDataStore/pact/ddm_sd2_20260810/matched_local_n600"))
    assert admitted.is_relative_to(sd2.BULK_ROOT)
    with pytest.raises(ValueError, match="bulk evidence must remain"):
        sd2.require_bulk_store(Path("/Volumes/VertigoDataTier/pact/ddm_sd2_20260810/matched_local_n600"))


def test_full_active_chunk_is_finalized_without_recomputing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sd2, "CAMERA_H", 2)
    monkeypatch.setattr(sd2, "CAMERA_W", 2)
    monkeypatch.setattr(sd2, "EVAL_H", 2)
    monkeypatch.setattr(sd2, "EVAL_W", 2)
    monkeypatch.setattr(sd2, "NUM_CLASSES", 2)
    monkeypatch.setattr(sd2, "POSE_OUTPUTS", 3)
    out_dir = tmp_path / "out"
    chunk_dir = out_dir / "retained/chunks/pairs_0000_0000"
    specs = sd2.chunk_array_specs(1)
    arrays = sd2.open_chunk_arrays(chunk_dir, specs)
    for array in arrays.values():
        array[:] = 0
    sd2.flush_chunk_arrays(arrays)
    del arrays

    frame_bytes = sd2.SEQ_LEN * sd2.CAMERA_H * sd2.CAMERA_W * 3
    base_raw = tmp_path / "base/0.raw"
    candidate_raw = tmp_path / "candidate/0.raw"
    base_raw.parent.mkdir()
    candidate_raw.parent.mkdir()
    base_raw.write_bytes(bytes(frame_bytes))
    candidate_raw.write_bytes(bytes(frame_bytes))
    progress_path = out_dir / "progress.json"
    progress = {
        "chunks": [],
        "active_chunk": {
            "pair_start": 0,
            "pair_end_exclusive": 1,
            "filled_pairs": 1,
        },
    }
    manifest = sd2.finalize_active_chunk(
        args=SimpleNamespace(out_dir=out_dir),
        progress=progress,
        progress_path=progress_path,
        base_decode={"raw": {"path": str(base_raw)}},
        candidate_decode={"raw": {"path": str(candidate_raw)}},
        scorer_axis="[test]",
    )

    assert manifest["pair_count"] == 1
    assert progress["active_chunk"] is None
    assert len(progress["chunks"]) == 1
    assert Path(progress["chunks"][0]["manifest"]["path"]).is_file()


def test_resume_progress_refuses_a_noncontiguous_chunk_prefix(tmp_path: Path) -> None:
    manifest_path = tmp_path / "chunk.json"
    manifest_path.write_text(
        json.dumps(
            {
                "pair_start": 60,
                "pair_end_exclusive": 120,
                "pair_count": 60,
            }
        )
    )
    progress = {
        "chunks": [
            {
                "pair_start": 60,
                "pair_end_exclusive": 120,
                "manifest": sd2.artifact(manifest_path),
            }
        ],
        "active_chunk": None,
    }
    with pytest.raises(ValueError, match="contiguous configured prefix"):
        sd2.validate_progress_chunks(progress, pair_count=600, chunk_pairs=60)
