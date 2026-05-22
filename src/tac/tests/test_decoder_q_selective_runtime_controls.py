# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from tac.optimization.decoder_q_selective_runtime_packet import pack_dqs1_payload
from tools.run_decoder_q_selective_runtime_locality_controls import (
    compare_raw_triplet,
    parse_selected_pairs,
    selected_frame_indices_for_pairs,
    validate_selective_runtime_contract,
)


def _write_raw(path: Path, frames: list[bytes]) -> None:
    path.write_bytes(b"".join(frames))


def _frames(prefix: int, count: int = 6) -> list[bytes]:
    return [bytes([prefix + idx, 255 - idx]) for idx in range(count)]


def _write_stored_zip(path: Path, payload: bytes) -> None:
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, payload)


def _write_selective_manifest(
    submission_dir: Path,
    *,
    payload: bytes,
    pairs: list[int],
    frames: list[int],
    frame_policy: str,
) -> None:
    (submission_dir / "selective_runtime_manifest.json").write_text(
        json.dumps(
            {
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "rank_or_kill_eligible": False,
                "promotable": False,
                "dqs1_payload": {
                    "frame_policy": frame_policy,
                    "pair_indices": pairs,
                    "affected_frame_indices": frames,
                    "payload_bytes": len(payload),
                    "payload_sha256": hashlib.sha256(payload).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )


def test_frame_policy_maps_selected_pairs_to_expected_frames() -> None:
    assert selected_frame_indices_for_pairs(
        [1],
        frame_policy="pair_all_frames",
    ) == [2, 3]
    assert selected_frame_indices_for_pairs(
        [1],
        frame_policy="segnet_last_frame_only",
    ) == [3]


def test_validate_selective_contract_cross_checks_manifest_and_archive(
    tmp_path: Path,
) -> None:
    payload = pack_dqs1_payload(
        pair_indices=[1],
        frame_policy="pair_all_frames",
        storage_index=26,
        q_offset=0,
        delta=1,
    )
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    _write_stored_zip(submission_dir / "archive.zip", b"FP11_STUB" + payload)
    _write_selective_manifest(
        submission_dir,
        payload=payload,
        pairs=[1],
        frames=[2, 3],
        frame_policy="pair_all_frames",
    )

    result = validate_selective_runtime_contract(
        selective_submission_dir=submission_dir,
        selected_pairs=[1],
        frame_policy="pair_all_frames",
    )

    assert result["selected_frame_indices"] == [2, 3]
    assert result["dqs1_tail_bytes"] == len(payload)


def test_validate_selective_contract_rejects_cli_pair_mismatch(tmp_path: Path) -> None:
    payload = pack_dqs1_payload(
        pair_indices=[1],
        frame_policy="pair_all_frames",
        storage_index=26,
        q_offset=0,
        delta=1,
    )
    submission_dir = tmp_path / "submission"
    submission_dir.mkdir()
    _write_stored_zip(submission_dir / "archive.zip", b"FP11_STUB" + payload)
    _write_selective_manifest(
        submission_dir,
        payload=payload,
        pairs=[1],
        frames=[2, 3],
        frame_policy="pair_all_frames",
    )

    with pytest.raises(ValueError, match="selected pairs mismatch"):
        validate_selective_runtime_contract(
            selective_submission_dir=submission_dir,
            selected_pairs=[2],
            frame_policy="pair_all_frames",
        )


def test_parse_selected_pairs_sorts_and_rejects_duplicates() -> None:
    assert parse_selected_pairs("501, 7 9") == [7, 9, 501]

    try:
        parse_selected_pairs("7,7")
    except ValueError as exc:
        assert "duplicates" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate pairs must fail")


def test_compare_raw_triplet_accepts_selective_locality(tmp_path: Path) -> None:
    parent_frames = _frames(10)
    global_frames = list(parent_frames)
    global_frames[2] = b"g2"
    global_frames[3] = b"g3"
    selective_frames = list(parent_frames)
    selective_frames[2] = global_frames[2]
    selective_frames[3] = global_frames[3]

    parent = tmp_path / "parent.raw"
    global_mutated = tmp_path / "global.raw"
    selective = tmp_path / "selective.raw"
    _write_raw(parent, parent_frames)
    _write_raw(global_mutated, global_frames)
    _write_raw(selective, selective_frames)

    result = compare_raw_triplet(
        parent_raw=parent,
        global_mutated_raw=global_mutated,
        selective_raw=selective,
        selected_frame_indices=[2, 3],
        frame_count=6,
        frame_bytes=2,
    )

    assert result["raw_byte_sizes_match"] is True
    assert result["mismatch_counts"]["selected_frame_mismatch_count"] == 0
    assert result["mismatch_counts"]["unselected_frame_mismatch_count"] == 0
    assert result["blockers"] == []
    assert (
        result["hashes"]["selected_frames"]["selective"]
        == result["hashes"]["selected_frames"]["global_mutated"]
    )
    assert (
        result["hashes"]["unselected_frames"]["selective"]
        == result["hashes"]["unselected_frames"]["parent"]
    )


def test_compare_raw_triplet_reports_selected_and_unselected_mismatches(
    tmp_path: Path,
) -> None:
    parent_frames = _frames(30)
    global_frames = list(parent_frames)
    global_frames[3] = b"g3"
    selective_frames = list(parent_frames)
    selective_frames[3] = b"xx"
    selective_frames[4] = b"yy"

    parent = tmp_path / "parent.raw"
    global_mutated = tmp_path / "global.raw"
    selective = tmp_path / "selective.raw"
    _write_raw(parent, parent_frames)
    _write_raw(global_mutated, global_frames)
    _write_raw(selective, selective_frames)

    result = compare_raw_triplet(
        parent_raw=parent,
        global_mutated_raw=global_mutated,
        selective_raw=selective,
        selected_frame_indices=[3],
        frame_count=6,
        frame_bytes=2,
    )

    assert result["mismatch_counts"]["selected_frame_mismatch_count"] == 1
    assert result["mismatch_counts"]["unselected_frame_mismatch_count"] == 1
    assert "selected_frame_locality_mismatch:0.raw" in result["blockers"]
    assert "unselected_frame_parent_regression:0.raw" in result["blockers"]
    assert {sample["frame_index"] for sample in result["mismatch_samples"]} == {3, 4}


def test_compare_raw_triplet_reports_raw_size_mismatch(tmp_path: Path) -> None:
    parent = tmp_path / "parent.raw"
    global_mutated = tmp_path / "global.raw"
    selective = tmp_path / "selective.raw"
    parent.write_bytes(b"aabbcc")
    global_mutated.write_bytes(b"aabbcc")
    selective.write_bytes(b"aabb")

    result = compare_raw_triplet(
        parent_raw=parent,
        global_mutated_raw=global_mutated,
        selective_raw=selective,
        selected_frame_indices=[1],
        frame_count=3,
        frame_bytes=2,
    )

    assert result["raw_byte_sizes_match"] is False
    assert result["mismatch_counts"]["raw_size_mismatch_count"] == 1
    assert result["blockers"] == ["raw_size_mismatch:0.raw"]
