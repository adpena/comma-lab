# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tac.substrates.pretrained_driving_prior.dataset_source import (
    build_dp1_dataset_source,
    collect_local_video_manifest,
)


def test_collect_local_video_manifest_hashes_video_hevc(tmp_path: Path) -> None:
    root = tmp_path / "comma2k19"
    video = root / "dongle" / "route" / "0" / "video.hevc"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"dp1-test-video-bytes")

    rows = collect_local_video_manifest(root)

    assert rows == [
        {
            "relpath": "dongle/route/0/video.hevc",
            "bytes": len(b"dp1-test-video-bytes"),
            "sha256": rows[0]["sha256"],
        }
    ]
    assert len(rows[0]["sha256"]) == 64


def test_dp1_dataset_source_local_chunks_has_complete_sha_coverage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "chunks"
    (root / "a" / "video.hevc").parent.mkdir(parents=True)
    (root / "a" / "video.hevc").write_bytes(b"a")

    source = build_dp1_dataset_source(
        dataset_name="comma2k19",
        source_mode="local_chunks",
        distillation_mode="single_pass",
        seed=1,
        max_distillation_frames=32,
        max_distillation_chunks=1,
        codebook_metadata={
            "license_tags": ["comma2k19:MIT"],
            "dataset_provenance": "comma2k19_local_chunks",
            "num_frames_used": 32,
        },
        chunks_dir=root,
    ).to_dict()

    assert source["chunk_ids"] == ["a/video.hevc"]
    assert source["chunk_sha256_coverage"]["complete"] is True
    assert source["reproducibility_blockers"] == []
    assert source["score_claim_allowed"] is False


def test_dp1_dataset_source_real_stream_requires_sha_manifest() -> None:
    source = build_dp1_dataset_source(
        dataset_name="comma2k19",
        source_mode="stream_log",
        distillation_mode="log_incremental",
        seed=1,
        max_distillation_frames=32,
        max_distillation_chunks=1,
        schedule_log=[],
    ).to_dict()

    assert "dp1_real_dataset_source_has_no_chunk_ids" in source[
        "reproducibility_blockers"
    ]
    assert "dp1_real_dataset_source_missing_pinned_sha256" not in source[
        "reproducibility_blockers"
    ]


def test_dp1_dataset_source_rejects_unwired_dataset() -> None:
    source = build_dp1_dataset_source(
        dataset_name="bdd100k",
        source_mode="local_chunks",
        distillation_mode="single_pass",
        seed=1,
        max_distillation_frames=32,
        max_distillation_chunks=1,
    ).to_dict()

    assert "dp1_dataset_not_trainer_wired_bdd100k" in source[
        "reproducibility_blockers"
    ]
