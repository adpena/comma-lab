# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.substrates.hprc.mlx_prefilter_coverage import (
    HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
    summarize_mlx_prefilter_coverage,
)


def _write_profile(
    path: Path,
    *,
    pairs: int,
    batch_pairs: int,
    score: float,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
                "max_pairs": pairs,
                "num_pairs": pairs,
                "n_samples": pairs,
                "scorer_batch_pairs": batch_pairs,
                "scope_status": {"full_video": "executed"},
                "score_components": {"canonical_score": score},
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_batched_full_video_prefilter_counts_for_acquisition_not_replay(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "batched_gpu_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=8, score=91.0)

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["full_video_profile_paths"] == [profile_path.as_posix()]
    assert summary["local_replay_mlx_prefilter_passed"] is False
    assert summary["local_replay_profile_paths"] == []
    assert summary["best_full_video_mlx_score"] == 91.0
    assert "full_video_mlx_scorer_replay_not_attached" not in summary["blockers"]
    assert summary["blockers"] == ["mlx_profile_batch_pairs_not_singleton"]
    assert summary["profile_records"][0]["full_video_prefilter"] is True
    assert summary["profile_records"][0]["local_replay_prefilter"] is False


def test_singleton_full_video_prefilter_can_unlock_local_replay(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "singleton_full600.json"
    _write_profile(profile_path, pairs=600, batch_pairs=1, score=0.25)

    summary = summarize_mlx_prefilter_coverage(
        (profile_path,),
        root=tmp_path,
    )

    assert summary["has_full_video_mlx_prefilter"] is True
    assert summary["local_replay_mlx_prefilter_passed"] is True
    assert summary["local_replay_profile_paths"] == [profile_path.as_posix()]
    assert summary["blockers"] == []

