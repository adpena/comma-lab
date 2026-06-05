# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

import tac.analysis.snerv_official_tub_source_forward_replay as replay_mod
from tac.analysis.snerv_official_tub_source_forward_replay import (
    DEFAULT_OFFICIAL_SNERV_REPO,
    TUB_CLOSED_BY_FIXTURE_REPLAY,
    TUB_FRAME_RECONSTRUCTION_BLOCKER,
    build_snerv_official_tub_source_forward_replay_artifact,
)


def _official_repo() -> Path:
    if not DEFAULT_OFFICIAL_SNERV_REPO.exists():
        pytest.skip(f"official SNeRV checkout is absent: {DEFAULT_OFFICIAL_SNERV_REPO}")
    return DEFAULT_OFFICIAL_SNERV_REPO


def test_tub_source_forward_replay_requires_real_frame_reconstruction() -> None:
    artifact = build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=_official_repo(),
        generated_utc="20260605T000000Z",
    )

    assert artifact["source_forward_replay_executed"] is True
    assert (
        artifact["official_tub_temporal_encoder_output2_source_fixture_replay_passed"]
        is True
    )
    assert artifact["source_forward_parity_proven"] is False
    assert artifact["full_tub_source_forward_parity_proven"] is False
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False

    frame = artifact["frame_reconstruction_equivalence"]
    assert frame["component_id"] == "tub_mfu_hfr_frame_reconstruction_equivalence"
    assert frame["source_forward_frame_reconstruction_matches_official"] is True
    assert frame["max_abs_error"] == 0.0
    assert frame["output_hashes_bit_identical"] is True
    assert frame["output_shapes"]["img_yl"] == [1, 3, 16, 16]
    assert frame["output_shapes"]["yh_out"] == [1, 3, 3, 16, 16]
    assert frame["output_shapes"]["img_out"] == [1, 3, 32, 32]
    assert frame["portable_frame_reconstruction_metadata"]["output_shapes"][
        "frame"
    ] == [1, 3, 32, 32]
    assert frame["score_claim"] is False
    assert frame["ready_for_exact_eval_dispatch"] is False

    component_ids = {row["component_id"] for row in artifact["component_rows"]}
    assert "tub_mfu_hfr_frame_reconstruction_equivalence" in component_ids
    assert (
        "snerv_official_tub_frame_reconstruction_source_forward_replay_missing"
        in frame["closed_blockers"]
    )
    for blocker in TUB_CLOSED_BY_FIXTURE_REPLAY:
        assert blocker in artifact["closed_blockers"]
    assert (
        "snerv_official_trained_checkpoint_state_dict_not_loaded"
        in artifact["preserved_blockers"]
    )


def test_tub_replay_frame_mismatch_stays_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = replay_mod.official_tub_frame_reconstruction_numpy

    def corrupt_frame(*args: object, **kwargs: object) -> object:
        out = original(*args, **kwargs)
        return replace(out, frame=out.frame + 1.0)

    monkeypatch.setattr(
        replay_mod,
        "official_tub_frame_reconstruction_numpy",
        corrupt_frame,
    )

    artifact = replay_mod.build_snerv_official_tub_source_forward_replay_artifact(
        official_repo_dir=_official_repo(),
        generated_utc="20260605T000000Z",
    )

    assert (
        artifact["official_tub_temporal_encoder_output2_source_fixture_replay_passed"]
        is False
    )
    frame = artifact["frame_reconstruction_equivalence"]
    assert frame["source_forward_frame_reconstruction_matches_official"] is False
    assert frame["max_abs_error"] > 0.0
    assert TUB_FRAME_RECONSTRUCTION_BLOCKER in frame["blockers"]
    assert TUB_FRAME_RECONSTRUCTION_BLOCKER not in frame["closed_blockers"]
    assert TUB_FRAME_RECONSTRUCTION_BLOCKER in artifact["blockers"]
    assert TUB_FRAME_RECONSTRUCTION_BLOCKER not in artifact["closed_blockers"]
    assert artifact["score_claim"] is False
    assert artifact["ready_for_exact_eval_dispatch"] is False
