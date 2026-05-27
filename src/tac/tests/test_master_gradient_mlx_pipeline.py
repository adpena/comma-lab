# SPDX-License-Identifier: MIT
"""Tests for the MLX per-pair master-gradient auto-trigger pipeline + a
deterministic GOLDEN-VECTOR regression guard on the extractor's per-byte
projection.

Covers (per the de-orphan pipeline landing 2026-05-27):

1. GOLDEN-VECTOR: a tiny deterministic ``TensorByteSpan`` + sensitivity-dict
   fixture -> a hand-computed (N_bytes, N_pairs, 3) tensor. Locks the
   ``project_per_tensor_sensitivity_to_per_byte`` math so the extractor's
   output cannot silently drift.
2. Auto-trigger seam: idempotency, scheduled-row emission, frontier-sha
   derivation, already-landed-artifact de-dup, fail-quiet error path.
3. Catalog #341 canonical non-promotable markers on every verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tac.master_gradient_mlx_extractor import (
    TensorByteSpan,
    project_per_tensor_sensitivity_to_per_byte,
)
from tac.master_gradient_mlx_pipeline import (
    EVENT_COMPLETED,
    EVENT_NO_OP,
    EVENT_SCHEDULED,
    MLXExtractionScheduleVerdict,
    auto_schedule_mlx_per_pair_extraction_for_frontier,
    latest_extraction_state_for_sha,
    load_extraction_state_lenient,
    resolve_latest_mlx_artifact_for_sha,
)


# --------------------------------------------------------------------------- #
# GOLDEN VECTOR — deterministic per-byte projection regression guard           #
# --------------------------------------------------------------------------- #


def test_golden_vector_per_byte_projection() -> None:
    """A hand-computed (N_bytes, N_pairs, 3) regression guard.

    Two decoder tensors at known offsets + known fp16_scale + known per-pair
    sensitivity. The projection distributes ``sens * scale_mag / numel``
    uniformly across each tensor's mantissa-byte span; rate (axis 2) is zero.
    This locks the canonical math so the extractor cannot silently drift.
    """
    archive_bytes_count = 32
    n_pairs = 2
    decoder_blob_offset = 4

    spans = [
        # tensor A: 4 bytes at decompressed offset 0 -> archive bytes [4, 8)
        TensorByteSpan(
            name="conv.weight",
            storage_index=0,
            shape=(2, 2),
            numel=4,
            mantissa_byte_offset=0,
            fp16_scale=2.0,
        ),
        # tensor B: 2 bytes at decompressed offset 10 -> archive bytes [14, 16)
        TensorByteSpan(
            name="head.bias",
            storage_index=1,
            shape=(2,),
            numel=2,
            mantissa_byte_offset=10,
            fp16_scale=0.5,
        ),
    ]
    sens_seg = {
        "conv.weight": np.array([1.0, 3.0]),  # per-pair
        "head.bias": np.array([8.0, 0.0]),
    }
    sens_pose = {
        "conv.weight": np.array([0.5, 0.25]),
        "head.bias": np.array([2.0, 4.0]),
    }

    G = project_per_tensor_sensitivity_to_per_byte(
        spans,
        sens_seg,
        sens_pose,
        archive_bytes_count=archive_bytes_count,
        decoder_blob_offset=decoder_blob_offset,
        n_pairs_used=n_pairs,
    )

    assert G.shape == (archive_bytes_count, n_pairs, 3)
    assert G.dtype == np.float64

    # Tensor A: scale=2.0, numel=4 -> per-byte = sens * 2.0 / 4 = sens * 0.5
    #   archive bytes [4, 8), 4 bytes, each pair-row identical.
    expected_a_seg = sens_seg["conv.weight"] * 2.0 / 4.0  # [0.5, 1.5]
    expected_a_pose = sens_pose["conv.weight"] * 2.0 / 4.0  # [0.25, 0.125]
    for b in range(4, 8):
        np.testing.assert_allclose(G[b, :, 0], expected_a_seg)
        np.testing.assert_allclose(G[b, :, 1], expected_a_pose)
        np.testing.assert_array_equal(G[b, :, 2], np.zeros(n_pairs))

    # Tensor B: scale=0.5, numel=2 -> per-byte = sens * 0.5 / 2 = sens * 0.25
    #   archive bytes [4+10, 4+10+2) = [14, 16).
    expected_b_seg = sens_seg["head.bias"] * 0.5 / 2.0  # [1.0, 0.0]
    expected_b_pose = sens_pose["head.bias"] * 0.5 / 2.0  # [0.25, 0.5]
    for b in range(14, 16):
        np.testing.assert_allclose(G[b, :, 0], expected_b_seg)
        np.testing.assert_allclose(G[b, :, 1], expected_b_pose)

    # Bytes outside the two spans are zero.
    for b in (0, 1, 2, 3, 8, 9, 13, 16, 31):
        np.testing.assert_array_equal(G[b], np.zeros((n_pairs, 3)))

    # Rate column is identically zero everywhere (byte-value sensitivities do
    # not move the rate term).
    np.testing.assert_array_equal(G[:, :, 2], np.zeros((archive_bytes_count, n_pairs)))

    # Per-pair axis preserved: pair-0 and pair-1 differ where sens differs.
    assert not np.allclose(G[4, 0, :], G[4, 1, :])


def test_golden_vector_clamps_span_past_archive_end() -> None:
    """A span whose decompressed offset runs past the archive size is clamped."""
    spans = [
        TensorByteSpan(
            name="big.weight",
            storage_index=0,
            shape=(100,),
            numel=100,
            mantissa_byte_offset=0,
            fp16_scale=1.0,
        ),
    ]
    sens = {"big.weight": np.array([1.0])}
    G = project_per_tensor_sensitivity_to_per_byte(
        spans,
        sens,
        sens,
        archive_bytes_count=8,
        decoder_blob_offset=0,
        n_pairs_used=1,
    )
    assert G.shape == (8, 1, 3)
    # All 8 bytes attributed (clamped to archive end); none past it.
    assert np.all(G[:, 0, 0] == 1.0 * 1.0 / 100.0)


def test_golden_vector_skips_zero_or_nonfinite_scale() -> None:
    spans = [
        TensorByteSpan(
            name="zero.scale",
            storage_index=0,
            shape=(2,),
            numel=2,
            mantissa_byte_offset=0,
            fp16_scale=0.0,
        ),
    ]
    sens = {"zero.scale": np.array([5.0])}
    G = project_per_tensor_sensitivity_to_per_byte(
        spans, sens, sens, archive_bytes_count=4, decoder_blob_offset=0, n_pairs_used=1
    )
    np.testing.assert_array_equal(G, np.zeros((4, 1, 3)))


# --------------------------------------------------------------------------- #
# AUTO-TRIGGER seam                                                            #
# --------------------------------------------------------------------------- #


def _write_pointer(tmp_path: Path, sha: str, axis: str = "contest_cpu") -> None:
    """Write a minimal canonical frontier pointer file under tmp_path."""
    state_dir = tmp_path / ".omx" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    anchor = {
        "score": 0.1920282830,
        "axis": axis,
        "archive_sha256": sha,
        "lane_id": "test_frontier_lane",
        "hardware_substrate": "linux_x86_64_cpu",
        "measured_at_utc": "2026-05-27T00:00:00+00:00",
        "evidence_grade": "[contest-CPU]",
    }
    pointer = {
        "schema_version": "canonical_frontier_pointer_v1",
        "our_local_frontier_contest_cpu": anchor if axis == "contest_cpu" else None,
        "our_local_frontier_contest_cuda": anchor if axis == "contest_cuda" else None,
        "submitted_pr_number_for_current_frontier": None,
        "upstream_leaderboard_snapshot": None,
        "upstream_leaderboard_snapshot_at_utc": None,
        "last_refreshed_utc": "2026-05-27T00:00:00+00:00",
        "auto_update_on_dispatch_completion": True,
        "pointer_refresh_command": "tools/refresh_canonical_frontier.py",
        "refresh_provenance": {},
    }
    (state_dir / "canonical_frontier_pointer.json").write_text(
        json.dumps(pointer, sort_keys=True)
    )


def test_auto_trigger_no_pointer_is_no_op(tmp_path: Path) -> None:
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path, state_path=state_path
    )
    assert isinstance(verdict, MLXExtractionScheduleVerdict)
    assert verdict.fired is False
    assert verdict.event_type == EVENT_NO_OP
    assert verdict.frontier_archive_sha256 is None
    # Catalog #341 markers.
    assert verdict.promotable is False
    assert verdict.score_claim is False


def test_auto_trigger_schedules_on_new_frontier(tmp_path: Path) -> None:
    sha = "a" * 64
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    manifest_path = tmp_path / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        frontier_axis="contest_cpu",
        state_path=state_path,
        manifest_path=manifest_path,
    )
    assert verdict.fired is True
    assert verdict.event_type == EVENT_SCHEDULED
    assert verdict.frontier_archive_sha256 == sha
    assert verdict.state_row_written is True
    assert verdict.promotable is False
    # Scheduled CLI command is surfaced for out-of-band run.
    assert "extract_master_gradient_mlx.py" in verdict.extra["scheduled_cli_command"]

    rows = load_extraction_state_lenient(state_path)
    assert len(rows) == 1
    assert rows[0]["event_type"] == EVENT_SCHEDULED
    assert rows[0]["frontier_archive_sha256"] == sha
    assert rows[0]["promotable"] is False


def test_auto_trigger_idempotent_after_scheduled(tmp_path: Path) -> None:
    """Re-firing a sha with a prior scheduled row is a no-op (deterministic)."""
    sha = "7" * 64
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    manifest_path = tmp_path / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
    first = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        state_path=state_path,
        manifest_path=manifest_path,
    )
    assert first.event_type == EVENT_SCHEDULED
    second = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        state_path=state_path,
        manifest_path=manifest_path,
    )
    assert second.fired is False
    assert second.event_type == EVENT_NO_OP
    # No duplicate scheduled row appended.
    assert len(load_extraction_state_lenient(state_path)) == 1


def test_auto_trigger_idempotent_after_completed(tmp_path: Path) -> None:
    sha = "b" * 64
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    manifest_path = tmp_path / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
    # Seed a completed row.
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "event_type": EVENT_COMPLETED,
                "frontier_archive_sha256": sha,
                "n_pairs": 64,
                "artifact_path": "/some/artifact.npy",
                "schema_version": "x",
            },
            sort_keys=True,
        )
        + "\n"
    )
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        n_pairs=64,
        state_path=state_path,
        manifest_path=manifest_path,
    )
    assert verdict.fired is False
    assert verdict.event_type == EVENT_NO_OP
    assert verdict.artifact_path == "/some/artifact.npy"
    # No new row appended.
    assert len(load_extraction_state_lenient(state_path)) == 1


def test_auto_trigger_dedups_already_landed_artifact(tmp_path: Path) -> None:
    sha = "c" * 64
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    manifest_path = tmp_path / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "archive_sha256": sha,
                "npy_path": ".omx/state/landed_artifact.npy",
                "evidence_grade": "macOS-MLX research-signal",
            },
            sort_keys=True,
        )
        + "\n"
    )
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        state_path=state_path,
        manifest_path=manifest_path,
    )
    assert verdict.fired is True
    assert verdict.event_type == EVENT_COMPLETED
    assert verdict.artifact_path == ".omx/state/landed_artifact.npy"
    rows = load_extraction_state_lenient(state_path)
    assert rows[-1]["event_type"] == EVENT_COMPLETED


def test_auto_trigger_force_reschedules(tmp_path: Path) -> None:
    sha = "d" * 64
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    manifest_path = tmp_path / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "event_type": EVENT_COMPLETED,
                "frontier_archive_sha256": sha,
                "n_pairs": 64,
                "artifact_path": "/old.npy",
                "schema_version": "x",
            },
            sort_keys=True,
        )
        + "\n"
    )
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        n_pairs=64,
        force=True,
        state_path=state_path,
        manifest_path=manifest_path,
    )
    assert verdict.fired is True
    assert verdict.event_type == EVENT_SCHEDULED


def test_derive_frontier_sha_reads_pointer(tmp_path: Path) -> None:
    from tac.master_gradient_mlx_pipeline import derive_frontier_archive_sha

    sha = "e" * 64
    _write_pointer(tmp_path, sha, axis="contest_cpu")
    derived_sha, axis = derive_frontier_archive_sha(repo_root=tmp_path)
    assert derived_sha == sha
    assert axis == "contest_cpu"


def test_auto_trigger_end_to_end_via_pointer(tmp_path: Path) -> None:
    """Full seam: pointer file -> derive sha -> schedule row."""
    sha = "f" * 64
    _write_pointer(tmp_path, sha, axis="contest_cpu")
    state_path = tmp_path / ".omx" / "state" / "mlx_per_pair_extraction_state.jsonl"
    manifest_path = tmp_path / ".omx" / "state" / "mlx_research_signal_manifest.jsonl"
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path, state_path=state_path, manifest_path=manifest_path
    )
    assert verdict.fired is True
    assert verdict.event_type == EVENT_SCHEDULED
    assert verdict.frontier_archive_sha256 == sha
    assert verdict.frontier_axis == "contest_cpu"


def test_resolve_latest_artifact_newest_wins(tmp_path: Path) -> None:
    sha = "1" * 64
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        json.dumps({"archive_sha256": sha, "npy_path": "old.npy"}, sort_keys=True)
        + "\n"
        + json.dumps({"archive_sha256": sha, "npy_path": "new.npy"}, sort_keys=True)
        + "\n"
    )
    assert (
        resolve_latest_mlx_artifact_for_sha(sha, manifest_path=manifest_path)
        == "new.npy"
    )


def test_latest_extraction_state_filters_by_n_pairs(tmp_path: Path) -> None:
    sha = "2" * 64
    state_path = tmp_path / "state.jsonl"
    state_path.write_text(
        json.dumps(
            {"frontier_archive_sha256": sha, "n_pairs": 64, "event_type": "scheduled"},
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {"frontier_archive_sha256": sha, "n_pairs": 600, "event_type": "scheduled"},
            sort_keys=True,
        )
        + "\n"
    )
    row64 = latest_extraction_state_for_sha(sha, n_pairs=64, path=state_path)
    row600 = latest_extraction_state_for_sha(sha, n_pairs=600, path=state_path)
    assert row64 is not None and row64["n_pairs"] == 64
    assert row600 is not None and row600["n_pairs"] == 600


def test_auto_trigger_never_raises_on_corrupt_state(tmp_path: Path, monkeypatch) -> None:
    """The seam is fail-quiet: an internal error returns an error verdict."""
    sha = "3" * 64

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("simulated internal failure")

    import tac.master_gradient_mlx_pipeline as mod

    monkeypatch.setattr(mod, "_append_state_row_locked", _boom)
    verdict = auto_schedule_mlx_per_pair_extraction_for_frontier(
        repo_root=tmp_path,
        frontier_archive_sha256=sha,
        state_path=tmp_path / "s.jsonl",
        manifest_path=tmp_path / "m.jsonl",
    )
    assert verdict.event_type == "error"
    assert verdict.fired is False
    assert verdict.error is not None
    assert verdict.promotable is False
