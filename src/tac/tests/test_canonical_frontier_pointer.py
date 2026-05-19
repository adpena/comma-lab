"""Tests for tac.canonical_frontier_pointer (Catalog #343 canonical helper).

Per CLAUDE.md "Frontier scores are pointer-only - NON-NEGOTIABLE" + Catalog
#343: the canonical pointer file is the SoT for operator-facing frontier +
upstream leaderboard. Tests cover schema validation, fcntl-locked atomic
write, strict + lenient load, refresh paths, network graceful degradation,
DX auto-update wire-in, sister Catalog #131 + #138 discipline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tac.canonical_frontier_pointer import (
    AnchorRecord,
    CANONICAL_FRONTIER_POINTER_PATH,
    CanonicalFrontierPointer,
    FrontierPointerCorruptError,
    POINTER_SCHEMA_VERSION,
    auto_refresh_canonical_frontier_after_dispatch_outcome,
    load_canonical_frontier_pointer_lenient,
    load_canonical_frontier_pointer_strict,
    refresh_canonical_frontier_from_local_state,
    refresh_canonical_frontier_from_upstream_leaderboard,
    write_canonical_frontier_pointer_locked,
)


# ─────────────────────────────────────────────────────────────────────────
# Schema + dataclass tests
# ─────────────────────────────────────────────────────────────────────────


def test_pointer_schema_version_is_canonical() -> None:
    """Schema version literal is the canonical v1 string."""

    assert POINTER_SCHEMA_VERSION == "canonical_frontier_pointer_v1_20260519"


def test_anchor_record_dataclass_round_trip() -> None:
    """AnchorRecord as_dict/from_dict round-trips losslessly."""

    anchor = AnchorRecord(
        score=0.1920513169,
        axis="contest_cpu",
        archive_sha256="6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf",
        lane_id="lane_pr101_frame_exploit_selector_fec6",
        hardware_substrate="linux_x86_64_cpu",
        measured_at_utc="2026-05-15T02:01:39.355227+00:00",
        evidence_grade="[contest-CPU]",
        source_path=".omx/state/continual_learning_posterior.json",
    )
    payload = anchor.as_dict()
    restored = AnchorRecord.from_dict(payload)
    assert restored.score == anchor.score
    assert restored.archive_sha256 == anchor.archive_sha256
    assert restored.evidence_grade == anchor.evidence_grade


def test_canonical_frontier_pointer_dataclass_round_trip() -> None:
    """CanonicalFrontierPointer as_dict/from_dict round-trips."""

    anchor = AnchorRecord(
        score=0.19205,
        axis="contest_cpu",
        archive_sha256="6bae0201" + "0" * 56,
        lane_id="lane_test",
        hardware_substrate="linux_x86_64_cpu",
        measured_at_utc="2026-05-15T00:00:00+00:00",
        evidence_grade="[contest-CPU]",
    )
    pointer = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=anchor,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=None,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc="2026-05-19T00:00:00+00:00",
        auto_update_on_dispatch_completion=True,
        pointer_refresh_command="tools/refresh_canonical_frontier.py",
        refresh_provenance={"refresh_kind": "local_state"},
    )
    restored = CanonicalFrontierPointer.from_dict(pointer.as_dict())
    assert restored.schema_version == pointer.schema_version
    assert restored.our_local_frontier_contest_cpu is not None
    assert restored.our_local_frontier_contest_cpu.score == anchor.score
    assert restored.our_local_frontier_contest_cuda is None


def test_is_stale_detects_old_pointer() -> None:
    """is_stale returns True for pointers >24h old."""

    pointer = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=None,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=None,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc="2020-01-01T00:00:00+00:00",  # ancient
        auto_update_on_dispatch_completion=True,
        pointer_refresh_command="x",
        refresh_provenance={},
    )
    assert pointer.is_stale() is True


def test_is_stale_returns_false_for_fresh_pointer() -> None:
    """is_stale returns False for pointers <24h old."""

    from datetime import datetime, timezone

    pointer = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=None,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=None,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc=datetime.now(timezone.utc).isoformat(),
        auto_update_on_dispatch_completion=True,
        pointer_refresh_command="x",
        refresh_provenance={},
    )
    assert pointer.is_stale() is False


def test_is_stale_handles_malformed_timestamp() -> None:
    """is_stale returns True for malformed last_refreshed_utc."""

    pointer = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=None,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=None,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc="not-a-timestamp",
        auto_update_on_dispatch_completion=True,
        pointer_refresh_command="x",
        refresh_provenance={},
    )
    assert pointer.is_stale() is True


# ─────────────────────────────────────────────────────────────────────────
# Persistence helpers (fcntl-locked atomic write + strict load)
# ─────────────────────────────────────────────────────────────────────────


def _make_minimal_pointer(refreshed_utc: str = "2026-05-19T00:00:00+00:00") -> CanonicalFrontierPointer:
    return CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=None,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=None,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc=refreshed_utc,
        auto_update_on_dispatch_completion=True,
        pointer_refresh_command="x",
        refresh_provenance={},
    )


def test_write_canonical_frontier_pointer_locked_atomic(tmp_path: Path) -> None:
    """Write produces a readable JSON file at the canonical path."""

    pointer = _make_minimal_pointer()
    written_path = write_canonical_frontier_pointer_locked(pointer, repo_root=tmp_path)
    assert written_path.is_file()
    data = json.loads(written_path.read_text())
    assert data["schema_version"] == POINTER_SCHEMA_VERSION


def test_write_creates_state_dir_if_missing(tmp_path: Path) -> None:
    """Writer creates parent dir on first use."""

    pointer = _make_minimal_pointer()
    write_canonical_frontier_pointer_locked(pointer, repo_root=tmp_path)
    assert (tmp_path / ".omx" / "state").is_dir()


def test_write_no_tmp_leakage(tmp_path: Path) -> None:
    """Atomic write should leave no stale .tmp.* files after success."""

    pointer = _make_minimal_pointer()
    write_canonical_frontier_pointer_locked(pointer, repo_root=tmp_path)
    state_dir = tmp_path / ".omx" / "state"
    tmp_files = list(state_dir.glob("canonical_frontier_pointer.json.tmp.*"))
    assert tmp_files == []


def test_load_strict_raises_on_missing_file(tmp_path: Path) -> None:
    """Strict load raises FrontierPointerCorruptError on missing file."""

    with pytest.raises(FrontierPointerCorruptError, match="missing"):
        load_canonical_frontier_pointer_strict(repo_root=tmp_path)


def test_load_strict_raises_on_malformed_json(tmp_path: Path) -> None:
    """Strict load raises on corrupted JSON content."""

    target = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    target.parent.mkdir(parents=True)
    target.write_text("{not valid json")
    with pytest.raises(FrontierPointerCorruptError, match="json parse"):
        load_canonical_frontier_pointer_strict(repo_root=tmp_path)


def test_load_strict_raises_on_non_dict_root(tmp_path: Path) -> None:
    """Strict load refuses non-object root."""

    target = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    target.parent.mkdir(parents=True)
    target.write_text('["not", "a", "dict"]')
    with pytest.raises(FrontierPointerCorruptError, match="must be object"):
        load_canonical_frontier_pointer_strict(repo_root=tmp_path)


def test_load_lenient_returns_none_on_missing(tmp_path: Path) -> None:
    """Lenient load returns None on missing file."""

    assert load_canonical_frontier_pointer_lenient(repo_root=tmp_path) is None


def test_load_lenient_returns_none_on_corrupt(tmp_path: Path) -> None:
    """Lenient load returns None on parse failure."""

    target = tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json"
    target.parent.mkdir(parents=True)
    target.write_text("garbage")
    assert load_canonical_frontier_pointer_lenient(repo_root=tmp_path) is None


def test_write_then_load_round_trip(tmp_path: Path) -> None:
    """Write + load preserves all fields."""

    anchor = AnchorRecord(
        score=0.19205,
        axis="contest_cpu",
        archive_sha256="6bae0201" + "0" * 56,
        lane_id="lane_test",
        hardware_substrate="linux_x86_64_cpu",
        measured_at_utc="2026-05-15T00:00:00+00:00",
        evidence_grade="[contest-CPU]",
    )
    original = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=anchor,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=108,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc="2026-05-19T00:00:00+00:00",
        auto_update_on_dispatch_completion=False,
        pointer_refresh_command="x",
        refresh_provenance={"refresh_kind": "test"},
    )
    write_canonical_frontier_pointer_locked(original, repo_root=tmp_path)
    loaded = load_canonical_frontier_pointer_strict(repo_root=tmp_path)
    assert loaded.our_local_frontier_contest_cpu is not None
    assert loaded.our_local_frontier_contest_cpu.score == anchor.score
    assert loaded.submitted_pr_number_for_current_frontier == 108
    assert loaded.auto_update_on_dispatch_completion is False


# ─────────────────────────────────────────────────────────────────────────
# Refresh from local state
# ─────────────────────────────────────────────────────────────────────────


REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def test_refresh_from_local_state_writes_pointer(tmp_path: Path) -> None:
    """Refresh from local state writes the pointer when state is empty."""

    pointer = refresh_canonical_frontier_from_local_state(repo_root=tmp_path)
    assert pointer.schema_version == POINTER_SCHEMA_VERSION
    # Empty repo: no anchors expected.
    assert pointer.our_local_frontier_contest_cpu is None
    assert (tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json").is_file()


def test_refresh_from_real_repo_finds_canonical_anchors() -> None:
    """Refresh against real repo discovers the 2026-05-15 anchors."""

    # Use a temp pointer path so we don't clobber the live repo state.
    # The refresher writes to the canonical path; for live-repo test we
    # just call write=False to avoid mutating the committed pointer.
    pointer = refresh_canonical_frontier_from_local_state(
        repo_root=REPO_ROOT, write=False
    )
    cpu = pointer.our_local_frontier_contest_cpu
    cuda = pointer.our_local_frontier_contest_cuda
    assert cpu is not None, "expected contest-CPU anchor in real repo"
    assert cuda is not None, "expected contest-CUDA anchor in real repo"
    # Canonical anchors per Catalog #316.
    assert cpu.score == pytest.approx(0.1920513169, rel=1e-6)
    assert cpu.archive_sha256.startswith("6bae0201fb08")
    assert cpu.hardware_substrate == "linux_x86_64_cpu"
    assert cuda.score == pytest.approx(0.20533002902, rel=1e-6)
    assert cuda.archive_sha256.startswith("9cb989cef519")


def test_refresh_preserves_pr_number_across_calls(tmp_path: Path) -> None:
    """Refresh preserves submitted_pr_number_for_current_frontier from prior pointer."""

    first = refresh_canonical_frontier_from_local_state(
        repo_root=tmp_path,
        submitted_pr_number_for_current_frontier=108,
    )
    assert first.submitted_pr_number_for_current_frontier == 108
    # Re-refresh without supplying PR number: should preserve from prior.
    second = refresh_canonical_frontier_from_local_state(repo_root=tmp_path)
    assert second.submitted_pr_number_for_current_frontier == 108


def test_refresh_no_write_does_not_persist(tmp_path: Path) -> None:
    """When write=False, no pointer file is created."""

    pointer = refresh_canonical_frontier_from_local_state(
        repo_root=tmp_path, write=False
    )
    assert pointer is not None
    assert not (tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json").is_file()


# ─────────────────────────────────────────────────────────────────────────
# Upstream leaderboard fetch (with injected fetcher)
# ─────────────────────────────────────────────────────────────────────────


def test_refresh_upstream_with_ok_snapshot(tmp_path: Path) -> None:
    """Upstream refresh records OK snapshot when fetcher succeeds."""

    def fake_fetcher(*, timeout_sec: int = 30) -> dict:
        return {
            "source": "test",
            "url": "test://leaderboard",
            "fetched_at_utc": "2026-05-19T00:00:00+00:00",
            "fetch_status": "ok",
            "pulls": [{"number": 101, "title": "PR101 GOLD", "state": "closed"}],
        }

    pointer = refresh_canonical_frontier_from_upstream_leaderboard(
        repo_root=tmp_path, fetcher=fake_fetcher
    )
    snap = pointer.upstream_leaderboard_snapshot
    assert isinstance(snap, dict)
    assert snap.get("fetch_status") == "ok"
    assert isinstance(snap.get("pulls"), list)
    assert len(snap.get("pulls")) == 1


def test_refresh_upstream_graceful_network_failure(tmp_path: Path) -> None:
    """Upstream refresh records failure status when fetcher fails."""

    def failing_fetcher(*, timeout_sec: int = 30) -> dict:
        return {
            "source": "test",
            "url": "test://leaderboard",
            "fetched_at_utc": "2026-05-19T00:00:00+00:00",
            "fetch_status": "network_failure",
            "fetch_error": "connection refused",
            "pulls": [],
        }

    pointer = refresh_canonical_frontier_from_upstream_leaderboard(
        repo_root=tmp_path, fetcher=failing_fetcher
    )
    snap = pointer.upstream_leaderboard_snapshot
    assert isinstance(snap, dict)
    assert snap.get("fetch_status") == "network_failure"
    assert "connection refused" in str(snap.get("fetch_error", ""))


def test_refresh_upstream_preserves_cached_snapshot_on_failure(tmp_path: Path) -> None:
    """On network failure, prior cached snapshot is preserved."""

    def ok_fetcher(*, timeout_sec: int = 30) -> dict:
        return {
            "source": "test", "url": "x", "fetched_at_utc": "2026-05-19T00:00:00Z",
            "fetch_status": "ok",
            "pulls": [{"number": 100, "title": "first", "state": "closed"}],
        }

    def failing_fetcher(*, timeout_sec: int = 30) -> dict:
        return {
            "source": "test", "url": "x", "fetched_at_utc": "2026-05-19T01:00:00Z",
            "fetch_status": "network_failure",
            "fetch_error": "timeout",
            "pulls": [],
        }

    # First populate with OK fetch.
    refresh_canonical_frontier_from_upstream_leaderboard(
        repo_root=tmp_path, fetcher=ok_fetcher
    )
    # Then trigger failure: should preserve cached.
    pointer = refresh_canonical_frontier_from_upstream_leaderboard(
        repo_root=tmp_path, fetcher=failing_fetcher
    )
    snap = pointer.upstream_leaderboard_snapshot
    assert snap.get("fetch_status") == "network_failure"
    cached = snap.get("cached_snapshot")
    assert cached is not None
    assert cached.get("pulls")[0]["number"] == 100


# ─────────────────────────────────────────────────────────────────────────
# DX auto-update hook
# ─────────────────────────────────────────────────────────────────────────


def test_auto_refresh_fires_on_harvested_outcome(tmp_path: Path) -> None:
    """auto_refresh_canonical_frontier_after_dispatch_outcome fires on harvested + numeric score."""

    result = auto_refresh_canonical_frontier_after_dispatch_outcome(
        status="harvested",
        score=0.193,
        score_axis="contest_cuda",
        archive_sha256="abc123" + "0" * 58,
        repo_root=tmp_path,
    )
    assert result is not None
    assert (tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json").is_file()


def test_auto_refresh_skips_non_harvested(tmp_path: Path) -> None:
    """auto_refresh returns None for non-harvested status."""

    for status in ("failed", "stale", "dispatched"):
        result = auto_refresh_canonical_frontier_after_dispatch_outcome(
            status=status,
            score=0.193,
            score_axis="contest_cuda",
            repo_root=tmp_path,
        )
        assert result is None, f"expected None for status={status}"


def test_auto_refresh_skips_none_score(tmp_path: Path) -> None:
    """auto_refresh returns None when score is None."""

    result = auto_refresh_canonical_frontier_after_dispatch_outcome(
        status="harvested",
        score=None,
        repo_root=tmp_path,
    )
    assert result is None


def test_auto_refresh_skips_nan_score(tmp_path: Path) -> None:
    """auto_refresh returns None for NaN score."""

    result = auto_refresh_canonical_frontier_after_dispatch_outcome(
        status="harvested",
        score=float("nan"),
        repo_root=tmp_path,
    )
    assert result is None


def test_auto_refresh_respects_disabled_flag(tmp_path: Path) -> None:
    """auto_refresh returns None when prior pointer has auto_update=False."""

    # First, write a pointer with auto_update_on_dispatch_completion=False.
    pointer = CanonicalFrontierPointer(
        schema_version=POINTER_SCHEMA_VERSION,
        our_local_frontier_contest_cpu=None,
        our_local_frontier_contest_cuda=None,
        submitted_pr_number_for_current_frontier=None,
        upstream_leaderboard_snapshot=None,
        upstream_leaderboard_snapshot_at_utc=None,
        last_refreshed_utc="2026-05-19T00:00:00+00:00",
        auto_update_on_dispatch_completion=False,  # disabled
        pointer_refresh_command="x",
        refresh_provenance={},
    )
    write_canonical_frontier_pointer_locked(pointer, repo_root=tmp_path)

    result = auto_refresh_canonical_frontier_after_dispatch_outcome(
        status="harvested",
        score=0.193,
        repo_root=tmp_path,
    )
    assert result is None


# ─────────────────────────────────────────────────────────────────────────
# Sister Catalog #131 sister discipline
# ─────────────────────────────────────────────────────────────────────────


def test_canonical_pointer_path_registered_in_shared_state_markers() -> None:
    """Catalog #131 sister: pointer path must be registered as shared state."""

    from tac.preflight import _SHARED_STATE_PATH_MARKERS

    assert "canonical_frontier_pointer" in _SHARED_STATE_PATH_MARKERS
    assert "CANONICAL_FRONTIER_POINTER_PATH" in _SHARED_STATE_PATH_MARKERS


def test_canonical_pointer_helpers_registered_in_canonical_helper_tokens() -> None:
    """Catalog #131 sister: pointer helpers must be in canonical helper allowlist."""

    from tac.preflight import _BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS

    assert "write_canonical_frontier_pointer_locked" in _BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS
    assert "auto_refresh_canonical_frontier_after_dispatch_outcome" in _BARE_WRITE_CANONICAL_HELPER_CALL_TOKENS


def test_canonical_pointer_module_exempt_from_bare_write_gate() -> None:
    """Catalog #131 sister: canonical helper file is on exemption list."""

    from tac.preflight import _BARE_WRITE_CANONICAL_HELPERS

    assert "src/tac/canonical_frontier_pointer.py" in _BARE_WRITE_CANONICAL_HELPERS
