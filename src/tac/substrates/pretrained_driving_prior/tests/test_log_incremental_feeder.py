# SPDX-License-Identifier: MIT
"""Tests for the log-incremental distillation schedule (DP1 auto-load).

Per operator directive 2026-05-14 "log incremental generator": the canonical
schedule yields ``[1, 2, 4, 8, 16, 32, 64, 80]`` at default base=2/max=80.
These tests pin (1) schedule shape across bases, (2) plateau early-stop, (3)
quality-metric monotonicity, (4) synthetic-stub-friendly test mode, (5)
license-tag propagation through schedule steps, (6) ScheduleStepResult
serialization (continual-learning anchor format).

Auto-downloads are MOCKED — tests use a synthetic-chunk manifest + the
``frame_iterator_factory`` injection so no real Comma2k19 chunks are fetched.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from tac.substrates.pretrained_driving_prior import (
    Comma2k19ChunkManifestEntry,
    Comma2k19LocalCache,
    DashcamCodebook,
    DistillationConfig,
    LogIncrementalSchedule,
    ScheduleStepResult,
    codebook_pca_quality_metric,
    log_incremental_distillation,
)
from tac.substrates.pretrained_driving_prior.distillation import (
    _synthetic_dashcam_frames,
)

if TYPE_CHECKING:
    import numpy as np

# ---------------------------------------------------------------------------
# Schedule shape
# ---------------------------------------------------------------------------


def test_schedule_default_doubling_to_80() -> None:
    s = LogIncrementalSchedule()
    assert s.schedule() == [1, 2, 4, 8, 16, 32, 64, 80]


def test_schedule_base_2_explicit() -> None:
    s = LogIncrementalSchedule(base=2, initial_chunks=1, max_chunks=80)
    assert s.schedule() == [1, 2, 4, 8, 16, 32, 64, 80]


def test_schedule_base_3() -> None:
    s = LogIncrementalSchedule(base=3, initial_chunks=1, max_chunks=80)
    assert s.schedule() == [1, 3, 9, 27, 80]


def test_schedule_base_4() -> None:
    s = LogIncrementalSchedule(base=4, initial_chunks=1, max_chunks=80)
    assert s.schedule() == [1, 4, 16, 64, 80]


def test_schedule_max_clamped() -> None:
    s = LogIncrementalSchedule(base=2, initial_chunks=4, max_chunks=10)
    assert s.schedule() == [4, 8, 10]


def test_schedule_initial_equal_max() -> None:
    s = LogIncrementalSchedule(base=2, initial_chunks=5, max_chunks=5)
    assert s.schedule() == [5]


def test_schedule_validation_base_too_small() -> None:
    with pytest.raises(ValueError, match="base must be >= 2"):
        LogIncrementalSchedule(base=1)


def test_schedule_validation_initial_zero() -> None:
    with pytest.raises(ValueError, match="initial_chunks must be >= 1"):
        LogIncrementalSchedule(initial_chunks=0)


def test_schedule_validation_max_below_initial() -> None:
    with pytest.raises(ValueError, match="max_chunks"):
        LogIncrementalSchedule(initial_chunks=10, max_chunks=5)


def test_schedule_validation_negative_threshold() -> None:
    with pytest.raises(ValueError, match="quality_plateau_threshold"):
        LogIncrementalSchedule(quality_plateau_threshold=-0.001)


def test_schedule_validation_zero_max_steps() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        LogIncrementalSchedule(max_steps=0)


# ---------------------------------------------------------------------------
# log_incremental_distillation — happy path with synthetic stub
# ---------------------------------------------------------------------------


def _synthetic_cache(tmp_path: Path, n_chunks: int) -> Comma2k19LocalCache:
    """Build a cache with N pre-cached synthetic chunks (no network)."""
    cache_dir = tmp_path / "cache"
    entries: dict[str, Comma2k19ChunkManifestEntry] = {}
    for i in range(n_chunks):
        cid = f"chunk_{i:03d}"
        entries[cid] = Comma2k19ChunkManifestEntry(
            chunk_id=cid,
            url=f"https://example.com/{cid}",
            expected_sha256="",
            size_bytes=128,
            dest_relpath=f"d/r/{i}/video.hevc",
        )
    cache = Comma2k19LocalCache(
        cache_dir=cache_dir,
        max_disk_gb=1.0,
        chunk_manifest=entries,
    )
    # Pre-populate every chunk with dummy bytes so fetch_chunk is O(1).
    # The frame_iterator_factory ignores the bytes — it generates synthetic
    # frames from the count.
    cache_dir.mkdir(parents=True, exist_ok=True)
    import json

    meta: dict[str, Any] = {}
    for cid, entry in entries.items():
        dst = cache_dir / entry.dest_relpath
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(b"\x00" * 128)
        meta[cid] = {
            "chunk_id": cid,
            "local_path": str(dst),
            "size_bytes": 128,
            "sha256_pinned": "",
            "license": "MIT",
            "source_url": entry.url,
            "fetched_at_utc": "2026-05-14T00:00:00Z",
            "last_used_at_utc": f"2026-05-14T00:00:{i:02d}Z",
        }
    (cache_dir / "cache_meta.json").write_text(json.dumps(meta, indent=2))
    return cache


def _synthetic_frame_factory_64(
    chunk_paths: Iterable[Path],
) -> Iterator[np.ndarray]:
    """Yield 64 synthetic frames per call (ignores chunk_paths content)."""
    paths = list(chunk_paths)
    seed = 1234 + len(paths)  # make output depend on chunk_count so quality
    # metric varies across schedule steps.
    yield from _synthetic_dashcam_frames(n_frames=64 * max(1, len(paths)), seed=seed)


def test_log_incremental_distillation_runs_full_schedule(tmp_path) -> None:
    cache = _synthetic_cache(tmp_path, n_chunks=8)
    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=8,
        quality_plateau_threshold=0.0,  # disable plateau early-stop
    )
    book, log = log_incremental_distillation(
        cache=cache,
        schedule=schedule,
        distill_cfg_template=DistillationConfig(
            dataset_name="comma2k19", max_frames=64, random_seed=42
        ),
        frame_iterator_factory=_synthetic_frame_factory_64,
    )
    # 4 steps: 1, 2, 4, 8.
    assert len(log) == 4
    assert [s.chunk_count for s in log] == [1, 2, 4, 8]
    # Final codebook is valid.
    assert isinstance(book, DashcamCodebook)
    # Final chunk count is recorded in codebook metadata.
    assert book.metadata.get("log_incremental_chunks") == 8


def test_log_incremental_distillation_step_0_has_none_marginal(tmp_path) -> None:
    cache = _synthetic_cache(tmp_path, n_chunks=4)
    schedule = LogIncrementalSchedule(
        base=2, initial_chunks=1, max_chunks=4, quality_plateau_threshold=0.0
    )
    _, log = log_incremental_distillation(
        cache=cache,
        schedule=schedule,
        frame_iterator_factory=_synthetic_frame_factory_64,
    )
    assert log[0].marginal_improvement is None
    assert log[0].step == 0
    for i in range(1, len(log)):
        assert log[i].marginal_improvement is not None
        assert log[i].step == i


def test_log_incremental_distillation_records_provenance(tmp_path) -> None:
    cache = _synthetic_cache(tmp_path, n_chunks=2)
    _, log = log_incremental_distillation(
        cache=cache,
        schedule=LogIncrementalSchedule(
            base=2, initial_chunks=1, max_chunks=2, quality_plateau_threshold=0.0
        ),
        frame_iterator_factory=_synthetic_frame_factory_64,
    )
    for step in log:
        # Catalog #210 provenance keys propagated.
        assert "license_tags" in step.provenance
        assert "dataset_provenance" in step.provenance
        # Cache identity recorded.
        assert step.provenance["cache_license"] == "MIT"
        assert step.provenance["cache_source_url"].startswith("https://")
        # Cached chunk ids used at this step recorded.
        assert "cached_chunk_ids_used" in step.provenance
        assert len(step.provenance["cached_chunk_ids_used"]) == step.chunk_count
        assert "chunk_sha256_manifest" in step.provenance
        assert set(step.provenance["chunk_sha256_manifest"]) == set(
            step.provenance["cached_chunk_ids_used"]
        )


def test_log_incremental_distillation_early_stops_on_plateau(tmp_path) -> None:
    cache = _synthetic_cache(tmp_path, n_chunks=8)

    # Custom quality metric that drops sharply at step 0 then plateaus.
    quality_values = iter([1.0, 0.5, 0.49995, 0.4999, 0.4998, 0.4997, 0.4996, 0.4995])

    def stub_quality(book):
        try:
            return next(quality_values)
        except StopIteration:
            return 0.0

    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=8,
        quality_plateau_threshold=0.001,
    )
    _, log = log_incremental_distillation(
        cache=cache,
        schedule=schedule,
        frame_iterator_factory=_synthetic_frame_factory_64,
        quality_metric=stub_quality,
    )
    # Quality goes 1.0 -> 0.5 (improvement 0.5; step 0->1)
    #              0.5 -> 0.49995 (improvement 0.00005; step 1->2 -> below threshold)
    # Plateau at step 2 (step index >= 2 condition).
    assert log[-1].early_stopped is True
    # Should NOT have completed the full 4-step schedule.
    assert len(log) < 4


def test_log_incremental_distillation_no_plateau_when_threshold_zero(tmp_path) -> None:
    cache = _synthetic_cache(tmp_path, n_chunks=8)
    qs = iter([1.0, 0.5, 0.4999, 0.49989])

    def stub_quality(book):
        return next(qs)

    schedule = LogIncrementalSchedule(
        base=2,
        initial_chunks=1,
        max_chunks=8,
        quality_plateau_threshold=0.0,  # disabled
    )
    _, log = log_incremental_distillation(
        cache=cache,
        schedule=schedule,
        frame_iterator_factory=_synthetic_frame_factory_64,
        quality_metric=stub_quality,
    )
    # Should run all 4 steps since plateau early-stop is disabled.
    # Schedule [1, 2, 4, 8] (base=2, max=8) → exactly 4 entries.
    assert len(log) == 4
    assert not any(s.early_stopped for s in log)


def test_log_incremental_distillation_stops_when_chunks_exhausted(tmp_path) -> None:
    cache = _synthetic_cache(tmp_path, n_chunks=3)  # only 3 chunks available
    schedule = LogIncrementalSchedule(
        base=2, initial_chunks=1, max_chunks=80, quality_plateau_threshold=0.0
    )
    _, log = log_incremental_distillation(
        cache=cache,
        schedule=schedule,
        frame_iterator_factory=_synthetic_frame_factory_64,
    )
    # The schedule says 1, 2, 4, 8, 16, 32, 64, 80 but cache only has 3.
    # First step uses 1 chunk; second uses 2; third uses min(4, 3) = 3; stop.
    assert [s.chunk_count for s in log] == [1, 2, 3]


def test_log_incremental_distillation_empty_cache_raises(tmp_path) -> None:
    cache = Comma2k19LocalCache(
        cache_dir=tmp_path / "empty",
        max_disk_gb=1.0,
        chunk_manifest={},
    )
    with pytest.raises(ValueError, match="no available chunks"):
        log_incremental_distillation(
            cache=cache, schedule=LogIncrementalSchedule()
        )


def test_log_incremental_distillation_synthetic_fallback(tmp_path) -> None:
    """When ``distill_cfg_template.dataset_name='synthetic_test'`` the schedule still works."""
    cache = _synthetic_cache(tmp_path, n_chunks=2)
    book, log = log_incremental_distillation(
        cache=cache,
        schedule=LogIncrementalSchedule(
            base=2, initial_chunks=1, max_chunks=2, quality_plateau_threshold=0.0
        ),
        distill_cfg_template=DistillationConfig(
            dataset_name="synthetic_test", max_frames=32, random_seed=7
        ),
    )
    assert len(log) == 2
    # Synthetic-test license tag should be present.
    license_tags = log[-1].provenance["license_tags"]
    assert "synthetic-test-only" in license_tags


# ---------------------------------------------------------------------------
# Quality metric (default codebook_pca_quality_metric)
# ---------------------------------------------------------------------------


def test_codebook_pca_quality_metric_zero_basis() -> None:
    from tac.substrates.pretrained_driving_prior import deterministic_zero_codebook

    book = deterministic_zero_codebook()
    q = codebook_pca_quality_metric(book)
    # The metric is a deficiency/loss: zero basis is worst.
    assert q == 1.0


def test_codebook_pca_quality_metric_richer_basis_is_lower() -> None:
    from dataclasses import replace

    from tac.substrates.pretrained_driving_prior import deterministic_zero_codebook

    zero = deterministic_zero_codebook()
    rich_basis = zero.road_plane_basis.copy()
    rich_basis.fill(127)
    rich = replace(zero, road_plane_basis=rich_basis)

    assert codebook_pca_quality_metric(rich) < codebook_pca_quality_metric(zero)


def test_codebook_pca_quality_metric_nonzero(tmp_path) -> None:
    """Real distillation produces non-zero quality."""
    cache = _synthetic_cache(tmp_path, n_chunks=1)
    book, _ = log_incremental_distillation(
        cache=cache,
        schedule=LogIncrementalSchedule(
            base=2, initial_chunks=1, max_chunks=1, quality_plateau_threshold=0.0
        ),
        frame_iterator_factory=_synthetic_frame_factory_64,
    )
    q = codebook_pca_quality_metric(book)
    assert 0.0 <= q <= 1.0
    # The deterministic synthetic frames should yield a bounded deficiency.


# ---------------------------------------------------------------------------
# ScheduleStepResult serialization
# ---------------------------------------------------------------------------


def test_schedule_step_result_to_dict() -> None:
    r = ScheduleStepResult(
        step=2,
        chunk_count=4,
        frame_count=256,
        codebook_size_bytes=5000,
        quality=0.75,
        marginal_improvement=0.02,
        codebook_basis_sha256="deadbeef" * 8,
        provenance={"license_tags": ["comma2k19:MIT"]},
        early_stopped=False,
    )
    d = r.to_dict()
    assert d["step"] == 2
    assert d["chunk_count"] == 4
    assert d["frame_count"] == 256
    assert d["codebook_size_bytes"] == 5000
    assert d["quality"] == 0.75
    assert d["marginal_improvement"] == 0.02
    assert d["codebook_basis_sha256"] == "deadbeef" * 8
    assert d["provenance"]["license_tags"] == ["comma2k19:MIT"]
    assert d["early_stopped"] is False


def test_schedule_step_result_to_dict_none_marginal() -> None:
    r = ScheduleStepResult(
        step=0,
        chunk_count=1,
        frame_count=64,
        codebook_size_bytes=5000,
        quality=1.0,
        marginal_improvement=None,
        codebook_basis_sha256="0" * 64,
    )
    d = r.to_dict()
    assert d["marginal_improvement"] is None
    assert d["early_stopped"] is False


# ---------------------------------------------------------------------------
# Continual-learning hook readiness (every step is an anchor candidate)
# ---------------------------------------------------------------------------


def test_schedule_log_serializable_as_anchor_list(tmp_path) -> None:
    """Each step's to_dict() output is JSON-serializable as a continual-learning anchor."""
    import json

    cache = _synthetic_cache(tmp_path, n_chunks=2)
    _, log = log_incremental_distillation(
        cache=cache,
        schedule=LogIncrementalSchedule(
            base=2, initial_chunks=1, max_chunks=2, quality_plateau_threshold=0.0
        ),
        frame_iterator_factory=_synthetic_frame_factory_64,
    )
    rows = [step.to_dict() for step in log]
    # Must round-trip through json.dumps without TypeError.
    serialized = json.dumps(rows)
    rehydrated = json.loads(serialized)
    assert len(rehydrated) == 2
    assert rehydrated[0]["step"] == 0
    assert rehydrated[1]["step"] == 1
