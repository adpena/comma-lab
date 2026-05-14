# SPDX-License-Identifier: MIT
"""Log-incremental codebook distillation schedule.

Operator directive 2026-05-14 (verbatim): *"you can download the dataset locally
and configure the chunks dir to be fed dynamically cached with a log incremental
generator"*.

This module wraps :func:`distill_codebook` in an exponential-growth chunk
schedule (default ``[1, 2, 4, 8, 16, 32, 64, 80]``) with marginal-quality
plateau detection. The motivation per the Contrarian Round 1 review of the
DP1 Phase 2 hardening v2 council
(``.omx/research/dp1_phase_2_hardening_v2_council_20260514.md``) is that
the SegNet + PoseNet contest scorers were ALREADY trained on driving data,
so the dashcam prior's score-relevant signal saturates at a small chunk
count. The log-incremental schedule:

1. Distills a tiny codebook (1 chunk) first to verify the pipeline.
2. Doubles the chunk count until either the schedule maximum or the
   marginal-quality plateau threshold is reached.
3. Stops early when adding 2x chunks no longer improves the quality
   metric by ``quality_plateau_threshold`` — typically PCA reconstruction
   error or codebook entropy.

Why log-incremental:

- **Cheap early evaluation**: see if distillation produces a non-trivial
  codebook at all with 1 chunk before downloading 80.
- **Diminishing-returns aware**: per the Contrarian's "scorer internalizes
  the prior" verdict (predicted Δ contest-CPU bounded at -0.012), the
  marginal score gain from chunk 32 → 80 is likely below measurement noise.
- **Cost-aware**: smaller schedule → less Modal/Vast.ai wall-clock + smaller
  PCA design matrix → faster distillation. Scenario B in the
  hardening-v2 cost matrix runs Comma2k19 medium codebook at 10k-50k
  frames; a log-incremental schedule can hit that range in 5-7 steps.
- **Reproducibility**: deterministic ordering of cumulative chunks
  (``cache.list_available_chunks()`` is sorted) → seed-stable.

The schedule is also a CONTINUAL-LEARNING primitive: each schedule step's
codebook + measured quality is an empirical anchor we can feed to
``tac.continual_learning.posterior_update_locked`` (Catalog #128) to
build a per-substrate "frames vs ΔS" posterior across many dispatches.

**License attribution**: every codebook emitted by the log-incremental
schedule carries the cache's license tag (``"MIT"`` for Comma2k19) per
:class:`Comma2k19LocalCache`. Catalog #210 STRICT preflight enforces that
this attribution survives all the way into archive packaging.

**Per CLAUDE.md "Subagent coherence-by-default"**: this module is the
``cathedral_autopilot dispatch hook`` AND ``continual-learning posterior
update`` wire-in for DP1 auto-load — the log-incremental schedule's
schedule-log is exactly the per-step empirical anchor list the autopilot
ranks against. The other 4 wire-in hooks are declared in the lane memory.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np  # noqa: E402

from tac.substrates.pretrained_driving_prior.codebook import (  # noqa: E402
    DashcamCodebook,
)
from tac.substrates.pretrained_driving_prior.distillation import (  # noqa: E402
    DistillationConfig,
    distill_codebook,
)
from tac.substrates.pretrained_driving_prior.local_chunk_cache import (  # noqa: E402
    Comma2k19LocalCache,
)


__all__ = [
    "LogIncrementalSchedule",
    "ScheduleStepResult",
    "log_incremental_distillation",
    "log_incremental_distillation_streaming",
    "codebook_pca_quality_metric",
]


@dataclass(frozen=True)
class LogIncrementalSchedule:
    """Exponential-growth chunk sampling schedule with plateau early-stop.

    Args:
        base: Exponential base. ``2`` (default) yields the doubling
            schedule ``[1, 2, 4, 8, ...]``; ``3`` yields ``[1, 3, 9, ...]``;
            and so on. Must be >= 2.
        initial_chunks: First step's chunk count. Default 1.
        max_chunks: Cap. The schedule stops once cumulative chunk count
            reaches or exceeds this value (the last entry is clamped to
            ``max_chunks``). Default 80 (full Comma2k19 corpus size).
        quality_plateau_threshold: Relative-improvement threshold for the
            early-stop heuristic. If ``(quality_step_n - quality_step_n+1)``
            falls below this and we've done at least 3 steps, the schedule
            stops. Set to 0.0 to disable plateau early-stop.
        max_steps: Hard cap on schedule steps regardless of base/max_chunks.
            Default 16 (covers base=2 schedule up to ~32K chunks; well
            beyond Comma2k19's 80).
    """

    base: int = 2
    initial_chunks: int = 1
    max_chunks: int = 80
    quality_plateau_threshold: float = 0.005
    max_steps: int = 16

    def __post_init__(self) -> None:
        # frozen=True forces us to use object.__setattr__ for any
        # post-init mutation; instead we just validate.
        if self.base < 2:
            raise ValueError(f"base must be >= 2; got {self.base!r}")
        if self.initial_chunks < 1:
            raise ValueError(
                f"initial_chunks must be >= 1; got {self.initial_chunks!r}"
            )
        if self.max_chunks < self.initial_chunks:
            raise ValueError(
                f"max_chunks ({self.max_chunks}) must be >= initial_chunks "
                f"({self.initial_chunks})"
            )
        if self.quality_plateau_threshold < 0.0:
            raise ValueError(
                f"quality_plateau_threshold must be >= 0; got "
                f"{self.quality_plateau_threshold!r}"
            )
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1; got {self.max_steps!r}")

    def schedule(self) -> list[int]:
        """Return the planned chunk-count series.

        Examples:
            ``LogIncrementalSchedule(base=2, initial_chunks=1, max_chunks=80).schedule()``
                ``→ [1, 2, 4, 8, 16, 32, 64, 80]``
            ``LogIncrementalSchedule(base=3, initial_chunks=1, max_chunks=80).schedule()``
                ``→ [1, 3, 9, 27, 80]``
            ``LogIncrementalSchedule(base=2, initial_chunks=4, max_chunks=10).schedule()``
                ``→ [4, 8, 10]``
        """
        series: list[int] = []
        n = int(self.initial_chunks)
        for _ in range(self.max_steps):
            clamped = min(n, self.max_chunks)
            if not series or series[-1] < clamped:
                series.append(clamped)
            if clamped >= self.max_chunks:
                break
            n = max(n + 1, n * self.base)
        return series


@dataclass
class ScheduleStepResult:
    """One step's result inside the log-incremental schedule log.

    Fields:
        step: 0-indexed step number.
        chunk_count: cumulative number of chunks used at this step.
        frame_count: actual frames consumed by the distiller this step.
        codebook_size_bytes: serialized codebook size in bytes.
        quality: the quality metric value at this step (lower = better
            for reconstruction error; higher = better for entropy).
        marginal_improvement: ``quality[step-1] - quality[step]`` for
            step >= 1, ``None`` for step 0.
        codebook_basis_sha256: hash of the codebook's distilled basis bytes.
        provenance: arbitrary per-step extra metadata.
        early_stopped: True iff this step triggered plateau early-stop.
    """

    step: int
    chunk_count: int
    frame_count: int
    codebook_size_bytes: int
    quality: float
    marginal_improvement: float | None
    codebook_basis_sha256: str
    provenance: dict[str, Any] = field(default_factory=dict)
    early_stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "chunk_count": int(self.chunk_count),
            "frame_count": int(self.frame_count),
            "codebook_size_bytes": int(self.codebook_size_bytes),
            "quality": float(self.quality),
            "marginal_improvement": (
                None
                if self.marginal_improvement is None
                else float(self.marginal_improvement)
            ),
            "codebook_basis_sha256": self.codebook_basis_sha256,
            "provenance": dict(self.provenance),
            "early_stopped": bool(self.early_stopped),
        }


def codebook_pca_quality_metric(book: DashcamCodebook) -> float:
    """Default quality metric: codebook's road-plane basis spectral concentration.

    Returns the average per-element magnitude of the int8 road-plane basis.
    Larger magnitude → more concentrated PCA directions → richer prior. The
    metric is normalized to a stable [0, 1] range by dividing by 127 (the
    int8 saturation level). The plateau detector compares the marginal
    improvement of this metric between schedule steps; if it drops below
    ``quality_plateau_threshold`` the schedule stops early.

    NOTE: this is a CHEAP proxy. For Phase 3 wiring the operator can plug
    in a real downstream-loss proxy (e.g. SegNet/PoseNet inference loss
    against the contest video) by passing a custom ``quality_metric`` to
    :func:`log_incremental_distillation`.
    """
    arr = book.road_plane_basis.astype(np.float32)
    if arr.size == 0:
        return 0.0
    mean_abs = float(np.abs(arr).mean())
    return mean_abs / 127.0


def _chunk_id_window(
    chunk_ids: list[str], n_chunks: int
) -> list[str]:
    """Return the first ``n_chunks`` chunk ids in deterministic order.

    The cache's ``list_available_chunks()`` is sorted so the prefix
    ``chunk_ids[:n_chunks]`` is reproducible across runs.
    """
    if n_chunks < 0:
        raise ValueError(f"n_chunks must be >= 0; got {n_chunks!r}")
    return list(chunk_ids[: int(n_chunks)])


def log_incremental_distillation(
    cache: Comma2k19LocalCache,
    schedule: LogIncrementalSchedule,
    *,
    distill_cfg_template: DistillationConfig | None = None,
    quality_metric: Callable[[DashcamCodebook], float] | None = None,
    frame_iterator_factory: Callable[
        [Iterable[Path]], Iterator[np.ndarray]
    ] | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> tuple[DashcamCodebook, list[ScheduleStepResult]]:
    """Run log-incremental distillation. Returns (final_codebook, schedule_log).

    For each step ``s`` in ``schedule.schedule()``:

    1. Fetch the prefix of ``cache.list_available_chunks()[:s]`` (cache hits
       are O(1); only NEW chunks are downloaded).
    2. Build a frame iterator that walks every cached chunk in order
       (``frame_iterator_factory`` lets tests inject a synthetic iterator).
    3. Distill a codebook on the cumulative frames.
    4. Measure ``quality_metric(book)``.
    5. If ``step >= 2`` and the marginal improvement is below
       ``schedule.quality_plateau_threshold``, mark this step as the
       early-stop point and break.

    Args:
        cache: canonical :class:`Comma2k19LocalCache` that owns the chunk
            downloads. Per Catalog #213 ALL chunk auto-download goes
            through this helper.
        schedule: :class:`LogIncrementalSchedule` describing the sampling
            growth + plateau threshold.
        distill_cfg_template: prototype :class:`DistillationConfig`. The
            schedule re-derives a per-step config by adjusting
            ``max_frames`` upward as chunk count grows. Default uses
            ``DistillationConfig(dataset_name='comma2k19')`` if the cache
            knows real chunks, else ``synthetic_test``.
        quality_metric: callable :class:`DashcamCodebook` → float (default
            :func:`codebook_pca_quality_metric`).
        frame_iterator_factory: optional factory that takes a list of
            ``video.hevc`` paths and returns a frame iterator. Default uses
            :class:`Comma2k19FrameIterator` over the cache's
            ``cached_chunks_dir()``.
        extra_provenance: dict merged into each step's ``provenance`` field.

    Returns:
        ``(final_codebook, schedule_log)``: the codebook from the last
        (cumulative) step + the per-step ScheduleStepResult list.

    Raises:
        ValueError: schedule is empty (impossible for valid
            :class:`LogIncrementalSchedule`) or no chunks available.
    """
    quality_metric = quality_metric or codebook_pca_quality_metric
    extra_provenance = dict(extra_provenance or {})
    distill_cfg_template = distill_cfg_template or DistillationConfig(
        dataset_name=(
            "comma2k19" if cache.list_available_chunks() else "synthetic_test"
        ),
    )
    chunk_ids = cache.list_available_chunks()
    planned = schedule.schedule()
    if not planned:
        raise ValueError("log-incremental schedule is empty (impossible)")
    if not chunk_ids:
        raise ValueError(
            f"Comma2k19LocalCache at {cache.cache_dir!r} has no available "
            f"chunks; populate the manifest or pass a cache with at least "
            f"one entry before calling log_incremental_distillation"
        )

    schedule_log: list[ScheduleStepResult] = []
    last_quality: float | None = None
    final_book: DashcamCodebook | None = None
    final_chunk_count = 0

    for step, requested in enumerate(planned):
        usable_chunk_count = min(requested, len(chunk_ids))
        window = _chunk_id_window(chunk_ids, usable_chunk_count)
        # Download all chunks in the window (cache hits are O(1)).
        chunk_paths: list[Path] = []
        for chunk_id in window:
            chunk_paths.append(cache.fetch_chunk(chunk_id))
        # Build frame iterator.
        if frame_iterator_factory is None:
            frames_iter = _default_frame_iterator(
                cache=cache, chunk_paths=chunk_paths, distill_cfg=distill_cfg_template
            )
        else:
            frames_iter = frame_iterator_factory(chunk_paths)
        # Distill.
        cfg = DistillationConfig(
            dataset_name=distill_cfg_template.dataset_name,
            dataset_sha256=distill_cfg_template.dataset_sha256,
            num_road_plane_components=distill_cfg_template.num_road_plane_components,
            num_vehicle_components=distill_cfg_template.num_vehicle_components,
            num_lane_curvature_components=distill_cfg_template.num_lane_curvature_components,
            random_seed=distill_cfg_template.random_seed,
            max_frames=max(distill_cfg_template.max_frames, 32 * usable_chunk_count),
            allow_bdd100k_dataset_images=distill_cfg_template.allow_bdd100k_dataset_images,
        )
        book = distill_codebook(cfg, frames=frames_iter)  # COMMA2K19_LEAKAGE_VERIFIED_OK:frames_iter-came-from-Comma2k19FrameIterator-via-_default_frame_iterator-or-test-injected-factory
        # Measure.
        quality = float(quality_metric(book))
        marginal: float | None = None
        if last_quality is not None:
            marginal = float(last_quality - quality)  # positive = improving (lower is better)
        # Serialize once for size accounting.
        from tac.substrates.pretrained_driving_prior.codebook import serialize_codebook

        book_bytes = serialize_codebook(book)
        # Build the per-step result record.
        provenance_dict: dict[str, Any] = dict(extra_provenance)
        provenance_dict.update(
            {
                "license_tags": list(book.metadata.get("license_tags", [])),
                "dataset_provenance": book.metadata.get("dataset_provenance", ""),
                "schedule_base": int(schedule.base),
                "schedule_max_chunks": int(schedule.max_chunks),
                "schedule_quality_plateau_threshold": float(
                    schedule.quality_plateau_threshold
                ),
                "cache_dir": str(cache.cache_dir),
                "cache_license": cache.DATASET_LICENSE,
                "cache_source_url": cache.CANONICAL_SOURCE_URL,
                "cached_chunk_ids_used": list(window),
            }
        )
        step_result = ScheduleStepResult(
            step=step,
            chunk_count=usable_chunk_count,
            frame_count=int(book.metadata.get("num_frames_used", 0)),
            codebook_size_bytes=len(book_bytes),
            quality=quality,
            marginal_improvement=marginal,
            codebook_basis_sha256=str(book.metadata.get("basis_sha256", "")),
            provenance=provenance_dict,
        )
        schedule_log.append(step_result)
        final_book = book
        final_chunk_count = usable_chunk_count
        # Plateau detection (only meaningful with at least 3 steps).
        if (
            schedule.quality_plateau_threshold > 0.0
            and marginal is not None
            and step >= 2
            and 0.0 <= marginal < schedule.quality_plateau_threshold
        ):
            step_result.early_stopped = True
            logger.info(
                "log_incremental_distillation: plateau early-stop at step %d "
                "(chunks=%d, marginal=%.6f < threshold=%.6f)",
                step,
                usable_chunk_count,
                marginal,
                schedule.quality_plateau_threshold,
            )
            break
        # If we've exhausted available chunks, stop.
        if usable_chunk_count >= len(chunk_ids):
            break
        last_quality = quality

    assert final_book is not None, "schedule produced no codebook (empty plan?)"
    # Note the final cumulative chunk count for downstream wire-in.
    final_book.metadata.setdefault("log_incremental_chunks", final_chunk_count)
    return final_book, schedule_log


def _default_frame_iterator(
    *,
    cache: Comma2k19LocalCache,
    chunk_paths: list[Path],
    distill_cfg: DistillationConfig,
) -> Iterator[np.ndarray]:
    """Default frame iterator: walks the cache's chunks_dir layout.

    Per Catalog #209 the canonical entry to the distiller's frame pipeline
    is :class:`Comma2k19FrameIterator`, which runs the contest-video
    leakage guard before any decode. Using ``cache.cached_chunks_dir()``
    (an actual repo-relative durable path, NOT ``/tmp``) gives the
    iterator a chunks-tree root it can ``rglob('video.hevc')`` over.
    """
    from tac.substrates.pretrained_driving_prior.distillation import (
        Comma2k19FrameIterator,
        _synthetic_dashcam_frames,
    )

    if distill_cfg.dataset_name == "synthetic_test":
        # COMMA2K19_LEAKAGE_VERIFIED_OK:synthetic-mode-does-not-touch-disk
        return iter(
            _synthetic_dashcam_frames(
                n_frames=distill_cfg.max_frames, seed=distill_cfg.random_seed
            )
        )
    # Real chunks already live under cache.cached_chunks_dir().
    # COMMA2K19_LEAKAGE_VERIFIED_OK:routed-via-Comma2k19FrameIterator-which-runs-check_no_contest_video_leakage-internally
    it = Comma2k19FrameIterator(
        chunks_dir=cache.cached_chunks_dir(),
        max_chunks=max(1, len(chunk_paths)),
        max_frames_per_chunk=max(
            32, distill_cfg.max_frames // max(1, len(chunk_paths))
        ),
    )
    return iter(it)


# ===========================================================================
# Streaming-mode log-incremental distillation (operator pivot 2026-05-14)
# ===========================================================================
#
# Operator directive verbatim: "or instead of downloading just configure a
# local streamer and log". The cache-mode function above remains for
# back-compat; the streaming-mode function below is the NEW canonical path.
# It accepts a :class:`Comma2k19LocalStreamer` instead of a cache and drives
# the same log-incremental schedule against streamed (in-RAM, no permanent
# disk) chunks. Both share the schedule + plateau logic — only the chunk
# source differs.


def log_incremental_distillation_streaming(
    streamer: Any,  # Comma2k19LocalStreamer (forward-ref; avoids circular import)
    schedule: LogIncrementalSchedule,
    *,
    chunk_ids: list[str] | None = None,
    distill_cfg_template: DistillationConfig | None = None,
    quality_metric: Callable[[DashcamCodebook], float] | None = None,
    frames_per_chunk: int = 256,
    extra_provenance: dict[str, Any] | None = None,
) -> tuple[DashcamCodebook, list[ScheduleStepResult]]:
    """Streaming-mode log-incremental distillation (operator pivot 2026-05-14).

    Each schedule step streams the next batch of chunks via the streamer
    (no permanent disk cache), distills incrementally on the streamed
    frames, logs the access via JSONL, and discards the in-RAM buffer.

    Args:
        streamer: :class:`Comma2k19LocalStreamer` (or compatible). Provides
            ``list_chunk_ids()`` + ``stream_frames(chunk_id, max_frames)``.
        schedule: :class:`LogIncrementalSchedule` (shared with cache mode).
        chunk_ids: Optional explicit chunk-id list. When None, uses
            ``streamer.list_chunk_ids()``.
        distill_cfg_template: Prototype :class:`DistillationConfig`. The
            schedule re-derives a per-step config by scaling ``max_frames``
            with cumulative chunk count.
        quality_metric: Callable ``DashcamCodebook → float`` (default
            :func:`codebook_pca_quality_metric`).
        frames_per_chunk: How many frames to decode per streamed chunk.
        extra_provenance: Dict merged into each step's ``provenance`` field.

    Returns:
        ``(final_codebook, schedule_log)``: codebook from the last
        cumulative step + per-step :class:`ScheduleStepResult` list.

    Raises:
        ValueError: empty schedule or no chunk ids available.
    """
    quality_metric = quality_metric or codebook_pca_quality_metric
    extra_provenance = dict(extra_provenance or {})
    chunk_ids = list(chunk_ids if chunk_ids is not None else streamer.list_chunk_ids())
    if not chunk_ids:
        raise ValueError(
            "Comma2k19LocalStreamer has no chunk ids; pass an explicit "
            "chunk_ids list or populate the streamer's dataset_sha256_manifest"
        )
    planned = schedule.schedule()
    if not planned:
        raise ValueError("log-incremental schedule is empty (impossible)")
    distill_cfg_template = distill_cfg_template or DistillationConfig(
        dataset_name=(
            "synthetic_test" if streamer.is_synthetic else "comma2k19"
        ),
    )

    schedule_log: list[ScheduleStepResult] = []
    last_quality: float | None = None
    final_book: DashcamCodebook | None = None
    final_chunk_count = 0

    for step, requested in enumerate(planned):
        usable_chunk_count = min(requested, len(chunk_ids))
        window = list(chunk_ids[:usable_chunk_count])
        # Stream every chunk in the window; collect frames in RAM only for
        # the duration of this step. The streamer's JSONL log records every
        # chunk + frame access.
        # COMMA2K19_LEAKAGE_VERIFIED_OK:routed-via-Comma2k19LocalStreamer-which-runs-_verify_chunk_id_safe-internally
        cumulative_frames: list[np.ndarray] = []
        for chunk_id in window:
            for frame in streamer.stream_frames(
                chunk_id, max_frames=frames_per_chunk
            ):
                cumulative_frames.append(frame)
        cfg = DistillationConfig(
            dataset_name=distill_cfg_template.dataset_name,
            dataset_sha256=distill_cfg_template.dataset_sha256,
            num_road_plane_components=distill_cfg_template.num_road_plane_components,
            num_vehicle_components=distill_cfg_template.num_vehicle_components,
            num_lane_curvature_components=distill_cfg_template.num_lane_curvature_components,
            random_seed=distill_cfg_template.random_seed,
            max_frames=max(distill_cfg_template.max_frames, len(cumulative_frames)),
            allow_bdd100k_dataset_images=distill_cfg_template.allow_bdd100k_dataset_images,
        )
        book = distill_codebook(cfg, frames=iter(cumulative_frames))  # COMMA2K19_LEAKAGE_VERIFIED_OK:cumulative_frames-came-from-streamer-which-internally-runs-check_no_contest_video_leakage
        quality = float(quality_metric(book))
        marginal: float | None = None
        if last_quality is not None:
            marginal = float(last_quality - quality)
        from tac.substrates.pretrained_driving_prior.codebook import serialize_codebook

        book_bytes = serialize_codebook(book)
        provenance_dict: dict[str, Any] = dict(extra_provenance)
        provenance_dict.update(
            {
                "license_tags": list(book.metadata.get("license_tags", [])),
                "dataset_provenance": book.metadata.get("dataset_provenance", ""),
                "schedule_base": int(schedule.base),
                "schedule_max_chunks": int(schedule.max_chunks),
                "schedule_quality_plateau_threshold": float(
                    schedule.quality_plateau_threshold
                ),
                "streamer_source_url": streamer.source_url,
                "streamer_log_dir": str(streamer.log_dir),
                "streamer_is_synthetic": streamer.is_synthetic,
                "streamer_ram_buffer_gb": streamer.ram_buffer_gb,
                "streamed_chunk_ids_used": list(window),
                "frames_streamed": int(len(cumulative_frames)),
            }
        )
        step_result = ScheduleStepResult(
            step=step,
            chunk_count=usable_chunk_count,
            frame_count=int(book.metadata.get("num_frames_used", 0)),
            codebook_size_bytes=len(book_bytes),
            quality=quality,
            marginal_improvement=marginal,
            codebook_basis_sha256=str(book.metadata.get("basis_sha256", "")),
            provenance=provenance_dict,
        )
        schedule_log.append(step_result)
        final_book = book
        final_chunk_count = usable_chunk_count
        if (
            schedule.quality_plateau_threshold > 0.0
            and marginal is not None
            and step >= 2
            and 0.0 <= marginal < schedule.quality_plateau_threshold
        ):
            step_result.early_stopped = True
            logger.info(
                "log_incremental_distillation_streaming: plateau early-stop "
                "at step %d (chunks=%d, marginal=%.6f < threshold=%.6f)",
                step,
                usable_chunk_count,
                marginal,
                schedule.quality_plateau_threshold,
            )
            break
        if usable_chunk_count >= len(chunk_ids):
            break
        last_quality = quality

    assert final_book is not None, "schedule produced no codebook (empty plan?)"
    final_book.metadata.setdefault("log_incremental_chunks", final_chunk_count)
    final_book.metadata.setdefault("streaming_mode", True)
    return final_book, schedule_log
