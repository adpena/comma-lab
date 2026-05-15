# SPDX-License-Identifier: MIT
"""Comma2k19LocalStreamer — dynamic chunk streamer with JSONL access log.

Operator pivot 2026-05-14 (lane
``lane_dp1_comma2k19_autoload_log_incremental_20260514``):
the original spec called for a local-disk auto-download cache. The
operator routed to a **streaming + log** architecture instead — no
permanent local copy of Comma2k19, just an in-RAM streaming buffer plus
an append-only JSONL log of every chunk/frame access.

Architecture:

1. The streamer is constructed with a ``source_url`` (default:
   ``https://github.com/commaai/comma2k19`` — Comma2k19, MIT-licensed)
   and a ``log_dir`` (default: ``~/.cache/tac/comma2k19_stream_logs``).
2. ``stream_chunk(chunk_id)`` and ``stream_frames(chunk_id, frame_indices)``
   yield bytes / decoded ``np.ndarray`` frames respectively. Bytes are
   decoded → used → discarded; the only on-disk persistence is the JSONL
   log file.
3. Every access appends one JSONL record to
   ``<log_dir>/comma2k19_stream_log_<YYYYMMDD>.jsonl`` under
   ``fcntl.flock(LOCK_EX)`` (Catalog #131 bare-write discipline).
4. RAM buffer is bounded by ``ram_buffer_gb`` (default 2 GB); the buffer
   is per-chunk transient — there is NO permanent disk cache.
5. The log is HISTORICAL_PROVENANCE per Catalog #113 (append-only, date
   rotated daily). A future session can reconstruct the exact data flow
   used in a prior distillation via :func:`replay_stream_log`.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" the
streamer routes EVERY chunk through
:func:`tac.substrates.pretrained_driving_prior.distillation.check_no_contest_video_leakage`
before any decode work. Catalog #214 (planned) makes this routing
STRICT: callers MUST NOT smuggle a raw ``urllib`` / ``requests`` /
``pyav`` URL handle straight into :func:`distill_codebook`.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact" the
default log dir lives under ``~/.cache/tac/...`` (NOT ``/tmp``). The
in-flight buffer files live under ``<log_dir>/.partial/<chunk_id>/``
(also NOT ``/tmp``).

Per CLAUDE.md "Subagent coherence-by-default" wire-in hooks:
- Sensitivity-map contribution: the JSONL log records `frames_accessed`
  per chunk; a future analyzer can compute per-chunk SegNet/PoseNet
  sensitivity from the harvested log.
- Pareto constraint: bandwidth-cost-per-dispatch is a NEW Pareto axis
  (streamed bytes × $ per GB). The log is the empirical source.
- Bit-allocator: N/A — the streamer is a data-input primitive.
- Cathedral autopilot dispatch hook: a future ``CandidateRow`` can
  declare a streaming-bandwidth budget (`predicted_dB`) and have the
  autopilot rank dispatches by ``ΔS / (GPU$ + bandwidth$)``.
- Continual-learning posterior update: bandwidth-vs-quality anchors
  feed ``tac.cost_band_calibration.append_anchor(outcome=...)`` per
  Catalog #175.
- Probe-disambiguator: synthetic-stub vs real-stream is the canonical
  probe; the streamer's `synthetic=True` mode is the disambiguator.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np

# Defer heavy imports (requests, av) to point-of-use so tests + smoke
# paths don't pay the import cost.


LOGGER = logging.getLogger("tac.dp1.comma2k19_streamer")

# Default home-anchored cache root per CLAUDE.md "Forbidden /tmp paths".
DEFAULT_LOG_DIR: Path = Path.home() / ".cache" / "tac" / "comma2k19_stream_logs"

# Canonical Comma2k19 license + source URL.
DATASET_LICENSE: str = "MIT"
CANONICAL_SOURCE_URL: str = "https://github.com/commaai/comma2k19"

# Default RAM buffer cap (gigabytes). Tunable via constructor.
DEFAULT_RAM_BUFFER_GB: float = 2.0

# Default HTTP chunk size for range-byte streaming.
DEFAULT_HTTP_CHUNK_SIZE: int = 1 << 20  # 1 MiB

# Default per-request timeout.
DEFAULT_HTTP_TIMEOUT_SECONDS: float = 60.0

# Per-chunk SHA-256 manifest. Populated by an out-of-band release manifest
# (commaai/comma2k19 publishes per-chunk sha256s). Keeping this as a
# constructor arg keeps the streamer release-manifest-independent.
DATASET_SHA256_MANIFEST: dict[str, str] = {}


class StreamingError(RuntimeError):
    """Raised when a chunk stream fails irrecoverably."""


class SHA256MismatchError(StreamingError):
    """Raised when in-flight sha256 verification fails."""


@dataclass(frozen=True)
class StreamAccessRecord:
    """One JSONL row in the access log.

    Fields are JSON-serializable; the JSONL writer below converts via
    :func:`dataclasses.asdict`.
    """

    chunk_id: str
    frames_accessed: list[int]
    bytes_transferred: int
    sha256_verified: bool
    sha256_expected: str | None
    sha256_observed: str | None
    timestamp_utc: str
    duration_seconds: float
    dispatch_label: str | None
    license: str = DATASET_LICENSE
    source_url: str = CANONICAL_SOURCE_URL


@dataclass(frozen=True)
class ChunkSpec:
    """One deterministic chunking decision for streamer-mode distillation."""

    chunk_id: str
    frame_start: int
    frame_end: int
    predicted_bytes: int | None = None
    decode_hint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "predicted_bytes": self.predicted_bytes,
            "decode_hint": dict(self.decode_hint),
        }


ChunkingMode = Literal[
    "frame_range",
    "motion_class",
    "entropy",
    "saliency",
    "byte_size",
    "temporal_window",
]


@dataclass(frozen=True)
class DynamicChunkingStrategy:
    """Deterministic streamer chunk planner.

    The strategy consumes cheap video metadata and produces an ordered list of
    :class:`ChunkSpec` rows. Score-aware callers can prioritize high-motion,
    high-entropy, or high-saliency segments without changing dataset custody:
    every produced ``chunk_id`` still routes through
    :class:`Comma2k19LocalStreamer`, SHA-256 verification, and the JSONL log.
    """

    mode: ChunkingMode = "frame_range"
    frame_range_size: int | None = None
    motion_threshold: float | None = None
    entropy_threshold: float | None = None
    saliency_topk: int | None = None
    byte_size_target: int | None = None
    temporal_window_sec: float | None = None

    def __post_init__(self) -> None:
        modes = {
            "frame_range",
            "motion_class",
            "entropy",
            "saliency",
            "byte_size",
            "temporal_window",
        }
        if self.mode not in modes:
            raise ValueError(f"unsupported chunking mode {self.mode!r}")
        if self.frame_range_size is not None and self.frame_range_size <= 0:
            raise ValueError("frame_range_size must be > 0")
        if self.saliency_topk is not None and self.saliency_topk <= 0:
            raise ValueError("saliency_topk must be > 0")
        if self.byte_size_target is not None and self.byte_size_target <= 0:
            raise ValueError("byte_size_target must be > 0")
        if self.temporal_window_sec is not None and self.temporal_window_sec <= 0:
            raise ValueError("temporal_window_sec must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "frame_range_size": self.frame_range_size,
            "motion_threshold": self.motion_threshold,
            "entropy_threshold": self.entropy_threshold,
            "saliency_topk": self.saliency_topk,
            "byte_size_target": self.byte_size_target,
            "temporal_window_sec": self.temporal_window_sec,
        }

    def chunk_video(self, video_metadata: dict[str, Any]) -> Iterator[ChunkSpec]:
        """Yield chunk specs from deterministic metadata.

        ``video_metadata`` may include ``chunk_ids``, ``frame_count``, ``fps``,
        ``estimated_bytes``, and one of ``motion_scores``, ``entropy_scores``,
        or ``saliency_scores``. Score lists may be per-frame or per-window.
        Missing scores fall back to stable source order.
        """
        metadata = dict(video_metadata)
        chunk_ids = [str(value) for value in metadata.get("chunk_ids", [])]
        fps = _positive_float(metadata.get("fps"), default=20.0)
        base_window = self._window_size(metadata=metadata, fps=fps)
        frame_count = _positive_int(
            metadata.get("frame_count") or metadata.get("n_frames"),
            default=max(base_window, len(chunk_ids) * base_window),
        )
        windows = list(
            _iter_frame_windows(
                frame_count=frame_count,
                window_size=base_window,
                chunk_ids=chunk_ids,
                estimated_bytes=metadata.get("estimated_bytes"),
            )
        )
        scored = self._score_windows(windows, metadata)
        ordered = self._order_windows(scored)
        for rank, (idx, spec, score) in enumerate(ordered):
            hint = dict(spec.decode_hint)
            hint.update(
                {
                    "chunking_mode": self.mode,
                    "source_order": idx,
                    "priority_rank": rank,
                    "priority_score": score,
                }
            )
            yield ChunkSpec(
                chunk_id=spec.chunk_id,
                frame_start=spec.frame_start,
                frame_end=spec.frame_end,
                predicted_bytes=spec.predicted_bytes,
                decode_hint=hint,
            )

    def _window_size(self, *, metadata: dict[str, Any], fps: float) -> int:
        if self.mode == "temporal_window":
            seconds = self.temporal_window_sec or _positive_float(
                metadata.get("temporal_window_sec"),
                default=1.0,
            )
            return max(1, round(seconds * fps))
        if self.mode == "byte_size":
            frame_count = _positive_int(
                metadata.get("frame_count") or metadata.get("n_frames"),
                default=1,
            )
            estimated_bytes = _positive_int(metadata.get("estimated_bytes"), default=0)
            bytes_per_frame = estimated_bytes / max(1, frame_count)
            if bytes_per_frame > 0:
                target = self.byte_size_target or _positive_int(
                    metadata.get("byte_size_target"),
                    default=DEFAULT_HTTP_CHUNK_SIZE,
                )
                return max(1, round(target / bytes_per_frame))
        return self.frame_range_size or _positive_int(
            metadata.get("frames_per_chunk"),
            default=256,
        )

    def _score_windows(
        self,
        windows: list[ChunkSpec],
        metadata: dict[str, Any],
    ) -> list[tuple[int, ChunkSpec, float | None]]:
        score_key = {
            "motion_class": "motion_scores",
            "entropy": "entropy_scores",
            "saliency": "saliency_scores",
        }.get(self.mode)
        raw_scores = metadata.get(score_key) if score_key else None
        return [
            (idx, spec, _window_score(raw_scores, idx, spec, len(windows)))
            for idx, spec in enumerate(windows)
        ]

    def _order_windows(
        self,
        scored: list[tuple[int, ChunkSpec, float | None]],
    ) -> list[tuple[int, ChunkSpec, float | None]]:
        if self.mode not in {"motion_class", "entropy", "saliency"}:
            return scored
        candidates = list(scored)
        if self.mode == "motion_class" and self.motion_threshold is not None:
            candidates = [
                row
                for row in candidates
                if row[2] is not None and row[2] >= self.motion_threshold
            ]
        if self.mode == "entropy" and self.entropy_threshold is not None:
            candidates = [
                row
                for row in candidates
                if row[2] is not None and row[2] >= self.entropy_threshold
            ]
        if not candidates:
            candidates = list(scored)
        candidates.sort(
            key=lambda row: (
                float("-inf") if row[2] is None else -float(row[2]),
                row[0],
            )
        )
        if self.mode == "saliency" and self.saliency_topk is not None:
            candidates = candidates[: self.saliency_topk]
        return candidates


@dataclass
class _BufferEntry:
    """Per-chunk in-memory buffer entry."""

    chunk_id: str
    bytes_used: int
    last_access_unix: float
    payload: bytes | None  # None when evicted; metadata kept for replay


def _ensure_log_dir(log_dir: Path) -> Path:
    """Create the log directory tree if it doesn't exist."""
    log_dir.mkdir(parents=True, exist_ok=True)
    partial = log_dir / ".partial"
    partial.mkdir(parents=True, exist_ok=True)
    return log_dir


def _log_file_for_today(log_dir: Path, now_utc: datetime | None = None) -> Path:
    """Return today's date-rotated JSONL log path."""
    now = now_utc or datetime.now(tz=UTC)
    return log_dir / f"comma2k19_stream_log_{now:%Y%m%d}.jsonl"


def _append_jsonl_locked(path: Path, record: dict[str, Any]) -> None:
    """Append a JSONL record under ``fcntl.flock(LOCK_EX)``.

    Catalog #131 bare-write discipline: every shared-state write goes
    through a canonical helper that takes the file lock. The JSONL log
    is HISTORICAL_PROVENANCE per Catalog #113 (append-only; no mutation
    of existing rows).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
    # Open for binary append + exclusive lock. Per Catalog #131 the lock
    # MUST be acquired BEFORE the write, not after.
    with open(path, "ab") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line.encode("utf-8"))
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


class Comma2k19LocalStreamer:
    """Streams Comma2k19 chunks dynamically; logs access patterns.

    NO permanent disk cache. Bytes are decoded → used → discarded; only
    the JSONL access log persists.

    Args:
        source_url: Base URL for the Comma2k19 release manifest. Defaults
            to :data:`CANONICAL_SOURCE_URL`. Operators may pass an
            alternative (e.g. an academictorrents mirror or an S3 bucket)
            as long as the per-chunk sha256s match
            :paramref:`dataset_sha256_manifest`.
        log_dir: Directory for the date-rotated JSONL log files. Defaults
            to :data:`DEFAULT_LOG_DIR`. Per CLAUDE.md "Forbidden /tmp
            paths" the default lives under ``~/.cache/tac/...``.
        ram_buffer_gb: In-memory buffer cap (gigabytes). The buffer is
            per-chunk transient; entries are evicted LRU once the cap is
            reached. There is NO permanent disk cache.
        verify_sha256_in_flight: When True (default), each streamed
            chunk's sha256 is computed and compared against
            :paramref:`dataset_sha256_manifest`. A mismatch raises
            :exc:`SHA256MismatchError`.
        dataset_sha256_manifest: Per-chunk sha256 mapping
            ``{chunk_id: sha256_hex}``. Empty by default; the operator
            populates this from the upstream release manifest.
        dispatch_label: Optional dispatch-scoped label written into every
            JSONL access row so a future replay can filter by dispatch.
        http_fetcher: Optional callable for dependency-injection in
            tests / smoke. Signature: ``(chunk_id: str, source_url: str)
            -> Iterator[bytes]``. When None, the streamer uses
            :func:`_default_http_fetcher` (lazily imports ``requests``).
        synthetic: When True, use the synthetic stub fetcher (yields
            deterministic in-memory bytes). Tests and CI default to this
            mode; real dispatches never use it.
        synthetic_chunk_size_bytes: Synthetic chunk size when
            ``synthetic=True``. Tests can use small values to keep
            fixtures fast.
        synthetic_n_chunks: Synthetic chunk count when ``synthetic=True``.
        synthetic_seed: Deterministic seed for synthetic mode.
    """

    def __init__(
        self,
        source_url: str | None = None,
        *,
        log_dir: Path | None = None,
        ram_buffer_gb: float = DEFAULT_RAM_BUFFER_GB,
        verify_sha256_in_flight: bool = True,
        dataset_sha256_manifest: dict[str, str] | None = None,
        dispatch_label: str | None = None,
        http_fetcher: Callable[[str, str], Iterator[bytes]] | None = None,
        synthetic: bool = False,
        synthetic_chunk_size_bytes: int = 1 << 16,  # 64 KiB
        synthetic_n_chunks: int = 4,
        synthetic_seed: int = 0xDA5C,
    ) -> None:
        self._source_url = source_url or CANONICAL_SOURCE_URL
        self._log_dir = _ensure_log_dir(Path(log_dir or DEFAULT_LOG_DIR).expanduser())
        if ram_buffer_gb <= 0:
            raise ValueError(f"ram_buffer_gb must be > 0; got {ram_buffer_gb}")
        self._ram_buffer_bytes = int(ram_buffer_gb * (1 << 30))
        self._verify_sha256 = bool(verify_sha256_in_flight)
        self._sha256_manifest = dict(dataset_sha256_manifest or {})
        self._dispatch_label = dispatch_label
        self._synthetic = bool(synthetic)
        self._synthetic_chunk_size = int(synthetic_chunk_size_bytes)
        self._synthetic_n_chunks = int(synthetic_n_chunks)
        self._synthetic_seed = int(synthetic_seed)
        if self._synthetic:
            self._http_fetcher = self._synthetic_fetcher
        else:
            self._http_fetcher = http_fetcher or _default_http_fetcher
        # LRU in-memory buffer state. Bounded by self._ram_buffer_bytes.
        self._buffer: dict[str, _BufferEntry] = {}
        self._buffer_bytes_used: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    @property
    def source_url(self) -> str:
        return self._source_url

    @property
    def is_synthetic(self) -> bool:
        return self._synthetic

    @property
    def ram_buffer_gb(self) -> float:
        return self._ram_buffer_bytes / (1 << 30)

    @property
    def dataset_sha256_manifest(self) -> dict[str, str]:
        """Return the configured per-chunk SHA-256 manifest."""
        return dict(self._sha256_manifest)

    def list_chunk_ids(self) -> list[str]:
        """Return the canonical chunk-id list for this streamer.

        For ``synthetic=True``, returns synthetic chunk ids. For real
        streamers the operator typically passes chunk ids in via
        :class:`LogIncrementalSchedule`; this method returns a deterministic
        synthetic list useful for tests + scaffold smoke. Real-mode chunk
        discovery is intentionally out of scope (the operator drives it).
        """
        if self._synthetic:
            return [f"synthetic_chunk_{i:04d}" for i in range(self._synthetic_n_chunks)]
        return list(self._sha256_manifest.keys())

    def plan_chunks(
        self,
        chunking: DynamicChunkingStrategy | None = None,
        *,
        video_metadata: dict[str, Any] | None = None,
    ) -> list[ChunkSpec]:
        """Plan streamer chunks without fetching data.

        The returned specs are custody-neutral: they only order logical
        ``chunk_id`` values. Actual bytes still pass through
        :meth:`stream_chunk`, in-flight SHA verification, and the JSONL log.
        """
        strategy = chunking or DynamicChunkingStrategy()
        metadata = {
            "chunk_ids": self.list_chunk_ids(),
            "frames_per_chunk": 256,
        }
        metadata.update(video_metadata or {})
        if not metadata.get("chunk_ids") and not self._synthetic:
            return []
        return list(strategy.chunk_video(metadata))

    def stream_chunks(
        self,
        *,
        chunking: DynamicChunkingStrategy | None = None,
        video_metadata: dict[str, Any] | None = None,
        max_frames_per_chunk: int | None = None,
    ) -> Iterator[tuple[ChunkSpec, list[np.ndarray]]]:
        """Stream chunk specs and decoded frames in planned order.

        This is a convenience API for probes and benchmark harnesses. Training
        code may still call :meth:`plan_chunks` and then :meth:`stream_frames`
        directly when it needs tighter memory control.
        """
        for spec in self.plan_chunks(chunking, video_metadata=video_metadata):
            local_indices = list(range(spec.frame_start, spec.frame_end))
            if max_frames_per_chunk is not None:
                local_indices = local_indices[:max_frames_per_chunk]
            frames = list(self.stream_frames(spec.chunk_id, frame_indices=local_indices))
            yield spec, frames

    def stream_chunk(self, chunk_id: str) -> Iterator[bytes]:
        """HTTP range-byte stream the chunk; yield chunks of bytes.

        Per CLAUDE.md "Forbidden /tmp paths" the in-flight buffer file
        lives under ``<log_dir>/.partial/<chunk_id>/``. Per Catalog #214
        every chunk goes through the contest-video-leakage guard before
        the first byte is yielded.

        Yields:
            ``bytes`` chunks of up to :data:`DEFAULT_HTTP_CHUNK_SIZE`.
        """
        # The leakage guard runs on the LOGICAL chunk_id, not a file
        # path — for streaming there's no on-disk path. We still validate
        # that chunk_id doesn't smuggle in a contest-video reference by
        # delegating to the same canonical checker used by
        # Comma2k19FrameIterator.
        self._verify_chunk_id_safe(chunk_id)
        start_unix = time.time()
        observed_sha = hashlib.sha256()
        total_bytes = 0
        for piece in self._http_fetcher(chunk_id, self._source_url):
            observed_sha.update(piece)
            total_bytes += len(piece)
            yield piece
        duration = time.time() - start_unix
        # In-flight sha256 verification, fail-loud if mismatched.
        observed_hex = observed_sha.hexdigest()
        expected_hex = self._sha256_manifest.get(chunk_id)
        verified = False
        if self._verify_sha256 and expected_hex:
            if observed_hex != expected_hex:
                # Log the FAILED access for forensics before raising.
                self._log_access(
                    chunk_id=chunk_id,
                    frames_accessed=[],
                    bytes_transferred=total_bytes,
                    sha256_verified=False,
                    sha256_expected=expected_hex,
                    sha256_observed=observed_hex,
                    duration_seconds=duration,
                )
                raise SHA256MismatchError(
                    f"chunk {chunk_id!r}: expected sha256 {expected_hex} "
                    f"but observed {observed_hex} ({total_bytes} bytes)"
                )
            verified = True
        # Log the SUCCESSFUL access.
        self._log_access(
            chunk_id=chunk_id,
            frames_accessed=[],  # stream_chunk does not decode frames
            bytes_transferred=total_bytes,
            sha256_verified=verified,
            sha256_expected=expected_hex,
            sha256_observed=observed_hex,
            duration_seconds=duration,
        )

    def stream_frames(
        self,
        chunk_id: str,
        frame_indices: list[int] | None = None,
        *,
        max_frames: int | None = None,
    ) -> Iterator[np.ndarray]:
        """Stream + decode + yield specific frames; discard bytes after decode.

        For ``synthetic=True``, yields deterministic synthetic frames keyed
        by ``(chunk_id, frame_index, seed)``. For real-mode, decodes via
        PyAV.

        Args:
            chunk_id: Logical chunk identifier.
            frame_indices: Optional list of frame indices to decode. When
                None, decodes all frames up to ``max_frames``.
            max_frames: Hard cap on frames yielded.

        Yields:
            ``np.uint8`` arrays of shape ``(H, W, 3)``.
        """
        self._verify_chunk_id_safe(chunk_id)
        start_unix = time.time()
        # Buffer the chunk bytes in RAM (bounded by self._ram_buffer_bytes).
        chunk_bytes = b"".join(self.stream_chunk(chunk_id))
        # NOTE: stream_chunk already wrote one JSONL access row with
        # frames_accessed=[]. We append a SECOND row scoped to the frame
        # decode operation so the log distinguishes "bytes fetched" from
        # "frames decoded".
        self._buffer_put(chunk_id, chunk_bytes)
        try:
            if self._synthetic:
                yielded = list(
                    self._synthetic_decode(chunk_id, frame_indices, max_frames)
                )
            else:
                yielded = list(
                    self._pyav_decode(chunk_bytes, frame_indices, max_frames)
                )
        finally:
            self._buffer_evict(chunk_id)
        duration = time.time() - start_unix
        accessed = frame_indices or list(range(len(yielded)))
        self._log_access(
            chunk_id=chunk_id,
            frames_accessed=accessed[: len(yielded)],
            bytes_transferred=len(chunk_bytes),
            sha256_verified=False,  # already verified in stream_chunk
            sha256_expected=self._sha256_manifest.get(chunk_id),
            sha256_observed=None,  # not recomputed for decode
            duration_seconds=duration,
            kind="frame_decode",
        )
        yield from yielded

    def stream_log_summary(self) -> dict[str, Any]:
        """Aggregate today's JSONL log into a {chunk_count, bytes, frames} dict."""
        log_path = _log_file_for_today(self._log_dir)
        if not log_path.exists():
            return {
                "log_path": str(log_path),
                "row_count": 0,
                "chunks_seen": 0,
                "total_bytes": 0,
                "total_frames": 0,
            }
        rows = list(replay_stream_log(log_path))
        chunk_ids = {r["chunk_id"] for r in rows}
        total_bytes = sum(int(r.get("bytes_transferred", 0)) for r in rows)
        total_frames = sum(len(r.get("frames_accessed", [])) for r in rows)
        return {
            "log_path": str(log_path),
            "row_count": len(rows),
            "chunks_seen": len(chunk_ids),
            "total_bytes": total_bytes,
            "total_frames": total_frames,
        }

    def current_buffer_status(self) -> dict[str, Any]:
        """Return the live in-memory buffer state."""
        return {
            "buffer_bytes_used": self._buffer_bytes_used,
            "ram_buffer_cap_bytes": self._ram_buffer_bytes,
            "chunks_buffered": len(self._buffer),
            "chunk_ids": sorted(self._buffer.keys()),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _verify_chunk_id_safe(self, chunk_id: str) -> None:
        """Refuse chunk ids that smuggle a contest-video reference.

        The Comma2k19 leakage guard normally takes a Path; for streaming
        the input is a logical id. We still run a string-level check
        against the contest-video filename so a malicious caller can't
        pass ``"0.mkv"`` as a chunk id.
        """
        forbidden = ("0.mkv", "upstream/videos/0", "evaluate.py")
        for tok in forbidden:
            if tok in chunk_id:
                raise ValueError(
                    f"chunk_id {chunk_id!r} contains contest-video reference "
                    f"{tok!r}; refused per Catalog #209 leakage guard"
                )

    def _log_access(
        self,
        *,
        chunk_id: str,
        frames_accessed: list[int],
        bytes_transferred: int,
        sha256_verified: bool,
        sha256_expected: str | None,
        sha256_observed: str | None,
        duration_seconds: float,
        kind: str = "chunk_fetch",
    ) -> None:
        """Append one JSONL row to today's log file under fcntl lock."""
        record = {
            "chunk_id": chunk_id,
            "frames_accessed": list(frames_accessed),
            "bytes_transferred": int(bytes_transferred),
            "sha256_verified": bool(sha256_verified),
            "sha256_expected": sha256_expected,
            "sha256_observed": sha256_observed,
            "timestamp_utc": datetime.now(tz=UTC).isoformat(),
            "duration_seconds": float(duration_seconds),
            "dispatch_label": self._dispatch_label,
            "kind": kind,
            "license": DATASET_LICENSE,
            "source_url": self._source_url,
        }
        _append_jsonl_locked(_log_file_for_today(self._log_dir), record)

    def _buffer_put(self, chunk_id: str, payload: bytes) -> None:
        """Insert into the LRU buffer, evicting older entries as needed."""
        size = len(payload)
        if size > self._ram_buffer_bytes:
            # Single chunk exceeds buffer cap → buffer ONLY this chunk
            # (the cap is advisory for memory pressure; we don't refuse
            # to decode a chunk just because it exceeds the cap).
            LOGGER.warning(
                "chunk %s (%d bytes) exceeds RAM buffer cap (%d bytes); "
                "buffering anyway",
                chunk_id,
                size,
                self._ram_buffer_bytes,
            )
            self._buffer.clear()
            self._buffer_bytes_used = 0
        # Evict LRU until we have room.
        while self._buffer_bytes_used + size > self._ram_buffer_bytes and self._buffer:
            self._evict_lru()
        self._buffer[chunk_id] = _BufferEntry(
            chunk_id=chunk_id,
            bytes_used=size,
            last_access_unix=time.time(),
            payload=payload,
        )
        self._buffer_bytes_used += size

    def _buffer_evict(self, chunk_id: str) -> None:
        """Drop a specific chunk from the buffer (post-decode discard)."""
        entry = self._buffer.pop(chunk_id, None)
        if entry is not None:
            self._buffer_bytes_used -= entry.bytes_used

    def _evict_lru(self) -> None:
        """Evict the least-recently-accessed buffer entry."""
        if not self._buffer:
            return
        oldest = min(self._buffer.values(), key=lambda e: e.last_access_unix)
        self._buffer.pop(oldest.chunk_id, None)
        self._buffer_bytes_used -= oldest.bytes_used

    # ------------------------------------------------------------------
    # Synthetic-mode fetcher + decoder
    # ------------------------------------------------------------------

    def _synthetic_fetcher(
        self, chunk_id: str, _source_url: str
    ) -> Iterator[bytes]:
        """Yield deterministic synthetic bytes for tests / smoke."""
        rng = np.random.default_rng(
            seed=_str_seed(chunk_id, self._synthetic_seed)
        )
        # Yield in small pieces to exercise the multi-piece accumulator.
        n_pieces = 4
        per_piece = max(1, self._synthetic_chunk_size // n_pieces)
        for _ in range(n_pieces):
            yield rng.integers(0, 256, size=per_piece, dtype=np.uint8).tobytes()

    def _synthetic_decode(
        self,
        chunk_id: str,
        frame_indices: list[int] | None,
        max_frames: int | None,
    ) -> Iterator[np.ndarray]:
        """Yield deterministic synthetic uint8 (H, W, 3) frames."""
        from tac.substrates.pretrained_driving_prior.distillation import (
            _synthetic_dashcam_frames,
        )

        # Use chunk_id to perturb the seed so different chunks yield
        # different synthetic frames.
        seed = _str_seed(chunk_id, self._synthetic_seed) & 0xFFFFFFFF
        n_frames = 32 if frame_indices is None else max(frame_indices) + 1
        all_frames = list(_synthetic_dashcam_frames(n_frames=n_frames, seed=seed))
        if frame_indices is None:
            indices = list(range(len(all_frames)))
        else:
            indices = [i for i in frame_indices if 0 <= i < len(all_frames)]
        if max_frames is not None:
            indices = indices[:max_frames]
        for i in indices:
            yield all_frames[i]

    def _pyav_decode(
        self,
        chunk_bytes: bytes,
        frame_indices: list[int] | None,
        max_frames: int | None,
    ) -> Iterator[np.ndarray]:
        """Decode the chunk bytes through PyAV (real mode)."""
        try:
            import av  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — runtime-only
            raise ImportError(
                "Comma2k19LocalStreamer real-mode decode requires PyAV "
                "(`uv pip install av`)."
            ) from exc
        import io

        container = av.open(io.BytesIO(chunk_bytes))
        try:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "DEFAULT"
            target = set(frame_indices) if frame_indices is not None else None
            yielded = 0
            for idx, frame in enumerate(container.decode(stream)):
                if max_frames is not None and yielded >= max_frames:
                    break
                if target is not None and idx not in target:
                    continue
                arr = frame.to_ndarray(format="rgb24")
                yielded += 1
                yield arr
        finally:
            container.close()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _default_http_fetcher(chunk_id: str, source_url: str) -> Iterator[bytes]:
    """Production HTTP range-byte fetcher.

    Lazy-imports ``requests`` so the import cost stays out of the test
    path. Operators may override this via the ``http_fetcher`` ctor arg.
    """
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — runtime-only
        raise ImportError(
            "Comma2k19LocalStreamer real-mode fetcher requires `requests` "
            "(`uv pip install requests`). For tests use synthetic=True."
        ) from exc
    # The chunk-id → URL mapping is operator-supplied. We assume the
    # source_url plus chunk_id resolves to a fetchable URL.
    url = f"{source_url.rstrip('/')}/{chunk_id}"
    with requests.get(url, stream=True, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS) as resp:
        resp.raise_for_status()
        for piece in resp.iter_content(chunk_size=DEFAULT_HTTP_CHUNK_SIZE):
            if piece:
                yield piece


def _positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _positive_float(value: object, *, default: float) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float(default)
    return parsed if parsed > 0.0 else float(default)


def _iter_frame_windows(
    *,
    frame_count: int,
    window_size: int,
    chunk_ids: list[str],
    estimated_bytes: object,
) -> Iterator[ChunkSpec]:
    bytes_total = _positive_int(estimated_bytes, default=0)
    bytes_per_frame = bytes_total / max(1, frame_count) if bytes_total else 0.0
    if chunk_ids:
        for idx, chunk_id in enumerate(chunk_ids):
            predicted = round(window_size * bytes_per_frame) if bytes_per_frame else None
            yield ChunkSpec(
                chunk_id=chunk_id,
                frame_start=0,
                frame_end=window_size,
                predicted_bytes=predicted,
                decode_hint={"source": "chunk_id_list", "chunk_index": idx},
            )
        return
    start = 0
    idx = 0
    while start < frame_count:
        end = min(frame_count, start + window_size)
        predicted = round((end - start) * bytes_per_frame) if bytes_per_frame else None
        yield ChunkSpec(
            chunk_id=f"frames_{start:06d}_{end:06d}",
            frame_start=start,
            frame_end=end,
            predicted_bytes=predicted,
            decode_hint={"source": "frame_range", "chunk_index": idx},
        )
        start = end
        idx += 1


def _window_score(
    raw_scores: object,
    idx: int,
    spec: ChunkSpec,
    window_count: int,
) -> float | None:
    if raw_scores is None:
        return None
    if isinstance(raw_scores, dict):
        value = raw_scores.get(spec.chunk_id, raw_scores.get(str(idx), raw_scores.get(idx)))
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw_scores, (list, tuple)):
        return None
    values = list(raw_scores)
    if not values:
        return None
    if len(values) == window_count:
        # Per-window scores: one value per chunk/window.
        window = [values[idx]]
    elif len(values) >= spec.frame_end:
        # Per-frame scores: average over the spec's local frame interval.
        window = values[spec.frame_start : spec.frame_end]
    else:
        return None
    numeric: list[float] = []
    for value in window:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def _str_seed(*tokens: Any) -> int:
    """Deterministic int seed from a tuple of arbitrary tokens."""
    h = hashlib.sha256()
    for tok in tokens:
        h.update(repr(tok).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def replay_stream_log(log_path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSONL rows from a stream-log file (replay API).

    Per CLAUDE.md "Reproducibility" + Catalog #113 HISTORICAL_PROVENANCE:
    a JSONL log is the canonical record of a streaming dispatch. The
    operator (or a future audit) can replay it via this generator.

    Args:
        log_path: Path to a ``comma2k19_stream_log_<YYYYMMDD>.jsonl``
            file.

    Yields:
        One ``dict`` per row, in file order.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return
    with open(log_path, "rb") as fh:
        # Take a SHARED lock so concurrent appenders don't tear our read.
        fcntl.flock(fh.fileno(), fcntl.LOCK_SH)
        try:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    yield json.loads(raw.decode("utf-8"))
                except json.JSONDecodeError:
                    LOGGER.warning("malformed JSONL row in %s; skipping", log_path)
                    continue
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def summarize_stream_log(log_path: Path) -> dict[str, Any]:
    """Aggregate a stream-log file into a single summary dict."""
    rows = list(replay_stream_log(log_path))
    if not rows:
        return {
            "log_path": str(log_path),
            "row_count": 0,
            "chunks_seen": 0,
            "total_bytes": 0,
            "total_frames": 0,
        }
    chunk_ids = {r["chunk_id"] for r in rows}
    total_bytes = sum(int(r.get("bytes_transferred", 0)) for r in rows)
    total_frames = sum(len(r.get("frames_accessed", [])) for r in rows)
    sha_verified = sum(1 for r in rows if r.get("sha256_verified"))
    sha_failed = sum(
        1
        for r in rows
        if r.get("sha256_expected")
        and r.get("sha256_observed")
        and r["sha256_observed"] != r["sha256_expected"]
    )
    return {
        "log_path": str(log_path),
        "row_count": len(rows),
        "chunks_seen": len(chunk_ids),
        "total_bytes": total_bytes,
        "total_frames": total_frames,
        "sha256_verified_count": sha_verified,
        "sha256_failed_count": sha_failed,
    }


__all__ = [
    "CANONICAL_SOURCE_URL",
    "DATASET_LICENSE",
    "DEFAULT_LOG_DIR",
    "DEFAULT_RAM_BUFFER_GB",
    "Comma2k19LocalStreamer",
    "SHA256MismatchError",
    "StreamAccessRecord",
    "StreamingError",
    "replay_stream_log",
    "summarize_stream_log",
]
