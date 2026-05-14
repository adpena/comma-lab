# SPDX-License-Identifier: MIT
"""Offline codebook distillation from public dashcam datasets.

This module runs at BUILD time (or as a separate $0 step) — NEVER at dispatch
time and NEVER against the contest video. The output is a frozen
:class:`tac.substrates.pretrained_driving_prior.codebook.DashcamCodebook` that
the score-aware loss uses as a soft prior.

**Distillation contract:**

1. The input data MUST come from PUBLICLY-AVAILABLE dashcam datasets
   distributed under permissive license (Comma2k19 MIT by default; BDD100K
   dataset-images opt-in behind ``--allow-bdd100k-dataset-images``;
   Waymo SKIPPED per non-commercial restriction).
2. The distillation procedure is DETERMINISTIC given a fixed dataset SHA-256
   + fixed PCA components count + fixed seed. The output codebook is
   reproducibly bit-equal across runs.
3. The contest video ``upstream/videos/0.mkv`` MUST NEVER be passed to this
   module. The :func:`check_no_contest_video_leakage` guard refuses to
   process any path matching ``upstream/videos/*.mkv``.
4. License attribution is BAKED INTO the codebook metadata; the codebook
   itself carries its provenance.

**Production-deployment alignment** (operator 2026-05-13): the distillation
pipeline is designed so future Comma edge devices can compute LOCAL deltas
(per-vehicle calibration) and contribute upstream via federated aggregation.
The :func:`aggregate_local_codebooks` helper sketches the aggregation
mathematics; the actual federated infrastructure lives outside this scaffold.

This module is scaffold-only. Real distillation requires the dataset to be
downloaded locally; the canonical entry point :func:`distill_codebook` accepts
either real frame iterators OR a deterministic synthetic generator (for tests).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tac.substrates.pretrained_driving_prior.codebook import (
    LANE_CURVATURE_PCA_SHAPE,
    ROAD_PLANE_BASIS_SHAPE,
    SKY_HORIZON_PROFILE_SHAPE,
    VEHICLE_APPEARANCE_BASIS_SHAPE,
    DashcamCodebook,
    validate_codebook,
)


class ContestVideoLeakageError(RuntimeError):
    """Raised when distillation input paths match the contest video pattern."""


# Forbidden patterns: anything that looks like the contest video.
_CONTEST_VIDEO_FORBIDDEN_FRAGMENTS: tuple[str, ...] = (
    "upstream/videos/",
    "comma_video_compression_challenge",
    "0.mkv",  # the canonical contest video filename
)


def check_no_contest_video_leakage(input_paths: Iterable[Path]) -> None:
    """Refuse any input path that matches a contest-video fragment.

    Per CLAUDE.md HNeRV parity discipline L1: the codebook is distilled
    from OUT-OF-DISTRIBUTION dashcam data, not the contest video. Leaking
    the contest video into distillation would make the codebook a
    contest-specific overfit that does not generalize to production
    edge devices.
    """
    for path in input_paths:
        path_str = str(path).replace(os.sep, "/")
        for fragment in _CONTEST_VIDEO_FORBIDDEN_FRAGMENTS:
            if fragment in path_str:
                raise ContestVideoLeakageError(
                    f"distillation input {path_str!r} matches forbidden "
                    f"contest-video fragment {fragment!r}; refusing to "
                    f"distill the codebook from the contest video itself. "
                    f"Per CLAUDE.md HNeRV parity L1, the codebook is an "
                    f"OUT-OF-DISTRIBUTION prior."
                )


@dataclass(frozen=True)
class DistillationConfig:
    """Static parameters for the offline distillation run.

    Args:
        dataset_name: One of ``"comma2k19"``, ``"bdd100k"``, or
            ``"synthetic_test"``. Drives license_tags + provenance.
        dataset_sha256: SHA-256 of the dataset archive (or ``""`` for
            synthetic_test). Stored in codebook metadata for reproducibility.
        num_road_plane_components: PCA components for road-plane basis.
            Default 8 (matches ``ROAD_PLANE_BASIS_SHAPE[0]``).
        num_vehicle_components: PCA components for vehicle appearance.
            Default 4.
        num_lane_curvature_components: PCA components for lane curvature.
            Default 8.
        random_seed: Deterministic seed for any randomized sampling.
        max_frames: Maximum frames to use for distillation (cost cap).
        allow_bdd100k_dataset_images: Operator opt-in for BDD100K
            non-commercial-research dataset images. Default False.
    """

    dataset_name: str = "comma2k19"
    dataset_sha256: str = ""
    num_road_plane_components: int = ROAD_PLANE_BASIS_SHAPE[0]
    num_vehicle_components: int = VEHICLE_APPEARANCE_BASIS_SHAPE[0]
    num_lane_curvature_components: int = LANE_CURVATURE_PCA_SHAPE[0]
    random_seed: int = 0xDA5C
    max_frames: int = 50_000
    allow_bdd100k_dataset_images: bool = False


def _license_tags_for_dataset(
    name: str, *, allow_bdd100k_images: bool
) -> list[str]:
    """Return the license-attribution tag list embedded in codebook metadata."""
    if name == "comma2k19":
        return ["comma2k19:MIT", "github.com/commaai/comma2k19"]
    if name == "bdd100k":
        if not allow_bdd100k_images:
            raise ValueError(
                "BDD100K dataset-images require explicit "
                "--allow-bdd100k-dataset-images opt-in (UC Berkeley "
                "research/academic terms). The default scaffold uses "
                "Comma2k19 (MIT) only."
            )
        return [
            "bdd100k:UC-Berkeley-research-only",
            "bdd100k:operator-opted-in",
            "github.com/bdd100k/bdd100k",
        ]
    if name == "synthetic_test":
        return ["synthetic-test-only"]
    raise ValueError(
        f"unknown dataset_name {name!r}; expected comma2k19, bdd100k, or synthetic_test"
    )


def _quantize_to_int8(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8 quantization. Returns (quantized, scale).

    Reconstruction: ``arr_reconstructed = quantized / scale``. Scale is
    chosen so the max-abs value maps to 127.
    """
    abs_max = float(np.abs(arr).max())
    if abs_max <= 1e-12:
        return np.zeros_like(arr, dtype=np.int8), 64.0  # safe default
    scale = 127.0 / abs_max
    quantized = np.round(arr * scale).clip(-127, 127).astype(np.int8)
    return quantized, scale


def _pca_basis(samples: np.ndarray, n_components: int) -> np.ndarray:
    """Compute top-k PCA components of ``samples`` shape ``(N, ...feature_dims)``.

    Returns array of shape ``(n_components, *feature_dims)``. Uses numpy SVD
    on the mean-centered design matrix; deterministic per
    ``DistillationConfig.random_seed``.
    """
    if samples.ndim < 2:
        raise ValueError(
            f"PCA samples must be at least 2D (N, features); got shape {samples.shape}"
        )
    n_samples = samples.shape[0]
    feature_shape = samples.shape[1:]
    flat = samples.reshape(n_samples, -1).astype(np.float64)
    flat = flat - flat.mean(axis=0, keepdims=True)
    # SVD: flat = U @ diag(S) @ Vt. Top components are first rows of Vt.
    # full_matrices=False for memory efficiency.
    _u, _s, vt = np.linalg.svd(flat, full_matrices=False)
    top = vt[:n_components]  # shape (n_components, num_features)
    return top.reshape((n_components, *feature_shape))


def _aggregate_road_plane_samples(
    frames: Iterator[np.ndarray],
    *,
    grid_shape: tuple[int, int],
    max_frames: int,
) -> np.ndarray:
    """Sample road-plane patches from frames into a (N, H, W, 3) array.

    Each input frame is shape ``(H_img, W_img, 3)`` uint8. We take the
    bottom-third (road plane heuristic), downsample to ``grid_shape``,
    convert to float32 in [0, 1].
    """
    out: list[np.ndarray] = []
    grid_h, grid_w = grid_shape
    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
        if frame.ndim != 3 or frame.shape[2] != 3:
            continue
        h_img = frame.shape[0]
        road_band = frame[2 * h_img // 3 :, :, :].astype(np.float32) / 255.0
        # Simple block-mean downsample to (grid_h, grid_w).
        rh, rw = road_band.shape[:2]
        if rh < grid_h or rw < grid_w:
            continue
        bh = rh // grid_h
        bw = rw // grid_w
        cropped = road_band[: bh * grid_h, : bw * grid_w, :]
        reshaped = cropped.reshape(grid_h, bh, grid_w, bw, 3).mean(axis=(1, 3))
        out.append(reshaped)
    if not out:
        return np.zeros((0, grid_h, grid_w, 3), dtype=np.float32)
    return np.stack(out, axis=0)


def _aggregate_sky_horizon_profiles(
    frames: Iterator[np.ndarray],
    *,
    num_samples: int,
    max_frames: int,
) -> np.ndarray:
    """Sample vertical brightness/chroma profiles into a (N, num_samples, 3) array."""
    out: list[np.ndarray] = []
    for i, frame in enumerate(frames):
        if i >= max_frames:
            break
        if frame.ndim != 3 or frame.shape[2] != 3:
            continue
        h_img = frame.shape[0]
        col_means = frame.mean(axis=1).astype(np.float32) / 255.0
        # Resample column means to num_samples via linear interp.
        idx = np.linspace(0, h_img - 1, num_samples)
        idx_lo = np.floor(idx).astype(int)
        idx_hi = np.minimum(idx_lo + 1, h_img - 1)
        frac = (idx - idx_lo).reshape(-1, 1)
        sampled = (1.0 - frac) * col_means[idx_lo] + frac * col_means[idx_hi]
        out.append(sampled.astype(np.float32))
    if not out:
        return np.zeros((0, num_samples, 3), dtype=np.float32)
    return np.stack(out, axis=0)


def _synthetic_dashcam_frames(
    *, n_frames: int, seed: int, h: int = 480, w: int = 640
) -> Iterator[np.ndarray]:
    """Deterministic synthetic dashcam-like frames for tests / scaffold smoke.

    Generates frames with a sky band (lighter, blue-tinted top), a horizon
    transition, and a road band (darker, gray-toned bottom). NOT a real
    distillation source; tests use this to avoid downloading datasets.
    """
    rng = np.random.default_rng(seed)
    for _i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.float32)
        # Sky band (top third): blue-ish
        frame[: h // 3] = np.array([130, 160, 200], dtype=np.float32)
        # Horizon transition (middle third): mid-gray
        frame[h // 3 : 2 * h // 3] = np.array([100, 110, 120], dtype=np.float32)
        # Road band (bottom third): dark gray
        frame[2 * h // 3 :] = np.array([60, 60, 70], dtype=np.float32)
        # Small per-frame noise so PCA finds non-trivial directions.
        noise = rng.normal(0.0, 5.0, frame.shape).astype(np.float32)
        frame = np.clip(frame + noise, 0.0, 255.0).astype(np.uint8)
        yield frame


def distill_codebook(
    cfg: DistillationConfig,
    *,
    frames: Iterator[np.ndarray] | None = None,
) -> DashcamCodebook:
    """Distill a :class:`DashcamCodebook` from a dashcam-frame iterator.

    Args:
        cfg: Static distillation parameters.
        frames: Iterator of HxWx3 uint8 numpy arrays. If None and dataset_name
            is ``"synthetic_test"``, a deterministic synthetic generator is used.

    Returns:
        A validated :class:`DashcamCodebook` ready to serialize.

    Raises:
        ValueError: dataset_name unknown, missing frames for non-synthetic
            distillation, or PCA failed (too few frames).
        ContestVideoLeakageError: dataset_name suggested contest-video provenance.
    """
    if cfg.dataset_name != "synthetic_test" and frames is None:
        raise ValueError(
            f"distill_codebook requires a frames iterator for "
            f"dataset_name={cfg.dataset_name!r}. Pass frames= or set "
            f"dataset_name='synthetic_test' for the deterministic test "
            f"generator."
        )
    if cfg.dataset_name == "synthetic_test" and frames is None:
        frames = _synthetic_dashcam_frames(
            n_frames=min(cfg.max_frames, 1024), seed=cfg.random_seed
        )

    license_tags = _license_tags_for_dataset(
        cfg.dataset_name, allow_bdd100k_images=cfg.allow_bdd100k_dataset_images
    )

    # Tee the frames iterator: we need it for both road-plane and sky-horizon.
    # The cheapest approach: materialize once into memory (cost-capped by max_frames).
    frames_buffer: list[np.ndarray] = []
    assert frames is not None  # narrowed above
    for i, frame in enumerate(frames):
        if i >= cfg.max_frames:
            break
        frames_buffer.append(frame)

    if len(frames_buffer) < 4:
        raise ValueError(
            f"distill_codebook needs at least 4 frames; got {len(frames_buffer)}"
        )

    road_grid = (ROAD_PLANE_BASIS_SHAPE[1], ROAD_PLANE_BASIS_SHAPE[2])
    road_samples = _aggregate_road_plane_samples(
        iter(frames_buffer), grid_shape=road_grid, max_frames=cfg.max_frames
    )
    if road_samples.shape[0] < cfg.num_road_plane_components:
        # Pad with mean-only direction (synthetic guard).
        pad = np.zeros(
            (cfg.num_road_plane_components - road_samples.shape[0], *road_samples.shape[1:]),
            dtype=np.float32,
        )
        road_samples = (
            pad if road_samples.shape[0] == 0 else np.concatenate([road_samples, pad], axis=0)
        )

    road_basis_float = _pca_basis(road_samples, cfg.num_road_plane_components)
    road_basis_q, road_scale = _quantize_to_int8(road_basis_float.astype(np.float32))

    sky_samples = _aggregate_sky_horizon_profiles(
        iter(frames_buffer),
        num_samples=SKY_HORIZON_PROFILE_SHAPE[0],
        max_frames=cfg.max_frames,
    )
    if sky_samples.shape[0] < 2:
        sky_mean = np.zeros(SKY_HORIZON_PROFILE_SHAPE, dtype=np.float32)
    else:
        sky_mean = sky_samples.mean(axis=0)
    sky_q, sky_scale = _quantize_to_int8(sky_mean.astype(np.float32))

    # Lane curvature PCA — synthetic for now (real distillation requires
    # parsed lane-marker annotations, which Comma2k19 has in pose CSVs).
    # The scaffold uses a deterministic zero basis; real distillation
    # fills this from dataset-specific lane-marker labels.
    lane_basis = np.zeros(LANE_CURVATURE_PCA_SHAPE, dtype=np.float16)

    # Vehicle appearance basis — same scaffold stub.
    vehicle_basis = np.zeros(VEHICLE_APPEARANCE_BASIS_SHAPE, dtype=np.float32)
    vehicle_q, vehicle_scale = _quantize_to_int8(vehicle_basis)

    metadata: dict[str, object] = {
        "road_plane_scale": float(road_scale),
        "sky_horizon_scale": float(sky_scale),
        "vehicle_scale": float(vehicle_scale),
        "dataset_provenance": cfg.dataset_name,
        "dataset_sha256": cfg.dataset_sha256,
        "distillation_version": "v1_scaffold_pca",
        "license_tags": license_tags,
        "num_frames_used": len(frames_buffer),
        "random_seed": cfg.random_seed,
        # Hash the actual distilled basis bytes for downstream byte-determinism
        # checks. Consumers can compare this hash to detect codebook tampering.
        "basis_sha256": hashlib.sha256(
            road_basis_q.tobytes(order="C") + sky_q.tobytes(order="C")
        ).hexdigest(),
    }

    book = DashcamCodebook(
        road_plane_basis=road_basis_q,
        sky_horizon_profile=sky_q,
        lane_curvature_pca=lane_basis,
        vehicle_appearance_basis=vehicle_q,
        metadata=metadata,
    )
    validate_codebook(book)
    return book


def aggregate_local_codebooks(
    local_codebooks: list[DashcamCodebook], *, weights: list[float] | None = None
) -> DashcamCodebook:
    """Sketch of federated aggregation across local Comma edge codebooks.

    Per operator 2026-05-13 "production-deployment alignment + federated
    learning architecture" framing. Each edge device computes its local
    codebook from its own driving data; an aggregator computes the
    weighted average and quantizes back to int8.

    NOT WIRED to any real federated infrastructure; this is the math
    contract so future Comma edge devices know the upstream merge rule.
    The scaffold tests exercise the aggregation math; production federated
    rollout would add differential privacy + auth + transport.
    """
    if not local_codebooks:
        raise ValueError("aggregate_local_codebooks requires at least one codebook")
    n = len(local_codebooks)
    if weights is None:
        weights = [1.0 / n] * n
    elif len(weights) != n:
        raise ValueError(
            f"aggregate_local_codebooks weights count {len(weights)} != codebooks {n}"
        )
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(
            f"aggregate_local_codebooks weights must sum to 1.0; got {sum(weights):.6f}"
        )
    for book in local_codebooks:
        validate_codebook(book)

    def _aggregate_int8(name: str) -> tuple[np.ndarray, float]:
        floats = []
        for book, weight in zip(local_codebooks, weights, strict=True):
            arr = getattr(book, name).astype(np.float32)
            # Normalize the scale-key lookup.
            scale_lookup = {
                "road_plane_basis": "road_plane_scale",
                "sky_horizon_profile": "sky_horizon_scale",
                "vehicle_appearance_basis": "vehicle_scale",
            }[name]
            arr = arr / float(book.metadata[scale_lookup])
            floats.append(arr * weight)
        merged = np.sum(floats, axis=0).astype(np.float32)
        q, scale = _quantize_to_int8(merged)
        return q, scale

    road_q, road_s = _aggregate_int8("road_plane_basis")
    sky_q, sky_s = _aggregate_int8("sky_horizon_profile")
    veh_q, veh_s = _aggregate_int8("vehicle_appearance_basis")

    # Lane curvature is fp16 already — average directly.
    lane_avg = sum(
        book.lane_curvature_pca.astype(np.float32) * w
        for book, w in zip(local_codebooks, weights, strict=True)
    ).astype(np.float16)

    license_tags_union = sorted(
        {
            tag
            for book in local_codebooks
            for tag in book.metadata.get("license_tags", [])
            if isinstance(tag, str)
        }
    )

    meta: dict[str, object] = {
        "road_plane_scale": float(road_s),
        "sky_horizon_scale": float(sky_s),
        "vehicle_scale": float(veh_s),
        "dataset_provenance": "federated_aggregate",
        "distillation_version": "v1_federated_aggregate",
        "license_tags": license_tags_union,
        "num_constituent_codebooks": n,
        "aggregation_weights": weights,
    }
    book = DashcamCodebook(
        road_plane_basis=road_q,
        sky_horizon_profile=sky_q,
        lane_curvature_pca=lane_avg,
        vehicle_appearance_basis=veh_q,
        metadata=meta,
    )
    validate_codebook(book)
    return book


class Comma2k19FrameIterator:
    """Stream dashcam RGB frames from a Comma2k19 chunk directory tree.

    Comma2k19 (`github.com/commaai/comma2k19`, MIT) ships chunks in a
    nested directory layout:

        <chunk_root>/<dongle_id>/<route_id>/<segment_idx>/video.hevc

    Real Phase 2 dispatch points :func:`distill_codebook` at this iterator
    by way of an operator-supplied ``--comma2k19-chunks-dir``. The
    iterator delegates frame decoding to :mod:`av` (PyAV) when present,
    which the canonical Modal training image already mounts.

    **Contest-video-leakage refusal:** every chunk path is run through
    :func:`check_no_contest_video_leakage` BEFORE the iterator yields its
    first frame. Catalog #209 STRICT preflight enforces that EVERY caller
    of :func:`distill_codebook` constructs the iterator through this class
    (or carries a same-line waiver) so the leakage refusal cannot be
    bypassed by handing a raw ``pyav`` iterator straight to the distiller.

    **Deterministic synthetic stub.** Tests / scaffold smoke use
    ``Comma2k19FrameIterator(synthetic=True, n_frames=N, seed=S)`` which
    yields the canonical synthetic dashcam frames from
    :func:`_synthetic_dashcam_frames`. The contract is otherwise
    identical: ``__iter__`` yields ``np.uint8`` ``(H, W, 3)`` arrays.

    Args:
        chunks_dir: Root of the Comma2k19 download. Required when
            ``synthetic=False``.
        max_chunks: Cap on the number of (dongle, route, segment) triples
            visited (each chunk decodes ~60 sec @ 20 Hz ≈ 1200 frames).
        max_frames_per_chunk: Cap on frames decoded per chunk.
        synthetic: If True, yield deterministic synthetic frames instead.
        n_frames: Frame budget for ``synthetic=True``.
        seed: Random seed for the synthetic generator.
        frame_stride: Stride applied during decode (1 = every frame).
    """

    def __init__(
        self,
        chunks_dir: Path | None = None,
        *,
        max_chunks: int = 8,
        max_frames_per_chunk: int = 256,
        synthetic: bool = False,
        n_frames: int = 1024,
        seed: int = 0xDA5C,
        frame_stride: int = 1,
    ) -> None:
        if synthetic:
            if chunks_dir is not None:
                raise ValueError(
                    "synthetic=True is incompatible with chunks_dir; "
                    "the synthetic stub generates its own deterministic frames"
                )
            self._chunks_dir = None
        else:
            if chunks_dir is None:
                raise ValueError(
                    "Comma2k19FrameIterator(chunks_dir=None, synthetic=False) "
                    "is invalid; supply chunks_dir or set synthetic=True"
                )
            chunks_dir = Path(chunks_dir).resolve()
            # CONTEST-VIDEO-LEAKAGE GUARD — refuse before any decode work.
            check_no_contest_video_leakage([chunks_dir])
            if not chunks_dir.exists():
                raise FileNotFoundError(
                    f"Comma2k19 chunks_dir {chunks_dir!r} does not exist; "
                    f"download a chunk from "
                    f"github.com/commaai/comma2k19#download (MIT-licensed) "
                    f"and pass --comma2k19-chunks-dir <path>"
                )
            self._chunks_dir = chunks_dir
        self._max_chunks = int(max_chunks)
        self._max_frames_per_chunk = int(max_frames_per_chunk)
        self._synthetic = bool(synthetic)
        self._n_frames = int(n_frames)
        self._seed = int(seed)
        self._frame_stride = max(1, int(frame_stride))
        # Provenance — populated lazily as chunks/frames are consumed.
        self.provenance: dict[str, object] = {
            "iterator_class": "Comma2k19FrameIterator",
            "synthetic": self._synthetic,
            "chunks_dir": str(self._chunks_dir) if self._chunks_dir else None,
            "max_chunks": self._max_chunks,
            "max_frames_per_chunk": self._max_frames_per_chunk,
            "frame_stride": self._frame_stride,
            "license": "MIT" if not self._synthetic else "synthetic-test-only",
            "source_url": (
                "https://github.com/commaai/comma2k19" if not self._synthetic else None
            ),
            "chunks_visited": [],  # populated by __iter__
            "frame_count_used": 0,
            "dataset_sha256_chunks": [],
        }

    def _iter_chunks(self) -> Iterator[Path]:
        """Yield ``video.hevc`` paths under the Comma2k19 chunk root.

        The Comma2k19 layout is documented at
        ``github.com/commaai/comma2k19#chunk-layout``; we walk the tree
        deterministically (sorted) so distillation is reproducible.
        """
        assert self._chunks_dir is not None
        seen = 0
        for video in sorted(self._chunks_dir.rglob("video.hevc")):
            if seen >= self._max_chunks:
                return
            # Re-validate each candidate path against the leakage guard so
            # symlinks / nested mount points cannot smuggle the contest
            # video in via a deeper chunk subdir.
            check_no_contest_video_leakage([video])
            seen += 1
            yield video

    def _decode_chunk(self, path: Path) -> Iterator[np.ndarray]:
        """Decode frames from a single ``video.hevc`` path via PyAV.

        Raises ImportError if PyAV is not installed; the canonical Modal
        training image carries it.
        """
        try:
            import av  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover — runtime-only import
            raise ImportError(
                "Comma2k19FrameIterator non-synthetic mode requires PyAV "
                "(`uv pip install av`)."
            ) from exc

        container = av.open(str(path))
        try:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "DEFAULT"
            i = 0
            yielded = 0
            for frame in container.decode(stream):
                if yielded >= self._max_frames_per_chunk:
                    break
                if i % self._frame_stride != 0:
                    i += 1
                    continue
                arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                yielded += 1
                i += 1
                yield arr
        finally:
            container.close()

    def __iter__(self) -> Iterator[np.ndarray]:
        """Yield np.uint8 (H, W, 3) frames in deterministic order."""
        if self._synthetic:
            yielded = 0
            for frame in _synthetic_dashcam_frames(
                n_frames=self._n_frames, seed=self._seed
            ):
                yielded += 1
                self.provenance["frame_count_used"] = yielded
                yield frame
            return

        chunks_visited: list[str] = []
        chunks_sha256: list[str] = []
        total_yielded = 0
        for chunk_path in self._iter_chunks():
            chunks_visited.append(str(chunk_path))
            try:
                chunks_sha256.append(
                    hashlib.sha256(chunk_path.read_bytes()).hexdigest()
                )
            except OSError:
                # Skip the SHA on huge chunks where read_bytes is impractical.
                # The provenance still records the chunk path.
                chunks_sha256.append("sha_skipped_large_file")
            for frame in self._decode_chunk(chunk_path):
                total_yielded += 1
                self.provenance["frame_count_used"] = total_yielded
                yield frame
        self.provenance["chunks_visited"] = chunks_visited
        self.provenance["dataset_sha256_chunks"] = chunks_sha256


def write_codebook_to_disk(book: DashcamCodebook, out_path: Path) -> None:
    """Write codebook bytes + metadata sidecar JSON to disk for offline reuse.

    Per CLAUDE.md "Forbidden /tmp paths": ``out_path`` must NOT live under
    ``/tmp/``, ``/var/tmp/``, or ``/private/tmp/``. Use
    ``experiments/results/<lane_id>/<artifact>.bin`` for durable artifacts.
    """
    from tac.substrates.pretrained_driving_prior.codebook import serialize_codebook

    out_str = str(out_path).replace(os.sep, "/")
    if any(out_str.startswith(p) for p in ("/tmp/", "/var/tmp/", "/private/tmp/")):
        raise ValueError(
            f"refusing to write codebook to transient path {out_path!r}; "
            f"use experiments/results/<lane_id>/ per CLAUDE.md 'Forbidden /tmp paths'"
        )
    data = serialize_codebook(book)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar.write_text(json.dumps(book.metadata, sort_keys=True, indent=2))


__all__ = [
    "Comma2k19FrameIterator",
    "ContestVideoLeakageError",
    "DistillationConfig",
    "aggregate_local_codebooks",
    "check_no_contest_video_leakage",
    "distill_codebook",
    "write_codebook_to_disk",
]
