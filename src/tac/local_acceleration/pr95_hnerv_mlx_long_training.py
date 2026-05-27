# SPDX-License-Identifier: MIT
"""PR 95 HNeRV MLX long-training infrastructure.

Sister of ``tac.local_acceleration.pr95_hnerv_mlx`` (canonical MLX
bundle: ``HNeRVDecoderMLX`` + ``HNeRVSyntheticTrainingBundleMLX`` +
``run_pr95_mlx_synthetic_timing_smoke``) and
``tac.local_acceleration.mlx_to_pytorch_export`` (Slot 1 PyTorch export
parity bridge, commit ``44640a985``). This module addresses the
operator's CRITICAL INSIGHT (2026-05-25 verbatim): *"all of the paradigm
class shift candidates be very limited and suffer cargo culting unless
long training runs? that is what the MLX work is ultimately for"*.

The MLX work today (Slot 1 + Stage 3+4+6+7 landings) has been BUILD
scaffolding (synthetic 100-step smokes + canonical export bridge); the
canonical extension pattern proven 7x but NO LONG training has been done
on the canonical contest video ``upstream/videos/0.mkv``. Every
substrate-class-shift candidate (Hinton-distilled scorer / UNIWARD
per-instance x wavelet db4 / Z6 / Z7 / Z8 / Cooperative-receiver / DP1 /
Probe 9 substrate / etc.) is inherently cargo-cult-suspect at $0
short-training scale; LONG training at $0 macOS-MLX is the canonical
apparatus that resolves cargo-cult vs real-signal BEFORE paid dispatch
authorization.

# NUMERIC_TOLERANCE acknowledgment per sister Slot 1 verdict

The MLX outputs produced by this module are APPROXIMATE within rtol=1e-2
absolute (random init max_abs=3.997e-3; trained checkpoint 3.05e-5 per
sister codex parity probe in
``.omx/research/codex_findings_pr95_mlx_full_queue_execution_20260525T173024Z_codex.md``).
The MLX long-training outputs are NEVER promotable to contest-CUDA or
contest-CPU axes; per CLAUDE.md "MLX portable-local-substrate authority"
non-negotiable, every MLX-derived row carries
``evidence_grade="[macOS-MLX research-signal]"`` with
``score_claim=False`` + ``promotion_eligible=False``. Paid CPU+CUDA
auth-eval on Linux x86_64 + NVIDIA contest-compliant hardware remains
the FINAL authority for any score claim.

# Implemented training pattern (RGB-frame MVP; not scorer-faithful yet)

Per the canonical MLX bundle: each "pair" carries a per-pair latent
vector of shape ``(latent_dim,)``. Training maintains
``latents_full: (num_pairs, latent_dim)`` updated alongside decoder
weights. Per-batch sampling picks indices ``idx`` and gathers
``latents_batch = latents_full[idx]`` + ``targets_batch =
targets_full[idx]``; the loss_fn takes ``(decoder, latents_batch,
targets_batch)`` and returns scalar MSE in RGB space at canonical
``(B, 2, 3, 384, 512)`` shape per the canonical PR 95 HNeRVDecoder
output. This module's ``MLXPairIterator`` produces ``(idx,
targets_batch)`` pairs at PyAV-sourced canonical video targets; the
pipeline maintains ``latents_full`` as MLX trainable state.

This is deliberately labeled as a local RGB-frame reconstruction MVP, not a
complete PR95 scorer-faithful training reproduction. SegNet/PoseNet loss,
source-faithful optimizer/schedule/QAT semantics, full-frame shell parity, and
paired contest CPU/CUDA auth eval remain required before any dispatch,
promotion, rank, kill, or score authority can be inferred.

# Catalog discipline

- Catalog #1 MPS noise: this module uses MLX (Metal Performance Shaders
  framework) NOT torch.mps; MLX outputs labeled
  ``[macOS-MLX research-signal]`` per Catalog #192.
- Catalog #192 macOS-CPU advisory: MLX outputs are NOT a substitute for
  ``[contest-CPU]`` Linux x86_64; checkpoint outputs go through the
  canonical Slot 1 export bridge to PyTorch state_dict for paired auth
  eval on a contest-compliant runner.
- Catalog #205 inflate device: any submission archive derived from
  these MLX checkpoints routes inflate device selection through
  ``select_inflate_device`` per the canonical helper.
- Catalog #229 premise verification: every long-training run records
  the source video sha256 + stage hyperparameters + canonical curriculum
  citation (``.omx/research/pr95_8stage_curriculum_forensic_20260513.md``
  + sister recovery ``pr95_curriculum_recovery_20260513_codex.md``)
  BEFORE any checkpoint is persisted.
- Catalog #305 observability surface: emits per-stage + per-epoch loss
  curve telemetry + wall-clock + memory footprint + transition
  diagnostics as queryable JSONL artifacts.
- Catalog #303 cargo-cult audit: per-candidate adapter SHALL document
  which $0 short-training predictions REPLICATE at MLX long-training
  scale vs which DIVERGE (cargo-cult identified at $0; saved paid GPU).
- Catalog #307 paradigm-vs-implementation: candidates that DIVERGE at
  MLX long-training are IMPLEMENTATION-LEVEL falsified per the canonical
  paradigm-vs-implementation classification; paradigm INTACT;
  alternative reducer per Catalog #308 is the canonical research path
  forward.
- Catalog #344 canonical equations registry: queued FORMALIZATION_PENDING
  candidate equation ``mlx_long_training_validation_as_paid_dispatch_authorization_gate_v1``
  formalizes the canonical apparatus contract.
"""

# CHECKPOINT_DISCIPLINE_WAIVED:long_training_infrastructure_module_no_subagent_dispatches_within
# FORMALIZATION_PENDING:queued_for_canonical_equation_registration_landing_memo_documents_routing

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    _MLX_AVAILABLE = True
except ImportError:  # pragma: no cover - environment without MLX
    _MLX_AVAILABLE = False
    mx = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.pr95_hnerv_mlx import HNeRVDecoderMLX, require_mlx

__all__ = [
    "CANONICAL_8STAGE_CURRICULUM",
    "CANONICAL_BASE_CHANNELS",
    "CANONICAL_CONTEST_VIDEO_PATH",
    "CANONICAL_EVAL_SIZE",
    "CANONICAL_LATENT_DIM",
    "INITIAL_SUBSTRATE_ADAPTER_REGISTRY",
    "MLX_LONG_TRAINING_EVIDENCE_GRADE",
    "MLX_LONG_TRAINING_EVIDENCE_TAG",
    "NUMERIC_TOLERANCE_ATOL",
    "NUMERIC_TOLERANCE_RTOL",
    "PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS",
    "PR95_MLX_LONG_TRAINING_EXPORT_PLACEHOLDER_SCHEMA",
    "PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY",
    "PR95_MLX_LONG_TRAINING_FIDELITY_CLASS",
    "PR95_MLX_LONG_TRAINING_FIDELITY_STATUS",
    "PR95_MLX_LONG_TRAINING_PLAN_SCHEMA",
    "PR95_MLX_LONG_TRAINING_PT_EXPORT_BLOCKERS",
    "PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS",
    "PR95_MLX_LONG_TRAINING_TELEMETRY_SCHEMA",
    "CheckpointArtifact",
    "LongTrainingConfig",
    "MLXLongTrainingPipeline",
    "MLXPairIterator",
    "PyAVFrameSource",
    "StageHyperparameters",
    "StageTelemetryRow",
    "SubstrateAdapterScaffold",
    "TrainingTelemetry",
    "build_long_training_plan_report",
    "compute_video_sha256",
    "list_initial_substrate_adapter_registry",
    "register_canonical_provenance",
]

# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------

# Canonical contest video path per CLAUDE.md "Primary duties" + the
# upstream/ pinned snapshot per CLAUDE.md mutation frontier
# non-negotiable.
CANONICAL_CONTEST_VIDEO_PATH: Path = Path("upstream/videos/0.mkv")

# Canonical HNeRV hyperparameters per recovered public PR 95 source at
# .omx/research/pr95_8stage_curriculum_forensic_20260513.md +
# submissions/a1/src/model.py (HNeRVDecoder reference).
CANONICAL_LATENT_DIM: int = 28
CANONICAL_BASE_CHANNELS: int = 36
CANONICAL_EVAL_SIZE: tuple[int, int] = (384, 512)

# Per Slot 1 NUMERIC_TOLERANCE verdict (commit 44640a985 +
# pr95_mlx_pytorch_export_parity_bridge_landed_20260525.md): MLX-PyTorch
# framework arithmetic drift at FP32 is bounded by rtol=1e-2 absolute at
# random init; trained checkpoint drift is two orders tighter (~3.05e-5).
NUMERIC_TOLERANCE_RTOL: float = 1.0e-2
NUMERIC_TOLERANCE_ATOL: float = 5.0e-3

# Canonical evidence grade per CLAUDE.md "MLX portable-local-substrate
# authority" + Catalog #192. MLX long-training outputs are research
# signal; paired CPU+CUDA Linux x86_64 + NVIDIA is the FINAL authority.
MLX_LONG_TRAINING_EVIDENCE_GRADE: str = EVIDENCE_GRADE_MLX
MLX_LONG_TRAINING_EVIDENCE_TAG: str = EVIDENCE_TAG_MLX
PR95_MLX_LONG_TRAINING_PLAN_SCHEMA = "pr95_mlx_long_training_plan.v1"
PR95_MLX_LONG_TRAINING_TELEMETRY_SCHEMA = "pr95_mlx_long_training_telemetry.v1"
PR95_MLX_LONG_TRAINING_EXPORT_PLACEHOLDER_SCHEMA = (
    "pr95_mlx_long_training_deferred_pytorch_export.v1"
)
PR95_MLX_LONG_TRAINING_FIDELITY_CLASS = "rgb_frame_mse_local_mlx_research_mvp"
PR95_MLX_LONG_TRAINING_FIDELITY_STATUS = (
    "local_rgb_frame_mse_mvp_not_segnet_posenet_scorer_faithful"
)
PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS = (
    "not_pr95_1to1_rgb_mse_mvp"
)
PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS: tuple[str, ...] = (
    "local_mlx_long_training_is_research_signal_not_contest_auth_eval",
    "rgb_frame_mse_is_not_segnet_posenet_contest_scorer_loss",
    "source_optimizer_scheduler_qat_parity_not_yet_attested",
    "requires_pytorch_export_forward_parity_on_result_checkpoint",
    "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
    "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
    "requires_segnet_posenet_loss_or_exact_scorer_response_calibration_before_dispatch_authority",
)
PR95_MLX_LONG_TRAINING_PT_EXPORT_BLOCKERS: tuple[str, ...] = (
    "package_tool_must_use_latents_from_pt_for_long_training_checkpoints",
    "requires_pytorch_export_forward_parity_on_result_checkpoint",
    "requires_full_frame_inflate_parity_before_runtime_consumption_claim",
    "requires_paired_contest_cpu_and_cuda_auth_eval_before_score_claim",
)
PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "score_claim_eligible": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "reproduction_claim": False,
    "pr95_1to1_reproduction_claim": False,
    "reproduction_equivalence": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "dispatch_packet_ready": False,
}


# ---------------------------------------------------------------------------
# Stage hyperparameters per the recovered public PR 95 8-stage curriculum
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageHyperparameters:
    """Canonical hparams for one stage of the PR 95 8-stage curriculum.

    Per ``.omx/research/pr95_8stage_curriculum_forensic_20260513.md``
    plus the codex recovery ``pr95_curriculum_recovery_20260513_codex.md``.
    Sister Slot 2 (``lane_pr95_mlx_stage_hparams_source_faithful_audit_*``)
    is auditing whether the canonical 3000-epoch full reference should be
    abbreviated to ~500 epochs at MLX scale; the defaults below are the
    canonical public PR 95 values; per-call overrides at the pipeline
    level allow swap to Slot 2's verdict when it lands.
    """

    stage_index: int
    name: str
    epochs: int
    learning_rate: float
    batch_size: int
    notes: str = ""

    def __post_init__(self) -> None:
        if not (1 <= self.stage_index <= 8):
            raise ValueError(
                f"stage_index must be in [1, 8]; got {self.stage_index}"
            )
        if self.epochs < 1:
            raise ValueError(
                f"epochs must be >= 1; got {self.epochs}"
            )
        if not (1e-7 <= self.learning_rate <= 1.0):
            raise ValueError(
                f"learning_rate out of canonical band [1e-7, 1.0]: "
                f"got {self.learning_rate}"
            )
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1; got {self.batch_size}"
            )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# Canonical 8-stage curriculum per the forensic recovery memo. Values
# here MUST NOT be edited without a sister landing memo that updates
# both the public PR 95 forensic citation AND the operator-approved
# Slot 2 audit verdict (when it lands).
CANONICAL_8STAGE_CURRICULUM: tuple[StageHyperparameters, ...] = (
    StageHyperparameters(
        stage_index=1,
        name="warmup_low_lr",
        epochs=300,
        learning_rate=1e-4,
        batch_size=2,
        notes="Stage 1 warmup at low LR; per recovered public PR 95 source.",
    ),
    StageHyperparameters(
        stage_index=2,
        name="ramp_lr",
        epochs=300,
        learning_rate=5e-4,
        batch_size=2,
        notes="Stage 2 LR ramp; transition into main training band.",
    ),
    StageHyperparameters(
        stage_index=3,
        name="main_train_band_a",
        epochs=400,
        learning_rate=1e-3,
        batch_size=2,
        notes="Stage 3 main training band; canonical PR 95 reference hparams.",
    ),
    StageHyperparameters(
        stage_index=4,
        name="main_train_band_b",
        epochs=400,
        learning_rate=1e-3,
        batch_size=2,
        notes="Stage 4 main training band continuation.",
    ),
    StageHyperparameters(
        stage_index=5,
        name="refine_lr_decay",
        epochs=400,
        learning_rate=5e-4,
        batch_size=2,
        notes="Stage 5 refinement with LR decay step.",
    ),
    StageHyperparameters(
        stage_index=6,
        name="polish_lr_decay",
        epochs=400,
        learning_rate=2e-4,
        batch_size=2,
        notes="Stage 6 polish with further LR decay.",
    ),
    StageHyperparameters(
        stage_index=7,
        name="finetune_low_lr",
        epochs=400,
        learning_rate=1e-4,
        batch_size=2,
        notes="Stage 7 fine-tune at low LR.",
    ),
    StageHyperparameters(
        stage_index=8,
        name="converge_low_lr",
        epochs=400,
        learning_rate=5e-5,
        batch_size=2,
        notes="Stage 8 convergence at very low LR; canonical curriculum end.",
    ),
)


# ---------------------------------------------------------------------------
# Long-training pipeline configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LongTrainingConfig:
    """Canonical configuration for one MLX long-training run."""

    # Source video for source-faithful training; default
    # ``upstream/videos/0.mkv`` per CLAUDE.md pinned snapshot.
    source_video_path: Path = CANONICAL_CONTEST_VIDEO_PATH

    # Architecture hparams; defaults match the canonical PR 95 reference.
    latent_dim: int = CANONICAL_LATENT_DIM
    base_channels: int = CANONICAL_BASE_CHANNELS
    eval_size: tuple[int, int] = CANONICAL_EVAL_SIZE

    # Per-stage curriculum. Defaults to the canonical 8-stage reference.
    curriculum: Sequence[StageHyperparameters] = CANONICAL_8STAGE_CURRICULUM

    # Output checkpoint root (per-stage checkpoints will be written
    # under this root). MUST NOT be under ``/tmp/`` per CLAUDE.md
    # FORBIDDEN_PATTERNS "Forbidden /tmp paths in any persisted
    # artifact (the transient-evidence trap)"; canonical location is
    # ``experiments/results/<lane_id>_<timestamp>/``.
    checkpoint_root: Path = Path(
        "experiments/results/pr95_mlx_long_training_checkpoints"
    )

    # Telemetry output path; canonical
    # ``.omx/state/mlx_long_training_telemetry_<lane>_<utc>.jsonl``
    # OR per-checkpoint sister file.
    telemetry_path: Path | None = None

    # Smoke mode: epochs floor + checkpoint-every cadence used to verify
    # pipeline correctness within minutes rather than the full multi-hour
    # 3000-epoch run.
    smoke_mode: bool = False
    smoke_epochs_per_stage: int = 100
    checkpoint_every_epochs: int = 100

    # Max frames to load from source video (None = all). Defaults to a
    # smoke-safe ceiling; production-grade runs override via the CLI to
    # the canonical 1200-frame full set.
    max_frames: int | None = None

    # Random seed for the MLX-native frame index iterator; per Catalog
    # #305 observability + deterministic-reproducibility (9-dim Dim 7).
    random_seed: int = 0

    # Provenance fields per Catalog #229 + #323 canonical Provenance.
    lane_id: str = (
        "lane_pr95_mlx_long_training_infrastructure_and_substrate_class_shift_candidate_validation_pipeline_20260525"
    )
    canonical_citation: str = (
        ".omx/research/pr95_8stage_curriculum_forensic_20260513.md"
    )
    operator_run_label: str = ""

    def total_epochs(self) -> int:
        """Effective total epochs across all stages (smoke-aware)."""

        if self.smoke_mode:
            return self.smoke_epochs_per_stage * len(self.curriculum)
        return sum(s.epochs for s in self.curriculum)

    def effective_epochs_for_stage(
        self, stage: StageHyperparameters
    ) -> int:
        """Effective epoch count for a stage (smoke-aware)."""

        if self.smoke_mode:
            return min(self.smoke_epochs_per_stage, stage.epochs)
        return stage.epochs


# ---------------------------------------------------------------------------
# PyAV frame source
# ---------------------------------------------------------------------------


class PyAVFrameSource:
    """Source-faithful per-frame iterator over ``upstream/videos/0.mkv``.

    Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline"
    L1 (substrate must be score-aware), this iterator decodes the
    EXACT canonical contest video that the upstream evaluator uses, at
    the canonical 384x512 contest resolution. The frame count and
    sha256 of the source video are recorded in the telemetry for the
    Catalog #229 PV invariant.

    The iterator returns numpy arrays of shape ``(H, W, 3)`` in RGB
    uint8 order; downstream MLX adapters convert to ``mx.array`` in
    the desired layout via the pipeline's transform function.

    NUMERIC_TOLERANCE NOTE: PyAV decode is deterministic on macOS for a
    given input video + ffmpeg version; tiny variations in decode
    arithmetic across ffmpeg versions are within the
    NUMERIC_TOLERANCE_RTOL=1e-2 band declared above.
    """

    def __init__(
        self,
        video_path: Path,
        height: int = 384,
        width: int = 512,
        max_frames: int | None = None,
    ) -> None:
        if not video_path.is_file():
            raise FileNotFoundError(
                f"canonical contest video not found at {video_path!s}; "
                "ensure upstream/ pinned snapshot is in place per "
                "CLAUDE.md mutation frontier non-negotiable"
            )
        self.video_path = video_path
        self.height = height
        self.width = width
        self.max_frames = max_frames
        self._frame_count: int | None = None

    def frame_count(self) -> int:
        """Total frame count of the source video (cached)."""

        if self._frame_count is not None:
            return self._frame_count
        try:
            import av  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PyAV is required for source-faithful MLX long training; "
                "install via `uv pip install av`"
            ) from exc
        # The canonical contest video may report ``stream.frames == 0`` in
        # PyAV (no precomputed frame-count metadata); fall back to a manual
        # demux+decode count.
        container = av.open(str(self.video_path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            count = int(stream.frames) if stream.frames and stream.frames > 0 else 0
            if count == 0:
                count = sum(1 for _ in container.decode(video=0))
            self._frame_count = count
            return count
        finally:
            container.close()

    def iter_frames(self) -> Iterator[Any]:
        """Yield numpy uint8 RGB frames at the canonical resolution."""

        try:
            import av  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "PyAV + numpy required for MLX long training; "
                "install via `uv pip install av numpy`"
            ) from exc
        container = av.open(str(self.video_path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            yielded = 0
            for packet in container.demux(stream):
                for frame in packet.decode():
                    if self.max_frames is not None and yielded >= self.max_frames:
                        return
                    rgb = frame.to_ndarray(format="rgb24")
                    if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
                        rgb = _resize_rgb_uint8(
                            rgb, self.height, self.width
                        )
                    yield rgb
                    yielded += 1
        finally:
            container.close()


def _resize_rgb_uint8(rgb_array: Any, target_h: int, target_w: int) -> Any:
    """Resize a numpy uint8 RGB array to target H x W via PIL."""

    try:
        import numpy as np  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PIL + numpy required for frame resize; "
            "install via `uv pip install Pillow numpy`"
        ) from exc
    img = Image.fromarray(rgb_array)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return np.asarray(img, dtype=rgb_array.dtype)


# ---------------------------------------------------------------------------
# MLX-native pair iterator
# ---------------------------------------------------------------------------


class MLXPairIterator:
    """MLX-native iterator producing (frame_index, target_frame) pairs.

    Per the canonical PR 95 training reference, each training step samples a
    pair-index from adjacent video-frame pairs; the HNeRVDecoder consumes
    the per-pair latent (the pipeline owns + trains the ``latents_full``
    state) and produces two RGB frames that are compared against the source
    target pair at the canonical 384x512 resolution. The iterator preserves
    source provenance by keying every yielded pair against a (video_sha256,
    frame_index) tuple, which the telemetry layer records per Catalog #305.

    Frames are pre-loaded into memory (canonical 1200-frame contest video at
    384x512x3 = ~700 MB float32; well within M5 Max memory budget). Each
    ``sample_batch(B)`` yields ``(mx_pair_idx, mx_target_rgb_pair)`` with
    targets shaped ``(B, 2, H, W, 3)``. Therefore the full contest video yields
    600 trainable latents, matching the public PR95 archive contract.
    """

    def __init__(
        self,
        frame_source: PyAVFrameSource,
        random_seed: int = 0,
        pair_strategy: str = "next_frame_pair",
    ) -> None:
        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is required for MLXPairIterator; "
                "install via `uv pip install mlx`"
            )
        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "numpy required for MLXPairIterator"
            ) from exc
        if pair_strategy not in {"next_frame_pair", "single_frame"}:
            raise ValueError(
                f"unknown pair_strategy: {pair_strategy!r}; "
                "expected 'next_frame_pair' or 'single_frame'"
            )
        self.frame_source = frame_source
        self.pair_strategy = pair_strategy
        self._rng_state = np.random.default_rng(seed=random_seed)
        # Pre-load all frames into a numpy buffer for deterministic
        # sampling. Memory footprint: ~700 MB for 1200 frames at 384x512x3
        # float32; well within M5 Max budget.
        frames_uint8: list[Any] = list(frame_source.iter_frames())
        if len(frames_uint8) == 0:
            raise RuntimeError(
                f"no frames decoded from {frame_source.video_path!s}; "
                "check video file is valid + readable"
            )
        stacked = np.stack(frames_uint8, axis=0)  # (T, H, W, 3) uint8
        self._frames_np = stacked
        self._frame_count = stacked.shape[0]
        if self.pair_strategy == "next_frame_pair":
            self._pair_count = self._frame_count // 2
            if self._pair_count < 1:
                raise RuntimeError(
                    "next_frame_pair strategy requires at least two decoded frames"
                )
        else:
            self._pair_count = self._frame_count
        self._mx_frames_cache: Any = None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def pair_count(self) -> int:
        return self._pair_count

    def _mx_frames(self) -> Any:
        """Lazy-build MLX float32 normalized frames in (T, H, W, 3)."""

        if self._mx_frames_cache is None:
            assert mx is not None
            arr = mx.array(self._frames_np, dtype=mx.float32) / 255.0
            self._mx_frames_cache = arr
        return self._mx_frames_cache

    def sample_batch(self, batch_size: int) -> tuple[Any, Any]:
        """Sample a batch of ``(indices, targets)`` for one training step.

        Returns:
            indices: MLX int32 array of shape ``(batch_size,)``.
            targets: MLX float32 array of shape ``(batch_size, 2, H, W, 3)``
                containing RGB target frame pairs normalized to ``[0, 1]``.
        """

        assert mx is not None
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        sampled = self._rng_state.integers(low=0, high=self._pair_count, size=batch_size)
        indices_mx = mx.array(sampled.astype("int32"))
        frames_mx = self._mx_frames()
        if self.pair_strategy == "next_frame_pair":
            frame_indices_mx = mx.array(
                [[int(index) * 2, int(index) * 2 + 1] for index in sampled],
                dtype=mx.int32,
            )
            targets_mx = frames_mx[frame_indices_mx]
        else:
            target = frames_mx[indices_mx]
            targets_mx = mx.stack([target, target], axis=1)
        return indices_mx, targets_mx


class _LongTrainingBundleMLX(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Decoder plus trainable per-pair latents for long-training."""

    def __init__(
        self,
        *,
        latent_count: int,
        latent_dim: int,
        base_channels: int,
        eval_size: tuple[int, int],
        seed: int,
    ) -> None:
        require_mlx()
        super().__init__()
        key = mx.random.key(seed)  # type: ignore[union-attr]
        self.latents = mx.random.normal(  # type: ignore[union-attr]
            (latent_count, latent_dim),
            key=key,
        ) * 0.01
        self.decoder = HNeRVDecoderMLX(
            latent_dim=latent_dim,
            base_channels=base_channels,
            eval_size=eval_size,
            output_layout="n2chw",
        )

    def __call__(self, indices: Any) -> Any:
        return self.decoder(mx.take(self.latents, indices, axis=0))  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Telemetry per Catalog #305 observability surface
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StageTelemetryRow:
    """One row of stage-level telemetry per Catalog #305 observability."""

    stage_index: int
    stage_name: str
    epoch_within_stage: int
    global_epoch: int
    loss: float
    learning_rate: float
    batch_size: int
    wall_clock_seconds: float
    mlx_peak_memory_bytes: int
    timestamp_utc: str
    notes: str = ""

    def __post_init__(self) -> None:
        # NaN detection: NaN != NaN.
        if self.loss != self.loss:
            raise ValueError(
                f"loss is NaN at stage {self.stage_index}, "
                f"epoch {self.global_epoch}"
            )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class TrainingTelemetry:
    """Aggregated training telemetry for one long-training run."""

    rows: list[StageTelemetryRow] = dataclasses.field(default_factory=list)
    lane_id: str = ""
    source_video_sha256: str = ""
    source_video_frame_count: int | None = None
    source_video_frame_count_scope: str = "not_decoded"
    max_frames: int | None = None
    canonical_citation: str = ""
    run_started_utc: str = ""
    run_completed_utc: str = ""
    operator_run_label: str = ""
    evidence_grade: str = MLX_LONG_TRAINING_EVIDENCE_GRADE
    evidence_tag: str = MLX_LONG_TRAINING_EVIDENCE_TAG
    score_claim: bool = False
    score_claim_valid: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    promotable: bool = False
    dispatch_attempted: bool = False
    gpu_launched: bool = False
    dispatch_packet_ready: bool = False

    def append_row(self, row: StageTelemetryRow) -> None:
        self.rows.append(row)

    def persist(self, telemetry_path: Path) -> None:
        """Persist telemetry to a canonical JSONL file."""

        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        header = {
            "schema_version": PR95_MLX_LONG_TRAINING_TELEMETRY_SCHEMA,
            "lane_id": self.lane_id,
            "source_video_sha256": self.source_video_sha256,
            "source_video_frame_count": self.source_video_frame_count,
            "source_video_frame_count_scope": self.source_video_frame_count_scope,
            "max_frames": self.max_frames,
            "training_fidelity_class": PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
            "training_fidelity_status": PR95_MLX_LONG_TRAINING_FIDELITY_STATUS,
            "reproduction_equivalence_class": PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS,
            "canonical_citation": self.canonical_citation,
            "run_started_utc": self.run_started_utc,
            "run_completed_utc": self.run_completed_utc,
            "operator_run_label": self.operator_run_label,
            "evidence_grade": self.evidence_grade,
            "evidence_tag": self.evidence_tag,
            "row_count": len(self.rows),
            **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
        }
        with open(telemetry_path, "w") as handle:
            handle.write(json.dumps(header, sort_keys=True))
            handle.write("\n")
            for row in self.rows:
                handle.write(json.dumps(row.as_dict(), sort_keys=True))
                handle.write("\n")


@dataclasses.dataclass(frozen=True)
class CheckpointArtifact:
    """Canonical record of one MLX long-training checkpoint."""

    stage_index: int
    global_epoch: int
    mlx_checkpoint_path: Path
    pytorch_state_dict_path: Path
    latents_path: Path | None
    loss_at_checkpoint: float
    timestamp_utc: str
    source_video_sha256: str
    pytorch_export_succeeded: bool = False
    pytorch_export_deferred_path: Path | None = None
    pytorch_export_manifest_path: Path | None = None
    pytorch_export_forward_parity_established: bool = False
    runtime_consumption_proof_established: bool = False
    trained_latents_exported: bool = False
    evidence_grade: str = MLX_LONG_TRAINING_EVIDENCE_GRADE
    evidence_tag: str = MLX_LONG_TRAINING_EVIDENCE_TAG
    score_claim: bool = False
    score_claim_valid: bool = False
    score_claim_eligible: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    promotable: bool = False
    reproduction_claim: bool = False
    pr95_1to1_reproduction_claim: bool = False
    reproduction_equivalence: bool = False

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["mlx_checkpoint_path"] = str(self.mlx_checkpoint_path)
        d["pytorch_state_dict_path"] = str(self.pytorch_state_dict_path)
        d["latents_path"] = None if self.latents_path is None else str(self.latents_path)
        d["pytorch_export_deferred_path"] = (
            None
            if self.pytorch_export_deferred_path is None
            else str(self.pytorch_export_deferred_path)
        )
        d["pytorch_export_manifest_path"] = (
            None
            if self.pytorch_export_manifest_path is None
            else str(self.pytorch_export_manifest_path)
        )
        d["training_fidelity_class"] = PR95_MLX_LONG_TRAINING_FIDELITY_CLASS
        d["training_fidelity_status"] = PR95_MLX_LONG_TRAINING_FIDELITY_STATUS
        d["reproduction_equivalence_class"] = PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
        d.update(PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY)
        d["exact_readiness_refusal"] = {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": list(PR95_MLX_LONG_TRAINING_PT_EXPORT_BLOCKERS),
        }
        return d


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def compute_video_sha256(video_path: Path) -> str:
    """Compute the sha256 of a source video for Catalog #229 PV provenance."""

    hasher = hashlib.sha256()
    with open(video_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_video_frame_count_scope(
    config: LongTrainingConfig,
    source_video_frame_count: int | None,
) -> str:
    if source_video_frame_count is None:
        return "not_decoded"
    if config.max_frames is not None:
        return "max_frames_cap"
    return "full_video_decode"


def register_canonical_provenance(
    config: LongTrainingConfig,
    source_video_sha256: str,
    source_video_frame_count: int | None,
) -> dict[str, Any]:
    """Build a canonical Provenance row for the long-training run.

    Per Catalog #323 canonical Provenance umbrella + Catalog #287
    docstring-overstatement protection: every score-claim row that
    references this MLX long-training run carries this Provenance
    object so the downstream consumer (cathedral autopilot ranker /
    Catalog #341 routing-markers / etc.) can route correctly.
    """

    return {
        "schema": "pr95_mlx_long_training_provenance.v1",
        "kind": "macos_mlx_research_signal",
        "training_fidelity_class": PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
        "training_fidelity_status": PR95_MLX_LONG_TRAINING_FIDELITY_STATUS,
        "reproduction_equivalence_class": PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS,
        "evidence_grade": MLX_LONG_TRAINING_EVIDENCE_GRADE,
        "evidence_tag": MLX_LONG_TRAINING_EVIDENCE_TAG,
        "axis_tag": MLX_LONG_TRAINING_EVIDENCE_TAG,
        "score_axis": MLX_LONG_TRAINING_EVIDENCE_TAG,
        "hardware_substrate": "darwin_arm64_macos_mlx",
        "lane_id": config.lane_id,
        "canonical_citation": config.canonical_citation,
        "source_video_path": str(config.source_video_path),
        "source_video_sha256": source_video_sha256,
        "source_video_frame_count": source_video_frame_count,
        "source_video_frame_count_scope": _source_video_frame_count_scope(
            config,
            source_video_frame_count,
        ),
        "max_frames": config.max_frames,
        "latent_dim": config.latent_dim,
        "base_channels": config.base_channels,
        "eval_size": list(config.eval_size),
        "stage_curriculum_summary": [
            {
                "stage_index": s.stage_index,
                "name": s.name,
                "epochs": s.epochs,
                "learning_rate": s.learning_rate,
                "batch_size": s.batch_size,
            }
            for s in config.curriculum
        ],
        "smoke_mode": config.smoke_mode,
        "smoke_epochs_per_stage": config.smoke_epochs_per_stage,
        "checkpoint_every_epochs": config.checkpoint_every_epochs,
        "random_seed": config.random_seed,
        "numeric_tolerance_rtol": NUMERIC_TOLERANCE_RTOL,
        "numeric_tolerance_atol": NUMERIC_TOLERANCE_ATOL,
        "numeric_tolerance_anchor": (
            "sister_slot_1_landing_memo_pr95_mlx_pytorch_export_parity_bridge_landed_20260525.md"
        ),
        "captured_at_utc": _utc_now_iso(),
        **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": list(PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS),
        },
    }


def build_long_training_plan_report(
    config: LongTrainingConfig,
    *,
    mode: str = "plan_only",
    output_report_path: Path | None = None,
    source_video_sha256: str | None = None,
    source_video_frame_count: int | None = None,
    telemetry_path: Path | None = None,
    checkpoint_artifacts: Sequence[CheckpointArtifact] = (),
    command: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a queue-observable long-training plan or smoke report.

    This report is intentionally useful before an expensive local MLX run starts:
    the queue can observe the intended artifacts, candidate adapters, exact-
    readiness blockers, and false-authority contract without inferring score or
    dispatch authority from a local substrate.
    """

    provenance = register_canonical_provenance(
        config,
        source_video_sha256 or "",
        source_video_frame_count,
    )
    frame_count_scope = _source_video_frame_count_scope(
        config,
        source_video_frame_count,
    )
    telemetry_paths = [telemetry_path.as_posix()] if telemetry_path is not None else []
    checkpoint_rows = [artifact.as_dict() for artifact in checkpoint_artifacts]
    command_args = [str(item) for item in command]
    recommended_execution: dict[str, Any] | None = None
    if output_report_path is not None and command_args:
        output_report = output_report_path.as_posix()
        recommended_execution = {
            "tool": "tools/run_pr95_mlx_long_training.py",
            "resource_kind": (
                "local_mlx" if "--execute-smoke" in command_args else "local_cpu"
            ),
            "authority_kind": "macos_mlx_research_signal",
            "output_manifest": output_report,
            "telemetry_path": None
            if telemetry_path is None
            else telemetry_path.as_posix(),
            "python_command_args": command_args,
            "extra_artifact_postconditions": [
                {
                    "type": "json_equals",
                    "path": output_report,
                    "key": "schema",
                    "equals": PR95_MLX_LONG_TRAINING_PLAN_SCHEMA,
                },
                {
                    "type": "json_equals",
                    "path": output_report,
                    "key": "ready_for_exact_eval_dispatch",
                    "equals": False,
                },
                {
                    "type": "json_equals",
                    "path": output_report,
                    "key": "training_fidelity_class",
                    "equals": PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
                },
                {
                    "type": "json_equals",
                    "path": output_report,
                    "key": "reproduction_equivalence",
                    "equals": False,
                },
                {
                    "type": "json_false_authority",
                    "path": output_report,
                    "required_false": sorted(
                        PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY
                    ),
                    "false_or_missing": [],
                },
            ],
            **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
        }
    return {
        "schema": PR95_MLX_LONG_TRAINING_PLAN_SCHEMA,
        "schema_version": PR95_MLX_LONG_TRAINING_PLAN_SCHEMA,
        "generated_utc": _utc_now_iso(),
        "mode": mode,
        "lane_id": config.lane_id,
        "source_video_path": config.source_video_path.as_posix(),
        "training_fidelity_class": PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
        "training_fidelity_status": PR95_MLX_LONG_TRAINING_FIDELITY_STATUS,
        "reproduction_equivalence_class": PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS,
        "source_video_exists": config.source_video_path.exists(),
        "source_video_sha256": source_video_sha256,
        "source_video_frame_count": source_video_frame_count,
        "source_video_frame_count_scope": frame_count_scope,
        "max_frames": config.max_frames,
        "checkpoint_root": config.checkpoint_root.as_posix(),
        "telemetry_path": None if telemetry_path is None else telemetry_path.as_posix(),
        "artifact_paths": telemetry_paths,
        "pullback_artifact_paths": telemetry_paths,
        "operator_run_label": config.operator_run_label,
        "stage_count": len(config.curriculum),
        "total_epochs": config.total_epochs(),
        "smoke_mode": config.smoke_mode,
        "smoke_epochs_per_stage": config.smoke_epochs_per_stage,
        "checkpoint_every_epochs": config.checkpoint_every_epochs,
        "random_seed": config.random_seed,
        "latent_dim": config.latent_dim,
        "base_channels": config.base_channels,
        "eval_size": list(config.eval_size),
        "candidate_registry": [
            adapter.as_dict() for adapter in INITIAL_SUBSTRATE_ADAPTER_REGISTRY
        ],
        "candidate_registry_count": len(INITIAL_SUBSTRATE_ADAPTER_REGISTRY),
        "checkpoint_artifacts": checkpoint_rows,
        "checkpoint_artifact_count": len(checkpoint_rows),
        "evidence_grade": MLX_LONG_TRAINING_EVIDENCE_GRADE,
        "evidence_tag": MLX_LONG_TRAINING_EVIDENCE_TAG,
        "axis_tag": MLX_LONG_TRAINING_EVIDENCE_TAG,
        "score_axis": MLX_LONG_TRAINING_EVIDENCE_TAG,
        "readiness_blockers": list(PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS),
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": list(PR95_MLX_LONG_TRAINING_EXACT_READINESS_BLOCKERS),
        },
        "canonical_provenance": provenance,
        "command": command_args,
        "recommended_execution": recommended_execution,
        "authority_status": (
            "PR95 MLX long-training is a local RGB-frame reconstruction MVP. It "
            "can select follow-up candidates, but SegNet/PoseNet or calibrated "
            "scorer loss plus exact CPU/CUDA auth eval remain required for score, "
            "promotion, dispatch readiness, rank, or kill authority."
        ),
        **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
    }


def _utc_now_iso() -> str:
    """Return current UTC time in ISO-8601 (Z-suffix) format."""

    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# MLX long-training pipeline
# ---------------------------------------------------------------------------


class MLXLongTrainingPipeline:
    """Canonical MLX long-training pipeline for HNeRV reconstruction.

    Wires together: HNeRV hparams -> ``HNeRVDecoderMLX`` ->
    ``PyAVFrameSource`` -> ``MLXPairIterator`` -> per-pair latents
    (trainable) -> Adam optimizer -> canonical 8-stage curriculum loop
    -> per-stage checkpoint persistence (MLX-native + PyTorch state_dict
    via Slot 1 export bridge) -> ``TrainingTelemetry`` JSONL emission
    per Catalog #305.

    The pipeline ENFORCES the canonical NUMERIC_TOLERANCE semantics
    (outputs labeled ``[macOS-MLX research-signal]`` per Catalog #192;
    NEVER promotable to ``[contest-CPU]`` or ``[contest-CUDA]`` per
    CLAUDE.md "MPS auth eval is NOISE" sister discipline applied at the
    MLX framework boundary).
    """

    def __init__(self, config: LongTrainingConfig) -> None:
        if not _MLX_AVAILABLE:
            raise RuntimeError(
                "MLX is required for MLXLongTrainingPipeline; "
                "install via `uv pip install mlx`"
            )
        self.config = config
        self._source_video_sha256: str | None = None
        self._source_video_frame_count: int | None = None
        self._bundle: Any = None
        self._decoder: Any = None
        self._frame_source: PyAVFrameSource | None = None
        self._pair_iterator: MLXPairIterator | None = None
        self._optimizer: Any = None
        self._latents_full: Any = None  # MLX (num_pairs, latent_dim)
        self._global_epoch: int = 0
        self._telemetry: TrainingTelemetry = TrainingTelemetry(
            lane_id=config.lane_id,
            canonical_citation=config.canonical_citation,
            operator_run_label=config.operator_run_label,
        )
        self._checkpoint_artifacts: list[CheckpointArtifact] = []

    def setup(self) -> None:
        """One-time setup: compute provenance + initialize MLX components."""

        assert mx is not None
        # Catalog #229 PV: record source video sha256 + frame count
        # BEFORE any training begins.
        self._source_video_sha256 = compute_video_sha256(
            self.config.source_video_path
        )
        self._frame_source = PyAVFrameSource(
            video_path=self.config.source_video_path,
            height=self.config.eval_size[0],
            width=self.config.eval_size[1],
            max_frames=self.config.max_frames,
        )
        self._pair_iterator = MLXPairIterator(
            frame_source=self._frame_source,
            random_seed=self.config.random_seed,
        )
        self._source_video_frame_count = self._pair_iterator.frame_count
        # MLX bundle; canonical HNeRV hparams plus trainable per-pair latents.
        self._bundle = _LongTrainingBundleMLX(
            latent_count=self._pair_iterator.pair_count,
            latent_dim=self.config.latent_dim,
            base_channels=self.config.base_channels,
            eval_size=self.config.eval_size,
            seed=self.config.random_seed,
        )
        self._decoder = self._bundle.decoder
        self._latents_full = self._bundle.latents
        # Optimizer at the first stage's LR (per-stage LR swaps in
        # ``run_stage``).
        first_stage_lr = self.config.curriculum[0].learning_rate
        self._optimizer = optim.Adam(learning_rate=first_stage_lr)
        # Telemetry header.
        self._telemetry.source_video_sha256 = self._source_video_sha256
        self._telemetry.source_video_frame_count = (
            self._source_video_frame_count
        )
        self._telemetry.source_video_frame_count_scope = (
            _source_video_frame_count_scope(
                self.config,
                self._source_video_frame_count,
            )
        )
        self._telemetry.max_frames = self.config.max_frames
        self._telemetry.run_started_utc = _utc_now_iso()

    def loss_fn(
        self,
        bundle: Any,
        indices: Any,
        targets_batch: Any,
    ) -> Any:
        """Canonical MSE reconstruction loss in RGB space.

        This Phase-1 loss is plain RGB MSE. It is not a contest
        SegNet/PoseNet scorer-loss reproduction and it carries no score or
        dispatch authority. Sister substrate adapters (per
        ``SubstrateAdapterScaffold`` below) can extend it with scorer-aware
        losses for later queue-owned probes.

        Args:
            bundle: ``_LongTrainingBundleMLX`` with decoder + trainable latents.
            indices: MLX int32 ``(B,)`` selecting trainable latents.
            targets_batch: MLX float32 ``(B, 2, H, W, 3)`` normalized [0, 1].
        """

        assert mx is not None
        # Decoder output shape per canonical PR 95 reference:
        # ``(B, 2, 3, H, W)`` in [0, 255].
        decoded = bundle(indices)
        decoded_pair = decoded / 255.0  # (B, 2, 3, H, W) normalized
        decoded_b2hwc = mx.transpose(decoded_pair, (0, 1, 3, 4, 2))
        diff = decoded_b2hwc - targets_batch
        loss = mx.mean(diff * diff)
        return loss

    def training_step(
        self,
        indices: Any,
        targets: Any,
    ) -> float:
        """One training step: forward + loss + backward + optimizer.update.

        Returns the scalar loss value (Python float).
        """

        assert mx is not None
        assert nn is not None
        assert self._bundle is not None
        assert self._optimizer is not None

        # Compute gradients through the full bundle so both decoder weights and
        # per-pair latents are updated. Earlier drafts updated decoder weights
        # only, which made the PR95-faithful long-training claim false.
        loss_and_grad_fn = nn.value_and_grad(self._bundle, self.loss_fn)
        loss_value, grads = loss_and_grad_fn(
            self._bundle, indices, targets
        )
        self._optimizer.update(self._bundle, grads)
        # Force MLX to evaluate (lazy eval graph).
        mx.eval(self._bundle.parameters(), self._optimizer.state)
        self._decoder = self._bundle.decoder
        self._latents_full = self._bundle.latents
        return float(loss_value.item())

    def run_stage(
        self,
        stage: StageHyperparameters,
        on_epoch_end: Callable[[StageTelemetryRow], None] | None = None,
    ) -> None:
        """Run one stage of the canonical curriculum."""

        assert mx is not None
        assert optim is not None
        assert self._pair_iterator is not None
        # Adjust optimizer learning rate at stage boundary.
        self._optimizer = optim.Adam(learning_rate=stage.learning_rate)
        stage_epochs = self.config.effective_epochs_for_stage(stage)
        for epoch_in_stage in range(stage_epochs):
            self._global_epoch += 1
            epoch_start = time.time()
            indices, targets = self._pair_iterator.sample_batch(
                stage.batch_size
            )
            loss = self.training_step(indices, targets)
            epoch_wall = time.time() - epoch_start
            row = StageTelemetryRow(
                stage_index=stage.stage_index,
                stage_name=stage.name,
                epoch_within_stage=epoch_in_stage + 1,
                global_epoch=self._global_epoch,
                loss=loss,
                learning_rate=stage.learning_rate,
                batch_size=stage.batch_size,
                wall_clock_seconds=epoch_wall,
                mlx_peak_memory_bytes=_mlx_peak_memory_bytes(),
                timestamp_utc=_utc_now_iso(),
            )
            self._telemetry.append_row(row)
            if on_epoch_end is not None:
                on_epoch_end(row)
            if (
                (epoch_in_stage + 1) % self.config.checkpoint_every_epochs == 0
                or (epoch_in_stage + 1) == stage_epochs
            ):
                self._persist_checkpoint(
                    stage=stage,
                    loss_at_checkpoint=loss,
                )

    def run_curriculum(
        self,
        on_epoch_end: Callable[[StageTelemetryRow], None] | None = None,
    ) -> TrainingTelemetry:
        """Execute the canonical 8-stage curriculum end-to-end."""

        for stage in self.config.curriculum:
            self.run_stage(stage, on_epoch_end=on_epoch_end)
        self._telemetry.run_completed_utc = _utc_now_iso()
        if self.config.telemetry_path is not None:
            self._telemetry.persist(self.config.telemetry_path)
        return self._telemetry

    def _persist_checkpoint(
        self,
        stage: StageHyperparameters,
        loss_at_checkpoint: float,
    ) -> CheckpointArtifact:
        """Persist one MLX checkpoint + paired PyTorch state_dict export."""

        assert mx is not None
        assert self._decoder is not None
        assert self._source_video_sha256 is not None
        self.config.checkpoint_root.mkdir(parents=True, exist_ok=True)
        ts = _utc_now_iso().replace(":", "").replace("-", "")
        stem = (
            f"stage{stage.stage_index:02d}_{stage.name}_"
            f"epoch{self._global_epoch:06d}_{ts}"
        )
        mlx_path = self.config.checkpoint_root / f"{stem}.mlx.safetensors"
        pt_path = self.config.checkpoint_root / f"{stem}.pt"
        latents_path = self.config.checkpoint_root / f"{stem}.latents.npy"
        flat_params: dict[str, Any] = {}
        # Save MLX-native via safetensors when available; fall back to
        # npz for portability.
        try:
            def _flatten(prefix: str, obj: Any) -> None:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatten(f"{prefix}.{k}" if prefix else k, v)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        _flatten(f"{prefix}.{i}" if prefix else str(i), v)
                else:
                    flat_params[prefix] = obj

            _flatten("", self._decoder.parameters())
            mx.save_safetensors(str(mlx_path), flat_params)
        except (AttributeError, TypeError, ValueError):
            import numpy as np  # type: ignore[import-not-found]

            np_params: dict[str, Any] = {}

            def _flatten_np(prefix: str, obj: Any) -> None:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        _flatten_np(f"{prefix}.{k}" if prefix else k, v)
                elif isinstance(obj, list):
                    for i, v in enumerate(obj):
                        _flatten_np(f"{prefix}.{i}" if prefix else str(i), v)
                else:
                    np_params[prefix] = _mlx_to_numpy(obj)

            _flatten_np("", self._decoder.parameters())
            flat_params = dict(np_params)
            mlx_path = Path(
                str(mlx_path).replace(".safetensors", ".npz")
            )
            np.savez(str(mlx_path), **np_params)
        # PyTorch export is non-authority until a later parity/runtime proof
        # consumes the .pt. If the bridge fails, write a typed placeholder that
        # cannot be mistaken for a state_dict or dispatch-ready artifact.
        pt_placeholder = pt_path.with_suffix(
            ".pt.deferred_export.non_authoritative.json"
        )
        pytorch_export_succeeded = False
        pytorch_export_manifest_path: Path | None = None
        pytorch_export_deferred_path: Path | None = None
        try:
            import numpy as np  # type: ignore[import-not-found]

            from tac.local_acceleration.mlx_to_pytorch_export import (
                export_mlx_state_dict_to_torch_pt,
            )
            from tac.local_acceleration.pr95_hnerv_mlx import (
                pytorch_state_dict_from_mlx,
            )

            np_state: dict[str, Any] = pytorch_state_dict_from_mlx(self._decoder)
            latents_np = _mlx_to_numpy(self._bundle.latents).astype(  # type: ignore[union-attr]
                np.float32,
                copy=True,
            )
            np.save(latents_path, latents_np)
            np_state["latents"] = latents_np
            export_manifest = export_mlx_state_dict_to_torch_pt(
                np_state,
                pt_path,
                substrate_id="pr95_hnerv_mlx",
                run_id=stem,
                overwrite=True,
            )
            pytorch_export_succeeded = pt_path.is_file()
            pytorch_export_manifest_path = pt_path.with_suffix(".pt.export_manifest.json")
            pytorch_export_manifest_path.write_text(
                json.dumps(
                    {
                        **export_manifest,
                        "schema": "pr95_mlx_long_training_pytorch_export_manifest.v1",
                        "pytorch_export_succeeded": pytorch_export_succeeded,
                        "pytorch_export_forward_parity_established": False,
                        "runtime_consumption_proof_established": False,
                        "training_fidelity_class": PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
                        "training_fidelity_status": PR95_MLX_LONG_TRAINING_FIDELITY_STATUS,
                        "reproduction_equivalence_class": (
                            PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
                        ),
                        "exact_readiness_refusal": {
                            "schema": "exact_readiness_refusal.v1",
                            "ready": False,
                            "blockers": list(PR95_MLX_LONG_TRAINING_PT_EXPORT_BLOCKERS),
                        },
                        **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
                    },
                    sort_keys=True,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        except (
            ImportError,
            AttributeError,
            TypeError,
            ValueError,
            NameError,
            RuntimeError,
        ) as exc:
            pt_path.parent.mkdir(parents=True, exist_ok=True)
            placeholder = {
                "schema": PR95_MLX_LONG_TRAINING_EXPORT_PLACEHOLDER_SCHEMA,
                "schema_version": PR95_MLX_LONG_TRAINING_EXPORT_PLACEHOLDER_SCHEMA,
                "deferred_export": True,
                "pytorch_export_succeeded": False,
                "pytorch_state_dict_path": str(pt_path),
                "pytorch_state_dict_exists": pt_path.exists(),
                "placeholder_is_not_pytorch_state_dict": True,
                "reason": (
                    f"Slot 1 export bridge call failed: "
                    f"{type(exc).__name__}: {exc!s}; canonical helper is "
                    "tac.local_acceleration.mlx_to_pytorch_export"
                ),
                "mlx_checkpoint_path": str(mlx_path),
                "latents_path": str(latents_path),
                "trained_latents_exported": False,
                "evidence_grade": MLX_LONG_TRAINING_EVIDENCE_GRADE,
                "evidence_tag": MLX_LONG_TRAINING_EVIDENCE_TAG,
                "axis_tag": MLX_LONG_TRAINING_EVIDENCE_TAG,
                "training_fidelity_class": PR95_MLX_LONG_TRAINING_FIDELITY_CLASS,
                "training_fidelity_status": PR95_MLX_LONG_TRAINING_FIDELITY_STATUS,
                "reproduction_equivalence_class": (
                    PR95_MLX_LONG_TRAINING_REPRODUCTION_CLASS
                ),
                "exact_readiness_refusal": {
                    "schema": "exact_readiness_refusal.v1",
                    "ready": False,
                    "blockers": list(PR95_MLX_LONG_TRAINING_PT_EXPORT_BLOCKERS),
                },
                **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
            }
            with open(pt_placeholder, "w", encoding="utf-8") as f:
                json.dump(placeholder, f, sort_keys=True, indent=2)
                f.write("\n")
            pytorch_export_deferred_path = pt_placeholder
        artifact = CheckpointArtifact(
            stage_index=stage.stage_index,
            global_epoch=self._global_epoch,
            mlx_checkpoint_path=mlx_path,
            pytorch_state_dict_path=pt_path,
            latents_path=latents_path if latents_path.exists() else None,
            loss_at_checkpoint=loss_at_checkpoint,
            timestamp_utc=_utc_now_iso(),
            source_video_sha256=self._source_video_sha256,
            pytorch_export_succeeded=pytorch_export_succeeded,
            pytorch_export_deferred_path=pytorch_export_deferred_path,
            pytorch_export_manifest_path=pytorch_export_manifest_path,
            trained_latents_exported=latents_path.exists() and pytorch_export_succeeded,
        )
        self._checkpoint_artifacts.append(artifact)
        return artifact

    @property
    def checkpoint_artifacts(self) -> list[CheckpointArtifact]:
        return list(self._checkpoint_artifacts)

    @property
    def telemetry(self) -> TrainingTelemetry:
        return self._telemetry


def _mlx_peak_memory_bytes() -> int:
    """Best-effort MLX peak-memory snapshot in bytes."""

    if mx is None:
        return 0
    try:
        return int(mx.get_peak_memory())  # type: ignore[attr-defined]
    except (AttributeError, RuntimeError):
        try:
            return int(mx.metal.get_peak_memory())  # type: ignore[attr-defined]
        except (AttributeError, RuntimeError):
            return 0


def _mlx_to_numpy(arr: Any) -> Any:
    """Convert an MLX array to numpy via the canonical np.asarray bridge."""

    import numpy as np  # type: ignore[import-not-found]

    return np.asarray(arr)


# ---------------------------------------------------------------------------
# Per-substrate-class-shift candidate adapter scaffold
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SubstrateAdapterScaffold:
    """Scaffold for a per-substrate-class-shift candidate MLX adapter.

    Each substrate-class-shift candidate (Hinton-distilled scorer
    surrogate Top-1 per DQS1-ASYMPTOTIC-FLOOR / UNIWARD per-instance x
    wavelet db4 / Z6 Z7 Z8 / Cooperative-receiver / DP1 / Probe 9
    substrate / etc.) requires a CUSTOM loss function + optionally a
    CUSTOM forward path. This scaffold is the canonical entry point;
    sister subagent waves implement the per-candidate loss + forward.

    Per the operator's CRITICAL INSIGHT 2026-05-25, this scaffold is
    a foundation for per-candidate MLX long-training validation. A candidate
    that passes through this scaffold still needs the normal queue-owned
    scorer/parity/exact-auth gates before paid GPU dispatch authority.

    Cargo-cult-vs-real-signal resolution protocol per Catalog #303:

      1. Implement candidate-specific loss + forward via the adapter's
         ``custom_loss_fn`` + ``custom_forward_fn`` callbacks.
      2. Run MLX long-training via the canonical 8-stage curriculum at
         100-epoch smoke FIRST (verifies pipeline correctness).
      3. Run MLX long-training at full canonical 3000-epoch (or Slot 2's
         abbreviated audit verdict when it lands).
      4. Emit a per-candidate convergence verdict in
         ``CONVERGES_CONSISTENTLY / DIVERGES / OSCILLATES / SUB_PARADIGM``.
      5. For ``CONVERGES_CONSISTENTLY`` candidates: emit a predicted-DeltaS
         confidence band at MLX scale; route through the Slot 1 export
         bridge to PyTorch state_dict; queue paid-CPU+CUDA paired auth
         eval per CLAUDE.md "Submission auth eval - BOTH CPU AND CUDA"
         non-negotiable. The local MLX verdict remains advisory.
      6. For ``DIVERGES`` candidates: classify per Catalog #307
         (IMPLEMENTATION-LEVEL falsified; paradigm INTACT); queue
         alternative-reducer enumeration per Catalog #308.
      7. For ``OSCILLATES / SUB_PARADIGM`` candidates: queue per-substrate
         optimal-form symposium per Catalog #325 + cargo-cult audit
         section per Catalog #303 BEFORE further training spend.
    """

    candidate_id: str
    candidate_class_shift_paradigm: str
    custom_loss_fn: Callable[[Any, Any, Any], Any] | None = None
    custom_forward_fn: Callable[[Any, Any], Any] | None = None
    canonical_citation: str = ""
    operator_routable_summary: str = ""

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id must be non-empty")
        if not self.candidate_class_shift_paradigm:
            raise ValueError(
                "candidate_class_shift_paradigm must be non-empty; "
                "examples: 'hinton_distilled_scorer_surrogate' / "
                "'uniward_per_instance_x_wavelet_db4' / "
                "'cooperative_receiver_z4' / 'predictive_coding_z5_z6'"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_class_shift_paradigm": self.candidate_class_shift_paradigm,
            "has_custom_loss_fn": self.custom_loss_fn is not None,
            "has_custom_forward_fn": self.custom_forward_fn is not None,
            "canonical_citation": self.canonical_citation,
            "operator_routable_summary": self.operator_routable_summary,
        }


# Initial scaffold registry for the highest-EV substrate-class-shift
# candidates per the DQS1-ASYMPTOTIC-FLOOR ranking + operator
# routing-directive landscape 2026-05-25. Sister subagent waves
# implement the per-candidate ``custom_loss_fn`` + ``custom_forward_fn``.
INITIAL_SUBSTRATE_ADAPTER_REGISTRY: tuple[SubstrateAdapterScaffold, ...] = (
    SubstrateAdapterScaffold(
        candidate_id="hinton_distilled_scorer_surrogate_top1",
        candidate_class_shift_paradigm="hinton_distilled_scorer_surrogate",
        canonical_citation=(
            "DQS1-ASYMPTOTIC-FLOOR Top-1 per the asymptotic-pursuit "
            "horizon class; sister Slot 3 dispatch prep per "
            "lane_hinton_distilled_scorer_surrogate_dispatch_prep_20260525"
        ),
        operator_routable_summary=(
            "MLX long-training validation P0; ~$50 saved per cargo-cult "
            "identification BEFORE paid Modal A100 dispatch"
        ),
    ),
    SubstrateAdapterScaffold(
        candidate_id="uniward_per_instance_x_wavelet_db4_probe9",
        candidate_class_shift_paradigm="uniward_per_instance_x_wavelet_db4",
        canonical_citation=(
            "Probe 9 substrate per UNIWARD-detector-informed-embedding "
            "+ Daubechies db4 wavelet multi-scale partition prior; "
            "Catalog #277 Daubechies CO-LEAD + Catalog #258 ATW codec"
        ),
        operator_routable_summary=(
            "MLX long-training validation P1; orthogonal class-shift "
            "from Hinton-distilled scorer; ~$100 saved per cargo-cult"
        ),
    ),
    SubstrateAdapterScaffold(
        candidate_id="cooperative_receiver_z4",
        candidate_class_shift_paradigm="atick_redlich_cooperative_receiver",
        canonical_citation=(
            "Z4 cooperative-receiver loss per Atick-Redlich 1990 + "
            "Tishby memorial; CLAUDE.md grand council 8 new 2026-05-15 "
            "seats + Z6 Z7 Z8 design memo"
        ),
        operator_routable_summary=(
            "MLX long-training validation P2; F-asymptote-class "
            "candidate; ~$100 saved per cargo-cult"
        ),
    ),
    SubstrateAdapterScaffold(
        candidate_id="predictive_coding_z5",
        candidate_class_shift_paradigm="rao_ballard_predictive_coding",
        canonical_citation=(
            "Z5 predictive-coding world model per Rao-Ballard 1999; "
            "ego-motion-conditioned next-frame prediction per "
            "Catalog #311 hard discipline"
        ),
        operator_routable_summary=(
            "MLX long-training validation P3; F-asymptote-class "
            "candidate; ~$150 saved per cargo-cult"
        ),
    ),
    SubstrateAdapterScaffold(
        candidate_id="hierarchical_predictive_coding_z8",
        candidate_class_shift_paradigm=(
            "hierarchical_predictive_coding_rao_ballard_x_daubechies_x_dreamerv3_x_wyner_ziv"
        ),
        canonical_citation=(
            "Z8 canonical quadruple per Catalog #312; Rao-Ballard "
            "hierarchy + Mallat wavelet CDF + Hafner DreamerV3 latent "
            "dynamics + Wyner-Ziv side-information"
        ),
        operator_routable_summary=(
            "MLX long-training validation P4; class-shift binding all "
            "4 canonical primitives; ~$200 saved per cargo-cult"
        ),
    ),
    SubstrateAdapterScaffold(
        candidate_id="pretrained_driving_prior_dp1",
        candidate_class_shift_paradigm="dp1_comma2k19_distilled_codebook",
        canonical_citation=(
            "DP1 pretrained driving prior per Catalog #209 + #210 + #213; "
            "Comma2k19 OOD distillation"
        ),
        operator_routable_summary=(
            "MLX long-training validation P5; cross-dataset pretrain "
            "+ contest-video fine-tune class-shift; ~$150 saved per "
            "cargo-cult"
        ),
    ),
)


def list_initial_substrate_adapter_registry() -> tuple[
    SubstrateAdapterScaffold, ...
]:
    """Return the canonical initial substrate adapter registry."""

    return INITIAL_SUBSTRATE_ADAPTER_REGISTRY
