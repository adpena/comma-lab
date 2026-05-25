#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a Hinton-distilled scorer surrogate MLX long-training smoke.

This CLI binds the canonical Hinton KL T=2.0 ``custom_loss_fn`` (per
``tac.substrates.hinton_distilled_scorer_surrogate``) into the Slot 1 PR95
MLX long-training pipeline
(``tac.local_acceleration.pr95_hnerv_mlx_long_training.MLXLongTrainingPipeline``)
and emits a per-epoch loss-curve telemetry JSONL + a convergence verdict
JSON per Catalog #305 observability surface.

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192 +
Catalog #1: every artifact emitted is part of the
``[macOS-MLX research-signal]`` axis evidence; ``score_claim``,
``promotion_eligible``, ``rank_or_kill_eligible`` and
``ready_for_exact_eval_dispatch`` are all ``False`` by construction.

Convergence verdict classification:
  * ``CONVERGES_CONSISTENTLY``: loss decreases monotonically (allowing minor
    epoch-noise oscillations) AND final epoch loss is < ~50% of initial.
  * ``DIVERGES``: loss increases or becomes NaN at any epoch.
  * ``OSCILLATES``: loss neither monotonically decreases nor monotonically
    increases (bouncing above mid-point of stage tail).
  * ``SUB_PARADIGM``: loss decreases but final < 95% of initial (slow
    convergence; may need longer training).

Operator-routable signals:
  * ``CONVERGES_CONSISTENTLY`` → LOCAL_MLX_QUEUE_READY for the next
    queue-owned local training/proof step.
  * ``DIVERGES`` → DEFER per Catalog #307 IMPLEMENTATION-LEVEL falsification
    (paradigm INTACT per CLAUDE.md "Forbidden premature KILL"); operator
    queues alternative reducer per Catalog #308 (different T value, sister
    teacher provider, or different student head architecture).
  * ``OSCILLATES`` → Catalog #325 per-substrate symposium reactivation
    (LR too high / batch too small / regularization needed).
  * ``SUB_PARADIGM`` → operator decides whether to extend training
    (1000ep / 3000ep) before any paid GPU proposal.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.local_acceleration.pr95_hnerv_mlx_long_training import (  # noqa: E402
    CANONICAL_BASE_CHANNELS,
    CANONICAL_CONTEST_VIDEO_PATH,
    CANONICAL_EVAL_SIZE,
    CANONICAL_LATENT_DIM,
    EVIDENCE_GRADE_MLX,
    EVIDENCE_TAG_MLX,
    PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
    LongTrainingConfig,
    MLXLongTrainingPipeline,
    StageHyperparameters,
    compute_video_sha256,
)
from tac.substrates.hinton_distilled_scorer_surrogate import (  # noqa: E402
    DEFAULT_DISTILLATION_TEMPERATURE,
    DEFAULT_SEGNET_CLASSES,
    HintonMlxCustomLossFnConfig,
    MockTeacherLogitsProvider,
    build_real_segnet_teacher_cache,
    make_hinton_custom_loss_fn,
)

TEACHER_PROVIDER_MOCK = "mock"
TEACHER_PROVIDER_REAL_SEGNET = "real_segnet"
VALID_TEACHER_PROVIDERS = (TEACHER_PROVIDER_MOCK, TEACHER_PROVIDER_REAL_SEGNET)

SMOKE_CURRICULUM_DEFAULT = (
    StageHyperparameters(
        stage_index=1,
        name="hinton_smoke",
        epochs=100,
        learning_rate=1e-3,
        batch_size=2,
        notes=(
            "Single-stage 100-epoch smoke for Hinton-distilled KL T=2.0 "
            "convergence validation; canonical Slot 1 LR + batch_size "
            "preserved for apples-to-apples comparison to the MSE-only baseline."
        ),
    ),
)

HINTON_MLX_SMOKE_SCHEMA = "hinton_mlx_long_training_smoke_verdict.v1"
HINTON_LOCAL_QUEUE_READY = "LOCAL_MLX_QUEUE_READY"
HINTON_PAID_DISPATCH_BLOCKED = (
    "PAID_DISPATCH_BLOCKED_REQUIRES_CONTEST_TEACHER_AND_CPU_CUDA_AUTH_EVAL"
)


CONVERGENCE_VERDICT_CONVERGES = "CONVERGES_CONSISTENTLY"
CONVERGENCE_VERDICT_DIVERGES = "DIVERGES"
CONVERGENCE_VERDICT_OSCILLATES = "OSCILLATES"
CONVERGENCE_VERDICT_SUB_PARADIGM = "SUB_PARADIGM"
VALID_CONVERGENCE_VERDICTS = (
    CONVERGENCE_VERDICT_CONVERGES,
    CONVERGENCE_VERDICT_DIVERGES,
    CONVERGENCE_VERDICT_OSCILLATES,
    CONVERGENCE_VERDICT_SUB_PARADIGM,
)


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class HintonMlxLongTrainingPipeline(MLXLongTrainingPipeline):
    """Slot 1 pipeline subclass that delegates ``loss_fn`` to the Hinton helper.

    The Slot 1 ``SubstrateAdapterScaffold`` exposes a ``custom_loss_fn``
    field but the canonical ``MLXLongTrainingPipeline`` does NOT yet
    consume it directly. The cleanest non-mutation integration pattern
    (per CLAUDE.md "Forbidden premature KILL" + Catalog #110 / #113
    APPEND-ONLY HISTORICAL_PROVENANCE) is to subclass + override
    ``loss_fn``; the Slot 1 module itself is not mutated.
    """

    def __init__(
        self,
        config: LongTrainingConfig,
        hinton_config: HintonMlxCustomLossFnConfig,
    ) -> None:
        super().__init__(config)
        self._hinton_config = hinton_config
        self._hinton_custom_loss_fn = make_hinton_custom_loss_fn(hinton_config)

    def loss_fn(self, bundle: Any, indices: Any, targets_batch: Any) -> Any:  # type: ignore[override]
        """Delegate to the canonical Hinton KL T=2.0 ``custom_loss_fn``.

        The Slot 1 ``training_step`` invokes ``loss_fn(bundle, indices, targets)``
        via ``nn.value_and_grad``; subclass override is the canonical extension
        point per Slot 1's docstring ("Sister substrate adapters can extend
        it with scorer-aware losses for later queue-owned probes").
        """
        return self._hinton_custom_loss_fn(bundle, indices, targets_batch)


@dataclasses.dataclass(frozen=True)
class ConvergenceVerdict:
    """Canonical convergence verdict per Catalog #305 observability surface."""

    verdict: str
    initial_loss: float
    final_loss: float
    min_loss: float
    max_loss: float
    loss_reduction_percent: float
    nan_at_epoch: int | None
    oscillation_score: float
    smoke_epochs: int
    diverges_threshold_ratio: float
    sub_paradigm_threshold_ratio: float
    converges_threshold_ratio: float
    rationale: str

    def __post_init__(self) -> None:
        if self.verdict not in VALID_CONVERGENCE_VERDICTS:
            raise ValueError(
                f"verdict must be one of {VALID_CONVERGENCE_VERDICTS!r}; "
                f"got {self.verdict!r}"
            )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def classify_convergence(
    loss_curve: Sequence[float],
    *,
    diverges_threshold_ratio: float = 1.05,
    sub_paradigm_threshold_ratio: float = 0.95,
    converges_threshold_ratio: float = 0.50,
    oscillation_neighbor_window: int = 3,
) -> ConvergenceVerdict:
    """Classify a loss curve per the canonical 4-verdict taxonomy.

    Args:
        loss_curve: List of per-epoch scalar losses in chronological order.
        diverges_threshold_ratio: ``final / initial`` ratio above which the
            run is classified ``DIVERGES``. Default 1.05 (loss increased >5%
            from start to end).
        sub_paradigm_threshold_ratio: ``final / initial`` ratio above which
            the run (if not diverging) is classified ``SUB_PARADIGM``.
            Default 0.95 (final loss > 95% of initial = slow convergence).
        converges_threshold_ratio: ``final / initial`` ratio at or below
            which the run is classified ``CONVERGES_CONSISTENTLY``. Default
            0.50 (final loss <= 50% of initial = solid convergence).
        oscillation_neighbor_window: Look-back window for computing
            local-neighbor oscillation signature. The verdict treats a run
            as ``OSCILLATES`` when the loss is non-monotonic in a way that
            does NOT cleanly fit CONVERGES/SUB_PARADIGM/DIVERGES.

    Returns:
        ConvergenceVerdict carrying the verdict + telemetry-bearing
        scalar statistics for the canonical landing memo.
    """
    if not loss_curve:
        raise ValueError("loss_curve must be non-empty")
    losses = list(loss_curve)
    nan_idx = next(
        (i for i, v in enumerate(losses) if not math.isfinite(v)),
        None,
    )
    initial = float(losses[0])
    final = float(losses[-1])
    min_loss = float(min(losses))
    max_loss = float(max(losses))
    # NaN/inf → DIVERGES (mathematical pathology).
    if nan_idx is not None:
        return ConvergenceVerdict(
            verdict=CONVERGENCE_VERDICT_DIVERGES,
            initial_loss=initial,
            final_loss=float("nan"),
            min_loss=min_loss if math.isfinite(min_loss) else float("nan"),
            max_loss=max_loss if math.isfinite(max_loss) else float("nan"),
            loss_reduction_percent=float("nan"),
            nan_at_epoch=nan_idx + 1,
            oscillation_score=float("nan"),
            smoke_epochs=len(losses),
            diverges_threshold_ratio=diverges_threshold_ratio,
            sub_paradigm_threshold_ratio=sub_paradigm_threshold_ratio,
            converges_threshold_ratio=converges_threshold_ratio,
            rationale=(
                f"Non-finite loss at epoch {nan_idx + 1}; classified DIVERGES "
                f"per Catalog #307 IMPLEMENTATION-LEVEL falsification. "
                f"Paradigm INTACT per CLAUDE.md 'Forbidden premature KILL'."
            ),
        )
    # Avoid division by zero (degenerate initial==0 case).
    if initial <= 0.0:
        ratio = 0.0
        loss_reduction_percent = 100.0
    else:
        ratio = final / initial
        loss_reduction_percent = 100.0 * (1.0 - ratio)
    # Oscillation score: split the curve into 4 chronological quartiles
    # and measure quartile-mean monotonicity. A canonical converging
    # curve has each quartile's mean less than the previous quartile's
    # mean (mean(Q1) > mean(Q2) > mean(Q3) > mean(Q4)); a canonical
    # oscillating curve has quartile means that flip back and forth.
    # This treats SGD-style per-epoch noise as benign (averaged out at
    # quartile scale) while flagging macro instability where the curve
    # genuinely bounces over wide bands at the multi-epoch timescale.
    # Catalog #305 observability semantics: this MUST distinguish 'SGD
    # noise during healthy convergence' from 'macro instability'.
    n = len(losses)
    if n < 4:
        # Too few epochs to compute quartile structure; trust the ratio.
        oscillation_score = 0.0
    else:
        q_size = n // 4
        q_means = [
            sum(losses[i * q_size : (i + 1) * q_size]) / float(q_size)
            for i in range(4)
        ]
        # Count adjacent-quartile-mean INCREASES (each increase = a
        # macro non-monotone step). 3 possible transitions Q1->Q2,
        # Q2->Q3, Q3->Q4.
        macro_increases = sum(
            1 for i in range(3) if q_means[i + 1] > q_means[i]
        )
        oscillation_score = float(macro_increases) / 3.0
    # Verdict cascade:
    # 1. DIVERGES if final > diverges_threshold_ratio * initial.
    if ratio > diverges_threshold_ratio:
        verdict = CONVERGENCE_VERDICT_DIVERGES
        rationale = (
            f"final/initial = {ratio:.4f} > {diverges_threshold_ratio} = "
            f"loss INCREASED beyond noise floor; classified DIVERGES per "
            f"Catalog #307 IMPLEMENTATION-LEVEL falsification."
        )
    # 2. CONVERGES_CONSISTENTLY if final <= converges_threshold_ratio *
    #    initial AND oscillation is low.
    elif ratio <= converges_threshold_ratio and oscillation_score <= 0.40:
        verdict = CONVERGENCE_VERDICT_CONVERGES
        rationale = (
            f"final/initial = {ratio:.4f} <= {converges_threshold_ratio} "
            f"(reduction={loss_reduction_percent:.1f}%) AND oscillation "
            f"score = {oscillation_score:.3f} <= 0.40; canonical "
            f"CONVERGES_CONSISTENTLY per Hinton 2014 distillation paradigm + "
            f"Probe 6 W=2 11.29x + Probe 7 W=6 21.20x CCC anchors."
        )
    # 3. SUB_PARADIGM if final > sub_paradigm_threshold_ratio * initial
    #    (almost no reduction).
    elif ratio > sub_paradigm_threshold_ratio:
        verdict = CONVERGENCE_VERDICT_SUB_PARADIGM
        rationale = (
            f"final/initial = {ratio:.4f} > {sub_paradigm_threshold_ratio} "
            f"= almost no reduction; classified SUB_PARADIGM (longer "
            f"training may close gap); operator decides whether to "
            f"extend to 1000ep/3000ep before paid GPU."
        )
    # 4. OSCILLATES if intermediate ratio but high oscillation.
    elif oscillation_score > 0.40:
        verdict = CONVERGENCE_VERDICT_OSCILLATES
        rationale = (
            f"final/initial = {ratio:.4f} (reduction={loss_reduction_percent:.1f}%) "
            f"AND oscillation score = {oscillation_score:.3f} > 0.40; "
            f"classified OSCILLATES per Catalog #325 per-substrate "
            f"symposium reactivation criteria (LR / batch / regularization)."
        )
    # 5. Otherwise SUB_PARADIGM (intermediate convergence, low oscillation).
    else:
        verdict = CONVERGENCE_VERDICT_SUB_PARADIGM
        rationale = (
            f"final/initial = {ratio:.4f} (reduction={loss_reduction_percent:.1f}%) "
            f"is between converges threshold ({converges_threshold_ratio}) and "
            f"sub_paradigm threshold ({sub_paradigm_threshold_ratio}); "
            f"classified SUB_PARADIGM; operator may extend training."
        )
    return ConvergenceVerdict(
        verdict=verdict,
        initial_loss=initial,
        final_loss=final,
        min_loss=min_loss,
        max_loss=max_loss,
        loss_reduction_percent=loss_reduction_percent,
        nan_at_epoch=None,
        oscillation_score=oscillation_score,
        smoke_epochs=len(losses),
        diverges_threshold_ratio=diverges_threshold_ratio,
        sub_paradigm_threshold_ratio=sub_paradigm_threshold_ratio,
        converges_threshold_ratio=converges_threshold_ratio,
        rationale=rationale,
    )


def _parse_eval_size(raw: str) -> tuple[int, int]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("eval-size must be H,W")
    try:
        h, w = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("eval-size must contain integers") from exc
    if h < 1 or w < 1:
        raise argparse.ArgumentTypeError("eval-size dims must be positive")
    return h, w


def _execution_command_args(raw_argv: Sequence[str]) -> list[str]:
    args = list(raw_argv)
    if "--execute-smoke" not in args:
        args.append("--execute-smoke")
    return [".venv/bin/python", "tools/run_hinton_mlx_long_training_smoke.py", *args]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-report",
        required=True,
        type=Path,
        help="Path to write the canonical convergence verdict JSON.",
    )
    parser.add_argument(
        "--source-video-path",
        default=CANONICAL_CONTEST_VIDEO_PATH,
        type=Path,
    )
    parser.add_argument(
        "--checkpoint-root",
        default=Path(
            "experiments/results/hinton_mlx_long_training_smoke_checkpoints"
        ),
        type=Path,
    )
    parser.add_argument(
        "--telemetry-path",
        type=Path,
        default=None,
        help="Per-epoch telemetry JSONL output (Catalog #305).",
    )
    parser.add_argument(
        "--lane-id",
        default=(
            "lane_hinton_distilled_scorer_surrogate_mlx_long_training_validation_20260525"
        ),
    )
    parser.add_argument(
        "--operator-run-label",
        default="hinton_kl_t2_smoke",
    )
    parser.add_argument("--latent-dim", default=CANONICAL_LATENT_DIM, type=int)
    parser.add_argument("--base-channels", default=CANONICAL_BASE_CHANNELS, type=int)
    parser.add_argument(
        "--eval-size",
        default=f"{CANONICAL_EVAL_SIZE[0]},{CANONICAL_EVAL_SIZE[1]}",
        type=_parse_eval_size,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=8,
        help=(
            "Max frames to decode from the source video (default 8 for the "
            "$0 smoke surface; canonical 1200 for full reproduction)."
        ),
    )
    parser.add_argument("--smoke-epochs", default=100, type=int)
    parser.add_argument("--checkpoint-every-epochs", default=100, type=int)
    parser.add_argument(
        "--distillation-temperature",
        default=DEFAULT_DISTILLATION_TEMPERATURE,
        type=float,
    )
    parser.add_argument(
        "--distillation-weight",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--num-classes",
        default=DEFAULT_SEGNET_CLASSES,
        type=int,
    )
    parser.add_argument(
        "--spatial-downsample-factor",
        default=4,
        type=int,
        help=(
            "Spatial downsample factor for the canonical mock teacher "
            "(default 4 = matches SegNet stride-2 stem aggregation). "
            "IGNORED when --teacher-provider=real_segnet (real SegNet "
            "outputs at canonical 384x512 = downsample factor 1)."
        ),
    )
    parser.add_argument(
        "--teacher-provider",
        choices=VALID_TEACHER_PROVIDERS,
        default=TEACHER_PROVIDER_MOCK,
        help=(
            "Teacher logits source. 'mock' uses the deterministic cosine "
            "projection (the canonical $0 smoke surface; the previous "
            "100ep smoke at commit e3b8c0d8d used this). 'real_segnet' "
            "pre-computes real upstream-PyTorch SegNet logits via "
            "tac.scorer.load_default_scorers on every video frame ONCE "
            "at setup time then indexes the cache per-batch via MLX "
            "integer indexing (O(1) lookup per step). The real_segnet "
            "path is the REAL-TEACHER REFIRE per the HINTON-MLX-BUNDLE "
            "mandate; it falsifies whether the mock-teacher convergence "
            "translates to the contest SegNet teacher."
        ),
    )
    parser.add_argument(
        "--teacher-cache-device",
        default="cpu",
        choices=("cpu",),
        help=(
            "PyTorch device for the real-SegNet teacher cache build. "
            "Only 'cpu' is permitted per CLAUDE.md 'MPS auth eval is "
            "NOISE' non-negotiable (MPS forwards through SegNet produce "
            "2x distortion drift)."
        ),
    )
    parser.add_argument(
        "--random-seed",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--hash-source-video",
        action="store_true",
        help="Hash the source video in the verdict for Catalog #229 PV.",
    )
    parser.add_argument(
        "--execute-smoke",
        action="store_true",
        help=(
            "Actually run the MLX smoke. Without this flag, the CLI emits "
            "a plan-only report (canonical pre-flight surface)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv or sys.argv[1:])
    args = parse_args(raw_argv)

    source_video_path = args.source_video_path
    if not source_video_path.is_absolute():
        source_video_path = REPO_ROOT / source_video_path
    checkpoint_root = args.checkpoint_root
    if not checkpoint_root.is_absolute():
        checkpoint_root = REPO_ROOT / checkpoint_root
    output_report = args.output_report
    if not output_report.is_absolute():
        output_report = REPO_ROOT / output_report
    telemetry_path = args.telemetry_path
    if telemetry_path is not None and not telemetry_path.is_absolute():
        telemetry_path = REPO_ROOT / telemetry_path

    config = LongTrainingConfig(
        source_video_path=source_video_path,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        eval_size=args.eval_size,
        curriculum=SMOKE_CURRICULUM_DEFAULT,
        checkpoint_root=checkpoint_root,
        telemetry_path=telemetry_path,
        smoke_mode=True,
        smoke_epochs_per_stage=args.smoke_epochs,
        checkpoint_every_epochs=args.checkpoint_every_epochs,
        max_frames=args.max_frames,
        random_seed=args.random_seed,
        lane_id=args.lane_id,
        operator_run_label=args.operator_run_label,
    )

    # Per the HINTON-MLX-BUNDLE 2026-05-25 REAL-TEACHER REFIRE mandate:
    # when --teacher-provider=real_segnet the student-side projection MUST
    # match the real SegNet output spatial shape (B, 384, 512, 5) so the
    # KL term is well-defined. The mock provider with downsample_factor=1
    # is the canonical student-side path; real-SegNet teacher logits are
    # supplied via the RealSegNetTeacherLogitsCache (built post-setup).
    student_downsample = (
        1 if args.teacher_provider == TEACHER_PROVIDER_REAL_SEGNET
        else args.spatial_downsample_factor
    )
    hinton_config = HintonMlxCustomLossFnConfig(
        distillation_weight=args.distillation_weight,
        temperature=args.distillation_temperature,
        student_head_out_channels=args.num_classes,
        teacher_provider=MockTeacherLogitsProvider(
            num_classes=args.num_classes,
            spatial_downsample_factor=student_downsample,
        ),
        # real_teacher_cache is wired AFTER pipeline.setup() below, where
        # the frame buffer becomes available.
    )

    source_sha = None
    if args.hash_source_video and config.source_video_path.is_file():
        source_sha = compute_video_sha256(config.source_video_path)

    plan: dict[str, Any] = {
        "schema": HINTON_MLX_SMOKE_SCHEMA,
        "lane_id": config.lane_id,
        "operator_run_label": config.operator_run_label,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "source_video_path": str(config.source_video_path),
        "source_video_sha256": source_sha,
        "max_frames": config.max_frames,
        "smoke_epochs": args.smoke_epochs,
        "distillation_temperature": args.distillation_temperature,
        "distillation_weight": args.distillation_weight,
        "num_classes": args.num_classes,
        "spatial_downsample_factor": args.spatial_downsample_factor,
        "teacher_provider": args.teacher_provider,
        "teacher_cache_device": args.teacher_cache_device,
        "latent_dim": config.latent_dim,
        "base_channels": config.base_channels,
        "eval_size": list(config.eval_size),
        "random_seed": config.random_seed,
        "mode": "plan_only",
        "command": [".venv/bin/python", "tools/run_hinton_mlx_long_training_smoke.py", *raw_argv],
        "recommended_execution": {
            "schema": "local_training_recommended_execution.v1",
            "tool": "tools/run_hinton_mlx_long_training_smoke.py",
            "training_backend": "mlx",
            "device": "gpu",
            "scheduler_resource_kind": "local_mlx",
            "output_manifest": output_report.as_posix(),
            "python_command_args": _execution_command_args(raw_argv),
            "candidate_generation_only": True,
            "requires_exact_eval_before_promotion": True,
            **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
        },
        "ran_at_utc": _utc_now_iso(),
        **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
    }
    if telemetry_path is not None:
        plan["recommended_execution"]["extra_artifact_postconditions"] = [
            {"type": "path_exists", "path": telemetry_path.as_posix()}
        ]

    if not args.execute_smoke:
        plan["readiness_blockers"] = [
            "execute_smoke_flag_required_to_run_actual_training",
        ]
        plan["ready_for_exact_eval_dispatch"] = False
        output_report.parent.mkdir(parents=True, exist_ok=True)
        output_report.write_text(
            json.dumps(plan, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "mode": "plan_only",
                    "output_report": output_report.as_posix(),
                    "evidence_grade": EVIDENCE_GRADE_MLX,
                },
                sort_keys=True,
            )
        )
        return 0

    # Execute smoke.
    pipeline = HintonMlxLongTrainingPipeline(config, hinton_config)
    started_utc = _utc_now_iso()
    setup_t0 = time.time()
    pipeline.setup()
    setup_elapsed = time.time() - setup_t0
    if source_sha is None:
        source_sha = pipeline._source_video_sha256  # type: ignore[attr-defined]

    # REAL-TEACHER REFIRE: build the real-SegNet teacher cache from the
    # in-memory frame buffer (canonical Slot 1 pair iterator stores
    # uint8 frames at (T, H, W, 3) per upstream/videos/0.mkv decode).
    # This is the critical mock-vs-real falsification surface: the
    # convergence verdict on REAL SegNet teacher logits is the load-
    # bearing artifact, not the mock-teacher loss-curve.
    real_teacher_cache_seconds: float | None = None
    real_teacher_cache_frame_count: int | None = None
    real_teacher_cache_hwk: list[int] | None = None
    if args.teacher_provider == TEACHER_PROVIDER_REAL_SEGNET:
        cache_t0 = time.time()
        frames_buffer = pipeline._pair_iterator._frames_np  # type: ignore[attr-defined]
        cache = build_real_segnet_teacher_cache(
            frames_buffer,
            upstream_dir=REPO_ROOT / "upstream",
            device=args.teacher_cache_device,
        )
        real_teacher_cache_seconds = time.time() - cache_t0
        real_teacher_cache_frame_count = cache.frame_count
        real_teacher_cache_hwk = [cache.height, cache.width, cache.num_classes]
        # Rebuild the hinton_config with the cache + reinstall the loss fn
        # on the pipeline.
        hinton_config = dataclasses.replace(hinton_config, real_teacher_cache=cache)
        pipeline._hinton_config = hinton_config  # type: ignore[attr-defined]
        pipeline._hinton_custom_loss_fn = make_hinton_custom_loss_fn(hinton_config)  # type: ignore[attr-defined]

    run_t0 = time.time()
    loss_curve: list[float] = []

    def _on_epoch_end(row: Any) -> None:
        loss_curve.append(float(row.loss))

    telemetry = pipeline.run_curriculum(on_epoch_end=_on_epoch_end)
    run_elapsed = time.time() - run_t0
    completed_utc = _utc_now_iso()

    verdict = classify_convergence(loss_curve)

    local_training_queue_signal: str
    paid_dispatch_authorization_signal: str
    if verdict.verdict == CONVERGENCE_VERDICT_CONVERGES:
        local_training_queue_signal = HINTON_LOCAL_QUEUE_READY
        paid_dispatch_authorization_signal = HINTON_PAID_DISPATCH_BLOCKED
    elif verdict.verdict == CONVERGENCE_VERDICT_DIVERGES:
        local_training_queue_signal = "DEFER_CATALOG_307_IMPLEMENTATION_FALSIFIED"
        paid_dispatch_authorization_signal = "DEFER_CATALOG_307_IMPLEMENTATION_FALSIFIED"
    elif verdict.verdict == CONVERGENCE_VERDICT_OSCILLATES:
        local_training_queue_signal = "DEFER_CATALOG_325_SYMPOSIUM_REACTIVATION"
        paid_dispatch_authorization_signal = "DEFER_CATALOG_325_SYMPOSIUM_REACTIVATION"
    else:  # SUB_PARADIGM
        local_training_queue_signal = "DEFER_LONGER_LOCAL_TRAINING_OR_OPERATOR_REVIEW"
        paid_dispatch_authorization_signal = "DEFER_LONGER_TRAINING_OR_OPERATOR_REVIEW"

    plan["mode"] = "executed_smoke"
    plan["source_video_sha256"] = source_sha
    plan["source_video_frame_count"] = pipeline._source_video_frame_count  # type: ignore[attr-defined]
    plan["telemetry_row_count"] = len(telemetry.rows)
    plan["telemetry_path"] = (
        str(config.telemetry_path) if config.telemetry_path is not None else None
    )
    plan["started_at_utc"] = started_utc
    plan["completed_at_utc"] = completed_utc
    plan["setup_seconds"] = setup_elapsed
    plan["run_seconds"] = run_elapsed
    plan["real_teacher_cache_seconds"] = real_teacher_cache_seconds
    plan["real_teacher_cache_frame_count"] = real_teacher_cache_frame_count
    plan["real_teacher_cache_hwk"] = real_teacher_cache_hwk
    plan["loss_curve"] = loss_curve
    plan["convergence_verdict"] = verdict.as_dict()
    plan["local_training_queue_signal"] = local_training_queue_signal
    plan["paid_dispatch_authorization_signal"] = paid_dispatch_authorization_signal
    # Readiness blockers depend on teacher provider per Catalog #287/#323:
    # mock teacher path retains the strong-language blocker; real_segnet
    # teacher path retains only the canonical paired-CPU+CUDA discipline
    # blocker (because the scorer loss IS the contest SegNet under the
    # real_segnet path).
    if args.teacher_provider == TEACHER_PROVIDER_REAL_SEGNET:
        plan["readiness_blockers"] = [
            "scorer_loss_is_kl_to_real_segnet_teacher_BUT_no_paired_cpu_cuda_auth_eval_yet",
            "macos_mlx_research_signal_only_per_catalog_192",
        ]
    else:
        plan["readiness_blockers"] = [
            "scorer_loss_is_kl_to_mock_teacher_logits_not_contest_segnet",
            "no_paired_cpu_cuda_auth_eval_yet",
            "macos_mlx_research_signal_only_per_catalog_192",
        ]
    plan["ready_for_exact_eval_dispatch"] = False

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "mode": "executed_smoke",
                "verdict": verdict.verdict,
                "loss_reduction_percent": verdict.loss_reduction_percent,
                "initial_loss": verdict.initial_loss,
                "final_loss": verdict.final_loss,
                "paid_dispatch_authorization_signal": paid_dispatch_authorization_signal,
                "output_report": output_report.as_posix(),
                "evidence_grade": EVIDENCE_GRADE_MLX,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
