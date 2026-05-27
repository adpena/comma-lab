# SPDX-License-Identifier: MIT
"""The substrate-AGNOSTIC ``_full_main`` orchestrator (separation of concerns).

This module owns ONLY the orchestration: verify MLX (device gate) -> verify
inflate portability (optional) -> wrap the bundle in the Style-B adapter ->
build the canonical ``LongTrainingConfig`` -> route through canonical
``run_long_training``. Every step delegates to a focused sub-module; the
orchestrator composes them.

Non-promotable by construction per CLAUDE.md "MLX portable-local-substrate
authority" + Catalog #127/#192/#317/#341: every artifact is tagged
``[macOS-MLX research-signal]`` with ``score_claim=False``,
``promotion_eligible=False``, ``ready_for_exact_eval_dispatch=False``. The
canonical L2 harness auto-stamps these markers on the ``TrainingArtifact``.

Dispatch gating (Catalog #325): this orchestrator runs on the M5 Max via MLX at
$0; it NEVER triggers a paid GPU dispatch. The device gate fails closed on a
non-MLX host (no silent CPU/CUDA fallback per Catalog #1 + #317).

[verified-against: tac.training.long_training_canonical.run_long_training canonical L2 harness]
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
from tac.substrates._shared.mlx_score_aware.device_gate import (
    require_mlx_for_harness,
)
from tac.substrates._shared.mlx_score_aware.portability import (
    assert_numpy_portable_inflate,
)

if TYPE_CHECKING:
    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle

# Canonical contest constants (sister of pact_nerv_full_main).
CONTEST_NORMALIZER: float = 37_545_489.0
# MLX false-authority canonical marker (sister of pr95_hnerv_mlx FALSE_AUTHORITY).
MLX_EVIDENCE_GRADE: str = "[macOS-MLX research-signal]"


def run_mlx_score_aware_full_main(
    *,
    bundle: RendererBundle,
    substrate_id: str,
    lane_id: str,
    output_dir: Any,
    epochs: int,
    batch_pair_indices_per_step: int,
    learning_rate: float = 1e-3,
    seed: int = 0,
    ema_decay: float | None = None,
    checkpoint_interval_epochs: int = 10,
    early_stopping_patience: int | None = None,
    curriculum_stages: Any | None = None,
    inflate_py_path: Any | None = None,
    notes: str = "",
    on_epoch_end: Callable[[Any], None] | None = None,
) -> Any:
    """Run the canonical MLX-first score-aware ``_full_main`` body.

    This is the substrate-AGNOSTIC ``_full_main`` the MLX-first substrate
    trainers route through. It:

    1. Verifies MLX availability (fail-closed; no CPU/CUDA leak per Catalog
       #1 + #317 + #325).
    2. (Optional) Verifies the substrate's ``inflate.py`` is numpy-portable
       (8th directive; HNeRV parity L4) when ``inflate_py_path`` is supplied.
    3. Wraps the substrate ``RendererBundle`` in :class:`MlxScoreAwareAdapter`.
    4. Builds a canonical ``LongTrainingConfig`` (single full-stage curriculum
       by default; the substrate may pass a multi-stage curriculum).
    5. Routes through canonical ``run_long_training`` (EMA / OOM-safe /
       telemetry / checkpoint / Provenance / posterior anchor / archive
       export).

    Args:
        bundle: the substrate RendererBundle (UNIQUE axis).
        substrate_id: canonical substrate id.
        lane_id: canonical lane id per CLAUDE.md "Lane maturity registry".
        output_dir: canonical output dir (MUST NOT be ``/tmp`` per the
            transient-evidence trap; ``run_long_training`` validates this).
        epochs: total epoch budget.
        batch_pair_indices_per_step: training batch size.
        learning_rate / seed / checkpoint_interval_epochs: training hparams.
        ema_decay: optional EMA decay override (default = canonical 0.997).
        early_stopping_patience: optional override (default = epochs + 1, i.e.
            disabled; MLX-local runs are cheap so we run the full budget).
        curriculum_stages: optional ``tuple[CurriculumStage, ...]``; default is
            a single full-budget stage.
        inflate_py_path: optional path to the substrate ``inflate.py`` to
            verify numpy-portability before training (8th directive).
        notes: substantive rationale (Catalog #287 placeholder rejected by the
            config).
        on_epoch_end: optional per-epoch callback.

    Returns:
        the canonical ``TrainingArtifact`` from ``run_long_training``.

    Raises:
        MlxScoreAwareHarnessError: MLX unavailable OR inflate not portable.
    """
    require_mlx_for_harness()
    output_dir = Path(output_dir)

    from tac.training.long_training_canonical import (
        CANONICAL_EMA_DECAY,
        CurriculumStage,
        LongTrainingConfig,
        run_long_training,
    )

    if inflate_py_path is not None:
        assert_numpy_portable_inflate(inflate_py_path)

    if curriculum_stages is None:
        curriculum_stages = (
            CurriculumStage(
                name=f"{substrate_id}_mlx_score_aware_full",
                start_epoch=0,
                end_epoch=epochs,
                notes=(
                    "MLX-first score-aware full-budget stage via canonical "
                    "mlx_score_aware harness; reconstruction + "
                    "optional Hinton-KL T=2.0 scorer surrogate."
                ),
            ),
        )

    adapter = MlxScoreAwareAdapter(bundle, substrate_id=substrate_id)

    config = LongTrainingConfig(
        substrate_id=substrate_id,
        lane_id=lane_id,
        epochs=epochs,
        batch_pair_indices_per_step=batch_pair_indices_per_step,
        curriculum_stages=curriculum_stages,
        ema_decay=CANONICAL_EMA_DECAY if ema_decay is None else float(ema_decay),
        checkpoint_interval_epochs=checkpoint_interval_epochs,
        early_stopping_patience=(
            epochs + 1
            if early_stopping_patience is None
            else early_stopping_patience
        ),
        learning_rate=learning_rate,
        seed=seed,
        output_dir=output_dir,
        device="mlx",
        evidence_grade=MLX_EVIDENCE_GRADE,
        notes=(
            notes
            or (
                f"{substrate_id} MLX-first score-aware L2 via canonical "
                "mlx_score_aware harness; non-promotable "
                "[macOS-MLX research-signal] per Catalog #192/#317/#341."
            )
        ),
    )

    return run_long_training(adapter, config, on_epoch_end=on_epoch_end)


__all__ = [
    "CONTEST_NORMALIZER",
    "MLX_EVIDENCE_GRADE",
    "run_mlx_score_aware_full_main",
]
