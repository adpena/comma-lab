# SPDX-License-Identifier: MIT
"""Canonical detector-aware iterative training loop (Yousfi 2019 ALASKA Pattern #4).

Origin: ``external/alaska_yousfi/src/tools/train_estimator.py:155-194``
``train_estimator`` + ``cnn_model_fn:103-150`` + the canonical Adamax
optimizer at lines 24-77 + piecewise-constant LR schedule (boundaries
``[20000, 100000]`` / values ``[0.0001, 0.001, 0.0001]``).

The CANONICAL insight (Yousfi 2019 ALASKA-#1-winning):
The training loop is structurally a 3-stage curriculum:

1. **Warm-up** (steps 0-20000, LR=1e-4): low LR to absorb the warm-start
   checkpoint's prior without destroying it
2. **Main training** (steps 20000-100000, LR=1e-3): 10x higher LR for
   primary fitting
3. **Fine-tune** (steps 100000-200000, LR=1e-4): return to low LR for
   convergence

Combined with:
* **Adamax optimizer** (Kingma 2014; canonical for steganalysis per the
  upstream ``AdamaxOptimizer`` class)
* **Warm-start from single-branch checkpoint** to multi-branch via
  ``warm_start_dict`` rebinding ``branch+'/Layer'`` -> ``Layer``
* **Pair-constraint batch input** per :mod:`tac.composition.alaska_inverse_steganalysis_patterns.canonical_pair_constraint_batch`
* **Multi-scheme Dirichlet sampling** per :mod:`tac.composition.alaska_inverse_steganalysis_patterns.canonical_multi_scheme_prior`

5-axis adaptation taxonomy
--------------------------
* **Axis A (contest)** -- detector training -> generator+scorer joint training
* **Axis B (problem space)** -- 5-class softmax -> per-pair score components
* **Axis C (math)** -- canonical 3-stage piecewise-constant LR + Adamax 1:1
* **Axis D (data)** -- 200k iterations on QF95 256x256 -> contest-resolution
  iteration budget per substrate-engineering opt-out per HNeRV parity L7
* **Axis E (video)** -- single-image batches -> per-pair latent-shared batches

Sister of slot ``tac.training.EMA`` at the canonical-EMA-discipline surface
per CLAUDE.md "EMA - NON-NEGOTIABLE" (Yousfi's training did NOT use EMA;
the canonical comma adaptation REQUIRES EMA at decay=0.997 per the canonical
non-negotiable).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

__all__ = (
    "DetectorAwareIterativeTrainingStrategy",
    "DetectorAwareTrainingConfig",
    "CANONICAL_ALASKA_LR_SCHEDULE",
    "CANONICAL_ALASKA_MAX_ITER",
    "DetectorAwareTrainingError",
)


class DetectorAwareTrainingError(ValueError):
    """Raised when training config violates a canonical invariant."""


CANONICAL_ALASKA_LR_SCHEDULE: tuple[tuple[int, float], ...] = (
    (0, 1e-4),
    (20_000, 1e-3),
    (100_000, 1e-4),
)
"""Canonical Yousfi 2019 ALASKA-#1 LR schedule verbatim:
boundaries = [20000, 100000]; values = [0.0001, 0.001, 0.0001].
Per ``src/notebooks/tf_fine_tune_branch.ipynb`` cell 3 upstream."""


CANONICAL_ALASKA_MAX_ITER: int = 200_000
"""Canonical Yousfi 2019 ALASKA-#1 iteration count per
``src/notebooks/tf_fine_tune_branch.ipynb`` cell 3 ``max_iter = 200000``."""


class DetectorAwareIterativeTrainingStrategy:
    """Marker class for canonical detector-aware iterative training.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
    this module ships the CANONICAL training-protocol descriptor +
    configuration dataclass + LR schedule lookup; the actual training loop
    is per-substrate per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD" since
    every substrate has different gradient-through-SegNet/PoseNet routing.

    Sister substrates that should adopt this canonical training protocol:

    * Slot FF UNIWARD-cost trainer (3-stage LR + pair-constraint batching)
    * Slot YY HILL-cost trainer (sister)
    * Slot AAA MiPOD-cost trainer (sister)
    * Slot CCC HUGO-cost trainer (sister)

    Per CLAUDE.md "Catalog #303 cargo-cult audit per assumption":
    the canonical 200k iter budget MAY be excessive for comma's smaller
    substrate sizes; each substrate adopting this pattern MUST do a
    HARD-EARNED-vs-CARGO-CULTED classification of the iteration count vs
    its own empirical convergence curve.
    """


def _lr_schedule_invariant(
    schedule: Sequence[tuple[int, float]],
) -> None:
    if len(schedule) < 1:
        raise DetectorAwareTrainingError("schedule empty")
    last_step = -1
    for step, lr in schedule:
        if step < 0:
            raise DetectorAwareTrainingError(
                f"schedule step={step} must be >= 0"
            )
        if step <= last_step:
            raise DetectorAwareTrainingError(
                f"schedule steps must be strictly increasing; got step={step} <= prior={last_step}"
            )
        if not (lr > 0.0):
            raise DetectorAwareTrainingError(
                f"schedule lr={lr} must be > 0"
            )
        last_step = step


@dataclass(frozen=True)
class DetectorAwareTrainingConfig:
    """Canonical detector-aware training config.

    Mirrors Yousfi's upstream ``train_estimator`` parameter set; renamed
    + extended for COMMA contest contract clarity.

    Attributes
    ----------
    lr_schedule
        Tuple of (step, lr) ordered ascending by step. Defaults to
        :data:`CANONICAL_ALASKA_LR_SCHEDULE` (1e-4 / 1e-3 / 1e-4 at
        boundaries 0 / 20000 / 100000).
    max_iter
        Total training iterations. Defaults to
        :data:`CANONICAL_ALASKA_MAX_ITER` (200,000).
    optimizer_class
        Optimizer family. Default ``"Adamax"`` per Yousfi upstream;
        ``"AdamW"`` accepted for canonical sister substrates per
        CLAUDE.md "Bugs must be permanently fixed" Catalog #327
        master-gradient axis-custody discipline.
    pair_constraint_batch
        If True (default), input pipeline must use the canonical
        pair-constraint batch builder per
        :mod:`tac.composition.alaska_inverse_steganalysis_patterns.canonical_pair_constraint_batch`.
    multi_scheme_prior_active
        If True (default), input pipeline must sample perturbation modes
        per the canonical multi-scheme prior per
        :mod:`tac.composition.alaska_inverse_steganalysis_patterns.canonical_multi_scheme_prior`.
    warm_start_checkpoint_branch
        Optional name of the source branch to warm-start from
        (e.g. ``"YCrCb"`` for warm-starting a ``"CrCb"``-only training).
        None means cold-start.
    ema_decay
        Canonical EMA decay per CLAUDE.md "EMA - NON-NEGOTIABLE" 0.997.
        Yousfi upstream did NOT use EMA; this is a COMMA adaptation per
        the canonical non-negotiable.
    eval_roundtrip_active
        Canonical eval-roundtrip simulation per CLAUDE.md
        "eval_roundtrip - NON-NEGOTIABLE" True. Yousfi upstream did NOT
        simulate the contest eval roundtrip; this is a COMMA adaptation.

    Notes
    -----
    This is a DESCRIPTOR config: it does NOT execute training. Each
    substrate that adopts the canonical ALASKA protocol must thread this
    config through its own training script per CLAUDE.md
    "UNIQUE-AND-COMPLETE-PER-METHOD" operating mode.
    """

    lr_schedule: tuple[tuple[int, float], ...] = CANONICAL_ALASKA_LR_SCHEDULE
    max_iter: int = CANONICAL_ALASKA_MAX_ITER
    optimizer_class: str = "Adamax"
    pair_constraint_batch: bool = True
    multi_scheme_prior_active: bool = True
    warm_start_checkpoint_branch: str | None = None
    ema_decay: float = 0.997
    eval_roundtrip_active: bool = True

    def __post_init__(self) -> None:
        _lr_schedule_invariant(self.lr_schedule)
        if self.max_iter < 1:
            raise DetectorAwareTrainingError(
                f"max_iter={self.max_iter} must be >= 1"
            )
        if self.optimizer_class not in ("Adamax", "AdamW", "Adam", "SGD"):
            raise DetectorAwareTrainingError(
                f"optimizer_class={self.optimizer_class!r} must be one of Adamax/AdamW/Adam/SGD"
            )
        if not (0.0 < self.ema_decay < 1.0):
            raise DetectorAwareTrainingError(
                f"ema_decay={self.ema_decay} must be in (0, 1)"
            )

    def lr_at_step(self, step: int) -> float:
        """Lookup the LR at training ``step`` per the canonical schedule.

        Parameters
        ----------
        step
            Training step >= 0.

        Returns
        -------
        float
            LR at the step per piecewise-constant schedule.
        """
        if step < 0:
            raise DetectorAwareTrainingError(f"step={step} must be >= 0")
        current_lr = self.lr_schedule[0][1]
        for boundary_step, boundary_lr in self.lr_schedule:
            if step >= boundary_step:
                current_lr = boundary_lr
            else:
                break
        return current_lr
