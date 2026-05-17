# SPDX-License-Identifier: MIT
"""SWA + Polyak averaging — Izmailov-Podoprikhin 2018 + Polyak 1990.

Stochastic Weight Averaging (Izmailov, Podoprikhin, Garipov, Vetrov, Wilson —
"Averaging Weights Leads to Wider Optima and Better Generalization", UAI
2018): in the late phase of training, average weights at regular checkpoint
intervals; the average lands in a flatter region of the loss surface than any
single SGD trajectory point, which improves quantization-friendliness.

Polyak averaging (Boris Polyak — "New stochastic approximation type
procedures", 1990): running average of weights since training start; the
classical predecessor of EMA but with constant (1/N) weighting rather than
exponential decay.

This module differs from the existing :class:`tac.training.SWA` helper in
TWO ways:

1. **EMA-compatible from byte zero**: the existing SWA helper takes snapshots
   of the EMA shadow and ALSO applies them back to the EMA shadow at the end.
   This module separates the SWA buffer from the EMA buffer entirely — the
   SWA average is its own buffer that the substrate's archive packer can
   choose to use INSTEAD of the EMA shadow or AVERAGED WITH the EMA shadow.
2. **Scheduler-driven**: the existing SWA helper requires the caller to call
   ``update()`` manually each epoch in the final 20% of training. This
   module exposes a :class:`SWAScheduler` that wraps the call-cadence
   decision so the caller just calls ``maybe_update(epoch)`` and lets the
   scheduler decide.

`[literature-extrapolation]` claims:
- Izmailov 2018 reports +0.2 to +1.0 ImageNet top-1 vs. final-epoch SGD; the
  improvement is highest for quantization-aware models. Whether this
  generalizes to ~100k-param contest substrates at scorer-axis floor 0.193 is
  empirically unknown.
- Polyak 1990's theoretical bound: running average converges to a flat-region
  basin at rate O(1/N) where N is the number of averaged points.

`[derived]` claims:
- SWA buffer cost: O(state_dict_bytes) memory; one running average tensor per
  state-dict key. Update cost: O(state_dict_bytes) per snapshot.
- Constant-time per-step overhead: the scheduler decides whether to update
  via integer comparison; no per-step tensor ops in the "don't update" branch.

Cargo-cult audit per assumption
───────────────────────────────
* "SWA buffer should mirror EMA buffer in dtype + shape" — HARD-EARNED;
  the buffer must be loadable into the model later.
* "Late-phase SWA always beats EMA shadow" — CARGO-CULTED for arbitrary
  substrates; HARD-EARNED only for substrates with empirical anchors. We
  expose both buffers (EMA + SWA) and let the substrate choose at archive-
  build time.

Canonical-vs-unique decision per layer (Catalog #290)
─────────────────────────────────────────────────────
* Running-average arithmetic → ADOPT canonical (online mean: avg += (new -
  avg) / count).
* Buffer dtype → ADOPT canonical (match snapshot dtype per key; non-float
  buffers copied directly per :class:`tac.training.EMA` pattern).
* Cadence policy → DOCUMENTED FORK (Izmailov 2018 uses every-N-epochs; we
  expose the cadence callable so substrates can plug in their own policy).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


class SWASchedulerError(RuntimeError):
    """Raised when SWA scheduler invariants are violated."""


class PolyakAverager:
    """Running average of weights since first update (Polyak 1990).

    Online formula: ``avg += (new - avg) / count`` per state_dict key.
    Constant-memory; constant-time per update.

    Differs from :class:`tac.training.EMA` (exponential decay) by maintaining
    UNIFORM-WEIGHT running average; latest snapshot has equal weight to first.

    Usage::

        averager = PolyakAverager()
        for epoch in range(N):
            train_one_epoch(model)
            averager.update(model)
        averager.apply(model)  # load Polyak average into model
    """

    def __init__(self) -> None:
        self._state: dict[str, torch.Tensor] | None = None
        self._count = 0

    @property
    def count(self) -> int:
        return self._count

    def update(self, model: torch.nn.Module) -> None:
        """Add ``model``'s current state_dict to the running average."""
        sd = model.state_dict()
        with torch.no_grad():
            if self._state is None:
                self._state = {k: v.detach().clone() for k, v in sd.items()}
                self._count = 1
                return
            self._count += 1
            for k, v in sd.items():
                if k not in self._state:
                    # New module added after construction (codex finding 2
                    # hardening pattern from tac.training.EMA).
                    self._state[k] = v.detach().clone()
                    continue
                if not v.is_floating_point():
                    self._state[k].copy_(v)
                else:
                    # Online mean: avg += (new - avg) / count
                    self._state[k].add_(
                        (v.detach() - self._state[k]).to(self._state[k].dtype),
                        alpha=1.0 / self._count,
                    )

    def apply(self, model: torch.nn.Module) -> None:
        if self._state is None:
            raise SWASchedulerError("PolyakAverager.apply() before any update()")
        model.load_state_dict(self._state)

    def state_dict(self) -> dict[str, torch.Tensor]:
        if self._state is None:
            raise SWASchedulerError(
                "PolyakAverager.state_dict() before any update()"
            )
        return {k: v.clone() for k, v in self._state.items()}


class SWAScheduler:
    """Scheduled SWA buffer that decides when to take a snapshot.

    Usage::

        scheduler = SWAScheduler(
            total_epochs=1000,
            swa_start_fraction=0.8,  # last 20%
            update_every=5,
        )
        averager = PolyakAverager()
        for epoch in range(1000):
            train_one_epoch(model)
            if scheduler.should_update(epoch):
                averager.update(model)
        # archive_packer can choose averager.state_dict() vs ema.state_dict()

    Args:
        total_epochs: Total training epoch budget; SWA window is computed
            from this.
        swa_start_fraction: Fraction of total epochs after which SWA begins.
            Izmailov 2018 default ``0.75``-``0.8``; we default ``0.8`` per
            Kaggle community variant.
        update_every: SWA snapshot cadence (in epochs); default ``5`` per
            Izmailov 2018 §4.

    Raises:
        :class:`SWASchedulerError` on invalid args.
    """

    def __init__(
        self,
        *,
        total_epochs: int,
        swa_start_fraction: float = 0.8,
        update_every: int = 5,
    ) -> None:
        if total_epochs < 1:
            raise SWASchedulerError(
                f"total_epochs={total_epochs} must be >= 1"
            )
        if not (0.0 <= swa_start_fraction < 1.0):
            raise SWASchedulerError(
                f"swa_start_fraction={swa_start_fraction} must be in [0, 1)"
            )
        if update_every < 1:
            raise SWASchedulerError(
                f"update_every={update_every} must be >= 1"
            )
        self.total_epochs = total_epochs
        self.swa_start_epoch = int(total_epochs * swa_start_fraction)
        self.update_every = update_every

    def should_update(self, epoch: int) -> bool:
        """True if ``epoch`` is within the SWA window AND on the cadence."""
        if epoch < self.swa_start_epoch:
            return False
        if epoch >= self.total_epochs:
            return False
        if (epoch - self.swa_start_epoch) % self.update_every != 0:
            return False
        return True
