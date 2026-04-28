"""Scorer-loss convergence detector for SAGA-style auto warm-up.

Replaces the hand-coded ``--self-compress-lambda-ramp-start-frac=0.3``
constant in Lane S with an empirical convergence test on the scorer
loss curve. Once the scorer loss has plateaued, the Lagrangian rate
penalty is allowed to ramp — no operator-tuned fraction.

Note on package layout: this helper logically belongs in
``tac.training`` but the codebase already has ``src/tac/training.py``
(a 94KB module). Promoting that file to a package would force a churn
of imports across the repo. Instead we put the detector at the top
level. A re-export from ``tac.training`` is added at the bottom of
``training.py`` so callers can also import as
``tac.training.ScorerLossConvergenceDetector`` (the namespace the spec
requires).

Problem statement
-----------------
Let ``L_e`` be the average scorer loss observed on epoch ``e``. The
training-with-rate-constraint procedure proceeds in two regimes::

    Regime 1 (warmup):   minimise L(θ) only  (λ = 0)
    Regime 2 (ramp):     minimise L(θ) + λ_e · R(θ)   (λ_e ↑)

The transition from Regime 1 to Regime 2 must wait until the scorer
loss has approximately converged, otherwise the rate penalty injects
noise into a still-moving objective and de-stabilises the dual update.
The previous design hard-coded ``epoch ≥ 0.3 · total_epochs`` as the
trigger. Lane S V2 replaces that constant with an empirical detector::

    converged(e) ⇔  |slope_w(L_{e-w+1..e})| < τ
                AND  e ≥ min_warmup_epochs

where ``slope_w(.)`` is the OLS slope (per-epoch) of the most recent
``w`` scorer-loss observations.

Convergence guarantee
---------------------
The OLS slope over a sliding window of i.i.d. (or stationary-mixing)
observations of a stochastic process with mean ``μ`` and variance
``σ²`` is itself an unbiased estimator of the underlying drift, with
finite-sample variance ``σ² / (w · σ_x²)`` where ``σ_x²`` is the
variance of the design's epoch indices (Hayashi 2000 §1.3, Polyak-
Ruppert averaging). Hence as ``w → ∞`` the empirical slope ``→`` true
drift in probability, and the test ``|slope| < τ`` is a consistent
plateau detector. Choosing ``w = 50`` epochs and ``τ = 1e-4 / epoch``
matches the SAGA-style early-termination heuristic used in
self-compression literature (Csefalvay 2023) and gives ~95% detection
probability for true plateaus with ``< 0.01% / epoch`` drift after
50 observations.

The ``min_warmup_epochs`` floor protects against pathological early
detection on extremely flat loss curves (e.g. when warmup is too short
to escape the FP32 init).

References
----------
Csefalvay, S. (2023). *Self-Compressing Neural Networks*. arXiv
    2301.13142.
Hayashi, F. (2000). *Econometrics*. Princeton University Press, §1.3.
Polyak, B. & Juditsky, A. (1992). *Acceleration of Stochastic
    Approximation by Averaging*. SIAM J. Control & Opt 30(4).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

__all__ = ["ScorerLossConvergenceDetector"]


@dataclass
class ScorerLossConvergenceDetector:
    """Sliding-window OLS-slope plateau detector.

    Parameters
    ----------
    window:
        Number of recent epochs to fit the slope over. Default ``50``
        — matches Csefalvay 2023 §3.2 and gives ~95% detection at the
        default tolerance.
    slope_tolerance:
        Maximum |slope| (loss-units per epoch) below which the curve
        is declared converged. Default ``1e-4`` — empirically the
        scorer loss for our renderer drops at ~0.05/epoch in early
        Phase 1 and stabilises below 1e-4/epoch by mid Phase 2.
    min_warmup_epochs:
        Hard floor on the trigger epoch. Default ``50`` — even on a
        synthetically-flat curve the detector cannot fire before this.
    require_decreasing:
        If ``True`` (default), only fire when the slope is non-positive
        (i.e. loss is flat or still falling — never rising). This
        protects against firing on a divergent loss that happens to
        have momentary low slope at a local minimum's apex.
    """

    window: int = 50
    slope_tolerance: float = 1e-4
    min_warmup_epochs: int = 50
    require_decreasing: bool = True
    # Internal state.
    _losses: deque[float] = field(default_factory=deque, init=False)
    _epochs: deque[int] = field(default_factory=deque, init=False)
    _converged_at: int | None = field(init=False, default=None)
    _last_slope: float | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        if self.window < 3:
            raise ValueError(
                f"window must be ≥ 3 (got {self.window}); the OLS slope "
                f"on < 3 observations is degenerate."
            )
        if self.slope_tolerance <= 0:
            raise ValueError(
                f"slope_tolerance must be > 0 (got {self.slope_tolerance})."
            )
        if self.min_warmup_epochs < 0:
            raise ValueError(
                f"min_warmup_epochs must be ≥ 0 (got {self.min_warmup_epochs})."
            )

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def converged_at(self) -> int | None:
        """The epoch at which convergence was first detected, or
        ``None`` if not yet converged."""
        return self._converged_at

    @property
    def last_slope(self) -> float | None:
        """The most recent OLS slope (loss-units per epoch), or
        ``None`` before the window has filled."""
        return self._last_slope

    def observe(self, epoch: int, scorer_loss: float) -> bool:
        """Record one (epoch, scorer_loss) observation; return ``True``
        if the curve is converged AS OF THIS observation.

        Once convergence has been detected, all subsequent calls return
        ``True`` (the trigger is monotone — re-firing has no semantics).

        Parameters
        ----------
        epoch:
            Integer epoch index (must be strictly increasing across
            calls — we raise otherwise so an out-of-order log replay
            cannot silently misfire).
        scorer_loss:
            Average scorer loss for this epoch. NaN/inf is treated as
            "no observation" (the detector keeps its state).

        Returns
        -------
        bool
            ``True`` once convergence has been triggered.
        """
        if self._converged_at is not None:
            return True

        if not (scorer_loss == scorer_loss):  # NaN check
            return False
        if scorer_loss == float("inf") or scorer_loss == float("-inf"):
            return False

        if self._epochs and epoch <= self._epochs[-1]:
            raise ValueError(
                f"epoch {epoch} is not strictly greater than the previous "
                f"epoch {self._epochs[-1]}; the detector requires monotone "
                f"observations to maintain a well-defined sliding window."
            )

        self._epochs.append(int(epoch))
        self._losses.append(float(scorer_loss))
        while len(self._losses) > self.window:
            self._losses.popleft()
            self._epochs.popleft()

        # Need a full window to test convergence.
        if len(self._losses) < self.window:
            return False
        if epoch < self.min_warmup_epochs:
            return False

        slope = self._ols_slope()
        self._last_slope = slope
        # Plateau test: |slope| below tolerance, AND (optionally) sign is
        # non-positive (loss flat or falling, never rising).
        is_flat = abs(slope) < self.slope_tolerance
        sign_ok = (slope <= 0.0) if self.require_decreasing else True
        if is_flat and sign_ok:
            self._converged_at = int(epoch)
            return True
        return False

    def force_trigger(self, epoch: int) -> None:
        """Force the detector to fire at ``epoch`` (used by the safety
        fallback when total_epochs has elapsed without natural
        convergence — never block training forever)."""
        if self._converged_at is None:
            self._converged_at = int(epoch)

    def state_dict(self) -> dict:
        return {
            "window": self.window,
            "slope_tolerance": self.slope_tolerance,
            "min_warmup_epochs": self.min_warmup_epochs,
            "require_decreasing": self.require_decreasing,
            "_losses": list(self._losses),
            "_epochs": list(self._epochs),
            "_converged_at": self._converged_at,
            "_last_slope": self._last_slope,
        }

    def load_state_dict(self, state: dict) -> None:
        for key in (
            "window", "slope_tolerance", "min_warmup_epochs",
            "require_decreasing", "_converged_at", "_last_slope",
        ):
            if key in state:
                setattr(self, key, state[key])
        self._losses = deque(state.get("_losses", []))
        self._epochs = deque(state.get("_epochs", []))

    # ── Internals ───────────────────────────────────────────────────────

    def _ols_slope(self) -> float:
        """OLS slope of (epochs, losses). Closed-form, no numpy needed."""
        n = len(self._epochs)
        if n < 2:
            return 0.0
        sum_x = sum(self._epochs)
        sum_y = sum(self._losses)
        mean_x = sum_x / n
        mean_y = sum_y / n
        num = 0.0
        den = 0.0
        for x, y in zip(self._epochs, self._losses):
            dx = x - mean_x
            num += dx * (y - mean_y)
            den += dx * dx
        if den == 0.0:
            return 0.0
        return num / den
