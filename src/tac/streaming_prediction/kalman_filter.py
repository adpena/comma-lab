# SPDX-License-Identifier: MIT
"""Kalman state-space model for streaming master-gradient samples.

Per SLOT MG-5 of the 2026-05-19 master-gradient enhancement wave.

This module implements a SCALAR Kalman filter fusing streaming Taylor-extrapolated
score predictions into a running posterior mean + variance per substrate.

The state-space is intentionally scalar: at each epoch the substrate produces a
single predicted_score observation; the Kalman filter folds it into the running
posterior using the conjugate normal-normal update + EMA-decayed process noise
per CLAUDE.md "EMA — NON-NEGOTIABLE" sister discipline.

Why hand-rolled instead of scipy/pykalman?
──────────────────────────────────────────
1. Single dependency-free path so SCAFFOLD has zero new runtime requirements;
   sister modules (`gaussian_process_regression`) carry the heavier sklearn
   dependency which is already in the project.
2. The state is 1-D (predicted_score posterior); pykalman's matrix machinery
   is excessive and obscures the canonical-normal update we want to audit.
3. The canonical normal-normal posterior update (see Bishop 2006 PRML §2.3.3
   or Kalman 1960 §4 for the 1-D case) is one line: ``mean := (variance*y +
   sigma2_obs*mean) / (variance + sigma2_obs)``.
4. EMA decay α=0.997 mirrors CLAUDE.md "EMA — NON-NEGOTIABLE" weight-EMA
   default (Quantizr 0.33 anchor) so the operator sees a consistent decay
   across both the weight-update domain and the prediction-update domain.

State-space formalization (canonical Kalman 1960):

  Observation model:    y_n = x_n + v_n   (v_n ~ N(0, sigma2_obs))
  State model:          x_n = x_{n-1} + w_n  (w_n ~ N(0, sigma2_proc))

  Posterior:
    K_n = P_{n|n-1} / (P_{n|n-1} + sigma2_obs)         # Kalman gain
    x_{n|n} = x_{n|n-1} + K_n * (y_n - x_{n|n-1})
    P_{n|n} = (1 - K_n) * P_{n|n-1}

  Time update (EMA-style process noise):
    x_{n+1|n} = x_{n|n}
    P_{n+1|n} = alpha * P_{n|n} + (1 - alpha) * sigma2_proc_init

  where alpha = DEFAULT_EMA_DECAY = 0.997 inflates the posterior variance
  geometrically toward the initial process-noise prior so old observations
  decay in influence (the system is non-stationary; later epochs are more
  trustworthy than earlier ones).

[verified-against:Kalman 1960 "A New Approach to Linear Filtering and Prediction Problems"]
[verified-against:Bishop 2006 PRML §2.3.3 + §13.3]
[verified-against:scipy.stats.norm conjugate-prior update]

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against":
this is canonical helper code; the formula is dispatch-time arithmetic
that future subagents can audit without reference to the scipy reference
implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

# Per CLAUDE.md "EMA — NON-NEGOTIABLE": Quantizr decay = 0.997 default.
# Sister discipline for the prediction-update domain.
DEFAULT_EMA_DECAY = 0.997


@dataclass(frozen=True)
class KalmanState:
    """Frozen scalar Kalman state — running posterior over predicted_score.

    Attributes:
        posterior_mean: most-recent posterior mean (x_{n|n}).
        posterior_variance: most-recent posterior variance (P_{n|n}).
        n_observations: number of samples folded into the posterior.
        ema_decay: alpha in [0, 1]; defaults to 0.997 per CLAUDE.md.
        sigma2_obs: observation noise variance (per-sample); defaults to 1e-4
            (~ 0.01 score-units sigma; informed by Quantizr 0.33 anchor's
            sample-to-sample variability on cheap smoke).
        sigma2_proc_init: process noise variance prior; geometrically inflates
            posterior variance over time so old observations decay.
    """

    posterior_mean: float
    posterior_variance: float
    n_observations: int
    ema_decay: float = DEFAULT_EMA_DECAY
    sigma2_obs: float = 1e-4
    sigma2_proc_init: float = 1e-4

    def __post_init__(self) -> None:
        # Per CLAUDE.md "Beauty, simplicity, and developer experience": fail loud
        # at construction so silent NaN-propagation never escapes.
        if not math.isfinite(self.posterior_mean):
            raise ValueError(f"posterior_mean must be finite; got {self.posterior_mean!r}")
        if not math.isfinite(self.posterior_variance) or self.posterior_variance < 0:
            raise ValueError(
                f"posterior_variance must be a non-negative finite number; "
                f"got {self.posterior_variance!r}"
            )
        if self.n_observations < 0:
            raise ValueError(f"n_observations must be non-negative; got {self.n_observations!r}")
        if not (0.0 <= self.ema_decay <= 1.0):
            raise ValueError(f"ema_decay must be in [0, 1]; got {self.ema_decay!r}")
        if self.sigma2_obs <= 0:
            raise ValueError(f"sigma2_obs must be positive; got {self.sigma2_obs!r}")
        if self.sigma2_proc_init <= 0:
            raise ValueError(f"sigma2_proc_init must be positive; got {self.sigma2_proc_init!r}")

    @property
    def posterior_std(self) -> float:
        return math.sqrt(self.posterior_variance)

    def as_dict(self) -> dict[str, Any]:
        return {
            "posterior_mean": self.posterior_mean,
            "posterior_variance": self.posterior_variance,
            "posterior_std": self.posterior_std,
            "n_observations": self.n_observations,
            "ema_decay": self.ema_decay,
            "sigma2_obs": self.sigma2_obs,
            "sigma2_proc_init": self.sigma2_proc_init,
        }


def create_initial_state(
    *,
    initial_mean: float,
    initial_variance: float = 1e-2,
    ema_decay: float = DEFAULT_EMA_DECAY,
    sigma2_obs: float = 1e-4,
    sigma2_proc_init: float = 1e-4,
) -> KalmanState:
    """Construct an initial KalmanState before the first observation.

    Args:
        initial_mean: prior mean (e.g., baseline score for the substrate class).
        initial_variance: prior variance (default 1e-2 = ~0.1 score-units sigma;
            broad enough to be quickly overwhelmed by the first observation).
        ema_decay: defaults to DEFAULT_EMA_DECAY (0.997).
        sigma2_obs: observation noise variance default.
        sigma2_proc_init: process noise variance prior default.

    Returns:
        Frozen KalmanState at n_observations=0.
    """
    return KalmanState(
        posterior_mean=initial_mean,
        posterior_variance=initial_variance,
        n_observations=0,
        ema_decay=ema_decay,
        sigma2_obs=sigma2_obs,
        sigma2_proc_init=sigma2_proc_init,
    )


def update_state_with_sample(
    state: KalmanState,
    observation: float,
    *,
    observation_variance: float | None = None,
) -> KalmanState:
    """Fold a single observation into the Kalman state.

    Per Kalman 1960 §4 (1-D scalar case) + Bishop 2006 PRML §2.3.3:

      K = P_{n|n-1} / (P_{n|n-1} + sigma2_obs)         # Kalman gain
      x_{n|n} = x_{n|n-1} + K * (observation - x_{n|n-1})
      P_{n|n} = (1 - K) * P_{n|n-1}

    Time-update (EMA-style process noise propagation):
      P_{n+1|n} = alpha * P_{n|n} + (1 - alpha) * sigma2_proc_init

    Args:
        state: current frozen KalmanState (returned by ``create_initial_state``
            or a previous ``update_state_with_sample`` call).
        observation: new predicted_score sample.
        observation_variance: optional per-sample variance override; defaults
            to ``state.sigma2_obs``. Useful when M_sample size varies per call
            (smaller M = larger observation_variance).

    Returns:
        New frozen KalmanState with n_observations incremented and posterior
        updated per the canonical normal-normal conjugate update.

    Raises:
        ValueError: if observation is not finite.
    """
    if not math.isfinite(observation):
        raise ValueError(f"observation must be finite; got {observation!r}")

    sigma2_obs = (
        observation_variance if observation_variance is not None else state.sigma2_obs
    )
    if sigma2_obs <= 0:
        raise ValueError(f"observation_variance must be positive; got {sigma2_obs!r}")

    # Measurement update (canonical Kalman gain).
    p_pred = state.posterior_variance
    gain = p_pred / (p_pred + sigma2_obs)
    new_mean = state.posterior_mean + gain * (observation - state.posterior_mean)
    new_variance_post_update = (1.0 - gain) * p_pred

    # Time update: EMA-decay process noise injection. This inflates the
    # posterior variance toward the initial process-noise prior so old
    # observations decay geometrically per the EMA decay.
    alpha = state.ema_decay
    new_variance = alpha * new_variance_post_update + (1.0 - alpha) * state.sigma2_proc_init

    return KalmanState(
        posterior_mean=new_mean,
        posterior_variance=new_variance,
        n_observations=state.n_observations + 1,
        ema_decay=state.ema_decay,
        sigma2_obs=state.sigma2_obs,
        sigma2_proc_init=state.sigma2_proc_init,
    )


def detect_stop_loss(
    state: KalmanState,
    initial_estimate: float,
    *,
    deviation_sigma: float = 3.0,
    minimum_observations: int = 3,
) -> tuple[bool, str]:
    """Return ``(triggered, rationale)`` for stop-loss recommendation.

    Stop-loss fires when the Kalman posterior_mean has worsened by more than
    ``deviation_sigma`` standard deviations beyond the initial estimate AND
    at least ``minimum_observations`` samples have been folded into the
    posterior (so we don't trigger on the first noisy sample).

    For score-minimization (the contest), "worsened" means posterior_mean
    is GREATER than initial_estimate by deviation_sigma * posterior_std.

    Returns:
        ``(True, rationale)`` if stop-loss should trigger.
        ``(False, rationale)`` otherwise.
    """
    if state.n_observations < minimum_observations:
        return (
            False,
            f"insufficient_observations ({state.n_observations} < {minimum_observations})",
        )
    if state.posterior_std == 0:
        return (False, "posterior_std=0 (degenerate; refuse to trigger)")
    deviation = (state.posterior_mean - initial_estimate) / state.posterior_std
    if deviation > deviation_sigma:
        return (
            True,
            (
                f"posterior_mean {state.posterior_mean:.5g} worsened "
                f"{deviation:.2f}σ above initial_estimate {initial_estimate:.5g} "
                f"(threshold {deviation_sigma}σ)"
            ),
        )
    return (
        False,
        (
            f"posterior_mean {state.posterior_mean:.5g} vs initial {initial_estimate:.5g} "
            f"deviation {deviation:.2f}σ within threshold {deviation_sigma}σ"
        ),
    )


def detect_convergence(
    state: KalmanState,
    *,
    sigma_threshold: float = 5e-3,
    minimum_observations: int = 5,
) -> tuple[bool, str]:
    """Return ``(detected, rationale)`` for convergence recommendation.

    Convergence fires when posterior_std drops below ``sigma_threshold`` AND
    at least ``minimum_observations`` samples have been folded in.

    Returns:
        ``(True, rationale)`` if convergence is detected.
        ``(False, rationale)`` otherwise.
    """
    if state.n_observations < minimum_observations:
        return (
            False,
            f"insufficient_observations ({state.n_observations} < {minimum_observations})",
        )
    if state.posterior_std < sigma_threshold:
        return (
            True,
            (
                f"posterior_std {state.posterior_std:.3g} below threshold "
                f"{sigma_threshold:.3g} after {state.n_observations} observations"
            ),
        )
    return (
        False,
        (
            f"posterior_std {state.posterior_std:.3g} above threshold "
            f"{sigma_threshold:.3g}"
        ),
    )
