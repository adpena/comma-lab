# SPDX-License-Identifier: MIT
"""tac.streaming_prediction — canonical streaming master-gradient + Kalman state-space predictor.

Per SLOT MG-5 of the 2026-05-19 5-slot master-gradient enhancement wave.

This package operationalizes the **streaming master-gradient sampling SCAFFOLD**
for substrate trainers. During training, a `StreamingMasterGradientHook`
samples a small M_sample at every N epochs, computes a Taylor-expansion
predicted_score, and registers a row in the canonical fcntl-locked JSONL
ledger at `.omx/state/streaming_predictions.jsonl`. Operators then tail
the ledger via `tools/realtime_prediction_dashboard.py` to observe
convergence + trigger stop-loss when the Kalman posterior worsens
> 3-sigma beyond the initial estimate.

Canonical surfaces (per CLAUDE.md "Beauty, simplicity, and developer experience"):

- :func:`register_streaming_sample` — single-line write to canonical ledger
- :func:`load_streaming_samples` — lenient read for dashboards
- :func:`load_streaming_samples_strict` — fail-closed for mutating callers
- :func:`latest_for_substrate` — most-recent sample per substrate
- :class:`KalmanState` — running posterior mean + variance per substrate
- :class:`GaussianProcessSurrogate` — sklearn.gaussian_process wrapper

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag (the
docstring-overstatement trap)" + Catalog #323 canonical Provenance: every
streaming sample is tagged `evidence_grade=PREDICTED` and is NEVER a
score claim — it is the live posterior over the predictor's belief about
where the final score will land.

Sister-of:
- :mod:`tac.deploy.modal.call_id_ledger` — canonical 4-layer pattern this
  package mirrors (Catalog #245)
- :mod:`tac.master_gradient` — Taylor-expansion predictor (PREDICTED grade)
- :mod:`tac.continual_learning` — Catalog #128 fcntl-locked posterior store
- :mod:`tac.cathedral_consumers.streaming_prediction_consumer` — auto-discovered
  cathedral consumer that emits stop-loss + convergence recommendations

Catalog #351 STRICT preflight gate refuses bare writes to the ledger path
outside the canonical helpers below.

[verified-against:Kalman 1960 "A New Approach to Linear Filtering and Prediction Problems"]
[verified-against:Rasmussen+Williams 2006 "Gaussian Processes for Machine Learning"]
"""

from __future__ import annotations

from tac.streaming_prediction.streaming_prediction_ledger import (
    STREAMING_PREDICTION_LEDGER_PATH,
    STREAMING_PREDICTION_LEDGER_LOCK,
    SCHEMA_VERSION,
    EVENT_SAMPLED,
    EVENT_CONVERGENCE_DETECTED,
    EVENT_STOP_LOSS_TRIGGERED,
    VALID_EVENT_TYPES,
    StreamingPredictionLedgerCorruptError,
    register_streaming_sample,
    register_convergence_event,
    register_stop_loss_event,
    load_streaming_samples,
    load_streaming_samples_strict,
    latest_for_substrate,
    query_by_substrate,
    query_all_post_utc,
)
from tac.streaming_prediction.kalman_filter import (
    KalmanState,
    DEFAULT_EMA_DECAY,
    create_initial_state,
    update_state_with_sample,
    detect_stop_loss,
    detect_convergence,
)
from tac.streaming_prediction.gaussian_process_regression import (
    GaussianProcessSurrogate,
    fit_surrogate_from_samples,
)

__all__ = [
    # Ledger
    "STREAMING_PREDICTION_LEDGER_PATH",
    "STREAMING_PREDICTION_LEDGER_LOCK",
    "SCHEMA_VERSION",
    "EVENT_SAMPLED",
    "EVENT_CONVERGENCE_DETECTED",
    "EVENT_STOP_LOSS_TRIGGERED",
    "VALID_EVENT_TYPES",
    "StreamingPredictionLedgerCorruptError",
    "register_streaming_sample",
    "register_convergence_event",
    "register_stop_loss_event",
    "load_streaming_samples",
    "load_streaming_samples_strict",
    "latest_for_substrate",
    "query_by_substrate",
    "query_all_post_utc",
    # Kalman
    "KalmanState",
    "DEFAULT_EMA_DECAY",
    "create_initial_state",
    "update_state_with_sample",
    "detect_stop_loss",
    "detect_convergence",
    # Gaussian process surrogate
    "GaussianProcessSurrogate",
    "fit_surrogate_from_samples",
]
