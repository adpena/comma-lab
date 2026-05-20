# SPDX-License-Identifier: MIT
"""Trainer-side hook for streaming master-gradient sampling.

Per SLOT MG-5 of the 2026-05-19 master-gradient enhancement wave.

The :class:`StreamingMasterGradientHook` is a callback class compatible with
both bare PyTorch training loops AND lightning.LightningModule callbacks.
Every N epochs it:

  1. Extracts a small M_sample via the canonical master-gradient helper
  2. Computes the Taylor-extrapolation predicted_score over the current weights
  3. Folds the prediction into a Kalman posterior
  4. Registers the row in the canonical streaming-prediction ledger
  5. Optionally calls user-supplied callbacks for convergence + stop-loss

SCAFFOLD only — this slot deliberately DOES NOT modify existing substrate
trainers per the task scope. The integration pattern is documented in the
hook's docstring + landing memo. The next training run that imports this
hook (and the future MG-1+MG-2+MG-3+MG-4 sister slots that may auto-wire
it) starts the empirical-anchor cycle.

Integration pattern (bare PyTorch loop)::

    from tac.training.streaming_master_gradient_hook import StreamingMasterGradientHook
    from tac.master_gradient import predict_delta_s  # canonical predictor

    hook = StreamingMasterGradientHook(
        substrate="substrate_z3_g1",
        subagent_id="claude_slot_X_<utc>",
        sample_every_n_epochs=10,
        initial_estimate=0.198,  # prior expected score
        predictor_fn=lambda model, archive_path: predict_delta_s(...),
        archive_path_fn=lambda epoch: tmp_archive_dir / f"epoch_{epoch}.zip",
    )

    for epoch in range(total_epochs):
        train_epoch(model, ...)
        hook.on_epoch_end(model, epoch=epoch, wall_clock_seconds=time.time() - t0)

Integration pattern (lightning.LightningModule)::

    class MyLightning(LightningModule):
        def __init__(self, ...):
            super().__init__()
            self.streaming_hook = StreamingMasterGradientHook(...)

        def on_train_epoch_end(self) -> None:
            self.streaming_hook.on_epoch_end(self, epoch=self.current_epoch, ...)

Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
#323: the predicted_score the hook registers is ALWAYS
``evidence_grade=PREDICTED`` and the row carries a canonical Provenance
built by :func:`tac.provenance.builders.build_provenance_for_predicted`.

Catalog #324 phantom-random-init guard: epoch=0 samples are tagged
``phantom_random_init=true`` automatically by the ledger; the dashboard
ignores them for convergence/stop-loss decisions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Optional

from tac.streaming_prediction.kalman_filter import (
    DEFAULT_EMA_DECAY,
    KalmanState,
    create_initial_state,
    detect_convergence,
    detect_stop_loss,
    update_state_with_sample,
)
from tac.streaming_prediction.streaming_prediction_ledger import (
    register_convergence_event,
    register_stop_loss_event,
    register_streaming_sample,
)

# Canonical typing aliases.
ModelLike = Any  # torch.nn.Module / lightning.LightningModule / etc.
PredictorFn = Callable[[ModelLike, Optional[Path]], float]
"""Canonical predictor signature: ``(model, archive_path_or_None) -> predicted_score``.

The predictor SHOULD route through ``tac.master_gradient`` for the canonical
Taylor-extrapolation; this signature is intentionally generic so any
custom in-trainer predictor satisfies it.
"""

ArchivePathFn = Callable[[int], Optional[Path]]
"""Optional callable that returns the archive_path for ``epoch`` (or None).

When the trainer doesn't build an archive every N epochs (typical), this
returns None and the ledger row's ``archive_sha256`` is null. When the
trainer does build an archive, returning the Path lets the ledger record
the sha for cite-chain auditing.
"""


class StreamingMasterGradientHook:
    """Trainer-side hook: extract master-gradient + register Kalman-fused sample every N epochs.

    Attributes:
        substrate: canonical substrate id (e.g., "substrate_z3_g1").
        subagent_id: the dispatching subagent's id (for posterior attribution).
        sample_every_n_epochs: cadence (default 10).
        initial_estimate: prior expected score (used by stop-loss).
        predictor_fn: callable mapping (model, archive_path) -> predicted_score.
        archive_path_fn: optional callable mapping epoch -> archive_path.
        ema_decay: Kalman EMA decay (default 0.997 per CLAUDE.md).
        sigma2_obs: observation noise variance (default 1e-4).
        convergence_threshold: posterior_std threshold (default 5e-3).
        stop_loss_sigma: deviation_sigma for stop-loss (default 3.0).
        on_convergence_callback: optional callable invoked when convergence
            is detected, receives ``(state, epoch)``.
        on_stop_loss_callback: optional callable invoked when stop-loss
            triggers, receives ``(state, epoch)``.
        kalman_state: live KalmanState (mutable). None until first sample.
    """

    def __init__(
        self,
        *,
        substrate: str,
        subagent_id: str,
        sample_every_n_epochs: int = 10,
        initial_estimate: float = 0.2,
        predictor_fn: PredictorFn,
        archive_path_fn: ArchivePathFn | None = None,
        ema_decay: float = DEFAULT_EMA_DECAY,
        sigma2_obs: float = 1e-4,
        sigma2_proc_init: float = 1e-4,
        convergence_threshold: float = 5e-3,
        stop_loss_sigma: float = 3.0,
        agent: str = "claude",
        on_convergence_callback: Callable[[KalmanState, int], None] | None = None,
        on_stop_loss_callback: Callable[[KalmanState, int], None] | None = None,
    ) -> None:
        if not substrate:
            raise ValueError("substrate must be a non-empty string")
        if not subagent_id:
            raise ValueError("subagent_id must be a non-empty string")
        if sample_every_n_epochs <= 0:
            raise ValueError("sample_every_n_epochs must be positive")
        if not math.isfinite(initial_estimate):
            raise ValueError("initial_estimate must be finite")
        if not callable(predictor_fn):
            raise TypeError("predictor_fn must be callable")
        if archive_path_fn is not None and not callable(archive_path_fn):
            raise TypeError("archive_path_fn must be callable or None")

        self.substrate = substrate
        self.subagent_id = subagent_id
        self.sample_every_n_epochs = sample_every_n_epochs
        self.initial_estimate = initial_estimate
        self.predictor_fn = predictor_fn
        self.archive_path_fn = archive_path_fn
        self.ema_decay = ema_decay
        self.sigma2_obs = sigma2_obs
        self.sigma2_proc_init = sigma2_proc_init
        self.convergence_threshold = convergence_threshold
        self.stop_loss_sigma = stop_loss_sigma
        self.agent = agent
        self.on_convergence_callback = on_convergence_callback
        self.on_stop_loss_callback = on_stop_loss_callback

        # State accumulators.
        self.kalman_state: KalmanState | None = None
        self.last_sample_epoch: int | None = None
        self._convergence_fired = False
        self._stop_loss_fired = False

    def on_epoch_end(
        self,
        model: ModelLike,
        *,
        epoch: int,
        wall_clock_seconds: float,
        m_sample_size: int = 8,
        force_sample: bool = False,
    ) -> dict[str, Any] | None:
        """Called by trainer at end of each epoch.

        Args:
            model: the trainer's model (torch.nn.Module / LightningModule).
            epoch: current epoch (0-indexed).
            wall_clock_seconds: seconds since training started.
            m_sample_size: number of pairs in the Taylor-expansion sample
                (passed-through to the ledger row for downstream uncertainty
                weighting per sister Slot MG-1).
            force_sample: if True, take a sample regardless of cadence.

        Returns:
            The persisted ledger row dict if a sample was taken; None otherwise.
        """
        if not force_sample and (epoch % self.sample_every_n_epochs != 0):
            return None
        if epoch == self.last_sample_epoch:
            return None  # de-dup: don't sample same epoch twice

        # Compute predicted_score via canonical predictor.
        archive_path = self.archive_path_fn(epoch) if self.archive_path_fn else None
        try:
            predicted_score = float(self.predictor_fn(model, archive_path))
        except Exception as exc:  # noqa: BLE001 — robust to predictor failure
            # Predictor failure is NON-fatal: log it as ledger row with
            # predicted_score=NaN-equivalent sentinel and continue training.
            # Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
            # against": we don't crash the training run, but we do record the
            # failure so the dashboard sees the gap.
            return self._register_failure_row(
                epoch=epoch,
                wall_clock_seconds=wall_clock_seconds,
                m_sample_size=m_sample_size,
                exc_str=str(exc),
            )
        if not math.isfinite(predicted_score):
            return self._register_failure_row(
                epoch=epoch,
                wall_clock_seconds=wall_clock_seconds,
                m_sample_size=m_sample_size,
                exc_str=f"predictor returned non-finite value: {predicted_score!r}",
            )

        # Update Kalman state.
        if self.kalman_state is None:
            self.kalman_state = create_initial_state(
                initial_mean=predicted_score,
                ema_decay=self.ema_decay,
                sigma2_obs=self.sigma2_obs,
                sigma2_proc_init=self.sigma2_proc_init,
            )
        else:
            self.kalman_state = update_state_with_sample(
                self.kalman_state, predicted_score
            )

        # Compute archive sha if available.
        archive_sha = None
        if archive_path and archive_path.exists():
            archive_sha = self._sha256_of_file(archive_path)

        # Build canonical Provenance payload.
        provenance_payload = self._build_provenance_payload(
            epoch=epoch, archive_sha=archive_sha
        )

        # Register the sample row.
        row = register_streaming_sample(
            subagent_id=self.subagent_id,
            substrate=self.substrate,
            epoch=epoch,
            wall_clock_seconds=wall_clock_seconds,
            m_sample_size=m_sample_size,
            predicted_score=predicted_score,
            posterior_mean=self.kalman_state.posterior_mean,
            posterior_std=self.kalman_state.posterior_std,
            posterior_n_observations=self.kalman_state.n_observations,
            archive_sha256=archive_sha,
            provenance=provenance_payload,
            agent=self.agent,
        )
        self.last_sample_epoch = epoch

        # Check convergence + stop-loss (only once each).
        if not self._convergence_fired:
            conv_triggered, conv_rationale = detect_convergence(
                self.kalman_state, sigma_threshold=self.convergence_threshold
            )
            if conv_triggered:
                self._convergence_fired = True
                try:
                    register_convergence_event(
                        subagent_id=self.subagent_id,
                        substrate=self.substrate,
                        epoch=epoch,
                        posterior_mean=self.kalman_state.posterior_mean,
                        posterior_std=self.kalman_state.posterior_std,
                        convergence_threshold=self.convergence_threshold,
                        rationale=conv_rationale,
                        agent=self.agent,
                    )
                except Exception:  # noqa: BLE001 — recommendation, not score
                    pass  # dashboard will detect convergence independently
                if self.on_convergence_callback:
                    try:
                        self.on_convergence_callback(self.kalman_state, epoch)
                    except Exception:  # noqa: BLE001 — user callback, isolated
                        pass

        if not self._stop_loss_fired and epoch > 0:
            stop_triggered, stop_rationale = detect_stop_loss(
                self.kalman_state,
                self.initial_estimate,
                deviation_sigma=self.stop_loss_sigma,
            )
            if stop_triggered:
                self._stop_loss_fired = True
                try:
                    register_stop_loss_event(
                        subagent_id=self.subagent_id,
                        substrate=self.substrate,
                        epoch=epoch,
                        posterior_mean=self.kalman_state.posterior_mean,
                        posterior_std=self.kalman_state.posterior_std,
                        initial_estimate=self.initial_estimate,
                        deviation_sigma=self.stop_loss_sigma,
                        rationale=stop_rationale,
                        agent=self.agent,
                    )
                except Exception:  # noqa: BLE001
                    pass
                if self.on_stop_loss_callback:
                    try:
                        self.on_stop_loss_callback(self.kalman_state, epoch)
                    except Exception:  # noqa: BLE001
                        pass

        return row

    def _register_failure_row(
        self,
        *,
        epoch: int,
        wall_clock_seconds: float,
        m_sample_size: int,
        exc_str: str,
    ) -> dict[str, Any] | None:
        """Record predictor failure as a streaming row with safe sentinels."""
        # Use initial_estimate as fallback so the row is well-formed.
        fallback_score = self.initial_estimate
        if self.kalman_state is None:
            self.kalman_state = create_initial_state(
                initial_mean=fallback_score,
                ema_decay=self.ema_decay,
                sigma2_obs=self.sigma2_obs,
                sigma2_proc_init=self.sigma2_proc_init,
            )
        try:
            row = register_streaming_sample(
                subagent_id=self.subagent_id,
                substrate=self.substrate,
                epoch=epoch,
                wall_clock_seconds=wall_clock_seconds,
                m_sample_size=m_sample_size,
                predicted_score=fallback_score,
                posterior_mean=self.kalman_state.posterior_mean,
                posterior_std=self.kalman_state.posterior_std,
                posterior_n_observations=self.kalman_state.n_observations,
                archive_sha256=None,
                provenance=self._build_provenance_payload(
                    epoch=epoch, archive_sha=None, predictor_failure=exc_str
                ),
                agent=self.agent,
                extra={"predictor_failure": exc_str},
            )
            self.last_sample_epoch = epoch
            return row
        except Exception:  # noqa: BLE001
            return None

    def _build_provenance_payload(
        self,
        *,
        epoch: int,
        archive_sha: str | None,
        predictor_failure: str | None = None,
    ) -> dict[str, Any]:
        """Build canonical Provenance dict for the streaming sample.

        Routes through ``tac.provenance.builders.build_provenance_for_predicted``
        and serializes via ``as_dict`` for JSONL storage. Per Catalog #323.
        """
        try:
            from tac.provenance.builders import build_provenance_for_predicted

            # Use a deterministic inputs_sha256 derived from substrate + epoch
            # so the provenance is auditable. The Provenance contract requires
            # non-empty inputs_sha256; we use sha256 of "substrate:epoch" if no
            # archive is available, or the archive_sha256 if it is.
            if archive_sha:
                inputs_sha = archive_sha
            else:
                import hashlib

                inputs_sha = hashlib.sha256(
                    f"{self.substrate}:{epoch}:{self.subagent_id}".encode("utf-8")
                ).hexdigest()
            prov = build_provenance_for_predicted(
                model_id=f"streaming_master_gradient_hook.{self.substrate}",
                inputs_sha256=inputs_sha,
            )
            payload = {
                "artifact_kind": prov.artifact_kind.value,
                "source_path": prov.source_path,
                "source_sha256": prov.source_sha256,
                "measurement_axis": prov.measurement_axis,
                "hardware_substrate": prov.hardware_substrate,
                "evidence_grade": prov.evidence_grade.value,
                "promotion_eligible": prov.promotion_eligible,
                "score_claim_valid": prov.score_claim_valid,
                "captured_at_utc": prov.captured_at_utc,
                "canonical_helper_invocation": prov.canonical_helper_invocation,
            }
            if predictor_failure:
                payload["predictor_failure"] = predictor_failure
            return payload
        except Exception:  # noqa: BLE001 — fallback to bare dict if provenance fails
            return {
                "artifact_kind": "predicted_from_model",
                "measurement_axis": "[predicted]",
                "evidence_grade": "predicted",
                "promotion_eligible": False,
                "fallback_reason": "provenance_builder_failed",
                "predictor_failure": predictor_failure,
            }

    @staticmethod
    def _sha256_of_file(path: Path) -> str:
        """Compute sha256 of file contents."""
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
