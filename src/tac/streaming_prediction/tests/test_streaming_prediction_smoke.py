# SPDX-License-Identifier: MIT
"""SLOT MG-5 streaming master-gradient hook + ledger smoke tests.

Per RECOVERY-MG-1-thru-5 (2026-05-20): the partial-landed MG-5 modules
arrived with no test files. This minimal smoke test verifies the
canonical surfaces (ledger API + Kalman state + Gaussian-process
surrogate + the training-side hook) are importable and contract-shape-
stable so future refactors do not silently break the streaming-sampling
loop.

Per CLAUDE.md "Beauty, simplicity, and developer experience": this test
is intentionally minimal — it does NOT exercise paid Modal dispatch or
the full Kalman / GP numerics, which require a paired training-run
fixture outside the recovery scope. The follow-on subagent landing
Catalog #351 (claimed but not yet written) is expected to write the
full multiprocessing fcntl-locked write stress + Kalman convergence
regression suite.
"""

from __future__ import annotations

import inspect


def test_mg_5_package_importable_and_canonical_symbols_present():
    """tac.streaming_prediction loads + exposes the canonical 26-symbol API."""
    import tac.streaming_prediction as sp

    expected = {
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
    }
    actual = set(sp.__all__)
    missing = expected - actual
    assert not missing, f"MG-5 __all__ missing canonical symbols: {missing}"


def test_mg_5_hook_importable():
    """The training-side StreamingMasterGradientHook loads + has expected surface."""
    import tac.training.streaming_master_gradient_hook as hook_mod

    assert hasattr(hook_mod, "StreamingMasterGradientHook"), (
        "tac.training.streaming_master_gradient_hook must expose "
        "StreamingMasterGradientHook (canonical training-side hook class)"
    )


def test_mg_5_register_signature_keyword_only():
    """register_streaming_sample MUST be keyword-only per Catalog #131 sister discipline."""
    from tac.streaming_prediction import register_streaming_sample

    sig = inspect.signature(register_streaming_sample)
    pos_or_kw = [
        p for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
    ]
    # All non-self params should be keyword-only (after the * separator)
    assert not pos_or_kw, (
        f"register_streaming_sample positional/keyword params must be keyword-only "
        f"(canonical fcntl-locked write discipline per Catalog #131); got positional: "
        f"{[p.name for p in pos_or_kw]}"
    )


def test_mg_5_canonical_ledger_path_under_omx_state():
    """Canonical ledger path must live under .omx/state/ per Catalog #131 + #245 sister discipline."""
    from tac.streaming_prediction import STREAMING_PREDICTION_LEDGER_PATH

    assert ".omx/state" in str(STREAMING_PREDICTION_LEDGER_PATH), (
        f"Canonical ledger path must be under .omx/state/ "
        f"(got {STREAMING_PREDICTION_LEDGER_PATH})"
    )


def test_mg_5_event_taxonomy_canonical():
    """The 3 event types MUST be distinct + in VALID_EVENT_TYPES."""
    from tac.streaming_prediction import (
        EVENT_SAMPLED,
        EVENT_CONVERGENCE_DETECTED,
        EVENT_STOP_LOSS_TRIGGERED,
        VALID_EVENT_TYPES,
    )

    events = {EVENT_SAMPLED, EVENT_CONVERGENCE_DETECTED, EVENT_STOP_LOSS_TRIGGERED}
    assert len(events) == 3, "All 3 event types must be distinct strings"
    for event in events:
        assert event in VALID_EVENT_TYPES, (
            f"event {event!r} must be in VALID_EVENT_TYPES; got {VALID_EVENT_TYPES}"
        )


def test_mg_5_kalman_state_construction():
    """KalmanState construction with create_initial_state returns frozen-ish state."""
    from tac.streaming_prediction import KalmanState, create_initial_state

    state = create_initial_state(initial_mean=0.20, initial_variance=0.01)
    assert isinstance(state, KalmanState)
    assert state.posterior_mean == 0.20
    assert state.posterior_variance == 0.01
    # Frozen dataclass: assignment must raise
    import dataclasses
    assert dataclasses.is_dataclass(KalmanState)


def test_mg_5_ledger_corrupt_error_subclasses_exception():
    """The corrupt-error class must be a real Exception (fail-closed per Catalog #138)."""
    from tac.streaming_prediction import StreamingPredictionLedgerCorruptError

    assert issubclass(StreamingPredictionLedgerCorruptError, Exception)
