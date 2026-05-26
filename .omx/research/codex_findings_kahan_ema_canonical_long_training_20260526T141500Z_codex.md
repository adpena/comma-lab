# Codex Findings: Kahan EMA Canonical Long Training

UTC: 2026-05-26T14:15:00Z

## Verdict

The prior Kahan smoke and canonical long-training surface had a real numerical
analysis gap: live-vs-EMA lag was being treated as numerical-error mitigation,
and Kahan compensation was not checkpoint/resume durable. This landing makes
Kahan an explicit canonical EMA accumulation mode while keeping naive as the
default.

## What Changed

- Added `ema_accumulation="naive"|"kahan"` to `LongTrainingConfig`.
- Wired Kahan mode into `run_long_training(...)` with strict Kahan failure
  semantics when selected.
- Persisted EMA compensation state as a checkpoint sidecar and recorded the
  sidecar path plus fallback telemetry in checkpoint metadata.
- Added fail-closed resume checks for accumulation-mode mismatch and missing
  Kahan compensation state.
- Updated `tools/smoke_kahan_ema_vs_naive_z6.py` to capture the initial model
  state, replay identical live states through naive/Kahan EMA, and compare both
  against a NumPy fp64 reference.

## Authority Boundary

Kahan EMA is a training-numerics and local MLX substrate-hardening tool. It does
not make MLX scorer output a contest score, promotion signal, rank/kill signal,
or exact-dispatch authority. Exact CPU/CUDA auth eval remains the only
promotion axis.

## Verification

- `ruff` on touched training/smoke/test files passed.
- `py_compile` on touched training/smoke/test files passed.
- `pytest src/tac/substrates/_shared/tests/test_long_training_canonical.py -q`:
  64 passed.
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py -q`: 33 passed.

## Next

Run the updated Z6 Kahan smoke on a bounded MLX trajectory, then decide from the
fp64-reference error ratio whether Kahan mode should become a default for any
long-running local substrate family or remain an opt-in diagnostic.
