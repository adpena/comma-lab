# HPRC protected pose pathway trainability fix

Codex finding, 2026-06-01T14:41:17Z.

## Verdict

The HPRC protected high-resolution residual pathway was archive-bound and
runtime-consumed, but its only runtime control knob,
`protected_residual_gain`, was not exposed to the training adapter state,
gradient update, checkpoint, or training export. That made the pose/protected
sidecar a partially fake trainable pathway: bytes were being spent, but the
long-training loop could not tune the sidecar contribution.

## Fix

- Added `protected_residual_gain` to HPRC trainable state, numpy and MLX gain
  gradients, RDO plan export, artifact metadata, and score-aware telemetry.
- Kept the protected tensor itself archive-owned for now. Training the full
  tensor without a binary checkpoint would create false resume/EMA authority.
- Moved HPRC training checkpoints to `.npz` so mutated coarse residual tokens
  and protected residual tensors are resume-safe. JSON manifests now carry
  shape/byte/SHA summaries instead of embedding arrays.

## Proof

Tests added:

- protected gain changes under reconstruction loss while `protected_residual`
  tensor remains unchanged;
- exported packet consumes the trained gain in the numpy receiver RDO plan;
- `.npz` state round-trips gains, `train_steps`, coarse residuals, and protected
  residual tensors.

Validation:

```text
uv run ruff check src/tac/substrates/hprc/training_adapter.py src/tac/substrates/hprc/tests/test_training_adapter.py
PYTHONPATH=. uv run pytest src/tac/substrates/hprc/tests/test_training_adapter.py -q
PYTHONPATH=. uv run pytest src/tac/substrates/hprc/tests/test_learned_receiver.py src/tac/substrates/hprc/tests/test_rate_collapse.py src/tac/substrates/hprc/tests/test_training_adapter.py -q
```

## Next actuation

Run the queue-owned full600 HPRC pose-guard campaign with P18/P19 surfaces,
MLX prefilter batch-pairs=1, rate gate, receiver proof, CPU replay only for
MLX survivors, and exact auth only for local winners. If the trained protected
gain still leaves MLX score in the 20+ range, demote the compact HPRC sidecar
configuration and escalate to native pose-aware renderer/substrate training
rather than more posthoc sidecar tuning.

