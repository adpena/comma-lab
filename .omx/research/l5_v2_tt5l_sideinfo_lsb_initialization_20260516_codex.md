# L5 v2 TT5L Side-Info LSB Initialization

Date: 2026-05-16
Author: Codex
Axis: TT5L / L5 v2 trainer hardening
Evidence grade: source-and-test hardening; no score claim

## Finding

The TT5L smoke path emitted nonzero random side-info before archive packing,
but the full trainer initialized `per_pair_side_info_float` as exact zeros via
`0.0 * torch.randn(...)`.

That creates a misleading low-signal start state for short, interrupted, or
timing-only TT5L runs: the archive grammar and optimizer can appear wired while
the emitted side-info stream remains indistinguishable from a no-side-channel
packet unless training moves it far enough. This is especially risky after the
L5 v2 Dykstra/consumption gates because those gates require the stream to be
real, archive-basis-bound, and not cargo-cult metadata.

## Change

`experiments/train_substrate_time_traveler_l5_autonomy.py` now initializes the
trainable side-info tensor at roughly one archive int8 least-significant bit:

`torch.randn(n_pairs, per_pair_side_info_bytes) / int8_scale`

The helper validates dimensions and positive scale, preserves deterministic
global seeding from `_pin_seeds(args.seed)`, and does not alter TT5L archive
grammar, side-info byte budget, or inflate semantics.

## Guard

`src/tac/tests/test_train_time_traveler_full_cpu_mode.py` now asserts the
initializer produces a trainable tensor whose quantized int8-domain values are
nonzero at the same scale consumed by TT5L archive/inflate.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_train_time_traveler_full_cpu_mode.py -q`
  - `26 passed`
- `ruff check experiments/train_substrate_time_traveler_l5_autonomy.py src/tac/tests/test_train_time_traveler_full_cpu_mode.py`
  - `All checks passed`

## No Score Claim

This is not a contest result. It is a trainer correctness and signal-preserving
hardening patch before the next TT5L side-info proof, timing smoke, or full
contest-axis run.
