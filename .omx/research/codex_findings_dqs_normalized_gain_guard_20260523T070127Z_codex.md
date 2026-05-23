# Codex Findings - DQS Normalized Gain Guard

UTC: 2026-05-23T07:01:27Z
Author: Codex
Lane: lane_codex_dqs_normalized_gain_guard_20260523

## Scope

Adversarial hardening pass for decoder-q selective (DQS1) MLX response-surface
consumers. This pass targets a specific signal-loss class: treating raw
per-window MLX gain as if it were already normalized to full-video score units.

## Finding

DQS selector, packet, and feedback builders already carried explicit
`observed_mlx_window_gain`, `normalized_full_video_gain`, and
`full_video_denominator` fields, but consumers trusted the normalized field
after basic numeric parsing. A malformed bridge work unit could therefore
inflate selector ranking or feedback transfer metrics by roughly the full
denominator if raw gain leaked into the normalized alias.

## Permanent Fix

Added `require_normalized_full_video_gain(...)` in
`tac.optimization.normalized_objective` and wired it into:

- `tac.optimization.decoder_q_selective_selector_pareto._bridge_units`
- `tac.optimization.decoder_q_selective_runtime_packet._selected_mlx_gain_sums`
- `tac.optimization.decoder_q_selective_runtime_feedback._sum_mlx_gains`

The guard recomputes:

```text
normalized_full_video_gain =
    observed_gain * source_n_samples / full_video_denominator
```

and fails closed on mismatch. The DQS consumers also enforce singleton pair
windows at the gain-summation boundary so a selected work unit cannot silently
change its sample count after bridge planning.

## Tests

Executed:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_normalized_objective.py \
  src/tac/tests/test_decoder_q_selective_selector_pareto.py \
  src/tac/tests/test_decoder_q_selective_runtime_packet.py \
  src/tac/tests/test_decoder_q_selective_runtime_feedback.py

.venv/bin/python -m ruff check \
  src/tac/optimization/normalized_objective.py \
  src/tac/optimization/decoder_q_selective_selector_pareto.py \
  src/tac/optimization/decoder_q_selective_runtime_packet.py \
  src/tac/optimization/decoder_q_selective_runtime_feedback.py \
  src/tac/tests/test_normalized_objective.py \
  src/tac/tests/test_decoder_q_selective_selector_pareto.py \
  src/tac/tests/test_decoder_q_selective_runtime_packet.py \
  src/tac/tests/test_decoder_q_selective_runtime_feedback.py

git diff --check
```

Results:

- `24 passed in 0.83s`
- `ruff`: all checks passed
- `git diff --check`: clean

## Remaining Integration

The same normalized-objective guard should be propagated into broader
scorer-response planning surfaces whenever they ingest partial-window rows.
This DQS pass closes the immediate PR110/DQS1 selector-to-packet-to-feedback
path without claiming score authority or changing promotion eligibility.

## Meta-Bug Closed

During lane bookkeeping, two concurrent `tools/lane_maturity.py mark` calls
produced a lost update in `.omx/state/lane_registry.json`: one gate mutation
overwrote the other because the CLI performed read-modify-write without a
transaction lock. The lane was repaired immediately and the tool now serializes
mutating commands (`add-lane`, `mark`, `unmark`, `set-field`) with a
`.omx/state/lane_maturity.lock` advisory lock. Registry writes are atomic
`os.replace(...)` operations and audit-log appends are flushed/fsynced under
the same transaction boundary.

Regression coverage:

```bash
.venv/bin/python -m pytest -q tests/test_lane_maturity_mutation_lock.py
```

The test launches two independent Python processes against a temporary registry
and verifies both gate updates survive.
