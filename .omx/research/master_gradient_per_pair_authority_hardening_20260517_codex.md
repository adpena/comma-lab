# Master-Gradient Per-Pair Authority Hardening - 2026-05-17

## Scope

This landing hardens the master-gradient/per-pair-gradient WIP in response to
the operator question about whether the current gradient captures per-pair
signal. It does not. The averaged `(N_bytes, 3)` tensor collapses pair-wise
sign cancellation and variance.

This is a code/API hardening artifact, not a score claim, promotion, dispatch,
or archive candidate.

## Finding

The extractor already had a `--preserve-per-pair` path that emitted a sister
sidecar with shape `(N_bytes, N_pairs, 3)`, but it recorded that sidecar through
the same `MasterGradient` metadata contract as the aggregate `(N_bytes, 3)`
anchor.

Failure class:

- per-pair sidecar rows looked like canonical master-gradient anchors;
- `MasterGradient.load_gradient()` rejected the sidecar shape after ledger
  lookup;
- downstream consumers had no typed way to distinguish aggregate score
  sensitivity from per-pair score sensitivity;
- the exact signal the operator asked to preserve could be silently flattened
  or become unusable.

## Fix

- Added explicit tensor kinds:
  - `aggregate_per_byte_v1` for `(N_bytes, 3)`;
  - `per_pair_per_byte_v1` for `(N_bytes, N_pairs, 3)`.
- Added per-pair metadata validation: per-pair anchors require positive
  integer `n_pairs`; aggregate anchors must not set `n_pairs`.
- Added `load_per_pair_gradient()` and `predict_delta_s_per_pair()` so callers
  receive a vector over pairs instead of an accidentally averaged scalar.
- Updated `tools/extract_master_gradient.py --preserve-per-pair` to write
  `gradient_tensor_kind=per_pair_per_byte_v1` and `n_pairs=<N>`.
- Updated `update_from_anchor()` to preserve per-pair tensor metadata.

## Why This Matters For Score Lowering

The aggregate tensor can rank byte directions near the current operating point,
but it cannot expose:

- pair-wise sign cancellation;
- bytes that help hard pairs while hurting easy pairs;
- per-pair variance for risk-adjusted allocation;
- TT5L / side-info / Wyner-Ziv pair-specific routing opportunities;
- per-pair Pareto water-filling for stack-of-stacks candidates.

The per-pair tensor is therefore a frontier-search instrument, not merely a
debug artifact. It supports lower-score work by preserving the pair axis until
the allocator or candidate builder explicitly decides how to aggregate it.

## Verification

```bash
.venv/bin/ruff check src/tac/master_gradient.py tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py
.venv/bin/python -m pytest src/tac/tests/test_extract_master_gradient.py -q
.venv/bin/python -m py_compile src/tac/master_gradient.py tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py
git diff --check -- src/tac/master_gradient.py tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py
```

Results:

```text
All checks passed
15 passed
py_compile clean
diff --check clean
```

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

Next score-lowering step: run a cheap per-pair master-gradient smoke
(`--preserve-per-pair --n-pairs-used 8`) against the current FEC6 CPU anchor,
then use the resulting pair-axis variance/cancellation to prioritize byte-level
operators and TT5L side-info cells.
