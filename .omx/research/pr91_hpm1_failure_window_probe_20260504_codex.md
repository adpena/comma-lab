# PR91 HPM1 Failure Window Probe - 2026-05-04

Scope: PR91/HPM1 entropy-contract recovery only. No remote job, scorer load,
exact eval, or score claim was performed.

## Code And Artifacts

- `src/tac/pr91_hpm1_codec.py` now lets
  `run_pr91_hpm1_first_symbol_state_probe(...)` trace a bounded global-symbol
  window with `symbol_offset` plus `symbol_count`.
- `experiments/replay_pr91_hpm1_mask.py` exposes `--symbol-offset`.
- The CLI `--probability-variant-matrix` now probes all registered HPAC
  probability variants when no explicit `--probability-variants` list is
  supplied.
- Current all-variant matrix:
  `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_probability_variant_matrix_frame0_20260504_current_codex.json`
- Compact failure-window fixture:
  `experiments/results/public_pr91_intake_20260504_codex/diagnostics/pr91_hpm1_failure_window_20260504_current_codex.json`

## Current Blocker

All local HPAC probability variants fail closed before completing PR91 HPM1
frame 0:

| Variant | Failure coordinate | Decoded symbols before failure |
| --- | --- | ---: |
| `source_float64_perfect_false` | `frame=0 group=10 symbol_in_group=191` | `5951` |
| `source_float32_perfect_false` | `frame=0 group=24 symbol_in_group=561` | `30513` |
| `source_float64_perfect_true` | `frame=0 group=15 symbol_in_group=1534` | `13822` |
| `source_float32_perfect_true` | `frame=0 group=15 symbol_in_group=191` | `12479` |

The compact source-contract window requested `global_symbol=5948..5955` and
recorded only symbols `5948..5950` before the arithmetic decoder assertion at
`global_symbol=5951`. Those three decoded symbols all matched the corrected
PR85 render-order reference (`2,2,2`). At the failing symbol, the reference
symbol is also `2`, rank `0`, with source-contract probability
`0.9667295673977898`.

Interpretation: this is no longer a broad "frame 0 fails" finding. The next
concrete blocker is an unrecovered range-coder/probability-state contract at
one specific high-confidence symbol: `frame=0`, `group=10`,
`symbol_in_group=191`, `pixel=(y=37,x=480)`. The available local CPU source
contract cannot consume the submitted PR91 token stream even when the model
strongly prefers the PR85 reference symbol.

## Verification

- `.venv/bin/python -m py_compile src/tac/pr91_hpm1_codec.py experiments/replay_pr91_hpm1_mask.py src/tac/tests/test_pr91_hpm1_codec.py`
  passed.
- `.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py::test_real_pr91_symbol_window_probe_shrinks_entropy_failure_if_available -q`
  passed: `1 passed in 8.18s`.
- `.venv/bin/python -m pytest src/tac/tests/test_pr91_hpm1_codec.py -q`
  passed: `20 passed in 107.86s`.

## Readiness

`dispatch_unlocked=false`, `pr91_ready_for_exact_eval=false`, and
`score_claim=false`. PR91/HPM1 remains external forensic signal until the exact
token-generation/probability trace is recovered or a full local decode plus
byte-exact re-encode parity gate passes.
