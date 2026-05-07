# PR91/HPM1 Tile-Major Failure-Row Probability Scan - 2026-05-07

Scope: local CPU forensic probe only. `score_claim=false`,
`dispatch_allowed=false`, and HPM1 remains blocked until full decode/reencode
parity plus exact runtime closure.

Input:

- Archive: `experiments/results/public_pr91_intake_20260504_codex/archive.zip`
- Archive SHA-256: `4c16d04c746c981feb902e4dd508ffadaf3615e532d351993c3d2f6eccda1b4f`
- Prior best local HPM1 semantic prefix: `tile_major_row_major`
- Prior failure: frame `0`, group `12`, symbol-in-group `210`,
  decoded-before `8274`
- Prior suffix scan: `1134` remaining same-group rows tested, `0` decodable

New probe:

- Tool: `tools/audit_pr91_hpm1_failure_row_probability_scan_probe.py`
- Artifact: `.omx/research/pr91_hpm1_failure_row_probability_scan_tile_major_20260507_codex.json`
- Replayed to the same tile-major failure row, cloned the `RangeDecoder` state
  immediately before the failed symbol, and scanned:
  - HPAC variants: `source_float64_perfect_false`,
    `source_float32_perfect_false`, `source_float64_perfect_true`,
    `source_float32_perfect_true`
  - `prob_eps`: `1e-12`, `1e-9`, `1e-7`, `1e-6`, `1e-5`, `1e-4`, `1e-3`,
    `1e-2`
  - uniform mix masses: `0`, `1e-6`, `1e-5`, `1e-4`, `1e-3`, `1e-2`

Result:

- `192` candidate categorical rows tested.
- `0` rows decoded from the cloned failure state.
- Baseline source variant still does not decode.
- Failure row normalized probability SHA-256 remained
  `8216c3d82263ef0fc10c88ddf28439b0916ae83865c8d14d9e37bd785bd2b7cd`.
- Classification:
  `failure_row_not_decodable_under_probability_numeric_scan`.

Blocker movement:

- Narrowed false lead: the tile-major failure is not explained by a simple
  failure-row `prob_eps`, float32/float64, `perfect` flag, or bounded uniform
  smoothing difference.
- Remaining plausible classes move earlier than the failed row or outside the
  modeled categorical surface:
  - earlier context/order drift before frame `0`, group `12`, symbol `210`
  - range-coder construction/finalization contract drift
  - true PR91 encoder semantic tokens differing from public runtime semantics
  - unmodeled encoder-side probability numeric path outside this bounded scan
  - byte-exact full-stream reencode

Grand-council stop rule:

- `material_unlock_found=false`.
- Recommendation: stop HPM1 wall-clock unless a real encoder trace/source
  appears; redirect frontier time to other replacement or categorical
  candidates with clearer exact-evaluable paths.
