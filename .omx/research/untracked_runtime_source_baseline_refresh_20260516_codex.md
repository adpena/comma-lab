# Untracked Runtime Source Baseline Refresh

- timestamp_utc: `2026-05-16T22:41:24Z`
- scope: all-lanes Gate #10 no-signal-loss custody for `experiments/results/`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- dispatch_attempted: `false`

## Finding

`tools/all_lanes_preflight.py --timings` failed only Gate #10. The standalone audit showed
`undispositioned_count=0` and `invalid_disposition_count=0`, but the
`experiments/results/` runtime-source baseline was stale:

- previous count: `11171`
- previous sha256: `dfa6409b0015804f3e62f8ccf89f5ca808bb0cea9580bc1f0f68469f0dcf8a68`
- current count: `14212`
- current sha256: `3da990f294be3384274aba473c3dbb826e89055d65b4920ad26582b5d5aeaa95`

## Resolution

Refreshed `.omx/research/untracked_source_dispositions_20260505_codex.json`
for the `experiments/results/` prefix baseline. The disposition remains
`ignore_rebuildable`: raw runtime packet/source copies under
`experiments/results/` are custody state, not promoted source of truth. Durable
signal must still be summarized into tracked `.omx/research` ledgers, code,
tests, or compact manifests before promotion.

## Non-Claims

This is not a score result, not a lane promotion, and not exact-eval authority.
It only restores the no-signal-loss guard's ability to distinguish known raw
custody growth from undispositioned source drift before dispatch.
