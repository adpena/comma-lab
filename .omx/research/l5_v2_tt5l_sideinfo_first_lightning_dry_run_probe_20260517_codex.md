# L5 v2 TT5L sideinfo first Lightning dry-run probe

**Date:** 2026-05-17  
**Status:** evidence preserved, no score claim, no provider dispatch

This ledger preserves the first launcher dry-run parse result from the L5 v2
TT5L sideinfo Lightning execution bundle. The raw state remains in the ignored
results tree; the durable signal is summarized here to avoid losing the
custody nuance.

## Source artifacts

- Bundle:
  `.omx/research/l5_v2_tt5l_sideinfo_lightning_execution_bundle_20260517_codex.json`
  - SHA-256:
    `aa871f49c4403272972e5a5c68b1f9bbd1a657be9923d3a40d39332efc6d60f1`
  - Schema: `l5_v2_tt5l_sideinfo_lightning_execution_bundle_v1`
  - Dry-run cells exposed: 10
  - Non-dry-run/provider-ready: false
- Source plan:
  `.omx/research/l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan_20260517_codex.json`
  - SHA-256:
    `9b84463441126dbfd7618a5dbb8a29b810ba3a376f29c3e42f1e4d94d1f1983f`
- Ignored raw state:
  `experiments/results/lightning_batch/l5_v2_tt5l_sideinfo_effect_curve_paired_axes/zero/contest_cpu/launcher_dry_run_state.json`
  - SHA-256:
    `b3a0eb90feb7c22372e850d0d510e53a54ee6ed794a9b75f95771db2898010a1`
  - JSON shape: `array[1]`

## Exercised cell

| field | value |
|---|---|
| variant | `zero` |
| axis | `contest_cpu` |
| lane_id | `lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cpu` |
| role | `exact_cpu_eval` |
| required_device | `cpu` |
| required_samples | `600` |
| archive SHA-256 | `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3` |
| archive bytes | `34373` |
| dry-run return code | `0` |
| stderr | empty |

## Custody nuance

The source-spec command SHA and launcher queue command SHA intentionally differ:

- Source-spec command SHA:
  `60b76e129ee3c866d761579d03e42bb0f67503c0d10995c79aca217c7fde4efd`
- Launcher queue command SHA:
  `f1bda02af81d2a27e7409f2e29db6ee8122ddbd6dcf52387d7bf15a8b4ed8ec2`

Classification: `expected_submit_layer_delta`.

Rationale: the launcher dry-run adds source-manifest, dispatch-lane-id,
adjudication, and queue metadata around the source spec command. The full
verifier should compare custody invariants instead of requiring byte-identical
command SHA equality.

## False-authority flags

- `planning_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- `provider_spend_attempted=false`

## Required follow-up

Only 1 of 10 dry-run cells was exercised here. The next artifact should be a
full dry-run verifier that executes all 10 bundle cells and records per-cell
axis, archive, lane, command, and metadata invariants before any non-dry-run
Lightning dispatch is attempted.
