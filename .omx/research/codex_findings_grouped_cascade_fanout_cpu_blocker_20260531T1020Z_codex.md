# Codex Findings: Grouped P18/P19/P11/P15 Fanout CPU Blocker

- generated_at_utc: 2026-05-31T10:20:00Z
- author: Codex
- scope: grouped scorer-region cascade fanout, receiver-closed distortion budget, MLX/CPU gate calibration
- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false

## What Landed

The grouped cascade campaign was executed through `experiment_queue.v1` against
the current CPU-frontier archive. The bounded fanout worker started 80 steps
across 6 variants and completed all 80 with no failures. The queue proved real
receiver output changes through `inflate.sh`, ran local CPU scorer gates, built
MLX response rows where configured, and emitted exact-ready bridge blockers.

Live report:

`./.omx/research/scorer_region_selector_cascade_campaign_20260531T023734Z/fanout_campaign_report_20260531T1018Z.json`

## Empirical Result

Observed local CPU rows: 5.

Best local CPU row:

- variant: `nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch`
- local CPU score: `0.1920003362662307`
- CPU-frontier pointer used by the eureka gate: `0.19198533626623068`
- delta: `+0.000015000000000015001`

All observed local CPU variants failed the CPU eureka gate. The receiver patches
are real, but this operator family currently spends SegNet distortion without
buying enough rate/composition improvement.

The MLX acquisition rows were optimistic in all observed cases:
`mlx_positive_full_cpu_negative_split_count == 5`. This is now encoded in the
campaign report's `aggregate_learning` block, with:

- `recommended_next_queue_policy`: `acquisition_first_or_cpu_gate_only_no_post_cpu_mlx`
- `posterior_routing_decision`: `demote_post_cpu_mlx_for_current_operator_family_until_acquisition_model_changes`

## Engineering Fix

The campaign report CLI no longer requires manual variant-root enumeration.
It now accepts `--variant-root-dir`, discovers immediate child variant roots,
and harvests a directory-owned fanout report. This closes a manual/orphan-prone
operator step.

Follow-on acquisition policy landed:

`./.omx/research/scorer_region_selector_cascade_campaign_20260531T023734Z/acquisition_policy_20260531T1036Z.json`

The policy consumes the fanout report, the MLX full-600 per-pair master-gradient
tensor, and the UNIWARD per-pixel scorer-gradient cache. It emits the next
queue mode as `vectorized_mlx_acquisition_then_cpu_gate_only`, preserves the
master-gradient and per-pixel priors as advisory only, and records the current
blockers: exact auth required, all observed CPU rows failed, MLX/CPU split,
archive-specific master-gradient anchor missing for this campaign, and the
pixel-gradient cache covering only a partial sample.

## Current Blocker

The P18/P19/P11/P15 receiver-closed cascade is not exact-auth ready from the
current observed family. The precise blocker is:

`local_cpu_score_not_below_auth_frontier`

This is not a receiver-patch premise failure. It is a distortion-budget economics
failure under the current RGB/YUV proxy deltas and grouped region/pair settings.

## Next 12-Week Tranche

The next huge sprint should pivot from post-CPU MLX rows to acquisition-first
candidate construction:

1. Build an MLX-first acquisition model that predicts CPU scorer deltas from
   P18/P19 features before full CPU inflate/evaluate.
2. Generate new grouped operator families that change the sign of the SegNet
   spend: smaller deltas, native YUV patching, boundary-aware masks, and
   per-region selector coding that actually saves rate.
3. Run CPU-gate-only queues for this current family unless the acquisition model
   changes materially.
4. Dispatch exact CPU auth only after a local CPU eureka pass; CUDA remains an
   anchor after CPU clears.
5. Keep PR95/HNeRV MLX as the substrate-training control arm, but route score
   lowering through receiver-closed candidates that beat the CPU gate first.
