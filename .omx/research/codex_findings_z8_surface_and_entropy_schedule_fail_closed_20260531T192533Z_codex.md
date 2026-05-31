# Codex Findings: Z8 Surface And Entropy Schedule Fail-Closed

`[macOS-CPU advisory]` `score_claim=false` `promotion_eligible=false`

## Trigger

Sidecar adversarial review found that the live Z8 full-video VJP bundle and per-subband entropy schedule could be read too strongly:

- the bundle surface contained non-finite `joint_weight` entries while its manifest carried gradient/budget authority;
- the per-subband delta schedule was derived from `6/600` pairs but marked materializer-ready;
- rate-only mutated search rows could be accepted without full-video replay or receiver-MSE proxy.

## Code Fixes

- `joint_coefficient_waterfill._normalize_surface_array` now rejects non-finite P18/P19 surfaces before any coefficient allocator work.
- `full_video_vjp_acquisition` now rejects non-finite shard/bundle arrays before assembly or NPZ writing.
- relinearized search now blocks mutating rate-only candidates unless receiver proxy or full-video replay ran; storage-only no-mutation probes remain allowed and explicitly labelled.
- `entropy_delta_schedule` now records source/report hashes and fails closed on partial headroom coverage unless `--allow-partial-coverage` is explicit.
- materializer/search CLIs now share the same fail-closed schedule loader.

## Artifact Supersession

The prior live bundle at `.omx/research/z8_full_video_mlx_vjp_live_20260531T181115Z/mlx_vjp/iteration_0000/bundle/z8_full_video_vjp_surface_bundle.npz` is superseded as `joint_p18_p19_surface_nonfinite` until regenerated. Current guard output:

```text
ValueError joint_weight contains non-finite values (14082048/44236800)
```

The sampled schedule at `.omx/research/z8_full_video_mlx_vjp_live_20260531T181115Z/per_subband_delta_schedule_codex/schedule_max_subband_mse_2e-5.json` is regenerated with:

```json
{"ready_for_materializer": false, "blockers": ["partial_headroom_coverage:6/600"]}
```

The explicit sampled/advisory variant remains separate as `schedule_max_subband_mse_2e-5_sampled_allow_partial.json`.

## Remaining Blocker

The current MLX P19 surface is still a scalar pose-loss gradient surface, not a true six-axis PoseNet Jacobian/null-subset Mahalanobis surface. It is useful as advisory ranking signal only. A regenerated authority bundle must either provide the real per-axis P19 surface or carry an explicit blocker instead of `budget_spend_authority=true`.
