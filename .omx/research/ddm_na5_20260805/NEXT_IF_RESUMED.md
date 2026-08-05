# ddm_na5 Next If Resumed

## Immediate State

NA5 has a local #931 ratio receipt, fire orders, anti-pattern registration, and
an explicit attempted render-cache prerequisite. It does not have a new n120
pose-family scorer verdict and does not have sigma_eff n600.

## Resume Order

1. Re-run the render cache on the SSD tier and let it finish:

```
.venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py build-render-cache --selection .omx/research/ddm_na3_20260805/stratified_pose_selection_923.json --checkpoint experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/levelset_witness_ema_mlx.npz --out-cache /Volumes/VertigoDataTier/pact/ddm_na5_20260805/pose923_run1_stratified_n120_oracle_render_cache.npz
```

2. Only after the cache exists, claim the local frozen-scorer slot and run:

```
.venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py score --selection .omx/research/ddm_na3_20260805/stratified_pose_selection_923.json --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz --render-cache /Volumes/VertigoDataTier/pact/ddm_na5_20260805/pose923_run1_stratified_n120_oracle_render_cache.npz --item pose_carrier_arms --out .omx/research/ddm_na5_20260805/pose_carrier_arms_stratified_n120_retest.json
```

3. Then run the rebuilt A0 depthwarp surface:

```
.venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py score --selection .omx/research/ddm_na3_20260805/stratified_pose_selection_923.json --gt-cache /Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz --render-cache /Volumes/VertigoDataTier/pact/ddm_na5_20260805/pose923_run1_stratified_n120_oracle_render_cache.npz --item pose_mladder_depthwarp --rungs A0 --out .omx/research/ddm_na5_20260805/pose_mladder_depthwarp_a0_stratified_n120_retest.json
```

4. Do not rerun `pose_l2_truedepth` or `pose_stratified_texture` until the
missing sidecars/harnesses are recovered. Current blockers are recorded in
`pose_family_rerun_status_923.json`.

5. For sigma_eff, claim the full n600 scorer slot first, then run:

```
.venv/bin/python experiments/probe_lever_d_selective_fullstack.py --state-ckpt experiments/results/torch_vehicle_full_mps_basin_bc20_n600/torch_vehicle_checkpoint_state.pt --video upstream/videos/0.mkv --which ema --n-pairs 600 --tau 0.5 --batch 6 --targets-cache /Volumes/VertigoDataTier/pact/ddm_na5_20260805/lever_d_selective_targets_n600 --out-json /Volumes/VertigoDataTier/pact/ddm_na5_20260805/lever_d_selective_probe_n600_sigma_eff.json
```

## Boundaries

- No contest score can be claimed from these rows.
- No route-changing pose-family verdict exists until a scored n>=120 result is
  present.
- Keep outputs on `/Volumes/VertigoDataTier/pact/ddm_na5_20260805` for bulky
  generated caches.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
