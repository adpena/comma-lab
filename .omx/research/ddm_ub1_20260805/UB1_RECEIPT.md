# UB1 Receipt - scorer-free selector + #923 recovery preflight

## Answer first

UB1 ran zero scorer forwards and no `upstream/evaluate.py`. GR1 now has an opt-in
stratified selection mode with prefix preserved as the default, and the scorer-free
n48 stratified selector receipt is written at
`.omx/research/ddm_ub1_20260805/gr1_selection_stratified_n48_seed20260805.json`.

#923 recovery is partial and honest. The run-1 checkpoint carries
`pose_carrier.xi_stored + pose_carrier.dxi`, so I rebuilt runnable surfaces for
`pose_carrier_arms` and the `pose_mladder_depthwarp` A0 rung behind a shared
oracle render-cache builder. I did not recover the true-depth sidecar/cache or
texture grid/cache, so `pose_l2_truedepth` and `pose_stratified_texture` remain
blocked in this searched scope rather than being called unblocked.

## Files changed

- `experiments/ddm_gr1_granularity_rerace.py`
  - Added `--selection-mode {prefix,stratified}` and `--selection-seed`.
  - Added `--mode selection`, which emits selection/provenance without loading the
    model or scorer.
  - Changed realized d_seg calls to consume explicit pair indices, preserving
    historical prefix behavior when `--selection-mode prefix`.
  - Receipt rows now carry selector provenance and full indices.
- `experiments/ddm_ub1_pose_family_923_harness.py`
  - Added scorer-free fire-order planning.
  - Added a future scorer-slot render-cache builder that uses the canonical
    level-set byte-close oracle path (`numpy_oracle_reference_frames`).
  - Added runnable scorer commands for `pose_carrier_arms` and `pose_mladder_depthwarp --rungs A0`.
  - Refuses `pose_l2_truedepth` and `pose_stratified_texture` until their missing
    sidecars/harnesses are recovered.
- `experiments/test_ddm_ub1_recovery.py`
  - Added scorer-free regression tests for GR1 selection and #923 recovery planning.
- `.omx/research/ddm_ub1_20260805/pose_family_923_fire_orders.json`
  - Scorer-slot fire-order plan and bounded blockers.

## Selector scope proof

GR1 stratified n48:

- mode: `stratified_blocks`
- seed: `20260805`
- n/population: `48/600`
- governing quantity: `seg_flip_density`
- subset/population ratio: `1.0120588943755269`
- seeded-random null band: `[0.9409174394797062, 1.0613633626400096]`
- verdict: `MATCHED`
- scorer forwards: `0`

## #923 recall evidence

Source receipts recovered and hash-bound:

- `.omx/research/pose_l2_truedepth_probe_measured_20260708.md`
  - sha256 `39b84b28b3eb47debac9dc199e13dd260d156eebedb596d8bee60bc545d5cb12`
- `.omx/research/pose_carrier_arms_measured_20260708.md`
  - sha256 `d2ebf07e473c14be738653e32175dad72e5a1baef7bdde23a5fc5072855e8821`
- `.omx/research/pose_mladder_depthwarp_measured_20260708.md`
  - sha256 `259bf0bd94c1e154e0b70134732450e10da6582e131902188fba41a63fda0706`
- `.omx/research/pose_stratified_texture_probe_measured_20260708.md`
  - sha256 `026dbcf5aa198ec30663d14ee0a59d21dee0e3df794298db50ff85ef004cd6e8`

Bounded absence scopes checked:

- Current repo + `.omx` + `.claude` + `experiments` + `tools` + `src` + SSD root
  exact-name `rg --files` search for:
  `pose_l2_truedepth_probe.py`, `pose_mladder.py`,
  `pose_stratified_texture_probe.py`, `pose_aperture_probe.py`,
  `renders_n24.npz`, `l2_n24.json`, `l2_n8.json`, `l2_depths_n24.npz`,
  `a1t_grid_n24.json`, `scale_sweep_n24.json`, `a2_n24.jsonl`,
  `a2plus_n8.jsonl`.
- `git log --all --name-only` exact-name search for the same harness/cache names.
- `find .omx experiments /Volumes/VertigoDataTier/pact` exact artifact-name
  search for depth/texture/m-ladder sidecars.

Outcome in that scope: receipts were found; original scratch harnesses and named
sidecar outputs were not found.

## Fire order summary

Full JSON is in `.omx/research/ddm_ub1_20260805/pose_family_923_fire_orders.json`.

- `gr1_stratified_n48_selector_then_rerace`: `READY_SELECTOR_BUILT_NOT_RUN`
- `pose923_oracle_render_cache_n120`: `READY_CACHE_BUILD_NOT_RUN`
- `pose_carrier_arms_stratified_n120_retest`: `READY_REBUILT_FROM_RECEIPT_NOT_RUN`
- `pose_mladder_depthwarp_a0_stratified_n120_retest`: `PARTIAL_READY_A0_REBUILT_FROM_RECEIPT_NOT_RUN`
- `pose_l2_truedepth_stratified_n120_retest`: `BLOCKED_DEPTH_CACHE_ABSENT`
- `pose_stratified_texture_stratified_n120_retest`: `BLOCKED_TEXTURE_GRID_ABSENT`

## Tests run

- `.venv/bin/python -m pytest experiments/test_ddm_ub1_recovery.py src/tac/tests/test_subset_selection.py`
  - `45 passed`
- `.venv/bin/python -m pytest experiments/test_ddm_ub1_recovery.py`
  - `4 passed`
- `.venv/bin/python -m py_compile experiments/ddm_gr1_granularity_rerace.py experiments/ddm_ub1_pose_family_923_harness.py experiments/test_ddm_ub1_recovery.py`
  - passed
- `.venv/bin/python experiments/ddm_gr1_granularity_rerace.py --mode selection --pairs 48 --selection-mode stratified --selection-seed 20260805 --outdir .omx/research/ddm_ub1_20260805`
  - wrote `gr1_selection_stratified_n48_seed20260805.json`
- `.venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py plan --out .omx/research/ddm_ub1_20260805/pose_family_923_fire_orders.json`
  - wrote `pose_family_923_fire_orders.json`

## Next if resumed

1. If a scorer slot is available, first claim it, then build the #923 oracle
   render cache:
   `.venv/bin/python experiments/ddm_ub1_pose_family_923_harness.py build-render-cache --selection .omx/research/ddm_na3_20260805/stratified_pose_selection_923.json --checkpoint experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/levelset_witness_ema_mlx.npz --out-cache .omx/research/ddm_ub1_20260805/pose923_run1_stratified_n120_oracle_render_cache.npz`
2. Run `pose_carrier_arms` and `pose_mladder_depthwarp --rungs A0` from that cache.
3. Do not fire `pose_l2_truedepth` or `pose_stratified_texture` until their missing
   depth/texture sidecars or original scratch harnesses are recovered or rebuilt with
   exact formulation parity.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
