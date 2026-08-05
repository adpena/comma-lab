# ddm_na3 2026-08-05 receipt

## Verdict

NA3 did not produce an upheld/overturned/weakened pose-family verdict because
the four source formulations still depend on unlanded scratch harnesses/render
caches. It did reproduce the #931 pose-prefix ratio law, registered the two
missing canonical anti-pattern classes, staged the full n600 sigma_eff command
without running the scorer slot, and left a population-matched n120
stratified-block selection receipt for the recovered pose reruns.

## Measured here

- #931 re-derived from
  `/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d2/d2_ep_solve.partial.jsonl`
  (`sha256 d2853c92090c28ebe558ece4a21b2847b55e25c9d768bef167bcba9dc67b72e5`,
  600 rows, field `d_pose_shipped_f16`).
- Population mean: `0.15950891917937635`.
- Prefix ratios: n24 `2.535475579649216`, n48 `2.640181689154513`,
  n64 `2.6477688499984713`, n96 `4.206770932037034`.
- 60-pair block profile: hardest/easiest `79.43661398538532`; first120/easiest
  `59.49651372432368`.
- NA3 retest selection: `stratified_blocks`, n120/600, seed `20260805`,
  10 contiguous blocks, governing ratio `1.0057539935665503`, random null band
  `[0.5357086950651611, 1.5802082483144484]`, verdict `MATCHED`.

All numbers in this memo are `[macOS-CPU frozen-scorer advisory]` and are not
contest score claims.

## Registered

The live canonical anti-pattern registry was appended through the locked writer
with these IDs:

- `prefix_bias_sign_inversion_pose_axis_v1`
- `subset_default_silent_under_sampling_v1`

Latest registry payloads are in `.omx/state/canonical_anti_patterns_registry.jsonl`.
The builder source is `src/tac/canonical_anti_patterns/na3_subset_bias_builders.py`.

## Staged only

The Lever-D sigma_eff n600 command is staged in
`.omx/research/ddm_na3_20260805/sigma_eff_n600_stage.json`. It was not run
because the n600 scorer slot is not NA3's slot. The staged output paths are on
`/Volumes/VertigoDataTier/pact/ddm_na3_20260805/`.

The four pose-family reruns are staged as fire orders in
`.omx/research/ddm_na3_20260805/pose_family_rerun_status_923.jsonl`. Each row
names `OD1 Stage-2 pose-recovery adjudication` as the consumer and has
`upheld_overturned_weakened: null` because the rerun did not execute.

## RECALL EVIDENCE

Queries and sources consulted:

- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Charter files: `.omx/tmp/codex_runs/na3_prompt.md`,
  `.omx/tmp/codex_runs/_common_contract.md`.
- Seed audit: `.omx/research/ddm_na2_negative_audit_20260803.md`.
- Prior prep beyond the seed: `.omx/research/ddm_ng1_20260805/ng1_negative_results_audit.md`
  and `.omx/research/ddm_ng1_20260805/pose_rerun_fire_orders_923.jsonl`.
- Four pose source receipts:
  `.omx/research/pose_l2_truedepth_probe_measured_20260708.md`,
  `.omx/research/pose_carrier_arms_measured_20260708.md`,
  `.omx/research/pose_mladder_depthwarp_measured_20260708.md`,
  `.omx/research/pose_stratified_texture_probe_measured_20260708.md`.
- Exact search terms included `pose_l2_truedepth_probe`,
  `pose_carrier_arms`, `pose_mladder`, `pose_stratified_texture_probe`,
  `MEASURED DEAD`, `2.535475`, `4.206770`, `prefix bias`,
  `inflate-out`, `a16aa984d0`, and `sigma_eff`.

Findings beyond the charter seeds:

- NG1 had already staged #923 fire orders and the n600 sigma_eff command, but
  explicitly did not run either class of scorer work.
- NG1's fire orders identified the same blocker NA3 confirms: the four source
  receipts name scratch harnesses or render caches that are not landed files in
  the bounded repo index.
- The reachable quote surfaces checked in this pass already reflect the weak
  evidence correction; no additional quote edit was required in the checked
  scope.
- `a16aa984d0` is an exact-eval inflate-output contract for
  `experiments/ddm_fz2_byteclose_and_eval.py`; the staged sigma_eff probe has no
  `--inflate-out` argument, so the contract is recorded for downstream
  composition rather than applied to this read-only command.

What changed because of recall:

- NA3 did not duplicate NG1's broad missing-harness search or fire unsafe
  scorer work.
- NA3 added a fresh stratified n120 selection receipt with governing pose ratio
  proof, registered the missing anti-pattern classes, and left the remaining
  work as bounded fire orders.

## NEXT-IF-RESUMED

1. Recover or re-land the four scratch harnesses/render caches named in
   `pose_family_rerun_status_923.jsonl`.
2. Run each formulation on
   `.omx/research/ddm_na3_20260805/stratified_pose_selection_923.json` and fill
   `upheld`, `overturned`, or `weakened` for OD1 Stage-2.
3. Only after the pe2/sq2 scorer slot clears, run the staged n600 sigma_eff
   command from `sigma_eff_n600_stage.json`.
4. If a rerun becomes exact-eval-composed through `ddm_fz2_byteclose_and_eval.py`,
   set `--inflate-out <sub-dir>/inflated` unless `--skip-eval` is used.

## Frontier

Own-vehicle frontier unchanged: `S = 0.7539807296911207 @ 357,836 B
[macOS-CPU advisory]`; contest pointer remains borrowed/unmoved at
`0.1910828242`.
