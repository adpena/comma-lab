# Codex Findings - Modal Harvest Recovery - 2026-05-22T04:51:09Z

## Scope

Operator request: harvest any ready Modal results before provider retention
scrubs them.

Canonical surfaces checked:

- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/modal_call_id_ledger.jsonl`
- `tools/recover_modal_auth_eval.py`
- `tools/harvest_modal_calls.py`

## Recovered exact auth eval

HFV9 magic explicit-row sidecar CPU exact eval was recovered through
`tools/recover_modal_auth_eval.py`.

- call_id: `fc-01KS63YEGTKQ5EQFKC2FRGRGMJ`
- output_dir: `experiments/results/modal_auth_eval_cpu/hfv9_magic_explicit_row_pr101_9a32b1311da1_cpu`
- evidence_grade: `contest-CPU`
- score_axis: `contest_cpu`
- score_recomputed_from_components: `0.32067828057415293`
- avg_posenet_dist: `0.00176691`
- avg_segnet_dist: `0.00068862`
- archive_sha256: `9a32b1311da1076b1659ff6652481383527279905c8a135eeedff6ec913888ac`
- archive_bytes: `178553`
- returncode: `0`
- score_claim: `true`
- terminal claim: `completed_contest_cpu_modal_auth_eval_recovered`

The CUDA sibling was already present:

- output_dir: `experiments/results/modal_auth_eval/hfv9_magic_explicit_row_pr101_9a32b1311da1_cuda`
- evidence_grade: `contest-CUDA`
- score_axis: `contest_cuda`
- score_recomputed_from_components: `0.33713201858942626`
- avg_posenet_dist: `0.00194092`
- avg_segnet_dist: `0.00078924`

## Recovered training / infrastructure calls

`tools/harvest_modal_calls.py --execute` recovered terminal state for all
remaining filesystem-discovered pending Modal calls and rebuilt the generated
summary surface.

Active A100 selfcomp / grayscale LUT:

- call_id: `fc-01KS5YG9W26T72D6Z8Y3N44JEN`
- lane_id: `lane_overnight_xx_selfcomp_tier_2_paid_modal_a100_first_anchor_dispatch_20260521`
- job: `substrate_grayscale_lut_lut_bits_5_modal_a100_dispatch_20260521T185859Z`
- rc: `124`
- timed_out: `true`
- elapsed_seconds: `14400.841477592001`
- recovered artifacts: `8`
- recovered checkpoint: `lane_substrate_grayscale_lut_results/output/best.pt` (`29906874` bytes)
- terminal claim: `failed_modal_training_timeout`
- score_claim: `false`
- promotion_eligible: `false`

Additional pending filesystem-discovered calls were also terminalized:

- NSCS06 v8 `20260521T155951Z`: `failed_modal_training_rc_22`, elapsed `2.1043871859999967`
- NSCS06 v8 `20260521T170122Z`: `failed_modal_training_rc_22`, elapsed `1.722390666999999`
- NSCS06 v8 `20260521T184408Z`: `failed_modal_training_rc_1`, elapsed `6.913175322000001`
- DP1 original baseline `20260521T031607Z`: `failed_modal_training_timeout`, elapsed `5401.02630825`
- DP1 original baseline `20260521T062845Z`: `failed_modal_training_timeout`, elapsed `3600.579290009`
- DP1 procedural `20260521T031839Z`: `failed_modal_training_timeout`, elapsed `5400.327533146`
- DP1 procedural `20260521T062934Z`: `failed_modal_training_timeout`, elapsed `3600.439569727`

These rows are training/infrastructure custody evidence only. They are not
score authority and remain `score_claim=false`.

## Post-harvest state

- `tools/claim_lane_dispatch.py summary --ttl-hours 24`: `active=0`; only stale
  nonterminal is the unrelated local master-gradient phase-1 row.
- `tools/harvest_modal_calls.py --from-ledger`: `unharvested call_ids: 0`.
- Generated summary repaired at `experiments/results/_modal_harvest_summary.json`.

## Follow-up

Review the A100 grayscale LUT `best.pt` and run log for whether the timeout
left a usable intermediate candidate. Any archive or promotion path must go
through exact result review and contest-axis auth eval; the recovered training
row itself is not rank or kill authority.
