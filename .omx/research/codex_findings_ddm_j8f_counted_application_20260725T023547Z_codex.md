# Codex findings — DM4 → J5 counted application operator

UTC: 2026-07-25T02:35:47Z
Lane: `lane_ddm_j8f_counted_application_20260724`
Delegation checkpoint: `codex_delegate:ddm_j8f_counted_application_operator:20260724T181414Z`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
Score claim: `false`
Pointer: `0.1910828242 [contest-CPU]` — **UNMOVED**

## Disposition

`READY_TO_FIRE_DDM_EVENT_CONTINUATION`, subject to independent MAIN landing
review. This lane did **not** FIRE, dispatch, promote, or move the frontier
pointer. The emitted ticket binds `fire_authority=MAIN_ONLY_AFTER_REVIEW` and
`execution_allowed=false`.

## What landed

- Typed DM4 proposal-descriptor → exact integer J5 receiver-coordinate
  application operator.
- Exact `+1/-1` receiver-secant enumeration with sparse, SHA-bound per-pair
  inventories.
- MS4d minimum-norm `-pinv(H)g` proposal ordering. No global learning rate or
  hand-tuned damping is present.
- Every proposal is projected through the #580 `range(A)` / gauge projector,
  then mapped to the nearest unused exact integer J5 secant.
- Twelve-step, one-quantum-per-coordinate, no-reuse smoke horizon with an
  atomic cumulative checkpoint after every stage.
- Exact archive parse-back and frozen-scorer n600 A/B for raw and
  range/gauge-projected arms.

The sealed conservative gaps remain explicit:

- `J5_BUCKET_VALIDITY_RADIUS_CURVE_ABSENT_NO_SHRINK_GROW_TRANSFER`
- `J5_NCDE_REENTRY_TIME_CUSTODY_ABSENT_USING_CANONICAL_WINDOW_12`

They scope this result to the bounded 12-step smoke; they do not authorize a
learned trust-radius transfer or a longer continuation horizon.

## Exact measured result

Reference Step 4:

- archive SHA-256:
  `9601e777010b1dc45ed0841e118fcf34c58452324f8730fe9958a3440502e3a4`
- bytes: `138804`
- `d_seg=0.0702156745062934`
- `d_pose=36.37587755493872`

Raw application arm:

- archive SHA-256:
  `b56380f73c3e7f46235eae82965fb4372cb7d5f93df0d81fa9844f5aed742c49`
- bytes: `138804`
- `d_seg=0.07043056064181857`
- `d_pose=36.31873163697579`
- exact joint delta versus Step 4: `+0.006501460375847978`

Range/gauge-projected arm:

- archive SHA-256:
  `0c9cb2235292eb4fe109e88847ec6dbd1f9b23e9d5fd1e6f01a48fa2c9151977`
- bytes: `138804`
- `d_seg=0.07026273939344618`
- `d_pose=36.15701340893894`
- Seg term versus Step 4: `+0.004706488715278123`
- Pose term versus Step 4: `-0.05746357277241998`
- Rate term versus Step 4: `0.0`
- exact joint delta versus Step 4: `-0.052757084057141856`
- exact joint delta versus raw: `-0.059258544432989835`

Therefore the projected arm satisfies both preregistered gates:

1. exact projected joint delta versus Step 4 is strictly negative;
2. exact projected joint delta is unchanged-or-better than raw.

The twelve measured rejected null/gauge energy fractions were:
`[0.6447443697758621, 0.6944055600935847, 0.6867504995023084,
0.5481433103801491, 0.5903146942668522, 0.6003331306205649,
0.5825373351255735, 0.5782064190068217, 0.5782064190068217,
0.572890918183736, 0.523330670169047, 0.5251753804416491]`.

All 12 raw coordinates and all 12 projected coordinates were unique; coordinate
reuse was disabled. All 12 cumulative stage checkpoints were preserved.

## Durable custody

Output root:
`/Volumes/VertigoDataTier/pact/experiments/results/ddm_j8f_counted_application_20260724T181414Z`

- Final receipt:
  `ddm_j8f_counted_application_receipt.json`, bytes `138748`, SHA-256
  `aa1a51dd34017d26b685966779b17139d780328f7cfcc294da94f78c1cc500f3`
- READY ticket:
  `READY_TO_FIRE.ticket.json`, SHA-256
  `3b3e503f0f32f888c606f814dfa8cf7ea1e69c99cf6648ea0c03f300ea0a8b3e`
- Raw n600 verdict SHA-256:
  `1c71137011e22f97bc129b82f7313aba77ead57b8761fae1fa4961a82ce81112`
- Projected n600 verdict SHA-256:
  `c11820455267546430a81ac041726838baaf41d1a8306bb18ac78ddd7384c25e`
- Final typed-config hash:
  `563748abc2c86298bcef11bef32b8e468377aa574e2a4dd82fff12de4b423377`
- Final preflight SHA-256:
  `64ab2ea272e2b6539285acb0c4573f7decf86c222c287fe60161c937488f5b33`
- Deterministic algorithms: `true`
- Observed Torch threads: `4`

Exact preserved inventory manifest SHAs used by the smoke:

- pair 16:
  `b0579d535a54d25cc97ba31b6989058c741981662787e5e8d2e943fa8af45105`
- pair 60:
  `354ed66b5003eba1deb8ef7ee0ecc4ec0cda09964e1ac210d9875aef06b6a87b`
- pair 90:
  `f4391656e197275b8336d4c0cf1192018f33a4ad78f0ce49286722e5f499a40d`

The completed-receipt loader revalidated archives, verdicts, deltas, decision,
and READY ticket without recomputation.

## Adversarial review and fixes

Three post-fix clean passes were recorded:

1. all six real J8e proposal pair/bucket joins resolved uniquely against the
   SHA-bound MS4d Seg-custody artifact;
2. 63 adjacent operator/adapter/joint-descent/#580 tests passed, with ruff,
   `py_compile`, and `git diff --check` clean;
3. immutable preflight reloaded and SHA-validated all three exact pair
   inventories without recomputation.

Earlier review passes found and fixed:

- discarded pair inventories on interruption;
- mutable/recomputed preflight;
- missing cumulative clipping/parse-back and resume-chain validation;
- stale memory/storage admission and unverified Torch thread count;
- completed-result reruns that did not revalidate immutable receipts;
- a config binding to the composite MS4d schema instead of the required Seg
  custody schema.

The first scoring attempt also failed closed because the isolated worktree cwd
did not contain the inherited repo-relative 5,078,017,610-byte GT cache.
The same 12 stage checkpoints were resumed from the canonical repository cwd
against the ticket-bound cache SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
No stage was recomputed and no custody field was coerced.

## STORES CONSULTED

- Delegated authority file, SHA-256
  `8b86a8f6e2016b10ab70053e5d5ee5444f9e3bfb5001a169bbd5a6cbe8803d9d`
- `CLAUDE.md`, `AGENTS.md`, top-10 Claude memory entries
- current lane registry, lane maturity audit, and subagent progress ledger
- Step-4 typed ticket, checkpoint, and exact n600 reference verdict
- J8e proposal-source ticket and measured memory receipt
- DM4 typed config and realization receipt
- MS4d direct Seg metric custody artifact
- v17 validity law/receipt, NCDE observer/event wiring, and #580 projector
- latest Codex findings/session summaries and latest Claude design/council
  surfaces required by preflight

## MAIN review required

MAIN must independently review:

1. the serializer commit and changed-file scope;
2. the six source bindings and typed config;
3. all 12 stage receipts and no-reuse proof;
4. both exact archive SHAs and n600 verdict SHAs;
5. the negative projected joint delta and projected-better-than-raw gate;
6. the two named validity/horizon gaps before deciding whether to FIRE.

This memo is advisory and historical provenance. It does not promote the
candidate or claim a contest-CPU/CUDA score.
