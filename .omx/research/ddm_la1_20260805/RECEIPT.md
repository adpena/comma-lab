# ddm_la1 2026-08-05 Receipt

## Scope

Arm: `ddm_la1` terminal JD1 LR-anneal lever for the TR1 joint-descent line.

This receipt is build/preregistration only. No launcher, scorer, eval, archive, or long run was
started. `tr1_jd4_cont_ep1646` / jd6 was read only; no live run directory was edited.

Pointer status: own-vehicle frontier remains
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; borrowed contest pointer unchanged.

## Recall

Reviewed before implementation:

- Witness-line Muon anneal laws: `muon_finisher_schedule_warmstart_and_lr_anneal_v1` and
  `muon_switch_conditioning_criterion_v1` establish the pattern of warm-started terminal
  anneal, but their `--muon-lr-final-frac 0.1` is vehicle-scoped and was not transferred.
- #518 / beta2 warmup geometry: optimizer variance memory uses the `c/(1-beta2)` shape; this
  lever uses `c=2` converted to epochs at the live batch geometry.
- Cross-regime constant discipline: no inherited scalar is treated as a TR1 constant. The only
  default final fraction is derived from parent telemetry for the actual boundary window.

## Derivation

Source telemetry:
`/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/tr1_jd4_cont_ep1526/telemetry.jsonl`

Endpoint mechanism evidence:
`/Volumes/VertigoDataTier/pact/ddm_jd4_20260805/jd5_endpoint_n600_both_bases.json`

Measured endpoint deltas, `[macOS-CPU frozen-scorer advisory]`, n600 both bases:

- Live basis: `delta d_pose = +0.048132309262887904`
- EMA basis: `delta d_pose = -0.054913158711088456`
- Live-vs-EMA endpoint d_pose gap: `0.10388646797397635`

The parent telemetry did not contain a pose-itemized `loss_terms.terms.*pose*` value. It did
contain active JD1 epoch rows and later `loss_terms` rows with `seg`, `rate`, and
`delta_sparsity`, so the implemented derivation labels the fallback source as
`epoch.ep_loss[jd1_pose_finish_active]`.

Inputs:

- Boundary: `start_epoch=1526`, `end_epoch=1646`
- `steps_per_epoch=150`
- `beta2=0.999`
- Active EMA decay: `0.9997777777777778`
- Base LR: `0.002`

Derived time constants:

- beta2 memory: `ceil(2/(1-0.999)/150) = 14` epochs
- active EMA memory: `ceil(2/(1-0.9997777777777778)/150) = 60` epochs
- tail length: `max(14, 60) = 60` epochs
- onset epoch for a 1646 boundary: `1586`

Measured last-60 active JD1 epoch-loss oscillation:

- `n=60`
- mean `0.8992388238112132`
- sd `0.011274632999681105`
- half_range `0.032927233179410265`
- relative half_range `0.03661678333666233`
- sign changes `38/58`

Derived default final fraction:

`final_frac = sd / (sd + half_range) = 0.2550714251294281`

Thus for `--lr 0.002`, final LR is:

`0.002 * 0.2550714251294281 = 0.0005101428502588562`

## Edits

- `experiments/train_tr1_partition_renderer_mlx.py`
  - Added `--jd1-lr-anneal {off,derived_tail}` default `off`.
  - Added `--jd1-lr-final-frac` default `0.0`, meaning derive.
  - Added pure parent-telemetry derivation helpers.
  - Added validation to refuse inert or unresumable LR-anneal shapes.
  - Wired ON-only schedule resolution at JD1 resume/engagement and ON-only
    `optimizer.learning_rate` assignment per epoch.
  - Added ON-only telemetry fields; OFF writes no schedule row and adds no `TR1Config` field.
- `src/tac/witness_dsl/spec_tr1_renderer_20260728.py`
  - Added `lever_jd1_lr_anneal()`, active-shape only, with runtime receipt schema and falsifier.
- `src/tac/tests/test_ddm_bp1_boundary_reset_race.py`
  - Added parser fail-closed tests, pure derivation test, and AST check that main reaches
    `optimizer.learning_rate`.
- `src/tac/witness_dsl/tests/test_jd1_joint_pose_finish_lever.py`
  - Added parser/DSL tests for the new lever.

## Default-Off Proof

Absent flags parse as:

- `jd1_lr_anneal == "off"`
- `jd1_lr_final_frac == 0.0`

OFF does not enter `TR1Config`, so config hashes and checkpoint metadata are not extended by
this arm. The derivation helper is only called from `_resolve_jd1_lr_anneal_schedule()` when
`args.jd1_lr_anneal == "derived_tail"`. Epoch rows are extended only when
`jd1_lr_schedule is not None`. The optimizer LR setter is also guarded by the active schedule.

## Preregistered Boundary A/B

At the jd7-or-Case-B boundary, run matched ON vs OFF from the same checkpoint:

- OFF: existing flat LR, no `--jd1-lr-anneal`.
- ON: `--jd1-lr-anneal derived_tail`, omit `--jd1-lr-final-frac` unless MAIN pre-supplies an
  explicit boundary override.

Both arms must use the same checkpoint, seed, epoch window, `batch_pairs`, EMA mode, and scorer
endpoint protocol. Measure n600 endpoint deltas on both live and EMA bases.

Prediction: live/EMA endpoint divergence closes if terminal LR oscillation is the cause.

Falsifier: ON endpoint EMA is no better and live divergence is unchanged; classify as not
LR-driven.

## Verification

- `python -m py_compile` on the trainer, DSL, and two test files: PASS.
- Focused pytest selection: `6 passed in 0.51s`.
  - The sandbox emitted the known MLX atexit `No Metal device available` warning after pytest
    completion; tests still exited 0.
- `tools/review_tracker.py scan`: PASS, then mark-file pass 1 on all four Python files.
- `tools/review_tracker.py scan`: PASS with `New: 0`, `Changed (stale): 0`, then mark-file
  pass 2.
- Review query after pass 2:
  - trainer: `117 reviewed`
  - bp1 tests: `50 reviewed`
  - TR1 DSL: `45 reviewed`
  - JD1 DSL tests: `12 reviewed`

No score claim. No scorer slot consumed. No live jd6 modification.
