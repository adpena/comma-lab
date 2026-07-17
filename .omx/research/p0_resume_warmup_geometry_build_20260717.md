# p0_resume_warmup_geometry (#518) — BUILD memo (2026-07-17)

Ledger row: `p0_resume_warmup_geometry_20260717` · branch
`claude/p0_518_resume_warmup_geometry_20260717` (isolated worktree; merges at the post-c2
boundary, NOT by this agent). Pointer 0.19108 UNMOVED — everything here is MEANS (boundary
apparatus for the c2 lineage), no score claim.

## STORES CONSULTED (recall-before-design)

- `tools/graph_memory_recall.py "resume warm start LR rewarmup"` → surfaced the ALREADY-SETTLED
  sister law `rewarmup_beta2_memory_window_v1` (rewarmup_steps ≥ 1/(1−β₂); callable
  `curriculum_derivation_laws_20260705:min_rewarmup_epochs`; PROVISIONAL, "cert8_satisfies:
  false" — the config 8 ep is UNDER the bound) + `warm_start_schedule_reconstruction_v1` +
  `muon_finisher_schedule_warmstart_and_lr_anneal_v1`. The new law was built AS A GENERALIZATION
  (c=1 reproduces the sister bound bit-exactly — tested), not a re-derivation.
- `.omx/state/canonical_equations_registry.jsonl` (both laws read in full before writing).
- Trainer sources read at the exact cited lines: `_stage_rewarmup_factor` (~3709), resume block
  (~9848-10180), baseline_v0 block (~10180-10270), lambda_pre probe mirror (~11370), pose engage
  (~11975), stage/tau-octave boundary registrations (12250/12337), LR consumption (12460-12490),
  clip/update site (12720-13010), `_maybe_preserve_best`/`_emit_verdict_row` (9290/9445),
  `_build_resume_state_arrays` (1024), hardness/RNG restore (10360-10410).
- DSL: `curriculum_dsl.Lever` + `HeadOffsetSolver`/`MarginBandSatisficing` factory patterns;
  `lawref.py`/`lawref_builtins.py`/`evaluators.py` (LawRef evaluator registration);
  `lever_registry.completeness()`; `spec_c2_surgical_20260716` (sealed constants; NOT touched).
- Launcher: `parse_dry_start_run_metrics` / `dry_start_boot_ok` / `dry_start_resume_ok`.
- `activation_ledger.py` (duty-to-measure is automatic: new levers are never-fired until a run
  consumes them — no manual registry needed).

## Commits (this branch)

1. `f53d40b0d3` — trainer items 1, 3, 4, 5a(#517), 6a, 6b, 7, 8-partial, 9 + six DSL Lever
   factories + `adam_v_variance_warmup_length_v1` (module + evaluator + registry row).
2. `c0de210d35` — launcher item 5b (fork-verdict receipt fields).
3. (tests commit) — `src/tac/tests/test_p0_resume_warmup_geometry.py` (25 tests).
4. (this memo).

## Per-item verdicts

| item | status | anchor |
|---|---|---|
| 1 trigger widening | **DONE** | trainer: widened `_retreatment`-alone trigger at the resume block (8-space level, structural test enforces) + `resume_lr_rewarmup` row {boundary_epoch, rewarmup_epochs, floor, shape, reason ∈ stiff_term_drift/warm_start_weights_only/lever_drift_retreatment}. Byte-identity: plain continuation resumes + window=0 unchanged; stiff-only path is a strict subset. |
| 2 β₂-derived length | **DONE** | `src/tac/canonical_equations/adam_v_variance_warmup_20260717.py` (`adam_v_variance_warmup_epochs`; c2: ceil(2/(1−0.999)/75) = **27 ep**) + evaluator `adam_v_variance_warmup_length_v1` in `evaluators.py` + registry row appended + DSL `ResumeLRWarmup()` (LawRef-resolved 27; fallback **8** = CONFIG rung recorded with waiver reason). NOT composed into the c2 spec (additive; spec imports verified intact). |
| 3 ForkHeadSolve | **DONE (wired, default OFF, never-fired)** | trainer `--fork-head-solve {menon,ot_newton,flip_weighted,flip_median}` + `--fork-head-solve-tau` + `--fork-head-freeze-epochs`; solve runs pre-v0 on restored EMA deploy (post item-5 schedule positioning), APPLIES b* to `model.out_sdf.bias` + `ema.shadow["out_sdf.bias"]`, fail-closed (requires `--resume-from`; decoupled-field incompat registered); freeze zeroes `clipped["out_sdf"]` for the window. v0 verdict = measured receipt. DSL `ForkHeadSolve()`. |
| 4 MarginStepCap | **DONE (wired, default OFF, never-fired)** | trainer `--margin-step-cap` + `--margin-step-cap-window` (−1 → rewarmup window): post-update per-group ‖ΔW‖ projection `w = w_pre + ΔW·cap/‖ΔW‖`, before Stiefel re-projection + EMA; `margin_step_cap` row (per-epoch throttled). DSL `MarginStepCap(cap, window)`. Reference is PARAMETERIZED (no margin-field surface is cached at the update site — checked; the flag help documents deriving the cap from measured margin telemetry). |
| 5 #517 fold | **DONE** | (a) resume schedule positioning `model.softmax_temp/_hosc_beta ← *_for_epoch(start_epoch)` mirroring the lambda_pre probe, gated `start_epoch > 1` (fresh byte-identical); placed BEFORE the resume reorient (which reads `model.hosc_beta` under `--gpu-reorient`) AND the v0 verdict (`_fwd_numpy` reads both — the CONFIRMED phantom-baseline gap). `baseline_v0_schedule_positioned` row carries tau/beta/seg_form. (b) launcher receipt: `baseline_v0_d_seg/_d_pose/_implied_S/_skipped_reason` extracted phase-keyed from run.log. |
| 6 pose-engage boundary | **DONE** | (a) `last_boundary_epoch = ep` at pose_finish_engage under `not muon_switched` (mirrors 12250's guard); (b) `--pose-engage-wpose-ramp` cosine 0→full over the rewarmup window from `engaged_epoch`, placed AFTER the engage block so both gate paths ramp consistently from the engage epoch; DSL `PoseEngageWPoseRamp()`. Default OFF → incumbent step byte-identical. |
| 7 EMA clearance | **DONE (minimal)** | `--fork-ema-clearance`: arms `until = ema_warmup_updates(decay)` at a re-treatment fork; `_maybe_preserve_best` suppresses banking inside the window (both async+sync callsites thread snapshot `ema_updates`; loud first-suppression row); async verdict rows stamped `ema_warmup: true/false`. DSL `ForkEmaClearance()`. |
| 8 boundary-state persistence | **PARTIAL** | FINDING: `__hardness_prob` + `__rng_*` + `__recent_losses` are ALREADY persisted in the full sidecar; the c2 fork's `hardness_restored/np_global_restored: false` had TWO causes — (i) hardness restore is warm-start-GATED (intentional), (ii) the fork source was an EMA-BEST npz which persists NO run state (keyless). Landed: `--warm-start-restore-boundary-state` opt-in restoring the hardness baseline under warm-start when the source HAS it + DSL `WarmStartRestoreBoundaryState()`. NOT landed (named residual): boundary-state keys in the BEST/deploy npz — needs race-free async snapshot threading through `_capture_verdict_snapshot`/`_build_ema_checkpoint_arrays`. |
| 9 reorient at ramp end | **DONE** | forced one `recompute_self_orient` at `last_boundary_epoch + rewarmup_epochs` (OR-folded into the cadence condition; `ramp_end_forced: true` on the row; guard `use_self_orient`; window=0 → byte-identical). NOT flag-gated: it fires only when a rewarmup window is configured — the boundary-fix semantics itself. |

## MEASURED / DERIVED / INFERRED / ASSUMED

- **MEASURED (pre-dispatch, cited from the coordinator + verified in-source):** the c2 fork's
  missing ramp (trigger only inside `if _stiff_added:` — verified at source), sigma_min
  transient 0.0025→0.0084 settling ~ep674 (~23 ep), steps/epoch = 75 (600/8, matches the sister
  law's run-2 anchor), `hardness_restored/np_global_restored: false`, `_fwd_numpy` consuming
  `model.softmax_temp`/`model.hosc_beta` at the v0 verdict (source-verified — the #517 gap is
  real and score-visible in the v0 receipt).
- **DERIVED:** 27 ep = ceil(c/(1−β₂)/S) at c=2, β₂=0.999, S=75; c=1 ≡ sister bound (tested on a
  grid); the cosine ramp factor; the per-group projection algebra.
- **INFERRED (labeled so in the equation):** the RAdam variance-rectification rationale for
  c≈2 — `INFERRED_FROM_DOMAIN_LITERATURE`, PROVISIONAL-PENDING-VERIFICATION, mirroring the
  sister law; the isolating A/B (8 vs 27 ep at fixed β₂) is the owed anchor.
- **ASSUMED (residual risk):** that the loop's per-epoch schedule application fully overwrites
  the item-5 pre-loop tau/beta mutation (verified for the tau/beta assignment sites at ~12368;
  not exhaustively for every consumer between loop-top and first use); that the c2 spec's
  compiled hash is unchanged (verified by module import + unchanged factory outputs; a full
  compile-hash recompute was NOT run in this worktree — the gt cache/compile-custody artifacts
  are absent here, see pre-existing failures below).

## Round-1 adversarial self-review (findings → fixes, all pre-commit)

1. **Ramp-ordering inconsistency (6b):** first draft computed the w_pose ramp BEFORE the engage
   block sets `engaged_epoch` → the engage epoch ramped on the sigma path but not the
   muon/backstop path. Fixed: moved after the engage block; both paths ramp from the engage
   epoch.
2. **`tree_unflatten` scope bug (4):** the projection used `tree_unflatten`, which was imported
   only inside the resume block → NameError on a FRESH run with the cap armed. Fixed: hoisted
   into the `run_train`-level mlx.utils import.
3. **DSL orphan (8):** `--warm-start-restore-boundary-state` initially had no Lever owner
   (`lever_registry.completeness().unmapped` caught it). Fixed: `WarmStartRestoreBoundaryState()`
   factory; all 8 new flags now held.
4. **Fake-test smell:** dropped a scaffold test that could pass while verifying nothing
   (NO-FAKE class 2); replaced by source-structural + behavior tests.
5. **Attribution discipline:** 3 failing launcher end-to-end tests were stash-verified to fail
   IDENTICALLY on the clean base in this worktree (missing untracked
   `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` + #406 compile-custody artifacts) —
   pre-existing/environmental, NOT a regression:
   `test_fix4_opt_out_no_per_group_grad_clip`, `test_main_extra_bank6_is_caught_by_the_fixed_mem_preflight`,
   `test_main_dry_run_prints_safe_frac_policy_branch`.

Tests: 25 new + 19 existing warm-start/resume = 44 passed; adjacent keyword sweep 388 passed /
10 failed-pre-existing (the 3 unique above, parametrized) / 4 skipped. ruff --select F clean on
all touched files.

## Honest residuals (named, not silently deferred)

- **R1 (item 1):** `last_boundary_epoch` is NOT persisted — a crash mid-ramp on a PLAIN
  continuation resume drops the remaining ramp (pre-existing limitation of the stiff path,
  inherited). Follow-up: additive `__last_boundary_epoch` sidecar key.
- **R2 (item 6b):** `engaged_epoch` not persisted → a resume inside the pose ramp restarts the
  ramp at the resume epoch (conservative, bounded by the window).
- **R3 (item 7):** event detectors consuming `history` are not warmup-filtered (rows are
  stamped; the banking gate covers both verdict paths; the SYNC-path row does not carry the
  stamp — async does).
- **R4 (item 8):** BEST/deploy npz carries no boundary-state keys (the c2 fork source class);
  needs async-snapshot threading.
- **R5 (items 3/4):** efficacy UNMEASURED — both are default-OFF fireable levers in the
  duty-to-measure queue (activation ledger: never-fired). ot_newton remains the
  MEASURED-worse formulation (#288 verdict, FORMULATION scope); flip_median is the default arm.
- **R6 (item 2):** the law is PROVISIONAL; the A/B converts INFERRED→VERIFIED and calibrates c.
- **R7:** DAG FEED append is left to the post-c2 merge session (this worktree's DAG would
  conflict with the live main DAG); ready-to-paste summary: *"FEED-518: resume-warmup geometry
  landed on branch claude/p0_518_resume_warmup_geometry_20260717 — widened resume LR-rewarmup
  trigger + #517 v0 schedule positioning + pose-engage boundary + β₂-derived window law
  (adam_v_variance_warmup_length_v1, 27 ep vs config 8) + 5 default-off fork levers
  (ForkHeadSolve/MarginStepCap/ForkEmaClearance/PoseEngageWPoseRamp/WarmStartRestoreBoundaryState);
  pointer UNMOVED (means/apparatus)."*

## Merge instruction (post-c2 boundary, NOT this agent)

```
git checkout main
git merge --no-ff claude/p0_518_resume_warmup_geometry_20260717
# then: rerun src/tac/tests/test_p0_resume_warmup_geometry.py on main (gt cache present there;
# the 3 environmental launcher tests should pass on main), append the FEED-518 DAG block (R7),
# and queue the A/B (R6: 8 vs 27 ep at fixed beta2) + the duty-to-measure fires (R5).
```

The registry row for `adam_v_variance_warmup_length_v1` was appended in THIS worktree's
`.omx/state/canonical_equations_registry.jsonl` (committed): on merge, if main's registry moved,
resolve by re-running
`populate_adam_v_variance_warmup_length_equation()` on main instead of taking the worktree hunk
(registration is idempotent/append-only).
