# ddm_m1c1 — executable M1 stopping amendment

**Date:** 2026-08-08  
**Scope:** M1 trainer + live ticket + canonical stopping module only.  
**Axis:** source/tests are scorer-free; the only consumed scorer number is the already-banked n32
`[macOS-CPU advisory torch upstream SegNet]` K8 tail-average row.  
**Pointer delta:** NONE. No Metal burn, CPU scorer job, archive, or `upstream/evaluate.py` was run.

## Reproduction before build

- **MEASURED:** the charter pins reproduced before edit: ticket SHA-256
  `9c8373b5b352cacc2456a21eac0deb53e32f445eb942e4675043825a1d896500`; trainer SHA-256
  `1ef18faf37e2f171d480b4e8073c453185f9ae00a1b3200b46d5bb258cd60895`.
- **MEASURED, scoped correction to the charter:** quoted-glob census
  `rg -n "evaluate_trajectory_stop" src tools experiments --glob '*.py'` found four production call
  sites already in the live tree, not exactly one: terminal pose at
  `src/tac/optimization/terminal_pose_gn.py:1231`, SL2 at
  `experiments/ddm_sl2_sq2_persist_and_compose.py:221`, and SQ1 at
  `experiments/ddm_sq1_stage_decomposition_and_solved_paint.py:178,207`. None was on the M1 trainer.
  The load-bearing finding reproduced: the pre-amendment M1 loop had no stopping call, no durable
  per-eval append, and no `stop_policy` consumer.
- **MEASURED:** post-edit ticket SHA-256 is
  `90cf28d390999ef9cda47340d9ec01bc65a15fb9ab3f88c60625abc29b414ec9`.

## B1 — resume-safe real-path eval journal

Built `O_APPEND` + one `os.write` + file/directory `fsync` at
`experiments/ddm_mx1_pr130_semantic_renderer.py:139-156`. The active reader at `:159-179` never
deletes history: a resume segment rewinds the logical view to its checkpoint, retains the abandoned
tail physically, then admits the new segment's rows. The real loop writes immediately after every
actual MLX eval at `:3487-3513`, before consulting the controller. Each row carries step,
`objective_S`, `d_seg_batch_mlx`, best d_seg, loss, LR, wall time, `weights_stepped`,
`accepted_batch_fraction`, and prior-checkpoint path/SHA.

Numbers:

- **DERIVED:** `objective_S = 100*d_seg_batch_mlx`, from the contest Seg coefficient already in the
  ticket.
- **MEASURED at runtime:** wall seconds, loss, d_seg, LR, completed update count, accepted-batch
  fraction, and checkpoint SHA.

Both-direction control:

- Bad/resume state fires: `test_m1_journal_resume_rewinds_active_view_without_deleting_tail` writes an
  abandoned step-100 row, resumes from step 50, and verifies the active step 100 is the new value.
- Good history stays intact: the same control verifies the accepted step-50 row remains active and all
  five physical JSONL records remain on disk. A short/incomplete JSONL row fails closed in the reader.

## B2 — executable controller and both terminal modes

Ticket values are consumed and re-derived at
`experiments/ddm_mx1_pr130_semantic_renderer.py:192-231`; drift in the one-flip constant versus
`N*H*W` refuses. Every eval invokes canonical `evaluate_staircase_aware_stop` in process at
`:3514-3554`, appends a typed decision, forces a live/EMA checkpoint on every halt, and writes an
atomic event/step-cap/calibration receipt at `:3567-3615`.

The `controlled-train` wrapper at `:4320-4415` launches only the ticket's governed `safe_run` argv and
survives exit 124. It reads the durable safe-run status receipt, evaluates the journal at a
`wall_clock_seconds` boundary, appends the boundary decision, emits canonical `CapStopReceipt`, and
queues the exact ticketed resume argv. A timeout before the first eval is still receipted as
`min_eval_rows_not_met`; it is not silently lost. Fresh and resume routes plus separate status,
pidfile, and terminal paths live at ticket lines 60-225 and 860-934.

Both-direction control:

- Bad boundary fires: `test_m1_controlled_train_receipts_a_safe_run_wall_cap` mocks only process
  completion (no Metal), feeds a real durable journal/status receipt, and verifies
  `terminal_mode=wall_clock_cap`, `action=QUEUE_RESUME`, and canonical
  `bound_kind=wall_clock_seconds`.
- Good/non-boundary stays live: staircase controls below return `CONTINUE`; a resolved event-free
  plateau returns the distinct `STOP_CONVERGED`, never cap-bound. `test_live_m1_ticket...` also proves
  both exact child argvs carry `--status-receipt` and the correct `--fire-argv-key`.
- **MEASURED source gate:** `tools.mx1_fire_guard.evaluate_guard` returned
  `passed/fire_guard_passed` with `flag_classification_ok` for both fresh and resume child keys. It was
  read-only; no guard verdict or run artifact was rewritten.

## B3 — staircase-aware convergence gate

Canonical additions are `StaircaseStopConfig`/`StaircaseStopDecision` at
`src/tac/optimization/trajectory_stopping.py:269-344` and the evaluator at `:775-900`. A smooth-fit
stop is admitted only when all of these hold: a full event-free horizon, statistically flat loss,
trajectory-own objective-slope uncertainty below the score marginal, advancing weight-update
liveness, and no sustained score erosion while loss falls. Boundaries with unresolved evidence
queue resume; erosion rolls back.

Numbers:

- **DERIVED from ticket geometry:** event quantum
  `100/(120*384*512) = 4.238552517361111e-6 S`.
- **DERIVED from ticket geometry:** event-free horizon `5 evals * 50 steps/eval = 250 steps`; this
  requires a full new ticket window after the most recent one-flip event.
- **DERIVED from ticket geometry:** erosion slope threshold
  `100*1e-6 d_seg/eval / 50 steps/eval = 2e-6 S/step`.
- **ASSUMED statistical convention, explicitly labeled in canonical source:** two-sided 2-sigma
  (~95.4%) detection; the decision resolution itself is estimated from this trajectory, not copied
  from another vehicle.

Both-direction controls at `src/tac/optimization/tests/test_trajectory_stopping.py:168-245`:

- Required plateau-then-drop trace: `CONTINUE`, blocker `event_free_horizon_not_met`.
- Required flat-objective/falling-loss trace: `CONTINUE`, blocker `loss_tail_not_flat`.
- Good resolved live plateau: `STOP_CONVERGED` with no blockers.
- Frozen weights/zero accepted batches: `CONTINUE`, blocker
  `weight_update_liveness_not_clear`.
- Wall cap below threshold refuses; at/above threshold receipts successfully.

## B4 — LawRef EMA plus bound K8 candidate

The ticket-driven LawRef derivation is at
`experiments/ddm_mx1_pr130_semantic_renderer.py:234-251`; the real Polyak update is at `:254-267` and
is called after every optimizer update at `:3453-3457`. Checkpoint bundle code at `:3231-3308` writes
distinct atomic live and EMA NPZs at step 0, every 250 steps, and every controller halt; resume refuses
without the paired EMA checkpoint. Once eight live checkpoints exist, it materializes a loadable K8
simple-mean candidate with exact member paths. No candidate is auto-adopted.

Numbers:

- **MEASURED, banked n32 advisory:** K8 mean beat final live by
  `-5.880991617838397e-6 d_seg`; source
  `.omx/research/ddm_mx1t_20260807/mx1t_facets_receipts.jsonl`, K8 row.
- **DERIVED:** K8 x 250-step checkpoint cadence binds a 2,000-update two-time-constant window;
  `phi=2000/3250=0.6153846153846154`, `U=3250`, and
  `ema_decay_run_geometry_v1(decay_from_warmup_fraction)` gives exactly `d=0.999`.
- **UNDETERMINED on n120:** whether live, EMA, or K8 wins. The ticket queues exact same-object CPU
  facet argvs for all three and forbids adoption before a selector names the minimum.

Both-direction controls:

- EMA LawRef control derives 0.999 and refuses a ticketed 0.997 drift; real update control moves a
  shadow from 0 to 0.5 for live=2/decay=0.75 and refuses parameter-tree drift.
- K8 materializer control averages 1 and 3 to 2 and refuses mismatched checkpoint key sets.
- Good selection behavior stays neutral: checkpoint metadata remains
  `selection_status=QUEUED_SAME_OBJECT_CPU_FACETS`; no EMA or average is called authoritative.

## B5 — same-object schedule calibration and no-jump extension

The LR function at `experiments/ddm_mx1_pr130_semantic_renderer.py:182-189` clamps its coordinate at
the original 3,250-step horizon. The trainer uses that same fixed horizon for LR and curriculum on
resume. Fresh training is forced to checkpoint/stop at step 250 until exact ticketed step-0 and
step-250 n120 CPU facets are compared. The selector at `:4285-4317` admits continuation only for at
least one n120 Seg lattice flip; otherwise it writes `rollback_no_resume`. Resume refuses a missing or
failed selection receipt before MLX setup.

Numbers:

- **SOURCE-VERIFIED/BORROWED_CANDIDATE_NOT_ADOPTED:** base LR `2e-7`, PR130 `train.sh:113`.
- **DERIVED:** old 3,250-step terminal LR is `2e-9`.
- **DERIVED control:** recomputing a 6,500-step horizon at step 3,250 would produce
  `1.0097607188262213e-7`, a `50.48803594131106x` jump. The implemented resume remains exactly
  `2e-9` at steps 3,250 and 6,499.
- **DERIVED:** schedule-admission improvement is one n120 flip in d_seg,
  `4.238552517361111e-8`.

Both-direction controls:

- `test_m1_resume_cosine_holds_terminal_lr_without_horizon_jump` verifies the bad re-horizon jumps
  over 50x and the production path stays at 2e-9.
- `test_m1_schedule_selection_admits_one_flip_and_refuses_no_gain` admits a >one-flip same-object CPU
  improvement and refuses zero gain.

## Exact ticket semantic leaf diff

Method: recursive JSON-object comparison from `HEAD` to the working ticket; object key ordering is
ignored and each argv list is one semantic leaf (so inserting two safe-run flags does not masquerade
as 52 shifted-index changes). Result: **50 semantic leaves: 39 scalar leaves + 11 argv/list leaves;
0 removals.** List hashes are SHA-256 of compact JSON (`separators=(',', ':')`); exact tokens are the
value at the named JSON pointer in the post-edit ticket.

List leaves:

| op | JSON pointer | old | new |
|---|---|---|---|
| ADD | `/argv_m1_controller_fresh` | absent | len 10, `25c92031428a340fd9ff78365f7b509101f0ebb0f467df4abbb50a7bb9d27b9e` |
| ADD | `/argv_m1_controller_resume` | absent | len 10, `fd82b3e830840194aed2c1df57eea37a811ca270c1de0de0145658c6ae32b088` |
| CHANGE | `/argv_m1_n120_cap_saturated` | len 62, `8ff9105c33be4feacdc4ffad98a35fdb85b55bf8c58acd4fc7f65dd2c65734ca` | len 66, `2b253093aa2354e042f59eacc116abf0b37298dacc0bec19ea80310100ab8aae` (adds status receipt + child pidfile) |
| ADD | `/argv_m1_n120_cap_saturated_resume` | absent | len 68, `6f085a4f733898aea1b8f8bf13fa3d4897dbf0d1d1fe5aa8b22f9989556c82fc` |
| ADD | `/stop_policy/executor/child_argv_keys` | absent | len 2, `f9faf6a47f2b26981c69cd19afbf53624c1df2886a3843eac67cff55ca7953c3` |
| ADD | `/stop_policy/executor/same_object_cpu_selection/schedule_baseline_cpu_argv` | absent | len 27, `85799555af380c3cc2007e92c0a2715f1d7fa26cffdd1dea2ca29fb8c6e8339d` |
| ADD | `/stop_policy/executor/same_object_cpu_selection/schedule_candidate_cpu_argv` | absent | len 27, `12211ba9bf6cd55b00ed20dc4802374b6c44e10906055c95bedfe79faf04acb6` |
| ADD | `/stop_policy/executor/same_object_cpu_selection/schedule_select_argv` | absent | len 12, `4e03c5920546e9e28a95bc660bb5e477898bf18b0df0d7b178fb097244674d7d` |
| ADD | `/stop_policy/executor/same_object_cpu_selection/terminal_live_cpu_argv` | absent | len 27, `e4e7a96a46676d16cb00d2eb5464d8a80d1c3a231fba8a9bd4bf4b38a72f4b3a` |
| ADD | `/stop_policy/executor/same_object_cpu_selection/terminal_ema_cpu_argv` | absent | len 27, `d2063b68b9453c3c368571313ae7cfee391a439ff96bbcb07244fce27ec161f6` |
| ADD | `/stop_policy/executor/same_object_cpu_selection/terminal_tail_average_cpu_argv` | absent | len 27, `6d9a0351c692f00d7411bf13c6aafc4afb4e5732913f890b80bf75ab7ae9b8aa` |

Scalar leaves (exact new value; `CHANGE` includes the old semantic in parentheses where useful):

- ADD `/mem_probe_receipt_paths/argv_m1_n120_cap_saturated_resume` =
  `.omx/research/ddm_m1_20260808/run/n120_metal/mem_probe/mem_probe_receipt.json`.
- CHANGE `/resumability` = live + LawRef EMA + K8 candidates, 250-step/halt checkpointing, and
  resume consumes live checkpoint, paired EMA, journal, schedule receipt, and controller argv.
- CHANGE `/stop_policy/doctrine` = in-process durable eval-before-decision controller plus timeout
  survivor; old value delegated evaluation to MAIN/monitor.
- ADD controller route scalar leaves: fresh child key `argv_m1_n120_cap_saturated`, fresh status
  `.../safe_run_fresh_status.json`, fresh terminal `.../terminal_fresh.json`; resume child key
  `argv_m1_n120_cap_saturated_resume`, resume status `.../safe_run_resume_status.json`, resume terminal
  `.../terminal_resume.json`.
- ADD `/stop_policy/executor/decision_path` = `.../stop_decisions.jsonl`;
  `/journal_path` = `.../eval_journal.jsonl`; `/resume_argv_key` =
  `argv_m1_controller_resume`; `/event_free_horizon_evals` = `5`.
- ADD safety bounds: fresh `3250`, resume `6500`.
- ADD EMA leaves: `law_ref=ema_decay_run_geometry_v1`, `derived_decay=0.999`,
  `updates_per_run=3250`, `warmup_fraction=0.6153846153846154`, `tail_average_k=8`, measured bank
  delta `-5.880991617838397e-6`, exact bank source path/axis, and the stated 2,000-update derivation.
- ADD selection adoption rule = no live/EMA/K8 adoption until all three n120 CPU facet commands pass
  at the same step and a selector names the minimum.
- ADD schedule leaves: `base_lr=2e-7`, status `BORROWED_CANDIDATE_NOT_ADOPTED from PR130
  train.sh:113`, `calibration_step=250`, `horizon_steps=3250`, step0→step250 one-flip continuation
  rule, clamp-at-3249/hold-2e-9 extension rule, and selection path `.../schedule_selection.json`.
- ADD terminal map leaves: fresh `.../terminal_fresh.json`; resume `.../terminal_resume.json`.
- CHANGE predicate leaves: `EXTEND_WITH_RESUME` now requires typed queue + schedule receipt;
  `QUEUE_RESUME` names step/wall/calibration bounds; `STOP_CONVERGED` names all staircase/loss/noise/
  liveness/erosion gates; `evaluator` points to `evaluate_staircase_aware_stop` with canonical smooth
  fit inside.
- CHANGE telemetry leaves: exact step0/step250 and live/EMA/K8 CPU fire order; exact O_APPEND row
  schema and resume-rewind semantics.

The count above is 11 list rows plus 39 non-list rows = 50. The prose grouping does not omit or add a
semantic leaf; it folds only sibling scalar paths with the values stated here.

## Controls and validation

- `.venv/bin/python -m pytest -q src/tac/optimization/tests/test_trajectory_stopping.py
  src/tac/pr130_lift/tests/test_mx1_pr130_lift.py` → **26 passed** (scorer-free; an existing
  `mlx_device_probe(device='cpu')` test emitted a no-Metal-device atexit warning after success; no
  training or Metal command fired).
- `py_compile` passed for trainer and canonical stopping module.
- `git diff --check` passed.
- Ruff import/zip/export checks passed (`I001,B905,RUF007,RUF022`).
- Ticket parses with `python -m json.tool`.
- Fresh and resume `mx1_fire_guard.evaluate_guard` both passed read-only, including unclassified-flag
  refusal control.
- The four modified Python files each received two successful `review_tracker.py mark-file` passes
  (`m1c1-code-pass-1`, then `m1c1-code-pass-2`) after the final source/test hashes were frozen, plus
  the registered `council` policy-approval mark; per-file policy checks report zero violations.

## RECALL EVIDENCE

Searched beyond charter seeds:

1. `rg -n "tail[-_ ]average|ema_decay_run_geometry|event[-_ ]driven.*stop|plateau[-_ ]then[-_ ]drop|safe_run_status_receipt" .omx/research src/tac experiments tools .omx/state --glob '!*.log' --glob '!*.jsonl'`
   - Found the already-built safe-run durable status surface, `ema_decay_run_geometry_v1`, DY2's
     separate JD1 plateau-tail law, and the live MX1T K={2,4,8} analyzer.
   - Changed plan: reused safe-run status and the canonical run-geometry EMA evaluator; bound M1's
     measured K8 protocol without importing JD1's formulation-specific plateau law.
2. `.venv/bin/python tools/list_canonical_equations.py --json | rg
   'ema_decay_run_geometry|trajectory_derived_stopping|score_marginal'`
   - Found both required canonical laws. Changed plan: extended the canonical stopping surface instead
     of creating a twin and executed the registered EMA evaluator at runtime.
3. `rg -n "M1|MX1|trajectory stopping|tail average|EMA" .omx/research/CANONICAL_RESEARCH_INDEX*
   .omx/research/sub015_DAG_* .omx/state/main_hot_state.md`
   - Found no later M1 executable-stop implementation beyond the charter/reviews; found the K8 banked
     receipt and current own-vehicle pointer. Changed plan: preserved K8 as a candidate protocol and
     used hot state, not the older common-contract pointer literal.
4. Read all M1R4 A/B/C reviews. B supplied B1-B5; C confirmed the executable-stop defect while also
   containing findings outside this charter's legal M1 trainer/ticket/stopping scope. Those were not
   silently declared cured here.

## Boundaries, blockers, and follow-on disposition

- **NOT MEASURED:** real n120 Metal execution of the amended loop. The charter forbids firing it.
- **NOT MEASURED:** n120 step0/step250 CPU schedule result. Status is
  `QUEUED-WITH-A-FIRE-ORDER` via the three exact schedule argvs in the ticket.
- **NOT MEASURED:** n120 live versus EMA versus K8 CPU selection. Status is
  `QUEUED-WITH-A-FIRE-ORDER` via the three exact terminal facet argvs; no basis is adopted.
- **NOT MEASURED:** archive bytes, pose, exact score, contest-CPU, or contest-CUDA. No pointer moved.
- **BLOCKED BY REQUIRED PROCESS, not implementation:** this amendment invalidates the three prior
  review passes. Ticket remains `COMPOSED_UNSEALED`; the required three new independent empty-finding
  reviews must complete before any fire.
- **FORMULATION scope only:** the controller governs this M1 n120 CAP route. It makes no family-level
  claim about other trainers or stopping laws.

`NEXT_IF_RESUMED:` run three independent review passes over this single amendment; if sealed, refresh
the main mem probe, run the ticketed fire guard, fire `argv_m1_controller_fresh` (which must stop at the
step-250 calibration boundary), execute the two CPU schedule facets plus `schedule_select_argv`, refresh
the resume guard, then fire `argv_m1_controller_resume`; at a terminal checkpoint run the exact
live/EMA/K8 CPU facet order before adopting any basis.

Own-vehicle frontier remains **S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]**.
