---
title: Codex findings - DDM J6 #366 prefire adversarial review
utc: 2026-07-23T07:08:32Z
lane_id: ddm_j6_366_prefire_adversarial_review_20260723T065554Z
verdict: BLOCK
verdict_scope: EXACT SEALED J5 TICKET AND CURRENT CONSUMER/LAUNCHER PREFIRE CONTRACT ONLY
research_only: true
execution_allowed: false
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Gate verdict

**BLOCK.** The J5 one-step result is genuine, but the sealed
`13e194a8...b6e8` campaign contract can report fire readiness or schedule
completion without proving the cumulative component gate, the required pose
finish, the worst-stage memory geometry, or the final stage targets.

No campaign, paid dispatch, score claim, promotion, or pointer movement
occurred. The pointer remains `0.1910828242 [contest-CPU]`.

# Six-target disposition

| Target | Disposition | Fresh-eyes result |
|---|---|---|
| 1. Active-pair/lifecycle mask | **PASS** | The sealed eight-pair proposal selects 18 whole-lifecycle-feasible `center_x` coordinates. Lane is intentionally held in stage 1 and is active in stages 2/3. |
| 2. Q8 acceptance/rollback | **PASS** | Q8 state is deterministic; exact admission uses compiled receiver bytes through uint8, R, and frozen CPU scorers. Rejected candidate objects do not mutate the admitted state, and an exhausted ladder preserves an immediately parse-backed rollback checkpoint. |
| 3. C1 bucket split | **BLOCK** | The measured `-1314 + -2013 = -3327` split is exact, but fire readiness uses component/residual safety versus the **last admitted** state while the computed cumulative-versus-baseline row is telemetry only (`tools/launch_ddm_joint_descent.py:1125-1233`). A two-move sequence can be locally safe and cumulatively regress d_seg. |
| 4. Pose finish | **BLOCK** | The ticket's engage condition is string-only (`src/tac/optimization/direct_description_joint_descent.py:208`). The launcher uses an immediate binary 0→1 switch after any strict Seg admission (`tools/launch_ddm_joint_descent.py:1018`), not a checkpointed #383 pose-finish detector. The banked-R1 `0.001610` comparator is incorrectly a binding stage target (`config:169-179`); no same-vehicle trajectory from measured `163.0613` is established. |
| 5. Real memory/resume | **BLOCK** | Checkpoint content, atomicity, distinct names, EMA/Adam/cursor state, and mid-stage load path pass. Memory custody does not: the measured stage-1 pair447 window had 8 island secants; the sealed stage-3 maximum has DERIVED 28 island + 24 Lane secants. Basis residency alone rises from 0.7276554108 to 4.7297601700 GiB. The likely headroom is not a substitute for the required worst-geometry measurement. The ticket also still cites the older J3 preflight SHA (`config:78-81`). |
| 6. Convergence/exit/stop | **BLOCK** | Stage limits advance on `REALIZED_STAGE_DESCENT_CONTINUE` even when the target is unmet (`launcher:1402-1436`). Flat/worse plateau rows first become blockers, so the plateau exit is unreachable as documented. Three maximum-step advances yield `FULL_RUN_SCHEDULE_COMPLETE` without a final-target gate (`launcher:1536-1537`). The fraction targets are thresholds, not a measured trajectory or floor. |

# What is real and should be preserved

- Ticket file SHA `2ae7da9058f8c5a421ffc494ab0407947391ea28ede1b1384248f298906fdf43`,
  semantic SHA
  `13e194a8a354d53489f0ff68a5042237e69b4b6841a6b7959a15873fffa7b6e8`,
  and typed hash
  `d43608af799b2f2d04e248413ceb944c093701441eafb222f2b3cdf3d32b8d80`
  independently re-derived.
- The exact proposal archive SHA is
  `d4eb1450f461437e714d08a9349cc735fe79b53a1739a2de92ef4850287dfd0d`.
- The n600 row is MEASURED:
  `delta_d_seg=-0.000028203328450521203`,
  `delta_d_pose=-0.00016296301816964842`, five bytes saved, and
  `delta_S=-0.002843840398518996`.
- C1 partition custody is exact and global/per-class totals fail closed.
- The J5 checkpoint SHA
  `1399292164682955bd9937d204692521542687f69223ddc9cc1e669ae05a944e`
  contains theta, EMA, both Adam moments, typed identity, run cursor, exact
  archive identity, and resume-registry state.
- Focused inherited tests remain green: `34 passed in 79.52s`. They do not
  exercise the cumulative fire latch, target-unmet stage limit, plateau
  preemption, pose detector, or worst-stage memory geometry.

# One reformulated $0 fix arm

Registered exactly one L0 fix arm:
`ddm_j6a_366_prefire_contract_hardening` — **Pose engage, cumulative fire
gate, worst-geometry memory, and governed-stop hardening**.

Minimal closeout:

1. At `tools/launch_ddm_joint_descent.py:1018`, replace the declarative/binary
   pose switch with a typed, checkpointed engage state driven by the run's own
   exact verdict history. Preserve the exact sqrt-pose objective after engage;
   keep `0.001610 / 0.127 / 7.2KB` as non-binding comparator metadata only.
2. At `tools/launch_ddm_joint_descent.py:1153-1233`, keep local pure-price
   admission but derive the component/residual **fire latch** versus stage-00,
   using the already-computed cumulative bucket row.
3. At `tools/launch_ddm_joint_descent.py:832-865`, add a bounded
   all-groups/worst-window memory bootstrap at pair start 498 or 499, bind the
   resulting receipt and final launcher/consumer hashes, and prove a fresh
   J5-typed process resume.
4. At `tools/launch_ddm_joint_descent.py:1427-1537`, make target-unmet limits
   and plateaus governed non-promoting stops and require an exact final-target
   verdict before schedule completion.
5. Reseal the ticket and add behavioral regressions, then require independent
   MAIN landing review. No campaign launch belongs in this fix arm.

# Re-derivation

```bash
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest --timeout=180 -q src/tac/optimization/tests/test_direct_description_joint_descent.py tools/tests/test_launch_ddm_joint_descent_j5.py src/tac/optimization/tests/test_pure_priced_realized_objective.py
```

The machine-readable custody, six dispositions, derived geometry, and
synthetic control-flow counterexamples are in
`.omx/research/ddm_j6_366_prefire_adversarial_review_receipt_20260723.json`.

# STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `PROGRAM.md`;
`docs/operating_manual_craft_handoff.md`; sealed J5 ticket; J2-J5 and V19/V19b
findings/session anchors; tracked J5 smoke/preflight/review/reseal receipts;
external immutable J5 full-run, baseline, proposal, memory, checkpoint, and
dry-run artifacts; consumer, launcher, focused tests; lane/subagent/frontier
state; prior #205/pose-finish custody notes; delegation
inboxes.

MAIN must independently review this branch diff and re-derive the four blockers
before landing. This memo does not authorize a fire.
