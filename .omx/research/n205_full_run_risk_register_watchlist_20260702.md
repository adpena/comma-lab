# #205 full-run RISK REGISTER + watch-list — be diagnostic-ready when real results land

**Date:** 2026-07-02 · **Directive:** operator "update the triality and tasks so we're on the lookout for those risks and others when full results come back from real #205." · **Purpose:** every risk below is pre-staged with (SIGNAL = what to measure when #205/R1 results land) → (DIAGNOSTIC = how to tell *which* mechanism) → (RESPONSE = the already-designed fix). A wall is never "fail" — it routes to a response. Pointer 0.19110 UNMOVED; this is apparatus/means. Each risk maps to a triality leg (DAG watch-node ∧ equation ∧ DSL lever) + a task.

## A. POSE risks (R1 is the leading indicator; the store_nothing A/B arm carries these)
- **A1 — render-legibility wall.** The non-photoreal witness render gives PoseNet a weak/ill-conditioned gradient → d_pose plateaus above ~0.018. **SIGNAL:** d_pose(epoch) plateaus; PoseNet-Jacobian σ_max/σ_top6/eff-dim on the *trained* render vs real GT. **DIAGNOSTIC:** `tools/levelset_pose_gate.py` conditioning (not just rank-6). **RESPONSE:** annealed-w_pose render-co-adaptation phase → else **warp-real-luma fallback** (rate 0.03–0.07, sub-0.19 at-threshold). Eq: `store_nothing_pose_carrier_rate_collapse_vs_dpose_v1`.
- **A2 — co-adaptation coupling.** w_pose>0 pose training flows back through the INR and perturbs the render → **d_seg REGRESSES.** **SIGNAL:** d_seg rises during/after the pose stage (Monitor logs d_seg + d_pose per verdict). **DIAGNOSTIC:** d_seg(epoch) with-vs-without w_pose. **RESPONSE:** freeze-witness-then-joint two-phase; lower pose-LR; trunk-stopgrad (the measured seg⊥pose exact freeze-and-add, cos 5.9e-5). Task #227.
- **A3 — one-stage insufficiency (the operator's core question).** Pose needs a *curriculum* like d_seg. **SIGNAL:** slow / non-monotone / oscillating d_pose descent (fit rate k, plateau ratios). **RESPONSE:** residual-LR warm-up+anneal, a staged pose schedule — a DIFFERENT curriculum than d_seg's (pose is a smooth 6-DoF twist, not a partition to nucleate).

## B. d_seg convergence risks (the primary axis — the whole run's purpose)
- **B1 — training-not-representation gap unclosed.** AA-floor 0.00086 < need-band (0.00077–0.00118), but the WITNESS *reaching* it is TRAINING; the full run may plateau ABOVE the band. **SIGNAL:** d_seg(epoch) plateau level vs need-band + AA-floor. **DIAGNOSTIC:** per-class d_seg + lane/movable recall (task #209 tool). **RESPONSE:** islands levers already ON (structured-init/persistence/island-amplify); extend epochs; θ* margin-field #218.
- **B2 — rare-class erasure.** lane/movable seeded at 0 → do they GROW? **SIGNAL:** per-class recall + d_seg for classes 1/3. **RESPONSE:** rare-class-protected init #208; amplify weight.
- **B3 — curriculum/stage risk.** Is l7-demotion correct? does Muon finish or hurt? stage-transition spikes? **SIGNAL:** stage-diff d_seg (which stage moved it) off the per-stage checkpoints. **DIAGNOSTIC:** #216 stage-diff attribution. **RESPONSE:** #188 decide_next_stage (EXTEND/ADVANCE/RERUN/ROLLBACK).
- **B4 — under-training / plateau.** Run may need more epochs / different schedule (the operator's pose concern applies to d_seg too). **SIGNAL:** end-of-schedule d_seg slope (still-descending → extend; plateau → advance). **RESPONSE:** the reactive stacking (#189 campaign.decide_next_stage), operator-gated resume.

## C. RATE risks
- **C1 — store-nothing d_pose sets the rate.** Closes → rate 0.049; walls → table 0.51 → sub-0.19 at-threshold. **SIGNAL:** R1 / A-arm store_nothing d_pose. **RESPONSE:** if walls, table + minimize the keyframe (temporal HEVC/AV1-VCM, measured 0.004–0.018).
- **C2 — mod-dim 32 vs rate-saving 19 (#223 deferral).** 32 may leave rate on the table. **SIGNAL:** L13 byte-close rate @ mod-dim 32. **RESPONSE:** mod-dim 19 A/B (shape-changing → fresh arm).
- **C3 — adam-beta2 0.999 vs derived 0.9999999 (#222 deferral).** First-row byte-identity choice; re-open post-row.
- **C4 — counted-payload minimality.** partition + pose + residual bytes minimal? **SIGNAL:** byte-close section breakdown. **RESPONSE:** L13 format levers + the flat-minima/MDL weight-compression lever (#242).

## D. APPARATUS / runnability / honesty risks
- **D1 — full-run OOM at a *different* stage.** The mem-preflight projects the launch peak, but a stage (pose-carrier warp @ native 874×1164, or a lever's all-pairs materialization) may spike differently. **SIGNAL:** safe_run RSS + per-stage RSS log. **DIAGNOSTIC:** the OOM-law eq `oom_verdict_batch_spike_peak_rss_v1` + mem-preflight. **RESPONSE:** chunk the offending full-P scorer forward (the OOM law).
- **D2 — determinism / resumability.** crash-resume bit-faithful? per-stage checkpoints intact? **SIGNAL:** resume-validity + crash-resume smoke (proven). **RESPONSE:** the durability non-negotiable.
- **D3 — byte-close honesty (axis-9).** Exact S through the real decode, NO borrowed/ancestor numbers. **SIGNAL:** the `CorrectnessDemonstration` validates fail-closed. **RESPONSE:** axis-9 / the executable gate.
- **D4 — A/B apples-to-apples.** Two SEQUENTIAL runs (2×67.6>128 GiB); store_nothing's render co-adapts, table's doesn't. **SIGNAL:** the two arms' d_seg trajectories must be comparable. **RESPONSE:** hold the witness-training config identical; only the pose carrier differs.

## E. SYSTEMIC / "and others" (proactive-recall — do not forget these when the row is claimed)
- **E1 — exact-eval axis (CPU vs CUDA).** The submission axis is OURS; the witness CPU/GPU gap is UNMEASURED. **SIGNAL:** exact `upstream/evaluate.py` on BOTH axes at sub-0.19. **RESPONSE:** measure both, never infer one from the other.
- **E2 — inflate.py 30-min budget (#214).** The store_nothing warp @ native 874×1164 must decode in-budget. **SIGNAL:** inflate wall-clock. **RESPONSE:** multiprocess → fp32 → torch-T4 → Rust lowering.
- **E3 — contest legality / rule-118.** NO scorer weights/SegNet/PoseNet in archive; generic generator FREE, learned/video-derived payload COUNTED. **SIGNAL:** payload-cleanliness audit. **RESPONSE:** the packet-compiler audit bundle.
- **E4 — the exact-row is a PROMOTION, not a LOCAL_SEAL.** Pointer moves ONLY via the byte-closed contest-CPU/CUDA row; advisory ≠ promotion. **SIGNAL:** evidence-grade of the S claim. **RESPONSE:** the CorrectnessDemonstration PROMOTION level (contest authority required).

## Response routing (the one-glance table)
| Risk | Watch signal | Pre-staged response | Triality anchor |
|---|---|---|---|
| A1 render-legibility | d_pose plateau + Jacobian σ | co-adapt phase → warp-luma fallback | eq store_nothing; DAG FEED-snx |
| A2 coupling | d_seg rises w/ w_pose | freeze-then-joint; trunk-stopgrad | task #227 |
| B1 d_seg gap | plateau vs need-band | islands levers; θ* margin #218 | AA-floor eq; DAG FEED-ma |
| C1 rate | store_nothing d_pose | table + keyframe codec | eq store_nothing |
| D1 OOM-stage | per-stage RSS | chunk full-P forward | eq oom_verdict_batch |
| E1 CPU/CUDA | both-axis exact eval | measure both | submission-axis memory |

**Consumers:** the R1 agent (A-risks live now), the #205 full-run watch (all), the byte-close/exact-eval path (C/D/E). Sisters: `n205_updated_recursive_adversarial_review_20260702T231003Z.md` · `n205_default_off_optimality_audit_20260702T230823Z.md` · the pose-open + OOM memories.
