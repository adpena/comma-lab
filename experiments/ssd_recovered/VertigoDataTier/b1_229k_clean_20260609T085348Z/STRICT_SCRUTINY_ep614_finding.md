# B1 clean run — strict-scrutiny finding (verified ep0→673, 2026-06-09T09:05Z)

**Run:** `b1_229k_clean_20260609T085348Z` (the clean+stabilized PR95 baseline that
supersedes the diverged/off-spec pilot `b1_229k_pilot_20260609T055851Z`).
**Authority:** `[macOS-MLX research-signal]` — proxy losses only. NOT a contest score.
The arbiter is the B2 exact eval (running). promotion_eligible=false.

## Liveness (durable signals, NOT `ps`)
- heartbeat ALIVE age ~60s, pid=6492; telemetry at ep673; sec/epoch ~1.05; nan_inf=0.
- The session shell's `ps` cannot see the nohup-detached daemon — heartbeat + telemetry
  growth are the liveness ground truth (see memory `durable-detached-daemons-...`).

## ep-250 harvest FIRED (off-by-one + self-bootstrap fixes worked)
- `harvest_status_ep250.json`: state=evaluating_b2; exported 256,072-byte HIV1 single-member
  archive from the ep249 EMA-best checkpoint; running 600-pair contest_auth_eval.py --device cpu
  (pid 11160). FIRST real backend-only exact eval; validates the 600-pair bridge the killed
  run's B2 failed on (ModuleNotFoundError, now fixed).

## The verdict on the relaunch agent's "pose resolved to 3.59" claim: FALSIFIED (cherry-pick)
Honest trajectory (615 telemetry rows, every ~40 ep):

| phase | loss_seg | loss_pose | grad_norm raw | verdict |
|---|---|---|---|---|
| Stage 1 CE (ep0-303) | 1.16→1.17 stable | 157→2.8 CONVERGED | 53K→95K | clean; BEST ckpt = ep249 total=3.16 |
| Stage 2 tau_softplus (ep304+) | 1.19→1.51 mild creep | osc 16-119 no descent | 192K→6.3M CLIMBING | destabilized a converged state |

- SEG chamber STABLE (1.12-1.55, never diverges) — the CORE FIX worked vs killed run's SEG 18→400.
- grad-clip fires 100% of steps; nan_inf=0 — no actual divergence (containment holds).
- "3.59" = ep280 (END of stage 1, where pose legitimately converged to 2.8). The moment stage 2
  (tau_softplus) engaged ~ep304, pose DESTABILIZED. Selector still ranks ep249 (stage-1 end) best.

## Why the EXPORT is safer than live weights look
1. Archive built from EMA shadow (ema_drift_l2=23.87 = smoothed lag of chaotic live weights).
2. Checkpoint selection on `total` metric is HOLDING ep249.
3. proxy loss_pose (PoseNet-MSE, scale ~tens) != score d_pose (~3e-5 at frontier, sqrt(10*d_pose)).
   => proxy loss_pose=37 does NOT imply a bad exact d_pose. B2 is the only authority.

## Decision (evidence-gated; Forbidden premature KILL; B1 = clean PR95 zero novelty)
- DO NOT touch run / change loss weights on proxy instability alone.
- Read `harvest_b2_work_ep250/contest_auth_eval.json` when 600-pair CPU eval finishes (~60-120 min).
- apply_campaign_decision(receipt, candidate_eval, frontier=0.1919853363, hard_fail_checks=
  [("muon_active_before_stage_8", <telemetry.muon at ep249>)]) -> machine-readable verdict.
- THEN harvest ep500 + ep750 to build the stage-2 trajectory (answers "does ep-250 trend justify ep-3000").

## Localized hypothesis (only act if B2 trend worsens AND grad_norm keeps climbing)
Instability is STAGE-2-SPECIFIC (tau_softplus margin objective on the pose path), NOT baseline-wide.
PR95-faithful targeted fix candidates: tau warmup, intra-stage-2 pose-distillation-weight ramp, or
tighter stage-2 grad-clip. Never a baseline rewrite.

## ep250 FIRST EXACT SCORE (2026-06-09T09:11Z) — the arbiter spoke
`harvest_b2_work_ep250/contest_auth_eval.json` (600-pair `upstream/evaluate.py --device cpu` on the
ep249 EMA-best archive; 256,072 bytes; ~9 min CPU scoring). `[macOS-CPU advisory]` — NOT 1:1 with the
Linux-x86_64 [contest-CPU] frontier 0.19199; promotion_eligible=false.

| term | value | contribution | share |
|---|---|---|---|
| d_seg  | 0.50482  | 50.48 | 56.0% (BINDING) |
| d_pose | 155.75   | 39.47 | 43.8% |
| bytes  | 256,072  | 0.17  | 0.2% |
| **final_score** | | **90.12** | vs frontier 0.19199 (~470x) |

DECISION (recorded, `ep250_campaign_decision.json`): **INSPECT_BINDING_CONSTRAINT** (binding=seg),
reason=early_high_score_binding_is_seg, auto_kill=False, muon_active@ep249=False (correct, stage 1).

### CRITICAL RECONCILIATION (strict scrutiny): proxy loss_seg stable != d_seg good
My proxy read called the "SEG chamber healthy" because loss_seg was STABLE (~1.16, did not diverge
like the killed run's 18->400). The EXACT eval reveals d_seg=0.50 — half the SegNet argmax pixels are
WRONG. Both true: the distillation LOSS didn't diverge (stability), but the rendered frames are still
POOR by the actual scorer (quality). Proxy stability is necessary-not-sufficient; only exact d_seg is
authority (CLAUDE.md "proxy is a training signal, not a measurement"). DO NOT read loss_seg stability
as seg quality again.

## TREND now building autonomously (the operator's burning question)
Trajectory harvester launched detached (pid 57037) targeting ep[500,750,1000,1250,1500,1750,2000,
2250,2500,2750,3000], ONE eval at a time (no concurrent 600-pair evals). ep500 export started 09:21Z.
Writes `harvest_trajectory_summary.json` + per-target `hi_nerv_backend_only_ep<N>_exact_eval.json`.

THE DISAMBIGUATING QUESTION the trend answers:
- If d_seg/score DESCENDS at ep500/750 (stage 2->3->QAT): the curriculum is working; continue to ep3000.
- If d_seg/score is FLAT/RISING: ep250's d_seg=0.50 is a structural problem (stage-2 pose instability
  OR the score-aware distillation not driving frame fidelity), NOT just "early" -> intervene before ep3000.
ep250 alone is a START point, not a verdict (8-stage curriculum; quality stages all ahead of ep249).

NEXT INVOCATION: read `harvest_trajectory_summary.json` + the ep500/ep750 results; if descending,
continue + plan dual CPU(Linux x86_64)+CUDA(T4) authoritative eval on the best checkpoint; if flat,
the stage-2 fix is the targeted intervention (NOT a baseline rewrite; NOT a kill).
