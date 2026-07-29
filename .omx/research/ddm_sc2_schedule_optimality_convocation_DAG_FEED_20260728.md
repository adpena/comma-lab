# DAG FEED — ddm_sc2 schedule-optimality convocation (2026-07-29)

Pointer: **0.1910828242 [contest-CPU] UNMOVED**. Apparatus unit (frontier_protecting); no score.
Parent memo: `.omx/research/ddm_sc2_schedule_optimality_convocation_20260728.md`.

## FEED-sc2-a — verdict node
tr1 burn schedule (v2 ticket d820f9dd…) adjudicated **PROCEED_WITH_REVISIONS**: structurally the
derived event-gated shape (14/20 elements DERIVED/MEASURED; 4 PROVISIONAL-labeled with rederivation
triggers); 3 named gaps, all routed FIRST-EXTENSION-WINDOW, none verdict-changing; burn fired
2026-07-29T04:54:08Z before convocation close ⇒ NO PRE-FIRE-FOLD (per the ticket's pre-registered
sc2_soft_gate). Live burn clean at ep19+ (COUPLED_DESCENT, zero alarms).

## FEED-sc2-b — the three findings (edges into the extension reseal)
- **F3 (top, MEASURED 9-15% epoch-budget leak):** half-applied #518 — Adam re-anchored fresh at every
  ratchet window; measured +114% same-epoch transient, 3-5 ep recovery, ×~11 boundaries. Mechanism
  source-verified: installed MLX Adam defaults `bias_correction=False` (raw `m/(sqrt(v)+eps)`).
  Cures (compose): persist opt state for SAME-config ratchets (schema already supports it) +
  `bias_correction=True` (one constructor arg) + lr-rewarmup per `adam_v_variance_warmup_length_v1`
  (#518 item 2; L(c=2)=2,000 updates ≈ 27 ep at β₂=0.999) at geometry changes. tb1 forces row 8 had
  self-recorded this as OWED at T3.
- **F2:** knee predicate `rel < 0.01` fires on RISING loss; the ONLY measured knee fire (T1-plain
  ep7) was that misfire ("off a transient CE rise", tb1 memo). Cure = `0 <= rel` guard + alarm;
  threshold 0.01 gets the basin-threshold provenance pattern. Burn projection: legit knee ~ep25-60,
  before F2@200 — first-ever legitimate event fire expected in window 1.
- **F1:** EMA clamp [0.9, 0.9995] silently binds at the TOP at burn scale (law 0.999867 → executed
  0.9995) and the provenance string records only the law. Behaviorally BENIGN-to-GOOD (shadow warm
  ep53 vs law-intent ep200 — basin instrument needs the early warm); cure = derive W from gate
  cadence explicitly + record clamp events in provenance.

## FEED-sc2-c — design questions closed
- **Form ladder COMPLETE:** CE→tau→(basin)→SOLVE; no third smooth leg (grounds: #459 satisficing,
  fd2 decoupling, #341 K=8 early-solve +5.1% n600 worsening; v17 ρ semantics). margin_hinge stays a
  raced START form. Contrarian fold ADOPTED: E2 quotes the solve from `stage_*_final.npz` even if
  basin never fires.
- **co7 tension adjudicated:** trainer's stop-first-quote-second deviates from `basin_solve_handoff_v1`
  never_auto_stop — RECONCILED by operator ×2 directive + saddle-resume rule; residual obligation =
  prompt solve attempt on receipt (E2/MAIN watch).
- **Rebalance:** as sealed (stage-boundary raced windows, ≥3× PROVISIONAL trigger, #312 honored;
  ν=0.0 composed). First marginal pair computable $0 at the knee boundary from existing telemetry.

## FEED-sc2-d — D5 optimizer story (per-step scale = the last INHERITED controller)
β₂ default 0.999 INSIDE the derived band [0.9967, 0.9992] at 75 steps/ep (window 13.3 ep); β₁<√β₂
guard passes everywhere in band. MLX Adam has NO bias correction → the F3 cold-v spike mechanism.
Per-type race designed (tokens/renderer-or-scores/masks; 40-ep warm-started matched-RMS windows,
px1 contract, pools-law winner-take-all). Original variant: scorer-metric preconditioning in the
ANALYTIC Fisher we own (ms3/ms4 rank-4 + margin≈Fisher 0.978) — rung 1 is the EXISTING
`margin_weighted` flag (default-off queue), rung 2 head-subspace projection, rung 3 IS the solve.
NOT redundant with fd1's terminal GN: preconditioning acts outside the solve validity radius
(#341/v17), buying conditioning speed toward the basin.

## FEED-sc2-e — corrections to standing records
Ticket `schedule_facts` gate estimate "~1.8h" is ~15× above measured (~10 s/gate → ~7 min + full-
confirm); charter ground-truth referenced the v1 ticket (v2 reseal supersedes: basin-handoff ON,
solve_project ADOPTED −28.9%); detector is the §16.1 3-gate predicate, NCDE remains shadow-only;
β₂ law registered under #518 item 2 (not "#222"); MD-Decoupling = #195/DAG-B2 (not "#175").
Optimizer-corpus verdict tokens custodied in memo §4.2: px1 "ADOPT-MEASUREMENT-CONTRACT;
DO-NOT-ADOPT-OPTIMIZER-CONSTANTS" + SOAP "DO NOT ADOPT NOW; FAMILY OPEN"; #552 "spec-only" /
#556 "pending, must not be described as live"; #496 rate-lever "REJECT (formulation-scoped)";
#469 "GO-BUILD-POLAR-CHART; NOT FIREABLE" (witness-FiLM-specific); Tilde "KEEP plain Muon;
Aurora reduces to Muon (max diff 0.0)"; Muon ancestor MUON_BITES_FROM_STAGE4 −32% d_seg gap
[contest-CPU advisory] = mechanism-hypothesis only (L18).
