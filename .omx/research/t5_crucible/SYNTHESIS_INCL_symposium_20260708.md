# T3 INCLUSION SYMPOSIUM — SYNTHESIS (2026-07-08) — per-item classes, arbitrations, v7.3 deltas

STORES CONSULTED: all five position files (S1 2805d8fd6 · S2 0bf175cd4 · S3 54680a7b2 ·
S4 153d196f1 · S5 80b0833d4) · CONVENING doc + docket status update · ORCHESTRATION_LEDGER ·
run-1 launch.sh/run.log (live argv + cadence) · contest_legal_inflate_20260705.md ·
r7_finishers/resume_registry/d15/d16/#330 landing memos. council_tier: T3 · quorum 5/5 seats ·
review_status: synthesis by main orchestrator, fresh-eyes on all five positions.
Pointer 0.19110 [contest-CPU] UNMOVED — everything here is MEANS.

## TALLY: 5/5 positions delivered; S2 CERTIFIES the composed set (conditions below);
## S5 veto WITHHELD pending engagement of 3 hypotheses — ENGAGED in §ARBITRATIONS.

## ORCHESTRATOR ERROR ACKNOWLEDGED (docket D16 premise)
The docket status update claimed "D16's consuming loss term is default-off in v7" — FALSIFIED
(verdict_scope: instance — that claim) independently by S3 (witness_autoconfig:1069/1457) and
S5 (live argv --persistence-loss-weight 1.0 + per-step persistence:0.37) and S4 (D-1 LOUD).
S2's REGISTERED call on item 5 inherited the false premise and is DISCOUNTED for that item.

## FINAL CLASSES (arbitrated)
| # | Item | Class | Basis |
|---|---|---|---|
| 1 | Basis rebalance | **IN-v7** | operator-decided; 793631e00; unanimous |
| 2 | Event mode | **DECIDED-EVENT** | operator-decided; S2 certifies stagger-invariant survives (DSL cap-check + nucleation gate) |
| 3 | Micro-batch | **v7.1-ARM, bounded A/B DURING v7** | 5/5 ARM; S5-H1 engaged below |
| 4 | Safe-compile hosc | **v7.1-ARM** | S4 D-2: flip INADMISSIBLE at launch (GPU manifest fingerprint-absent; CPU hosc affirmatively non-bit-identical 6e-8; the failing 1/9 IS the lever); gate = stop-time GPU re-cert |
| 5 | D16 pool | **IN-v7 (dispatch ON)** | premise corrected: term ACTIVE at w=1.0; bit-identity max|Δ|=0 incl. FULL-LOSS flag-on-vs-off on REAL n600 GT (=S3's named parity check, ALREADY MEASURED by the D16 landing); N=5 cross-process; measured 2.3–3.9× on a live hot term — same class as fused-R |
| 6 | #330 verdict reclaim | **v7.1-ARM (default-OFF at launch)** | 2 IN vs 3 not-in; the artifact's own default-OFF-until-n600 rec (S4 D-3) + S2 COND-4 + the ratchet is NOT live in run-1 (RSS stable over 3 verdicts — the urgency premise is stale); named trigger = verdict-correlated RSS ratchet in v7 telemetry; crash-composition seam (S5-A3) owed before load-bearing |
| 7 | Adaptive-ε | **REGISTERED** | unanimous; trigger = eikonal re-entry signature |
| 8 | R-7 | **SPLIT: Polyak IN-v7 (start_epoch SIZED to TAIL window, derived-at-config); β2-rewarmup v7.1-ARM** | Polyak observe is read-only/EMA-untouched/fail-open → score-neutral extra ckpt candidate, free byte-close arm; sizing per S1/S2/S5 flag; rewarmup trajectory-affecting + law INFERRED |
| 9 | Resume registry | **IN-v7 — BINDING PRECONDITION of event mode (S2 COND-1), CONTINGENT on #358 landing** | unanimous; #358 builder in flight |
| 10 | GPU-verdict | **REGISTERED** | unanimous; D1 stop-time agreement probe |
| 11 | fp16 cf-feats | **REGISTERED** | unanimous; not needed (37.26 GiB headroom post-basis, S1); re-waterfill gate if ever armed |

## ARBITRATIONS — S5's veto hypotheses ENGAGED
**H1 (micro-batch circular baseline):** partially UPHELD. Resolution is NOT
baseline-completion-gated: v7's OWN first ~300 ep = arm A; arm B (twin config, micro-batch ON,
same seed, ep0–300) fires ~day 1, admission-gated on (a) v7's measured uncontended n600 RSS
curve by ~ep100 (the waterfill input it is honestly missing) and (b) the governor's 2-job
envelope. NOT pre-launch: delaying the operator-decided leap ~1.5d to A/B a lever is dominated,
and single-GPU contention pre-launch would corrupt both. This names the bounded A/B S5 demanded.
**H2 (3.1 anchor optimistic):** UPHELD. Live incl-startup cadence 3.62 min/ep → 7.55d > sealed
7.427d. v7.3 DELTA: re-derive wall_clock_budget_days from the LIVE cadence (admission-bench
value at launch, floor 3.62) — the anchor's provenance flips to config-conditional MEASURED;
rc=8's real bench remains the final arbiter at admission.
**H3 (D16 premise):** UPHELD in full — class corrected to IN-v7 (table above).
**A3 (composition bit-identity):** engaged — #330 stays OFF at launch; #358's crash-resume
tests + S2's source-verified disjoint-tmpfile/killpg analysis cover the seam before any arm.
**4th class (IN-v7-with-bounded-auto-revert):** acknowledged as ontology addition for FUTURE
dockets; not needed for this one (D16 admits via bit-identity; safe-compile lacks its GPU cert
so even auto-revert cannot admit it at launch).

## v7.3 COMPILE DELTAS (from this synthesis)
1. D16 persistence-pool dispatch flag ON (bit-identity + real-n600 full-loss parity already measured).
2. PolyakFinisher lever composed with start_epoch = TAIL-window start (DERIVED-AT-CONFIG; never default 0).
3. wall_clock_budget_days re-derived from live cadence (≥3.62 min/ep; admission bench final).
4. NEW-1 epochs (config-sealed 3000) — already landed; verify at compile.
5. Items 3/4/6 default-OFF verbatim with their named triggers stamped in the activation ledger.
PRECONDITIONS: #358 lands (item 9 contingency + S2 COND-1) → then v7.3 compile → seal ROUND 2
(4 lenses, fix-all-severities, zero-unfixed precondition) → 3 clean passes → knee re-derive (D2)
→ governed stop of run-1 (checkpoints preserved) → EVENT-mode launch through the full gate chain.
