# T5 CRUCIBLE — ORCHESTRATION LEDGER (durable; any session resumes from here)

Convened: 2026-07-07 (operator, verbatim in convening record §1).
Record: `.omx/research/T5_crucible_convening_arm_A_full_stack_20260707.md`
Grounding: `.omx/research/t5_crucible/GROUNDING_PACKET_20260707.md`
Dossier: `.omx/research/DRAFT_derived_optimal_next_run_for_council_20260707.md`

## Protocol phases
- [x] P0 grounding (never-fired 36 enumerated; 113 unmapped DSL flags; ledger-semantics caveat)
- [ ] P1 independent positions (6 seats, anti-anchored, STAGGERED 2 waves of 3 for rate limits)
- [ ] P2 red-team pre-mortem (reads all positions; "how this fails to move the pointer")
- [ ] P3 debate round(s) (≥2; disagreements enumerated)
- [ ] P4 EMPIRICAL RECESS (measurable disputes → n600-real data; HVP-Lanczos first; serial, governed)
- [ ] P5 synthesis (one stack; survives second red-team pass)
- [ ] P6 #363 self-reflection seal (assumption tags; PROVISIONAL where unmeasured)
- [ ] P7 deliverables 1-7 assembled (DSL WitnessProgram · ledger resolution · schedule ·
      costate · curriculum · measurement plan · wall-clock plan) — triality-consistent landing

## Seats (P1)
| Seat | Charter | Wave | Status | Output |
|---|---|---|---|---|
| S1 BASIS | Daubechies/Mallat — Arm A basis design (along-tangent bandwidth, bank ladder, activation, chroma) | 1 | DONE (7a052eed0) | position_S1_basis_20260707.md |
| S2 SCHEDULE+CURRICULUM | witness-native derivation; PR95 cargo-cult audit table; stages/exits/priming | 1 | LAUNCHED | position_S2_schedule_curriculum_20260707.md |
| S3 CONTROL/COSTATE | SENSE→DECIDE→ACT; GN/Fisher 2nd-order wiring; HVP-Lanczos probe design; trust-region | 1 | LAUNCHED | position_S3_costate_20260707.md |
| S4 RATE | Shannon — two-regime allocation, weight-entropy, #157 compress-half, derive-H, byte accounting | 2 | QUEUED | position_S4_rate_20260707.md |
| S5 LEVER-LEDGER | Fridrich/Yousfi — 36 never-fired per-lever BUILD/DEFER; annulus geometry; lane-band; islands | 2 | LAUNCHED | position_S5_lever_ledger_20260707.md |
| S6 POSE+BYTE-CLOSE | §5B pose ON/OFF; store-nothing carrier; byte-close + exact-eval path readiness; measurement-plan skeleton | 2 | QUEUED | position_S6_pose_byteclose_20260707.md |

## Rate-limit resilience rules
- Waves of ≤3 concurrent seats; wave 2 launches when wave 1 returns (or one at a time if limits bite).
- All context via durable files + pointers (no giant inline prompts).
- Seats checkpoint positions incrementally → a killed seat is resumed by relaunching with
  "continue position_S<N> from its current on-disk state".
- Every phase output committed immediately (serializer, [no-triality]).

## Recess queue (P4; grows from positions + red-team)
1. $0 HVP-Lanczos GN/Hessian spectrum on saved #205 checkpoint (pinned §2 first-measurement).
2. **FEED-08l FRESH-EYES ADVERSARIAL REVIEW (operator-flagged 2026-07-07).** The freq_along ladder
   probe was pre-registered (721c764fd) with the 4db610af2-inherited GT-validity controls, BUT the
   verdict was written by post-credit-death RECOVERY, never fresh-eyes reviewed; only 2 rungs
   scoreable (0, 8 — 16/25/32 indeterminate); ORACLE-form only (form-a retrain UNSUPPORTED,
   explicitly a T5 candidate). Review the durable JSON
   (experiments/results/freq_along_ladder_probe_20260707/) + the verdict's inferences.
   → Until it survives: S1's lane_carried DEMOTION is PROVISIONAL (#363 tag), and the comb-favored
   claim is additionally gated on item 3.
3. **Comb-REGISTRATION audit (L65 owed).** "cCOMB best 0.00695" is load-bearing for both FEED-08l's
   comparative verdict and S1's dash-carrier allocation; mis-phase risks scoring against real lane.
   Audit before any comb verdict is load-bearing.
4. **form-a retrain arm decision** (freq_along raised AT TRAINING, not oracle-injected) — the
   in-training question FEED-08l could not answer; fold into the S1 basis A/B design (R2).
(…seats append here via their RECESS proposals…)

## Log
- 2026-07-07T~18:00 convening record committed (294fb8119).
- 2026-07-07T~19:0x session cut (rate limits) before seat launch; recovery confirmed ZERO signal
  loss (no seats had launched). P0 re-grounded; wave 1 launching.
- 2026-07-07T~19:5x GROUNDING CORRECTION (operator catch): mod32cap = council-designed CLEAN
  BASELINE per council_symposium_clean_config_20260705.md (explicitly excluded seeding/lane-prior/
  analytic-lane-band/island-birth). Packet fixed + pushed to S1/S2/S3 mid-flight; memory sharpened.
- 2026-07-07T~20:0x S1 BASIS DONE (7a052eed0): basis role NARROWS (band+comb carry lane/dash; basis
  carries C2-cartoon separatrix); lane_offloaded(32) primary, lane_carried DEMOTED (FEED-08l);
  Nyquist-clean across=8 arm; hosc anneal kept; ChromaBoundarySharpen stub-fold; R1-R6 recess items;
  seam conflicts declared (GFC×self-orient fail-close, AA×band/seed incompatible). S5 LEVER-LEDGER
  launched (keeps concurrency at 3).
