# T5 CRUCIBLE — ORCHESTRATION LEDGER (durable; any session resumes from here)

Convened: 2026-07-07 (operator, verbatim in convening record §1).
Record: `.omx/research/T5_crucible_convening_arm_A_full_stack_20260707.md`
Grounding: `.omx/research/t5_crucible/GROUNDING_PACKET_20260707.md`
Dossier: `.omx/research/DRAFT_derived_optimal_next_run_for_council_20260707.md`

## Protocol phases
- [x] P0 grounding (never-fired 36 enumerated; 113 unmapped DSL flags; ledger-semantics caveat)
- [ ] P1 independent positions (6 seats, anti-anchored, STAGGERED 2 waves of 3 for rate limits)
- [ ] P2 red-team pre-mortem (reads all positions; "how this fails to move the pointer") —
      **MANDATORY FIRST PASS: PROVENANCE AUDIT** (operator directive 2026-07-07 "I shouldn't have
      had to catch that"): for EVERY load-bearing measured claim in EVERY position, verify
      {anchor path · review_status (pre-registered-only / recovery-written-UNREVIEWED /
      fresh-eyes-reviewed(N)) · form-limitations (oracle-vs-retrain, scoreable-rung count,
      truncated-anneal state, subset-vs-n600)}. Any claim resting on a recovery-written or
      unreviewed verdict → PROVISIONAL until its review clears. Run-config claims must cite
      launch.sh or the council design memo, never the activation ledger. Memory:
      verdict_review_status_metadata_operator_should_never_catch_provenance_20260707.
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
| S2 SCHEDULE+CURRICULUM | witness-native derivation; PR95 cargo-cult audit table; stages/exits/priming | 1 | DONE (f1e3e8b21) | position_S2_schedule_curriculum_20260707.md |
| S3 CONTROL/COSTATE | SENSE→DECIDE→ACT; GN/Fisher 2nd-order wiring; HVP-Lanczos probe design; trust-region | 1 | LAUNCHED | position_S3_costate_20260707.md |
| S4 RATE | Shannon — two-regime allocation, weight-entropy, #157 compress-half, derive-H, byte accounting | 2 | LAUNCHED | position_S4_rate_20260707.md |
| S5 LEVER-LEDGER | Fridrich/Yousfi — 36 never-fired per-lever BUILD/DEFER; annulus geometry; lane-band; islands | 2 | LAUNCHED | position_S5_lever_ledger_20260707.md |
| S6 POSE+BYTE-CLOSE | §5B pose ON/OFF; store-nothing carrier; byte-close + exact-eval path readiness; measurement-plan skeleton | 2 | QUEUED | position_S6_pose_byteclose_20260707.md |

## Rate-limit resilience rules
- Waves of ≤3 concurrent seats; wave 2 launches when wave 1 returns (or one at a time if limits bite).
- All context via durable files + pointers (no giant inline prompts).
- Seats checkpoint positions incrementally → a killed seat is resumed by relaunching with
  "continue position_S<N> from its current on-disk state".
- Every phase output committed immediately (serializer, [no-triality]).

## SYNTHESIS HARD REQUIREMENTS (operator-directed 2026-07-07; the stack fails synthesis without these)
A. **SOLVE-WHERE-SOLVABLE, from the basin.** The final stack includes a per-block solve schedule:
   (i) TerminalSolve from the quadratic basin — GN/CG solve replacing late-stage training, gated on
   the S2 recess GN/CG-from-ep650 probe (bar: beat 0.0033662) + the S3 in-basin spectrum detector;
   (ii) the FULL #342 solve-don't-train inventory folded — every block solvable by
   linear/quadratic/KKT/closed-form is SOLVED not trained, with where/when/conditions stated.
   Training is reserved for what is provably not solvable.
B. **HARDENED EVENT DETECTION, TESTED END-TO-END.** Every event trigger in the derived schedule
   (CE→tau hand-off, per-class meat exits, plateau, anneal-complete precondition, finisher
   regression guard) ships ONLY with all three: (1) BACKTEST against the real mod32cap log
   (anti-anchor: the #315 trigger that backtested near-vacuous — fires ep251 mid-descent,
   reorient-50 confound); (2) INJECTION TEST through the LIVE trainer code path (synthetic
   scenarios: fires-when-it-should + stays-silent-when-it-shouldn't, exercising the actual
   witness_control wiring, not a unit stub); (3) FAIL-SAFE epoch CAPS so a dead/vacuous trigger
   degrades to the capped schedule, never to an unbounded or truncated run (anti-anchor: the M2
   anneal-truncation defect). A trigger without all three is NOT in the launch config.
C. **LADDER curriculum techniques (operator-directed).** The curriculum's difficulty axis is a
   LADDER: easier surrogate targets annealed to real targets under per-class-λ gates — the PROVEN
   mapping (L56/#322): LADDER ⊂ our costate (LADDER = 1-channel/const-λ special case; ours
   generalizes to per-class λ). Concretely: island-birth via the #323 per-class-λ-gated homotopy
   (movable = dilation-GO; lane = curve-prior), composed with the treatment-arm memo's
   margin-GATED support (net-positive iff n_isl > n_big3). Synthesis verifies S2's derived
   schedule carries the island-birth/LADDER entry points explicitly (charter item; S2's summary
   did not surface it — red-team gap-check).
D. **POWERPLAY campaign-meta (operator-directed).** The costate DECIDE layer + the crucible's own
   measurement plan honor the registered `powerplay_variant_ii_cost_isomorphism_v1`
   (`tac.witness_dsl.powerplay` → `campaign`, BUILT): the next task/probe/stage is the CHEAPEST
   NEW UNSOLVED one whose solution provably extends the stack (a measured ΔS, a lever verdict, or
   a capability the prior stack lacked). Deliverable 6's probe ordering is PowerPlay-ordered
   (cheapest-decisive first) and the duty-to-measure queue ranking is PowerPlay-consistent.

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
5. **SYNTHESIS ORDERING GUARD (operator-flagged, the wrong-dimension lesson).** S1's primary arm
   drops freq_along 8→6 on the premise that band+comb carry the along-tangent dash energy. That
   premise is UNPROVEN until (a) the comb-registration audit passes (item 3) and (b) the band
   survives through-R at n600. **Constraint: no along<8 arm may be the launch primary until BOTH
   clear; else arm (a) (control along=8) is primary.** The along-tangent axis is the measured
   starved dimension (FEED-03t, 3.2×) — never re-starve it on an ungated premise. Also: the
   √-at-base-only anisotropy approximation is "acceptable per FEED-08l" and therefore inherits
   item 2's review dependency.
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
- 2026-07-07T~20:3x S2 SCHEDULE+CURRICULUM DONE (f1e3e8b21) — 5 NEW $0 MEASURED rows on the clean
  baseline: M1 Muon fired COLD ep726 → +27.5% quench, never re-beat ep650; M2 **ANNEAL-TRUNCATION
  DEFECT** — Muon freeze truncated β at 3.177/4.00 + τ at 0.216/0.05 (denominators ep1000 vs freeze
  ep726) → the control's 0.0033662 best sat on an INCOMPLETE anneal; M3 #315 plateau trigger
  backtests near-vacuous on ep_loss (reorient-50 confound); M4 τ-stage ran 76-125ep past meat
  exhaustion; M5 finisher failure = TRANSIENT×BUDGET not paradigm (τ_e=305ep > 274ep budget).
  Derived: event exits w/ epoch CAPS; anneal-complete = finisher-fire PRECONDITION;
  --anneal-epochs=Muon-cap is the $0 flag fix; 34-row PR95 cargo table; exits alone save ~35%
  wall-clock at zero score cost. M2 forwarded to S3 (spectrum ckpts are truncated-anneal states;
  ep650 = primary target, ep1000 = cold-quench artifact). S4 RATE launched (concurrency 3).
