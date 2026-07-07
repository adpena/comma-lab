# T5 CRUCIBLE — ORCHESTRATION LEDGER (durable; any session resumes from here)

Convened: 2026-07-07 (operator, verbatim in convening record §1).
Record: `.omx/research/T5_crucible_convening_arm_A_full_stack_20260707.md`
Grounding: `.omx/research/t5_crucible/GROUNDING_PACKET_20260707.md`
Dossier: `.omx/research/DRAFT_derived_optimal_next_run_for_council_20260707.md`

## Protocol phases
- [x] P0 grounding (never-fired 36 enumerated; 113 unmapped DSL flags; ledger-semantics caveat)
- [x] P1 independent positions (6 seats DONE) + 20-store compendium DONE + pursuit-chain-A in flight
- [ ] P2 **POSITIVE SYNTHESIS DRAFT — reordered per operator 2026-07-07 ("agents finding the
      obvious but not doing the positive work of proposing optimal")**: a CHIEF-DESIGNER pass
      that DRAFTS THE OPTIMAL FULL STACK from all six positions + the compendium. Contract
      (REFINED per operator 2026-07-07 "a value is not enough … ramp up or down or self deriving
      or dynamic or DE or fractional or partial"): **every knob gets a fully-specified CONTROL
      LAW**, one of — (a) a CONSTANT; (b) a RAMP/ANNEAL with derived shape + endpoints + a
      COMPLETION GUARANTEE (anti-M2: every anneal provably completes before its consumer fires,
      or its truncation is event-safe — denominators bound to events, never to a fixed clock);
      (c) a SELF-DERIVING/ADAPTIVE law (e.g. the DE-derived adaptive-ε #318/#320: the formula +
      its clamps, not a number); (d) an EVENT-CONDITIONED dynamic (costate-driven predicate +
      response, tested per requirement B); (e) a FRACTIONAL/PARTIAL application (per-class λ,
      margin-gated support, partial weights — the gate law specified). Each law carries its
      derivation (or anchor, or default-with-named-recess), its parameters pinned, and its
      stability/completion argument. "TBD", "gated with no default", and a bare constant where
      the physics demands a law are ALL forbidden in the draft. **EXTENSION (operator
      2026-07-07): the SCHEDULE and CURRICULUM as wholes obey the same contract** — stage
      sequence as a costate-choosable POLICY (stage DAG, #334/#339 levels-as-paths, D-5) with the
      fixed sequence only as the fail-safe default path; FRACTIONAL/BLENDED stage transitions
      (ramp-in/out laws with completion guarantees) where derivable, binary switches only where
      blending is measured harmful; PER-CLASS SUB-CURRICULA (lane/movable LADDER homotopies on
      their own laws, coupling guards specified); SELF-DERIVING schedule parameters (caps as laws
      of measured constants, e.g. finisher budget ≥ k·τ_e with τ_e re-estimated online). No bare
      fixed clock where an event/law is derivable. Output = the exact
      WitnessProgram + schedule + curriculum + costate config + rate plan + measurement plan,
      launchable as written. Proposing-optimal-with-tags is NOT a NO-FAKE violation — asserting
      it as MEASURED would be; the draft states its epistemic status per knob and pre-registers
      falsification. THEN:
- [ ] P3 red-team attacks THE DRAFT (not an abstraction) + MANDATORY PROVENANCE AUDIT —
      **MANDATORY FIRST PASS: PROVENANCE AUDIT** (operator directive 2026-07-07 "I shouldn't have
      had to catch that"): for EVERY load-bearing measured claim in EVERY position, verify
      {anchor path · review_status (pre-registered-only / recovery-written-UNREVIEWED /
      fresh-eyes-reviewed(N)) · form-limitations (oracle-vs-retrain, scoreable-rung count,
      truncated-anneal state, subset-vs-n600)}. Any claim resting on a recovery-written or
      unreviewed verdict → PROVISIONAL until its review clears. Run-config claims must cite
      launch.sh or the council design memo, never the activation ledger. Memory:
      verdict_review_status_metadata_operator_should_never_catch_provenance_20260707.
- [ ] P3b debate round(s) (≥2; designer defends/revises the draft against the attacks)
- [ ] P4 EMPIRICAL RECESS (measurable disputes → n600-real data; HVP-Lanczos first; serial, governed)
- [ ] P5 revised synthesis (the draft, amended by recess results; survives second red-team pass)
- [ ] P6 #363 self-reflection seal (assumption tags; PROVISIONAL where unmeasured)
- [ ] P7 deliverables 1-7 assembled (DSL WitnessProgram · ledger resolution · schedule ·
      costate · curriculum · measurement plan · wall-clock plan) — triality-consistent landing

## Seats (P1)
| Seat | Charter | Wave | Status | Output |
|---|---|---|---|---|
| S1 BASIS | Daubechies/Mallat — Arm A basis design (along-tangent bandwidth, bank ladder, activation, chroma) | 1 | DONE (7a052eed0) | position_S1_basis_20260707.md |
| S2 SCHEDULE+CURRICULUM | witness-native derivation; PR95 cargo-cult audit table; stages/exits/priming | 1 | DONE (f1e3e8b21) | position_S2_schedule_curriculum_20260707.md |
| S3 CONTROL/COSTATE | SENSE→DECIDE→ACT; GN/Fisher 2nd-order wiring; HVP-Lanczos probe design; trust-region | 1 | DONE (17843a820) | position_S3_costate_20260707.md |
| S4 RATE | Shannon — two-regime allocation, weight-entropy, #157 compress-half, derive-H, byte accounting | 2 | DONE (e4819f0eb) | position_S4_rate_20260707.md |
| S5 LEVER-LEDGER | Fridrich/Yousfi — 36 never-fired per-lever BUILD/DEFER; annulus geometry; lane-band; islands | 2 | DONE (f01c8dea6) | position_S5_lever_ledger_20260707.md |
| S6 POSE+BYTE-CLOSE | §5B pose ON/OFF; store-nothing carrier; byte-close + exact-eval path readiness; measurement-plan skeleton | 2 | DONE | position_S6_pose_byteclose_20260707.md |

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
E. **RATE-LEVER COMPLETENESS (operator-directed).** Every class of the rate arsenal considered —
   in-training (flat-minima #242, WeightEntropyPenaltyMLX, entropy-penalty, latent-structure,
   variable-grid QAT — ⚠ LAUNCH-BLOCKING run-1 decisions) · bit-depth (waterfill) · structural
   (low-rank/prune/TropNNC/KD) · invariance/orbit (permutation canonicalization, orbit coding) ·
   post-hoc coding (receipts required; base AT floor) · payload-specific (flip/band/pose/latent) ·
   scorer-invariance (#153 P-SUFF). Full taxonomy in the compendium. Per lever: FOLD /
   DEFER-with-reason / DEAD-with-receipt. Named RECESS: Class-D×Class-B interaction (waterfill an
   entropy-shaped ckpt vs unshaped ep650, $0).

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
- 2026-07-07T~21:0x S5 LEVER-LEDGER DONE (f01c8dea6): ISLANDS NECESSARY for T_3 (63.9% of flips
  = un-born movable+lane; big-3-only floors ~0.00215 > 0.00092 → island-first ranking).
  GROUND-TRUTH CORRECTIONS to the packet: ~10 of "36 never-fired" DID raw-flag fire per launch.sh
  — AnalyticLaneRenderBand fired ep300+ in the FULL-STACK #205 (20260702) run, UNATTRIBUTED, and
  is NOT near-zero-byte (+0.0206 LBND4 rate MEASURED); did NOT fire in mod32cap (by design).
  FIRE verdicts: islands-arm core, band trained-with, AACoverageRender, MuonWarmStart,
  LengthSigma, StepNative+FinerBiasInit, WeightEntropyPenaltyMLX. DEFER: UniWARD(chance),
  MarginSaliency, GFC(seam), lane_carried regime, SegFocalGamma(γ*=0 conflict), FiLM-family.
  RETIRE: MicroBatch(no-win), SoftBoundary. Apparatus fix R1: argv→lever reverse-map +
  engagement-predicate backfill.
- 2026-07-07T~21:1x OPERATOR DIRECTIVE (comprehensive signal sweep — "design cannot be optimal
  otherwise"): CONTEXT_COMPENDIUM_20260707.md skeleton written (14-store STANDING CHECKLIST +
  group-theory first-pass: orbit-coding = THE unifying rate principle, equation candidate
  rule118_orbit_coding_free_action_counted_coords_v1, 387B permutation slack, NO-equivariant-arch)
  + dedicated sweep agent LAUNCHED to complete all 14 stores. Compendium = MUST-READ for P2/P5.
  Class-fix: the STANDING STORE CHECKLIST makes context-completeness a P0 convening step forever.
- 2026-07-07T~21:3x S3 COSTATE DONE (17843a820) — RAN the pinned D-3/4/5 first measurement ($0
  inline, de-orphaned #341 SolveCtx.hvp_pair): ep650 EMA-best Ritz spectrum K=8 = [-369.7 ...
  +139.3], STRONGLY INDEFINITE (|λ₋|=2.65×λ_max, grad_norm 0.787) [macOS-CPU advisory,
  PROVISIONAL-K=8, honors truncated-anneal state]. Reading: best point NOT 2nd-order exhausted;
  TerminalSolve in-basin NOT met; cold Muon fire was curvature-blind. Full-P RECESS-R1
  pre-registered (kill band |λ₋|/λ_max<0.1 @K=128 → capacity/basis wall → strengthens Arm A
  either way). SENSE: +gn_spectrum.checkpoint_lanczos producer; DECIDE: curvature-aware
  exhaustion (1st-order meat ∧ PD-decrement ∧ no-usable-λ₋) + spectrum-rate mixture replaces
  linear-λ (the ep450-miss fix) + λ_max-sized trust region + POWERPLAY never-regress; ACT: NEW
  quadratic_basin ExitEvent (warm-Muon fire predicate) + margin-gated island support; advisory-
  only, CONTAINMENT unchanged. Harness landed: experiments/t5_s3_hvp_lanczos_probe.py.
  S6 POSE+BYTE-CLOSE launched (last seat; concurrency 3 with S4 + context-sweep).
- 2026-07-07T~22:1x S4 RATE DONE (e4819f0eb) — FIRST measured archive row for the clean
  baseline: 83,406 B → rate 0.05553 (ep650 EMA-best, real byte-close accounting). Base weights AT
  order-0 entropy floor (brotli 100.9% of H0 — zero coder slack; only bit-depth/waterfill or
  in-training entropy shaping move base rate; code stream 36% below H0 = structure remains). Lane
  band +24.1-30.9 KB (+0.016-0.021 rate — corroborates S5: NOT near-zero; net-ΔS positive
  mid/upper of FEED-07d ceiling, NOT guaranteed at conservative edge). Pose ξ +2.7 KB ≈ 1e-2
  S/byte = best buy (gated #248). mod32→48 = +20.2 KB. Uniform int6/5/4 = 54.2/41.3/29.0 KB;
  waterfill pre-registered [52,68] KB at Δd_seg ≤ +5e-5 (RECESS R1 = $0 probe). Free grammar rev2
  = −2.8 KB measured; PR95-L25 temporal-delta on code MEASURED NEGATIVE (+64%) = cargo-cult DROP
  with receipt. BUDGET TENSION: mod32+band+ξ = 110-117 KB > 105 KB sub-0.15 headroom → the
  compress-half is NOT optional if band+pose+capacity all compose. Board: 5/6 seats done; S6 +
  20-store sweep in flight.
- 2026-07-07T~22:4x CONTEXT SWEEP DONE (506a301bd; compendium 803 lines, all 20 stores). Top
  unpinned items now in: (1) CORRECTION to my group-theory first-pass — §C permutation slack is
  MEASURED NO (387 B was theoretical; best arm −8 B, most arms HURT +72/+251/+339 B) — the
  provenance machine caught MY error this time; (2) per-group-grad-clip ON in mod32cap but ABSENT
  in every stacked run = baseline confound for any stacked-vs-clean comparison; (3) 15 witness
  equations sit ASSUMED_AWAITING_VERIFICATION (both anisotropic-basis laws, step-native, adaptive-
  eps, Muon-finisher, S_R, chroma-at-annulus…) → reliance is PROVISIONAL per #363; (4) self-orient
  state NOT persisted in ckpts (+4.3% reconstruction gap — affects every ckpt-probe incl. chain-A);
  (5) never-reset-moments law + GradNorm-canary warning bound to costate ACT; (6) v_h=174
  MEASURED-OPTIMAL (lane-geometry closed); (7) durable-state files 3-8 weeks STALE (do not cite);
  (8) serializer absorption recurred TODAY — crucible committers must use --base-content-sha256
  (56fc4e19 resolution); (9) FEED-08g overturn: τ-crossover FLAT-H = instrumented-blind; (10)
  pose break-even error bars soft (0.018 borrowed-ancestor / 0.026 operator-stated) → §5B must
  not treat as measured. Chain-A pursuit notified of item (4).
- P2 chief-designer contract REFINED: knob → CONTROL LAW (constant | ramp+completion-guarantee |
  self-deriving/DE | event-conditioned | fractional/partial), per operator.
- 2026-07-07T~23:0x S6 DONE — §5B: pose ON two-track (diagnostics seg-only; pointer run
  w_pose>0 + FiLM + store-nothing ξ staged at tau boundary). POSE = THE SECOND WALL: pose-blind
  term ≈35.5 (measured 125.833 raw) unsubmittable; R1 floor 0.0011 → term 0.105 misses T_1-feasible
  d_pose ≤ 1.51e-4 by ~7× — perfect d_seg CANNOT cross 0.19110 unless the unfired L3 mechanism
  beats it. Kill: converged d_pose > 1.5e-4 → L1 Jacobian fallback. FIRST byte-close rows (bit-exact
  ×3): weights 83,430 B / 0.0556 (cross-checks S4's 83,406 within 24 B); band LBND2 +41,562 B /
  +0.02767 (CONFLICT vs S4 LBND4 24-31KB — designer decides); pose +6,929 B / +0.00464 (derive-H
  live, H_bytes=0). Gaps G1-G5 incl. G1 silent freq_along tool confound + G3 AA-decode shipping
  blocker + G4 exact-eval never fired e2e. Inflate budget: 13.9 min measured, 15-18 projected.
  **P1 COMPLETE. P2 CHIEF-DESIGNER LAUNCHED** (control-law contract + schedule/curriculum-as-laws
  extension delivered mid-flight). In flight: designer + pursuit-chain-A.
