# T5 CRUCIBLE — ORCHESTRATION LEDGER (durable; any session resumes from here)

Convened: 2026-07-07 (operator, verbatim in convening record §1).
Record: `.omx/research/T5_crucible_convening_arm_A_full_stack_20260707.md`
Grounding: `.omx/research/t5_crucible/GROUNDING_PACKET_20260707.md`
Dossier: `.omx/research/DRAFT_derived_optimal_next_run_for_council_20260707.md`

## Protocol phases
- [x] P0 grounding (never-fired 36 enumerated; 113 unmapped DSL flags; ledger-semantics caveat)
- [x] P1 independent positions (6 seats DONE) + 20-store compendium DONE + pursuit-chain-A in flight
- [x] P2 **POSITIVE SYNTHESIS DRAFT — reordered per operator 2026-07-07 ("agents finding the
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
- [x] P3 red-team attacks THE DRAFT (not an abstraction) + MANDATORY PROVENANCE AUDIT —
      **MANDATORY FIRST PASS: PROVENANCE AUDIT** (operator directive 2026-07-07 "I shouldn't have
      had to catch that"): for EVERY load-bearing measured claim in EVERY position, verify
      {anchor path · review_status (pre-registered-only / recovery-written-UNREVIEWED /
      fresh-eyes-reviewed(N)) · form-limitations (oracle-vs-retrain, scoreable-rung count,
      truncated-anneal state, subset-vs-n600)}. Any claim resting on a recovery-written or
      unreviewed verdict → PROVISIONAL until its review clears. Run-config claims must cite
      launch.sh or the council design memo, never the activation ledger. Memory:
      verdict_review_status_metadata_operator_should_never_catch_provenance_20260707.
- [x] P3b debate round(s) — v2 REVISION landed (305d884ce); second red-team verify pass launched
- [ ] P4 EMPIRICAL RECESS (measurable disputes → n600-real data; HVP-Lanczos first; serial, governed)
- [~] P5 second red-team DONE: SEAL-TO-RECESS (3c707e2f5) — zero FAILS, 6 amendment-grade PARTIALs bind at recess close; revised synthesis follows recess
- [ ] P6 **RECURSIVE SEAL (operator 2026-07-07: "recursive adversarial review and deep math pass
      like normal")**: TWO LENSES per round — (i) the finding-disposition VERIFY pass + (ii) the
      DEEP-MATH MEAT HUNT (re-derive every law · hunt unconsumed levers vs the compendium · bug
      hunt the config-as-written · fresh-math sweep). Findings at either lens → revision → NEW
      round, counter RESETS. **SEAL = 3 consecutive clean rounds across BOTH lenses** (per the
      canonical 3-clean-pass protocol + tac.review_counter). Then the #363 self-reflection seal
      (assumption tags; PROVISIONAL where unmeasured). Round 1: verify pass + meat hunt BOTH in
      flight on v2. Also pinned to both: the lane-anisotropy scope guard (u_min-isotropic is
      lane-BLIND — measured at a lane-less checkpoint; must not be cited against anisotropic
      allocation levers).
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

## AUTONOMOUS CONTINUATION RULE (operator 2026-07-07: "requiring me to have an insane memory")
The system holds the thread — NOT the operator, NOT a session's memory:
- The costate digest auto-surfaces this ledger at EVERY SessionStart (tools/costate_digest.py
  section_active_convening) — phase state + next action + last log line.
- The MAIN LOOP advances phases on every agent landing WITHOUT being asked: landing → fold into
  ledger → launch the next written action. Current standing queue: (1) on next free slot →
  launch the #346 retrieval-layer build (contract clauses + corpus_query + convene.py) alongside
  the requirement F/G build wave; (2) on draft landing → P3 red-team vs the draft (provenance
  audit charter already written above); (3) on chain-A landing → fold + recess queue update;
  (4) on integration-tranche-1 landing → verify drift detector, then task/DSL/costate legs with
  the build wave. Any session resuming cold executes from THIS list — the ledger IS the memory.
- NOTHING here waits for operator prompting except the ONE GO gate (heavy pointer-run launch).

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
P. **SIGNAL-COMPLETENESS TO THE THEORETICAL LIMIT (operator 2026-07-07: "it must generate and
   record and expose all signal necessary for extreme optimization and realization to the
   theoretical limit").** The instrument framing (N) made TOTAL: the stack is signal-complete
   only when, for EVERY term separating current S from S_floor = 0.118, there is a NAMED signal
   that localizes the residual gap, and run-1 GENERATES it (run conditions produce it), RECORDS
   it (telemetry row / checkpoint / artifact, durable), and EXPOSES it (queryable by corpus_query
   + consumed by a named DECIDE-layer consumer: costate controller, response surface, DSL lever
   default, campaign ILC error term). Binding consequences:
   (1) v4 carries a **SIGNAL-COMPLETENESS LEDGER** section: gap-term decomposition {d_seg
   per-class/per-stage/annulus/along-across spectrum · d_pose witness-side (UNMEASURED — the row
   itself is a required signal) · rate per-section vs entropy floor · decode-gap g_dec
   per-stage-boundary · schedule (anneal truncation, meat exits, TAIL_k per-cycle yield) ·
   capacity (response-surface seeds) · basis (orientation quality, freq_along utilization)} —
   each row: signal name → generated? → recorded-where → exposed-to-whom. A gap term with NO
   signal row = a design FINDING (blocks seal), not a footnote.
   (2) This is the ILC OBSERVABILITY condition made explicit: the campaign converges
   (‖I−LP‖<1, CT-1) only if the per-run measurement set makes the gap IDENTIFIABLE — signal
   completeness is not hygiene, it is the campaign's convergence precondition.
   (3) No write-only telemetry: every recorded signal names its consumer at design time
   (sister of "Results must become system intelligence" + the default-off ledger); a signal with
   no consumer is queued signal-loss.
O. **FIELDS-MEDAL-GRADE CONTROL-THEORY DEEP RESEARCH (operator 2026-07-07: "deep research fields
   medal grade against control theory and our level set witness and Morse-Smale and full stack
   and meta and meta meta all aspects all dimensions all schedule and curriculum and controls and
   controller and DSL and everything").** Two research seats fired (CT-1 training/campaign optimal
   control · CT-2 PDE/geometric/topological control of the level-set object), positive-design
   contract per requirement + L82 (derived LAWS in our notation with values, not surveys), corpus
   retrieval-first, WebSearch authority. Deliverables feed v4 + the campaign layer + the costate
   controller's mature form (λ̇ = adjoint of the score functional, per M).

S. **CITATION PROVENANCE — record the literature (operator 2026-07-08: "make sure we are
   recording arxiv paper citations and stuff too for scientific rigor and provenance and the
   marimo contest if we get to it").** Every research-derived law/import/theorem-claim records
   its resolvable citation (authors · year · exact title · arXiv ID or DOI) at the point of
   derivation. MEASURED GAP at binding: CT-2 = 11 citations, CT-1 = 0 (Pontryagin/Tabuada/
   Heemels/Liberzon/Feldbaum/ILC all cited by name only), v5 = 1. Remedies fired: (1) crucible
   BIBLIOGRAPHY file (claim → citation → which v5 decision consumes it), backfilling CT-1 fully;
   (2) CITATION clause added to tac.subagent_contract so every future research charter carries
   the requirement structurally; (3) each bibliography entry tagged for marimo-contest
   candidacy (#347, ⚠ deadline Jul 9 11:59PM PST — implement-a-paper: the bibliography IS the
   candidate list). Citations are provenance for claims exactly as anchors are for
   measurements — an uncited imported theorem is the literature-side analog of an unanchored
   verdict (req R's FAMILY-level kills especially: those REQUIRE a citable theorem).

R. **VERDICT-SCOPE TAXONOMY — one failed formulation is NOT a dead family (operator 2026-07-08:
   "ensure no falsifications or negative interpretations made based on naive or toy or binary;
   many techniques have multiple ways of formulation and one failure does not mean family
   dead").** Extends requirement L (scale-relativity) with the FORMULATION axis and CLAUDE.md's
   paradigm-vs-implementation (Catalog #307) with the intermediate level. Every negative verdict
   MUST name its scope on the 4-level ladder and defaults to the NARROWEST level the measurement
   supports: (1) INSTANCE (this config/checkpoint/constant) · (2) FORMULATION (this mathematical
   form of the technique — e.g. K≤64 max-plus annulus fit; pooled-unsigned UniWARD; fixed-β
   hosc) · (3) FAMILY (all known formulations — requires EITHER a theorem/impossibility bound
   OR kills across ≥2 structurally distinct formulations at adequate scale) · (4) PARADIGM
   (operator + council consensus only, per the premature-KILL non-negotiable). Binding rules:
   (a) a binary pass/kill gate decides the DISPOSITION of one formulation (ships/stays-gated) —
   it is NEVER evidence about the family; (b) at kill time the verdict memo ENUMERATES the known
   alternative formulations not yet tested (the reformulation queue — e.g. max-plus: different
   K, different basis elements, log-sum-exp relaxation, per-class vs global; UniWARD asymmetry:
   other cost fields, hinge vs linear, per-range); (c) "naive/toy/binary" implementations can
   only produce INSTANCE-level negatives (the L-discipline already bans citing them beyond their
   scope); (d) the seal review checks every negative v5 consumes for correct scope level —
   an over-scoped kill is a BLOCKER-class finding. The measured anchor for this rule: viscosity
   (confound-poisoned "family kill" that was instance-level), UniWARD pooled null (formulation-
   level read as family-level, now the Q1 reopen), mod-dim ("refuted" = one formulation at one
   scale), l7 ("defect" = as-formulated-in-PR95-order).
   ENFORCED (2026-07-08): verdict-scope leg in tools/triality_drift_detector.py (rides the
   existing Stop-hook registration) + fmtools on-device-FM advisory layer (over-scope check,
   family/paradigm-calibrated, persisted → costate_digest), commit aa4cd41f0.

Q. **PROBES BECOME INSTRUMENTS — the toolbelt rule (operator 2026-07-08: "pursue those
   validation paths as well to expand our toolbelt and understanding and improve our
   v[ehicle]").** Every recess probe is a PROTOTYPE INSTRUMENT, not a one-shot gate check: its
   implementation lands as a durable reusable tool (canonical path under tools/ or
   src/tac/witness_control/, docstring with the law + band provenance, artifact schema), its
   output feeds a NAMED consumer (per req P), and the METHOD joins the standing toolbelt usable
   across runs and vehicles: ν-refit (exponential-meat fitting) · forfeit-arm backtester ·
   cadence-law replayer · signed per-class-pair per-direction correlation (the asymmetry
   instrument) · Conley persistence certifier · uint8-deadzone census · max-plus annulus fitter.
   Sister of "Results must become system intelligence": a probe whose code is scratch is a
   queued re-derivation; a probe promoted to instrument is campaign capital — run-2 re-runs the
   SAME instruments on run-1's artifacts for free, which is what makes the ILC error term e_k
   cheap to measure every iteration. Understanding compounds through the tools, not the memos.

--- LANDING FOLDED 2026-07-08T~13:0x: SEAL ROUND 1 ON V6 = CLEAN, COUNTER 1/3 (b17c09f3f) ---
- 0 BLOCKER / 0 MAJOR / 2 MINOR + 2 nits (editorial; bar tests printed). ALL arithmetic
  reproduces independently at artifact precision (crossing chain, asymptote decomposition —
  reviewer also caught that MY charter's suggested floor form would DOUBLE-COUNT; v6's
  composition verified correct — τ* table, ν laws unrounded, B17 bars, comb, K=64).
- MOST LOAD-BEARING SURVIVOR: source-read of both wave-B instruments hunting a SECOND tautology
  inheritance — conley_persistence_certifier + uint8_deadzone_census both read the TRUE gt-cache
  margins field, NOT the corrupted gt_margin key. Corrupted-axis blast radius = exactly the one
  law (τ*) already re-derived. V=5 reversal verified measured-grounded (trigger-vs-estimator
  consumer split), not rationalization.
- MINOR-1 (fixed as v6.1 errata): P-DZ/P-CON census vehicle = the θ* per-stage-attribution run
  (2026-06-30), not mod32cap — vehicle named; alternative fraction-transfer floor 0.2180 still
  > 0.19110, all dispositions stand; run-1 F-rows re-measure the fraction on its own vehicle.
  MINOR-2 (fixed): width label 0.4989. Flags 22/22 real.
- Counter 1/3 on v6 (+v6.1 errata, attested no-substance-change). Round 2 FIRES now; the FINAL
  round waits on P-TAU2 + P-DITHER (in flight) per NO-OPEN-GATES.

--- LANDING FOLDED 2026-07-08T~12:0x: HARDENING SWEEP COMPLETE (all 6 items, tree clean) ---
- LAUNCHER (run-1 gate) FIXED+GUARDED (e28ff371e): root cause MEASURED — launch.sh was rewritten
  IN-PLACE (same inode) ~5.5h into the live mod32cap run; bash reads scripts LAZILY and resumed
  at a shifted byte offset at trainer exit → the orphaned `--ckpt-every` executed. Fix = atomic
  tmp+os.replace in write_launch_sh; guards = bash -n + no-orphaned-continuations + inode-replace
  test (41/41). The failing test was STALE (builder was right; test never updated for the
  deliberate #!/usr/bin/env bash move) — aligned, not weakened. NEW failure class recorded:
  launch_sh_inplace_rewrite_under_live_bash (opened/measured + gate-landed).
- #265 already folded (FEED-09b; task completed).
- LEDGER CLOSURES DONE: all log→dir consumers through resolve_run_dir_for_log; 9 bare JS catches
  now log + null-guarded setTxt on 8 DOM writes + fd-leak fix at 3 spawn sites; 2 resolution
  rows appended; dashboard SANDBOX diff triaged+committed.
- v5.1 errata correctly LEFT (v6 superseded mid-window; v5 pinned append-only — verified).
- Orphans recovery-committed with review_status flags. Suite: 205/205 mandated subsets;
  pre-existing repo lint debt noted out-of-scope.

--- LANDING FOLDED 2026-07-08T~11:0x: V6 (feec6e7af+b5771b531) — THE LAUNCH CANDIDATE ---
- Crossing re-executed unrounded, UNMOVED: central 0.1897336 (margin 0.0013664) · win9
  0.1817034 (0.0093966) · bars 0.0010137/0.0010940 · ILC 9.9573e-4. Central run-1 ≈ 0.26 does
  NOT cross; crossing = engineered gated tail, now with a THIRD named binding constraint
  (locked-mass coverage).
- τ*: convention DERIVED = fixed-point mass(m < τ*·ln5) = f_target (Maslov intent); f_target →
  named probe P-TAU2; LAUNCH VALUE τ_end = 0.31 (ep650-best sits inside its own live-field
  [τ*(q80), τ*(q90)] = [0.277, 0.408]) + SC-3 live-law promotion (q̂=0.85). Anneal narrative
  HONESTLY INVERTED: the control OVER-descended on the τ-leg. 10-row c(τ) cascade + TAIL
  ladder live-form.
- #149 → B19 decode-side seeded dither (~15-25 LOC, 0 bytes rule-118, byte-close-selectable),
  RUN-1 gated on P-DITHER ($0 A/B on the ep650 byte-close). No launch-blocking dependency.
- ASYMPTOTE (M1 measured term — the decisive honesty row): decoded residual 0.0036146 =
  smooth-reachable 2.0351e-3 + quantum-locked 1.5795e-3 (43.7%). **Smooth-perturbative-only
  floor S ≥ 0.2373 — that family CANNOT cross 0.19110.** Crossing requires the locked-mass
  levers (band/comb/dither) to work; composed-family band [0.154, 0.181] retained (lower edge
  conditional on locked-mass × lever-support overlap = named run-1 computation). T_3 = 0.15
  still requires the family step.
- Also: F-DET in config + B-DET preflight · ν laws (settle 237.1, cycle 387.1, s* 6.8971e-6,
  k_max 6; V=5 RETAINED — window-covers-settle refuted-as-necessary) · forfeit arm FIRING
  (B-INJ owed pre-GO) · B17 fitted bar (1.750/1.302) · B16 regrounded + SC-20 · P-MP autopsy ·
  comb 79.33 · 21-row ledger · 9 minors · zero invented flags.
- SEAL: round 1 of 3 FIRED on v6. NEW $0 GATES (P-TAU2, P-DITHER) fired in parallel — final
  seal round waits for them per NO-OPEN-GATES.

--- LANDING FOLDED 2026-07-08T~10:0x: τ-CONFIRM (f4dcad1a6) — THE 0.10 ANCHOR WAS AN APPARATUS ARTIFACT ---
- **τ*_end = 0.062 DOES NOT STAND (scope=INSTANCE; the law τ*=m_q/ln5 untouched).** The m_q=0.10
  anchor was TAUTOLOGICAL: birth_death_persistence_dseg binned flips on the maps-npz `gt_margin`
  key = the SIGNED witness margin-toward-GT (≤0 at flips by definition, max −1.1e-5) — so "all
  flip mass below 0.10" was true of ANY vehicle (bit-reproduced 0.7644972239). Apparatus-vs-
  anchor, NOT physics.
- TRUE GT-margin quantiles (16-pair advisory): END ep1000 m_q90 = 0.743 → τ*(q90) = 0.462;
  BEST ep650 m_q90 = 0.656 → 0.408; ANCHOR-l7 re-measured 0.818 → 0.508. Even q50 → 0.11-0.12.
  The QUANTILE CONVENTION is a design decision (undefined on the real heavy-tailed field) →
  SC-3 + v6 derives it from the law's ORIGINAL INTENT (Maslov error budget).
- **v5 cross-field consistency row (a) is CIRCULAR** (τ_end·ln5 ≈ 0.0998 vs "measured 0.10 edge"
  — both sides consumed the same corrupted apparatus output) — STRUCK. Coherence check: P-CON's
  independent finding (survival = ABSOLUTE ~1.3-1.75 logit bar, NOT τ·ln5) was the first hint;
  wave-A's 0.2432 (true-axis recompute at ep300) agrees with 0.269-0.322 here. Everything
  consistent once the corrupted axis is removed.
- CASCADE (v6): all c(τ) anchors + adaptive-ε clamp re-anchoring + TAIL_k τ-ladder + the
  "anneal TRUNCATED" narrative re-examined (control τ 0.216 is BELOW the new τ*(q90) band
  0.41-0.46 — the control may have OVER-descended, not under). Instrument landed
  (tools/witness_tau_mq_confirm.py, Wave-A bit-for-bit cross-check, 4 tests).
- NO-OPEN-GATES VINDICATED AT FULL SCALE: v5 would have shipped an anneal endpoint derived from
  a tautology. The probe caught it for ~20 min of advisory render.
- ALL PROBE GATES NOW RESOLVED → V6 SYNTHESIZER FIRED (complete fold list in its charter).

--- LANDING FOLDED 2026-07-08T~09:0x: PROBE WAVE-B RESPAWN (instruments inherited+committed) ---
- Q1 **BETWEEN (no-fire/no-kill)**: max |ρ| = 0.1242 < 0.3 fire bar; 3 sides ≥ 0.1 block
  robust-dead → B16 stays default-off. MECHANISM CONFIRMED REAL: per-direction effects carry
  OPPOSITE SIGNS (Lane→Road erasure high-texture +0.135 · Movable→Road low-texture −0.103) —
  exactly what pooled −0.033 averaged away — but at 40% of fire bar. NEW PHYSICS ROW: S_R
  positive-control at chance while margin control passes ⇒ annulus flips are MARGIN-driven,
  not reachability-driven (retro-validates LEVER-4 msal_uni inertness mechanism).
- P-CON **KILL scope=FORMULATION**: raw pers > τ·ln5 certifies only 0.4408/0.5636 (band ≥0.95).
  Fitted s = 21.75/3.75 → survival behaves as a ~τ-INDEPENDENT ABSOLUTE bar ~1.3-1.75 logit;
  Lane islands = the entire failure. B17 ships the FITTED form, not τ·ln5.
- P-DZ **FIRES — the headline**: uint8-deadzone flip mass = **1.5795495775e-3 d_seg-equivalent**
  = 88.7× the duty band; **38.4% of ALL flip mass is sub-quantum** (far-range lane rows 176-224,
  horizon shadow edges, hood boundary). #149 (camera-res sub-pixel placement) GRADUATES from
  DEFER to FIRST-ORDER duty-queue lever. Impossibility bound M1 = MEASURED BINDING → the
  family-asymptote estimate gains a measured term: ~44% of mod32cap's residual is unreachable
  by smooth-witness corrections alone (dither/phase/#149-class levers required for that mass).
- P-MP **KILL scope=FORMULATION** (K≤64 concave-max at annulus): agreement ~0.41 vs 0.95 band.
  SHARPENED: oracle-selection capacity PASSES (K=64 rms 0.076-0.13 ≈ the 0.0998 bound) — the
  max-envelope SELECTION mechanism binds, not representation richness; K=64 payload = 1.53 S
  rate-dead regardless. Max-plus stays solve-inventory/campaign.
- Stragglers: FEED-08l **UPHELD-WITH-EVIDENCE-CORRECTION** (scoreability column mis-mapped;
  flatness now 5-point = STRONGER; compensated recall-vs-gapFP trade) · comb-registration:
  deciding measurement = phase-sweep + registration score, n600-render-pass cost class; this
  JSON recomputes comb gap-FP removal **79.33% not 86%** (solid-baseline discrepancy — v6 must
  cite 79.33 or resolve).
- V6 FOLD LIST (running): F-DET · ν amendments · P-CT3 promotion · B17 fitted-form · B16 stays
  off · #149 graduation + M1-binding asymptote update · P-MP formulation-dead · cadence
  antagonism · comb 79.33 correction · τ_end AWAITING the ep1000 re-render (in flight).

--- CREDIT-DEATH ×3 + RECOVERY 2026-07-08T~08:1x (operator: "recover and respawn staggered,
ensure no signal loss") ---
- Probe wave-B died @83 uses: 4 instruments BUILT + IN TREE uncommitted (conley_persistence_
  certifier / maxplus_annulus_fit / signed_flip_asymmetry_correlator / uint8_deadzone_census) —
  the expensive half survived; respawn inherits + commits them, then runs the probes.
- τ-confirm died @18 uses (early, no artifacts) — clean respawn.
- Hardening sweep died @43 uses (no commits) — clean respawn on next landing (staggered).
- Session-limit reset 01:10 CDT; respawns staggered 2-now-1-next. No signal loss.

--- LANDING FOLDED 2026-07-08T~07:4x: VERDICT-SCOPE HOOK BUILT + LIVE-FIRED (aa4cd41f0/5adfbd70f) ---
- Requirement R is now ENFORCED WITHOUT VOLITION: deterministic leg BLOCKS (added-diff negative
  tokens without verdict_scope; family without citation-or-2-formulations; kills without
  reformulation queue; quote/negation/waiver exemptions; fail-open); fmtools leg ADVISES inline
  (0.24-0.82s measured — well under budget; fires only on declared family/paradigm per its own
  3-case calibration, which honestly EXCLUDED the formulation case it misjudged); advisories
  persist to .omx/state/verdict_scope_advisories.jsonl + 14-day costate-digest line.
- Builder's "not verified on a live turn" caveat: SUPERSEDED WITHIN MINUTES — the hook fired on
  BIBLIOGRAPHY_20260708.md during the live session (both legs), compliance cost 1 line + honest
  waiver. Tuning candidate: table-cell quoted verdicts not covered by the quote exemption
  (waiver path handled it; fold at leisure).
- 56 tests (23 new), ruff clean, two review passes, rides the existing Stop-hook registration.

--- LANDING FOLDED 2026-07-08T~07:3x: CITATIONS (798634dc7 + 2f8c6bf2b) — REQUIREMENT S DISCHARGED ---
- 57 claim rows: 21 fetch-verified + 24 search-verified + 7 honestly-UNRESOLVED. CT-1's
  zero-citation gap fully backfilled (16 canonical records). NO fabricated citations (21/21
  fetched IDs resolve to the named papers).
- Provenance findings → seal audit: arXiv:1301.4777 = Zheng Qu 2013 (NOT McEneaney; CT-2
  misattribution corrected) · weak-KAM O(1/t) anchor = Fathi unpublished lecture notes
  (folklore-without-venue, so-stated).
- CITATION_CLAUSE composed into standard_contract() (integrity-gated, 41 tests, no override) —
  future research charters carry the requirement structurally.
- Hook meta-event: the verdict-scope + recall-evidence legs FIRED on this doc (first real
  firing of the new leg); compliance cost = 1 header line + honest waiver; quote-exemption
  tuning candidate noted for the hook builder's fold.
- MARIMO #347 (deadline Jul 9 11:59PM PST): top candidate = Tabuada/Heemels event-triggered
  control LIVE on our real mod32cap 41-row trace (the P-CT2 backtest IS the notebook; trace +
  instruments on disk). OPERATOR GO/NO-GO surfaced.

--- LANDING FOLDED 2026-07-08T~07:0x: PROBE WAVE-A (4 verdicts, instruments landed per req Q) ---
- P-CT3 **PASS**: forfeit arm first-sustained-fires ep675 (band 670-700, both estimator forms
  agree); EMA-best-at-fire = ep650 = stage true best, forfeit EXACTLY 0; +5.450779e-4 S recovery
  VERIFIED full-precision vs shipped ep625. ARM PROMOTES to firing (pending req-B injection
  test only). Fire epoch INVARIANT to the ν dispute.
- P-CT1 **KILL, scope=FORMULATION**: ν(tau)=0.012653/ep (band [0.02,0.035], kill <0.01 for CE?
  no — muon_fin 0.003289 < 0.01) — the REGISTERED ν=0.026210 is NOT REPRODUCIBLE from the trace
  (its 3.3e-3 S/ep @ep350 input doesn't exist; measured 1.4812e-3; suspected rel-vs-abs units
  mix). RECOMPUTED LAWS (v6 amendment-grade): settle 237 ep · cycle floor 387 · dwell ≥237 ·
  s* = 6.8971e-6 S/ep · V=5 window NO LONGER covers settle (125 < 237 → V must grow or the
  window law re-derives). TAIL k_max shrinks under cycle floor 387.
- P-CT2 **BAND-FAIL (kill not triggered)**: 5/41 skipped vs band 12-17 — B-CT3 stays unbuilt.
  SEAM FINDING (req I paying rent): composed with the promoted forfeit arm, the cadence law
  would SKIP ep650 and hand back the ENTIRE +5.450779e-4 S — the two laws are ANTAGONISTIC as
  formulated; any future cadence law must carry a stage-best-protection conjunct.
- τ-CONFIRM **PARTIAL → τ_end=0.062 PROVISIONAL**: arithmetic ✓ (0.0621335) but flip-mass share
  below the m_q=0.10 edge = 0.2432 on THIS run's cached ep300 maps (anchor expected ~1.0; 75.7%
  of flip mass ABOVE the edge). Note the cross-field consistency row survives as a LAW (Dirac
  layer width = τ·ln5) — the VALUE of m_q moves both together. End-checkpoint confirm = 16-pair
  advisory re-render of the ep1000 ckpt (cheap, FIRED as a follow-up probe).
- Instruments landed: src/tac/witness_control/trace_probes.py + tools/witness_trace_probes.py
  + 19 tests (reviewed properly) + artifact JSON.
- V6 FOLD LIST grows: F-DET (fused-r) · ν-law amendments (settle/cycle/dwell/s*/V) · cadence-law
  antagonism note · τ_end pending the ep1000 re-render · P-CT3 promotion executes.

--- LANDING FOLDED 2026-07-08T~06:0x: #348 GO — THE L70 WALL FELL (ec660ca41/6175362f5/596fee22d) ---
- LOCALIZED: MLX-GPU cross-process nondeterminism = ONE op class — dup-index atomic scatter-add
  (`arr.at[idx].add`, 10 unique hashes/10 procs) + `mx.take` strided-cotangent VJP = the
  reference-R gather-based bicubic-UP BACKWARD. Everything else (GEMM all shapes, conv2d,
  reductions, softmax, seeded random, custom grouped-backward, fused-R fwd+VJP) DETERMINISTIC
  1-hash-in-10. This one op poisoned all 28 witness gradients from ep1.
- CURE ALREADY IN-TREE: `--fused-r-kernel` (fixed-order transpose VJP, no atomics) → full
  launch-path trainer 0/28 diverged cross-process N=10 (Muon arm 0/28 N=5). OVERHEAD: −8%
  (determinism is FASTER, 25.35s→23.44s 200-ep smoke). 25/25 numpy-authority parity tests pass.
  Verified at SMOKE scale; n600/self-orient COMPOSITE CHECK OWED before relying there.
- P0 EN-ROUTE FIX: per_class telemetry `round(dict)` crashed EVERY fresh trainer launch (both
  baseline_v0 + sync-verdict sites) — fixed + verified ~30 real runs. Run-1 launch was BLOCKED
  without this; the trainer working-tree diff mystery is resolved (it was #348's, now committed).
- INSTRUMENT LANDED (req Q): tools/mlx_gpu_determinism_probe.py (19-op-cell, reusable) + 5 tests
  + canonical equation mlx_gpu_crossprocess_nondeterminism_v1 (closes risk-register D6 gap) +
  memory L70 refined.
- RUN-1 CONFIG CONSEQUENCE (v6 fold item F-DET): ship `--fused-r-kernel` in the launch config —
  strictly dominant at smoke scale (determinism + speed + parity-gated); pre-GO verification =
  the n600/self-orient composite determinism check (cheap, rides the launch preflight).
  UNLOCKS: bit-exact proofs on GPU (byte-close/parity F13 rows GPU-accelerable, campaign ILC
  e_k cheaper); deterministic-repro hard-limit #1 now EXTENDS to the GPU.
- SEAL COUNTER: honest handling — F-DET is a DESIGN CHANGE (new lever in the config) → fold into
  V6 with probe-wave resolutions when they land → counter RESETS; rounds run on v6 with
  no-open-gates satisfied.

--- LANDING FOLDED 2026-07-08T~05:0x: SEAL ROUND 1 ON V5 = CLEAN, COUNTER 1/3 (872d2e76c) ---
- 0 BLOCKER / 0 MAJOR / 9 MINOR-nits (none decision/number/build-item-changing; per-item bar
  tests printed; all bind to the P7/v6 editorial fold).
- Crossing arithmetic INDEPENDENTLY re-executed unrounded — every digit reproduces (0.1897336 /
  0.0013664 · 0.1817034 / 0.0093966 · targets 0.0010137/0.0010940 · ILC bar 9.9573e-4 ·
  s*=1.41536e-5 · τ_end·ln5=0.0997852 · hood 5.32688e-6 · TAIL 265×7≤2350).
- Flags: 37/39 exist vs live 248-flag argparse; top nit: `--persistence-loss-warmup-epochs/
  -classes` wrong spellings in prose (real: `--persistence-warmup-epochs`/`--persistence-classes`)
  — DSL factory emits TRUE spellings, launch path unaffected (editorial fix owed).
- Regression: all 9 round-2 findings still fixed; CT IMPORT-NOW folds 5/5 + 5/5 faithful.
- Requirement-R scope audit: NO operative inflation; the 4 known prior over-scopings correctly
  de-inflated in v5; 3 scope-WORDING minors.
- The 3 cross-field consistency rows recomputed and HOLD.
- SEQUENCING: round 2 HELD until probe waves A+B land (NO-OPEN-GATES + pre-registered arm
  promotions execute first; in-band gate resolutions = the design executing, NOT a design
  change — counter holds; out-of-band results → v6 fold → counter resets honestly).

--- SEQUENCING RULE BOUND 2026-07-08T~04:0x (operator: "when are the probes run? ... how can we
say the designs are optimal?") — NO OPEN GATES AT SEAL ---
The FINAL seal round may NOT certify while any $0-runnable gate probe remains unfired. Disposition
classes (every design decision in exactly one): (i) ANCHORED (measured row) — may bear load;
(ii) UNANCHORED-DERIVED — rides ONLY as would-fire arm / default-off+duty-to-measure / build item
with pre-registered kill, AND its gate probe must be RESOLVED before the final seal round;
(iii) UNANCHORED-NO-PROBE — excluded from run-1 with a written reason + reactivation path (the
family-step ideas per req N). Optimality claim after probes: "every decision anchored, gated-and-
resolved, or deferred-with-reason" — optimal RELATIVE TO the free-measurable set; absolute
optimality is never claimed (req N). PROBE WAVE 2 FIRED (PowerPlay order): wave-A control/schedule
on the mod32cap trace (P-CT3 · P-CT1 · P-CT2 · τ-confirm) + wave-B geometry/class on cached
fields (Q1 per-side signed ρ = the B16 gate · P-CON Conley backtest · P-DZ deadzone census ·
P-MP max-plus fit). Stragglers (FEED-08l fresh-eyes · comb-registration audit · K=128 eigen
finish · P6 shares) assigned to wave-B tail.

--- LANDING FOLDED 2026-07-08T~03:3x: V5 (b241cf466) — the seal target ---
- DRAFT_OPTIMAL_STACK_v5_20260707.md (596 lines): all 10 CT fold items 1:1 (+2 gated extras).
  Crossing RE-VERIFIED unrounded, UNCHANGED: central S=0.1897336 (margin 0.0013664), win9
  S=0.1817034 (margin 0.0093966); train targets ≤0.0010137 / ≤0.0010940; ILC-formal bar
  0.0011−Δ̂ = 9.9573e-4. Signal ledger = 19 rows SC-1..SC-19, ZERO no-signal gap terms (req P).
  τ-indexed constants: 9-row c(τ) enumeration. B16 signed shape-gradient (Q1-gated) · B17
  Conley certificate (P-CON backtest) · B18 release law r*(t)=0.95·σ_eff(t) · B-CT1 would-fire
  TAU→FIN arm (P-CT3 backtest queued).
- APPARATUS CATCH (never-invent-flags WORKING): synthesizer grep-verified all flags against the
  real 250-row argparse — BOTH CT seats had invented one (`--copred-verdict-window`,
  `--island-dilation-radius-end`); corrected; V=4→5 routed to the B1 spec (no in-trainer flag
  exists). CT provenance tagged fresh-research-round-1-unreviewed on every derived row.
- SEAL COUNTER: v5 = seal target; round 1 of 3 (both lenses) FIRED.

--- LANDING FOLDED 2026-07-08T~02:5x: CT-1 (d9325a44e) — REQUIREMENT O COMPLETE (both seats) ---
- ct_deepresearch_1_training_campaign_control_20260707.md (552 lines, fresh-research-round-1).
  Top-3 imports:
  1. FORFEIT-MATCHED TAU→FIN EXIT (MPC): fire at slope s* = ν·forfeit = 1.41e-5 S/ep — shipped
     trigger 4.8× coarser, fires ~60 ep early; later firing recovers ≈ +5.4e-4 S (0.30 margins)
     for ~42 min (mandatory under L59). $0 backtest P-CT3 (band ep670-700; kill <650 or >726);
     would-fire arm first (~10 LOC, req-B).
  2. DECODE-GAP ILC FEEDFORWARD: Δ̂ is ILC-repeatable → train-side bar 0.0011−Δ̂ = 0.00100
     (CONSISTENT with v4's independently-chosen 0.0010); campaign EWMA ω=0.5, Newton-ILC γ=0.7
     → contraction 0.055-0.545 → 2-3 runs to identified floor. Req-P identifiability condition
     formalized (excitation rank + σ_meas + matched instruments) = campaign convergence precond.
  3. PMP + TWO-TIMESCALE RATIFICATIONS: PMP stop-rate 7.1e-5 S/ep ≈ shipped eps_rel 6.8e-5
     (within 5% — INDEPENDENT ratification); fix: co-predicate window V=4→5 (settle 115 ep vs
     100); turnpike validates cap_fin + gives TAIL_k budget law (≥265 ep/cycle, k_max 3-7).
- Req-P §12.2 five signals (overlaps CT-2's): verdict-replicate σ_meas · per-stage Δ̂_k parity
  trajectory (EVSI ~2e-3 S) · live m_q(t) row (TAIL τ*_k input) · forecast-residual row ·
  signed per-class-pair flip-mass row.
- EVSI TABLE (dual control, req N ratified): run-1 instrument value ≈ 0.05 S of v5 decision
  value (pose row dominant 0.044) — ~10× its direct crossing value.
- CROSS-FIELD CONSISTENCY TRIPLE (the fields-grade payoff): (a) τ_end·ln5 = 0.0998 ≈ measured
  0.10 flip-support edge [CT-2]; (b) PMP stop-rate ≈ shipped eps_rel within 5% [CT-1];
  (c) ILC bar 0.00100 = v4's independently-chosen train bar [CT-1×v4]. Independent derivations
  agreeing with measured constants = the design is sitting on real structure.
- NEXT: V5 SYNTHESIZER folds both CT §12 IMPORT-NOW sets into v4 → v5; THEN seal rounds on v5.

--- LANDING FOLDED 2026-07-08T~02:3x: CT-2 (8be6251ee) ---
- ct_deepresearch_2_pde_geometric_topological_control_20260707.md (653 lines, §0-§13,
  fresh-research-round-1 unreviewed). Top-3 imports (PowerPlay-ordered):
  1. CONLEY PERSISTENCE CERTIFICATE (~30 LOC + $0 backtest): island survives stage k + decode
     ⟸ pers(I) > τ_k·ln5 + Δ_dec^logit (~0.10 logit at τ_end=0.062) → per-island pass/fail
     ledger + death alarm. Kill: certified-survival < 80% on the 0630 birth-death ledger.
  2. SIGNED PER-CLASS-PAIR SHAPE-GRADIENT WEIGHT (0 bytes): exact d_seg shape derivative =
     signed one-sided boundary density — the THEOREM-form of the UniWARD pooled-null mechanism.
     CROSS-FIELD CONSISTENCY: τ_end·ln5 = 0.0998 ≈ measured 0.10 flip-support edge (Maslov
     semiclassical + Hadamard layer width = SAME law from two fields). Band ≈ 2.2× crossing
     margin on the lane leg; gated on the queued $0 per-side ρ probe.
  3. τ-INDEXED CONSTANTS + CRITICAL-NUCLEUS RELEASE LAW: Γ-convergence licenses finite-τ design
     LAW-wise only — every τ-adjacent constant ships as c(τ); island release r*(t)=0.95·σ_eff(t)
     (matches measured dilation knee r*≈1.43 px).
- Req-P: per-section REQUIRED SIGNAL lines; 5 run-1 signals to add (per-stage parity rows incl.
  Δ_dec^logit · per-direction margin histograms · live per-island persistence ledger ·
  interface-geometry row · per-class logit export + per-primitive residuals) → v5/§4c fold.
- §13: 5 impossibility bounds (uint8 deadzone, homogenization pinning, Godunov barrier, Hajek,
  max-plus blow-up), each with its deciding measurement → family asymptote COMPUTABLE from run-1.
- Backstepping: DERIVED-dead for this plant (campaign detour saved); surviving essence =
  max-plus band-residual decomposition.

--- LANDING FOLDED 2026-07-08T~02:0x: V4 (d06e7edbd) ---
- DRAFT_OPTIMAL_STACK_v4_20260707.md: all 13 fold items 1:1 + mid-fold requirement P.
- CROSSING NOW BINDS ON THE DECODED SURFACE: 100·(d_seg_train+g_dec)+√(10·d_pose)+rate < 0.19110
  with g_dec = +1.0427e-4 MEASURED (R6 ep650). v3 triple does NOT cross (0.199734, over by
  4.85× margin — stated plainly). v4 triple (train 0.0010, central rate): 0.189734, margin
  0.001366 (train must be ≤1.4% above design edge). win9 arm (18,832 B measured): S=0.181703,
  margin 0.009397 (≈9.4% headroom — restores v3's condition). g_dec = SELECTION variable via
  per-stage F13 parity rows.
- FAMILY ASYMPTOTE (req N, honest): S_asymptote ≈ 0.165 central, band [0.154, 0.181] — this
  family claims ~36% of the 0.07313 frontier-to-floor gap; **T_3=0.15 sits at/beyond the
  family's optimistic edge** → the family step (quotient codec #155 / compress-half #336 /
  0.0005-regime) is named run-2+ work. Run-1 = instrument (F13/F14/F15, per-class F-rows,
  RS-1..5, TAIL yields) regardless of crossing.
- Signal-completeness ledger (§4c) found + fixed 2 no-signal gap terms (F14 along/across
  spectrum, F15 TAIL yield). review_status: pre-registered-only.
- SEAL SEQUENCING: hold seal round on v4 until CT-1/CT-2 land (their IMPORT-NOW items fold
  first — avoid burning a round on a draft the research is about to move).

--- LANDINGS FOLDED 2026-07-08T~01:1x ---
- SEAL ROUND 2: NOT CLEAN (counter 0/3). BLOCKER: v3 ships AACoverageRender(ss=2) that recess-R3
  measurably REFUSES (105.9 GiB > 89.6; ipe = surviving form); MAJOR: stale K=128 0.011 (measured
  0.163) ×4 sites; MAJOR: per-class amplify weights have NO trainer flag/DSL param/§10 item
  (config-orphan); 6 MINORs (flag name --softmax-temp-end; hood threshold 10× slip; adaptive-ε
  saturation alarm owed under τ_end 0.062; recon-gap not margin-ranked; AA×island row; reactive-
  only laws where forecasts exist per M). Architecture + crossing arithmetic + lane-first SURVIVE.
- WAVE-1 R1/R3/R6: R3 AA gate amended+verified (a0b82ba6c) REFUSE as-designed; R1 LBND4 win9 =
  18,832 B exact (saves 0.008031 S; win5 roundtrip FALSE, quarantined; LBND2 = 41,526 stale-fix);
  R6 PARITY ROW PASS-WITH-FLAG: decoded 0.0036146 vs training 0.0035103, Δ = +1.0427e-4 d_seg =
  +0.010427 S = 5.86× crossing margin — decode TRUSTED, but realized-gap allowance CONSUMED; the
  crossing triple must carry the decode-gap term explicitly.
- NEGATIVES SCALE-VALIDITY: 9 ROBUST · 11 BOUND · 4 SUSPECT · 1 REOPENED (viscosity, confound-
  poisoned — run-1 eikonal ramp = first fair test). UniWARD "at chance" = flagship asymmetry
  case ($0 per-class-pair per-DIRECTION signed re-test queued); τ-crossover window-scoped (v3
  trains 3.5× below it); adaptive-ε clamps → τ-law re-derivation; TAIL VERDICT: END → TAIL_k
  warm-restart cycles (~40 LOC, all pieces built), END demoted to req-B fail-safe cap.
- V4 DESIGNER FIRED with the full fold list (blocker+majors+minors+R6 decode-gap+TAIL_k+reopens).

N. **REFLECTION · RECURSION · INFLECTION · INTROSPECTION — v3 IS NOT THE OPTIMUM (operator
   2026-07-07: "v3 is definitely a step in the right direction but I doubt it's optimal or
   really even that close relatively").** CALIBRATION, by v3's own numbers: engineered crossing
   0.18932 vs frontier 0.19110 vs measured floor 0.118 — the frontier-to-floor gap is ~0.073 S
   and v3's best case claims ~0.002 of it (~3%). The SEAL certifies CORRECTNESS AND
   LAUNCH-WORTHINESS, never optimality. Binding consequences:
   (1) **REFLECTION**: run-1 is a MEASUREMENT INSTRUMENT for the campaign — its purpose is the
   parity row, the binding-constraint measurements, and the telemetry corpus that make v4/v5
   derivable, at least as much as its own S. Frame the launch that way in the synthesis.
   (2) **RECURSION**: the design loop does NOT stop at the seal — v4 design work (deeper floor
   attack: d_seg → the 0.0005-0.0009 regime, rate → 0.05 via the compress-half, the quotient-
   codec paradigm #155, the asymptotic-tail stages) continues in parallel with run-1's burn,
   consuming its live telemetry. Each run is one iterate of the campaign's outer loop.
   (3) **INFLECTION**: per design family, watch the returns curve — when a family's improvements
   per round bend toward ITS asymptote, and that asymptote sits above the target, the correct
   move is a NEW FAMILY (reframe/paradigm step), not more polishing. Estimate each family's
   asymptote explicitly (the L-discipline applied to the DESIGN process itself); v3's family
   (composed-lever witness on mod32) has an asymptote the crucible should ESTIMATE, not assume.
   (4) **INTROSPECTION**: each seal round includes a self-critique pass that attempts to
   generate OPERATOR-GRADE reframings ("what would the operator catch?") — the H-M requirements
   all originated as operator catches; the machine's maturity metric is generating them first.
M. **CAPACITY HUMILITY · MEASURE→SWEEP→DERIVE · SELF-DERIVING PDE CONTROL (operator
   2026-07-07).** Three binding principles:
   (1) **CAPACITY IS NOT FULLY UNDERSTOOD — treat it empirically.** Capacity's interaction with
   basis/format/schedule has surprised us repeatedly (bc20 starvation · capacity-alone +6%
   HARMFUL on isotropic · basis-prior-to-capacity · the two-surface island shares). No capacity
   claim rests on derivation alone; the hedge that wins REGARDLESS is TRAIN-BIG-COMPRESS-SMALL
   (big capture → #157/#336 sensitivity waterfill down; the compress-half is measured, so the
   train-side capacity choice only needs to be GENEROUS, not optimal). Where capacity matters,
   fit RESPONSE SURFACES (#170: Q-bits × E-epochs × C-capacity) from sweeps — measure → sweep →
   DERIVE the law from the measured surface → step toward optimal — not theory-first guessing.
   (2) **MEASURE→SWEEP→DERIVE-TOWARD-OPTIMAL is the standing loop** for any knob whose law we
   can't derive a priori: cheap sweep → fitted surface → derived local law (with validity
   domain) → next sweep centered at the implied optimum. Response surfaces are first-class
   apparatus (they feed the costate DECIDE layer as models, not just plots).
   (3) **SELF-DERIVING PDE CONTROL — responsive AND proactive.** The #318/#320 exemplar
   (adaptive-ε from viscous-HJ CFL stability) is the TEMPLATE, not a one-off: control laws are
   DIFFERENTIAL EQUATIONS derived from the system's own physics + live measurements, with BOTH
   feedback (respond to measured state) and FEEDFORWARD/model-predictive terms (act before the
   event, using the fitted forecast models — powerlaw meat-flux, spectrum trend, per-class λ
   trajectories — as the internal model). The costate controller's mature form: λ̇ = the adjoint
   equation of the score functional; anneal rates governed by measured meat-flux ODEs; stage
   transitions = MPC decisions over the response surfaces. Programmed intelligence = the DE +
   its clamps + its falsification band, in the DSL as class-(c) laws.
L. **SCALING: COARSE → FINE → EXTREMELY FINE → ASYMPTOTE, under INFINITE COMPUTE + SEPARATRIX
   ASYMMETRY (operator 2026-07-07).** ASYMMETRY ADDENDUM: the separatrix is NOT symmetric —
   flip costs and responses are one-sided per class-pair (Road→Lane FP ≠ Lane→Road erasure;
   winner-vs-runner-up logit perturbations act asymmetrically on the decision surface; the two
   sides of the margin field carry different measured flip masses and UNIWARD costs). Any
   negative that measured a SYMMETRIC perturbation/response and concluded "inert/flat/at-chance"
   may have AVERAGED AWAY a one-sided effect — re-review such negatives per-side. Design
   consequence: losses/gates/amplifiers on the boundary may be SIGNED (one-sided hinge per
   class-pair direction), not symmetric bands.** Every "exhausted/flat/NO-GO" verdict is SCHEDULE-RELATIVE unless proven
   scale-robust: exhaustion at a coarse point of the anneal ladder (τ=0.216 truncated, β=3.177,
   1000-ep budget) does NOT bound the asymptote (τ→τ*→0, β→∞ effective, PR95's 29,650-ep /
   8-stage existence proof — each stage exhausts before the next reopens descent; that IS
   annealing). Training time is lexicographically SECONDARY (L59) — with infinite compute the
   schedule is OPEN-ENDED: event exits fire REFINEMENT TRANSITIONS (fine → extremely-fine
   stages: smaller LR, τ→τ*, engaged sharpening, cyclic finishing) rather than termination;
   "wall-clock savings" reframe as "epochs redeployed to finer stages". The extremely-fine
   regime near τ→0 is where the tropical/Maslov math lives — the asymptote is the argmax itself.
   BINDING: every negative finding carries a SCALE-VALIDITY tag {scale-robust (holds at all
   schedule points, e.g. instrument facts, winner's-curse catches) | scale-suspect (measured at
   a coarse point, may reopen at finer scales) | scale-bound (explicitly valid only at the
   measured point)}; a scale-suspect negative may gate run-1 arms but may NOT retire a paradigm.
K. **NATIVE FORMAT + ADEQUATE CAPACITY PER TECHNIQUE (operator 2026-07-07: "Give all
   techniques the format and capacity they need").** Every technique runs in ITS OWN
   mathematical format, matched to its object, with capacity at ITS OWN optimum — never a
   borrowed format or a pooled hand-me-down budget. The measured anchors of this law: the dash
   wanted a COMB (modulation carrier), not more linear along-frequency (FEED-08l — wrong format
   loses at any capacity); pose wanted a 6-dim SCREW ξ, not pixels; the lane wanted a POLYNOMIAL
   band; the hood wanted a CLAMP; the cartoon bulk wants CURVELETS; bc20 died of capacity
   starvation in the right paradigm. Audit lens (binding on every review round): per technique
   in the stack — (1) FORMAT: is its representation the native chart for its object, or is it
   squeezed through another technique's format? (2) CAPACITY: is its budget (params/harmonics/
   coeffs/epochs/bytes) at its OWN derived or measured optimum, or starved by sharing? (a comb
   with too few harmonics, a band with too few coeffs per lane, a FiLM head too narrow for its
   conditioning, an island class under its critical nucleus — each = a K-finding); (3) the same
   for APPARATUS formats: per-technique telemetry rows, checkpoint persistence (self-orient!),
   DSL factory, and verdict surface — a technique whose state can't be saved/measured/expressed
   is format-starved even if its math is right. UNIQUE-AND-COMPLETE-PER-METHOD applied at the
   technique grain; the fractal discipline's representation half.
J. **PRECISION + MARGIN-DENOMINATED SIGNIFICANCE (operator 2026-07-07: "very small numbers are
   relatively significant; even the thousandth, ten-thousandth, hundred-thousandth place
   matters").** The crossing margin is 0.00178 S. The exchange rates: 1 byte = λ_bytes =
   6.6586e-7 S · 1 KB = 6.82e-4 S = 38% of the margin · 1e-5 d_seg = 1e-3 S = 56% ·
   1e-6 d_pose ≈ 2.9e-4 S = 16% (√-steepened at 3e-5) · the ep625-vs-ep650 θ delta (2.67e-5
   d_seg) = 2.67e-3 S > the ENTIRE margin. Binding rules: (1) NO premature rounding — d_seg
   carried to ≥5 decimals, d_pose ≥6, bytes to the byte, S to ≥5 wherever arithmetic chains;
   (2) every "negligible/small/below-noise" claim DENOMINATED in S-units as a % of the 0.00178
   margin — undenominated smallness = a finding; (3) "unattributable" ≠ "insignificant": effects
   below the +4.3% reconstruction gap (~0.015 S) are significant-if-real and the gap fix
   (persist self-orient, req-F #6) is MARGIN-CRITICAL apparatus — it gates attribution of every
   win smaller than 8× the crossing margin, i.e. every win that matters; (4) every lever prints
   benefit AND cost in the same S-unit currency.
H. **PER-CLASS TREATMENT + APPLES-TO-APPLES (operator 2026-07-07: "Don't compare apples to
   oranges. Different classes need different treatment just like different stages").** The five
   SegNet classes are FIVE DIFFERENT MATHEMATICAL OBJECTS and the design/reviews treat them so:
   **lane** = anisotropic dashed curve (comb carrier + band + along-tangent laws — its own
   physics); **movable** = sparse mobile islands (dilation homotopy + persistence protection);
   **hood/MyCar** = STATIC region (the 0-byte clamp #139 — is it IN? free d_seg if not);
   **road/undrivable** = bulk cartoon (the basis's C² job). Each class gets its own laws, its own
   telemetry rows, its own kill thresholds, its own curriculum (per-class sub-curricula per the
   schedule-as-law extension). REVIEWS MUST NOT POOL ACROSS CLASSES: a pooled d_seg verdict
   averages apples with oranges; per-class comparisons only under MATCHED CONDITIONS (same load
   path, same checkpoint class, same coder when comparing byte costs — the LBND2-vs-LBND4
   "conflict" was two coders, not one number). Sister of the measured per-stage-treatment law
   (L58) — classes are to space what stages are to time.
I. **SYNERGY MATRIX — between, among, and through (operator-directed).** The synthesis carries a
   lever×lever×stage×class SYNERGY/ANTAGONISM matrix, each cell a coupling law not a vibe. Rate
   synergies are the obvious shallow layer; the DEEPER classes to hunt:
   - **GEOMETRY-SHARING** (one stored object serves N score terms): the openpilot lane polynomial
     serves the band AND the comb phase (ego-ξ) AND the basis orientation; the screw ξ serves
     d_seg warp AND d_pose (the canonical dual-use); chroma serves the d_seg annulus AND
     PoseNet's YUV6 input.
   - **GRADIENT-SHARING** (one loss improves two terms): entropy-shaping improves rate AND acts
     as a regularizer; persistence loss protects births AND stabilizes boundaries.
   - **SCHEDULE-COUPLING** (one stage completing enables another lever's efficacy): anneal
     completion × warm-Muon (the finisher only pays on a completed anneal — M1/M2 measured);
     island birth × AA render (AA raises birth SURVIVAL through R); WeightEntropy(train) ×
     waterfill(post) — the Class-D×B recess.
   - **TELEMETRY-SHARING**: one sensor feeds N laws (per-class λ feeds exits AND amplify gates).
   Antagonisms get the same treatment (declared seams: GFC×self-orient, AA×seed ordering,
   per-class homotopy interference). The FRACTAL principle binds it: the task-space level-set
   witness + Morse-Smale complex demand RECURSIVE FRACTAL OPTIMIZATION — the same
   optimize-at-its-own-optimum discipline applies at every scale (class → stage → lever → layer →
   run → campaign), AND the couplings between scales are themselves optimized objects.
G. **TRIALITY + TASK INTEGRATION (operator-directed 2026-07-07: "all must be integrated into
   triality especially DSL and tasks or it's ephemeral").** Crucible-local files are ORCHESTRATION
   STATE, not landings. Nothing counts as KNOWN until it lives in all three legs + the task
   system: (1) **equations** — every measured crucible finding registered in
   `tac.canonical_equations` (S2's M1-M5 anneal-truncation/cold-quench/meat laws; S4's
   entropy-floor + measured byte/rate rows; S6's pose-second-wall bound + byte-close rows; S3's
   indefinite-spectrum row [PROVISIONAL-K=8]; the L25-temporal-delta NEGATIVE receipt), and the
   15 AWAITING_VERIFICATION rows updated wherever the crucible measured their anchors; (2)
   **DSL** — every control law in the draft = a Lever/Schedule/Curriculum factory (new laws →
   new factories or emit_stub_lever; the requirement-F telemetry rows wired as DSL-held,
   default-on observability); (3) **DAG** — crucible FEED entries for the measured rows + the
   design decision (FEED-09x series); (4) **tasks** — the draft's build list + recess queue →
   TaskCreate rows with owners; superseded tasks (#183, #124, #285 per the compendium) marked
   with supersession notes; (5) **COSTATE CONTROLLER (operator addition)** — the crucible's
   control designs land as CODE in the live controller surfaces, not as memo prose:
   S3's SENSE additions (gn_spectrum.checkpoint_lanczos producer + requirement-F rows: anneal
   state, online meat, trigger would-fire audits, transition health) wired into
   `witness_control`/`costate_digest`; S3's DECIDE laws (curvature-aware exhaustion predicate,
   spectrum-rate exponential-mixture replacing linear-λ, λ_max-sized trust region, POWERPLAY
   never-regress) landed shadow-mode-first; the NEW `quadratic_basin` ExitEvent registered as a
   DSL gap-kind; the activation-ledger launch-time argv→lever ENGAGED ingest (S5 R1) wired so
   the ledger finally reads real runs; the duty-to-measure queue RE-RANKED from crucible
   verdicts (S5's FIRE/DEFER/RETIRE table replaces the stale never-fired ranking). Actuation
   boundary UNCHANGED: advisory-only autonomous, heavy/stop/config = operator-GO (CONTAINMENT).
   P7 is NOT complete until the drift detector is satisfied on all legs AND the controller
   consumes the new senses — integration commits drop the [no-triality] tag because they ARE
   the triality landing.
F. **TELEMETRY ENHANCED FROM FINDINGS (operator-directed 2026-07-07).** Every crucible finding
   discovered by post-hoc forensics becomes a LIVE telemetry row + alarm in the next run —
   "anything found by archaeology must be observable in flight." The findings→telemetry map
   (each row score-neutral read-only → defaults ON per the default-off-is-orphaned rule):
   1. ANNEAL STATE (M2): per-epoch effective β(t)/τ(t)/anneal-progress% + completion flags;
      ALARM when any consumer (Muon fire, stage exit) triggers with its precondition-anneal
      incomplete. The M2 defect becomes impossible to miss live.
   2. TRANSITION HEALTH (M1 cold-quench): optimizer-state rows at every stage boundary (momentum
      norms, effective LR, restored-vs-reset moments per the never-reset-moments law) + a quench
      detector (loss-spike attribution to the transition, +X% threshold alarm).
   3. ONLINE MEAT (M4): per-epoch remaining-meat estimate (the AIC mixture fit) emitted as a row
      — exits become observable, 76-125 wasted epochs impossible silently.
   4. TRIGGER AUDIT (M3): every event trigger logs its would-fire state + inputs per epoch
      (reorient-cycle-aware windows) so vacuous/confounded triggers are visible BEFORE they gate.
   5. SPECTRUM SENSE (S3): checkpoint-cadence Lanczos summary row (top-k λ±, negative-curvature
      count, in-basin flag) — the curvature-blindness closed as a standing SENSE organ.
   6. CKPT FIDELITY (sweep +4.3%): persist self-orient state in checkpoints (BUILD) + a
      save-time reconstruction-gap check row (saved-state verdict vs live verdict delta; ALARM
      over threshold).
   7. LEVER ENGAGEMENT (S5): launch-time argv→lever ingest + per-lever ENGAGED (not just
      flagged) predicate rows; activation ledger wired to real runs (S5's R1 fix).
   8. SINGLE-LEVER ATTRIBUTION (S5 band-unattributed): any render-time lever activating mid-run
      emits a paired with/without verdict delta at its activation epoch — no more unattributed
      firings.
   9. EFFECTIVE-CONFIG PROVENANCE (S6 G1): byte-close + trainer emit the EFFECTIVE config
      (persisted cfg vs CLI-default diff); FAIL LOUDLY on silent mismatch (the freq_along
      confound class).
   10. CONFIG-DELTA-VS-BASELINE (sweep grad-clip confound): every launch emits a config-diff row
       vs its named control (mod32cap) — baseline confounds visible at launch, not discovery.
   11. POSE WALL WATCH (S6): once w_pose>0, per-verdict d_pose through the L3 mechanism + FiLM
       read-back check rows, with the 1.5e-4 kill threshold monitored live.
   12. STAGE WALL-CLOCK (S2 35% waste): per-stage wall-clock + epochs-past-meat rows.
   Designer §4/§10 carries the telemetry build list; requirement B's tested-end-to-end contract
   applies to every ALARM (backtest + injection + fail-safe).
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
- 2026-07-07T~19:2x **P2 DRAFT LANDED** (467daadd2, 565 lines, launchable-as-written):
  ARM-PRIMARY = Mod32SegOnlyControlBase + along=8 (guard honored; Rebalance=gated A/B) +
  islands-first (S5 FIRE core, per-class LADDER laws) + band trained-with (LBND4 +30.9KB booked)
  + pose ON two-track (kill 1.5e-4) + event-exit schedule w/ caps (M2 fix; warm-Muon; chain-A
  branch) + rate WeightEntropy-λ15-IN / flat-minima+QAT-OUT + waterfill gated → archive central
  93.1 KB / rate 0.0620. Predicted S central ≈0.29, band [0.186, 0.47] — run-1 central does NOT
  cross 0.19110 (stated plainly per NO-FAKE); crossing = joint favorable tail; T_3 = run-2
  levers (named). 3 riskiest: islands-net-positivity-in-training, pose 7×-vs-R1-floor, waterfill
  ∧ band-conservative-edge.
- 2026-07-07T~19:15 **RECALL-EVIDENCE HOOK FIRST CORRECT FIRING**: blocked the draft for missing
  'STORES CONSULTED:'; complied honestly (incl. NOT-consulted list). Class recall-before-decide =
  HOOK-ACTIVE-AND-FIRED. **P3 RED-TEAM LAUNCHED** (provenance audit + design attack, per charter).
  In flight: P3 red-team + pursuit-chain-A + integration tranche 1. Next per queue: #346 build on
  next free slot.
- 2026-07-07T~19:4x OPERATOR CRITIQUE of the P2 draft ("lazy and naive design not grounded in
  deep math") — adjudicated honestly by main-loop audit: laws ARE derived (1-Lipschitz easing,
  CFL adaptive-ε, Dykstra-capped bands, per-class birth laws) BUT the charge LANDS at the
  architecture level: (1) stack is VERDICT-COMPOSED not variationally derived; (2) central fails
  T_1 with NO binding-constraint/ceiling derivation — "run-2 headroom" (comb, AACoverage) is an
  ADMISSION of unexploited run-1 headroom deferred by schedule not math; (3) #149 subpixel
  closed-form ~absent. → P3 red-team gains MANDATORY PASS 3: DEEP-MATH GROUNDING AUDIT
  (DERIVED vs VERDICT-COMPOSED per choice) + DERIVE-THE-CEILING-OR-DESIGN-HARDER per S-term +
  adjudicate comb/AACoverage/#149 run-1-vs-run-2 with math not convenience. If ceiling
  underivable ∧ headroom real → REVISE-THEN-RECESS with levers moved to run-1.
- 2026-07-07T~20:0x OPERATOR COUNTER-FRAME (binding on P3/P3b/P5): "pose is solvable, rate is
  small with wiggle room, d_seg tameable — you are being pessimistic; we are in a SANDBOX." The
  optimist's arithmetic folded into the red-team charter: rate = measured 0.056 structural moat
  (crossing condition reduces to 100·d_seg + √(10·d_pose) < 0.129); pose = BUILT-UNFIRED
  mechanism with an on-scorer existence proof (never pushed at w_pose>0 — pre-measurement number,
  not a wall); d_seg needs 2.8× with 63.9% of flips in never-fired levers + anneal recovery +
  warm finisher + measured negative curvature. PROBABILITY-MODEL CRITIQUE: central 0.29 =
  joint-independent-tail (one-shot lottery); the stack's checkpoints/event-exits/costate/kill-
  fallbacks make SEQUENTIAL-DESCENT-WITH-REPAIR the candidate honest model — synthesis must
  present BOTH bands with per-lever repair mechanisms named, and justify the central. Sandbox
  prior: iterate-ability HIGH → marginal cost of including derisked levers in run-1 is LOW.
  Guard: optimism about MECHANISMS, rigor about CLAIMS (pose measured-through-witness only, L68).
- 2026-07-07T~20:4x P3 RED-TEAM DONE (d33df2be9): **REVISE-THEN-RECESS**. Provenance audit
  CLEAN (anchors re-verified BY EXECUTION: byte-closes bit-exact ×3; 31-flag argparse check zero
  invented). F1 BLOCKER: draft's own printed crossing tail sums 0.2207 > 0.19110 — doesn't cross;
  0.186 band-lower irreproducible + leaned on forsworn ancestor pose; honest crossing needs pose
  ~2-3e-5 ∧ d_seg at islands ceiling. F3: anneal-epochs=726 makes the event exit VACUOUS by
  construction (re-locks the 76-125 wasted ep; §8 projection arithmetically impossible). F4:
  AACoverageRender excluded on a STALE seam (FEED-07g built compose-after-downsample WITH proofs
  same-day) — **the retrieval-failure class caught by the MACHINE, not the operator** (the curve
  bent). F2/F6/F7: req-A(ii) claimed-absent · armed trigger no backtest · WeightEntropy no λ=0
  comparator. F8/F9: SOLVE needs measured-acceptance clause (chain-A ~35% instrument gap) ·
  islands ep0 abort gate calibrated against behavior the measured seed run never produced.
  Counter-frame adjudicated 2/3 RIGHT and strengthens: rate moat → ceiling-raising levers are
  the crossing ENABLERS; F17 dual-band (independent-tail AND sequential-with-repair) demanded.
  **P3b REVISION DESIGNER LAUNCHED** (DRAFT v2: honest crossing engineered true, AA into run-1,
  vacuous exit fixed, requirement letters closed, chain-A folded live).
- 2026-07-07T~21:0x #346 RETRIEVAL-FIRST LAYER COMPLETE (4 commits, 94 tests): corpus_query.py
  (7-store deterministic query, ~1.0s whole-corpus: research 5343 · equations 527 · memory 1866 ·
  DAG 331 · council 275 · tasks 57 · docs 92) + convene.py (auto grounding packets from the
  codified 20-store checklist — every future convening starts complete BY CONSTRUCTION) +
  subagent_contract clauses (RECURSION/CONTROL_LAW/RETRIEVAL_FIRST/REVIEW_STATUS in every future
  agent contract, integrity-gated anti-self-waive) + costate digest recall-push ("the corpus
  knows:" at SessionStart when a convening is live). The L83 root cause (writes-better-than-reads)
  now has its structural cure: retrieval is a 1-second query + an auto-packet + a contract clause
  + a push — no longer volitional, no longer the operator's memory. Task #346 CLOSED.
- 2026-07-08T~00:5x **P3b DRAFT v2 LANDED** (`DRAFT_OPTIMAL_STACK_v2_20260707.md`, supersedes v1
  append-only; all F1-F17 dispositioned §0.1). Headline: crossing arithmetic recomputed — verified
  triple d_seg ≤0.0011 ∧ d_pose ≤3e-5 ∧ rate ≤0.062 → S 0.1893 < 0.19110 (pose SUCCESS bar 3e-5;
  1.5e-4 demoted to milestone/hard-kill); AA IN ARM-PRIMARY ep0 (P11 $0 memory+throughput gate +
  BA decode LB-at-byte-close; byte-close-selectable repair); comb P1-pass conditional inclusion
  law; `--anneal-epochs 600` event-margin law makes TAU→FIN REAL — **$0 co-predicate BACKTEST RUN
  this session on the mod32cap 41-row trace: first sustained fire ep625 (one cadence before the
  ep650 best); CE cap-fires, stated**; twin re-purposed to λ=0 (single-dim entropy attribution +
  clean Class-D×B recess; per-stage kills STACK-level); chain-A folded (λ₋ collapses ~1/K, u_min
  ISOTROPIC — no basis-mechanism shortcut, no cheap descent at ep650 ⇒ TerminalSolve-from-ep650
  measured NO-GO; SOLVE acceptance = measured-verdict HARD, K=128 branch carried); S5-R5 composed
  ceiling folded (island share 0.562; **big-3 anneal-completion recovery = the named binding
  constraint**); #342 solve inventory produced (§11, 15 blocks); dual probability model printed
  (independent 2-6% / with-repair 8-15%); central ≈0.26 does NOT cross (stated per NO-FAKE);
  wall-clock corrected 35%→5-27%. NEW pre-GO probes: P11 (AA gate) + P12 (ep0 init probe, F9).
  Next per queue: second red-team pass on v2 → P4 empirical recess (P5/P7 first per PowerPlay).
- 2026-07-07T~21:3x **DRAFT v2 LANDED** (305d884ce; v1 preserved append-only). CROSSING CASE NOW
  ARITHMETICALLY TRUE + ENGINEERED: d_seg ≤0.0011 ∧ d_pose ≤3e-5 ∧ rate ≤0.062 → S=0.1893 <
  0.19110 (row-verified; v1's false tail printed as the negative example). Central ≈0.26 does NOT
  cross — stated plainly; the TWO BINDING CONSTRAINTS exhibited + instrumented (big-3
  anneal-completion recovery ~4-9e-4 UNMEASURED; lane composed-band efficacy). AA IN from ep0
  (P11 $0 gate pre-GO) · comb conditional on P1 · pose bar 3e-5 (1.5e-4 → milestone) · λ=0 twin
  (attribution + Class-D×B recess) · anneal-epochs 600 event-margin (F3 fixed; **backtest RUN:
  first sustained fire ep625 on the real mod32cap trace — one cadence before ep650 best**) ·
  #342 15-block solve inventory produced · dual probability model printed (independent 2-6% /
  with-repair 8-15% run-1 crossing). **CHAIN-A FOLDED (honest negative):** λ₋ collapses ~1/K,
  u_min ISOTROPIC → no cheap spectral descent at ep650; TerminalSolve-from-ep650 measured NO-GO;
  SOLVE branch = measured-acceptance HARD. **P5 SECOND RED-TEAM PASS LAUNCHED** (verify each
  F1-F17 disposition + regression/interaction/launch-readiness).
- 2026-07-07T~21:5x OPERATOR: standard recursive review + deep-math pass demanded → DEEP-MATH
  MEAT HUNT launched as lens 2 (re-derive laws: anneal-completion guarantee, adaptive-ε clamps,
  1-Lipschitz easing, logit-adjust constants, dual-band arithmetic, byte sums · meat vs the full
  compendium universe: chroma-annulus, #207 $0 decode-side winners, #149, hood clamp, persistence
  ORDER, per-class couplings · bug hunt: cadence arithmetic, flag existence, λ=0 twin validity,
  worst-case memory, AA×band pixel-write seam · fresh math: Fisher trust-region metric,
  persistence-weighted loss, τ-ħ endgame, #288 OT). P6 upgraded to RECURSIVE SEAL (3 clean rounds
  × 2 lenses). Lane-anisotropy scope pin delivered to P5 verifier (operator: "lane is anisotropic"
  — the u_min-isotropic negative is measured at a LANE-LESS checkpoint, structurally blind to the
  signal-level lane anisotropy; category error if cited against Rebalance/comb/band/along-tangent).
- 2026-07-07T~22:3x P5 SECOND RED-TEAM DONE (3c707e2f5): **SEAL-TO-RECESS, zero FAILS** —
  verified BY RE-EXECUTION (crossing rows to the digit; ep625 backtest reproduced BIT-FOR-BIT
  from the on-disk trace; 18/18 flags exist; FEED-07g confirmed). 6 PARTIALs bind at recess
  close: (1) AA MEMORY GATE IS AA-BLIND — witness_memory_preflight lacks the supersample term
  (the exact C4 false-green class it exists to extinct) + smoke never fires an n600 verdict
  (~15 LOC + one forced verdict, pre-GO); (2) anneal-SPEED confound un-named (600 = 1.67× faster
  than the backtest's control trace); (3) F11 worst-tail label missing pose-byte leg; (4) F7
  twin needs stage-relative comparison; (5-6) lane-anisotropy pin CLEAN (no lever demoted on
  u_min negative) but scope sentence + registry domain_of_validity caveat owed (tranche 2).
  Regression v1→v2 CLEAN; worst-case bytes close; launchable modulo honestly-labeled LB builds.
  **11-item recess list finalized.** RECESS WAVE 1 LAUNCHED (independent items): R1 LBND4-smoothed
  ($0, kill ≥24,149B) · R3 AA-gate fix+run (the ss² term + forced verdict, pre-GO) · R6 THE
  PARITY ROW (byte-close→decode→n600 verdict vs training-side 0.0035103, kill Δ>+5e-4 =
  fix-before-any-run). Lens-2 meat hunt still in flight (H/I + naive-collapses + openpilot nine).
- 2026-07-07T~23:0x MEAT HUNT (lens 2) DONE (318d9b94b): **NOT CLEAN — 1 BLOCKER · 5 MAJOR ·
  13 MINOR → COUNTER RESETS.** BLOCKER-1: v2 SELF-CONTRADICTS on island-dominant class (§0.3
  lane-dominant vs §3.2 movable 44.8%/lane 19.1%) — crossing design + curriculum λ run on
  SWAPPED classes (req-H violation inside one doc). MAJOR-2: the celebrated ep625 fire lands on
  θ WORSE than ep600 and ep650-best never trains — transition-from-stage-best UNSPECIFIED;
  forfeits up to +2.7e-3 S ("one cadence before best" was a cost spun as validation — main-loop
  self-correction: I cheered it). MAJOR-3: τ_end=0.2 ≈ control's truncation 0.216 → anneal
  "recovery" rides β confounded; FIX = semiclassical τ_end* = margin_q/ln5 ($0, the τ=ħ math).
  MAJOR-4: openpilot GEM confirmed (0.966 > 0.893-0.909; pay-for-polynomial-serve-only-band).
  MAJOR-1: joint λ_bytes law replaces per-section thresholds. MAJOR-5: per-class island weights
  (3.6× asymmetry). MEAT: hood clamp 8-byte no-regret · L3 NTK preconditioner 3-10× speed ·
  kinematic-ξ · MUTCD · v_h pin; #207 exclusion JUSTIFIED (measured dead — main-loop charter
  premise corrected); CERTIFIED: crossing table, byte sums, anneal-600 semantics (source-read),
  Lipschitz, AA ordering. **v3 REVISION LAUNCHED** (folds both lenses + meat adoptions + wave-1
  results-or-branches). Round 2 of the recursive seal begins on v3.
- 2026-07-07T~20:2x CREDIT-DEATH ×3 (v3 designer @17 uses — NO output, clean respawn; chain-A
  @466 uses — TERMINAL VERDICT recovered from uncommitted tree; recess wave-1 @54 uses — partial
  AA-gate fix (+80 LOC) in tree, .md absent). ZERO SIGNAL LOSS after recovery. **CHAIN-A TERMINAL
  VERDICT (recovered + committed):** S3's indefinite-spectrum headline KILLED at full-loss (K=128
  ratio 0.011 — subset noise; my own registered PROVISIONAL equation now needs the kill update,
  tranche 2); ep650 exhausted BOTH orders; **TerminalSolve measured NO-GO** (all solve steps worsen
  on holdouts, fp32+int8) → DEMOTED from the stack; quadratic_basin ExitEvent survives K≥32-
  disciplined (HOLD_STAGE_NEGATIVE_CURVATURE disarmed; basin-entered predicate correctly fires
  finisher/stop at ep650 — agrees with S2's meat exit); **the wall is REPRESENTATION/BASIS — Arm A
  carries the entire burden.** RESPAWN STAGGER: (1) v3 designer relaunched w/ chain-A verdict
  folded (TerminalSolve OUT, req-A(i) updated to the measured negative); (2) recess wave-1
  finisher relaunched (complete+verify the in-tree AA-gate fix, R1, R6). K=128 JSON append =
  optional cheap tail item.
- 2026-07-07T~20:4x OPERATOR IDEA registered (#348): deterministic GPU accumulation via
  fixed-point int64 (associative → order-independent → bit-identical) or fixed-order segmented
  reduction (no atomics) — attacks the L70 MLX-GPU cross-process bit-identity wall at its root.
  Phase 0 = $0 per-op localization (reuse chain-A Link-0 repeatability harness); sweet spot = OUR
  custom Metal kernels (grouped-backward/fused-R — we own the reductions); PRIZE = GPU verdict
  promotion (CPU-locked chunked n600 verdicts → GPU = wall-clock synergy, note for synthesis §8);
  deep cut = full-integer pipeline as the only cross-device (Metal↔CUDA) bit-identity. Sequenced
  post-seal; numpy-fp32 stays THE authority regardless. Queue unchanged: v3 designer + wave-1
  finisher in flight.
- 2026-07-07T~21:xx **DRAFT v3 LANDED** (3 incremental commits e65cfdf76/1097a49e7/83224c801 —
  credit-death-proof checkpointing honored). BLOCKER-1 RESOLVED: the two share tables are TWO
  SURFACES (witness-alone ep225 movable-dominant vs composed ep300 verdict surface
  lane 0.4396/movable 0.1226/big-3 0.4378) with a bridging law (CE birthed 98% of movable mass →
  movable largely solved composed; lane dash-residual dominates); ALL design binds to the composed
  set — §0.3 verified unchanged (already on it), §3.2 REBUILT lane-first, MAJOR-5 weights
  w_lane=1.0/w_movable=0.28 + per-island 1/pers (R13-gated). MAJOR-2: TAU→FIN restores TAU-window
  EMA-best; forfeit printed (+5.4e-4 S event-leg cost vs +2.7e-3 unspecified worst case). MAJOR-3:
  τ_end* = m_q/ln5 ADOPTED → τ_end = 0.062 (flip support GT-margin<0.10; 0.2 was 3.2× above the
  flip-core law value); recovery split β-leg vs τ-leg, 4-9e-4 band DEMOTED to DPR, anneal-speed +
  path-shape confounds NAMED, B9 → PREFERRED. MAJOR-1: ONE KKT byte law at λ_bytes=6.6586e-7
  (0.002 pose threshold deleted — net-negative arithmetic printed). MAJOR-4: hybrid deterministic
  lane-annulus orientation (0.966 vs 0.893-0.909) gated on NEW $0 probe R12 + fallback law — the
  req-I geometry-sharing headline (ONE polynomial → band + comb phase + lane orientation).
  CHAIN-A TERMINAL FOLDED: TerminalSolve REMOVED from the stage graph; §11 rows 11-13 terminal;
  quadratic_basin = K≥32+K-trend SENSOR, HOLD disarmed; lane-anisotropy scope sentence; registry
  kill-update → tranche 2. All 6 P5 PARTIALs + 13 MINORs + meat adoptions dispositioned (hood
  clamp +8 B IN; NTK verified UNBUILT → §9.4 build-spec; MUTCD period + ξ phase; v_h=174 pinned;
  kinematic-ξ + comma2k19 init → P9; mirror-schedule twin; Model A 1.6-6.4% / Model B 3-11%
  run-1-campaign with per-axis estimates; wall-clock 10-27% + B1-contingency). NEW probes: R12,
  R13, τ-confirm. **Round 2 of the recursive seal begins on v3 (both lenses).**
- 2026-07-08T~00:0x **DRAFT v3 LANDED** (respawned designer; checkpoint-early honored:
  e65cfdf76→1097a49e7→83224c801). BLOCKER-1 resolved as TWO SURFACES (witness-alone ep225
  movable-dominant vs composed ep300 lane-dominant; bridging law: CE birthed ~98% movable mass by
  ep300, within_flip 5.3% → solved; lane dash-residual stays) — design binds to the COMPOSED set;
  §0.3 crossing residuals were already on it (re-verified, no crossing number moved); §3.2
  curriculum REBUILT lane-first, movable amplify 1.0→0.28. **τ_end 0.2 → 0.062 DERIVED**
  (τ*=m_q/ln5; 0.2 was 3.2× above the flip-core law — v2 was re-encoding the control's truncation
  by design). TerminalSolve REMOVED (chain-A fold). ONE KKT byte law @ λ_bytes=6.6586e-7 (the
  0.002 pose threshold that could eat the whole 0.00178 margin is dead). Hybrid deterministic
  lane orientation (R12-gated + fallback). Hood clamp +8B IN. Comb: MUTCD period + ξ phase +
  20-ep ramp. Honest: central ≈0.26 does NOT cross; crossing triple verified (0.0011 ∧ 3e-5 ∧
  0.062 → 0.18932); Model A 1.6-6.4% / Model B 3-11% run-1 (8-15% only with run-1.5 branch,
  labeled). New $0 probes R12/R13/τ-confirm. **SEAL ROUND 2 LAUNCHED** (both lenses, delta-scoped,
  one agent — rate-limit-resilient stagger; wave-1 finisher holds slot 2).
- 2026-07-08T~00:2x CHAIN-A FORMALLY COMPLETE (post-reset self-resurrection; commits
  ebd9ccaaa/91b3d37d4/42fa00812; 629 tool-uses, $0, no launches). **PROVENANCE CORRECTION to the
  recovery commit:** recovered draft cited K=128 ratio 0.011; FINAL converged numbers are
  λ₋=−16.85 / ratio 0.163 / 1/√K-extrapolated ≈0.08 at full P (strict kill <0.1 not formally
  reached; persist >0.5 decisively excluded). The DECISIVE evidence is the TRANSFER TEST: holdout
  curvature along u_min ±1.2 vs −175 claimed = 150× smaller + sign-unstable → subset noise;
  u_min isotropic across curvelet/along/across columns; g_true ≈0.08 (exact 1/√K scaling), per-
  pair σ≈2.0 = near-stationary; ALL solve steps worsen holdouts (the one int8 "gain" −6.1e-4
  flipped +5.8e-4 on a disjoint holdout — winner's curse caught). BINDING INSTRUMENT FACTS:
  analytic HVP ≠ true landscape ~35% through uint8-STE → measured-acceptance only; K≤8 spectra =
  NOISE for DECIDE (require K≥32 + K-trend); extreme Ritz magnitudes fp32-path-fragile (2×
  CPU/GPU). Verdict unchanged from recovery: TerminalSolve NO-GO · HOLD disarmed · basin sensor
  K-disciplined agrees with S2 meat-exit · Muon-quench sharpened (no 2nd-order gold at ep726) ·
  wall = BASIS, Arm A carries the burden. TRANCHE-2 NOTE: the PROVISIONAL spectrum equation's
  kill-update must cite the FINAL numbers (0.163/extrap-0.08/transfer-150×), not the recovered
  draft's 0.011.
- 2026-07-08T~00:4x REQUIREMENTS J/K/L BOUND (precision+margin-denominated significance ·
  native-format+capacity per technique · scaling-to-asymptote under infinite compute + separatrix
  asymmetry). **NEGATIVES SCALE-VALIDITY RE-REVIEW LAUNCHED** (12-item inventory: chain-A
  exhaustion verdicts, meat-exit, FEED-08l, #207, viscosity, LEVER-4/UniWARD symmetric-measurement
  suspects, hosc, subset-solve, basin-scoped wall + v3's asymptotic-tail check). Rationale: every
  negative was measured at ONE coarse schedule point (τ 0.216-frozen, 1000-ep) vs the PR95
  29,650-ep/8-stage reference — exhaustion is schedule-relative; exits should refine, not
  terminate; symmetric probes may average away one-sided separatrix effects. 3 in flight:
  seal-round-2 (J+K pins) · wave-1 finisher · negatives re-review.
