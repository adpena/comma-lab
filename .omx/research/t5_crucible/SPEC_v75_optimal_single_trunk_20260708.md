# SPEC — v7.5 OPTIMAL SINGLE TRUNK (the crucible launch vehicle) — 2026-07-08

STORES CONSULTED: ORCHESTRATION_LEDGER (full session) · SYNTHESIS_seal_v73_round2 + all r2/r3/r4
lens reports · crucible_v73_compile + LAUNCH_PACKAGE_v7 · v75_birth_counterforce · road_anomaly_probe
· probe_PA_paintfloor_perclass · perclass_carriers_design · DAG FEEDs v7crux/mbcrux/v74r3fix/
roadfloor/roadfloorfix/missingforces/PA/v8risks/mergediff · run-1 telemetry (ep0-166) · canonical
equations (chan_vese_area_constraint_birth_balance_v1 · safe_compile_hosc_device_bitidentity_v1 ·
frozen_scorer_forward_batch_dependence_v1 · tail_stop_forfeit_floor). Author: outgoing Fable session
(operator-directed handoff). Pointer contest-CPU **0.19110 UNMOVED** — everything here is MEANS; the
END is a byte-closed `upstream/evaluate.py` n600 exact row < 0.19110.

## 1. WHAT v7.5 IS

The sealed EVENT-mode single-trunk witness config (`crucible_v7` in `tac.witness_autoconfig`,
compiled by `_build_crucible_v7`, launched ONLY via `tools/launch_witness_run.py` governed path).
7 DSL levers: seg_form_unify_tau · tail_k_warm_restart · n323_ladder_island_homotopy ·
FEED_07a_directional_basis_rebalance · R7_polyak_finisher · + the 2 counter-force levers
(Chan-Vese area constraint; birth-completion event). EVENT mode is the OPERATOR'S BINDING DECISION
(verbatim in LAUNCH_PACKAGE: "We want to transition to event based now and accept the risk, this
is a new baseline") — do NOT re-litigate; the two-token clock revert is documented there.

## 2. SEALED CONSTANTS (all derived/measured, provenance in the named artifacts)

| constant | value | provenance |
|---|---|---|
| epochs | 3000 CONFIG-SEALED (launcher `--epochs` default=None resolves it; NEVER pass explicit) | NEW-1 fix + tests |
| hosc_beta_end | **3.177** (event-frozen value := clock β(726)=1+3·725/999; trajectory bounded [1.0,3.177]; ≤4.0 divergence bound; GPU bit-cert covers [1,10]) | r2 BLOCKER fix, r3/r4 re-derived twice |
| budget | **8.314 d** = amortized 3.47 min/ep (r_ss ep25→125 = 3.4537 + startup 59.5/3000); refuse ceiling 3.99 | DM-MINOR-1, re-derived from raw ts twice |
| Polyak finisher | start = 2546 (averages exactly 455 ep); degenerate clamp start=epochs+1 (count=0 verified over the real loop) | A3 + r4 |
| persistence-classes / logit-adjust-classes | **"3"** = class INDEX Movable (canonical order Road0/Lane1/Undriv2/Movable3/MyCar4 — NEVER "first 3") under lane_offloaded; companion laws persistence_/logit_adjust_classes_for_basis_regime agree | A5 + counter-force lever-3; consumer traced trainer L4061 |
| Chan-Vese λ | lane **683.8** / movable **322.6** DERIVED-LIVE = W_birth/(δ·A_GT_c), δ=0.25; 51×/14× dominance at ep125 runaway; equilibrium 1.25×GT returns ~0.1145 area = **~96% of** the 0.1189 Road deficit (substantially un-floored, NOT fully at δ=0.25 — CORRECTED from the earlier '≥ deficit' overstatement per R5 confound MINOR-1; last ~4% rides higher δ / the λ A/B). λ-scale = ASSUMED_AWAITING_VERIFICATION | chan_vese equation module |
| lane band coupling | lane_offloaded ⇒ `--lane-render-band` ASSERTED at compile (fail-loud, both entry paths) | F-3 |
| perf env | GROUPED_BACKWARD=1 (~17×) + PERSISTENCE_POOL=1 (D16, loud fallback + fingerprint gate) + `--safe-compile-regions hosc_activation` (b2 fingerprint-gated; cert = per-{chip,os,mlx,device} EMPIRICAL fact, never transfer) + fused-R (bit-identity, L70) | v7.3 compile + crux certs |
| event-gate telemetry | fire rows carry sensor_data_epoch in the LEVER frame (= _fired_epoch frame; additive shift cancels ⇒ lag = real verdict-cadence lag). FRAME CONTRACT in ExitEvent docstring — never mix frames | F-1 + r4 affine proof |
| micro-batch | OFF (scorer forward is batch-DEPENDENT: GPU 2.26e-2 drift/11 argmax flips — bit-identity-at-speedup impossible; bounded n600 d_seg A/B is the only admission path) | frozen_scorer_forward_batch_dependence_v1 |

## 3. WHY THE COUNTER-FORCE EXISTS (the run-1 lesson — binding context)

Run-1 (crucible_v6 clock-mode, pid 63069, sacred READ-ONLY) measured the birth-stack
recall-without-precision imbalance: Lane painted 13.8× GT area, Movable 4.6×, EXACT mass
conservation with the Road+Undriv deficit (0.1191≈0.1189) ⇒ Road d_seg FLOORED ~0.40 ⇒ composite
floored ~0.10-0.13. NOT the analytic band (falsified 3 ways: gated to ep350, init lane_px=0,
1-4K px ≪ 18K needed). Run-1 stays valid AS the birth-arm measurement — never compare it to
bulk-only floors (#205 CE 0.005). The Chan-Vese term is the missing area-Lagrange of the level-set
energy (operator: "the level set and Morse smale are perfect for engineering the precisely desired
annealing behavior"); its equilibrium IS the spec — no ramp schedule to tune.

## 4. OWED BEFORE LAUNCH (the exact remaining chain — DO THESE IN ORDER)

1. **Lever-2 RAMP ACTUATION** (builder a4bb7ebf536596b2f IN FLIGHT at handoff): per-class ramp-down
   of birth pressures on completion-event fire; resume-safe __bc_* additive; derived post-birth
   levels from the Lever-1 balance; default-OFF byte-identical; appends RAMP-LANDED to
   v75_birth_counterforce memo. If it died: its brief is reproducible from the memo's HONEST SCOPE
   section — respawn with the same constraints.
2. **P0 FORCES phase-2 code wave** (operator P0 elevation, task #360; phase-1 LANDED f7209667a:
   .omx/research/p0_forces_derivation_20260708.md = the mechanical spec). SETTLED by phase-1:
   4 forces → 3 (R-phase FOLDS into tie-locus — do NOT build two terms); the tie-locus term is
   ALREADY BUILT in the trainer (~L4559, default-off — wrap + FEED-PA edge weighting only);
   δ_R = 0.0196 MEASURED (m_safe ≈ 0.06; n600 re-run of the existing tool rides phase-2);
   temporal = stop-grad GT-ξ default (pose-safe; live-ξ arm separate + tripwired), w_t 0.1
   cold-start + stage-boundary ramp; satisficing anneals in at l7 (masks, never replaces CE).
   Phase-2 = 2 new DSL levers + 1 wrap, default-OFF, from the spec verbatim. FIRE ONLY when the
   ramp builder (#361) frees the trainer/DSL files. Activation: one force per crucible increment,
   ≤15% each / ≤40% total loss share.
3. **SEAL rounds → 3 clean passes** (counter 1/3 after round 4 CLEAN). Round 5 = all 4 lenses on
   the composed delta (v7.5 counter-force + ramp + P0 levers). Protocol: fix-ALL severities per
   round (zero-unfixed precondition); reviewers RE-DERIVE from primary artifacts (never confirm
   memos); verdict_scope on every negative token; blind derivation for structural claims
   (allowlist committed BEFORE memos — vindicated 2×); [no-triality] only for true apparatus.
4. **Knee re-derive (D2)** on run-1's checkpoint at stop (deferral ledger row, named trigger).
5. **Decode wall-time re-measure** on the FINAL checkpoint (30-min budget binds on contest
   hardware — T4 16GB or CPU 4c/16GB; #214 measured 6.59 min torch-fp32-CPU on M5 Max at
   contest-shape thread-cap; the 1:1 Linux x86 replay is the unproven-margin caveat).
6. **Governed STOP of run-1** (operator standing GO for stop+relaunch AFTER seal completes;
   anything anomalous surfaces first; PRESERVE every per-stage checkpoint).
7. **EVENT-mode LAUNCH** through the FULL gate chain of `tools/launch_witness_run.py` (b-perf ·
   b0.5 zero-naked · b0.6 DSL manifest VERIFY · b2 safe-compile fingerprint · memory preflight ·
   system admission — the governor REFUSE during run-1's life is CORRECT accounting, clears at
   stop). Raw-python bypass = FORBIDDEN P0. No --epochs.
8. **Stop-window items armed** (one-line reactivations in their memos): D1 GPU-verdict agreement
   probe · D17 safe-compile whole-step bench · D18 truncate-at-export A/B · micro-batch bounded
   n600 A/B (~day-1, admission-gated) · #357 speed bundle · #356 megakernel.

## 5. WATCH-LIST (full facet set — operator holistic directive; LAUNCH_PACKAGE has the table)

Per-class d_seg vs anchors — **Road PRIMARY, now a SUCCESS TARGET not just an alarm** (counterforce_insufficiency_deepmath 87bb5adaa): the Chan-Vese area constraint should drive Road within_flip ~0.38→**≈0.018** (0.015 δ-residual + ~0.002 placement); DECISION RULE — if Road FLOORS at ~0.015–0.035 and STOPS, that is the PLACEMENT floor (area-theft solved) ⇒ the next lever is P0 Force-3 (tie-locus displacement), NOT more area constraint / not a basis-helps claim. If Road stays >0.30 @ep200 the area constraint itself under-fired (check λ/δ). run-1 baseline Road 0.392@ep150, NO counter-force) · island birth (part_frac vs GT area — now
ALSO watch the Chan-Vese equilibrium: part_frac should approach ≈1.25×GT then hold; OVERSHOOT
persisting ⇒ λ-scale wrong, the pre-registered response is the A/B anchor) · **POSE (the blocker:
run-1 d_pose 1.80 ⇒ contribution ≈4.2 of S; need ~3e-5-scale for ~0.018; flag if no
order-of-magnitude descent by mid-run)** · rate (blob ~88KB) · CE deceleration vs the event
schedule's exits · liveness/spike-guard/jitter rows. EMA-shadow lag caveat on early verdicts (L-memory).

## 6. MEASUREMENT AUTHORITY (unchanged, absolute)

Only `upstream/evaluate.py` on exact archive bytes, contest-CPU/CUDA 1:1, is a score. All local =
[macOS-MLX research-signal]/[macOS-CPU advisory], NON-PROMOTABLE. Byte-close via
tools/levelset_byte_close_and_eval.py — 3-arm selection (ema/live/polyak) now RECORDED (B1);
per-arm d_pose at export is the pose facet's first real measurement. n600 or it is not evidence.
GT class order is canonical comma10k [Road,Lane,Undrivable,Movable,MyCar] — NEVER luma-sort.

## 8. OPERATING CONTRACT (BINDING on every session/agent touching v7.5 or v8 — operator-directed handoff hardening)

### A. RESUMABILITY IS P0 FOR ALL (operator verbatim 2026-07-08)
Every run: resumable-from-disk + per-stage checkpoints + EMA-shadow save + atomic writes
(CLAUDE.md non-negotiable — loop-end-only saving FORBIDDEN). Every NEW lever/controller: register
under the canonical resume registry (src/tac/witness_control/resume_registry.py static gate) with
ADDITIVE, legacy-compatible persistence (the __bc_*/__mg_* sentinel pattern: old sidecars restore
to un-fired ⇒ byte-identical pre-fire). Every builder: commit early and often via the serializer
(two builders died on transient API-500s today and survived ONLY because edits were on disk).
Never half-wire under unverifiable resume risk — do LESS but REAL (the Lever-2 precedent).

### B. ALREADY SETTLED — do NOT re-derive, re-measure, re-open, or re-litigate (the artifact is the authority)
| settled | authority |
|---|---|
| β_end 3.177 (derived 3× independently); epochs 3000 sealed (NEVER pass --epochs); budget 8.314d | r2-r4 seal reports + tests |
| EVENT mode is the operator's BINDING decision; two-token clock revert documented, not proposed | LAUNCH_PACKAGE |
| `--persistence/logit-adjust-classes "3"` = Movable class-INDEX (consumer traced); class order = canonical comma10k [Road0,Lane1,Undriv2,Movable3,MyCar4] — luma-sort is WRONG (bit us 3×) | r3 confound lens; CLAUDE.md §SegNet |
| Road floor actuator = birth-stack recall-without-precision, NOT the analytic band (falsified 3 ways) | road_anomaly_probe b9da25aa6 |
| Micro-batch bit-identity-at-speedup IMPOSSIBLE (scorer forward batch-dependent) — bounded n600 A/B is the ONLY admission | frozen_scorer_forward_batch_dependence_v1 |
| Bulk-paint interiors near-free AT THE ORACLE BOUND; residual = 100% separatrix placement; Road = adjacency hub | probe P-A bf1ee1fa8 (n600) |
| R-phase FOLDS into tie-locus (do NOT build two terms); tie-locus term ALREADY BUILT (trainer ~L4559) | p0_forces_derivation f7209667a |
| δ_R = 0.0196 (tool exists: tools/measure_delta_R_noise_floor.py — for n600, RE-RUN THE TOOL, never rebuild) | reports/delta_R_noise_floor.json |
| Pose 3.4e-5 is ANCESTOR-BORROWED; this vehicle's d_pose is OPEN (~1.8) and THE blocker | L68 + run-1 verdicts |
| Run-1 = the birth-arm measurement; NEVER compare to bulk-only floors (#205 CE 0.005 / mod32cap) | FEED-roadfloor |
| Safe-compile/bit-identity certs are per-{chip,os,mlx,device} FACTS — never transferred, always fingerprint-gated | safe_compile_hosc_device_bitidentity_v1 |

### C. NO-STRAY / NO-HALLUCINATION RULES
Config exists ONLY as a typed DSL WitnessProgram compile (never hand-assembled argv; never-invent-
flags — grep add_argument or the DSL validates fail-closed). Every constant rides the value-
provenance ladder (derived-live > derived-at-config > measured-anchor > waivered — bare literals
are a bug class). Every negative verdict carries verdict_scope at the NARROWEST supported level.
Every decision doc carries STORES CONSULTED. RECALL BEFORE DECIDE: grep the DAG + memories + these
two SPECs + the ledger before concluding anything — if it feels like a new idea, it is probably in
FEED-* already. Loss weights adapt at STAGE BOUNDARIES only, never per-step (the GradNorm-would-
have-muted-the-canary warning). Council-flagged equations are NOT registered until their anchors
land.

### D. EXECUTION GUARDRAILS
Heavy launches ONLY via tools/launch_witness_run.py (raw-python bypass = P0 FORBIDDEN; a governor
REFUSE is correct accounting, not a bug to work around). pid 63069 + all run dirs READ-ONLY sacred;
preserve every per-stage checkpoint. Serializer --patch-file on hot files + post-edit sha +
post-commit `git show --stat` verify (two absorption incidents today — the discipline is not
optional). Seal protocol: fix ALL severities per round; 3 consecutive clean passes; reviewers
RE-DERIVE from primary artifacts (a memo is a claim, not evidence); structural claims get BLIND
derivation (allowlist committed before memos — vindicated twice). Tasks marked done only when
landed AND verified. MPS never authority; n600 or it is not evidence; only byte-closed
upstream/evaluate.py rows move the pointer.

### E. THE CATHEDRAL INVARIANT
Every unit of work ends with: (1) pointer-delta honesty (0.19110 moved or not — say it plainly),
(2) the triality legs it touched (DSL / DAG FEED / equations), (3) a durable committed artifact.
A chat-only insight is a lost insight. When a wall appears, one crisp verdict then pivot — never
a second unit characterizing the wall. When in doubt: smaller and real beats larger and fake.

## 9. LEVER SCOPE — BUILT vs ACTIVATED (operator scope question 2026-07-08; binding)

Two distinct states, never conflated:
- **BUILT / registered / triality / tasks / costate-surfaced — ALL new levers, always.** Every
  lever is a DSL `Lever` factory ⇒ `lever_registry.completeness()` auto-derives it; the activation
  ledger holds it with duty-to-measure `{default, derived-reason, ever_fired, last_measured_verdict,
  state}` ⇒ the #247 costate SENSE layer ranks never-fired high-value levers into DECIDE + the
  Stop-hook nags. Applies to the counter-force levers AND the P0 forces. VERIFIED (not assumed) by
  task #363 (gated on #361 — the query imports the mid-edit DSL).
- **ACTIVATED (composed ON) in the FIRST v7.5 launch — counter-force ONLY.** Chan-Vese area
  constraint + birth-completion event + ramp compose ON (the launch-critical Road-floor fix;
  dsl_levers 5→7). The 3 P0 forces (temporal screw-consistency · margin-band satisficing ·
  tie-locus displacement) default OFF and activate ONE PER CRUCIBLE INCREMENT with measured A/B
  justification — mandated by their own derivation (≤15% loss-share each / ≤40% total; satisficing
  sequenced ≥ l7; attribution requires isolation). Turning all three on by default is a SPEC
  VIOLATION (confounds attribution + risks term_domination), not a completeness win.
- **Why default-off is NOT orphaning:** the tracked-ranked-nagged queue (costate + activation
  ledger + Stop-hook) is the anti-orphan mechanism, per CLAUDE.md "default-off is orphaned signal"
  — "off" is a queue state the controller drains, never a grave. The failure mode to guard is
  "built but never registered in the ledger" — that is exactly what #363 verifies to zero.

## 10. #363 COVERAGE VERIFY — MEASURED RESULT (2026-07-08, Opus)

VERIFIED (registry/composition axis): all 6 new levers are DSL `Lever` factories
(AreaConstraintBirth [Chan-Vese] · BirthCompletionEvent · TemporalScrewConsistency ·
MarginBandSatisficing · TieLocusDisplacement); crucible_v7 composes AreaConstraintBirth +
BirthCompletionEvent ON (autoconfig L2394/L2405), the 3 P0 forces ABSENT from composition
(default-off per §9) — MEASURED in-source, not asserted.
VERIFIED (activation-ledger / costate axis) — RESOLVED 2026-07-08: my first re-verify queried the
WRONG module (`lever_registry`); the correct entry point is `tac.witness_dsl.activation_ledger`.
Measured there: `known_levers()`=51 includes ALL 5 new levers; `duty_to_measure()`=50 owed
includes ALL 5; `never_fired()` includes ALL 5 (correct — default-off, no run has fired them).
This is exactly what `tools/costate_digest.py::section_duty_to_measure` reads at SessionStart ⇒
the costate controller SURFACES, RANKS, and NAGS the P0 forces + counter-force levers.
Independently confirmed by me, not just the builder test. (The #332 gap is DISTINCT: it is the
older non-factory levers absent from activation_ledger; the NEW levers are fully wired.)
PRE-EXISTING GAP (routed to #332, NOT a v7.5 regression, NOT a seal blocker): completeness().
unmapped = 123 flags; **26 are genuine designed levers** (--logit-adjust-classes,
--persistence-classes, --*-start-event gates, --closed-loop-*, --curriculum-reanchor-levers,
--margin-saliency-reachability, --pose-carrier-*, --seg-chroma-boundary-*, --eikonal-* cluster)
that are score-relevant but NOT DSL-Lever-held ⇒ NOT in the costate duty-to-measure queue ⇒ the
signal-loss surface #332 exists to close. The regime-derived companions (--logit-adjust-classes,
--persistence-classes) are set by companion LAWS from the held DirectionalBasisRebalance lever,
so they are law-derived config outputs, not free orphans — but completeness() still flags them.
