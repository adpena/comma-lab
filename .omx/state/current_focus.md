# Current Focus - 2026-07-31 (ENDPOINT DECISION RECORDED: no window_04 by CAP not by gate; cn3's reversal read WITHDRAWN — the rise itself reversed; gd1 proves the gate amplifies drift 16.7×)

> **⚖️ ENDPOINT DECISION LANDED (`/Volumes/VertigoDataTier/pact/ddm_b4s_20260731/burn4_endpoint_decision_MAIN.json`).**
> Three parts, each receipt-grounded:
> **(1) NO WINDOW_04 — on GOVERNANCE, not measurement.** b4r's prospective wall-cap check already
> decides this mechanically: a window may legally run to its own 130-min trainer wall cap, so admitting
> one at the ~4.4h boundary would end ~6.5h > the 6.0h cap / 20:17:42Z hard stop. The endpoint
> MEASUREMENT stage is explicitly NOT truncated by that cap.
> **(2) cn3's REVERSAL READ IS WITHDRAWN as a decision input.** It reported mid-audit "7 monotone rises,
> 347% of the boundary step surrendered, OLS t=+6.28 ⇒ B5-C." The full trajectory shows the rise was a
> TRANSIENT that has itself reversed: ep884 0.0040827 → ep914 **0.0043281 (peak)** → ep919 0.0042619 →
> ep924 0.0041589 → ep929 **0.0041162**, three consecutive COUPLED_DESCENT recovering 0.0002119 of the
> 0.0003879 rise. Correct read at the time; wrong basis for a terminal verdict. **Two INDEPENDENT gd1
> (#817) findings say this instrument can never carry one alone:** the A1 gate reduces a NON-probability
> sample (block 447–450 + 32 SRS) with an **unweighted mean**, over-weighting the block 16.67× ⇒ it
> **amplifies block drift 16.7×**, and the block is **+6.3–8.3% harder** than the population and
> **−16.2% in Lane**; and `n_points=5` × `--gate-every 5` **aliases a ~30-gate oscillation into a sign
> flip** — a 7-gate monotone run is ≈1 alias period. L3 verdict-clearance: apparatus-validity
> precondition FAILS ⇒ no load-bearing verdict admissible from the gate alone.
> **(3) ENDPOINT SELECTION RULE = rank on the gate, DECIDE on n600.** Every gate epoch wrote a
> checkpoint (~54 candidates). Measured gate↔n600 relationship: at ep805 gate 0.0040519 vs
> full_confirm **0.0040671** — the LEVEL is nearly unbiased (0.374%) but the DELTA overstates the descent
> by **~7.6%** (independently matching gc14's 7.2%). Shortlist to n600: **ep809 (0.0039402, window-best
> AND the boundary state = highest information)**, ep854 (0.0039406), ep879 (0.0039510), + the final.
> Control ALREADY measured — do not re-run: window_02 ep805 full_confirm **0.004067128**.
> **HONEST BURN-4 DESCENT TO DATE (n600, not gate):** 0.004277157 (w01 ep665) → 0.004067128 (w02 ep805)
> = **−0.000210029 d_seg = −0.0210 S** `[macOS-CPU advisory]`. Pointer **0.1910828242 [contest-CPU] UNMOVED.**
>
> **What this does NOT decide:** whether the descent is optimizer artifact or learning. gc15's source
> derivation (`bias_correction=False`, `train_tr1:1543`) predicts EXACTLY the ep809 boundary gain.
> **#815 Arm B′ (`bias_correction=True`, byte-identical otherwise, ~2h, $0) retains FIRST CLAIM** on the
> post-endpoint slot. cn3 proposed firing burn-2 there instead — that CONFOUNDS composition with
> optimizer. **ORDER: B′ (isolates optimizer) → burn-2 (isolates composition, at B′'s decided setting).**
>
> **Hygiene done this turn:** orphan PID 77097 (`observe_m1_banded_checkpoints.py`, retired M1 lineage,
> **10d 15h**) REAPED. #817/#818 flipped completed.
>
> **⛔ THE STRUCTURAL ROOT the operator's stub correction pointed at — cn3 MEASURED it (lines 135–143):
> ALL THREE anti-orphan gates are VACUOUS.** `lever_registry.completeness()` returns `stale=[]` while
> ASTing **1 of 171 modules (0.6%)** — `_module_source()` (`lever_registry.py:107`) returns only
> `curriculum_dsl.__file__` — which is *why* the 5 DESIGNED-STUB fh1 forces never surfaced: the registry
> cannot see the file they live in. `check_codex_findings_memos_consumed` reports LIVE COUNT 0 while
> scanning `mtime<3days` ⇒ **0 of 1,260 files**. Catalog #396 has the right scan set but **433 live
> violations, never strict-flipped**. A gate returning a clean marker over 0.6% of its domain IS NO-FAKE
> class #1 at the gate layer. **ROUTED to sb2 (#819) as in-scope**, with the instruction to fix the
> registry scope FIRST and re-run its inventory, because any stub count taken before that fix is an
> undercount by construction.

# Current Focus - 2026-07-31 (gc15 DERIVES THE MECHANISM FROM SOURCE: bias_correction=False ⇒ every boundary is an unpriced 3.16–6.57× LR SPIKE)

> **⚡ gc15 (#816 DONE, ca4850be52) — THE SOURCE DERIVATION.** MLX `Adam` defaults
> **`bias_correction=False`** and `train_tr1:1543` NEVER overrides it. With zeroed moments the
> effective step multiplier is **η(t)=(1−β₁ᵗ)/√(1−β₂ᵗ) — 3.16× at t=1, PEAK 6.5685× at t=12**,
> decaying over ~1000 steps. At the burn's measured argv (batch-pairs 8, 600 pairs ⇒ 75
> steps/ep, lr 2e-3) **EACH BOUNDARY INJECTS ~1,212.6 extra sign-steps ≈ 16.17 EPOCHS OF FREE
> DISPLACEMENT, 81.7% inside the first 13 epochs.** This DERIVES ALL FOUR gc14 boundary
> observations (step · flat in-window slope t=−0.09 · ep_loss spike at normal gnorm · ratio
> 1.36) **from source with NO new experiment** ⇒ **the campaign's only measured seg descent this
> month is plausibly an OPTIMIZER ARTIFACT, not learning.**
> **THE $0 SETTLER — ARM B′: `bias_correction=True`, all else byte-identical, ~2h.** Highest
> information per dollar in the campaign; alone separates H1(spike) from H2/H3(conditioning/
> exploration). FIRES FIRST at the window_03 endpoint, ahead of everything in #815. If the
> descent vanishes, every seg verdict since needs re-grading against a corrected baseline
> (r̂=0.310 itself must be re-derived from bias-corrected data before R3 can bind).
> **OPERATOR Q ANSWERED (fresh-vs-warm):** NEITHER POLE. Warm beats fresh MECHANICALLY (k windows
> = k kicks PLUS the preserved path; fresh gets one kick and must re-learn) ⇒ **do NOT birth
> fresh for the seg axis**. But restart count is a KNOB, not a property of warmness, and the
> bigger lever is WHAT WE RESET *TO*: zero makes the post-reset step a uniform SIGN step = the
> maximally generic metric-free direction, violating our own generic_basis_metric_never_optimal
> law + sy1 S1-POLICY, at the ONE point where inserting the custodied frozen-scorer metric costs
> ZERO extra wall-clock (we discard the metric anyway) ⇒ **a STRUCTURED RESET ON A WARM RUN
> plausibly DOMINATES BOTH POLES.** Caveats NOT softened: the benefit may BE the disorder (arms
> magnitude-matched, vary DIRECTION only) · H5 pure time-travel (a bigger step arriving sooner
> where the run was already going).
> **MAIN CORRECTED:** my "the protection stack never ran from ep0" argument is OVERRATED —
> **5 of 6 forces are DESIGNED-STUB**; Carmack: *wiring is the blocker, not birth*.
> **FRESH-RUN DECISION FUNCTION:** FIRE ⟺ T1 ∨ T2 ∨ T3. **T1 STRUCTURAL-IMPOSSIBILITY is the
> only currently-TRUE trigger, ONE instance: QA84 rowband D8** — no D8 ckpt exists anywhere
> (every tr1 ckpt is D16), RowBandGrammar BUILT, `--token-rowband-spec` exists, band
> [160,240)/1248 theorem-certified rate-optimal ⇒ warm CANNOT buy a D8 parent. T2 matched-compute
> reversal PROVISIONAL-TRUE for QA24 (fires only on re-measurement). T3 path-dependence ceiling
> E≥0.30 UNMEASURED (Rudin: bare constant, needs derivation). Config if fired: rowband D8 ×
> solve_project init × **kd_warm_start_dir ← the warm endpoint** × the §7 reset winner; ~13h, $0.
> **RE-GRADES:** QA24 from-birth **MEASURED-DOMINATED → CONDITIONALLY RE-OPENED** (bc1@ep399 vs
> warm@ep641 = +60.7% compute, unmatched bytes; at matched ep499 the sign REVERSES to fresh
> ~0.035 S ahead — PROVISIONAL/extrapolated). **K-FAC 07-13 DISCARD conditionally re-opened at
> FORMULATION scope** (both stated grounds fail for a boundary insertion).
> **ORPHAN NAMED:** the DEFERRED from-birth-KD and rowband D8 cells are **ONE RUN**, and their
> actuator (#74/#129 KD-warm-start, BUILT w/ 6 NO-FAKE tests) has **NEVER FIRED** — 6 weeks old.
> **#815 AMENDED:** 5-arm magnitude-matched reset-operator race (A no-reset · B zero-reset
> incumbent · B′ bias-corrected · C momentum-only · D± scorer-metric from ms4d+#725+rank-4) with
> H1–H5; **φ-preflight gates D± with pre-registered withdrawal at φ>0.8** (fraction of a uniform
> kick landing in range(A) — Yousfi upgraded it to campaign-level diagnostic); **cadence A/B
> DEMOTED 1st→4th** (the impulse is now a DERIVED constant).
> POINTER NOTE (gc15 accuracy): canonical_frontier_pointer.json's local contest-CPU field carries
> **0.18804** = the Modal 07-12 PR128-on-PR110 **NON-SUBMITTABLE bank**; the SUBMITTABLE custody
> row is **0.1910828242**. BOTH UNMOVED.

# Historical - 2026-07-31 (gc14 VERDICT: the "descent" is plausibly a BOUNDARY-STEP APPARATUS ARTIFACT — 3 MAIN claims falsified)

> **⛔ gc14 (#814 DONE, 0068807a62) FALSIFIED THREE MAIN CLAIMS — carry the corrections:**
> (1) **"λ_Lane guard MEASURED WORKING" is FALSE** — `lambda_lane` was exactly **0.0 at all 38
> gates**; the guard NEVER actuated ⇒ lg1 is **UNTESTED, not validated**. (2) **"Lane strongly
> birthing +42 t=+5.54" is a 5-GATE PHASE ARTIFACT** — full-window slope −0.497 (t=−1.50),
> all-38-gate −0.158 (t=−0.69). (3) The 36-pair gate **overstates descent by 7.2%** ⇒ honest
> n600 net = **−0.018303 S**, not −0.019934.
> **THE FINDING — BRANCH (c) BOUNDARY-STEP / APPARATUS ARTIFACT (currently winning):** over
> window_02's 29 gates realized d_seg OLS slope = **−1.46e-7/gate, t=−0.09 (statistically
> ZERO)** while ep_loss fell 13.4% ⇒ **139 epochs of real loss descent bought NO realized
> d_seg**. The level moves as a STEP at the boundary: w02→w03 −1.118e-4, then flat;
> **|step|/|window Δ| = 1.36** (the boundary jump EXCEEDS the whole window's movement).
> **Mechanism SOURCE-CONFIRMED:** train_tr1 passes `opt_state_flat={}` at every save + rebuilds
> `optim.Adam` fresh unconditionally (L1543) ⇒ **every boundary = a full Adam warm restart**
> (ep_loss spikes +9.6%/+22.8% at normal gnorm), with ema_decay re-derived per window (shadow τ
> 166→202→236 ep) = an **unintended SWA regime**. Causation INFERRED not proven — R1 is the
> experiment. **⇒ "the first genuine coupled descent" may be an artifact of HOW MAIN WINDOWED
> THE BURN, not a property of training.**
> **CROSSOVER ARITHMETIC (kills the 16-window extrapolation):** measured per-window decay
> **r = 0.310** (−0.06769 → −0.02101 S); TWO independent routes agree runway = **1.2–1.8
> windows**; **w04 goes NET-POSITIVE (+0.00068)**; geometric remaining **0.00946 S**; closing
> the 0.36640 S seg debt needs r ≥ 0.9458. **Burn-4 closes 2.3% of the own-vehicle gap.**
> **BURN-5 DECISION FUNCTION (Test 1 gates all; ep834 already returns BOUNDARY-LOCALIZED,
> ratio 1.36):** B5-A restart-cadence A/B · B5-B pool-drain ONLY AFTER R2 · **B5-C hand off seg
> → RATE = THE DEFAULT on current arithmetic** (cell_drop50's banked −0.098 S is **5.4× larger
> than anything burn-4 produced**) · B5-D sustained (pre-registered predicted-NOT-to-fire).
> Point prediction for w03 endpoint: n600 d_seg **0.0040019**, D_tot ∈ [−0.0110,−0.0075].
> **WHAT SURVIVES:** `gt_components_erased` fell **567→508 (Lane −53)** in-loop unaided — the
> strongest live receipt for the ERF-collateral in-loop-birth doctrine (unpriced in S: no
> scorer slot; Yousfi dissent recorded). **GUARD IMPLICATION:** falling betti0 may be
> CONVERGENCE TOWARD GT, not erosion — the predicate has NO GT-reference term ⇒ MAIN's CONTINUE
> adjudication holds but partly for the wrong reason; R2 must precede any guard.
> **#815 QUEUED** (R1 causal A/B w/ derived margin-density-at-zero preflight · R2 predicate
> re-calibration ×2 · R3 derived STOP rule) — fires at the ~18:40Z endpoint.
> **#809 amendments** (memo §16): sensor leg AHEAD of dual leg · GT-reference mandatory ·
> estimator window derived not n_points=5 · boundary_event = first-class ledger row · MyCar
> dual RETIRES (betti0 = GT = 36, zero variance, 38/38) · **window length was a SILENT UNPRICED
> HYPERPARAMETER for the entire campaign** (contrast: the pre-auth bound WAS class-4 labelled).
> Pointer 0.1910828242 [contest-CPU] UNMOVED.

# Historical - 2026-07-31 (RECOVERED + window_03 LIVE-DESCENDING; gc14 16th convocation on the first working descent)

> **RECOVERY COMPLETE (b4r, commits c72d1e7b75/fb58d89af9/1f53a78bdc/b08f750268).** Both procs
> verified dead → ALARM retired to alarms_resolved/ sha 08bce4ca… as
> **UNDRIV_EROSION_ADJUDICATED_CONTINUE** with `supersedes_fields: []` — the arm deliberately
> did NOT overlay the measured field: the derived-ε machinery is CORRECT and the erosion is
> REAL; retired as a PRICED TRADE, never a measurement error (contrast ALARM #1's
> SPURIOUS_PORTED_PREDICATE). Pre-auth bound |net betti0| ≤ 10 labeled **class-4 governance
> value** (owner MAIN, re-derivation trigger = λ_undriv via cg1 #809) — constants-are-poison
> applied to a governance knob. **window_03 FIRING**: ticket c286049aac3f…, ep806→946, trainer
> 49743 / supervisor 65276, launched 17:09:57Z, boundary ~18:40Z, all 6 governed gates PASS,
> argv-diff = only epochs/out-dir/resume (NO reseal needed, zero unsanctioned drift).
> **LIVE-DESCENDING: ep829 d_seg 0.003998 < window_02's 0.004052 endpoint** (in-window a1
> telemetry, NOT an endpoint verdict — early-window rates run hot). Measured window_02 net ΔS
> = **−0.019934 S** (seg −0.022634 + rate +0.002700, w01→w02 baseline; the w01-vs-rung1
> baseline ambiguity is owed to gc14 §3). **CAP UNCHANGED 20:17Z** but enforcement made
> PROSPECTIVE — b4r found the retrospective-only check would have admitted a window_04 that
> breaches by ~33 min; class-fixed across every launch path incl. the lg1 rollback-relaunch
> (new LANE_EROSION_ROLLBACK_EXCEEDS_CAP). window_03 = THE LAST WINDOW. Endpoint bundle
> producer BUILT+verified (experiments/ddm_b4r_endpoint_extras.py: per-class n600 d_seg in
> xp1's convention, partition sum-check 8.7e-19; 5×5 class-pair flip matrix; per-class descent
> rates parent-ep641→w01→w02→w03; Undriv priced in S vs the fl1 floor; ZERO extra SegNet
> forwards). Waiter v3 armed (4×135s sub-call protocol).
> **gc14 (#814) LIVE — 16th convocation, on OPUS (Fable at limit).** Seeded with THE
> DISCRIMINATOR: window_03 tests reachable-headroom-drain (fl1 ~0.026 S ⇒ 1.15 windows runway
> ⇒ flatten) vs FLOOR-PIERCING (renderer learning temporal phase-faithfulness in-loop,
> descending THROUGH the FORMULATION-scoped flicker floors as PR130 does 18× below) —
> per-class decomposition discriminates; Road-at-its-floor ⇒ any Road descent = piercing
> evidence. Plus the unclaimed CROSSOVER derivation (d_seg −0.0226/window decelerating vs rate
> +0.00205/window growing ⇒ computable window where net ΔS flips positive = the DERIVED
> stopping rule superseding judgment-based E2 handoff) + burn-5 decision function +
> phase-faithfulness axis (#425/#535/W1-COH, Lane #1 at 43.6% of floor mass) + us1's
> unconsumed pose geometry (2×2 polyphase luma, camera_fl=910) + cg1 amendments.
> Pointer 0.1910828242 [contest-CPU] UNMOVED.

# Historical - 2026-07-31 (RECOVERY 16:58Z: Fable-5 limit → MAIN on Opus 5 → burn-4 respawned)

> **INTERRUPTION + RECOVERY (no signal lost).** The b4s arm hit a **Fable-5 usage limit**
> mid-execution of MAIN's CONTINUE adjudication (received, zero steps executed). Operator
> switched MAIN to **Opus 5** (deviation from the Fable-main-thread directive — EXPLICIT
> operator-initiated, not a silent reroute). Recovery state at 16:58Z: supervisor daemon
> 83959 + trainer 79174 both **DEAD**; burn4.ALARM NOT retired; window_03 never launched;
> ~3.2h of the 6h cap remain (t0 14:17Z → hard stop 20:17Z) ⇒ ONE full 140-ep window fits.
> Uncommitted hot-state COMMITTED (6498ab4577) before anything else. **ddm_b4r respawned on
> Opus** with full disk-grounded context: verify → retire ALARM (UNDRIV_EROSION_ADJUDICATED_
> CONTINUE, priced trade NOT spurious, append-only supersession) → relaunch window_03 from
> ep805 same seal, fresh daemon, both watches armed → endpoint obligations = **the gc13 R1
> consumption bundle** (per-class re-measure · 5×5 class-pair flip matrix · protected descent
> rates · Undriv exact S-pricing · R6_PAYS/R6_CLOSES). PRE-AUTHORIZED: a repeat UNDRIV alarm
> with |Δbetti0| ≤ 10 AND window ΔS < 0 continues automatically (no dead stop between wakes).
> Waiter re-arms on the fire receipt. MEMORY.md is 18.3KB (>17KB load cap) = hygiene debt.

# Historical - 2026-07-31 (window_02 COUPLED_DESCENT: NEW BEST d_seg 0.0040519 → UNDRIV alarm adjudicated CONTINUE → window_03)

> **WINDOW_02 (ep665→805, 2.03h): THE CAMPAIGN'S FIRST GENUINE COUPLED DESCENT.** d_seg
> 0.0042778→**0.0040519** (Δ −0.000226 ≈ −0.0226 S seg-axis; NEW BEST realized, beats rung-1
> 0.0042641). **Lane STRONGLY BIRTHING: betti0 498→540 (net +42, slope +11.66/gate, t=+5.54)**
> — R6 1.3 + λ_Lane guard MEASURED WORKING (λ stayed 0.0, within budget). Rate +3,074B
> (→276,078B, +0.00205 S) ⇒ net window ΔS ≈ **−0.0206 S**. Birth key NOT fired (births active
> ⇒ chain continues). **UNDRIV_EROSION alarm #2 fired at boundary** (slope −0.799 vs ε 0.633,
> net −3 comps — REAL but small, already inside the net-positive aggregate; guard-era erosion
> SLOWER than unprotected). **MAIN ADJUDICATED: CONTINUE** — b4s resumed to retire the ALARM
> (UNDRIV_EROSION_ADJUDICATED_CONTINUE, priced trade not spurious) + launch window_03 from
> ep805, same seal, watch armed, cap unchanged (hard stop 20:17Z, ~2 windows feasible);
> λ_undriv routes to cg1 (#809) with endpoint calibration, NOT mid-burn. fl1 context: Undriv
> reachable headroom +0.0164 S total vs −0.02 S/window descent — trade clearly favors continue.
> **fl1 (#813 DONE, 0c671abe00):** per-class flicker floors sum-check +4.4e-7; falsifier
> honestly SCOPED (smooth-label floors are FORMULATION-bound, pierced by phase-faithful
> renderers — NO re-waterfill); live value = PHASE-DEBT RANKING Lane #1 (43.6%); Road AT its
> formulation floor (capacity exhausted under current render family); reachable non-flicker
> headroom = Undriv +0.0164 + Movable +0.0095 ≈ 0.026 S; the rest of the corner-C gap =
> phase-faithfulness debt → #425/#535/W1-COH axis after the burn. {Lane, Undriv} guard set
> TRIPLE-CONFIRMED (gc13 receipt × live watch × fl1 headroom).
> **dg1 (#812 DONE, e66f225934+36a1f01f59+4a5504fe51):** dynamic-denominator guard — line-56/64
> asymmetry verified from source; fail-closed at rate_term (83 import sites, one chokepoint) +
> Catalog #407 warn-only-deliberate. **Waiter laws memorialized** (foreground sleep >~3min ALSO
> SIGURG-killed in waiter contexts → 4×135s sub-calls; memory waiter_agent_protocol_laws).
> NEXT WAKE: b4s resume receipt → re-arm waiter → burn4.done endpoint → R1 consumption bundle
> (per-class exact re-measure + 5×5 flip matrix + protected descent rates + Undriv S-pricing).
> Pointer 0.1910828242 [contest-CPU] UNMOVED.

# Historical - 2026-07-31 (burn-4 alarm adjudicated + gc13/us1 landed → b4s fix-chain relaunch pending)

> **BURN-4 ARC:** fired lg1-protected (ticket 6206cf56, daemon 40118) → first ALARM at ~18min:
> term_domination = **seg ITSELF 67.8%** on the lean 2-term loss (seg .349/rate .147) — MAIN
> adjudicated **FALSE POSITIVE by ported-predicate semantics** (v9 intent = NON-scored term
> crowding seg; port fires on ANY >40%; would fire on EVERY telemetry-on TR1 run). NEW LAW:
> **alarm predicates/thresholds are per-vehicle calibration objects** (constants-are-poison →
> the guard system's own sensors; hold+adjudicate chain = the calibration mechanism).
> Window_01 science VALID: re-smoke FLAT (Δ +4.2e-7, below own ~1.1e-6 resolution), Lane guard
> healthy, birth key fired. **b4s RESUMED executing:** scored-term-exempt predicate fix (two-
> landing) → reseal → carry-or-rerun smoke evidence → retire ALARM → +seal-neutral supervisor
> **UNDRIV_EROSION watch** (gc13 receipt: Undriv eroded +0.00204 > Lane +0.00151 in the
> unprotected ep499→641 window — "Lane=the one need" FALSIFIED, INSTANCE scope) → relaunch
> full 140-ep windows. **RELAUNCH DONE (b4s d7d7c2fa87+84b5a695c1): predicate intent-restored
> (scored-term exempt; floor 0.60 DERIVED as caps-law complement; 16/16 tests; per-vehicle-
> calibration doctrine documented) · smoke evidence CARRIED (3-hunk telemetry-only diff receipt)
> · ALARM retired via append-only supersession + overlay reader · window_02 FULL LIVE (ticket
> bf1141d5fa594eb3, trainer pid 79174, ep665→806, ~90min/window, ≤2 more in 6h cap t0 14:17Z)
> · UNDRIV_EROSION watch LIVE seal-neutral (rebuild hash == live hash; daemon pid 83959).
> ⚠ FIRST READING: Undriv watch reads ERODING on window_01 smoke telemetry (slope −0.804/gate
> vs ε 0.650, net −4 over 5pts/24ep — MARGINAL, short window); window_02's 140-ep boundary =
> the clean adjudication; if it fires, MAIN adjudicates (no trainer-side λ_undriv yet — cg1
> build item). Marker waiter v2 ARMED.**
> **gc13 (#810 DONE, 71659bd3d7):** missing shape = the closed bar↔burn loop; truly-optimal =
> discretized Pontryagin TPBVP in scorer coords (per-EDGE dual trajectories from a bar-backcast
> band-box, gate-cadence PID duals + SE deadbands, ONE KKT waterfill settle, distortion-only
> per-class floors, graduation-to-solve, sensor-calibration leg). 3 seed-gap CORRECTIONS:
> rd1 duals EXCLUDED as price source · per-class rate floors ill-posed · guarded set =
> {Lane, Undriv}. Backcast (derived, advisory): corner C 0.16182 sub-bar @130KB · corner D
> 0.14418 sub-0.15 @149KB · 42.6% of 0.36640 seg debt in the two positive-slope classes.
> #809 cg1 AMENDED (11 items). R1=endpoint consumption bundle (per-class re-measure + 5×5 flip
> matrix + protected descent rates + Undriv typing; falsifier: protected slopes ≥0 at λ_max ⇒
> dual form falsified ⇒ #208 projection successor). R3 FIRED: **fl1 arm LIVE (#813)** —
> per-class GT-flicker floors vs corner-C allocations (falsifier: floor>allocation ⇒
> re-waterfill).
> **us1 (#811 DONE, 64030ffc6d):** upstream re-read scorecard **0 CRITICAL** (13 laws re-
> derived+HOLD; drift = operational only: 2 DRIFTED/4 FORGOTTEN/11 NEW). Top: rate denominator
> is DYNAMIC rglob-sum (evaluate.py:64; stray ._*/.DS_Store silently shifts rate → **guard
> task #812**) · rule-118 stakes +62.8 S · PoseNet luma = 2×2 POLYPHASE + camera_fl=910 (pose-
> solve inputs) · timm 1.0.22-vs-1.0.27 parity OWED (advisory-only impact).
> Pointer 0.1910828242 [contest-CPU] UNMOVED.

# Historical - 2026-07-31 (lg1 LANDED → b4s resumed: fold → re-seal → re-smoke → RE-FIRE burn-4)

> **lg1 (#808) LANDED (c009a2e123/c66acf4d79/8afe941864/9ac4aff09d):** scorer-coordinate
> lane-guard WIRED — λ_Lane primal-dual (budget 0.12589 S custodied via
> dsl_custodied_scalar_identity_v1, = xp1's exact ep641 Lane level; η_λ 66.2252 derived
> from the measured 0.00151 unprotected erosion; step cap 0.1/gate, λ_max 5.0; dual reads
> the a1 gate's EXISTING realized argmax = zero new scorer passes) + born-lane protection
> mask (×1.19607 rank-4 head sensitivity) + PIXEL margin floor (p10 of Lane-restricted
> QA80 margins, derived at first gate). DEFERRED: gradient surgery (~1.8× step tax) +
> per-component rank-4 hinge wiring (helper landed+tested, insertion point named).
> Byte-identity: structural default-off + ep0 realized d_seg BIT-EQUAL 0.5078303019205729
> across 8 arms (vehicle is rerun-nondeterministic on both devices — pre/post-OFF inside
> that scatter). 3 DSL Lever factories lever_lane_guard_{lambda,born,margin_floor} +
> --lane-guard-lambda-init warm-λ rollback path. Custody ddm_lg1_20260731/ + lg1.done.
> **HANDOFF EXECUTED:** #808 completed · b4s (a82f149e…) RESUMED with the turnkey fold
> instruction (verify argparse → 3 levers into ticket builder → flip LG1_DUAL_ENGAGED →
> re-seal fresh hash → bounded re-smoke ΔS<0 → RE-FIRE under standing GO, daemon+markers
> burn4.done/ALARM + LANE-EROSION guard armed). NEXT WAKE: b4s fire receipt (then P0
> ledger + marker watch) or burn4.done endpoint. **NEW (operator 07-31): #809 ddm_cg1**
> registered — generalized per-class/per-edge guard LEDGER (targets/caps/duals per
> class+edge+saddle vs realized outcome+telemetry+archive), gated on burn-4 endpoint
> telemetry as its calibration table; burn-4 seal untouched (Lane = the one measured need).
> Pointer 0.1910828242 [contest-CPU] UNMOVED.

# Historical - 2026-07-31 (burn-4 SEALED+HELD: R6+lane-guard config, awaiting lg1 fold)

> **b4s LANDED (e53c0720bf/e5edf0f354/18ee715068, #807 blocked-on-lg1):** all skeleton
> slots FILLED (xp1 P 0.04401 S w/ per-class [Road .188, Lane .126, Undriv .056, Mov
> .038, MyCar .018]). RECEIPTS-FORCED RECONCILIATION: from-birth QA24 rate cell
> MEASURED-DOMINATED by warm ep641 (0.686 vs 0.608, grid capped D16) · continuation-KD
> CLOSED (realization-gap reversal) · paint SKIPPED ⇒ **burn-4 = bounded continuation
> from ep641 + R6 class-weight-lane 1.0→1.3 (never-fired lever FIRES) + telemetry ON,
> seg-only, no injection**; composed advisory 0.564-0.608 — does NOT reach 0.172141;
> from-birth-KD/rowband = DEFERRED from-scratch cells (the residual rate axis).
> Sealed ticket 098b5aea (argv-diff = exactly 6 sanctioned flags, all gates PASS);
> R6-only re-smoke fired pre-amendment then HELD cleanly (killed ep645, 4 ep, procedural
> supersession, custody preserved, burn4.HOLD). **Supervisor carries the LANE-EROSION
> GUARD** (P2 key inverted, derived ε — catches xp1's erode-while-improving mechanism).
> **BLOCKER:** lg1's trainer flags (--lane-guard family, budget default→0.12589 = xp1's
> exact ep641 Lane level) are wired-but-UNCOMMITTED (lg1 resumed after an API drop,
> in flight); seal-freshness recompiles COMMITTED state ⇒ fold waits. ON lg1 LANDING:
> resume b4s (turnkey §6: verify argparse → lever_lane_guard in ticket builder → flip
> dual gate → re-seal → re-smoke → re-fire under standing GO).

# Historical - 2026-07-31 (RUNG-1 LANDED: new best S 0.6082 → burn-4 compose+seal LIVE)

> **RUNG-1 ENDPOINT (manifest /Volumes/VertigoDataTier/pact/ddm_r1c_20260731/):** NEW BEST
> realized point — **n600 d_seg 0.00426407708, S_additive 0.6081898657 @ 273,004 B**
> (parent 0.0049411/0.67325: ΔS **−0.0651** in 1.498h, ONE 140-ep window). stop_reason
> = birth_completion_fired (P2 key: slope −10.30 comp/gate, t=−2.81 — births plateaued/
> declining; the sealed window → key → stop → endpoint-obligations → marker chain ran
> END-TO-END mechanically, halt_events=[]). Burn-4 parent = ep641 ckpt sha 40553db8be98…;
> QA80 NOT_CONSUMED_FRESH_BY_CONSTRUCTION; P 4-conn estimate 474 above-nucleus (exact
> owed → xp1). c telemetry 0.0775 (from 0.0885, cross-burn trend down).
> **xp1 LANDED (cc55db90d5/cff7ce033e, #806):** exact P = **0.04401 S** (8-conn, 4228/9035
> erased; chain validated to ~1e-7 vs the manifest). THE TWIST: the pool GREW +0.00212 S
> during rung 1 — the −0.0651 S win was entirely Road/Movable/MyCar while **Lane
> REGRESSED +0.00151** (two-instrument agreement w/ the birth gate's betti0 534→500).
> Plain continuation now MEASURABLY erodes Lane → the fh1 seg-cell forces (R6
> class-weight-lane · σ_cc′ pair-weighting) are measured-motivated, not just harvested;
> no measured crossing mechanism for the Lane pool yet (KD arm carries its falsifier).
> **OPERATOR 07-31 ×2 ("sophisticated techniques to constrain and protect" + "remember
> the upstream channels and hyperplane and basis"):** burn-4 seal HELD for ddm_lg1
> (#808, LIVE) — the scorer-coordinate LANE-GUARD layer: λ_Lane primal-dual hard
> constraint (budget = ep641 measured level, gate-cadence bounded dual) · born-lane
> protection mask weighted by lane-channel sensitivity (rank-4 head rows + #725 BN
> capacities, #208 lifted to train-time) · margin floor in the EXACT head flip-distance
> metric d=|m|/‖Δw‖ · optional Fisher-metric gradient surgery (dual-metric readback).
> b4s amended: supervisor gains the LANE-EROSION event key (P2 inverted) + window-level
> realized acceptance w/ rollback; protection = constraint layer, orthogonal to the
> raced force cells. ddm_b4s (#807, scorer-free) — fill last skeleton slots → compose the burn-4
> arm matrix (QA24 from-birth rate cell × KD-from-birth seg cell w/ R6/R7 raced × fh1
> force stack per §3b × telemetry ON) → reseal → bounded re-smoke after xp1 frees the
> slot → FIRE under standing GO w/ the adapted daemon+marker supervisor. Honest
> arithmetic: burn-4's rate re-race (~0.19-0.28 S) is the gap-sized lever; ladder does
> NOT reach 0.172141 by itself.

# Historical - 2026-07-31 (BR-D wall branch: THE BIRTH-COMPLETION LADDER adopted)

> **LADDER STATE (gc12 b4d317538d; QA92 LANDED 6c23883be6):** rung-0 QA92 FIRED-MEASURED
> and the **Contrarian bound fired: P·O = 0.017 S < 0.05 ⇒ RUNG 2 SKIPPED** — no carrier
> build, no solve-seeded births; **burn-4 is the terminal deliverable**. The measurement:
> P (remaining erased super-nucleus Lane pool) = **0.04189 S**, sharply refining QA91's
> 0.134 upper bound — the ep399→ep499 continuation already recovered ~0.09 S for free.
> O = 0.407 / F = 0.194, but collateral dominates: **joint ΔS +0.30 S (oracle) / +0.22 S
> (flat) — BOTH NET WORSE even with perfect GT-RGB content** (identity-fill control
> bit-identical; metric identity JOINT = −P·O + collateral verified exact). NEW MEASURED
> LAW (fp1+QA92 convergent, memory `erf_collateral_law_...20260731`): SegNet's ~85px-r50
> ERF re-reads any injected stroke's neighborhood ⇒ post-hoc injection on textured
> renders is net-negative; erased-structure recovery must be BORN IN-LOOP. Re-pricing:
> carrier (b) gate now carries a hard collateral clause (~5-8× unfavorable); (e1)
> priority-dropped behind burn-4; birth arm defaults to continuation/KD-from-birth.
> **LIVE (mechanical execution):** lp2 LANDED (8fd0f7836b/d9bd131824/34d2354ac4 — P2
> birth_completion key w/ DERIVED ε [28 tests, measured-runnable: births still rising
> at the burn tail, correctly does not fire] + P3 build-only seeding harness w/
> preregistered survival<0.50 falsifier + P5 burn-4 skeleton w/ ⟪UNKNOWN⟫ slots).
> **Rung 1 LIVE (task #803, ddm_r1c, owns the scorer slot):** seal per the gc12 demand
> (control_tail ep499 parent, NO new levers, argv-diff exactly {window,outdir,resume,
> wall-cap}, QA80 staleness gate, F1-F4+A1, governed+DSL-hashed) → FIRE → supervise
> windowed loop w/ the P2 key between ~120-150ep windows, ≤2 extensions, ~8h cap →
> endpoint obligations (n600 realized verdict + P re-measure + QA80 staleness re-check)
> = the burn-4 parent. **fh1 LANDED (72ac061bd3, task #805):** ranked adapted-force
> table CONSUMED — burn-4 skeleton §3b FORCE STACK FILLED (df2402d22b): R6
> `--class-weight-lane` raced (existing lever, NEVER-FIRED — activation-duty row) ·
> R1+R4 tie-locus × per-class-pair W (placement pool, composes w/ birth pool) ·
> R3 satisfice cap · R7 KD attack detail · R2 ξ-advected base GATED on the QA90 $0
> read (pre-seal item) · R8 plateau conjunct · R9/R10/R13 guards ride the tp1 port;
> renderer-rate force demoted (measured pool 0.0022 S); R14 ERF birth-context
> co-adaptation = FRESH derived force, speculative rung behind burn-4 window-1;
> 5 default-OFF DSL lever stubs landed, trainer untouched. **tp1 LANDED (15aad5a28b/fe55c9e599, task #804):** the
> v9→TR1 telemetry port — 6 core rows ported (per-term loss · term-domination ·
> term-inert · liveness · positive-control · lever_engage), flag default-off, CPU
> byte-identity PROVEN w/ OFF-vs-OFF control (112 arrays bit-identical; sole diff =
> flag-independent wall-clock float), DSL Lever, 15 tests — **burn-4 prereq-1 CLEARED**.
> Skeleton slots now: force stack FILLED (fh1) · rung-2 verdict FILLED (QA92 SKIP,
> ffbac5f3e5) · REMAINING = rung-1 endpoint {ckpt sha, d_seg, P} + QA80 staleness +
> QA24 d_seg-cost + validity-r — all fill at the rung-1 endpoint boundary. On rung-1 endpoint:
> MAIN fills the P5 skeleton from receipts → burn-4 seal chain → fire. Honest
> arithmetic unchanged: ladder ⇒ ~0.65-0.70 advisory, does NOT reach 0.172141;
> burn-4's rate re-race (~0.19-0.28 S) is the gap-sized lever.

> **Pointer honesty:** submittable local baseline `0.1910828242 [contest-CPU]` UNMOVED;
> effective competitive bar `0.172141` (official PR #130, [contest-CUDA] reconstruction).
> All local rows `[macOS-CPU advisory]`, `score_claim=false`.
>
> **State:** THE AUTONOMOUS WORKFLOW POLICY (gc11, fe12a36a14, rules W1-W11/OP1) is
> ADOPTED AND BINDING — the standing GO removes WAITING, never RIGOR. The pa1r Pool-A
> race LANDED (fdb48e2c26): `hull_moved_s=FALSE` everywhere; **control_tail = NEW BEST
> realized point** (n600 d_seg 0.0049411, S_additive 0.67325 @ 265,528 B SMEVR, c
> 0.08851; custody /Volumes/VertigoDataTier/pact/ddm_pa1r_20260730/control_tail/).
> delta_sparsity + margin_quant `worse_s`; rowband BLOCKED_NO_D8_PARENT; Pool-A levers
> EXIT burn-3 at instance scope. Closed fork-space to date: post-hoc deletion (Gate-B),
> finishing-distill (dw1), post-hoc pose (ps1/L68), encode-side null-snap (nv1,
> co-location −0.51), all Pool-A in-loop token-byte levers (pa1r).
>
> **Active critical path (fp1 LANDED 07-31 → BR-D ADJUDICATED → gc12 LIVE):**
> fp1 (#799, c90254b5ef/8870930cc4) returned a decisive typed negative — **receiver
> floor f′ ≥ 0.008305 BY CONSTRUCTION** (perfect-GT-argmax flat paint through
> R+SegNet already loses to the RGB parent's 0.00494; rg3 split 100% RECEIVER-limited,
> 0% token-limited; trained head 0.499 = second wall). BR-B graft INSTANCE-DEAD
> (FORMULATION scope: flat-prototype-paint class-field output); reformulation
> re-routed to the RECEIVER (AA/smooth paint, margin-channel modulation). QA91:
> erased Lane mass is NOT GT-flicker (97.67% of Lane area super-nucleus; births
> +8.75/gate STILL RISING at burn end) → BIRTH_PLATEAU_KEY_CANDIDATE, recoverable
> ~0.134 S ≈ 26% of the endpoint seg term. gc11 §2 mechanical branch-select:
> BR-A FALSE ∧ BR-B FALSE ∧ f′ > 2e-3 ⇒ **BR-D THE WALL BRANCH** (fp1's "BR-A
> stays live" was a loose reading; MAIN adjudicated per the receipt table, W11).
> **gc12 = the 14th convocation (task #800, Fable arm LIVE)**: compose {from-birth
> distill (dw1 row 1) · v8 per-class-carrier-IN-RGB derivation · continuation-to-
> birth-plateau · receiver-paint reformulation} → ONE plan + seal demand + prep
> charters → fire under W1.
>
> **Apparatus:** canonical_task_status 603-fold REPAIRED (1b95ab4f09; 3 sc2-folded
> rows quarantined to .omx/research sidecar + sha256 manifest; strict appends restored;
> #793 completion + #799 registration backfilled). Burn-3 design inputs banked: pa1r
> reformulation rows (w-sweep {3e-3,1e-2} · pre-knee parent · rowband from-birth at the
> theorem band [160,240)/1248) + un-exhausted plain-continuation dividend at B's knee.

<!-- prior focus (2026-07-25, effective frontier correction) preserved below as history -->
# Current Focus - 2026-07-25 (effective frontier correction + DDM closure)

> **Competitive score to beat: `0.172`, official leaderboard PR #130.** The
> canonical pointer now selects the minimum of current official leaderboard
> best and custody-specific local exact anchors. The scanner's `0.1880443980`
> row is a banned/non-submission defensive bank; `0.1910828242` remains the
> current original/submittable local baseline. Neither is the global frontier.
> PR130's rounded official components reconstruct to `0.1721412975
> [contest-CUDA]`; no contest-CPU equivalence is inferred.
>
> **Active critical path:** preserve candidate identity and exact coupled score
> from the existing DDM/J12 receiver actions through
> `tac.applied_action_receipt.v1`, `tac.action_effect.v1`, measured ordered
> commutators, lawful stream byte homes, and a hardened one-SHA packet compiler.
> Harvest `v2_compose` framing/inflate/accounting without making its stale n96
> scientific semantics canonical. PF3 has materialized 29/37 occupied buckets;
> its 16 measured scalar directions are adverse and 52 bounded probes remain,
> so it is a control arm rather than the whole program. In parallel, price the
> current archive at EV2's lawful seven stream homes, harden the J12 nonfinite
> warp/reseal edge, byte-close one v8/v9 support-renewing carrier, and reopen
> frame-0 Pose only as nonlinear real-PoseNet descent. Exact joint `delta_S`,
> not independent Seg/Pose/rate caps, is the admission authority. A local exact
> row below the original baseline but above `0.172` is useful progress, not an
> effective-pointer move.

<!-- prior focus (2026-06-10, CAPSTONE viability) preserved below as history -->
# Current Focus - 2026-06-10 (CAPSTONE viability daemon LIVE — first ORIGINAL basis)

> **Pointer: 0.19109982 [contest-CPU] `b46897267` — UNMOVED.** The mission is the capstone (our own
> small learned basis), NOT post-hoc compression of the frozen frontier (all such levers now CLOSED,
> incl. #79 packaging). This session: fixed the null canonical pointer; closed #79 (packaging lever,
> rigorous negative); built `experiments/run_capstone_campaign.py` (the #65/#78 actuator) + validated
> it (base_ch=16 int8 → 64,369 B archive = rate 0.043, sub-0.15-capable budget MEASURED); LAUNCHED the
> decisive 100-pair viability daemon (pid in `.omx/tmp/capstone_daemon/`).
> **CRUX FIX (2026-06-10, commit `11f15a56d`):** the capstone pose-FiLM was a single SHARED FiLM on the
> common feature — but PoseNet scores the frame0↔frame1 DIFFERENTIAL, so it had ~0 Jacobian in the pose
> direction (d_pose bounced 0.437). Replaced with PER-FRAME film0/film1 (matching #84 that held 2.7e-4);
> identity-init preserved, +0.2 KB, 22 tests pass + new shared-FiLM-regression guard. Killed the buggy
> daemon, relaunched corrected (out-dir `…_perframe`).
> **RESUME:** harvest `experiments/results/capstone_daemon_b16_n100_perframe/capstone_result.json` (or tail
> the latest daemon log via `.omx/tmp/capstone_daemon/LATEST_LOG.txt`) — the decisive read is whether
> d_pose now HOLDS (not bounces) at the 64 KB budget. Full doc:
> `.omx/research/capstone_campaign_launch_and_session_state_20260610.md`. Gate ladder: clean 100-pair
> descent+pose-hold → fund the 600-pair CUDA candidate → paired CPU+CUDA exact eval → pointer move.

<!-- prior focus (2026-06-07, NeRV launch-contract blocker closure) preserved below as history -->
# Current Focus - 2026-06-07 (NeRV launch-contract blocker closure)

<!--
DERIVED_OUTPUT regeneration header (Catalog #113 sister discipline):
generated_at: 2026-06-07T01:09:12Z
from_state_hash: 223703aa7 (git HEAD)
source_state_hash: 58fd05af63e1 (.omx/state/canonical_frontier_pointer.json sha256 prefix)
input_sha: derived from reports/latest.md + .omx/state/canonical_frontier_pointer.json + active_lane_dispatch_claims.md per src/comma_lab/research_state.py operator_control_plane registry
classification: DERIVED_OUTPUT per .omx/state/artifact_kind_registry.yaml ("canonical operator focus snapshot; refreshed from latest ledgers and reports")
canonical_writer: operator-curated control-plane snapshot (not auto-generated by any single tool); refreshed manually as ledgers + reports advance
backfill_provenance: Slot D HISTORICAL_PROVENANCE refactor wave 2026-05-29 (sister of canonical helper tac.research_pipeline_output_dir_safety) per CLAUDE.md "Operator gates must be wired and used" + Catalog #113 artifact_lifecycle compliance
-->

## 2026-06-07 P0 Routing Overlay

This overlay supersedes the older May queue text below for active routing. The
historical May sections remain for provenance; do not use them as the current
operator queue when they conflict with this section.

- Main source of truth is `main` at `223703aa7`. The just-landed validated
  slice tightens NeRV launch proof contracts, keeps public/frontier authority
  axis-labelled, and preserves false-authority local MLX evidence as planning
  signal only.
- Public frontier intake was refreshed through the canonical pointer tool at
  `2026-06-07T01:09:05Z` with `fetch_status=ok`; no expensive dispatch should
  start from an older public snapshot.
- Canonical scanner best anchors remain:
  `0.19198533626623068 [contest-CPU]` and
  `0.20533002902019143 [contest-CUDA T4]`. These remain citation surfaces, not
  new score claims.
- HiNeRV current state: the v6 same-region action bundle is blocker-closure
  evidence, not launch authority. The launch contract now requires A/B/C/D/E
  ActionEffect rows, including the real reverse-order
  `frame0_pose_then_birth_composite` arm. Long HiNeRV remains blocked until a
  real accepted birth survives receiver/fakequant/parseback/inflate gates
  without support spill or Pose harm.
- HiNeRV active next move: harvest/classify the current local false-authority
  v26 spatial-scope forced-region claim, then select the next receiver-closed
  parseback/export replay gate. v24/v25 duplicate-smoke attempts are recorded
  as refused due active same-family campaign in the dispatch ledger.
- SNeRV current state: source-forward/export/runtime binding remains the gate.
  The source-forward proof row now carries an explicit output2 boundary verdict
  and launch clearance fails closed unless output2 is `SOURCE_IDENTICAL` and
  receiver-consumed. If output2 requires adapter/rename/drop-basis handling,
  route to LF/HF/MFU/HFR/TUB causal-basis work, not exact-eval admission.
- Final-rate/ActionEffect stack: promote only executable materializers with
  receiver proof. Refuse duplicate PR110++/ActionAtlas scaffolds unless they
  consume the canonical `frontier_rate_attack_materializer_stack` and emit
  `tac.action_effect.v1` value-per-byte rows under
  `100*d_seg + sqrt(10*d_pose) + rate`.
- Exact authority: no current HiNeRV/SNeRV macOS MLX result is promotion,
  rank, kill, or score authority. Exact CPU/CUDA dispatch waits for byte-closed
  receiver proof, parseback survival, inflate survival, and full-video replay
  surfaces.

## Frontier

- Canonical scanner-derived best CPU anchor:
  `0.1880443979880752`
  `[contest-CPU; Linux x86_64 1:1]`, archive
  `196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5`,
  lane `lane_pr128_click_import_NONSUBMISSION_defensive_bank_20260712`.
  Refresh from `reports/latest.md` and
  `.omx/state/canonical_frontier_pointer.json`; this file is a mirror, not a
  frontier source of truth.
- Canonical scanner-derived best CUDA anchor:
  `0.20533002902019143`
  `[contest-CUDA T4]`, archive
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`,
  lane `lane_pr106_format0d_latent_score_table_20260516_contest_cuda`.
- A1 remains the Rule #6 control substrate, not the best current axis floor:
  `0.19284757743677347` `[contest-CPU; GHA Linux x86_64 1:1]` and
  `0.2263520234784395` `[contest-CUDA T4]`.
- A1 archive bytes/SHA-256:
  `178262` /
  `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`.
- Public medal band remains the immediate score target:
  PR101/PR102/PR103 around `0.193`/`0.195`/`0.195`, axis-specific and
  external until exact replay custody proves otherwise.

## 2026-05-22 MLX Portable-Local-Substrate State

- MLX is now a calibrated local research/spend-triage substrate, not score
  authority. Current allowed axis tag is `[macOS-MLX research-signal]`; exact
  CPU/CUDA auth eval remains mandatory before any score, rank/kill, promotion,
  or submission claim.
- MLX score calibration is strict-pass for the current public-frontier set:
  `6/6` pairwise decisions certified, `0/6` uncertain, and minimum MLX gap for
  spend triage `8.801772121230789e-05`.
- MLX transfer calibration now also requires the auth-side payload to pass the
  strict contest-auth-axis contract in `tac.auth_eval_schema`: only full-sample
  `contest-CPU` / `contest_cpu` or `contest-CUDA` / `contest_cuda` payloads
  qualify. Advisory/proxy/local diagnostic payloads are historical or debug
  evidence only.
- The decoder-q surface-guided batch is blocked for exact-eval dispatch:
  candidates `a2f90a216aac4184`, `a9b04920db67ec71`, and
  `8f3a33e49b9b7906` all regressed on `[macOS-CPU advisory decoder-q]`.
- Current first MLX follow-up is sign calibration:
  `ll_decoder_q_surface_sign_calibration_repair`. Use the advisory-negative
  batch as labeled response data; do not send the three surface-guided
  candidates to exact CUDA.
- DQS1 pairset rank021/pair0371 exact Modal CPU recovery is now the CPU
  frontier: `0.19202828295713675 [contest-CPU]`. DQS1 top32 compact
  `sorted_gap_uleb` is superseded on CPU at
  `0.19202894881608987 [contest-CPU]`; its paired CUDA result
  `0.22619043540195719 [contest-CUDA T4]` is not a CUDA frontier.
- Next local-substrate portability work is full-weight PyTorch/NumPy/MLX trace
  parity plus CPU-stable response harvest expansion. MLX GPU/batch behavior
  remains research-only until batch invariance and transfer checks pass.

## Active Strategic Rebaseline

The May 17 T4 symposium supersedes the stale May 15 queue framing without
retiring the L5/L5-v2 staircase:

1. **Immediate frontier-breaking path**: Rule #6 bolt-ons on verified A1.
   Build small, byte-closed, PR101-style additions on the working A1 substrate
   before spending another wave on high-risk substrate-class guesses.
2. **L5 v2 / TT5L priority remains active**: TT5L side-info effect curve,
   L5-v2 probe gates, and architecture-lock custody remain the primary
   asymptotic campaign and must keep moving in parallel.
3. **High-risk substrate cluster**: 35-substrate per-pair-conditioning cluster
   is deferred pending SCORER-AWARENESS probes, not killed.
4. **Original Z6 FiLM path**: do not dispatch as-is; replace with
   per-frame-renderer-axis ego-motion conditioning.
5. **PR106/HNeRV local-basin work**: useful as forensic control and byte
   lessons only; do not let it crowd out Rule #6 or L5-v2 actuation. The
   2026-05-17 format0D WIP adversarial review classifies format0D as a local
   best `[contest-CUDA T4]` forensic/control anchor, not a public
   `[contest-CPU]` frontier or submission candidate.

## Current L5 v2 / TT5L State

- TT5L paired diagnostic exact eval is terminal and non-promotional:
  `3.8987840060549908` `[contest-CPU]` and
  `3.9007398365396795` `[contest-CUDA]` for archive
  `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`.
- Architecture lock remains forbidden. Current blocker class:
  missing complete L5-v2 gate evidence, missing C1/Z5/TT5L probe gate evidence,
  and missing paired CPU/CUDA side-info effect curve harvest.
- TT5L side-info Modal paired dispatch plan now consumes the shared exact
  dispatch authority gate and is unblocked at custody level:
  `ready_work_unit_count=5`, per-variant `dispatch_blockers=[]`, source runtime
  `report.txt` materialized, and all five adjacent `archive_manifest.json`
  files materialized. Ledger:
  `.omx/research/l5_v2_tt5l_dispatch_custody_materialization_20260517_codex.md`.
- TT5L side-info Lightning paired-axis plan has 10 cells, execution preflight
  has `ready_cell_count=10`, execution bundle has
  `ready_dry_run_cell_count=10`, dry-run verifier has `10/10` passing cells,
  route packet has `artifact_blocker_count=0`, and required doctor plan is
  `ready_for_operator_doctor=true`. The route and doctor packets were
  refreshed from a source-relevant tree with no paired-axis-plan source-path
  drift. Do not use the recorded generated-at commit as self-referential
  dispatch authority; rerun the route/doctor packet immediately before
  non-dry-run execution. Non-dry-run provider execution remains blocked on
  Lightning identity/quota, `LIGHTNING_TEAMSPACE`, `LIGHTNING_SSH_TARGET`,
  per-cell source manifests, active lane claims, remote CUDA/machine-inventory
  doctor checks, exact harvest, terminal claim rows, and architecture-lock
  packet refresh after harvest.
- Lightning required-doctor plan exists at
  `.omx/research/l5_v2_tt5l_lightning_required_doctor_plan_20260517_codex.md`;
  it is planning-only and confers no dispatch or score authority. The route
  packet no longer hashes architecture-lock as an upstream source, so the
  route -> doctor -> architecture chain has no circular custody dependency.

## Active P0 Work

1. **Rule #6 A1 bolt-on #1**: Ballé-2018 hyperprior on A1 per-pair latent,
   with KL-on-logits `T=2.0` distillation from frozen A1 teacher.
   Existing Z3HV2 direct-residual export is not this implementation: it is now
   classified as a byte-negative direct-residual control with no active Ballé
   entropy residual decoder. See
   `.omx/research/rule6_z3v2_direct_residual_unwind_20260517_codex.md`.
2. **Rule #6 A1 bolt-on #2**: PR101-style per-tensor byte map plus
   Brotli/LZMA/Huffman sidecar on A1 weights/latents.
   Current A1 byte-escape profiler is saturated under the existing runtime:
   raw-LZMA latent sweep best equals source at `15387` bytes, current
   607-byte sidecar has only a 4-byte oracle entropy gap but no smaller
   runtime-supported representation for current semantics, and no candidate
   archive was emitted. Ledger:
   `.omx/research/a1_rule6_byte_escape_profile_20260517_codex.md`.
3. **Rule #6 A1 bolt-on #3**: VQ-codebook on A1 per-pair latent.
4. **TT5L side-info effect curve**: custody unblocker is complete. Next run
   the Lightning doctor, stage per-cell source manifests, claim each lane, and
   execute the 10 paired CPU/CUDA cells only if doctor and source-manifest
   custody are green.
5. **SCORER-AWARENESS probe wave**: measure whether substrate distinguishing
   features reach scorer attention/argmax maps before deferring high-risk
   per-pair-conditioning substrates.
6. **Z6 replacement design**: per-frame-renderer-axis ego-motion variant,
   not FiLM-bottlenecked Z6.

## Latest WIP Review

- `.omx/research/alien_tech_reverse_engineering_pr106_format0_family_20260517.md`
  is untracked partner WIP and was left unmodified.
- `.omx/research/full_problem_space_reverse_engineering_cpu_gpu_both_20260517.md`
  is also untracked partner WIP and was left unmodified. Its executive
  summary reinforces the paired CPU/CUDA hardware-axis split and scorer
  decomposition, but it is not a current score, dispatch, or promotion
  authority until landed and reviewed.
- `.omx/research/cpu_frontier_master_gradient_campaign_plan_20260517.md`
  is untracked partner WIP and was left unmodified. Its L5 row inherits the
  stale `-0.008` to `-0.015` rate-only expectation; current routing should use
  the L5 rate-only bound review instead.
- Master-gradient raw archive-byte finite differences are now blocked as an
  invalid probe grain for ZIP/entropy-coded packets. Current routing should
  use `.omx/research/master_gradient_raw_byte_finite_difference_adversarial_review_20260517_codex.md`
  and `tac.master_gradient_feasibility`: replace `(N_archive_bytes, 3)` bit/byte
  flip probes with `(N_valid_mutation_operators, 3)` packet-valid score-response
  rows that rebuild ZIP metadata/CRC and prove inflate success.
- Master-gradient operator-response planning surface is now materialized in
  `tac.master_gradient_operator_plan` and
  `tools/build_master_gradient_operator_plan.py`. Ledger:
  `.omx/research/master_gradient_operator_response_plan_landed_20260517_codex.md`.
  This supersedes raw-byte/autograd-per-archive-byte wording in untracked WIP
  campaign notes without editing partner WIP: the only valid next object is a
  grammar-aware operator-row manifest, still `score_claim=false` and
  dispatch-ineligible until packet closure proofs exist.
- Master-gradient dirty WIP adversarial review:
  `.omx/research/master_gradient_partner_wip_false_authority_review_20260517_codex.md`.
  The untracked `src/tac/master_gradient.py` and dirty cathedral-autopilot
  hook should not land as authority-bearing code while they expose an
  `(N_archive_bytes, 3)` tensor, `finite_difference_bit_flip` method naming,
  raw `{byte_idx: delta}` projection API, notes-parsed archive SHA lookup, and
  a "rerank" hook that currently leaves ordering unchanged. Valid routing
  remains `CandidateModificationSpec` / `grammar_aware_operator` response rows
  with packet rebuild, inflate proof, byte-consumption proof, and exact
  axis-labelled result review.
- First executable master-gradient operator row is now materialized for
  PR106-format Brotli sections:
  `tac.master_gradient_brotli_operator_candidate` and
  `tools/build_master_gradient_brotli_operator_candidate.py`. Ledger:
  `.omx/research/master_gradient_brotli_operator_candidate_landed_20260517_codex.md`.
  Local candidate signal: `decoder_packed_brotli` recompression on public PR106
  reduced archive bytes by `151` with CRC/header/parser/Brotli closure proven,
  but runtime inflate proof and byte-consumption proof are still missing, so
  the row remains non-dispatchable and non-promotional.
- Reviewed disposition:
  `.omx/research/pr106_format0d_wip_adversarial_review_20260517_codex.md`.
- FEC6 CPU frontier submission-surface adversarial review:
  `.omx/research/fec6_cpu_frontier_submission_surface_adversarial_review_20260517_codex.md`.
  Verdict at the 2026-05-17 review time: FEC6 remained the best
  `[contest-CPU]` anchor, but the WIP packet was not submission-ready. FEC6 is
  now superseded as best CPU anchor by the 2026-05-22 compact DQS1 gap-ULEB
  exact CPU result; the submission-surface blockers below remain relevant for
  any FEC6-derived packet. Strict compliance output is
  preserved at
  `experiments/results/fec6_cpu_submission_surface_review_20260517_codex/pre_submission_compliance_cpu.json`
  with `passed=false`; blockers include missing `archive.zip`/`report.txt` in
  `submission_dir`, incomplete member manifest, missing public packet README
  axis/source/repro labels, CPU score-claim schema mismatch, and terminal
  dispatch claim missing runtime-tree SHA binding.
- FEC6 writeup pose-marginal correction:
  `.omx/research/fec6_writeup_pose_marginal_correction_20260517_codex.md`.
  The live `docs/pr_writeups/cpu_frontier_fec6_20260517.md` WIP now uses
  `5/sqrt(10*d_pose) = 291.44`, not `922`, and its 1000-byte pose example is
  corrected to a net score regression unless `d_pose` drops by at least about
  `2.24e-6`. Future FEC6/Rule #6 byte-spending claims should use
  `tac.score_geometry.pose_byte_tradeoff`.
- FEC6 selector operator-space audit:
  `.omx/research/omx_parent_markdown_and_fec6_selector_operator_followup_20260517_codex.md`.
  Reusable surface:
  `tac.fec6_selector_operator_space` and
  `tools/audit_fec6_selector_operator_space.py`. Local artifact:
  `experiments/results/fec6_selector_operator_space_20260517_codex/operator_space_manifest.json`.
  Result: `operator_row_count=40`, `raw_archive_byte_rows_emitted=0`,
  `proxy_and_nonpositive_bit_rows=[]`, selector payload entropy gap `8` bytes
  versus `78` required charged bytes to strictly cross `<0.192`. Same-runtime
  FEC6 selector byte-only polish from the current pair table is blocked; next
  FEC6 work must use new paired component rows or materialize a
  component-moving packet operator with byte-consumption proof.
- L5-v2 12-month foreseeable-failures sidecar:
  `.omx/research/l5_v2_next_12_months_foreseeable_failures_20260517_subagent.md`.
  Key tripwires: architecture-lock false authority, TT5L custody refresh
  loops without harvested cells, non-causal side-info controls, no-op Rule #6
  hyperprior/entropy work, axis drift, and optimizer/probe overfitting.
- L5 Wyner-Ziv rate-only bound review:
  `.omx/research/l5_wyner_ziv_rate_only_bound_adversarial_review_20260517_codex.md`.
  Corrected decision: L5 remains P0 because `4800 -> 2000` pose-stream shrink
  would lower FEC6 CPU to about `0.19019`, but a rate-only pose-stream shrink
  cannot justify `-0.008` to `-0.015` or `0.17-0.18` claims. Larger L5 claims
  now require component-moving proof or a larger charged-byte section.
- Key action change: harvest format0D's two-pass additive grammar as a donor
  primitive for Rule #6 A1/FEC6 byte-closed bolt-ons, but do not route the P0
  queue back into PR106-only local-basin polish. Any direct PR106 revisit must
  start with CPU/CUDA raw-output xray and extra-stream ablations.
- Parent-scope cargo-cult / Quantizr staircase follow-up:
  `.omx/research/omx_parent_markdown_cargo_cult_and_quantizr_staircase_review_20260517_codex.md`.
  Non-research `.omx` Markdown added no new score authority, but did preserve
  live no-retread signals: full-score curriculum difficulty, PoseNet
  preprocessing sensitivity, scorer-input anatomy, zero-order arithmetic
  failures versus Brotli, and stack-order discipline. The Quantizr staircase
  helper now has package discoverability, ruff hygiene, and softened
  score-authority wording; it remains a training scaffold, not a frontier
  claim, until byte-closed trainer adoption proves component movement.

## Dispatch Discipline

- No provider dispatch without `tools/claim_lane_dispatch.py claim`.
- No CPU/CUDA promotion without axis-labeled paired custody.
- No architecture lock until the shared authority predicate allows it.
- No score claim from planning, dry-run, macOS, proxy, or diagnostic anchors.
- Every result review must preserve failure class, custody, recomputed formula,
  and reactivation criteria.

## Parent-Scope OMX Markdown Scan

On 2026-05-17, the Markdown scan was widened from `.omx/research` to all
`.omx/**/*.md`, then repeated with `--hidden --no-ignore` so ignored
`.omx/auto_memory_snapshot_*` and `.omx/tmp` Markdown were not silently
excluded. Relevant non-research control surfaces checked:

- `.omx/state/current_focus.md` - refreshed by this file.
- `.omx/state/next_experiments.md` - refreshed alongside this file.
- `.omx/state/active_lane_dispatch_claims.md` - current source for dispatch
  conflict/terminal status.
- `.omx/auto_memory_snapshot_20260504T230223Z/*.md` - ignored historical
  Claude/OMX memory snapshot; no current L5/TT5L authority, but it preserves
  no-signal-loss, stack-order, entropy-coder, overfit-to-one-video,
  PoseNet-sensitivity, FiLM-pose-plumbing, derive/sweep/learn, and
  remote-tarball lessons.
- `.omx/tmp/*.md` - ignored temporary appendices and detached clone READMEs;
  useful as forensic inputs only, not current score authority.
- `.omx/notepad.md` - stale April AV1/Track-B notebook, not current L5
  authority.
- `.omx/release_manifest_v0.2.0-rc1.md` - release hygiene context, not current
  score authority.
- `.omx/state/dispatch_queue.md` - historical HTD queue; not the May 17
  Rule #6/L5-v2 priority list.

Detailed scan ledger:
`.omx/research/l5_v2_omx_parent_markdown_scope_refresh_20260517_codex.md`.
No-ignore follow-up ledger:
`.omx/research/l5_v2_omx_parent_markdown_no_ignore_refresh_20260517_codex.md`.
FEC6 selector operator follow-up:
`.omx/research/omx_parent_markdown_and_fec6_selector_operator_followup_20260517_codex.md`.
Parent-scope active-claims bugfix:
`.omx/research/omx_parent_markdown_modal_cpu_dispatch_bugfix_20260517_codex.md`
records the fresh no-ignore `.omx` Markdown scan that found
`master_gradient_fec6_modal_cpu_dispatch` failing with rc=2 after claim
creation because `experiments/modal_train_lane.py` did not support `--gpu CPU`.
The Modal dispatcher now has a CPU function target for tool dispatches; future
real master-gradient work still needs a fresh active claim and normal custody.
Latest TT5L route/doctor custody refresh is source-relevant-path clean but must
be regenerated immediately before provider execution; the generated packet's
recorded commit is a reproducibility input, not self-referential dispatch
authority.
Current-main full parent Markdown scan:
`.omx/research/omx_parent_markdown_current_main_full_scan_20260517_codex.md`
records `2410` total `.omx/**/*.md` files, `636` non-research Markdown files,
and `379` keyword-matching non-research files. It found no new authority beyond
`current_focus`, `next_experiments`, and `active_lane_dispatch_claims`, but it
did expose the proxy/advisory JSONL append-lock mismatch now fixed in the MPS
and macOS-CPU advisory helpers.

## Required Refresh Cadence

- Refresh this file after any Rule #6 dispatch result, TT5L side-info harvest,
  L5-v2 architecture-lock packet change, or public frontier intake that changes
  the score target.
- Refresh `.omx/state/next_experiments.md` whenever the active P0 work order
  changes.
- Catalog #316 now checks this file, `reports/latest.md`, and
  `.omx/state/next_experiments.md` against `tac.frontier_scan` so stale
  frontier citations fail preflight instead of becoming hidden control-plane
  signal loss.
