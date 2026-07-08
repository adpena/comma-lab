---
doc_type: t5_crucible_p3d_revised_synthesis_draft_v5
role: v5 SYNTHESIZER (operator-convened T5 crucible; requirement-O fold — NOT a redesign: v4's
  architecture, crossing-arithmetic FORM, lane-first curriculum, and all 13 v4 fold resolutions
  are carried forward WHOLE; this revision folds the two control-theory research seats' IMPORT-NOW
  sets (CT-1 §11/§12 · CT-2 §11/§12/§13) per the 10-item fold list, 1:1)
date: 2026-07-07
supersedes: DRAFT_OPTIMAL_STACK_v4_20260707.md (d06e7edbd) — v4 preserved append-only. v5 folds:
  ct_deepresearch_1_training_campaign_control_20260707.md (fresh-research-round-1, unreviewed) +
  ct_deepresearch_2_pde_geometric_topological_control_20260707.md (fresh-research-round-1,
  unreviewed). Both CT sources carry review_status fresh-research-round-1 — every CT-derived law
  below inherits that provenance tag until the seal rounds review it (L81 discipline).
epistemic_contract: unchanged — every knob carries a CONTROL LAW class {(a) CONSTANT · (b)
  RAMP/ANNEAL+completion guarantee · (c) SELF-DERIVING · (d) EVENT-CONDITIONED · (e)
  FRACTIONAL/PARTIAL} + a tag {V-S · V-A · D · DPR}. Every load-bearing number labeled
  MEASURED / DERIVED / INFERRED / ASSUMED (docs/operating_manual_craft_handoff.md). Nothing
  unmeasured asserted as measured.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; archive bytes exact (zip stat).
  Pointer contest-CPU 0.19110 UNMOVED — this whole file is MEANS.
review_status: pre-registered-only (v5, awaiting seal round 1 of 3)
---

STORES CONSULTED: ORCHESTRATION_LEDGER.md (full — reqs A–P binding; the CT-1/CT-2/v4 landing
folds at top = the fold spec) · DRAFT_OPTIMAL_STACK_v4_20260707.md (full — the base, carried
whole; §0.1 13-item table re-checked, none regressed) ·
ct_deepresearch_1_training_campaign_control_20260707.md (full, esp. §1.4/§2.3/§3.3/§4.2/§5.1/
§6.2/§7/§9.2/§10.2/§11/§12) · ct_deepresearch_2_pde_geometric_topological_control_20260707.md
(full, esp. §1/§2/§5/§6/§7/§8/§9/§10/§11/§12/§13) · trainer argparse RE-VERIFIED this session
(full 250-row `add_argument` extraction from experiments/train_levelset_witness_realized_
through_R_mlx.py — every flag named below checked against it; findings in §1.0) ·
DRAFT_OPTIMAL_STACK_v2_20260707.md §2.2/§2.2b (the co-predicate spec + backtest — resolves the
V=4 flag question) · src/tac/witness_control/ module listing (powerlaw_exit.py, costate_
estimator.py, shadow_controller.py exist — the out-of-process co-predicate surface) ·
corpus_query "birth death persistence ledger 20260630" (ledger EXISTS:
tools/birth_death_persistence_dseg.py + .omx/research/birth_death_persistence_dseg_
20260630T172510Z.md — the Conley backtest target is real). NOT consulted: durable-state files
(stale per sweep); position_S* full re-reads (carried via v3/v4); no training launched, no live
config touched ($0 reading + arithmetic + grep only).

# T5 CRUCIBLE — P3d DRAFT v5: THE OPTIMAL FULL STACK (control-theory fold: forfeit-matched exit · decode-gap ILC · Conley certificates · τ-indexed constants · signal-ledger union · computable asymptote)

## §0 — REQUIREMENT-N FRAMING, EXTENDED (fold item 8: the asymptote becomes COMPUTABLE FROM RUN-1)

**Answer first (v4 §0 inherited unchanged in its numbers).** Frontier-to-floor gap = 0.19110 −
0.11797 = **0.07313 S**. Family asymptote estimate: **S_asymptote ≈ 0.165 central, band
[0.154, 0.181]** — this family claims ~36% of the gap central, band [14%, 51%]; **T_3 = 0.15
sits at/beyond the family's optimistic edge**; the family step (quotient codec #155 /
compress-half #336 / d_seg 0.0005–0.0009 regime) is named run-2+ work. v5 does not claim T_3
for this family. What v5 ADDS: the asymptote stops being an estimate-by-stacking-edges and
becomes **computable from run-1's own signals**, via CT-2 §13's five impossibility bounds —
each with the measurement that decides whether it binds:

### §0.0a The five impossibility bounds (CT-2 §13; fold item 8) — each with its deciding measurement

| # | bound | our form | deciding measurement (all run-1-generated or $0) |
|---|---|---|---|
| M1 | **uint8 deadzone** (quantizer information loss; no linear repair) | flips requiring through-R intensity change < 1/255 are unreachable by ANY smooth witness without dither/phase (#149 class). DERIVED floor: δx_min = (1/255)/(0.842·g_I); at lane-paint contrast g_I ≈ 0.2–0.5 ⇒ δx_min ≈ 0.009–0.023 px — sub-pixel control IS observable where contrast is healthy; the deadzone binds only on LOW-contrast boundaries (far-range lane, shadow edges) — a per-class-pair, per-range statement (req H) | **$0 deadzone census** (recess queue, §7c): on cached S_R + frames, count flip-annulus pixels whose required through-R intensity change < 1/255. Band: < 0.3× margin (< 5.3e-6 d_seg-equivalent, i.e. < 5.34e-4 S… see §0.0b precision note) ⇒ NOT binding, #149 stays DEFER; > 1× margin ⇒ family floor term, #149 enters the duty queue with a real prior. Edge-contrast g_I histograms per class-pair = SC-16 fields |
| M2 | **homogenization/pinning** (sub-δ structure unrecoverable by the coarse flow; Dirr–Yip class — viscosity hunt §3) | dash structure below the (τ, ε, R-Nyquist) crossover needs the comb corrector; NO capacity/epoch budget recovers it | the queued $0 τ-crossover probe (Q2, F12 dash-contrast at τ ∈ {0.216, 0.12, 0.062}) + comb-corrector A/B. If comb-OFF floors the dash residual: binding, corrector mandatory for T_3 |
| M3 | **Godunov monotone barrier** (monotone schemes ≤ 1st-order accurate) | any GLOBAL viscosity/damping fallback pays O(ε) boundary accuracy everywhere — the fixed-ε tax | clamp-binding fraction + jitter-share rows (SC-18): if the global-ε path is active > x% of epochs, its O(ε) d_seg tax is measurable vs the ca-band filtered form |
| M4 | **Hajek annealing lower bound** (global-basin guarantees need log-slow cooling; finite budget forfeits the guarantee) | run-1 CANNOT certify the global partition basin; wrong-basin risk is structural. TAIL_k warm restarts (§2.2e) are the adopted repair | seed-pair partition-Hamming A/B ($0-cheap at two seeds; viscosity hunt §2 spec) + inter-TAIL-cycle partition-Hamming (an F15/SC-10 field, added in v5) to see basin hops |
| M5 | **max-plus approximation lower bound** (basis count blows up for smooth non-max-structured targets) | the bulk cartoon must stay INR/curvelet; the tropical layer only for the separatrix — the two-semiring split is FORCED, not chosen | the §11-row-16 max-plus fit probe's K-vs-accuracy curve on BULK patches (expected blow-up) vs ANNULUS (expected small K) |
| — | NOT binding (verified): Bardos–Lebeau–Rauch geometric control condition | full-domain observation ⇒ observability unconstrained; the along-tangent 3.2× deficit is CHART CAPACITY, not observability | already MEASURED: \|H_R\| ∈ [0.842, 1.0] all-pass to render-Nyquist (scale-robust, negatives review item 5) — no further measurement owed |

**Asymptote composition (INFERRED, honest — CT-2 §13 verbatim posture):** the composed-lever
family's d_seg floor ≥ deadzone mass (M1) + uncorrected sub-δ dash mass (M2, →0 with comb) +
selection-regime residue below τ_end·ln5 resolution (bounded by the §3.4 certificate's
uncertified-island population). No term is asserted as a number (all currently unmeasured); every
term has a named run-1 signal or $0 census above — **run-1 generates every input, so the v6/run-2
asymptote row is a COMPUTATION, not a re-estimate.** Design consequence regardless: M1/M2 are
attacked by #149/comb (representation), not by more epochs — "the wall is BASIS" (chain-A) from a
third independent direction.

### §0.0b Precision note on the M1 band (req J)

CT-2's census band "< 0.3× crossing margin (< 5.3e-6 d_seg)" mixes units: 0.3 × 0.00178 S =
5.34e-4 S = **5.34e-6 d_seg**. The band as adopted here: deadzone flip mass < 5.34e-6 d_seg
(= 5.34e-4 S = 0.3× margin) ⇒ not binding; > 1.78e-5 d_seg (= 0.00178 S = 1× margin) ⇒ binding.
(The CT-2 sentence's "5.3e-6" is the d_seg figure; its "S-units" phrasing was loose — corrected
at adoption, no number moved.)

### §0.0c THE EVSI TABLE (CT-1 §9.2; fold item 8 — requirement N quantified: run-1 as instrument)

Dual-control pricing of run-1's information (EVSI ≈ min(p, 1−p)·|ΔS_arms| per binary v5/run-2
decision; priors stated; all DERIVED estimates):

| run-1 signal | v5/run-2 decision it flips | swing \|ΔS_arms\| | prior p | EVSI (S) |
|---|---|---|---|---|
| pose-through-witness row (F11/SC-5, w_pose>0) | ship L3 FiLM mechanism vs L1 Jacobian fallback | ~0.088 (pose term 0.105 at R1-floor vs ~0.017 at 3e-5) | 0.5 (never fired) | **~0.044 — dominates everything by 10×** |
| per-stage decode parity Δ̂_k trajectory (SC-7) | trust/repair byte-close per stage; catches within-run decode drift | insurance on 0.0104 S | 0.2 | ~2e-3 |
| big-3 anneal-completion recovery (F1+F3 split legs) | v5 allocates schedule-completion vs representation | ~5e-4 within-run + ~2e-3 arm-selection | 0.5 | ~1e-3 |
| verdict replicate σ_meas (SC-14) | threshold/floor calibration for every trigger + ILC resolution limit | mis-set trigger costs 5.4e-4–2.7e-3/run | 0.4 | ~6e-4 |
| per-class ν_c contraction fits (F3 tail fits, emitted explicitly) | eps_c veto calibration | forfeit-class 5.4e-4–2.7e-3 | 0.3 | ~4e-4 |
| signed per-class-pair flip rows (SC-16) | admit/kill the one-sided-hinge lever family (B16) | ~1e-3 if real | 0.3 | ~3e-4 |
| live margin-quantile m_q row (SC-3) | TAIL τ*_k self-derivation + τ_end confirm + adaptive-ε τ-law | enables the tail (1e-4–1e-3/cycle) | enabling | ~3e-4·k |
| forecast-residual row (SC-15) | MPC horizon validity (N* online) + λ-decay model selection | ~5e-4 | 0.4 | ~2e-4 |

**Sum ≈ 0.048–0.05 S of expected v5/run-2 decision value — ~10× run-1's direct crossing value
(1.6–6.4% × (0.19110 − S_run1)), plus one avoided wasted run (12–35 h at 42 s/ep).** This table
IS the PowerPlay-consistent duty-to-measure ranking for the signal adds (req D): pose row first
(already planned), decode-parity trajectory second. Run-1 is a two-wall measurement instrument
with a real, engineered, gated crossing tail — framed exactly per requirement N.

## §0.1 — FOLD RESOLUTIONS, 1:1 WITH THE v5 FOLD LIST

| # | fold item | resolution in v5 |
|---|---|---|
| 1 | TAU→FIN forfeit-matched exit s\* = ν·forfeit | §2.2f: the QUANTITATIVE form of v4's feedforward co-predicate (they are the SAME law — under exponential decay, "forecast remaining meat s/ν < forfeit" ⟺ "s < ν·forfeit"). s\* = 0.026210 × 5.4e-4 = **1.4154e-5 S/ep** (fold-list 1.41e-5 = same to 3 s.f.). Ships as the WOULD-FIRE second arm FIRST (B-CT1, ~10 LOC on the per-epoch-normalized slope); promotes to firing arm iff **P-CT3** passes (recess queue §7c; band: first fire ep670–700; kill: fires < ep650 or > cap 726). Req-B all-three binds: backtest (P-CT3) + injection test + fail-safe cap 726 unchanged. Wall-clock cost ~60 ep × 42 s ≈ **42 min for +5.4e-4 S = 30% of the crossing margin** — mandatory under L59 (time lexicographically secondary). Shipped slope arm is NOT deleted: it stays the tested fallback until P-CT3 promotes the new arm |
| 2 | Decode-gap ILC | §2.4: train-side bar FORMALLY = **0.0011 − Δ̂**, Δ̂₀ = 1.0427e-4 (R6 MEASURED) ⇒ bar = **9.9573e-4 ≈ 0.0010** — v4's independently-chosen design target is now DERIVED, not chosen (cross-field consistency row (c), §0.4). Δ̂ updated PER-STAGE from F13/SC-7 parity rows by EWMA ω = 0.5 (stable for gain-ratio ξ ∈ (0,4)). Campaign update law (Newton-ILC γ = 0.7, contraction \|1−γξ\| ∈ [0.055, 0.545] at ξ ∈ [0.65, 1.35] ⇒ **2–3 runs to the identified floor**) in the campaign layer §8. Crossing-exact ceilings UNCHANGED (§0.2): ≤ 0.0010137 central / ≤ 0.0010940 win9 |
| 3 | PMP fixes | (a) co-predicate window **V = 4 → 5** — flag VERIFIED by grep (§1.0): NO in-trainer flag exists for the VERDICT co-predicate window (it is a B1 build parameter; the trigger runs out-of-process advisory today, v2 §2.2). V=5 binds the B1 spec + the advisory implementation; CT-1's `--copred-verdict-window` was an INVENTED name — corrected. The sister EXISTING flag `--curriculum-plateau-windows` (default 4, L7672) is the EP_LOSS-plateau window, a DIFFERENT surface — NOT changed (its transient-discount state covers it per CT-1 §10.2; silently recalibrating it would be the per-epoch-normalization bug class). (b) PMP↔eps_rel ratification = consistency row (b), §0.4 (7.12e-5 vs 6.8e-5 S/ep, within 5% at the exhaustion operating point — operating-point-dependent, caveat carried). (c) turnpike TAIL_k budget law folded into B14/§2.2e: **≥ 265 ep/cycle (settle 3/ν ≈ 115 + exit floor 150), k_max ≈ 3–7 within the 3000-ep budget after ep650 entry; dwell_TAIL ≥ 115 ep lower bound** — a TAIL cycle shorter than 115 ep is measuring its own transient |
| 4 | Conley persistence certificate | §3.4 + B17 (~30 LOC on existing persistence machinery): island I survives stage k AND decode ⟸ **pers(I) > τ_k·ln5 + Δ_dec^logit**. At τ_end = 0.062: τ_end·ln5 = 0.0998 ⇒ threshold ≈ **0.10 logit** + Δ_dec^logit (Δ_dec^logit currently UNMEASURED — initialized 0; supplied by the F13/SC-7 logit-unit extension). Per-island pass/fail ledger + "born-to-die" stage-boundary count + DEATH ALARM (island dies WHILE certified = controller/instrument failure, req-B tested). $0 BACKTEST in the recess queue (§7c) on the 20260630 birth-death ledger (EXISTS: tools/birth_death_persistence_dseg.py); band: certified-survival ≥ 95%; kill: < 80% ⇒ fit safety factor s·(τ·ln5). Honest boundary carried: certificate is SUFFICIENT-not-necessary; pers measured on the smoothed field (sides not fully independent) |
| 5 | Signed per-class-pair shape-gradient weight | §1.3 + B16: GATED lever, 0 bytes — DSL Lever slots (σ_ij,dir per class-pair per DIRECTION) on the EXISTING margin-saliency surface (`--margin-saliency-*` family verified; new argparse rows = PROPOSED, §1.0), **default-off, registered with duty-to-measure** (activation-ledger row; the default-off-is-orphaned rule honored), contingent on **Q1** (the $0 per-side ρ UniWARD-asymmetry re-test, v4 §7b — cited as the gate; fire iff \|ρ\| ≥ 0.3 any side). Grounding = consistency row (a), §0.4: τ_end·ln5 = 0.0998 ≈ the MEASURED 0.10 flip-support edge — the shape-calculus Dirac-layer width and v3's τ_end derivation are the SAME law from two fields. Mechanism note: the pooled-unsigned UNIWARD "at chance" (ρ = −0.033) is this law with the sign field integrated out — the theorem PREDICTING the measured null. Efficacy bound: lane-leg 10% relative ≈ 4e-5 d_seg ≈ 4e-3 S ≈ **2.2× margin** |
| 6 | τ-indexed constants law | §1.4: the full enumeration (9 constants + audit rule) with c(τ) forms, coarse-τ anchors, declared exponents, re-validated at the F12 τ-samples {0.216, 0.12, 0.062}. Island release law **r\*(t) = 0.95·σ_eff(t)** adopted (B18, on the REAL surface `--seed-island-eased`/`--island-dilate-px`/`--seed-anneal-*` — CT-2's `--island-dilation-radius-end` was an invented name, corrected to a build item); plugged consistency: r\* = 0.674·√2·σ = 0.953σ; at the probe's σ = 1.5 ⇒ r\* ≈ 1.43 px, matching the MEASURED dilation knee (native 44.6% survival straddling r\*, +1 px → 90.0%, +2 px → 98.3%) |
| 7 | Signal-completeness ledger UNION | §4c: v4's 13 rows + CT-1 §12.2 (5) + CT-2 §12 (5) MERGED — dedupes: per-stage parity appears in BOTH CT lists + v4's F13 → ONE row SC-7 (extended to logit units Δ_dec^logit); live m_q appears in CT-1 + v4's annulus row → merged into SC-3 (per-cadence emission); per-direction histograms appear in CT-1 #5 + CT-2 #2 → ONE row SC-16. Net: **19 rows, SC-1..SC-19 (6 NEW: SC-14..SC-19; 3 EXTENDED: SC-3/SC-7/SC-9)**, each {signal → generated? → recorded-where → NAMED consumer}. Completeness re-checked §4c-end; zero no-signal gap terms remain |
| 8 | §13 impossibility bounds + EVSI | §0.0a (five bounds + deciding measurements → asymptote computable-from-run-1) + §0.0c (EVSI table, run-1 instrument value ≈ 0.048–0.05 S, pose row 0.044 dominant) folded into the req-N framing |
| 9 | DEAD verdicts recorded | §11 row 16: **max-plus band-residual decomposition** = the surviving essence of backstepping (placed IN the #342 solve inventory where solve-don't-train will find it); §12 DEAD/CAMPAIGN ledger: backstepping kernel machinery DERIVED-dead (no 1-D spatial causality; actuator not at domain boundary; descent already Lyapunov-stable) · CT-1 DEAD: LQR/Riccati direct, grid/NN HJB, continuous ES dither on θ · DEFER-with-reason: Griewank revolve · CAMPAIGN-ONLY routed to §8 (ILC gain law, excitation-rank check, EVSI ranking, pair-admission quarantine, family-inflection watch, birth-scheduler fit, adaptive continuation step) — nothing dropped |
| 10 | Consistency pass | §0.2: crossing arithmetic RE-VERIFIED end-to-end at full precision this session — **no fold moves any crossing number** (verification printed). Flags: every flag named in v5 re-grepped against the trainer argparse (§1.0); two CT-invented names caught and corrected (`--copred-verdict-window`, `--island-dilation-radius-end`); zero invented flags remain |
| +A | CT-1 §12.1(4) self-triggered verdict cadence (IMPORT-NOW set member beyond the numbered list — folded per the ledger's "both CT §12 IMPORT-NOW sets" charter) | §2.5/B-CT3 (gated on P-CT2): Δt_next = clamp(floor_S/\|Ŝ′\|, 25, 100) ep — verdict cadence stretches 25→100 exactly where verdicts are least informative; −30–40% n600 verdicts late-run at zero score cost (every skipped verdict provably could not have changed a decision at the floor). Floor 25 = today's cadence ⇒ degrades safely; spike/liveness alarms stay per-epoch. P-CT2 band: 12–17 of 41 verdicts skipped on the mod32cap replay, no missed best > 1 cadence; kill: any missed best > 1 cadence |
| +B | CT-2 §12.1(5) washout transition damper (same charter basis) | §2.6/B-CT5 (gated + injection-tested): high-pass LR damper at stage boundaries d(t) = k·HP[dL/dt]₊ — acts ONLY on transients, ZERO DC gain ⇒ structurally cannot bias the converged score (the principled M1-quench form; measured quench +27.5%). Fail-safe = the v3 rewarmup ramps unchanged (`--stage-transition-rewarmup-*` flags verified). Falsification: A/B transition-transient area vs the open-loop ramp at matched schedule; kill: no reduction. ~20 LOC |

## §0.2 — THE CROSSING ARITHMETIC (fold item 10: RE-VERIFIED at full precision; NO fold moves it)

The condition binds on the DECODED surface (v4 §0.2 inherited):

  **100·(d_seg_train + g_dec) + √(10·d_pose) + rate < 0.19110**, g_dec = +1.0427e-4 d_seg
  [MEASURED, R6 ep650/mod32cap — one checkpoint; per-stage SC-7 rows re-measure].

Re-verification, unrounded chain (req J — d_seg ≥ 5 dp, d_pose ≥ 6 dp, bytes exact, S ≥ 6 dp):

- pose term: √(10 × 3e-5) = √(3.0e-4) = **0.0173205**.
- central rate: 93,092 B × 25/37,545,489 = 2,327,300/37,545,489 = **0.0619861**.
- win9 rate: 81,032 B × 25/37,545,489 = 2,025,800/37,545,489 = **0.0539559**.

| triple (d_seg_train, d_pose, rate) | S decoded | crosses 0.19110? |
|---|---|---|
| v3 triple as written (0.0011, 3e-5, central) | 100×0.00120427 + 0.0173205 + 0.0619861 = 0.120427 + 0.0173205 + 0.0619861 = **0.1997336** | **NO — over by 0.0086336 = 4.85× margin** (0.0086336/0.00178 = 4.850). Stated plainly (v4 inherited) |
| **v5 triple (0.0010, 3e-5, central)** | 0.110427 + 0.0173205 + 0.0619861 = **0.1897336** | **YES — margin 0.0013664** (v4's 0.001366 ✓) |
| **v5 + win9 arm (0.0010, 3e-5, win9)** | 0.110427 + 0.0173205 + 0.0539559 = **0.1817034** | **YES — margin 0.0093966** (v4 printed S 0.181703 / margin 0.009397 ✓ — v4's §0.2 win9 row's displayed column-sum "…= 0.181703" is confirmed from the unrounded chain; summing the ROUNDED column entries gives 0.181704, a 1e-6 display artifact only, no decision touched) |
| win9 at the OLD train target (0.0011, 3e-5, win9) | 0.120427 + 0.0173205 + 0.0539559 = **0.1917034** | NO — over by 0.0006034 (win9 alone does not rescue 0.0011; v4 ✓) |

**Required train-side targets (exact, re-derived):** central: d_seg_train ≤ (0.19110 − 0.0173205
− 0.0619861)/100 − 1.0427e-4 = 0.00111793 − 0.00010427 = **0.0010137** ✓. win9: 0.00119824 −
0.00010427 = **0.0010940** ✓. All four v4 rows reproduce to the printed digit. **None of folds
1–9 enters this arithmetic:** fold 1 changes WHEN the finisher fires (schedule), not the design
targets; fold 2 re-derives the 0.0010 bar the arithmetic already used; folds 3–9 are schedule/
curriculum/telemetry/campaign surfaces. The v4 plain statement stands verbatim: crossing at
central rate needs every class within ≤ 1.4% of its optimistic design edge simultaneously; with
win9 admitted (P8/F8 trained-with gate), ≤ 9.4% — the v3 condition essentially restored; g_dec
is an ENGINEERED selection variable (decoded-KKT, §5.0), not a hope.

## §0.4 — CROSS-FIELD CONSISTENCY ROWS (the fields-grade payoff; fold items 3b + 5)

Independent derivations agreeing with measured constants = the design is sitting on real
structure. Registered as consistency rows (tranche-2 equation candidates, PROVISIONAL until the
seal reviews the CT sources):

| row | field A | field B | agreement |
|---|---|---|---|
| (a) | Maslov semiclassical: τ_end·ln5 = 0.062 × 1.6094379 = **0.0997852** | Hadamard shape-calculus Dirac-layer width = MEASURED flip-support edge **0.10** (flip-rate 0.764 for m < 0.10, ~0 above) | **≈ 0.2% — the SAME law from two fields**; grounds both the τ_end choice and the B16 signed-weight δ_τ width |
| (b) | PMP transversality stop-rate ε_stop = floor_S/cadence = 0.00178/25 = **7.12e-5 S/ep** | shipped co-predicate eps_rel 5e-3/25ep converted at the exhaustion operating point (d_seg 0.0034): 5e-3 × 0.0034/25 × 100 = **6.8e-5 S/ep** | **within 5% — independent ratification** of the shipped trigger. Caveat carried: operating-point-dependent (at d_seg 0.001 the relative trigger = 2e-5 S/ep, 3.5× finer than the floor-derived stop — correct direction) |
| (c) | ILC feedforward bar: 0.0011 − Δ̂₀ = 0.0011 − 1.0427e-4 = **9.9573e-4** | v4's independently-chosen train-side design target **0.0010** | ≈ 0.4% — the chosen target was the ILC-correct one; now DERIVED (§2.4) |

---

## §1 — THE WitnessProgram (v5 deltas from v4 marked ★★★★; all v4 ★★★ / v3 ★★ rows inherited)

### §1.0 FLAG-VERIFICATION TABLE (fold item 10 — every flag in v5, grepped this session against the trainer argparse)

**Verified EXISTING** (line refs where load-bearing): `--anneal-epochs` (L7258) ·
`--softmax-temp-end` (L7365, default 0.05) · `--render-aa` (L8245, choices none/supersample/ipe)
· `--curriculum-plateau-windows` (L7672, default 4 — the EP_LOSS surface, NOT the verdict
co-predicate; unchanged) · `--curriculum-plateau-rel-eps` (L7666, default 1e-4) ·
`--curriculum-event-triggered` (L7661) · `--amplify-weight` (L8361, float, default 0) ·
`--amplify-persist` (L8368, choices uniform/inverse_thickness) · `--island-dilate-px` (L8345,
default 1) · `--seed-island-eased` (L8346) · `--seed-anneal-epochs`/`--seed-anneal-shape`
(L8384/L8390) · `--margin-saliency-weight/-tau/-target/-start-epoch/-uniward/-reachability` ·
`--eikonal-visco-eps-floor/-eps-upper/-margin-factor/-ca-band` · `--weight-entropy-penalty-lambda`
· `--muon-start-epoch` · `--stage-transition-rewarmup-epochs/-floor/-shape` ·
`--stage-transition-reset-moments` · `--persistence-loss-weight/-warmup-epochs/-classes` ·
`--stage-checkpoints` · `--verdict-batch/-pairs` · `--annulus-telemetry/-band`.

**PROPOSED-NEW (build items — never claimed existing):** the B1 verdict-co-predicate argparse
surface incl. its window param V and the B-CT1 second arm threshold (CT-1's suggested spelling
`--tau-fin-slope-star 1.4154e-5` is a PROPOSAL inside B1's design, not a flag) · B13's
`--amplify-weight-lane`/`--amplify-weight-movable` + `persistence_pairs` third `--amplify-persist`
kind (v4 inherited) · B16's signed σ_ij,dir slots on the margin-saliency surface · B18's
r\*(t)-release mode on the eased-homotopy surface · B-CT3's cadence-law flag.

**CT-invented names CAUGHT + CORRECTED (the never-invent-flags discipline doing its job on the
research seats):** `--copred-verdict-window` (CT-1 §12.1) — no such flag; the verdict
co-predicate has NO in-trainer flag yet (B1 owed, ~80 LOC; runs out-of-process advisory) ·
`--island-dilation-radius-end` (CT-2 §12.1) — no such flag; the real surface is
`--island-dilate-px` + `--seed-island-eased` + `--seed-anneal-*`.

### 1.1 Program sketch (v4 sketch inherited; v5 adds two gated levers + one law-binding)

```python
prog = WitnessProgram(
    purpose="T5 crucible ARM-PRIMARY v5: v4 + forfeit-matched exit + Conley certs + tau-indexed constants",
    base=Mod32SegOnlyControlBase(),
    curriculum=sealed_205_curriculum(cfg, handoff="event"),   # anneal-epochs 600; --softmax-temp-end 0.062
    levers=[
        SeedIslandBirth(), SeedIslandEased(release="r_star"),  # ★★★★ B18: r*(t)=0.95·sigma_eff(t);
                                                               # fail-safe = fixed 275-ep ramp (v3)
        EventTriggeredCurriculum(),                            # B1 spec: verdict co-predicate V=5 (was 4)
        LogitAdjust(tau=1.0),
        AmplifyIsland(form="hinge", weight={"lane": 1.0, "movable": 0.28},  # B13-gated; else pooled 1.0
                      gated="witness_alone"),
        PersistenceTopology(weight="1/pers clamped [0.25,4] (R13+B13-gated; else pooled)", warmup=275),
        ConleyCertificate(),                                   # ★★★★ B17: SENSE row + death alarm
                                                               # (advisory-only; pers > tau_k*ln5 + dec_logit)
        SignedBoundaryWeight(gated="Q1"),                      # ★★★★ B16: sigma_ij,dir slots, DEFAULT-OFF,
                                                               # duty-to-measure registered; byte-identical OFF
        CacheGtSkeleton(), LengthSigma("fitted-20260707"),
        AACoverageRender(mode="ipe"),                          # v4 (R3-surviving form)
        AnalyticLaneRenderBand(start=350, boundary_relative=True, v_h=174),
        ChromaBoundarySharpen(weight=0.1, margin_band=1.0, start="tau_fire"),
        MuonWarmStart(lr_final_frac=0.1),                      # entry = TAU-window EMA-best
        WeightEntropyPenaltyMLX(lam=15),                       # twin lam=0, mirror-schedule
        GNSpectrumProbe(k_pairs=">=32 + K-trend row"),
    ],
)
prog.validate()
```

### 1.3 ★★★★ THE SIGNED SHAPE-GRADIENT LEVER (fold item 5; B16 — gated, 0 bytes, default-off)

Exact law (CT-2 §1, Hadamard structure theorem applied to d_seg as a shape functional; the
τ-smoothed shape gradient = the signed Dirac layer):

    L_boundary = Σ_{ij} Σ_{dir∈{i→j, j→i}} σ_ij,dir · Σ_x 1[pair(x)=(i,j), dir(x)=dir]
                 · δ_{τ(t)}(m(x)) · |∇m(x)| · hinge_dir(ℓ(x)),
    δ_τ width = τ(t)·ln5 (RE-DERIVED each stage — a §1.4 τ-indexed constant),
    σ_ij,dir initialized from Q1 (the $0 per-side ρ re-test, v4 §7b — cached S_R + texture
    fields, < 5 min).

Class (e) fractional/partial. **Gate law:** OFF (byte-identical) unless Q1 returns |ρ| ≥ 0.3 on
any side of any major pair; if |ρ| < 0.1 both sides every major pair ⇒ SCALE-ROBUST dead, lever
retired-with-reason in the activation ledger. Registered default-off WITH duty-to-measure
(state: never-fired → Q1-adjudicated) per the default-off-is-orphaned rule. Grounding:
consistency row (a) — the δ_τ layer width 0.0998 ≈ the measured 0.10 flip edge. Through-R note
carried from CT-2: the sign structure survives R (linear parts adjoint-transport the boundary
layer, smeared ~2 px by the bicubic footprint); magnitude calibration is STE-distorted, which is
why S_R (#268, exact, θ-independent) is the realized weight of choice. Falsification: A/B
signed-hinge vs unsigned margin-gate at matched schedule; kill if Δd_seg < 0 or < attribution
floor. [Provenance: fresh-research-round-1 mechanism + MEASURED anchors]

### 1.4 ★★★★ THE τ-INDEXED CONSTANTS LAW (fold item 6 — Γ-convergence licenses finite-τ design LAW-wise only)

CT-2 §5: phase-field control results converge τ→0 INCLUDING the first variation — but only the
LAWS track the limit; a frozen constant does not. **Binding rule: every τ-adjacent constant in
the launch config ships as c(τ) with its coarse-τ anchor value, a declared scaling form, and
re-validation at the three F12 τ-samples {0.216, 0.12, 0.062}.** The enumeration ("others you
find" — audited against the full flag list this session):

| # | constant (real surface) | c(τ) form | anchor / status |
|---|---|---|---|
| 1 | adaptive-ε FLOOR (`--eikonal-visco-eps-floor`) | floor_t = max(0.25 [DPR fail-safe], ε_lower(t) = \|c_a(τ)\|·√(η·λ_eik/8)) — the v4 §1.2 window law; \|c_a\| GROWS as τ descends ⇒ the floor tracks the physics | v4 row inherited; saturation ALARM at ε_raw > 0.7 sustained (§4 rank-4) |
| 2 | adaptive-ε UPPER (`--eikonal-visco-eps-upper`) | 0.7 CONSTANT — measured-anchor (ε = 0.3 stable / 1.0 explodes), NOT derived; declared exponent 0 pending the $0 clamp-binding check (Q3) at fine-τ fields | V-A anchor; if binding > 90% at fine τ ⇒ re-derive as c(τ) via the §6.2-CT1 projection form (B-CT4, gated) |
| 3 | δ_τ margin-layer width (margin-gate band; B16's δ_τ; `--margin-saliency-tau` family) | **width = τ(t)·ln5 — EXACT, exponent 1** (Maslov/Hadamard, consistency row (a)) | D; re-derived each stage; at τ_end: 0.0998 |
| 4 | island release radius (B18 on `--seed-island-eased`/`--island-dilate-px`) | **r\*(t) = 0.95·σ_eff(t)**, σ_eff(t) = max(τ(t)-interface width, ε-viscous cutoff, R-Nyquist ≈ 1 px); RELEASE protection when r\*(t) < the native dash half-width — the homotopy END state comes from physics, not a hand ramp | D with MEASURED anchors (knee r\* ≈ 1.43 px at σ = 1.5: 44.6%/90.0%/98.3% survival at +0/+1/+2 px); fail-safe = v3's fixed 275-ep 1-Lipschitz ramp (req B) |
| 5 | island gate margin (`--amplify-margin-target` / margin-gated support) | gate band ∝ τ(t)·ln5 (the support IS the flip-prone set, which lives at m ≲ τ·ln5) | exponent 1 declared; coarse anchor = current value; F12 re-validation |
| 6 | c_cond (annulus-conditioning threshold for the event-conditioned λ_eik, §2-CT2) | class (e): measured from run-1's FIRST TAU-stage window; declared c(τ) with exponent fit from the F12 samples | first-run-measures posture (same as eps_c) |
| 7 | ChromaBoundarySharpen margin_band = 1.0 | candidate c(τ): band is in margin units ⇒ should scale ∝ τ·ln5; run-1 keeps the constant (measured-good at coarse τ), F12 decides the exponent | DPR; re-validation owed, non-blocking |
| 8 | Conley certificate threshold (B17) | **τ_k·ln5 + Δ_dec^logit — τ-indexed BY CONSTRUCTION** | D |
| 9 | TAIL τ\*_k (§2.2e) | **m_q(k)/ln5 re-derived from the LIVE margin field each cycle** — already law-form | D (v4 inherited) |
| — | audit rule | any constant whose consumer reads the τ-smoothed margin field carries a declared exponent OR a measured-flat verdict at the three F12 samples; a bare constant there = a seal finding (the "coarse-point constant wearing a law's clothing" class, negatives-review item 6) | binding on seal rounds |

All other §1.2 v4/v3 rows (ipe AA, decoded-KKT selection, along=8 regime sentence, TAU→FIN
restore law, joint λ_bytes law, hybrid orientation + R12, MUTCD comb, LogitAdjust pins,
determinism) inherited UNCHANGED.

---

## §2 — THE SCHEDULE (v5: forfeit-matched arm; TAIL budget law; ILC bar; gated cadence + damper)

### 2.1 Stage graph — v4 §2.1 inherited unchanged (P → CE → TAU → FIN → TAIL_1 → … ; END = fail-safe cap only).

### 2.2f ★★★★ THE FORFEIT-MATCHED TAU→FIN ARM (fold item 1 — the quantitative form of v4's feedforward co-predicate)

Derivation (CT-1 §3.3, MPC hand-off principle): exit when forecast remaining stage gain < the
measured transition cost. Under the MEASURED exponential decay (within-TAU erosion slope
+3.3e-3 → +2.4e-4 S/ep over ep350→ep450 ⇒ contraction ν = ln(13.75)/100 = **0.026210 /ep**,
settle 3/ν ≈ 115 ep — the turnpike signature), remaining gain from slope s = s/ν. Transition
forfeit (v3 §2.2c, MEASURED) = +5.4e-4 S. Therefore:

  **fire TAU→FIN when s < s\* = ν·forfeit = 0.026210 × 5.4e-4 = 1.4154e-5 S/ep.**

The shipped arm fires at s ≈ 6.8e-5 S/ep (consistency row (b)) — 4.8× coarser ⇒ ~60 ep early
(Δt = ln(6.8e-5/1.4154e-5)/ν = 1.5695/0.026210 ≈ 60 ep); firing at ~ep685 puts the TAU-window
EMA-best ≈ the ep650 true best ⇒ the +5.4e-4 forfeit → ≈ 0, at ~42 min wall-clock —
**mandatory under L59.** Self-consistency: as the transition law improves, forfeit shrinks, s\*
shrinks, fire moves later — a fixed point, not a constant. **Ship posture (req B, all three):**
B-CT1 lands the arm WOULD-FIRE ONLY (~10 LOC on the per-epoch-normalized slope, an F4 audit-row
variant); P-CT3 backtests it on the mod32cap 41-row trace (band: first fire ep670–700; kill:
< ep650 or > 726); injection test through the live witness_control wiring; fail-safe cap 726
unchanged. Promotion to the firing arm ONLY on P-CT3 + injection pass. The v4 §2.2 forecast-form
co-predicate is hereby made quantitative: "AIC-forecast remaining meat < 5.4e-4 S" and
"s < ν·forfeit" are the same predicate under the exponential model — BOTH would-fire rows are
logged (SC-9) so the equivalence is itself measured. Note (M2-class guard): the forfeit-matched
fire at ~ep685 still sits AFTER anneal-complete (ep600 denominator law) — the anneal-completion
precondition is untouched.

### 2.2e TAIL law — v4 inherited + the fold-item-3c budget law now IN the spec:

TAIL_k: τ_k = max(τ_{k−1}/2, τ\*_k = m_q(k)/ln5 live); LR_k ∝ τ_k; warm restart (never-reset
moments; §2.2c transition law); per-cycle powerlaw_meat exit; PowerPlay stop rule at the
attribution floor; END = req-B cap only (cap_tail = 2× realized TAU length, injection-tested).
**★★★★ Budget law (turnpike, CT-1 §2.3/§5.1): each cycle needs settle (3/ν ≈ 115 ep) + exit arc
(≥ 150 ep floor) ⇒ ≥ 265 ep/cycle; k_max ≈ 3–7 within the 3000-ep budget after ep650 entry —
the tail is τ\*-limited, not budget-limited, at 3000 ep (PR95's 29,650-ep/8-stage run = the k→∞
existence proof). Dwell lower bound: dwell_TAIL ≥ 115 ep — a shorter cycle is measuring its own
transient (M-S2 confound class); this is the derived LOWER bound to cap_tail's upper.** Per-cycle
Δ remains a BET (labeled); the crossing arithmetic books NO TAIL gain (§0.2 check unchanged).
New F15/SC-10 field (M4 bound): inter-cycle partition-Hamming (basin-hop visibility).

### 2.4 ★★★★ THE DECODE-GAP ILC LAW (fold item 2)

The R6 gap is an ILC-REPEATABLE disturbance of the byte-close/decode path (ILC theorem: the
repeatable component is driven to zero exactly, even under model mismatch; the non-repeatable
part passes through and is floored by σ_meas — SC-14):

  **train-side bar = decoded target − Δ̂:  0.0011 − Δ̂,  Δ̂₀ = 1.0427e-4 [MEASURED R6]
  ⇒ bar₀ = 9.9573e-4 ≈ 0.0010** (consistency row (c) — v4's chosen target, now derived).
  **Within-campaign update: Δ̂ ← ω·Δ_k^measured + (1−ω)·Δ̂ per F13/SC-7 stage-boundary parity
  row, ω = 0.5** (EWMA-R2R stable for gain-ratio ξ ∈ (0, 4) — safe against a 4× decode-model
  error). Falsified if the run-1 parity rows land Δ outside [0, 3e-4] d_seg.

The campaign-scale law (Newton-ILC, contraction arithmetic) lives in §8. Crossing ceilings
(§0.2) are UNCHANGED by this fold — the bar is the design target the arithmetic already used;
what is new is that it now MOVES with the measured per-stage Δ̂ instead of being frozen.

### 2.5 ★★★★ SELF-TRIGGERED VERDICT CADENCE (fold +A; B-CT3, gated on P-CT2)

  **Δt_next = clamp( floor_S / |Ŝ′(t)| , 25, 100 ) ep, floor_S = 0.00178 (post req-F#6).**

Plugged: ep350 (|Ŝ′| = 3.3e-3) → floor 25 binds; ep450 (2.4e-4) → 7.4 ⇒ floor binds; near
exhaustion (1.4e-5) → 127 ⇒ cap 100 binds. Verdicts stretch 25→100 exactly in late-TAU/FIN/TAIL
where they are least informative: **−30–40% n600 verdicts at zero score cost** (each skipped
verdict provably below the decision floor). Degrades safely (floor = today's cadence); alarms
stay per-epoch; req-B three tests + P-CT2 gate (band 12–17 of 41 skipped, kill: any missed best
> 1 cadence). Interaction note (fold-1 seam, declared): stretching the cadence stretches the
co-predicate's V-window in EPOCHS — B1 must window on VERDICT COUNT with per-epoch-normalized
slopes (already MINOR-9 discipline), so the two laws compose without recalibration.

### 2.6 ★★★★ WASHOUT TRANSITION DAMPER (fold +B; B-CT5, gated)

d(t) = k·HP[dL/dt]₊ on the effective LR at stage boundaries — high-pass ⇒ zero DC gain ⇒
CANNOT bias the converged score (structurally admissible even as an actuator); targets the
measured +27.5% cold-fire quench class. Fail-safe = v3 rewarmup ramps unchanged
(`--stage-transition-rewarmup-*`). Injection-tested (req B); A/B falsification vs the open-loop
ramp at matched schedule; kill: no transient-area reduction. ~20 LOC.

### 2.3 chain-A / SOLVE — v4 terminus-corrected record inherited UNCHANGED (ratio 0.163 K=128;
conjunction-dispositive; TerminalSolve OUT; HOLD disarmed; K≥32+K-trend sensor). One addition
from CT-1 §5.2, doc-row only: **mode-admission rule, theorem-named** — a stage/lever enters the
default graph only with a measured (or would-fire-audited) ΔS ≤ 0 record on the common Lyapunov
function S; else only behind a restore-guard (l7 = the measured counterexample that grounds its
standing demotion; the FIN regression guard = the compliant pattern). This is req-B restated
from switched-systems theory; no build. Dwell check recorded: worst measured switch jump
μ = 1.275 (cold Muon) ⇒ τ_d > ln(1.275)/0.026210 ≈ 9.3 ep; shipped min-stage 250 = 27× margin —
SATISFIED, no change.

---

## §3 — CURRICULUM (v5: Conley certificates; island release law; birth-scheduler record posture)

§3.1–§3.3 (lane-first per-class table, LADDER homotopies, per-class weights w_lane = 1.0 /
w_movable = 0.28, B13 fail-safe) inherited from v4 UNCHANGED. Additions:

### 3.4 ★★★★ THE CONLEY PERSISTENCE CERTIFICATE (fold item 4; B17, ~30 LOC, advisory-only)

    island I survives stage k AND decode  ⟸  pers(I) > τ_k·ln5 + Δ_dec^logit
    (pers(I) = birth-death margin amplitude — the existing persistence value)

At τ_end = 0.062: threshold = 0.0998 + Δ_dec^logit ≈ **0.10 logit** (Δ_dec^logit UNMEASURED —
initialized 0, supplied by SC-7's logit-unit extension at the first byte-close; until then the
certificate runs at the τ-term alone, stated). The measured erasure ∝ 1/persistence law (L75) is
this certificate's empirical shadow — the uncertified population dying first. Three surfaces:
1. **Per-island pass/fail ledger** (SC-17) at every stage boundary + the "born-to-die" count
   (births with pers below the NEXT stage's threshold — support spend on them is WASTED unless
   the island-forcing schedule raises pers above threshold before release; converts the island
   curriculum from hope to a ledger).
2. **DEATH ALARM:** an island dying WHILE certified = controller/instrument failure, not physics
   (req-B tested alarm; rides the simple-point island-death counter, CT-2 §2.4).
3. **Release coupling:** B18's r\*(t) release (§1.4 row 4) may drop protection exactly when the
   certificate says the island self-sustains — the two laws share σ_eff/τ inputs (SC-18).
Honest boundary: sufficient-not-necessary (sub-threshold islands MAY survive — matches 44.6%
native survival); pers is measured on the smoothed field so the inequality's sides are not fully
independent. **$0 backtest (recess §7c): survival-vs-pers curves on the 20260630 birth-death
ledger at the two known τ points; band: certified-survival ≥ 95%; kill: < 80% ⇒ fit safety
factor s·(τ·ln5).** Consistency: the MCF-forcing threshold (∝ 1/r ≈ 1/pers for near-critical
islands) is the same physics as the R13-gated 1/pers weighting — CT-2 §5 upgrades that weighting
from heuristic to the forcing-threshold law (doc-row on the v4 §0.3b law, no value change).

### 3.5 Birth-scheduler (CT-2 §6) — RECORD-ONLY run-1, fit run-2 (measure→sweep→derive, req M)

Island birth = a fold (saddle-node) of the margin field; the fold-advance law db_c/dw_c =
−(∂μ_c/∂w_c)/(dμ_c/dt) makes the per-class amplify weights a calibratable birth SCHEDULER (pick
target birth epochs — lane islands born BEFORE τ(t)·ln5 shrinks below their persistence — read
dμ_c/dt off the trace, set w_c(t)). Run-1 keeps v4's constants and RECORDS the pre-birth fold
telemetry (SC-17 fields: per-class near-threshold local-max count/max of ℓ_c − ℓ_runner, dμ_c/dt,
window ‖Δθ‖ response norms); the fit is run-2 work (§8). The adaptive continuation step
(Δλ ∝ 1/‖Δθ‖_window — the pseudo-arclength law behind the 1-Lipschitz easing) is likewise
record-only run-1 → backtest → run-2 (fail-safe = the fixed ramp). LPV bound recorded (CT-1
§6.1): any NEW ramp obeys ramp_length ≥ 3/ν ≈ 115 ep unless a measured deconflict row licenses
faster (the 20-ep band-engage ramp is 1.9× ν — MARGINAL, measured-good, flagged not changed).

---

## §4 — COSTATE + TELEMETRY (v5: the UNION signal ledger; would-fire rows extended)

§4a margin-critical ranking inherited (rank-1 self-orient persist 8.10× · rank-2 F13 5.86× ·
ranks 3–12 unchanged) with SC-9 extended: the trigger would-fire audit now carries THREE shadow
arms (slope arm · forecast arm · forfeit-matched arm) + the PMP ε_stop would-fire row
(7.12e-5 S/ep now; operating-point conversion printed) — zero behavior change run-1, feeds the
run-2 trigger re-derivation.

### 4c ★★★★ SIGNAL-COMPLETENESS LEDGER v5 — THE UNION (fold item 7; requirement P)

Union of v4 §4c (13 rows) + CT-1 §12.2 (5 signals) + CT-2 §12 (5 signals); dedupes stated in
§0.1 row 7. **19 rows.** Contract unchanged: no write-only telemetry; a gap term with no signal
row = a seal-blocking finding; all new rows score-neutral read-only ⇒ default-ON.

| ID | gap term | signal | run-1 generates? | recorded where | named DECIDE consumer |
|---|---|---|---|---|---|
| SC-1 | d_seg per-class | per-class F-rows (shares + rel slopes) | YES (LB set) | run JSONL | per-class λ gates + meat exits + amplify recalibration (RS-4); per-class veto; **ν_c fits emitted explicitly (CT-1 §4.1: eps_c = 0.5·ν_c·d_seg_c — the veto formula now supplied)** |
| SC-2 | d_seg per-stage | stage-boundary verdicts + per-stage EMA ckpts + wall-clock (F12) | YES | per-stage ckpts + F12 rows | TAIL stop rule; schedule MPC; campaign ILC error term |
| SC-3 ★ext | d_seg annulus/margin | #333 margin rows + **live m_q(t) percentile EMITTED PER VERDICT CADENCE** (CT-1 merge) + clamp-binding fraction | YES | annulus rows (default-ON) | τ\*_k re-derivation (TAIL); adaptive-ε window law; τ-confirm; **the §1.4 τ-indexed re-validations** |
| SC-4 | d_seg along/across spectrum | F14 coefficient-energy per along/across bin at the lane annulus, per stage boundary | YES (~20 LOC) | F14 rows | run-2 along-ladder / form-a arm; Rebalance regime check |
| SC-5 | d_pose witness-side | F11 pose-wall watch (first-ever witness-side row) | YES (w_pose>0) | F11 rows | pose kill law (1.5e-4); q\* selection (RS-1); §0.2 pose leg; **EVSI-dominant row (0.044)** |
| SC-6 | rate per-section vs entropy floor | F16 per-section manifest {bytes, H0, coder-vs-H0 %} | YES (every byte-close) | packet manifest JSON | decoded-KKT selection; waterfill stop (RS-2); compress-half go/no-go |
| SC-7 ★ext | decode gap g_dec | **F13 parity per stage boundary in d_seg AND logit units (Δ_dec^logit)** — the CT-1×CT-2 dedupe row | YES | r\*_parity JSONL + packet dirs | decoded-KKT; §0.2 crossing; RS-5; **Conley certificate (Δ_dec^logit term); σ(Δ) bias-variance split — bands: σ(Δ) < 0.5× margin (8.9e-4 S) ⇒ decode drift retires as an attribution term; > 2× margin ⇒ byte-close bug hunt before any sub-margin claim (fail-closed); until n ≥ 3 rows, every A/B delta below 0.0104 S is PROVISIONAL against decode drift (the #2 attribution floor); ILC Δ̂ EWMA (§2.4)** |
| SC-8 | schedule: anneal state | F1 β/τ/progress% + completion flags + M2 alarm | YES | F1 rows | finisher-fire precondition; B9 re-anchor; campaign ILC anneal shape |
| SC-9 ★ext | schedule: meat/exits | F3 AIC forecast + F4 would-fire audit **incl. slope + forecast + forfeit-matched (s\* = 1.4154e-5) + PMP ε_stop shadow arms** | YES | F3/F4 rows | TAU→FIN arms (§2.2f); exit caps; PowerPlay duty ranking; run-2 trigger re-derivation |
| SC-10 ★ext | schedule: TAIL yield | F15 per-cycle {τ_k, m_q(k), Δd_seg, epochs, ΔS/ep} + **inter-cycle partition-Hamming (M4 basin-hop row)** | YES (B14) | F15 rows | TAIL stop rule; req-N inflection curve; **M4 bound decision** |
| SC-11 | capacity (response surfaces #170) | RS-1..RS-5 (pose q · waterfill · WeightEntropy λ · per-class λ · g_dec) | YES | costate SENSE store (persisted models) | costate DECIDE (req M(3)); run-2 capacity; mod-dim 2-point |
| SC-12 | basis: orientation quality | F17 lane-annulus \|cos\| alignment + R12 oracle verdict | YES (~10 LOC; R12 pre-GO) | F17 rows + R12 artifact | R12 gate + fallback; run-2 orientation arm |
| SC-13 | attribution substrate | req-F#6 self-orient persist + recon-gap row (rank-1) + F13 (rank-2) | YES (LB builds) | ckpt fields + gap rows | EVERY consumer above (the attribution floor) |
| SC-14 ★NEW | measurement-noise floor | **verdict-replicate σ_meas** — once per stage boundary: repeat the n600 verdict on the same checkpoint + one across-decode replicate (~10 LOC) | YES once built | replicate rows | every threshold/SE denominator (currently INFERRED from fit residuals — this measures it); ILC resolution limit (no iterate attributes below σ_meas); rank-deficient directions DECLARED not guessed |
| SC-15 ★NEW | forecast/model validity | **forecast-residual row** — per cadence: (model id, predicted ΔS, realized ΔS) (~10 LOC) | YES once built | residual rows | MPC horizon validity online (N\* = 2 cadences, measured not asserted); spectrum-rate mixture model selection; the ep450-class band miss visible in flight |
| SC-16 ★NEW | separatrix asymmetry (req L) | **signed per-class-pair per-DIRECTION flip-mass + one-sided margin histograms** at stage boundaries (~15 LOC) + edge-contrast g_I histograms per pair (deadzone fields) | YES once built ($0 cached-field probe Q1 FIRST) | per-stage F-rows | σ_ij,dir fit (B16 gate); Q1 adjudication; τ\*_k per-side re-derivation; **M1 deadzone census + #149 adjudication**; L-asymmetry negative re-reviews |
| SC-17 ★NEW | topology/island fate | **per-island birth-death ledger with persistence, LIVE at stage boundaries** (not post-hoc forensics) + pre-birth fold fields (near-threshold local-max count/max, dμ_c/dt, window ‖Δθ‖) | YES (B17) | island ledger rows | Conley certificate + death alarm; birth-scheduler fit (§3.5, run-2); erasure-law tracking; adaptive continuation backtest |
| SC-18 ★NEW | interface conditioning / DtO health | **interface-geometry row**: measured interface width (m-profile fit transverse to Γ), σ_eff(t) components, r\*(t) trajectory, annulus conditioning E[(\|∇m\|−1)²], extension-violation E[(∇δm·∇m)²], off-annulus gradient-mass fraction | YES once built (cached fields, cheap) | F-family rows | B18 release law; event-conditioned λ_eik (class-(d): raise λ_eik on conditioning degradation, hold floor otherwise — fail-safe = the v3 ramp); §1.4 clamp-τ re-derivations; **DtO artifact audit: rising off-annulus mass late in TAU = the checkerboard precursor alarm**; M3 bound decision |
| SC-19 ★NEW | solve-inventory input | **per-class logit-field export + per-primitive (band/clamp/comb) annulus residuals** at stage boundaries; band-residual row extended from activation-only (req-F#8) to every stage boundary | YES once built | export + residual rows | §11-row-16 max-plus fit probe; band→INR hand-off certificate (freeze band → residual flip mass per pair → INR trains residual-only); per-class solve ledger; M5 bound decision |

**Completeness check vs the 0.07313 S gap:** d_seg (class ∧ stage ∧ annulus ∧ spectrum ∧
asymmetry ∧ topology ∧ conditioning) ✓ · d_pose (witness-side) ✓ · rate (per-section vs floor) ✓
· decode (g_dec per stage, d_seg + logit units, bias + variance) ✓ · schedule (anneal ∧ meat ∧
tail ∧ forecast-validity) ✓ · capacity (5 surfaces) ✓ · basis (orientation ∧ along-utilization)
✓ · measurement noise (σ_meas) ✓ · every §0.0a bound's deciding measurement mapped (M1→SC-16,
M2→Q2/F12, M3→SC-18, M4→SC-10, M5→SC-19) ✓. **Zero gap terms without a
generated-recorded-consumed row.** New-row loss review (CT discipline §3-CT2 applied to v5
itself): SC-14..SC-19 are all read-only reductions on existing caches/verdicts — no new loss
terms, no continuum-symbol declarations owed.

**Review-checklist row adopted (CT-2 §3, 0 LOC):** every NEW loss term declares its continuum
symbol {boundary-supported (shape-gradient class) | bulk-proper | NONE (artifact-risk — A/B
before adoption)} — binding on seal rounds and run-2 lever proposals (the DtO artifact-tax
audit).

---

## §5 — RATE PLAN — inherited from v4 UNCHANGED (R1 exact bytes: LBND2 41,526 · win9 18,832
roundtrip-exact gated P8/F8 · win5 QUARANTINED · central 93,092 B → 0.0619861 · win9 arm
81,032 B → 0.0539559 · worst tails unchanged · hood 5.32688e-6 S gate · decoded-KKT selection).
One v5 addition, 0 bytes: the byte-close emits **Δ_dec^logit** (one masked logit-unit comparison
at decode — SC-7 field) so the Conley certificate's decode term stops being 0-by-initialization
at the first byte-close.

---

## §7 — MEASUREMENT PLAN + REOPENS + ★★★★ RECESS-QUEUE ADDITIONS

§7 table (P11′ ipe · P8/F8 win9 trained-with leg · F13 per stage · τ-confirm · R12 · R13) and
§7b reopens (Q1 UniWARD per-side — NOW ALSO the B16 gate · Q2 F12 τ-samples · Q3 viscosity
first-fair-test + clamp-binding check) inherited from v4 UNCHANGED. New $0 items:

### §7c recess-queue additions (all $0, pre-registered bands + kills)

| # | probe | band · kill |
|---|---|---|
| P-CT1 | refit ν per stage from the mod32cap trace (the constant under §2.2f/§2.3/§3.5's window laws) | band ν ∈ [0.02, 0.035]/ep; kill: ν < 0.01 ⇒ recompute ALL window laws (115-ep settle, 265-ep cycle floor, dwell bounds) |
| P-CT2 | self-triggered cadence replay on the 41-row trace | band: 12–17 verdicts skipped, no missed best > 1 cadence; kill: any missed best > 1 cadence ⇒ B-CT3 stays unbuilt |
| P-CT3 | forfeit-matched arm backtest on the 41-row trace (fold item 1) | band: first sustained fire ep670–700 (vs shipped ep625), EMA-best-at-fire within 1 cadence of ep650; kill: fires < ep650 or > cap 726 ⇒ arm stays would-fire-only |
| P-CON | Conley certificate backtest on the 20260630 birth-death ledger (fold item 4) | band: certified-survival ≥ 95% at both known τ points; kill: < 80% ⇒ fit safety factor s |
| P-DZ | deadzone census on cached S_R + frames (§0.0a M1) | < 5.34e-6 d_seg ⇒ #149 stays DEFER; > 1.78e-5 d_seg ⇒ #149 enters the duty queue (census — no kill) |
| P-MP | max-plus annulus fit prototype on cached logit fields (§11 row 16) | kill: K ≤ 64 elements/class fails band-level annulus accuracy; pass ⇒ bytes of K coefficient sets enter the λ_bytes law |

Ordering is PowerPlay-consistent (req D): P-CT3/P-CT1 first (they gate a shipped-arm decision),
P-CON next (gates B17's alarm semantics), P-CT2/P-DZ/P-MP after (gate unbuilt/DEFER surfaces).

---

## §8 — ★★★★ THE CAMPAIGN LAYER (fold items 2 + 9; CT-1 §7/§9/§12.3 — run-to-run control, not run-1 config)

1. **Newton-ILC update:** u_{k+1} = u_k + γ·P̂⁻¹·e_k on the identified lever subspace, γ = 0.7;
   with model error bounded by the measured ~35% instrument gap (ξ ∈ [0.65, 1.35]), per-run
   contraction |1 − γξ| ∈ [0.055, 0.545] ⇒ **2–3 runs to the identified floor** — the
   control-theory form of Model B's with-repair band. Δ̂ (decode gap) rides the §2.4 EWMA.
2. **Identifiability = the req-P convergence precondition, formalized (CT-1 §7.3):** the gap is
   identifiable iff (i) EXCITATION RANK — the stacked config-delta matrix (+ within-run F8
   single-lever events as rank-1 excitations) has rank ≥ |controlled lever set| (the λ=0 twin +
   mirror-schedule twin + F8 activations are the rank suppliers; the 13-simultaneous-diff pair
   is the measured anti-anchor); (ii) σ_meas MEASURED not inferred (SC-14); (iii) MATCHED
   INSTRUMENTS (req H) — a verdict-semantics change invalidates the pair for ILC, full stop
   (the crutch↔fix anti-anchor). Missing any leg ⇒ the campaign converges to an UNATTRIBUTED
   plateau. **Excitation-rank check runs before each run's config freeze.**
3. **ILC pair-admission quarantine (~10 LOC campaign-ledger check):** runs with a parity-row/F9
   semantics fault are auto-excluded from ILC pairs (fault DETECTION exists; this adds the
   quarantine).
4. **EVSI duty-queue ranking:** §0.0c IS the duty-to-measure ordering for signal adds.
5. **ES admissibility rule (req J(2) formalized):** sweep step a\* = max(2·floor_S/ĝ, knob
   resolution). Rate knobs at ĝ = λ_bytes: ≥ 5.3 KB/arm at the margin floor, **≥ 45 KB at
   today's 0.015 recon-gap floor ⇒ fine rate sweeps are unattributable until req-F#6 lands —
   the ckpt-fidelity fix is a PRECONDITION of the rate-sweep program**, not hygiene. Any arm
   with predicted |ΔS| < floor is UNATTRIBUTABLE-BY-CONSTRUCTION and not launched.
6. **Family-inflection watch on the design process itself:** measured contraction record
   v1→v2→v3→v4 finding counts 17 → 6+13 → 1B+2M+6m → (seal pending); if the per-round finding
   rate stops contracting, req-N(3) fires on the DESIGN family — switch families, don't polish.
7. **Run-2 fits from run-1 records:** birth-scheduler w_c(t) calibration (§3.5) · adaptive
   continuation step Δλ ∝ 1/‖Δθ‖ (backtest first) · trigger re-derivation from the SC-9 shadow
   arms · λ(t)-decay model selection from SC-15.

---

## §9 — PREDICTED S LADDER — inherited from v4 UNCHANGED (central ≈ 0.26 does NOT cross —
stated plainly; the crossing is the engineered gated tail per §0.2; dual probability model:
independent 2–6% / with-repair 8–15% incl. run-1.5 branch, labeled). §9.4 run-2 additions
inherited + v5 adds: max-plus expansion lever (if P-MP passes) · signed one-sided viscosity ·
birth-scheduler w_c(t) · adaptive continuation · self-triggered cadence promotion (if P-CT2
passed but B-CT3 missed the GO train).

---

## §10 — BUILD LIST (v5 deltas; v4 B1–B15/F13–F17/I-6 rows stand)

| id | build | ~LOC | status/route |
|---|---|---:|---|
| ★★★★ **B-CT1** | forfeit-matched TAU→FIN second arm, WOULD-FIRE first (s\* = 1.4154e-5 S/ep on the per-epoch-normalized slope; an F4 audit-row variant inside the B1 trigger surface) | ~10 | promotes to firing arm iff P-CT3 + injection pass; cap 726 unchanged |
| ★★★★ **B1 spec change** | verdict co-predicate window **V = 5** (was 4; 125 ep ≥ 3/ν = 115 — two-timescale separation) — binds the B1 build (~80 LOC, owed) AND the out-of-process advisory | 0 (spec) | the ONLY V surface; `--curriculum-plateau-windows` (ep_loss) deliberately untouched |
| ★★★★ **B16** | signed σ_ij,dir per-class-pair per-direction slots on the margin-saliency lever surface + DSL Lever param + activation-ledger row; DEFAULT-OFF, byte-identical OFF | ~25 | GATED on Q1 (fire iff \|ρ\| ≥ 0.3 any side); duty-to-measure registered |
| ★★★★ **B17** | Conley certificate: per-island pass/fail ledger (SC-17) + born-to-die count + certified-death ALARM (req-B tested); consumes existing persistence machinery + τ_k + SC-7's Δ_dec^logit | ~30 | advisory-only; P-CON backtest pre-GO |
| ★★★★ **B18** | island release law r\*(t) = 0.95·σ_eff(t) on the eased-homotopy surface (`--seed-island-eased` + `--island-dilate-px` + `--seed-anneal-*`); release when r\*(t) < native dash half-width | ~10 | fail-safe = v3's fixed 275-ep ramp (req B); consumes SC-18's σ_eff row |
| ★★★★ **B-CT3** | self-triggered verdict cadence Δt = clamp(floor/\|Ŝ′\|, 25, 100) | ~15 | GATED on P-CT2; alarms stay per-epoch; composes with B1 via verdict-count windows (§2.5 seam) |
| ★★★★ **B-CT5** | washout LR damper at stage boundaries (zero-DC high-pass) | ~20 | GATED + injection-tested; fail-safe = rewarmup ramps |
| ★★★★ **B-CT4** | clamp→projection form for adaptive-ε (tangential, not saturating) | ~10 | GATED on Q3's clamp-binding check (> 90% binding ⇒ build; else moot) — v4 §1.2 window law unchanged |
| ★★★★ **F18–F23** | the six NEW SC rows: σ_meas replicate (SC-14, ~10) · forecast-residual (SC-15, ~10) · signed per-direction histograms + g_I (SC-16, ~15) · live island ledger + fold fields (SC-17, in B17) · interface-geometry (SC-18, ~20) · per-class logit export + per-primitive residuals (SC-19, ~15) | ~70 total | all score-neutral read-only ⇒ default-ON (orphan rule); consumers named in §4c |
| ★★★★ **I-7** | equation registrations, tranche 2: the three §0.4 consistency rows + `pmp_stop_rate_epsilon_v1` (ε_stop(t) = floor_S(t)/cadence(t)) + `forfeit_matched_exit_v1` (s\* = ν·forfeit) + `conley_island_certificate_v1` (pers > τ_k·ln5 + Δ_dec^logit) + `critical_nucleus_release_v1` (r\* = 0.95σ_eff) + `decode_gap_ilc_feedforward_v1` (bar = target − Δ̂, ω = 0.5) — ALL tagged PROVISIONAL / fresh-research-round-1-derived until the seal reviews CT-1/CT-2 (#363 discipline) | 0 | rides the P7 triality landing |

---

## §11 — SOLVE INVENTORY (req A(ii)) — v4 rows 1–15 stand (TerminalSolve terminus-corrected). One addition:

| row | block | verdict |
|---|---|---|
| ★★★★ 16 | **max-plus band-residual decomposition** (fold item 9 — the surviving essence of backstepping, placed here so #342 solve-don't-train finds it): re-express the plant in coordinates where the hard part is solved, control only the residual — m_lane = max(band_analytic, m_INR); freeze band → measure per-pair residual flip mass (SC-19) → INR trains residual-only (margin-gated). Generalization: fit a max-plus expansion (max of K quadratics/low-order polys) DIRECTLY to the frozen scorer's per-class logits on the annulus — per-basis fitting = small least-squares over the tropical polyhedral decomposition; band/clamp/comb are the K=1 special cases; rule-118 makes the basis GENERATORS free at decode (McEneaney curse-of-dim-free = the PDE-numerics name for our free-decode column). Coupling bound: per-class decomposition error confined to {m < max_c δ_c + τ·ln5} — free for the bulk, exactly-marginal on the flip population (τ_end coupling = 0.0998). Bulk stays INR/curvelet (M5 bound — the two-semiring split is forced) | SOLVABLE-candidate; P-MP $0 prototype (kill: K ≤ 64/class fails band-level annulus accuracy); bytes of passing K enter the λ_bytes KKT law |

## §12 — ★★★★ DEAD / CAMPAIGN-ONLY LEDGER (fold item 9 — routed, not dropped)

| item | verdict | surviving essence / route |
|---|---|---|
| PDE backstepping kernel machinery (Volterra/Goursat; Koga–Krstic Stefan) | **DERIVED-DEAD for this plant** (no 1-D spatial causality — 2-D curve network; actuator acts through the loss, not the domain boundary; the descent is already Lyapunov-stable) — campaign detour saved | essence = §11 row 16 (max-plus band-residual decomposition) + the per-stage band-residual certificate row (SC-19) |
| LQR/Riccati direct; grid/NN HJB value solve | DEAD (no linear plant; dim(θ) ~1e5 kills grids; the value function enters only via λ = ∇V estimation) | costate estimator (existing) |
| continuous ES dither on θ | DEAD (per-step dither fights the optimizer) | campaign-timescale FD-ES only (§8.5 sizing law) |
| Griewank revolve checkpointing | DEFER-with-reason (per-stage ckpts + EMA already exceed 1-level adjoint needs; relevant only if a through-time inner-PDE lever lands) | reactivation criterion named |
| Hajek log-τ cooling | already REFUTED (viscosity hunt §6) | M4 bound + TAIL_k repair stand |
| Imbert–Monneau full flux-limiter formalism | NOT imported (the σ_ij Young-angle fit is its actionable projection, already queued) | viscosity hunt §7 queue |
| CAMPAIGN-ONLY (not run-1 config): Newton-ILC gain law · excitation-rank check · EVSI ranking · pair-admission quarantine · family-inflection watch · birth-scheduler fit · adaptive continuation | routed to §8, each with its owner-law | — |

---

## SELF-ATTACK (operating-manual discipline; v4's five attacks stand — three new ones)

1. **Both CT sources are fresh-research-round-1 UNREVIEWED — am I building on sand?** The
   posture is graded: (i) laws that only ADD would-fire/telemetry rows (B-CT1 shadow, SC-14..19,
   PMP row) are risk-free by construction — wrong theory = wasted rows, zero score path;
   (ii) laws that CHANGE behavior are ALL gated on $0 probes with pre-registered kills (P-CT3,
   P-CT2, P-CON, Q1, Q3) or ship with fail-safes (B18's fixed-ramp, B-CT5's rewarmup, cap 726);
   (iii) the three §0.4 consistency rows are the strongest evidence the CT derivations are
   real — independent agreement with measured constants to 0.2–5%. The seal rounds review the
   CT sources themselves (their provenance tags are carried on every derived row).
2. **The forfeit-matched arm's ν = 0.026210 comes from ONE stage of ONE run (ep350→ep450
   mod32cap).** Correct — and every window law (115, 265, 9.3, 60) inherits it. P-CT1 refits ν
   per stage BEFORE any of these numbers is load-bearing (kill ν < 0.01 ⇒ recompute all);
   the arm ships would-fire-only regardless. Labeled DERIVED-from-one-trace.
3. **Did the union ledger smuggle in compute that breaks the verdict budget?** Checked: SC-14
   adds ~1 extra n600 verdict per stage boundary (4–6 per run) + one across-decode replicate;
   SC-16..19 are cached-field reductions; F13 was already booked (~4.7 min/ckpt). Against
   B-CT3's −30–40% late-run verdicts, the net verdict count DROPS if P-CT2 passes; if it fails,
   the adds are ~5 verdicts on ~41 — +12%, bounded, and each is EVSI-positive per §0.0c.
   No new loss terms (§4c end), so no continuum-symbol or term-domination risk added.
4. (v4's attacks 1–5 re-checked against v5: the win9 single-point-of-failure statement, the
   TAIL-books-no-gain check, the k_max² anchor caveat, the no-re-litigation check, and the
   asymptote-low-edge caveat all still hold verbatim; §0.2 confirms no crossing number moved.)

Pointer 0.19110 UNMOVED — this draft is MEANS until the §7 ROW lands.
