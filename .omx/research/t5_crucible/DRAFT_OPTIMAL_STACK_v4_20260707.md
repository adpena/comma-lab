---
doc_type: t5_crucible_p3c_revised_synthesis_draft_v4
role: v4 DESIGNER (operator-convened T5 crucible; recursive-seal round-2 fold — precision fold,
  NOT a redesign: the architecture, crossing arithmetic FORM, and lane-first curriculum SURVIVED
  round 2; this revision folds the 13-item fold list 1:1)
date: 2026-07-07
supersedes: DRAFT_OPTIMAL_STACK_v3_20260707.md (11017be0e) — v3 preserved append-only. v4 folds:
  seal_round2_verdict (1 BLOCKER · 2 MAJOR · 6 MINOR) + recess_wave1 R1/R3/R6 measured rows +
  negatives_scale_validity_review (TAIL_k verdict + reopens) + requirements J/K/L/M/N + the
  mid-fold operator pin requirement P (signal-completeness ledger, §4c).
epistemic_contract: unchanged — every knob carries a CONTROL LAW class {(a) CONSTANT · (b)
  RAMP/ANNEAL+completion guarantee · (c) SELF-DERIVING · (d) EVENT-CONDITIONED · (e)
  FRACTIONAL/PARTIAL} + a tag {V-S · V-A · D · DPR}. Every load-bearing number labeled
  MEASURED / DERIVED / INFERRED / ASSUMED (operating-manual discipline,
  docs/operating_manual_craft_handoff.md). Nothing unmeasured asserted as measured.
axis: all numbers [macOS-CPU/MLX advisory] unless tagged; archive bytes exact (zip stat).
  Pointer contest-CPU 0.19110 UNMOVED — this whole file is MEANS.
review_status: pre-registered-only (v4, awaiting seal round 3)
---

STORES CONSULTED: ORCHESTRATION_LEDGER.md (full; reqs A–O binding, esp. J/K/L/M/N + the
2026-07-08 landings-folded block) · DRAFT_OPTIMAL_STACK_v3_20260707.md (full — the base) ·
seal_round2_verdict_20260707.md (full — BLOCKER-R2-1, MAJOR-R2-2/3, MINOR-R2-4..9, §4 certified
list) · recess_wave1_R1_R3_R6_20260707.md (full — R3 AA-gate table, R1 LBND rows, R6 parity row)
· negatives_scale_validity_review_20260707.md (full — 9 ROBUST/11 BOUND/4 SUSPECT/1 REOPENED,
§3 TAIL_k law, §4 summary) · trainer argparse RE-VERIFIED this session (grep add_argument:
`--softmax-temp-end` L7365 EXISTS, default 0.05; `--tau-anneal-end` DOES NOT EXIST) ·
witness_config_differential_equations_derivation_20260705.md (the #318/#320 derivation — the
two-sided CFL window L145–L196 used for the §1.2 clamp τ-law re-derivation). NOT consulted:
durable-state files (stale per sweep); position_S* full re-reads (carried via v3 + the two
verdicts); no training launched, no live config touched ($0 reading + arithmetic only).

# T5 CRUCIBLE — P3c DRAFT v4: THE OPTIMAL FULL STACK (round-2 precision fold; g_dec in the crossing; TAIL_k; ipe AA)

## §0 — REQUIREMENT-N FRAMING: WHAT THIS FAMILY CAN CLAIM, HONESTLY (fold item 13)

**Answer first.** The frontier-to-floor gap is 0.19110 − 0.11797 = **0.07313 S** (floor MEASURED,
rate-dominated S_floor per the d_seg floor analysis / CLAUDE.md GOAL section). v4's estimate of
THIS family's asymptote (composed-lever level-set witness on the mod32 control basis, with band +
islands + anneal-completion + TAIL_k cycles + the within-family run-2 levers):

- **S_asymptote ≈ 0.165 central, band [0.154, 0.181]** (DERIVED from: d_seg family-asymptote
  0.0009–0.0011 — the §0.3 optimistic edges plus TAIL_k/per-class-τ_c headroom, floored by the
  lane-band isolated 0.00087-class bound [MEASURED]; pose term 0.01732 at the 3e-5 bar
  [mechanism BUILT-UNFIRED]; rate 0.047–0.054 — waterfill-lower 52,000 + win9 18,832 [MEASURED]
  + pose 2,700 + manifest − grammar).
- **Fraction of the frontier-to-floor gap this family claims: ~36% central, band [14%, 51%].**
- **T_3 = 0.15 sits AT or BEYOND this family's optimistic asymptote edge.** Reaching sub-0.15
  most likely requires the FAMILY STEP (req-N(3) inflection): quotient-codec paradigm #155,
  compress-half to rate ≈0.04–0.05, and/or the d_seg 0.0005–0.0009 regime — all named run-2+
  design work that consumes run-1's telemetry. v4 does not claim T_3 for this family.
- **Run-1's instrument value (req-N(1)), independent of crossing:** the per-stage decode-gap
  parity rows (F13, §4), the per-class F-row share-transfer measurement (P6), the lane-bearing
  finer-τ spectrum rows (the negatives-review reopening condition 1d), the response-surface
  seeds RS-1..RS-5 (§4b), and the TAIL_k per-cycle Δ measurements — the corpus that makes
  v5/run-2 DERIVABLE. Run-1 is a two-wall measurement instrument with a real, engineered,
  gated crossing tail (§0.2) — framed that way per requirement N.

## §0.1 — FOLD RESOLUTIONS, 1:1 WITH THE FOLD LIST (the v4 revision contract)

| # | fold item | resolution in v4 |
|---|---|---|
| 1 | BLOCKER-R2-1: AA(ss=2) measurably REFUSED; R1 exact bytes | **AA ships as `--render-aa ipe` in run-1** (§1.1/§1.2); ss=2 → run-2 with its measured cost WRITTEN (§9.4: aa_fine 34.36 GiB @ ndf4/full → projected 105.90 GiB > 89.6 REFUSE rc=3; ndf=2/full 87.91 GiB margin 1.69 GiB < 10 GiB assumed-margin ⇒ margin-REFUSE; batch memory-SAFE but ~29 s/ep EDT thrash = wall-clock-killed; plus the trainer Wave-D MEASURED −49% witness-harm on supersample). §0.3 AA citations re-scoped to ipe; §8 s/ep base re-scoped. R1 bytes folded into §5.1: LBND2 = **41,526 B** (41,562 was the stale seat), LBND4 **win9 = 18,832 B** roundtrip-exact (saves 12,060 B = **0.008031 S = 4.51× margin** vs 30,892 central; **win5 QUARANTINED** — decode→re-encode identity FALSE all 3 schemes) |
| 2 | MAJOR-R2-2: 4 stale "K=128 ratio 0.011" sites | ALL replaced with the measured chain-A TERMINUS: **ratio 0.163 at K=128** (strict kill <0.1 NOT reached — middle zone; persist >0.5 decisively excluded; full-P extrapolation **≈0.08 DERIVED**). Re-derivation §2.3: the terminal DISPOSITION survives on the CONJUNCTION (1/√K collapse + holdout ±sign transfer + isotropy + every solve step measured non-descending), NOT on the strict-kill number; §11 row 13 + I-6 registry spec REWRITTEN to the terminus framing (§10/§11) |
| 3 | MAJOR-R2-3: per-class amplify weights unbuildable | **B13 build item added** (§10): exact flags `--amplify-weight-lane` / `--amplify-weight-movable` + third `--amplify-persist` kind `persistence_pairs` (R13-gated); DSL `AmplifyIsland.weight` widened float→float\|dict + compile branch + validate(); ~60 LOC total. **Fail-safe demotion written:** if B13 unlanded at GO, run-1 ships POOLED weight=1.0 with the per-class law a NAMED run-2 build (matching the per-class τ_c handling) — the §1.1 sketch carries both forms explicitly |
| 4 | MINOR-R2-4: invented flag `--tau-anneal-end` | Fixed everywhere: the real flag is **`--softmax-temp-end`** (RE-VERIFIED this session: trainer argparse L7365, default 0.05; DSL gauge emits it). §1.2 row header corrected |
| 5 | MINOR-R2-5: hood pay-threshold 10× slip | Fixed at full precision (req J): 8 B × 6.6586e-7 S/B = **5.32688e-6 S** ⇒ pays iff **Δd_seg > 5.3269e-8** (equivalently 5.3269e-6 S = **0.2993% of the 0.00178 margin** — same-currency per J(4)). Operative gate unchanged (the stricter paired verdict) |
| 6 | MINOR-R2-6: adaptive-ε "rarely fires" stale; clamps → τ-law | §1.2 row REWRITTEN with the τ-coupled statement + the **clamp τ-LAW derived inline** ($0, from the #318 memo's own two-sided window) + the **upper-clamp-saturation ALARM row** (§4). See §1.2 |
| 7 | MINOR-R2-7: recon-gap fix not margin-ranked | §4 gains the **margin-critical telemetry ranking**: self-orient persist (req-F #6) is **rank #1** at +1.4411e-4 d_seg = 0.014411 S = **8.10× margin** (R6 MEASURED — it gates attribution of every win that matters); promoted from pooled "strongly-wanted" to a NAMED LB-class build |
| 8 | MINOR-R2-8: AA×island row + along=8 regime sentence | §4 adds the **AA×island-survival attribution row** (per-class AA paired deltas at stage boundaries — req I); §0.3/§1.2 add the **along=8 regime-boundary sentence** (req K): along=8 is justified ONLY in the lane_offloaded regime (band trained-with ⇒ Candès-Donoho along-optimum ≈6; the DSL Rebalance factory brands across=32/along=8 BACKWARDS in the lane-carried regime); if P1 fails or the band reverts, run-1 stays along=8 (control) but the lane-carried regime + the along-bandwidth question REOPEN (negatives-review item 3/4 branch arm) |
| 9 | MINOR-R2-9 / req M: reactive-only laws; response surfaces unnamed | (i) TAU→FIN co-predicate gains the **FEEDFORWARD form** (§2.2): fire when the F3 online-meat AIC FORECAST of remaining stage meat < the forfeit it would recover (measured forfeit 5.4e-4 S) — the trailing-slope form stays as fallback; (ii) adaptive-ε gains its **ε_ff(t) = ε(ĉ_a(τ(t)))** feedforward term (§1.2 — the τ-coupling IS the internal model); (iii) **response-surface seeds NAMED** feeding #170 (§4b): RS-1..RS-5 persisted to the costate SENSE store as DECIDE models, not one-shot gates |
| 10 | R6 decode-gap (load-bearing) | **g_dec enters the crossing triple explicitly** (§0.2): measured **+1.0427e-4 d_seg = +0.010427 S = 5.86× margin** at ep650/mod32cap [MEASURED, R6]. Crossing recomputed honestly (§0.2) — plain statement included. **F13 telemetry row added** (§4): per-checkpoint byte-close parity at stage boundaries (chunked, resumable — the R6 driver machinery, ~4.7 min inflate + chunked verdict per checkpoint). §5.0 joint KKT law extended: byte-close SELECTION runs on the DECODED verdict, making g_dec a selection variable, not a fixed tax |
| 11 | TAIL_k warm-restart cycles | §2.1 stage-graph edge REPLACED: `per-class-meat-exhausted → TAIL_k` (was → END); the full law §2.2e (τ_k = max(τ_{k-1}/2, τ*_k re-derived from the LIVE margin field); LR_k ∝ τ_k; per-cycle meat exits; PowerPlay stop rule at the attribution floor; **END demoted to the req-B fail-safe cap**, cap_tail = 2× TAU stage length, injection-tested). **B14 build item** (~40 LOC — all pieces built) + DSL `TailCycles` Schedule object (§10) |
| 12 | Reopens queue (not run-1 blockers) | §7b: (i) UniWARD per-class-pair per-DIRECTION signed re-test ($0, cached S_R + texture fields, <5 min; kill \|ρ\|<0.1 both sides ⇒ SCALE-ROBUST dead; \|ρ\|≥0.3 any side ⇒ signed-hinge UNIWARD term enters the queue with a real prior); (ii) F12 dash-contrast sampling at **τ ∈ {0.216, 0.12, 0.062}** (separates the τ-completion leg from comb in dash attribution); (iii) **viscosity REOPENED** (FEED-06h confound — never fairly tested): run-1's eikonal ramp under live spike/liveness guards **IS the first fair test** — noted as such in the §4 telemetry expectations |
| 13 | Req-N §0 framing | §0 above: family-asymptote estimate (≈36% of the 0.07313 gap central, band [14%, 51%]) + run-1 instrument-value framing |
| P | ★ OPERATOR PIN requirement P (arrived during this fold): signal-completeness | **§4c SIGNAL-COMPLETENESS LEDGER added** — the S-vs-S_floor gap decomposed term-by-term, every term with {signal → run-1 generates? → recorded where → NAMED consumer}; two gap terms found with NO signal row (along/across spectrum utilization; TAIL_k per-cycle yield) — FIXED in v4 as F14/F15, not footnoted. Fold-item-9's RS-1..5 + telemetry additions are FOLDED INTO the ledger (single table, no scattered rows) |

## §0.2 — THE CROSSING ARITHMETIC v4 (g_dec IN; fold item 10; req J full precision)

**The crossing condition now binds on the DECODED surface** (R6 proved decode is trusted but not
free). With g_dec = decode-leg d_seg gap (int8-quantize + brotli roundtrip + fp64 inflate +
chunked-verdict read-back), MEASURED **+1.0427e-4 d_seg** at ep650/mod32cap:

  **100·(d_seg_train + g_dec) + √(10·d_pose) + rate < 0.19110**

Rate legs (§5.1, R1-folded): central 93,092 B → **0.061986** · win9-band arm 81,032 B →
**0.053956** (gated: P8/F8 trained-with leg — smoothing is a LOSSY geometry change; win9 is
roundtrip-exact but its NET-S effect is unmeasured until the trained-with probe).

| triple (d_seg_train, d_pose, rate) | S decoded (g_dec = 1.0427e-4) | crosses 0.19110? |
|---|---|---|
| v3 triple as written (0.0011, 3e-5, 0.061986) | 0.110000 + 0.010427 + 0.017321 + 0.061986 = **0.199734** | **NO — over by 0.008633 = 4.85× margin.** Stated plainly: the v3 engineered crossing does NOT survive g_dec unmodified |
| **v4 triple (0.0010, 3e-5, 0.061986)** | 0.100000 + 0.010427 + 0.017321 + 0.061986 = **0.189734** | **YES — margin 0.001366** (77% of the old 0.00178) |
| **v4 + win9 arm (0.0010, 3e-5, 0.053956)** | 0.100000 + 0.010427 + 0.017321 + 0.053956 = **0.181703** | **YES — margin 0.009397 = 5.28× the old margin** |
| win9 arm at the OLD train target (0.0011, 3e-5, 0.053956) | 0.191703 | NO — over by 0.000603 (win9 alone does not rescue 0.0011) |

**Required train-side targets (exact):** central rate ⇒ **d_seg_train ≤ 0.0010137**; win9 rate ⇒
**d_seg_train ≤ 0.0010940**.

**The plain statement (fold item 10):** the engineered crossing STILL CLEARS 0.19110, but the
train-side design target TIGHTENS from ≤0.0011 to **≤0.0010137 at central rate** — a sliver only
1.37e-5 d_seg (1.4%) above the §0.3 optimistic design-sum edge (0.0010), i.e. every class must
land essentially AT its optimistic edge simultaneously (v3's "≤10% of edge" condition collapses
to **≤1.4%**). The R1 win9 coder (0.008031 S saved, 4.51× margin) nearly exactly offsets the
g_dec hit (0.010427 S, 5.86× margin): with win9 admitted, the requirement relaxes to
**≤0.0010940 ≈ 9.4% above the optimistic edge — the v3 condition essentially restored.** The
crossing is therefore CONDITIONAL on (a) the win9 trained-with gate (P8/F8) passing, OR (b) the
run beating its own optimistic d_seg edge, OR (c) g_dec shrinking below its ep650 measurement —
and (c) is now an ENGINEERED lever, not a hope: §5.0's joint KKT selection runs on the DECODED
verdict (per-checkpoint F13 parity rows), so byte-close picks the (checkpoint × quantization
depth) that minimizes decoded S, trading waterfill bytes against the quantize leg of g_dec under
the same λ_bytes law. Honesty notes: g_dec = 1.0427e-4 is ONE checkpoint's measurement (mod32cap
ep650); per-stage F13 rows re-measure it (band [0.5e-4, 2e-4] is my ASSUMED prior, not data).
Pose decode-gap: unmeasured on a pose-bearing run (R6 was pose-blind); the store-nothing ξ
carrier decodes exactly by construction (derive-H) so pose g_dec is EXPECTED ~0 [ASSUMED —
F13 verifies at the first pose-bearing byte-close].

Consequences: pose success bar 3e-5 unchanged; **d_seg design target ≤ 0.0010 (tightened)**;
rate ≤ 0.062 central with the win9 arm as the margin restorer; §9.1's "byte-closed realized
+0..+1e-4" rung is REPLACED by the measured g_dec rung (its prior allowance is CONSUMED, not
spare — R6's exact words).

## §0.3 — d_seg DESIGN ARITHMETIC (Surface B; AA re-scoped to ipe; the tightened edge condition)

Per-class table unchanged in shares/mechanisms (Surface-B: lane 0.4396 / movable 0.1226 /
big-3 0.4378 — certified round 2, not re-litigated) with TWO edits:

1. **Every "AA sub-px" lever citation now reads "ipe cone-AA (basis-level, ~0 memory, self-orient-
   compatible, wired)"** — the ss=2 form is REFUSED by its own launch gate (R3 MEASURED) and
   carried a MEASURED −49% witness-harm negative; its removal is plausibly S-favorable, not a
   loss [the ipe form's d_seg contribution is UNMEASURED — design bands held, AA-leg re-tagged
   ipe-form/unmeasured].
2. **The edge condition (from §0.2):** crossing at central rate needs the design sum ≤ 0.0010137
   — every class within **≤1.4%** of its optimistic edge simultaneously; with win9, ≤9.4%.
   Islands-only honesty unchanged: full island fix alone floors at ≈0.0015; below that is the
   big-3/anneal leg (τ_end=0.062) + the TAIL_k cycles (§2.2e) — which is exactly where the
   negatives review says the coarse-point "exhausted" verdicts do not bind.

**along=8 regime-boundary sentence (fold item 8, req K):** the across=32/along=8 capacity split
is NATIVE only in the **lane_offloaded regime** — band trained-with carries the lane's
along-tangent energy, moving the basis's Candès-Donoho along-optimum toward ≈6, so along=8 is
GENEROUS there (train-big-compress-small compliant). If P1 (comb-registration) fails or the band
reverts at byte-close, run-1 still ships along=8 (the control value, ordering-guard primary),
but the regime flips to lane-carried — where the DSL Rebalance factory itself brands this split
BACKWARDS — and the along-bandwidth ladder + lane_carried demotion REOPEN as run-2 questions
(negatives review items 3/4: form-a retrain arm at along∈{8,26} under τ_end=0.062, comb OFF).

§0.3b (per-class weights): law unchanged (w_lane=1.0, w_movable=0.28 from Surface-B shares;
per-island w_i ∝ 1/pers_i clamped [0.25,4.0], R13-gated) — **now with a build surface: B13**
(§10, fold item 3). If B13 is unlanded at GO: pooled weight=1.0 ships, per-class law → run-2
NAMED (the honest fallback the seal demanded; no dict-into-float fake config).

---

## §1 — THE WitnessProgram (v4 deltas from v3 marked ★★★; v3 ★★ inherited)

### 1.1 Program sketch

```python
prog = WitnessProgram(
    purpose="T5 crucible ARM-PRIMARY v4: lane-first islands + ipe-AA + band(win9-armed) + pose + TAIL_k",
    base=Mod32SegOnlyControlBase(),
    curriculum=sealed_205_curriculum(cfg, handoff="event"),   # anneal-epochs 600; --softmax-temp-end 0.062
    levers=[
        SeedIslandBirth(), SeedIslandEased(), EventTriggeredCurriculum(),
        LogitAdjust(tau=1.0),                                  # priors pinned (v3 §1.2)
        # ★★★ B13-gated per-class form; fail-safe = pooled 1.0 if B13 unlanded at GO:
        AmplifyIsland(form="hinge", weight={"lane": 1.0, "movable": 0.28},  # requires B13
                      gated="witness_alone"),                  # else: AmplifyIsland(form="hinge", weight=1.0, ...)
        PersistenceTopology(weight="1/pers clamped [0.25,4] (R13+B13-gated; else pooled)", warmup=275),
        CacheGtSkeleton(), LengthSigma("fitted-20260707"),
        # RENDER SUBSTRATE ★★★ (BLOCKER-R2-1 fold: ipe is the surviving AA form)
        AACoverageRender(mode="ipe"),                          # trainer: --render-aa ipe (R3-verified wired);
                                                               # DSL param-hold VERIFIED at build (B15 if factory lacks mode)
        AnalyticLaneRenderBand(start=350, boundary_relative=True, v_h=174),
        ChromaBoundarySharpen(weight=0.1, margin_band=1.0, start="tau_fire"),
        MuonWarmStart(lr_final_frac=0.1),                      # entry = TAU-window EMA-best (v3 §2.2c)
        WeightEntropyPenaltyMLX(lam=15),                       # twin lam=0, mirror-schedule
        GNSpectrumProbe(k_pairs=">=32 + K-trend row"),
    ],
)
prog.validate()
```

### 1.2 Knob → control-law table (v4 DELTA rows only; all unlisted v3/v2/v1 rows inherited)

| knob | value/law | class | tag |
|---|---|---|---|
| ★★★ **`--softmax-temp-end`** (τ_end) | **0.062** — flag name CORRECTED (fold item 4; argparse L7365 RE-VERIFIED; `--tau-anneal-end` never existed). Law unchanged: τ_end* = m_q/ln5 from the measured flip-annulus support; τ-confirm $0 probe pre-GO | b | D |
| ★★★ `--render-aa` | **`ipe`** (basis-level cone AA; ~0 memory/compute; self-orient-compatible; wired) — the R3-surviving form. ss=2 supersample → run-2 with measured cost written (§9.4). The P11′ gate re-runs on the ipe config (expected trivially SAFE — aa_fine term 0) | a | **V-A (R3 MEASURED refuse of ss=2; ipe wiring verified; ipe d_seg effect UNMEASURED)** |
| ★★★ adaptive-ε (eikonal) — **the clamp τ-LAW, derived inline** (fold item 6; $0, from the #318 memo's own two-sided window L145–L196) | Raw law unchanged: ε_raw(t) = \|c_a(t)\|·√(η(t)·λ_eik(t)/8)·(1+m), m=0.5. **The derived stability window is** `ε_lower(t) = \|c_a\|√(ηλ_eik/8) ≤ ε ≤ ε_upper = √(2/(ηλ_eik))/k_max²` — BOTH edges are functions of live quantities; the shipped constants (0.3, 0.7) were the MEASURED-SAFE sub-window at COARSE τ (ε=0.3 stable, ε=1.0 explodes) wearing a law's clothing (negatives-review item 6). **τ-LAW replacement (class c):** floor_t = max(0.25 [DPR fail-safe], ε_lower(t)) — the floor TRACKS the live CFL lower edge, so descending τ (which GROWS \|c_a\| per the memo's own coupling L162) raises the floor with the physics instead of silently re-freezing the adaptive branch; upper stays 0.7 (measured-anchor, below the ε=1.0 explosion). **Window-closure law (the deep consequence):** window width ratio ε_upper/ε_lower = 4/(\|c_a\|·η·λ_eik·k_max²) NARROWS ∝ 1/\|c_a\| as τ descends; saturation (ε_raw > 0.7) occurs at \|c_a\| ≳ 93 with current constants (vs measured 11.2 coarse) [DERIVED]. **Feedforward/MPC term (req M(2), fold item 9):** ε_ff(t) = ε(ĉ_a(τ(t))) with ĉ_a(τ) forecast from cached margin fields per τ (the known τ-schedule IS the internal model); on sustained saturation (ALARM, §4) the derived RESPONSE is feedforward, not reactive: scale η·λ_eik down multiplicatively (both window edges scale as √(ηλ) and 1/√(ηλ) — reducing ηλ OPENS the window from both sides) until ε_raw ≤ 0.6; falsification band: if ε_raw > 0.7 persists 3 further cadences post-scale ⇒ eikonal OFF (fail-safe). The old "clamp binds / rarely fires" sentence is DELETED — under τ_end=0.062 the adaptive path is MORE likely to fire late; **that is the law working, and run-1's eikonal ramp under these guards is the FIRST FAIR TEST of viscosity physics** (FEED-06h confound; REOPENED per negatives review item 6) | c (+d response) | D (window derived; 0.25/0.7 anchors DPR/V-A; ĉ_a forecast UNMEASURED at fine τ — the $0 clamp-binding check on cached fine-τ fields is queued §7b) |
| ★★★ byte-close SELECTION surface | **The §5.0 joint KKT law selects on the DECODED verdict** (fold item 10): argmin over (checkpoint × quantization depth × section options) of [100·d_seg_decoded + √(10·d_pose_decoded) + λ_bytes·bytes] — g_dec becomes a selection variable; F13 parity rows supply d_seg_decoded per candidate (chunked, ~5 min inflate each, resumable) | c | D (score-law) + R6 MEASURED machinery |
| ★★★ along=8 regime sentence | see §0.3 (req K; one sentence, lane_offloaded regime justification + the P1-fail flip condition) | — | honesty tag |

All other v3 rows (TAU→FIN restore law, joint λ_bytes law, hybrid orientation + R12, MUTCD comb,
LogitAdjust pins, ChromaBoundarySharpen, determinism) inherited unchanged.

---

## §2 — THE SCHEDULE (v4: TAIL_k replaces END; feedforward co-predicate; terminus-corrected chain-A)

### 2.1 Stage graph ★★★ (fold item 11 — the asymptotic tail added; END demoted to fail-safe cap)

```
P(prime, ep0) → CE → TAU → FIN(warm-Muon, entry = TAU-window EMA-best) → TAIL_1 → TAIL_2 → …
   FIN --regression-guard-trip--> RESTORE-BEST --DECIDE--> { TAU-continue | TAIL_1 }
   TAIL_k --per-cycle-meat-exhausted--> TAIL_{k+1}
   TAIL_k --STOP-RULE (PowerPlay)--> END
   ANY --fail-safe cap (req B)--> END        (cap_tail = 2× realized TAU stage length; injection-tested)
```

### 2.2 Event exits — v3 items inherited (per-class veto, per-epoch slope norm, B9 PREFERRED) + one v4 addition:

**★★★ Feedforward co-predicate form (fold item 9, req M(2)):** the TAU→FIN trigger gains the
forecast-consuming form — fire when the F3 online-meat AIC mixture FORECAST of remaining
TAU-stage meat (in S-units) < the transition forfeit it would recover (**5.4e-4 S**, the §2.2c
MEASURED forfeit at ep625-fire). The trailing-slope co-predicate stays as the FALLBACK (its
backtest is the only tested form; the forecast form ships shadow-mode first — logged would-fire
epochs vs the slope form — and is promoted only if its injection tests pass; req-B all-three
applies to BOTH forms). Rationale: B5's powerlaw-meat exit already uses the forecast form — the
TAU→FIN trigger just didn't consume it; this shrinks the §2.2c forfeit the same way the
event-adaptive cadence would, at zero verdict cost.

### 2.2e ★★★ THE TAIL LAW (fold item 11 — verbatim adoption of the negatives-review §3 law, with builds)

```
FIN --meat-exhausted--> TAIL_k (k = 1, 2, ...)          [replaces --> END as the default path]
TAIL_k: τ_k = max(τ_{k-1}/2, τ*_k),   τ*_k = m_q(k)/ln5 re-derived from the CURRENT witness
        margin field each cycle (the LAW, not the number, is the commitment — §2.2d discipline);
        LR_k ∝ τ_k (parabolic-scaling consistent); warm restart (never-reset moments; per-stage
        EMA-best entry per the §2.2c transition law); annealed-hosc β continues geometric
        (equal-epochs-per-octave); per-cycle meat exit (powerlaw_meat, BUILT) fires TAIL_{k+1}.
STOP RULE (PowerPlay-consistent, req D): exit when the last cycle's measured Δd_seg converts to
        < the attribution floor in S-units (0.014411 S until req-F #6 lands; the 0.00178 margin
        after) OR the costate duty-queue ranks another probe/arm higher per marginal-S-per-epoch.
END = the req-B fail-safe CAP only (a dead tail trigger degrades to the capped schedule, never
        unbounded): cap_tail = 2× the realized TAU stage length, injection-tested.
```

Honesty notes carried verbatim: each cycle's Δ is a BET until run (like warm-Muon); ~40 LOC —
the pieces (powerlaw_exit, per-stage ckpts, schedule_readback, geometric τ) all EXIST; req-B's
three tests (backtest where a log exists, injection, fail-safe cap) bind every tail trigger.
The tropical/Maslov regime (τ→0) is where the extremely-fine cycles live — the tail is the only
path in the stack that approaches the asymptote, and without it every "exhausted" verdict run-1
emits would be a new coarse-point negative for the next crucible to re-review. Build: **B14** +
DSL **`TailCycles`** Schedule object (§10).

### 2.3 SOLVE / chain-A — TERMINUS-CORRECTED (fold item 2; MAJOR-R2-2)

All four stale citations replaced. The corrected record:

1. TerminalSolve remains **OUT of run-1** — unchanged.
2. **The K-trend measurement (TERMINUS, commit 42fa00812):** ratio |λ₋|/λ_max = **0.163 at
   K=128** [MEASURED]. The strict kill band (<0.1) was **NOT formally reached** — middle zone;
   persist (>0.5) decisively excluded; full-P=600 extrapolation **≈0.08 [DERIVED]**. The
   recovery-written "0.011" did not survive fresh-eyes — exactly the L81 review-status class.
3. **Re-derived conclusion (what leaned on 0.011):** the terminal DISPOSITION does NOT rest on
   the strict-kill number — it rests on the CONJUNCTION [MEASURED]: 1/√K Ritz collapse (×5.2
   then ×1.97) + every solve step non-descending on independent holdouts (fp32 AND int8-deploy)
   + winner's-curse sign flip on disjoint holdout + u_min isotropy (lane-dilute + coarse-τ
   scoped). "ep650-EMA exhausted to both orders" SOFTENS to: exhausted **at the frozen schedule
   point, with residual 2nd-order structure in the kill-middle-zone (0.163) whose full-P
   extrapolation (≈0.08) is DERIVED not measured** — the sensor (K≥32 + K-trend), not this
   number, adjudicates any future basin. HOLD_STAGE_NEGATIVE_CURVATURE stays DISARMED.
4. **Negatives-review re-scope folded (§2.3(3) wording, items 1/11):** "the basis levers carry
   the REPRESENTATION burden; the τ/β-completion legs (§2.2d — unmeasured, instrumented) carry
   the SCHEDULE burden; optimizer/solve moves at THIS basin carry nothing (measured at the
   frozen point)." The F5 lane-bearing note gains "**AND finer-τ**" (annulus pixels gain Hessian
   weight as τ descends — anisotropy can EMERGE at 0.062 where invisible at 0.216). The ~35%
   HVP-vs-true-curvature magnitude is point-bound — re-measure per checkpoint ($0), never cite
   "35%" at a τ=0.062 checkpoint.
5. Registry (I-6, tranche 2) REWRITTEN: `gn_hessian_spectrum_indefinite_at_ema_best_v1` update
   registers **"K=128 ratio 0.163; kill-middle-zone; conjunction-dispositive; full-P
   extrapolation 0.08 DERIVED"** + the lane-blind + coarse-τ domain_of_validity caveats;
   `hessian_negative_curvature_subset_artifact_v1` carries both at birth. No falsified number
   enters `tac.canonical_equations`.

---

## §4 — COSTATE + TELEMETRY (v4: margin-critical ranking; F13; AA×island row; saturation alarm)

### 4a ★★★ THE MARGIN-CRITICAL TELEMETRY RANKING (fold item 7; req J(3) — ranked by S-units gated)

| rank | row/build | S-units it gates | × margin |
|---|---|---|---|
| **1** | **self-orient state PERSISTED in checkpoints + save-time recon-gap check row (req-F #6)** — promoted to NAMED LB-class build | +1.4411e-4 d_seg = **0.014411 S** attribution floor on EVERY probe/branch/byte-close (R6 MEASURED live→same-load-path) | **8.10×** |
| **2** | **F13 decode-gap parity row** ★★★ (fold item 10): per-checkpoint byte-close parity at stage boundaries — chunked REAL-path inflate (~4.7 min measured) + chunked n600 verdict, resumable JSONL state; emits g_dec(ckpt) per stage | g_dec measured **0.010427 S** at ep650; gates the crossing arithmetic + the decoded-surface KKT selection | **5.86×** |
| 3 | anneal-state row + completion flags (M2 class; F1) | the M2 defect class sat on ~0.001-class S misattribution | ~0.5-1× |
| 4 | ★★★ **adaptive-ε saturation ALARM** (fold item 6): ε_raw > 0.7 sustained ≥3 verdict cadences ⇒ π_eik>1 risk ⇒ feedforward ηλ_eik scale-down (§1.2 law); logs \|c_a\|(t), ε_raw, clamp-binding fraction | prevents a silent re-freeze of the adaptive branch (the vacuous-trigger disease at the clamp surface) | guard |
| 5 | ★★★ **AA×island-survival attribution row** (fold item 8; req I; the meat-§C3 gap): per-class paired with/without-AA verdict deltas at stage boundaries (ipe form) | attributes AA's island-birth-survival coupling per class | attribution |
| 6 | trigger would-fire audit (M3; F4) incl. the NEW feedforward co-predicate shadow rows (§2.2) | — | guard |
| 7–12 | remaining F-rows (online meat, spectrum sense w/ finer-τ note, lever engagement, single-lever attribution, config provenance, pose wall watch, stage wall-clock) | — | — |

**Viscosity fair-test expectation (fold item 12):** the eikonal ramp rows should be read as the
FIRST FAIR TEST of viscosity physics (every prior "eikonal failure" was the spike-guard
median-freeze confound, FEED-06h) — pre-registered expectation: eikonal term neither dominates
(>40% of loss ⇒ term_domination alarm) nor freezes; clamp-binding fraction <90% at fine τ or
the §1.2 window law fires.

### 4b Response-surface seeds — FOLDED into §4c (req M(3)/#170; the RS rows are the CAPACITY
block of the signal ledger below; each is a persisted costate-SENSE DECIDE model, not a
one-shot gate).

### 4c ★★★ SIGNAL-COMPLETENESS LEDGER (requirement P — operator pin: "generate and record and
expose ALL signal necessary for extreme optimization and realization to the theoretical limit")

The gap under decomposition: S_frontier 0.19110 − S_floor 0.11797 = **0.07313 S**. Every gap
term gets {signal → run-1 generates it? → recorded where (durable) → NAMED DECIDE consumer}.
Contract: **no write-only telemetry** (every row names its consumer); **a gap term with no
signal row = a seal-blocking design finding** — two were found in this pass and are FIXED
(F14, F15), not footnoted. This ledger SUBSUMES fold-item 9's scattered additions.

| gap term | signal | run-1 generates? | recorded where | named DECIDE consumer |
|---|---|---|---|---|
| d_seg per-class | per-class F-rows (lane/movable/big-3/hood d_seg shares + rel slopes) | YES (LB set) | run JSONL telemetry (levelset_train_result rows) | per-class λ gates + meat exits + amplify recalibration (RS-4); costate per-class veto (§2.2) |
| d_seg per-stage | stage-boundary verdicts + per-stage EMA ckpts + stage wall-clock (F12) | YES | per-stage ckpts (mandate) + F12 rows | TAIL_k stop rule; schedule MPC; campaign ILC error term (v5 schedule derivation) |
| d_seg annulus/margin | #333 margin-field rows + live m_q re-derivation + clamp-binding fraction | YES | annulus rows (default-ON, score-neutral) | τ*_k re-derivation law (TAIL_k, §2.2e); adaptive-ε window law (§1.2); τ-confirm |
| d_seg along/across spectrum | ★ **F14 (NEW — was a NO-SIGNAL gap term):** per-stage-boundary coefficient-energy spectrum per along/across frequency bin at the lane annulus (basis utilization) | YES once built (~20 LOC read-only reduce over existing coefficient tensors) | F14 rows per stage boundary | run-2 along-ladder / form-a retrain arm decision (negatives item 3/4); Rebalance regime check (§0.3 sentence) |
| d_pose witness-side | F11 pose-wall watch: per-verdict d_pose through the L3 mechanism + FiLM read-back checks — **currently UNMEASURED on any witness; the row itself is a required run-1 signal** | YES (pose ON two-track; w_pose>0) | F11 rows + verdict JSONL | pose kill law (1.5e-4) + q* selection (RS-1); §0.2 pose leg |
| rate per-section vs entropy floor | **F16:** byte-close per-section manifest {bytes, order-0 H0, coder-vs-H0 %} (the S4 machinery, recorded not ad-hoc) | YES at every byte-close | packet manifest JSON (durable, per F13 cadence) | joint decoded-KKT selection (§5.0); waterfill stop rule (RS-2); compress-half go/no-go (run-2) |
| decode gap g_dec | **F13** parity row per stage boundary (chunked inflate + chunked n600 decoded verdict) | YES | r*_parity JSONL (resumable) + packet dirs | decoded-KKT selection; §0.2 crossing arithmetic; RS-5 g_dec(ckpt, quant-depth) surface |
| schedule: anneal state | F1 β(t)/τ(t)/progress% + completion flags + M2 alarm | YES | F1 rows | finisher-fire precondition; B9 re-anchor; campaign ILC (v5 anneal shape) |
| schedule: meat/exits | F3 online-meat AIC forecast + trigger would-fire audit (F4) incl. feedforward-form shadow rows | YES | F3/F4 rows | TAU→FIN feedforward co-predicate (§2.2); exit caps; PowerPlay duty-queue ranking |
| schedule: TAIL yield | ★ **F15 (NEW — was a NO-SIGNAL gap term):** per-TAIL-cycle {τ_k, m_q(k), Δd_seg, epochs, Δd_seg/epoch in S-units} | YES (part of B14) | F15 rows | TAIL stop rule (attribution-floor comparison); req-N inflection curve (family-asymptote update per cycle); v5 tail design |
| capacity (response surfaces #170) | RS-1 pose (bytes,d_pose)(q) · RS-2 waterfill (bits,d_seg) marginal · RS-3 WeightEntropy (λ;bytes,d_seg) via twin+mid-λ · RS-4 per-class λ surface · RS-5 g_dec surface | YES (P9/P4/twin/F-rows/F13) | costate SENSE store (fitted models, persisted) | costate DECIDE layer (req M(3)); run-2 capacity choices; mod-dim 2-point |
| basis: orientation quality | **F17:** per-stage lane-annulus orientation alignment \|cos\| row (hybrid poly-tangent vs self-orient vs boundary normals) + the R12 oracle verdict | YES (~10 LOC read-only; R12 pre-GO) | F17 rows + R12 artifact | R12 gate + fallback law (§1.2 v3 row); run-2 orientation arm |
| attribution substrate | req-F #6 self-orient persistence + save-time recon-gap row (rank-1, 8.10× margin) + F13 (rank-2, 5.86×) | YES (LB builds) | ckpt fields + gap rows | EVERY consumer above — the attribution floor that gates whether any other signal's Δ is readable |

Completeness check against the gap: d_seg (class ∧ stage ∧ annulus ∧ spectrum) ✓ · d_pose
(witness-side, first-ever) ✓ · rate (per-section vs floor) ✓ · decode (g_dec per stage) ✓ ·
schedule (anneal ∧ meat ∧ tail) ✓ · capacity (5 surfaces) ✓ · basis (orientation ∧ along-
utilization) ✓. No remaining gap term lacks a generated-recorded-consumed signal row; F14/F15
are the two findings this ledger surfaced (both small read-only builds, added to §10).

---

## §5 — THE RATE PLAN (v4: R1 exact bytes; hood arithmetic fixed; decoded-surface selection)

### 5.0 Joint λ_bytes law — inherited, with two edits:

- ★★★ **Hood exemplar CORRECTED (fold item 5):** the clamp's 8 bytes cost 8 × 6.6586e-7 =
  **5.32688e-6 S** ⇒ it pays iff **Δd_seg > 5.3269e-8** (= 5.3269e-6 S = 0.2993% of the margin,
  same-currency). Operative gate unchanged (paired verdict).
- ★★★ **Selection runs on the DECODED verdict** (§1.2 row; g_dec a selection variable).

### 5.1 Byte budget (R1-folded: LBND2 = 41,526; win9 arm added; win5 quarantined)

Components: base+code post-waterfill 60,000 [52,000, 68,000] · grammar rev-2k −3,108 (measured)
· **band: LBND4 raw 30,892 central [MEASURED] / LBND4-win9 18,832 [MEASURED, roundtrip-exact —
byte-close-selectable under the decoded KKT law, GATED on the P8/F8 trained-with leg; win5
QUARANTINED (roundtrip FALSE all 3 schemes) until its identity defect is explained; independent
tail floor 18,832 replaces the 18,000 placeholder (was 832 B = 5.54e-4 S = 31% of margin
optimistic)]** · pose ξ 4,500 [2,700, 6,929] · manifest ~800 · hood clamp +8 · comb 0 B · AA
(ipe) 0 B.

| scenario | archive bytes | rate |
|---|---:|---:|
| central (LBND4 raw) | **93,092** | **0.061986** |
| ★★★ win9 arm (gated; central −12,060) | **81,032** | **0.053956** |
| component-consistent independent band | [70,400, 103,521] | [0.046877, 0.068930] |
| waterfill-fail, LBND4 holds | 115,285 | 0.076763 |
| worst joint tail — three legs (waterfill-fail ∧ B6-slip **LBND2 41,526** ∧ pose-upper 6,929) | **128,348** | **0.085462** |
| worst two-leg tail (pose central) | **125,919** | **0.083844** |

(Tail rows re-printed with the R1-measured LBND2 41,526 — −36 B vs v3, direction favorable.)

§5.2 receipts inherited unchanged.

---

## §7 — MEASUREMENT PLAN (v4 deltas) + §7b REOPENS QUEUE

§7 table inherited with: P11′ re-runs on the **ipe** config (expected trivially SAFE — aa_fine
term 0; the amended gate is the standing guard for any future ss>1 arm) · P8/F8 gains the
**win9 trained-with leg** (does the smoothed-geometry band NET lower decoded S? — gates the
win9 arm of §0.2) · F13 parity row runs at every stage boundary (rank-2 telemetry) · τ-confirm,
R12, R13 unchanged.

**§7b ★★★ REOPENS QUEUE (fold item 12 — queued, NOT run-1 blockers):**

| # | reopen | cost / kill band |
|---|---|---|
| Q1 | UniWARD per-class-pair per-DIRECTION signed re-test (asymmetry-suspect flagship; pooled ρ −0.033 may average away one-sided masking) | $0, cached S_R + texture fields, <5 min; kill \|ρ\|<0.1 BOTH sides every major pair ⇒ SCALE-ROBUST dead; \|ρ\|≥0.3 any side ⇒ signed-hinge UNIWARD cost term enters the duty queue with a real prior; re-run once at a fine-τ checkpoint |
| Q2 | F12 dash-contrast sampled at **τ ∈ {0.216, 0.12, 0.062}** | telemetry line (score-neutral, default-ON); separates τ-completion vs comb in any run-1 dash improvement (the τ-crossover negative was window-scoped — v4 trains 3.5× below its floor) |
| Q3 | **Viscosity REOPENED** — run-1 eikonal ramp = the first fair test (FEED-06h confound); plus the $0 clamp-binding check of the §1.2 window law on cached fine-τ margin fields | $0 pre-GO check + the §4 alarm in-run; one-sided (erasure-side-only) viscosity stays the named §9.4 refinement |

---

## §8 — WALL-CLOCK (v4: ipe base)

Base s/ep: **~107 (control-class) stands** — ipe adds ~0 (R3: basis-level, no fine-grid memory
or EDT cost); the ss=2-driven "1.5× gate is TIGHT" pressure is GONE with the ss=2 form. Event
path / cap path / twin / probe-wave numbers inherited (event exits worth 10–27%;
B1-contingency tag stands). TAIL_k cycles extend the run beyond the old END point —
epochs REDEPLOYED to finer stages per req L; the fail-safe cap bounds them (cap_tail = 2×
realized TAU length); wall-clock is lexicographically secondary (L59).

---

## §9 — PREDICTED S LADDER + RUN-2 (v4 fixes)

§9.1 ladder edits: the "byte-closed realized +0..+1e-4" rung is REPLACED by **g_dec measured
1.0427e-4 central [MEASURED ep650], prior band [0.5e-4, 2e-4] [ASSUMED — F13 measures per
stage]**; rate rung gains the win9 arm 0.053956 (gated). Central S ≈ 0.26 UNCHANGED (does not
cross — stated plainly; the crossing is the engineered tail per §0.2). §9.3 dual probability
model inherited; the win9 gate adds one more sequential-repair lever to Model B's rate axis
(P(rate ≤ 0.062) ≈ 0.85–0.95 unchanged; the win9 arm shifts rate mass toward 0.054).

§9.4 run-2 additions ★★★: **AA supersample (ss=2) moved here WITH ITS MEASURED COST WRITTEN**
(fold item 1 / R3's own kill law): aa_fine 34.36 GiB @ ndf=4/full ⇒ projected peak 105.90 GiB >
89.6 ceiling REFUSE; ndf=2/full 87.91 GiB with only 1.69 GiB margin (< the 10 GiB assumed-margin
⇒ margin-REFUSE until the reconcile ledger measures a smaller p95 spike); batch mode
memory-SAFE but ~29 s/ep EDT thrash (wall-clock-killed); AND the trainer Wave-D measured **−49%
witness-harm** on supersample + decode-budget-disqualified — ss=2 re-enters ONLY with a
hardware/margin change AND a positive paired verdict vs ipe. Everything else inherited
(branch protocol incl. cosine_hold + along=26 form-a retrain arm; NTK build-spec; per-class
τ_c — now per-class-pair-DIRECTION when it lands, per the asymmetry sweep; one-sided viscosity;
signed LengthSigma hinge).

---

## §10 — BUILD LIST (v4 deltas)

| id | build | ~LOC | status/route |
|---|---|---:|---|
| B1-B12, W1/W1b/W2, I-1..I-5, F-rows, T-1..3, BA | inherited | — | v3 §10 stands |
| ★★★ **B13** | per-class amplify weights (fold item 3): trainer `--amplify-weight-lane` / `--amplify-weight-movable` argparse rows + per-class mask split at the amplify block (reuses the existing per-class masks, L3780-3820 region) + third `--amplify-persist` kind `persistence_pairs` (w_i ∝ 1/pers_i clamped [0.25,4] from CacheGtSkeleton pairs, R13-gated) + DSL `AmplifyIsland.weight` float→float\|dict widening + compile branch + validate() + tests | ~60 | **LB for the per-class law**; fail-safe: pooled 1.0 ships if unlanded at GO (deferral WRITTEN, §0.3b) |
| ★★★ **B14** | TAIL_k warm-restart cycles (fold item 11): the §2.2e law — τ-halving w/ live τ*_k re-derivation, LR∝τ_k, warm restart via the §2.2c transition path, per-cycle powerlaw_meat exit, PowerPlay stop rule, fail-safe cap; + DSL **`TailCycles`** Schedule object (new factory; new gap-kind) | ~40 | pieces all BUILT; req-B three tests bind |
| ★★★ **B15** | verify/land DSL hold for `AACoverageRender(mode="ipe")` (if the factory lacks a mode param — config-orphan guard; trainer flag `--render-aa ipe` is R3-verified wired) | ~10 | rides B13's commit |
| ★★★ **F13** | decode-gap parity row (fold item 10): per-stage-boundary chunked byte-close + chunked n600 decoded verdict (R6 driver machinery), resumable JSONL; emits g_dec(ckpt); feeds RS-5 + the decoded-KKT selection | ~40 (drivers exist) | rank-2 margin-critical |
| ★★★ **F14** | along/across coefficient-energy spectrum row at the lane annulus, per stage boundary (req-P ledger finding — basis-utilization signal; read-only reduce over existing coefficient tensors) | ~20 | score-neutral, default-ON; consumer: run-2 along-ladder decision |
| ★★★ **F15** | per-TAIL-cycle yield row {τ_k, m_q(k), Δd_seg, epochs, ΔS/epoch} (req-P ledger finding) | in B14 | consumer: TAIL stop rule + req-N inflection curve |
| ★★★ **F16/F17** | per-section rate-vs-H0 manifest row (byte-close) · lane-annulus orientation-alignment \|cos\| row | ~15 | req-P ledger rows; consumers named §4c |
| ★★★ req-F #6 | self-orient checkpoint persistence + save-time recon-gap row — PROMOTED from pooled F1-F12 to a NAMED LB-class build (rank-1 margin-critical, 8.10× margin) | part of F-set | **LB (attribution)** |
| ★★★ I-6 | equation registrations — REWRITTEN per §2.3(5): terminus framing (0.163 kill-middle-zone; conjunction-dispositive; 0.08 extrapolation DERIVED) + lane-blind + coarse-τ caveats; **no falsified 0.011 enters the registry** | 0 | tranche 2 |

## §11 — REQUIREMENT A(ii) row 13 (terminus-corrected)

Row 13 (trunk/basis weights) now reads: **NOT-SOLVABLE at this basin — conjunction-dispositive**
(every solve step measured non-descending on independent holdouts fp32+int8; K-trend ratio 0.163
at K=128 in the kill-middle-zone with full-P extrapolation ≈0.08 DERIVED; u_min isotropy scoped
lane-dilute + coarse-τ); TRAINED — and the wall being representation/basis is why Arm A is the
vehicle. Rows 11/12 carry the same terminus numbers. Requirement A(i) status unchanged.

---

## SELF-ATTACK (operating-manual discipline: attack your own conclusion before shipping it)

1. **The crossing story now leans on win9 — is that a new single point of failure?** Partly.
   Without win9 the crossing needs every class within 1.4% of its optimistic edge — honest but
   thin. Mitigations are real: g_dec is a per-checkpoint SELECTION variable (F13 + decoded-KKT),
   and the g_dec=1.0427e-4 input is ONE checkpoint on the CONTROL config — a run with waterfill
   tuned on the decoded surface may realize less. But if win9's trained-with leg (P8/F8) shows
   the smoothed geometry RAISES d_seg by > 12,060·λ_bytes-equivalent, the win9 arm dies and the
   1.4% sliver is the whole story. I did not find a third offset lever of comparable size in the
   certified set — that is the design's honest thinnest point, and it is printed.
2. **Is the TAIL_k law smuggling unbounded compute into a "run-1" claim?** No — END survives as
   the fail-safe cap (2× TAU length, injection-tested) and the stop rule is denominated in the
   attribution floor. But the per-cycle Δ is a BET (labeled), and if the first tail cycle's
   measured Δd_seg < floor, the tail contributes nothing to run-1's crossing — the crossing
   arithmetic above does NOT book any TAIL_k gain (check: it doesn't; §0.2 uses only the §0.3
   design band).
3. **The adaptive-ε window law uses k_max² I never measured.** Correct — the upper edge is
   anchored empirically (0.7 safe / 1.0 explodes), not derived to a constant; I kept 0.7 and
   made only the FLOOR track the live edge. The saturation |c_a|≈93 number inherits the same
   anchor. Labeled DERIVED-with-DPR-anchors, and the $0 clamp-binding check (Q3) is queued
   before the number is load-bearing.
4. **Did I re-litigate anything round 2 certified?** Checked against seal §4: crossing-table
   FORM, share model, τ*-law, transition law, joint KKT, §5.1 sums — all preserved; my §5.1
   tail-row edits are the R1-measured −36 B and the win9 arm addition (new rows, not
   re-derivations); the §0.2 change is the g_dec TERM the fold list itself demands.
5. **Family-asymptote estimate (§0) could be wishful at its low edge (0.154).** It stacks
   optimistic edges; the central 0.165 / ~36%-of-gap claim is the number to cite. Labeled.

Pointer 0.19110 UNMOVED — this draft is MEANS until the §7 ROW lands.
