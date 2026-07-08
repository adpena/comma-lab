# T5 CRUCIBLE — NEGATIVE-FINDINGS SCALE-VALIDITY ADVERSARIAL RE-REVIEW (requirement L)

Agent: NEG-SCALE-REVIEW-T5 · 2026-07-07 · review pass over ALL load-bearing negatives consumed by
DRAFT_OPTIMAL_STACK_v3, under the operator lens (verbatim): *"Remember scaling and coarse to fine to
extremely fine to asymptote. Infinite compute time. Need to adversarial review all negative findings
in light of all of this."* + *"Remember separatrix asymmetry too."*

STORES CONSULTED: ORCHESTRATION_LEDGER (req A–L, esp. L incl. asymmetry addendum) ·
pursuit_chainA_spectrum_solve_20260707.md (full) · deepmath_meat_hunt_v2_20260707.md (§C/§C2/§C3)
· DRAFT_OPTIMAL_STACK_v3_20260707.md (§2.1–2.3, §3.2, §5.0–5.2, stage graph, exits) ·
CONTEXT_COMPENDIUM (STORE 1b probe wave + STORE 3 DAG sweep incl. FEED-06h retraction,
FEED-08c/g/k/l, Muon thread) · position_S2 (M-S2-1..5, cargo-cult row 4) · position_S1 (lane_carried
demotion L45/L100) · P3/P5 red-team verdicts (FEED-08l provenance rows) · corpus_query
("msal_uni UNIWARD sensor at chance S_R reachability") → memory
`msal_uni_texture_proxy_inert_build_exact_sR_reachability_weight_20260703` + sweep-B ledger +
equation `margin_saliency_reachability_replaces_texture_proxy_v1` · memory index L65/L70/L76/L78/L79
· CLAUDE.md capstone-trainer hosc anchor (FEED 2026-06-25a/FEED-ly). NOT consulted: raw run dirs
beyond what the memos cite (no training, no n600 spend — reading + arithmetic only).

Axis discipline: every number here is [macOS-CPU/MLX advisory] provenance unless marked; pointer
contest-CPU **0.19110 UNMOVED**; this review is MEANS. review_status: self-executed,
fresh-eyes-unreviewed (it is itself a review pass; P6 counter treats it as a finding-producing
round if any disposition below is adopted).

Currency (req J): crossing margin **0.00178 S**. 1e-5 d_seg = 1e-3 S = 56% of margin ·
1 KB = 6.82e-4 S = 38% · attribution floor (self-orient reconstruction gap +4.3%) ≈ 0.015 S ≈ 8.4×
margin until req-F #6 lands.

---

## §0 The lens, applied honestly (what L does and does NOT license)

L does NOT license wishful reopening: an instrument fact (1/√K statistics, winner's curse, subset
overfit, an operator property of frozen R) holds at every schedule point. L DOES bind: any
"exhausted/flat/NO-GO" measured at ONE point of the (τ, β, LR, epoch-budget, checkpoint) ladder is
a statement about THAT point. The control's ladder point for nearly every negative below:
**τ frozen at 0.216** (v3's derived τ_end = 0.062, 3.5× finer; tropical asymptote τ→0 unreached),
**β frozen 3.177–4.0**, **1000-ep budget** (PR95 reference: 29,650), **truncated anneal**
(M-S2-2), **cold Muon** (M-S2-1). The τ-parameterized objective L_τ is a DIFFERENT function at
each τ: stationarity/convexity/meat-exhaustion of L_0.216 at θ_ep650 does not bound descent on
L_0.062 — that reopening is not a hope, it is the definition of annealing (each PR95 stage exhausts
before the next reopens descent).

Separatrix-asymmetry sub-lens: the boundary is one-sided per class-pair (Road→Lane FP ≠ Lane→Road
erasure; winner-vs-runner-up logit perturbations act asymmetrically; per-side flip masses and
UNIWARD costs differ). Any negative built from a SYMMETRIC estimator (pooled unsigned correlation,
symmetric second-difference, |margin| quantiles, σ_ij = σ_ji) may have averaged away a one-sided
effect. Checked per item below.

---

## §1 Per-finding verdicts

### 1. Chain-A terminal verdict (ep650 "exhausted BOTH orders" · TerminalSolve NO-GO · u_min isotropic)

**Split verdict — the chain itself already draws the line correctly; v3 needs one wording fix.**

**SCALE-ROBUST (the instrument facts — keep verbatim, they bind ALL future spectrum work):**
- K≤8 Ritz rows are NOISE for DECIDE (1/√K collapse ×5.2 then ×1.97 — pure statistics).
- Winner's-curse protocol (best-of-N on a 16-pair int8 surface flips sign on disjoint holdout).
- Extreme Ritz MAGNITUDES at small K are fp32-path-fragile (CPU −370 vs GPU −175).
- Measured-loss acceptance only / quadratic-model acceptance FORBIDDEN. *Refinement under L:* the
  qualitative fact (analytic HVP ≠ true curvature through un-disableable uint8-STE) is robust and
  STRENGTHENS at finer τ (the surface gets more piecewise-flat, rounding-jump curvature grows);
  the **~35% magnitude is point-bound** — do not cite "35%" at a τ=0.062 checkpoint, re-measure
  ($0, one pair, the landed harness).

**SCALE-BOUND (the conclusions — valid exactly where the chain says, nowhere else):**
- "ep650-EMA exhausted to BOTH orders **at the frozen schedule point**" — the chain's own phrase.
  Exhaustion of L_{τ=0.216} at θ_ep650 does NOT bound descent on L_{τ=0.062}: τ enters the loss;
  the finer objective's gradient at the same θ is a NEW measurement. v3 §2.3(2) carries the
  qualifier — correct. The full-P extrapolation (ratio ≈0.08, DERIVED) likewise binds only this
  (θ, τ) pair.
- "TerminalSolve NO-GO" — NO-GO **from this checkpoint on this objective**. v3 correctly removes
  it from run-1 and retains the run-2 spec at a NEW basin; under L, sharpen the revisit condition:
  the solve question re-poses at **every stage boundary of the finer ladder** (each anneal stage's
  exit basin is a new L_τ), which v3's per-stage GNSpectrumProbe already instruments. No change
  needed beyond noting the sensor IS the re-test (cost $0-in-run; reopen iff K≥32 K-STABLE
  coherent λ₋ or large PD Newton decrement at any lane-bearing, finer-τ stage boundary).
- "u_min isotropic ⇒ no basis shortcut in the Hessian" — DOUBLY bound: (a) lane-dilute checkpoint
  (P5-5 pin, v3 §2.3(6) carries it — lane ≈ 0.58% of an area-weighted loss); (b) **τ-bound, the
  new part**: at finer τ the loss concentrates on the annulus (the m<τ·ln5 population grows
  relatively as the smooth interior saturates), so annulus-pixel directions gain Hessian weight —
  anisotropy can EMERGE at τ=0.062 where it was invisible at 0.216. The F5 lane-bearing note
  should say "lane-bearing AND finer-τ".

**SEPARATRIX-ASYMMETRY check: PASS with one note.** The dispositive step measurements probed BOTH
signs individually (±u_min line searches, −ĝ at 4 η's) — not a pooled symmetric estimator; the
negative stands per-side. The symmetric second-difference [L(+s)+L(−s)−2L(θ)]/s² IS a two-sided
average and its observed sign instability (−1.2 at s=0.01 → +1.3 at s=0.02) is consistent with an
odd (cubic/one-sided) term at small s — but since the direct ± steps each failed, nothing
actionable was averaged away. Keep the note as an instrument caveat: curvature-transfer tests on
this vehicle should always report the two one-sided differences, not only their sum.

**v3 citation re-scope (the one fix, §2.3(3)):** "Arm A + the analytic lane render-band + basis
levers carry **the entire burden**" overstates — v3's OWN §2.2d books unmeasured τ-completion and
β-completion legs as REAL Δ candidates. Chain-A proves no OPTIMIZER/SOLVE move crosses the gap at
the frozen point; it says nothing against SCHEDULE moves (finer τ/β are objective changes, not
optimizer moves). Reword: "the basis levers carry the representation burden; the τ/β-completion
legs (§2.2d, unmeasured, instrumented) carry the schedule burden; optimizer/solve moves at this
basin carry NOTHING (measured)." This is exactly finding-11 below; one sentence.

### 2. S2 M-S2-4 "meat exhausted ep600–650"

**SCALE-BOUND — and constructively, it IS the refinement-transition trigger, not a terminator.**
The exponential fit (asymptote a=0.003377, τ_e=79 ep, remaining meat 5.5e-6 ≈ 0.55e-3 S ≈ 31% of
margin at +300ep) is a fit to the τ-STAGE trajectory of the CONTROL's truncated anneal — it
measures exhaustion of THAT stage's objective at τ→0.216. It generalizes to exactly nothing about
L_{τ=0.062} or any finer stage. Under L this negative is the SENSOR the ladder needs: per-stage
meat-exit = "this stage's objective is mined out → fire the NEXT, finer stage." v3 uses it
correctly as a stage exit (TAU→FIN, forfeit table §2.2c) but ALSO wires
`per-class-meat-exhausted (all classes) → END` — that second use is the tail defect (§3 below).
Re-test: none needed — powerlaw_meat re-fits per stage on run-1 by construction. Asymmetry: n/a
(scalar trajectory fit).

### 3. FEED-08l freq_along ladder FLAT

**SCALE-SUSPECT — the review's clearest case, already 80% flagged by P3/P5; confirm + add the arm.**
Every limitation compounds toward blindness at fine scale: (a) ORACLE-form substitution at the
FROZEN ep650 checkpoint — an along-bandwidth increase can only pay through RETRAINED coefficients,
and the along-tangent content it would carry (dash structure) is precisely the fine-τ population
(dashes live in the m<0.10 annulus; L65: unrecoverable below crossover **at any capacity** — but
that law was itself measured on the coarse-schedule family); (b) only 2 scoreable rungs (0, 8),
rungs ≥16 invalidated by the GT-control instrument floor — the interesting end of the ladder was
never scored; (c) recovery-written post-credit-death, never fresh-eyes-reviewed. The exact
finer-scale condition that could reopen it: **form-a retrain at along∈{8, 26} under the v3 anneal
to τ_end=0.062, comb OFF** (so along-bandwidth gets its unconfounded shot at the dash population
the finer τ exposes). Re-test: (i) the already-owed P2 fresh-eyes review ($0, reading); (ii) a
§9.4 tau-boundary BRANCH arm (form-a retrain, along=26) — fractional fine-tune cost, kill band:
along=26 fails to beat along=8 by more than the attribution floor (until req-F #6 lands that floor
is 0.015 S; after it, the honest bar is the 0.00178 margin itself). Note the ordering guard
already makes along=8 primary regardless — this re-test gates run-2 regime choice, not run-1.
Asymmetry: n/a (bandwidth allocation, unsigned).

### 4. lane_carried regime demotion (rests on 08l)

**SCALE-SUSPECT by inheritance — already PROVISIONAL (#363 tag, ledger L284); keep, re-scope the
citation.** v3 correctly keeps lane_carried as fallback with the FAIL⇒revert-OPEN path (P2 row).
Under L add: the demotion may not be cited in run-2 planning as a measured regime kill — it is a
frozen-point oracle-form reading. The §9.4 branch arm in item 3 IS its re-test (same arm, same
kill band). No separate cost.

### 5. #207 pre-emphasis / R-deconvolution "measured DEAD, R all-pass"

**SCALE-ROBUST — the rare negative that survives the lens completely.** |H_R| (1.0→0.842 at
render-Nyquist) is a property of the FROZEN operator R (bicubic↑ → uint8 → bilinear↓), not of any
checkpoint, τ, or schedule — it is measured on the operator, and the operator never anneals. The
max linear-deconvolution gain is bounded by the operator itself (~1/0.842 ≈ +1.5 dB) INDEPENDENT
of how sharp the witness gets; the Wiener ceiling (+1.25 dB) used a signal spectrum from the
measured checkpoint, and a sharper (finer-τ) witness shifts spectral mass toward HF — but the
gain bound is signal-independent, so the DEAD verdict on R-deconvolution/pre-emphasis-of-R holds
at the asymptote. Optional $0 confirm (minutes, NOT required): recompute the Wiener ceiling with
the run-1 sharp-witness spectrum and verify it stays under +1.5 dB. The surviving sibling (L2
sub-pixel phase lever, #149 class) is where fine-scale content actually routes — v3's DEFER-with-
build-spec + AA ss=2 partial cover is the right disposition. Asymmetry: n/a (linear-response
measurement). One caveat for the record: the uint8 nonlinearity inside R is NOT covered by |H_R|
(a linear reading); at very fine τ, quantization-interaction levers (dither/phase) remain the open
sibling — which is again the #149 class, already named.

### 6. Viscosity NO-GO (eikonal-viscosity arms)

**INVALID-MEASUREMENT / REOPENED — not scale-bound, not scale-robust: never fairly tested.**
FEED-06h retraction (DAG L8330) is binding: every ep103–114 "eikonal failure" (v5/v6 + 3 #205
runs) was the legacy spike-guard median-freeze deadlock, not eikonal physics. Reconciliation with
req-L is clean and adds one NEW item: even the eventual fair test must be schedule-aware, because
the viscosity ε-window is τ-coupled — the backward-heat region (|∇m|<1) and the two-sided CFL
window both move as the margin field sharpens under the anneal. v3's shipped adaptive-ε law
(ε = clamp(|c_a|·√(η·λ_eik/8)·(1+m), 0.3, 0.7)) was calibrated at coarse τ and v3 itself notes
the clamp BINDS (adaptive rarely fires — shipped "for the FORM"). **Finding: the clamp floor/upper
(0.3/0.7) are coarse-point constants wearing a law's clothing** — at τ_end=0.062 the derived ε*
may sit outside [0.3, 0.7], silently re-freezing the adaptive branch (the same default-off orphan
shape, req-B's vacuous-trigger disease at the clamp surface). Re-test: $0 — evaluate the ε formula
against the CFL edge-tracker on cached fine-τ margin fields (or first run-1 TAU-late checkpoint)
and check clamp-binding fraction; if it binds >90% at fine τ, re-derive the clamps as functions of
τ (law class (c)). Run-1's eikonal ramp under live spike/liveness guards IS the fair first test of
the physics. Asymmetry note (design consequence, req-L addendum): viscosity smooths BOTH sides of
the separatrix equally; a one-sided variant (smooth only the erasure-side sublevel set) is un-tried
and is the PDE-facet twin of the signed-hinge idea — name it in §9.4 as a fractal refinement, no
run-1 change.

### 7. LEVER-4 msal_uni texture proxy INERT + UniWARD sensor at chance (#268)

**SCALE-SUSPECT — specifically ASYMMETRY-SUSPECT; the flagship case for the operator's second pin.**
The measurement (memory L76 / `msal_uni_texture_proxy_inert_...20260703`): Pearson −0.033±0.022
between the texture multiplier 1/(1+β·tex) and through-R reachability S_R; top-5% Jaccard 0.024 ≈
chance. Both quantities are UNSIGNED, and the correlation is POOLED over all class-pairs and both
flip directions. Under the asymmetry addendum this is exactly the estimator shape that can average
a one-sided effect to zero: UNIWARD cost is a texture-masking argument — masking plausibly
predicts FP-side reachability (can we PUSH a new class in over textured background?) differently
from erasure-side (can we LOSE a thin structure?); per-side correlations of ±0.2–0.3 pool to ~0
if the sides carry opposite signs with comparable mass (lane: 36.5% of GT lane px flip = erasure-
dominated; Road-side FPs are the other population). Two independent conclusions:
- The DEPLOYED disposition is UNCHANGED and correct: S_R (exact, θ-indep, signed by construction
  at the flip level) replaces the proxy; a chance-level POOLED sensor must not fire (S5's reason).
- The KILL SCOPE must shrink: "texture ⊥ through-R reachability" is proven only for the pooled
  unsigned statistic. Re-test ($0, cached fields, <5 min arithmetic): recompute Pearson/Jaccard
  **per class-pair per flip DIRECTION** (Road→Lane vs Lane→Road at minimum) from the existing S_R
  + texture caches. Kill band: |ρ| < 0.1 on BOTH sides of every major pair ⇒ upgrade to
  SCALE-ROBUST dead; any side with |ρ| ≥ 0.3 ⇒ a one-sided UNIWARD cost term (signed hinge,
  req-L design consequence) enters the never-fired queue with a real prior. Same re-scope applies
  to any other pooled at-chance sensor verdict (sweep: none other found in v3's citations).
- Schedule axis (secondary): reachability S_R was computed at the coarse checkpoint; texture-
  masking's headroom grows as the witness sharpens (more HF content to hide in) — tag the per-side
  re-test to re-run once at a fine-τ checkpoint ($0 again).

### 8. "Muon didn't improve past ep650" re-attribution (cold + truncated)

**SCALE-BOUND; re-attribution COMPLETE under L, with ONE residual caveat to pin.** The story is
already the L-shaped one: M-S2-1..5 re-attributed the finisher failure to schedule (cold Muon
quench + truncated anneal + 76–125 past-exhaustion epochs), chain-A then SHARPENED it (no 2nd-order
structure at ep726 for ANY optimizer to mine — the failure was never about Muon), and #270 (L79)
fixed the restart semantics (resume PRE-switch + warm-start momentum + lr-final-frac 0.1). The
−32% keep-anchor plus v3's warm-Muon-on-completed-anneal (stated as a BET with the +5.4e-4 S
forfeit printed) is the correct L-compliant posture. The caveat: **M-S2-5's Muon asymptote fit
(0.003236, INFERRED) is a truncated-anneal-point artifact** — it must never be cited as a bound on
warm-Muon at completed anneal (different objective, different entry state). S2 labels it
advisory/INFERRED; keep that label load-bearing. No re-test beyond run-1 itself (the FIN stage is
the measurement). Asymmetry: n/a.

### 9. hosc fixed-β divergence

**SCALE-ROBUST — and it is req-L's own witness.** The negative (fixed β=4 from scratch: tanh(β·sin)
saturation → vanishing grad → AdamW random-walk → d_seg RISES; CLAUDE.md capstone anchor,
FEED 2026-06-25a/FEED-ly) is a COLD-START dynamics fact: initializing AT the fine point without
the ladder fails. It does not say "high β is bad" — annealed-hosc β 1.0→4.0 is the measured
survivor, i.e. the finding is the existence proof that the coarse→fine ladder is MANDATORY, the
same shape as PR95's stage ladder. One scope note so it is never misread: the negative does NOT
bound β→∞-effective at the END of a completed anneal (that region is unmeasured and is where the
tropical math lives); S2's geometric-β build (equal-epochs-per-octave, τ=ε=ħ) is the right
extension and needs no re-test before run-1. Asymmetry: n/a.

### 10. K=8 subset-solve gap +5.1% (#341)

**SCALE-ROBUST (the artifact mechanism) with one scale-scoped sub-claim.** Subset-solve overfit is
statistics (solving 8 pairs' quadratic overfits their idiosyncratic Hessian components — the same
per-pair σ≈2.0 vs g_true≈0.08 decomposition chain-A measured); it holds at every schedule point
and the derived rule ("only full-P in-trainer GPU solve admissible") is schedule-independent.
Scale-scoped sub-claim: the LM ρ 0.847/0.868 QUADRATIC-CHART confirmation is a property of the
ep650 basin — chart validity at any new/finer basin is a new measurement (v3's per-basin LM ρ row
covers this automatically). The +4.3% reconstruction-gap instrument fact is checkpoint-format-
bound, not schedule-bound — fixed by persisting self-orient (req-F #6), after which the
attribution floor drops from ~0.015 S toward the margin itself. Asymmetry: n/a.

### 11. "The 0.0034→0.00092 gap cannot be crossed by optimizer moves at this basin"

**SCALE-BOUND, CORRECT AS STATED — v3's citation needs the one-sentence re-scope from item 1.**
The claim's own words ("optimizer moves", "at this basin") are exactly right. v3 §2.3(3)'s "carry
the ENTIRE burden" is the only drift: it erases v3's own §2.2d schedule legs (τ-completion to
0.062, β-completion — both unmeasured, both instrumented, both potentially harvesting part of the
gap WITHOUT basis change). Arm A carries the burden EITHER WAY for run-1 (basis-match is prior to
capacity, −48% anchor unchanged) — but the extremely-fine tail stages may still harvest optimizer/
schedule gains post-basis-change, and the draft should not pre-commit the attribution. Fix: the
§2.3(3) rewording in item 1. Denominated: the gap is 2.5e-3 d_seg ≈ 0.25 S ≈ 140× the margin —
even a 5% schedule-leg harvest (~0.0125 S) is 7× the margin, far too large to write off by wording.

### 12. Sweep — other negatives v3 consumes as load-bearing (§-by-§)

| negative (v3 cite) | verdict | note |
|---|---|---|
| dash-comb CORRECTOR-AS-COMPOSITE NO-GO at frozen ep650 (§3.2 comb P1-cond.) | SCALE-BOUND, correctly consumed | v3 already routes comb IN-TRAINING; the frozen-point composite negative is the reason, properly scoped |
| τ-crossover trainflow "no crossover in τ∈[0.216,0.806]; dash contrast τ-insensitive" | SCALE-BOUND — window-scoped, and v3 EXITS the window | v3's τ_end=0.062 trains 3.5× below the measured window's floor; "no reachable τ-anneal buys dash resolution" was true of the CONTROL's reach only. **Attribution consequence:** if dashes improve in run-1, credit must split τ-completion vs comb — the per-class lane F-row + comb engage-epoch stamp (MINOR-13 ramp) are the instruments; add one line to F12: dash-contrast sampled at τ∈{0.216, 0.12, 0.062} so the τ-leg is separable |
| #280 Lever-D post-hoc corrector NO-GO | SCALE-BOUND | same class as dash-comb composite: frozen-checkpoint post-hoc form; in-training forms unaffected |
| AnalyticLaneRenderBand post-hoc NEUTRAL (+0.000012 gated) | SCALE-BOUND (form-bound) | correctly consumed — v3 trains WITH band; the post-hoc neutral says nothing about trained-with at finer τ |
| contour-string flip coding NO-GO (0.820 B/flip, "fragmented confetti") | SCALE-SUSPECT-lite | measured on ep650 residual; run-1's residual flip set (post band+comb+finer τ) is a different population — possibly MORE codeable (structured leftovers) or less. Re-test is FREE: re-run the coder on run-1's residual at byte-close (it is an offline pass); keep OUT of run-1 planning |
| orbit-coding permutation slack NO (−8 B best; most arms hurt) | SCALE-ROBUST-cheap-auto | byte-domain, this checkpoint's stream; re-runs automatically at run-1 byte-close for free; rev-2k −3,108 B stands |
| PR95-L25 temporal-delta on code +64% | SCALE-ROBUST | stream-statistics fact (ξ innovation white, code stream non-temporal); schedule-independent |
| se3-spline DIRECT ξ replacement DEAD (innovation white) | SCALE-ROBUST | property of the ξ trajectory statistics, not of any schedule; spline-as-predictor residual (~1 KB headroom) correctly kept |
| paint-seed starvation (init −36%, plateau 0.027) | SCALE-BOUND | capacity-pivot evidence at its config; v3 consumes only the seed-choice conclusion (GT-paint-then-SDF dominates) — fine |
| weight-perm canonicalization −8 B vs 387 B estimate | SCALE-ROBUST (this stream) | measured contradiction of the group-theory estimate; re-checks free at byte-close |
| FEED-06r amplify UNIFORM net-negative (0.121/0.026) | SCALE-BOUND | v3 already replaces with margin-GATED + per-class w (0.28 movable); the uniform-form negative doesn't bound gated forms — correctly consumed |

**Asymmetry sweep of v3's SYMMETRIC design constants (design consequences, not negatives):**
τ_end = m_q/ln5 uses the UNSIGNED |margin| quantile pooled over both sides — per-side m_q (per
class-pair direction) would give different τ*; the per-class τ_c build (meat §D-3, run-2) should
be per-class-pair-DIRECTION when it lands. LengthSigma σ_ij is symmetric in (i,j) — the junction
fit measured pair costs, not directed costs; a signed one-sided hinge variant is the §9.4 named
refinement (with the one-sided viscosity twin from item 6). Menon logit-adjust is already signed
per-class — no change.

---

## §3 The constructive half — ASYMPTOTIC-TAIL VERDICT on v3: **FINDING (missing tail)**

v3 §2.1: `P → CE → TAU → FIN → END`, with `TAU/FIN --per-class-meat-exhausted (all)--> END` and
FIN budget cap_fin = clamp(1.5·τ̂_e, 150, 350). **v3 TERMINATES where requirement L demands
refinement.** Under infinite compute, per-class meat exhaustion at stage k's objective is the
REFINEMENT-TRANSITION trigger (item 2), not a run terminator; PR95's existence proof is 8 stages
of exactly this. v3 has no extremely-fine stage family, no re-anneal/cyclic finishing, and its
finest point (τ_end=0.062) is reached once and then the graph ends.

**The tail-stage law to add (one §2.1 edge + one law row; run-1-compatible, costate-governed):**

```
FIN --meat-exhausted--> TAIL_k (k = 1, 2, ...)   [replaces --> END as the default path]
TAIL_k: τ_k = max(τ_{k-1}/2, τ*_k),  τ*_k = m_q(k)/ln5 re-derived from the CURRENT witness
        margin field each cycle (the law, not the number, is the commitment — §2.2d discipline);
        LR_k ∝ τ_k (parabolic-scaling consistent); warm restart (never-reset moments, per-stage
        EMA-best entry per the §2.2c transition law); optional engaged sharpening (annealed-hosc
        β continues geometric, equal-epochs-per-octave); per-cycle meat exit (powerlaw_meat,
        already built) fires TAIL_{k+1}.
STOP RULE (PowerPlay-consistent, req D): exit the tail when the last cycle's measured Δd_seg
        converts to < the attribution floor in S-units (0.015 S until req-F #6; the 0.00178
        margin after) OR the costate duty-queue ranks another probe/arm higher per marginal-S-
        per-epoch. END stays as the fail-safe CAP (req B: a dead tail trigger degrades to the
        capped schedule, never unbounded) — cap_tail = 2× the TAU stage length, injection-tested.
```

Two honesty notes: (i) the tail is a BET like warm-Muon — each cycle's Δ is unmeasured until run;
the law costs ~40 LOC (the pieces — powerlaw_exit, per-stage ckpts, schedule_readback, geometric
τ — all exist) and req-B's three tests; (ii) the tropical/Maslov regime (τ→0) is where the
extremely-fine cycles live — the asymptote is the argmax itself, and the tail is the only path in
the current draft that ever approaches it. Without the tail, every "exhausted" verdict run-1 emits
will be a NEW coarse-point negative for the next crucible to re-review — the tail law is the
structural fix for the entire class this review exists to catch.

---

## §4 Summary table

| # | finding | verdict | reopening condition (finer-scale) | re-test (cost / kill band) | asym check |
|---|---|---|---|---|---|
| 1a | chain-A instrument facts (K≤8 noise, winner's curse, fp32-fragile magnitudes, measured-acceptance-only) | **SCALE-ROBUST** | — (35% HVP magnitude: re-measure per checkpoint, $0) | — | PASS (± probed individually); report one-sided diffs henceforth |
| 1b | "ep650 exhausted BOTH orders" | **SCALE-BOUND** (frozen point; self-scoped) | L_{τ=0.062} is a new objective | GNSpectrumProbe at run-1 stage boundaries ($0-in-run; reopen iff K-stable λ₋) | — |
| 1c | TerminalSolve NO-GO | **SCALE-BOUND** | new basin at any finer stage boundary | same sensor; run-2 spec retained | — |
| 1d | u_min isotropic | **SCALE-BOUND** (lane-dilute AND coarse-τ) | lane-bearing + finer-τ checkpoint | F5 note gains "finer-τ" | — |
| 2 | M-S2-4 meat exhausted ep600–650 | **SCALE-BOUND** = the refinement trigger | by design: next stage | none (per-stage refit built) | — |
| 3 | FEED-08l freq_along FLAT | **SCALE-SUSPECT** | form-a retrain under τ_end=0.062, comb OFF | P2 review ($0) + §9.4 branch arm along=26 (fractional fine-tune; kill: < attribution floor over along=8) | n/a |
| 4 | lane_carried demotion | **SCALE-SUSPECT** (inherits 3) | same arm | same (no extra cost) | n/a |
| 5 | #207 pre-emphasis/deconv DEAD | **SCALE-ROBUST** (operator property) | none (gain bounded by \|H_R\| ≥ 0.842) | optional $0 Wiener recompute at sharp spectrum | n/a |
| 6 | viscosity NO-GO | **INVALID/REOPENED** (confound; never fairly tested) | run-1 eikonal ramp = first fair test | $0 clamp-binding check of adaptive-ε at fine-τ margin fields; re-derive clamps as τ-law if binding >90% | one-sided viscosity = named §9.4 refinement |
| 7 | msal_uni / UniWARD at chance | **SCALE-SUSPECT (asymmetry-suspect)** | per-side signal pooled to zero? | $0 per-class-pair per-DIRECTION ρ from cached fields (<5 min; kill: \|ρ\|<0.1 both sides ⇒ robust-dead; \|ρ\|≥0.3 any side ⇒ signed-hinge UNIWARD enters queue) | **THE flagship case** |
| 8 | Muon past-ep650 | **SCALE-BOUND**; re-attribution COMPLETE | run-1 FIN is the test | pin: M-S2-5 asymptote fit must stay advisory | n/a |
| 9 | hosc fixed-β diverges | **SCALE-ROBUST** (cold-start fact; req-L's own witness) | does NOT bound β at end-of-anneal | none | n/a |
| 10 | #341 subset gap +5.1% | **SCALE-ROBUST** (artifact mechanism) | LM chart per new basin (auto row) | none | n/a |
| 11 | gap uncrossable by optimizer moves | **SCALE-BOUND**, correct as stated | schedule legs are not optimizer moves | v3 §2.3(3) one-sentence re-scope | — |
| 12 | sweep (11 items) | 5 ROBUST · 5 BOUND (correctly consumed) · 1 SUSPECT-lite (contour-string; free auto re-test) | see rows | free/auto | τ_end + σ_ij symmetric-constant notes |
| — | v3 asymptotic tail | **FINDING — MISSING** | — | add TAIL_k law (~40 LOC + req-B tests; END demoted to fail-safe cap) | per-side τ* in tail cycles (run-2) |

**Counts:** SCALE-ROBUST **9** (1a, 5, 9, 10 + sweep×5) · SCALE-BOUND **11** (1b, 1c, 1d, 2, 8,
11 + sweep×5) · SCALE-SUSPECT **4** (3, 4, 7, contour-string-lite) · INVALID/REOPENED **1**
(viscosity). v3 text changes required: §2.3(3) re-scope (items 1/11) · F5 "finer-τ" note (1d) ·
F12 dash-contrast τ-sampling line (12) · adaptive-ε clamp τ-law check (6) · **§2.1 TAIL_k edge
(the tail law — the review's principal constructive finding)**. New $0 probes queued: msal_uni
per-side ρ (7) · adaptive-ε clamp-binding (6) · optional Wiener recompute (5). No paradigm
retired by any suspect finding stands retired (req-L binding clause honored: suspects may gate
run-1 arms, not retire paradigms).

Pointer 0.19110 UNMOVED — this review is MEANS; it changes what run-1's schedule DOES after its
exits fire and what run-2 may cite.
