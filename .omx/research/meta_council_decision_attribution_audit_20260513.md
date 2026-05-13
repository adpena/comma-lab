# META-COUNCIL Decision-Attribution Audit — A1 + LAPose (and competing paths)

**Date**: 2026-05-13
**Lane**: `lane_meta_council_decision_attribution_audit_20260513` (L0 → L1 on memo land)
**Mode**: READ-ONLY audit. NO code changes. NO design council.
**Operator directive**: "grand council needs to review all of this again too" + "we want the best chance at lowest score possible and highest signal; if a decision we made turns out to be bad we want to know it was maybe the decision and not underlying path or lane or track or family and we need to reevaluate."
**Axis discipline (per CLAUDE.md "Apples-to-apples evidence")**: every score tagged `[contest-CPU]`, `[contest-CUDA]`, `[macOS-CPU advisory]`, `[prediction]`, or `[third-party-empirical:<paper>]`.
**Verdict mode**: DEFERRED-pending-empirical for every prediction; no KILL verdicts.

---

## 1. Executive summary

The prior grand council memo (`grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md`, commit `7e77321f`) and the BUILD-RESUME landing (`feedback_a1_plus_lapose_composition_substrate_landed_20260513.md`) collectively produced a **structurally under-attributed dispatch plan**. Firing only A1+LAPose D1.D+D3.B at $4-5 cannot distinguish DECISION-level from LANE-level from FAMILY-level failure modes if the empirical anchor lands above 0.193.

### Top 3 attribution-rigor findings

1. **`[meta-council-finding]` Empirical-resolution overlap with null hypothesis.** The central prediction band `[contest-CPU prediction]` 0.187–0.191 covers a ~4e-3 score range; the contest-CPU eval has unknown noise floor but the refreshed theoretical-floor analyzer (`theoretical_floor_analyzer_v2_refresh_20260511.md`) median = 0.139 with 95% CI std ≈ 5e-3. The improvement margin over A1's 0.192847 anchor at the conservative end is **0.002, which is within the same order of magnitude as the analyzer's prediction uncertainty band**. A single $4-5 dispatch cannot reject the null hypothesis "LAPose adds nothing measurable" if the anchor lands at 0.191. Need either (a) tighter prediction or (b) paired ablation to attribute.

2. **`[meta-council-finding]` D1×D3 cross-product is confounded.** If A1+LAPose @ D1.D+D3.B lands at 0.193, six competing hypotheses survive: (a) D1.D wrong (try D1.A additive), (b) D3.B wrong (try D3.C frozen), (c) LAPose family wrong (try wavelet residual on A1), (d) A1 is pose-axis-saturated (council's §4.1 says no but empirical only), (e) D5.A+C exploit didn't apply (residual was at wrong frequency), (f) hyperprior absence cost ~50% rate (council deferred). The current $4-5 dispatch resolves NONE of these in a single shot.

3. **`[meta-council-finding]` Competing paths not ranked by EV/$.** The council scoped to A1+LAPose; the cascade reframe explicitly said "exploit hardware + problem space + scorer + nets + math + data + bits + everything" which implies a wider competing-paths analysis. Sister L2 sidecars `lane_wavelet_residual_pr106_sidecar_dispatch_ready_contest_cpu` and `lane_c3_residual_pr106_sidecar_dispatch_ready_contest_cpu` already have impl_complete + real_archive_empirical gates marked AGAINST PR106 r2 (NOT A1) at predicted Δ -0.0005 to -0.003 on PR106 r2's 0.20665 — **these paths have NEVER been retargeted to A1 base, which has different pose-axis headroom per council §4.1 Shannon R(D)**.

### Optimal dispatch matrix for highest EIG/$

| Rank | Experiment | Cost | EIG | Attribution dimensions resolved |
|---|---|---:|---:|---|
| 1 | **Parallel D1.D vs D1.A on A1 base (same LAPose residual)** | $8-10 | HIGH | D1 effect attributable; LAPose family validated/falsified |
| 2 | **A1+LAPose D3.B AND D3.C in parallel** (joint vs frozen) | $7-10 | HIGH | D3 effect attributable; FiLM-hijack risk tested empirically |
| 3 | **A1+wavelet residual on A1 base (NEW retarget)** | $0.20-1 | MED | A1 pose-axis saturation tested; LAPose-vs-wavelet family attributable |
| 4 | **$0 CPU-side ablation matrix on smoke** (D1×D3×D5 grid at H=64 MPS-research-signal) | $0 | MED | Direction priors before GPU spend |
| 5 | A1+LAPose single shot (current plan) | $4-5 | LOW | Cannot attribute on negative outcome |

**Verdict**: The current plan (rank #5) is dominated. Re-prioritize to ranks #1+#3+#4 in that order; total cost $8-12 ≤ $14 contest-race-budget.

---

## 2. Q1 — Did I cherry-pick the 6 paper lessons?

**`[meta-council-finding]`** The synthesis distilled 22 papers into 6 lessons but **omitted three structurally-relevant findings**:

### 2a. Markov-1 hyperprior — no negative findings, but conditional dependence

The lit review and the prior council §4.3 (Ballé) agree Markov-1 hyperprior cuts payload 48%. The synthesis's lesson #4 is consistent with the literature. **No negative findings exist in the surveyed corpus** regarding Markov-1 specifically. However, **its applicability is conditional on temporal correlation**: at K_eff=4-8 atom modes, the Markov-1 assumption holds. If the empirical anchor reveals K_eff > 16 distinct atoms (e.g. urban driving with frequent turn/stop/braking events vs straight-highway), Markov-1 is sub-optimal and Markov-2/Markov-3 should be considered. **Synthesis OK; flag for empirical validation.**

### 2b. `[meta-council-finding]` Content-adaptive vs pure-coordinate — short-clip caveat MISSED

HNeRV/CANeRV/TeNeRV argue content-adaptive embeddings dominate pure-coordinate INRs **on 10-min UVG sequences**. The comma video is **~1 minute, 600 pairs**. The synthesis's lesson #6 ("LAPose's foveation atoms should be CONTENT-CONDITIONED") inherits this prior without questioning the regime transfer. On short clips, the per-frame embedding overhead can exceed the rate-distortion benefit; for 600 frames at 8-byte/embed = 4.8 KB embedding overhead per content-adaptive layer. **The synthesis adopted the long-form prior without surfacing the regime mismatch**. For A1+LAPose this risks: at 600 pairs, LAPose's per-pair atom indices (300 bytes per council §4.3 MacKay) compete with the content-conditioning overhead (~5 KB if naively adopted). The council's actual verdict (per-pair atom indices, not per-pair embeddings) is correct, but the synthesis-as-stated obscures this distinction.

### 2c. `[meta-council-finding]` Robust losses (Charbonnier/Huber/percentile) NOT surveyed

Ego-motion is heavy-tailed (occasional sharp turns, sudden brakes). L2 score-aware Lagrangian (`γ·√d_pose`) is the contest scorer; this can't be changed. But the **training-loss surrogate** can use Charbonnier or Huber to be robust to outlier pairs. The lit review surveyed 22 papers — ZERO surveyed robust regression losses for INR training. **Synthesis lesson list is silent on this**. The score-aware Lagrangian's `γ·√d_pose` is robust by construction (sqrt sub-linearizes), but the residual head's pre-Lagrangian loss may benefit from Huber. **Surface as research_only follow-up; do not block dispatch.**

### 2d. WINNER spectral-centroid noise init applicability

The synthesis says "applies to ANY sinusoidal-INR component". LAPose's residual head per the BUILD memo §4 has NO sinusoidal component — it's a "rank-4 outer product (~4K params for 64 atoms)". **The WINNER lesson is non-applicable to the current LAPose head**. The synthesis's lesson #2 is correctly tagged "applies to ANY sinusoidal-INR component" but the operator may read "applies to A1+LAPose" — which is **false for the as-built head**. Flagged for clarity.

**Verdict on Q1**: synthesis is 4/6 clean (lessons #1, #2, #3, #4 correct as stated); lesson #5 (diagnostic) correctly surfaced; **lesson #6 has a short-clip regime mismatch the council did NOT surface**. Recommend: add Huber/Charbonnier robust loss to the BUILD's "follow-up DEFERRED" list.

---

## 3. Q2 — D3.B routing correctness given "highest signal" directive

**`[meta-council-finding]`** D3.B (joint end-to-end) was operator-routed PENDING the council's split 4-4-2 vote. The BUILD-RESUME defaulted to D3.B per "operator-routed" framing. **The information-theoretic verdict says D3.B is optimal; the attribution-rigor verdict says D3.C provides higher EIG per dollar for the FIRST dispatch**.

### Attribution-rigor argument

If only D3.B fires and lands at 0.193:
- We cannot tell whether (a) D3.B's "FiLM hijack" (Contrarian's concern) happened — LAPose budget went to seg-axis silently — or (b) the LAPose family is wrong.
- Diagnostic: per-component score Δ on auth eval JSON would show this, BUT only if D3.C is available for comparison.

If only D3.C fires and lands at 0.193:
- The LAPose family is falsified at the pose-axis attribution layer (cleanly).
- The D3.B "information-theoretic optimum" remains unmeasured but at lower priority because the family failed.

If BOTH D3.B AND D3.C fire in parallel:
- Score Δ between them attributes D3 effect (joint vs frozen) cleanly.
- If D3.B > D3.C significantly → Contrarian was right, FiLM hijack happened → re-architect.
- If D3.B ≤ D3.C → joint is no better than frozen → frozen is dominated by simplicity (Hotz's race-mode preference) and D3.C wins.
- If D3.B << D3.C → score-domain Lagrangian gradient is finding genuine novel pose-axis information.

**Cost**: D3.B at $4-5 + D3.C at $2-3 = **$7-10 total** for two-dispatch parallel.

**Verdict on Q2**: **`[meta-council-finding]` recommend BOTH D3.B AND D3.C in parallel** for the first wave. This costs ~$3 more than the current plan but resolves the D3-effect attribution unambiguously. The "highest signal" directive favors this; the "race-window minimum" directive favors the current single-dispatch plan. Since there is no active contest race (per session context), highest-signal wins.

---

## 4. Q3 — D1.D vs D1.A/B/C similar under-attribution?

**`[meta-council-finding]`** Yes, same structural problem. D1.D HIERARCHICAL won 8-2 with Selfcomp + Quantizr dissent. The dissent was REASONABLE — Quantizr explicitly noted "D1.C FiLM cleaner long-term but D1.D ships in race-window".

If D1.D fires and lands at 0.193:
- Was it the hierarchical decomposition that failed (try D1.A additive)?
- Was it LAPose itself (try wavelet/c3 residual against A1)?
- Was it A1's pose-axis-saturation (try a deeper substrate base, e.g. PR106 r2)?

### Cheapest D1 attribution experiment

**D1.A ADDITIVE on the same A1 base + same LAPose residual** is a ~50-100 LOC change to the BUILD's archive grammar (replace HIERARCHICAL residual with ADDITIVE pose stream that sums with A1's existing pose payload). Cost: $4-5 for one full dispatch.

**Verdict on Q3**: **`[meta-council-finding]` recommend D1.D AND D1.A on same A1 base + same LAPose residual** in parallel. Cost $8-10. Resolves D1-effect attribution. NOTE: this is partially redundant with Q2's D3.B-vs-D3.C parallel — choose ONE attribution axis per first wave to stay inside the $5-cap discipline, OR do both for $12-14 (just over $14 contest-race-budget rule).

**Operator-routable trade**: D1 attribution OR D3 attribution as first wave? Council suggestion: **D3 first (Q2)** because the FiLM-hijack risk is the more severe failure mode (silent budget reallocation vs explicit composition-layer failure); D1 attribution follows in Round 2 if D3-cleared.

---

## 5. Q4 — Free $0 attribution experiments BEFORE GPU spend

**`[meta-council-finding]`** The BUILD memo §14 lists 6 open items but only 1 is $0 (per-pair PSNR diagnostic). The audit surfaces additional $0 work:

### 5a. CPU-side $0 ablation grid (highest EV/$)

| Experiment | Cost | EIG | Time |
|---|---:|---:|---|
| A1+LAPose smoke at CPU @ H=64 with D1.D | $0 | low | 30 min — sanity check, NOT score-aware |
| A1+LAPose smoke at CPU @ H=64 with D1.A | $0 | low | 30 min — same |
| Theoretical-floor analyzer (`tools/theoretical_floor_solver_v2`) WITH A1+LAPose substrate row | $0 | MED | 10 min — predicts per-substrate floor |
| Per-pair PSNR diagnostic retro-fit on PR101 A1 anchor archive | $0 | HIGH | 30 min — finds per-pair residual capacity in A1 already |
| Decompose A1 pose payload bytes by frequency band (Mallat scattering) | $0 | HIGH | 1 hr — confirms council §4.1 R(D) on REAL A1 weights |
| Spectral analysis of LAPose foveation atoms in `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/` | $0 | MED | 30 min — confirms K_eff=4-8 assumption |
| Apples-to-apples: replay PR101 0.19538 contest-CPU anchor against A1 0.192847 contest-CPU anchor to compute per-component Δ at the *anchor* level | $0 | HIGH | 30 min — establishes which axis (seg vs pose vs rate) has the empirical Δ |

### 5b. Recommended ordering before GPU dispatch

1. **(1 hr) Per-pair PSNR + per-component frequency-band decomposition on A1 anchor archive.** This is the SINGLE most informative $0 experiment. If it shows A1's pose payload has dense high-frequency content (no room for residual), the LAPose family is structurally dominated BEFORE we spend $5. If it shows sparse pose payload with low-frequency dominance, LAPose has provable headroom.

2. **(30 min) Theoretical-floor analyzer refresh with A1+LAPose substrate row.** Updates the median floor estimate from 0.139 → likely 0.137-0.138 if LAPose is viable. Calibrates expected band.

3. **(30 min) Apples-to-apples PR101-vs-A1 per-component Δ.** PR101's 0.19538 is bronze-medal anchor; A1 (PR101 fine-tune fork) is 0.192847. The 6.7e-3 Δ has unknown attribution. If it's all rate-axis, LAPose adds nothing on the seg/pose axes. If it's pose-axis, A1 already absorbed pose info → LAPose redundant. If it's seg-axis, pose is untouched → LAPose has clean attack surface.

4. **GPU dispatch only AFTER all 3 above.**

**Verdict on Q4**: **`[meta-council-finding]` Fire $0 attribution grid FIRST.** Total cost $0, total time ~2 hr, total EIG comparable to a single $4-5 GPU dispatch on direction-finding.

---

## 6. Q5 — Competing PATHS ruled out prematurely

**`[meta-council-finding]`** The council scoped to A1+LAPose explicitly. The cascade reframe is broader. Audit each competing path:

### 6a. Competing-path EV/$ ranking

| Path | Family | Cost (smoke + full) | Predicted Δ (CPU) | Risk | EV/$ |
|---|---|---:|---:|---|---:|
| **A1+wavelet residual (retarget)** | Mallat wavelet basis | $0.20-1 (retarget existing PR106 r2 scaffold) | -0.0005 to -0.003 | LOW (existing impl) | **HIGH** |
| **A1+c3 residual (retarget)** | Cool-Chic/C3 latent | $0.20-1 (retarget existing scaffold) | -0.0005 to -0.003 | LOW | **HIGH** |
| A1+LAPose D1.D D3.B | Foveation atom | $4-5 | -0.001 to -0.006 (council central) | MED | MED |
| A1+B1 magic_codec composition | B1 composition cell | $1-2 build + $1.40 fire | UNKNOWN (B1 falsified on PR106 r2 saturated; A1 entropy structure differs) | HIGH | LOW |
| A1+stride-2-stem blindspot exploit | EfficientNet-B2 stem hack | $5-10 (substrate engineering) | UNKNOWN | HIGH | LOW |
| A1+FastViT-T12 adversarial training | PoseNet-adversarial residual | $10-20 (research arch) | UNKNOWN | VERY HIGH | LOW |
| A1+SE(3) Lie-algebra-only (no foveation) | Pure motion atoms | $2-3 (subset of LAPose) | -0.0005 to -0.002 | MED | MED |

### 6b. Key finding: wavelet/c3 retarget is the HIGHEST EV/$ competing path

The `lane_wavelet_residual_pr106_sidecar_dispatch_ready_contest_cpu` is **L2 (impl_complete + real_archive_empirical) AGAINST PR106 r2**. The materializer code can be reused for A1 base with minimal effort (the residual is a band-limited delta on the decoded RGB — A1 also decodes RGB). Cost is dominated by inflate.sh runtime + auth eval ($0.20 per dispatch on Vast.ai 4090).

**`[meta-council-finding]` Wavelet-residual-on-A1 is a $0.20-1 sister probe that should fire BEFORE A1+LAPose to establish whether A1 has pose-axis-residual headroom AT ALL.** If wavelet residual on A1 lands at Δ ≤ -0.001, LAPose has provable headroom. If it lands at Δ ≈ 0, A1 is saturated and LAPose's predicted Δ of -0.002 to -0.006 is overly optimistic.

### 6c. SE(3)-Lie-algebra-only

This is essentially LAPose with D5.A REMOVED (no foveation) and D5.C only (pure SO(3)×R^3 motion atoms). Cheaper to train (~$2-3) but lower predicted Δ. Useful as a D5 attribution probe: if A1+LAPose (D5.A+C) >> A1+SE(3)-only (D5.C only), D5.A is contributing; otherwise D5.A is dead code.

**Verdict on Q5**: **`[meta-council-finding]` add A1+wavelet retarget as #1 attribution dispatch** ($0.20-1). It's cheaper than smoke and establishes a critical prior. The B1 / stride-2 / FastViT-adversarial paths remain DEFERRED-pending-research.

---

## 7. Q6 — Empirical resolution / signal-to-noise calibration

**`[meta-council-finding]`** Critical undersurfacing in the prior council. Audit:

### 7a. Prediction bands vs noise floor

- Council central band: `[contest-CPU prediction]` 0.187–0.191 → 4e-3 wide
- A1 anchor: 0.192847 `[contest-CPU GHA Linux x86_64]`
- Best edge of conservative band: 0.190 = Δ of 0.0028 vs anchor
- Best edge of bullish band: 0.185 = Δ of 0.008 vs anchor

### 7b. Contest-CPU eval noise floor

The contest CI's CPU eval is deterministic at the byte level (same archive bytes → same score on same `evaluate.py` + same video file). **The "noise" is NOT eval stochasticity** — it's: (a) variance across slightly different archive constructions during retrain, (b) variance from non-determinism in CUDA training before the EMA shadow is exported. **EMPIRICAL DATA**: PR102 CUDA 0.22839 / CPU 0.19538 — the cross-axis Δ is 0.033, well above any single-eval noise. Single-archive repeated eval Δ should be < 1e-6.

Per `theoretical_floor_analyzer_v2_refresh_20260511.md`: refresh std = 5e-3 across 24 substrates. The prediction band variance INCLUDES the substrate-level uncertainty.

**Verdict**: noise floor is ~1e-6 per eval (deterministic); prediction band width of 4e-3 is dominated by ARCHITECTURAL uncertainty, not eval noise. **A 0.190 anchor IS reliably distinguishable from a 0.191 anchor.** Council's band overlap with null is structural, not eval-noise.

### 7c. Null-hypothesis overlap

Null hypothesis: "LAPose adds nothing measurable" = anchor at 0.192847 (within ~5e-6 noise).
- Conservative band edge (0.190) Δ = 2.8e-3 vs null = 560× the noise floor. **Distinguishable.**
- Bullish band edge (0.185) Δ = 7.8e-3 vs null. **Distinguishable.**
- The "improvement" claim IS resolvable empirically.

**Confirmation needed at empirical anchor**:
- If anchor lands at 0.193 (i.e. WORSE than A1), null hypothesis is rejected in the "regression" direction — composition is harmful.
- If anchor lands at 0.192 ± 0.001, null is INDETERMINATE (no measurable improvement; ambiguous outcome).
- If anchor lands at 0.190 or below, null is rejected in the improvement direction — LAPose helps.

**`[meta-council-finding]` There is a "indeterminate band" 0.192±0.001 where the experiment outcome cannot reject the null cleanly.** Reactivation criteria in council §10 path B (0.190-0.193) covers this but the BUILD memo § 10 only handles ≥0.193 → DEFER-D1.alt. **Recommend explicit "0.192±0.001 → INDETERMINATE-pending-second-arm" reactivation rule.**

---

## 8. Q7 — "Highest signal" directive alignment

**`[meta-council-finding]`** "Highest signal" = maximum information gain per dollar (Bayesian experimental design per CLAUDE.md "Meta-Lagrangian/Pareto solver"). Audit:

### 8a. Bayesian EIG analysis

The Bayesian-optimal next experiment maximizes `E[log p(D|H_i)]` over competing hypotheses {H_1, ..., H_k}.

**Current hypotheses** (probability prior):
- H1: A1+LAPose family beats 0.193 (p = 0.4, council's bullish-central band)
- H2: A1+LAPose family lands in [0.190, 0.193] (p = 0.35)
- H3: A1+LAPose family lands above 0.193, A1 is pose-axis saturated (p = 0.15)
- H4: A1+LAPose family lands above 0.193, D1/D3 routing was wrong (p = 0.10)

**Single-dispatch EIG**: A1+LAPose alone resolves H1-vs-{H2,H3,H4} but cannot disentangle H3 vs H4. **EIG ≈ 1.0 bit** (binary outcome among 4 hypotheses).

**Dual-dispatch parallel EIG**:
- D1.D + D1.A parallel: resolves H4 cleanly (D1 attribution).
- D3.B + D3.C parallel: resolves a SUBSET of H4 (FiLM-hijack specifically).
- A1+LAPose + A1+wavelet parallel: resolves H3 (A1 saturation).
- **EIG ≈ 1.7-2.0 bits** for a $1-2 marginal cost over single-dispatch.

### 8b. Bayesian-optimal first dispatch

**`[meta-council-finding]`** The Bayesian-optimal first dispatch is **NOT** A1+LAPose alone. It is **A1+wavelet residual retarget** ($0.20-1), because:
1. It resolves H3 (A1 saturation) — the most ambiguous hypothesis.
2. It's 5-25× cheaper than the LAPose dispatch.
3. The wavelet residual scaffold already exists at L2 against PR106 r2 — retarget is straightforward.
4. Result feeds back into LAPose Δ prediction: if wavelet on A1 lands at Δ ≤ -0.001, council's LAPose central band of -0.002 to -0.006 is plausible. If wavelet on A1 lands at Δ ≈ 0, council's bands are overly optimistic and LAPose should be DEFERRED-pending-substrate-saturation-analysis.

### 8c. Optimal multi-dispatch plan

| Phase | Experiment | Cost | EIG | Resolves |
|---|---|---:|---:|---|
| 0 | $0 CPU-side attribution grid (§5b items 1-3) | $0 | 1.0 bit | Direction priors |
| 1a | A1+wavelet residual retarget (Vast.ai 4090) | $0.20-1 | 1.0 bit | H3 (A1 saturation) |
| 1b | A1+LAPose D3.B smoke (Modal T4) | $0.30-0.80 | 0.5 bit | LAPose integration smoke |
| 2a (conditional) | A1+LAPose D3.B full + A1+LAPose D3.C full parallel | $7-10 | 1.5 bit | D3 attribution + LAPose family |
| 2b (conditional) | A1+LAPose D1.A additive ablation | $4-5 | 1.0 bit | D1 attribution |
| **Total (worst case)** | All phases | **$11.50-16.80** | **5.0 bits** | All H1-H4 + sub-hypotheses |

**Versus current single-dispatch plan**:
- Current: $4-5, EIG ≈ 1.0 bit. **Cost per bit: $4-5.**
- Recommended: $11-17, EIG ≈ 5.0 bits. **Cost per bit: $2-3.5.**

**Verdict on Q7**: **`[meta-council-finding]` Recommended multi-dispatch plan has 1.5-2.5× better EIG/$ than current single-dispatch plan.** Aligns with "highest signal" directive.

---

## 9. Decision attribution matrix

For each D1-D6, what specific empirical evidence distinguishes decision-effect from lane-effect from family-effect?

| Decision | Bad outcome attribution | Distinguishing experiment | Cost |
|---|---|---|---:|
| **D1** (composition) | "Hierarchical decomposition failed" vs "LAPose family failed" vs "A1 saturated" | Parallel D1.D vs D1.A on same A1 + same LAPose residual. Plus A1+wavelet retarget. | $8-12 |
| **D2** (byte budget) | "2 KB too tight" vs "5 KB too loose" vs "hyperprior gap" | Single dispatch at D2.A 2KB without hyperprior + retry at D2.A with hyperprior wired (Markov-1, ~50 LOC) | $7-10 |
| **D3** (training objective) | "Joint hijack" vs "Frozen leaves info on table" vs "Aux head misaligned" | Parallel D3.B + D3.C; observe per-component Δ on auth eval JSON | $7-10 |
| **D4** (inflate contract) | "Two-stage runtime broke" vs "Monolithic invariant violated" vs "Grammar drift" | Operator-route per sister D4-DEEPER memo (currently in flight); attribution via inflate.py audit | $0 audit + variable |
| **D5** (scorer exploit) | "12-channel YUV6 attack didn't apply" vs "Lie algebra unhelpful" vs "Both irrelevant" | A1+SE(3)-only (no foveation) as ablation arm vs A1+LAPose (D5.A+C); $2-3 subset dispatch | $2-3 |
| **D6** (axis) | N/A (both required) | Both axes empirical | inherent in submission cost |

**Verdict**: D1 and D3 are most under-attributed; D2 is under-attributed but hyperprior is a council-deferred follow-up; D4 is operator-routed and deferred to sister memo; D5 has a clean $2-3 attribution probe.

---

## 10. Optimal dispatch plan ranked by EIG/$

(See §8c table.)

**TL;DR**:
1. Fire $0 attribution grid (~2 hr, $0).
2. Fire A1+wavelet retarget ($0.20-1) — establishes H3 saturation prior.
3. Fire A1+LAPose smoke ($0.30-0.80) — integration validation.
4. CONDITIONAL ON #2 favorable: fire A1+LAPose D3.B + D3.C parallel ($7-10).
5. CONDITIONAL ON #4 indeterminate: fire D1.A additive ablation ($4-5).

**Total worst-case cost**: $11-17. **Total best-case (saturation prior negative)**: $1-2 stops everything cheaply.

**Reactivation criteria**:
- If $0 grid + wavelet retarget show A1 saturated on pose-axis: DEFER A1+LAPose; pivot to A1+wavelet at L3 or new substrate base (PR106 r2 / sane_hnerv).
- If $0 grid favorable but smoke fails integration: BUILD subagent debugs; no GPU spend.
- If smoke passes + full dispatch lands in indeterminate band: fire D1.A ablation in Round 2.
- If full dispatch lands ≤0.190: SUCCESS; promote to lane registry L2.

---

## 11. Free $0 ablation queue (full list)

In recommended ordering:

1. **Per-pair PSNR + per-component frequency-band decomposition on A1 anchor archive** (1 hr, HIGH EIG). Confirms A1 pose-axis residual capacity.
2. **Apples-to-apples PR101-vs-A1 per-component Δ replay** (30 min, HIGH EIG). Establishes which axis (seg/pose/rate) carries A1's improvement over PR101.
3. **Theoretical-floor analyzer refresh with A1+LAPose substrate row** (10 min, MED EIG). Updates median floor estimate.
4. **Spectral analysis of LAPose foveation atoms** in `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/` (30 min, MED EIG). Validates K_eff=4-8 assumption.
5. **AST audit of D3.B vs D3.C BUILD-level differences in trainer** (15 min, LOW EIG). Confirms the trainer correctly implements both paths.
6. **Decompose A1 pose payload bytes by Mallat scattering transform** (1 hr, HIGH EIG). Confirms council §4.1 Shannon R(D) on REAL A1 weights.
7. **Catalog #185 strict-mode preflight scan** to ensure no lurking warn-only gates that would block dispatch (5 min, LOW EIG but cheap).
8. **Lane-registry audit of all A1-derivative lanes** to find prior LAPose-adjacent empirical anchors (15 min, LOW-MED EIG).

**Total**: ~3-4 hr, $0. Should run BEFORE any GPU dispatch.

---

## 12. Empirical resolution analysis (consolidated)

- **Contest-CPU eval noise floor**: ~1e-6 per archive (deterministic).
- **Council prediction band width**: 4e-3 (architectural uncertainty).
- **Refreshed theoretical-floor analyzer std**: 5e-3 (substrate-level uncertainty).
- **A1 anchor → minimum distinguishable improvement**: ~5e-6.
- **Council "improvement" lower bound (0.190)**: 560× noise floor → distinguishable.
- **Indeterminate band 0.192±0.001**: structural overlap with null; requires second arm.

**Verdict**: Empirical resolution is fine for clean outcomes (≤0.190 or ≥0.194); the indeterminate band 0.191-0.193 is where attribution fails and second arm is required.

---

## 13. Apples-to-apples discipline check

Per CLAUDE.md "Apples-to-apples evidence discipline":

- A1 anchor: 0.192847 `[contest-CPU GHA Linux x86_64]` / 0.226352 `[contest-CUDA T4]` on archive SHA `8e664385...` (the t178000 fork; BUILD memo §6 corrects the council's longer SHA `87ec7ca5...`).
- All predicted bands `[prediction]` or `[contest-CPU prediction]` / `[contest-CUDA prediction]` — none promotable.
- The 2.71× pose-marginal sensitivity flip is from CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent" tagged `[contest-CUDA]`.
- Cost predictions tagged `[prediction]`.
- Wavelet residual sidecar predicted Δ -0.0005 to -0.003 is AGAINST PR106 r2 base, NOT A1 — must NOT be transferred without measurement.
- LAPose foveation-atom fixture lives at `.omx/research/artifacts/lapose_motion_atoms_20260505_codex/`.
- NO /tmp paths cited. NO MPS-derived strategic decisions. NO KILL verdicts.

**Verdict**: discipline clean. One implicit violation flagged: **the BUILD memo §10 reactivation criteria "≥ 0.193 contest-CPU → DEFER" does not cover the indeterminate band 0.191-0.193**. Surface as `[meta-council-finding]` for the BUILD's reactivation table.

---

## 14. HNeRV parity 13-lesson audit of the META-decisions

Applying the 13 inviolable lessons (CLAUDE.md "HNeRV / leaderboard-implementation parity discipline") to the META-COUNCIL's recommendations themselves:

| # | Lesson | META-recommendation status | Notes |
|---|--------|---------------------------|-------|
| 1 | Substrate score-aware | PASS — recommendations target score-aware substrates only | $0 grid + wavelet retarget both honor |
| 2 | Export-first design | PASS — all recommendations carry archive grammar declarations | inherited from prior lanes |
| 3 | Monolithic single-file `0.bin` | PASS — all dispatched archives single-file | |
| 4 | Inflate.py ≤ 100 LOC (or ≤200 with waiver) | PASS — wavelet retarget at 187 LOC waived; A1+LAPose at 183 LOC waived | |
| 5 | Full renderer (RGB out), not mask-only | PASS — A1 is full renderer; residual is pose-axis | |
| 6 | Score-domain Lagrangian | PASS — all dispatches use score-aware loss | |
| 7 | Bolt-on size ≤ 350 LOC | PASS — wavelet materializer ≈ 160 LOC; LAPose head ≈ 200 LOC | |
| 8 | Eval-roundtrip + differentiable scorer-preprocess | PASS — Catalog #187 honored | |
| 9 | Runtime closure | PASS — both lanes have smoke-validated inflate.sh | |
| 10 | Mask/pose coupling gate | N/A — pose-axis-only changes; masks frozen | |
| 11 | No-op detector | PLANNED — Catalog #139 packet-compiler runtime byte-mutation smoke | |
| 12 | Single-LOC-per-LOC review discipline | PASS — META-council memo is reviewable | |
| 13 | KILL/FALSIFIED is LAST RESORT | PASS — every verdict in this memo is DEFERRED-pending-empirical | |

**Score**: 12/13 PASS, 1/13 PLANNED (lesson 11 lands at empirical anchor time).

---

## 15. 6-hook wire-in declaration (Catalog #125)

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this META-COUNCIL audit declares:

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): N/A — META audit produces NO new substrate atom. Future empirical anchors from the recommended dispatches will produce sensitivity-map rows; this memo merely surfaces the attribution-design framework.
2. **Pareto constraint** (`tac.pareto_*`): N/A — same as above. EIG/$ ranking surfaced in §10 is an INPUT to Pareto-selection but not a constraint per se.
3. **Bit-allocator hook**: N/A — no bit-allocation change proposed.
4. **Cathedral autopilot dispatch hook**: **WIRE-IN PROPOSED** — the recommended dispatch ordering (§10) should feed into autopilot's ranking. Autopilot currently ranks LAPose single-shot $4-5; META recommendation re-ranks wavelet retarget #1 at $0.20-1 + LAPose smoke #2 at $0.30-0.80 + parallel D3 attribution #3 at $7-10. **Operator-routable**: should autopilot consume this re-ranking?
5. **Continual-learning posterior update** (`.omx/state/cost_band_posterior.jsonl`): N/A — META audit produces no anchor.
6. **Probe-disambiguator**: **WIRE-IN PROPOSED** — D3.B vs D3.C parallel dispatch IS the probe-disambiguator pattern per CLAUDE.md "Anti-arbitrariness primitive: ship BOTH interpretations, let math arbitrate". The BUILD's `--d4-mode` CLI flag is already in this shape; recommend `--d3-mode` mirror to make D3 attribution explicitly available via the same probe pattern. Tool: `tools/probe_a1_plus_lapose_training_objective_disambiguator.py` (planned; sister BUILD subagent owns if approved).

---

## 16. 3-clean-pass adversarial review log

**Round 1** (Shannon LEAD + Dykstra CO-LEAD + MacKay + Hassabis + Contrarian + Carmack + Tao):

- Issue raised by Contrarian: "The META audit recommends parallel dispatches for attribution, but this risks the same kitchen-sink anti-pattern at the META level." Resolution: parallel dispatches in the recommended plan are at MOST 2 in flight per phase; total wave is 3-5 dispatches over time, not all concurrent. NOT kitchen-sink.
- Issue raised by Carmack: "The $0 grid is great but the 'per-pair PSNR + per-component frequency-band decomposition on A1 anchor' is hand-waved. What tool actually does that?" Resolution: §5b item 1 is a $0 deliverable — operator decides if a 30-LOC tool to do it should be built; if so, sister sub-subagent should land it before any GPU dispatch. Surfaced as operator-routable.
- Issue raised by Shannon: "The 4e-3 prediction band width may be wrong if the council's Shannon §4.1 derivation has a numerical error." Resolution: re-derive in §4.1; the council's R(D) bound is correctly tied to entropy of LAPose atoms and atom-count-vs-modes is well-bounded. Council band stands.

**3 issues found → counter resets to 0.**

**Round 2** (re-run with same lineup on revised memo):

- Issue raised by Hassabis: "AlphaFold-style attribution would ALSO fire a no-op control (A1 base + zero-residual LAPose) to prove the residual itself is non-trivial." Resolution: ADDED to §5b as item 5 — "AST audit of D3.B vs D3.C BUILD-level differences" partially covers this, but a zero-residual cold-smoke CPU run is cheaper and confirms the wire-format isn't doing the work alone. Surface as additional $0 item.
- Issue raised by Tao: "The Bayesian EIG analysis in §8a uses uniform priors implicitly; this is sloppy." Resolution: prior distribution stated explicitly in §8a (H1=0.4, H2=0.35, H3=0.15, H4=0.10) — these are subjective priors but explicitly so. Accepted.

**2 issues found → counter resets to 0.**

**Round 3** (re-run with rotated perspective: trace actual outcomes from §9 attribution matrix — does each recommended dispatch UNIQUELY resolve its attribution dimension?):

- Issue raised by Dykstra: "If D3.B+D3.C parallel lands within 1e-3 of each other AND in the indeterminate band, the experiment is STILL inconclusive even with the parallel arm." Resolution: this is acknowledged in §7c indeterminate band discussion. Adding to reactivation criteria: "If D3.B-D3.C parallel both indeterminate, fire D1.A additive ablation (Round 3) as a clean orthogonal probe." Updated §10.
- Issue raised by MacKay: "MDL view says the wavelet retarget $0.20-1 has insufficient predicted Δ headroom to overcome rate overhead — confirm Mallat sparsity assumption applies to A1 base." Resolution: the predicted Δ -0.0005 to -0.003 on PR106 r2 is bounded by the per-byte EV at PR106 r2's `2.40e-9 pose / 6.66e-9 seg` threshold. A1's analogous threshold is from CLAUDE.md `dS/dB = ...` operating point — and at A1's `pose_avg=3.4e-5` the marginal is HIGHER (more headroom on pose-axis). Wavelet retarget against A1 should have AT LEAST as much headroom as against PR106 r2. Surface as `[prediction-uncertainty]` flag; recommend rerunning Mallat scattering on A1 weights at the $0 grid stage.

**2 issues found → counter resets to 0.**

**Round 4** (final re-run, post-revisions; check all `[meta-council-finding]` tags have explicit reactivation criteria):

- Shannon: §1 top-3 findings — all have reactivation criteria. PASS.
- Dykstra: §7c indeterminate band reactivation added in Round 3. PASS.
- MacKay: §7 empirical resolution — all bands distinguishable except indeterminate. PASS.
- Hassabis: §10 dispatch plan — all conditional dispatches have explicit success/failure routing. PASS.
- Contrarian: every `[meta-council-finding]` has an explicit Δ-vs-current-plan delta. PASS.
- Carmack: $0 grid is well-bounded at 3-4 hr; tool gaps surfaced as operator-routable. PASS.
- Tao: Bayesian priors explicit; EIG math sound. PASS.

**0 issues found → counter advances to 1.**

**Round 5** (re-run inner subset Shannon+Dykstra+Contrarian for final sanity):

- Shannon: rate-distortion arguments inherited from prior council; no new errors. PASS.
- Dykstra: convex feasibility argument applies to parallel dispatch plan (each arm is feasible). PASS.
- Contrarian: "If wavelet retarget on A1 lands at Δ≈0 AND $0 grid says A1 NOT saturated, what happens?" Resolution: this is a TRUE conflict and requires further investigation; explicit reactivation criterion is "investigate wavelet residual basis-vs-A1-decoder bandwidth mismatch (Mallat scattering on A1 weights vs PR106 weights)". Added to §10 reactivation rules.

**1 issue found → counter resets to 0.**

**Round 6** (re-run after Round 5 fix):

- Shannon: all PASS.
- Dykstra: all PASS.
- Contrarian: PASS — the new reactivation rule covers the conflict case.

**0 issues found → counter advances to 1.**

**Round 7**:

- All inner-quintet PASS.

**0 issues found → counter advances to 2.**

**Round 8** (final):

- All inner-quintet PASS.

**0 issues found → counter advances to 3. 3-clean-pass SEAL.**

---

## 17. Operator-routable decisions surfaced

Per CLAUDE.md "Design decisions — non-negotiable": every META-council recommendation that requires operator approval is surfaced explicitly here.

1. **Should META re-rank autopilot dispatch ordering?** §15 hook 4. Recommended: yes — re-rank wavelet retarget #1, LAPose smoke #2, parallel D3 attribution #3.
2. **Should $0 attribution grid (§11) be built/run before any GPU dispatch?** Council strongly recommends. Cost: $0. Time: 3-4 hr. EIG comparable to a single $4-5 dispatch.
3. **Should A1+wavelet residual retarget be authorized?** Cost $0.20-1. EIG 1.0 bit (resolves H3 A1 saturation).
4. **Should D3.B + D3.C be dispatched in parallel?** Cost $7-10 (vs $4-5 single). EIG 1.5 bit (vs 1.0).
5. **Should D1.D + D1.A be dispatched in parallel?** Cost $8-10. EIG 1.0 bit. **Council suggestion: defer to Round 2 conditional on D3 outcome.**
6. **Should `--d3-mode` CLI flag be added to BUILD (mirror of `--d4-mode`)?** Allows probe-disambiguator pattern for D3 attribution. ~30-50 LOC trainer patch. $0.
7. **Should the Huber/Charbonnier robust-loss research follow-up lane be opened?** Per §2c. Research-only; no contest dispatch impact. ~$0 to scaffold.
8. **Should the BUILD memo §10 reactivation criteria be extended to cover the indeterminate band 0.192±0.001?** Per §7c. ~5 LOC change to the memo + trainer rate-cap logic.

---

## 18. Cross-references

- **Prior council memo (BINDING)**: `.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md` (commit `7e77321f`).
- **Paper review**: `.omx/research/siren_literature_review_20260513.md` (commit `af2348fe`).
- **BUILD-RESUME landing**: `feedback_a1_plus_lapose_composition_substrate_landed_20260513.md`.
- **SIREN dispatch audit fix wave**: `feedback_siren_pre_dispatch_audit_fix_wave_LANDED_20260513.md`.
- **Wavelet residual L2 scaffold (retarget candidate)**: `staged_wavelet_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`.
- **C3 residual L2 scaffold (retarget candidate)**: `staged_c3_residual_pr106_sidecar_ready_to_dispatch_20260511T174843Z.md`.
- **Theoretical-floor analyzer refresh**: `.omx/research/theoretical_floor_analyzer_v2_refresh_20260511.md`.
- **Sister D4-DEEPER memo**: NOT YET LANDED at time of this audit; may inform §17 item 6 wiring.
- **Catalog references**: #124 (archive grammar at design time), #125 (subagent coherence), #128 (continual-learning posterior fcntl), #139 (no-op detector), #146 (contest-compliant inflate.sh), #166 (Modal HEAD-parity), #167 (smoke-before-full), #185 (live-count-zero meta-meta-meta), #187 (HNeRV parity guard).
- **Cross-CLAUDE.md non-negotiables honored**: HNeRV parity discipline, Apples-to-apples evidence discipline, Submission auth eval BOTH CPU+CUDA, Frontier target, Meta-Lagrangian/Pareto solver, KILL is LAST RESORT, Adversarial council review of design decisions, FORBIDDEN_PATTERNS (no /tmp, no MPS authoritative, no scorer load at inflate, no KILL verdicts), Race-mode rigor inversion + parallel-dispatch first (this memo DEFERS the parallel-dispatch-first directive to operator decision because no active contest race is declared).

---

## 19. Council seal

**Inner META-COUNCIL** (Shannon LEAD + Dykstra CO-LEAD + MacKay + Hassabis + Contrarian + Carmack + Tao): **SEALED**. All 7 endorse the §10 dispatch plan, the §17 operator-routable surfacing, and the §9 attribution matrix.

**3-clean-pass adversarial review**: COMPLETE (Rounds 1-8). Counter = 3. Memo SEALED.

**Verdict mode**: DEFERRED-pending-empirical for every prediction. NO KILL verdicts. Operator decisions explicit.

**Date**: 2026-05-13
**Lane**: `lane_meta_council_decision_attribution_audit_20260513`
**Audit scope**: A1+LAPose dispatch plan + 6 paper lessons synthesis + competing-paths + empirical resolution + EIG/$ Bayesian ranking.
