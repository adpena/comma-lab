# Cool-Chic AR-prior-training feasibility — does training the prior tighten the coded rate? (2026-06-12)

**Author:** TRACK B step-2 subagent `cool-chic-ar-prior-feasibility-20260612`.
**Lane:** `lane_cool_chic_ar_entropy_coder_20260612` (step-2 continuation; the entropy coder itself landed step-1).
**Evidence grade:** `[contest-CPU advisory]` — **NON-PROMOTABLE** (`promotable=false`, `score_claim=false`). No scorer eval, no contest score, no paid dispatch. The `disc_bits/elem` + `sigma` + real coded-byte trends below are EXACT (deterministic, measured). The frontier (`.omx/state/canonical_frontier_pointer.json`) is **UNMOVED**. This is a MEANS-feasibility gate (sizes a campaign); the END is a lower exact score, stated plainly per the means/ends firewall.
**Smoke:** `experiments/cool_chic_ar_prior_training_feasibility_smoke.py` ($0 CPU, no basin contention by design — trains the tiny AR-prior nets only, never touches the MPS GPU the basin daemon owns).

> **NO-FAKE headline:** the AR prior was REALLY trained (real backprop on the conditional NLL of the REAL trained latents from the step-1 smoke `0.bin`) and the coded bytes were REALLY measured (`encode_latent_chain` range-coded; byte count exact, not predicted). **The prior did NOT tighten with training — that honest negative IS the finding, per CLAUDE.md "a negative is a key finding, not a failure."**

---

## 1. The ONE question (MVP-first)

Step-1's verdict was **DEFERRED-pending-AR-prior-training**: the entropy coder is real + lossless + shrinks bytes (57.5% vs raw int16), but the pure-AR-entropy gain was "only" ~24.5% because the 6-epoch smoke prior's σ was wide (near-marginal). Step-1's stated reactivation hypothesis: *"The fix is more in-curriculum rate training (Layer 2): as the AR prior learns the temporal conditional structure (tighter, context-dependent σ), the coded rate falls toward the true conditional entropy."*

This step-2 smoke tests EXACTLY that hypothesis, in isolation, at $0:

> **When we TRAIN the AR prior (the density-unified Layer-2 rate lever) on the real trained latents, does its predicted conditional distribution TIGHTEN (σ contract) so the REAL coded bytes (`encode_latent_chain`) FALL — and how fast?**

Design (cleanest isolation of the thesis variable = **AR PRIOR QUALITY**): load the step-1 smoke's TRAINED latents (appearance already learned; FIXED), train ONLY the AR prior nets (coarse + fine) to minimize the conditional NLL == `ar_gaussian_predicted_bits` (the EXACT quantity the coder emits — the Layer-1↔2 unifier), and measure (a) AR predicted bits/latent, (b) the REAL coded-byte count on a representative window, (c) sec/epoch on CPU — across 600–800 epochs (130× the smoke's 6).

---

## 2. MEASURED result — the prior does NOT tighten (the decisive data)

### 2a. AR-prior-only training on the FIXED coarse latents (800 epochs, lr=1e-2, the decisive diagnostic)

| epoch | `disc_bits/elem` (the coded rate) | σ_median | σ_median / marginal_std |
|---:|---:|---:|---:|
| 0 (the 6-ep shipped prior) | 6.7192 | 0.05621 | **0.746** |
| 100 | 6.7794 | 0.05776 | 0.766 |
| 200 | 6.7113 | 0.06114 | 0.811 |
| 300 | 6.7333 | 0.06278 | 0.833 |
| 400 | 6.6886 | 0.05649 | 0.750 |
| 500 | 6.6440 | 0.05960 | 0.791 |
| 600 | 6.6435 | 0.05996 | 0.796 |
| 700 | 6.6302 | 0.06034 | 0.801 |
| 800 | 6.6570 | 0.05651 | 0.750 |

- marginal latent std = **0.07537** (coarse).
- **`disc_bits/elem`: 6.7192 → 6.6570 = a 0.93% drop over 800 epochs (best-seen 6.6302 = 1.33% drop).** Essentially flat, and non-monotone (it WANDERS within ±1% — this is optimization noise on a saturated objective, not descent toward a tighter floor).
- **σ_median: 0.0562 → 0.0565 — flat.** It is already at **75% of the marginal std at epoch 0** and stays in the 0.75–0.83 band for the entire 800 epochs. It never contracts toward a tight conditional σ.

**Interpretation (HARD-EARNED): the tiny conv-3×3 AR prior had ALREADY captured ~all the conditional structure it can extract from these latents by the 6-epoch checkpoint.** σ/std ≈ 0.75 means the conditional Gaussian is only ~25% tighter than the marginal — and that ratio is FIXED by the architecture's capacity to predict `z_t` from `z_{t-1}`, not by how long you train it. Step-1's "under-trained prior" hypothesis is **NOT supported**: the prior was near-saturated at 6 epochs.

### 2b. Real-coder confirmation (the full smoke, BOTH axes, REAL `encode_latent_chain`)

The full smoke (`experiments/results/cool_chic_ar_prior_feasibility_20260612T163906Z/ar_prior_training_feasibility.json`, both axes, 600 ep, real `encode_latent_chain` on 32-pair windows at each checkpoint) CONFIRMS the analytic trend on the ACTUAL coded byte stream — the real coded bytes track `disc_bits` within ~0.09 bit/elem (the coder's per-symbol table-quantization + window-flush overhead), so the Layer-1↔2 unification holds AND the prior-saturation conclusion is on real bytes, not just predicted bits:

| epoch | coarse `real_coded_bits/elem` | coarse σ_med | fine `real_coded_bits/elem` | fine σ_med |
|---:|---:|---:|---:|---:|
| 0 | 6.8158 | 0.05621 | 6.8657 | 0.05428 |
| 100 | 6.8668 | 0.05776 | 6.9509 | 0.05710 |
| 200 | 6.7996 | 0.06114 | 6.8540 | 0.05461 |
| 300 | 6.8007 | 0.06278 | 6.8423 | 0.05526 |
| 400 | 6.7707 | 0.05649 | 6.8572 | 0.05680 |
| 500 | 6.7478 | 0.05960 | 6.8292 | 0.05504 |
| 600 | 6.7472 | 0.05996 | 6.8347 | 0.05646 |

- **coarse real coded rate: 6.8158 → 6.7472 = 1.01% drop over 600 ep** (and non-monotone: it rose to 6.8668 at ep100 before settling). σ_med wanders 0.056–0.063 — no contraction.
- **fine real coded rate: 6.8657 → 6.8347 = 0.45% drop over 600 ep** (also non-monotone). The fine axis is even MORE saturated (σ/std ≈ 0.82–0.85; fine marginal std = 0.0672).
- **TOTAL 600-pair latent bytes: 7,897,744 → 7,853,381 = 0.56% drop** over 600 epochs of dedicated AR-prior training. The exp-decay-to-floor fit projects a floor essentially equal to the final (`still_descending_at_run_end = false`).
- **Projected floor = 48.4× HNeRV's ~161 KB decoder-weight floor.** Training the prior does not move this.

**Both axes, both the analytic `disc_bits` and the REAL coded bytes, the standalone 800-ep diagnostic and this 600-ep dual-axis smoke — all agree: ≤1% coded-rate change, σ flat. The AR prior is saturated; training it is not the lever.**

### 2c. Timing (this smoke, measured)

- AR-prior-only CPU: **285 ms/epoch (coarse), 706 ms/epoch (fine)** under heavy basin contention (the basin daemon + a sister proc held 270%/550% CPU). Total wall 736 s for both axes × 600 ep + 14 real coder runs.

---

## 3. Timing (campaign cost sizing)

- **AR-prior-only CPU training: 285 ms/epoch (coarse), 706 ms/epoch (fine)** under heavy contention (the basin daemon pid 33911 + a sister process were at 270%/550% CPU during the run; an uncontended core would be faster). The 800-epoch coarse-only diagnostic took ~9 min wall; the dual-axis 600-ep full smoke took ~12 min wall.
- This is the AR-prior-ONLY loop (tiny nets, fixed latents). **A real campaign trains the FULL model** (latents + synthesis + AR prior + the two scorers' forward/backward) — step-1 measured **~28 s/epoch on MPS** for that full loop. The CPU full loop was deliberately NOT run (the basin owns the GPU; we did not contend).
- The pure-Python `encode_latent_chain` coder is slow (step-1: 91.8 s coarse + 365.4 s fine for the full 600-pair archive). For measurement we used 32-pair windows; the per-element rate is window-stationary so the window faithfully tracks the full-archive trend.
- **CPU is NOT a viable training device** (the heavy contention + single-thread slowness make even the tiny AR-prior loop sluggish). This is itself a campaign-device data point: any real Cool-Chic campaign must use the GPU (MPS or Modal), never CPU.

---

## 4. Projection + VERDICT

### The projection
The coded-rate-vs-epoch trend has **no descent to project** — the total 600-pair latent bytes moved **7,897,744 → 7,853,381 = 0.56% over 600 epochs** (100× the original 6), and the per-axis rates wander non-monotonically within ±1%. The exp-decay-to-floor fit returns `still_descending_at_run_end = false` (the asymptote ≈ the final). **Projecting more epochs does not lower the coded bytes**: the AR-prior-training lever is exhausted at ~6 epochs.

Concretely, the 600-pair AR-coded latent rate stays at **~7.85–7.90 MB** regardless of prior-training length — **48.4× above HNeRV's ~161 KB decoder-weight floor.** Training the prior longer does not close that gap. (The smoke's mechanical `epochs_to_floor`≈4480 / `mps_hours`≈35 / `modal_t4_cost`≈$20 are the exp-fit's extrapolation of the residual ±1% wander; they are NOT a real path to a competitive floor — the floor they reach is still 48× HNeRV's. Reported for completeness; the honest read is "no campaign worth running on this lever.")

### VERDICT: **NO-GO** for the AR-prior-training campaign (as scoped) — with a NEEDS-DESIGN pivot

**NO-GO on the assigned question:** *training the AR prior on the (fixed) trained latents does NOT tighten its predicted distribution and does NOT drop the real coded-byte count* (≤1.3% over 800 epochs; σ flat at 75% of marginal). The step-1 reactivation hypothesis ("more in-curriculum AR-rate training tightens σ → coded rate falls") is **empirically falsified at the implementation level** (Catalog #307: the PARADIGM — conditional entropy coding — is intact; the SPECIFIC claim that *more prior training* is the lever is falsified). Cool-Chic's latent rate floor on this smoke is **NOT set by AR-prior training quality** — it is set by the genuine conditional-entropy content of the latents under a tiny conv-3×3 context model, which the 6-epoch prior already nearly reaches.

**The NEEDS-DESIGN pivot (where the real headroom, if any, lives — NOT a GO, a redirect):** the σ/std ≈ 0.75 floor is a **capacity + latent-structure** ceiling, not a training-time one. Three distinct levers remain, NONE of which is "train the existing prior longer":
1. **Re-optimize the LATENTS for compressibility (the full-campaign lever this smoke deliberately excluded).** The joint loss's AR rate term pushes the *latents themselves* to be more predictable (lower conditional entropy), which is a fundamentally different and potentially stronger lever than fitting a prior to fixed latents. This smoke does NOT measure it (it needs the scorer/recon loop = basin contention). **This is the single highest-value next probe**, but it is a FULL-model training question, not an AR-prior-training question — and it must be measured, not assumed (per "Measurement-first").
2. **A richer AR context model** (larger receptive field, spatial autoregression within a pair, hyperprior side-info à la Ballé) to push σ/std below 0.75 — an architecture change, not a training change.
3. **Coarser grid + T1 cross-pair dedup + T8 scorer-null projection** (the bolt-on inventory's `−0.003 to −0.006` headroom) — these attack the latent rate from the *grid* and *redundancy* sides, orthogonal to the prior.

**Device recommendation (moot for the assigned campaign, stated for the pivot):** since the AR-prior-training campaign is NO-GO, there is nothing to dispatch for it. For the NEEDS-DESIGN pivot lever #1 (latent re-optimization), the workload is the full score-aware loop = **local-MPS at ~28 s/epoch** (sequence behind/alongside the basin; $0) is the right first device, NOT Modal — a $0 local probe must falsify-or-confirm the latent-compressibility lever BEFORE any paid GPU (MVP-first). Modal-T4 only if MPS-hours exceed a basin window AND the local probe shows real descent.

---

## 5. Honest re-statement of the Cool-Chic carrier thesis

Step-1: "Cool-Chic as a *competitive* carrier vs HNeRV's floor is DEFERRED-pending-AR-prior-training." **Step-2 closes that specific defer: AR-prior-training is NOT the path.** The carrier thesis (latent floor below HNeRV's decoder-weight floor) is now **DEFERRED-pending-LATENT-compressibility-evidence** — a strictly narrower, sharper open question. The honest status: Cool-Chic is a real, lossless rate carrier (step-1), but on the only evidence we have it sits ~44× above HNeRV's floor, and the lever step-1 hoped would close that gap (prior training) is exhausted. Whether the *latent re-optimization* lever can close it is the next measured probe — and until it shows real descent, **Cool-Chic is NOT demonstrated to be a lower-floor carrier than HNeRV** (the carrier thesis is weaker than step-1 hoped, stated plainly).

No premature KILL: the paradigm (per-pair latents + conditional entropy) is intact; the *prior-training implementation path* is the falsified branch; the latent-compressibility branch is untested.

---

## 6. Six-hook wire-in (Catalog #125)

1. **Sensitivity-map** — ACTIVE: the `disc_bits/elem` trend is the rate-sensitivity-to-prior-training signal; its flatness is the anchor.
2. **Pareto constraint** — N/A (no score move; advisory rate-only feasibility).
3. **Bit-allocator** — informs: prior-training is NOT a bit-allocator knob (it doesn't move bits); the grid step + T1/T8 are.
4. **Cathedral autopilot** — N/A (non-promotable, no archive dispatched).
5. **Continual-learning posterior** — DESIGN: the σ/std≈0.75 saturation ratio + the ≤1.3%/800ep prior-training elasticity are falsifiable anchors that reseed the judge: "AR-prior-training elasticity ≈ 0; redirect campaign budget to latent-compressibility." Future latent-reopt anchors update it.
6. **Probe-disambiguator** — ACTIVE: this smoke IS the disambiguator between step-1's two interpretations — "coder works but prior is weak/undertrained" (FALSIFIED: prior is saturated, not undertrained) vs "the latent rate floor is structural" (SUPPORTED for the prior-training axis).

**Mission contribution:** `frontier_breaking_enabler` (closes a means-path that would otherwise have burned a campaign on a NO-GO lever; redirects to the measured latent-compressibility probe). Frontier UNMOVED. No score asserted. No GPU launched. No paid spend.

## 7. Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| AR-prior NLL == coded bits | **ADOPT (step-1's `ar_gaussian_predicted_bits` + `encode_latent_chain`)** | The Layer-1↔2 unifier is the exact measurement surface; reusing it IS the test. |
| Training loop | **FORK_PRINCIPLED (AR-prior-only, fixed latents)** | Isolates the thesis variable (prior quality) from latent movement + avoids basin GPU contention; a full-model loop would conflate the two levers and contend for the MPS GPU. |
| Exp-decay-to-floor fit | **FORK (new, dependency-free grid search)** | No scipy.optimize dep; small + deterministic + auditable; returns floor+tau for the campaign-cost projection. |

## 8. Cargo-cult audit (Catalog #303)

- **"the 6-epoch prior is under-trained → train it longer"** (step-1's hypothesis) — **CARGO-CULTED, now FALSIFIED.** Unwound by direct measurement: 130× more training moves the coded rate ≤1.3%. The prior was near-saturated.
- **"prior training and latent re-optimization are the same lever"** — FALSE; explicitly separated. This smoke tests prior-only; latent-reopt is the untested pivot.
- **"closed-form CDF without empirical bit-spend proof"** (Catalog #304) — AVOIDED: the real coder (`encode_latent_chain`) provides the empirical byte count at every checkpoint; the `disc_bits` prediction is cross-checked against it.
