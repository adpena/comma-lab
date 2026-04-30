# Council Design Review — Lane MDL/Bayesian (MacKay)

**Status:** Phase A council review for Level 0 → Level 1 graduation.
**Anchor:** Lane G v3 = 1.05 [contest-CUDA] (DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL).
**Predicted band [prediction]:** Lane MDL is primarily a **selection / weighting framework**, not a
direct codec. Its operational role is to RANK competing codec families (Lane Ω-W-V2, Lane J-NWC,
Lane 20 Ballé, Lane 17 IMP) by Bayesian model evidence and to allocate prior weight to the
best-evidence path. **Direct byte savings: 0 ± 200 B** (the framework itself ships nothing in the
archive; the meta-decision picks an existing codec). **Indirect score lift via better stack
selection: -0.005 to -0.015 [prediction]** if the framework reliably picks the best codec under
the realistic competing-prior regime where empirical winners differ by ≤0.01 bytes.
**Cost estimate:** $0 (pure analysis on existing artifacts; no GPU).
**Dependencies:** outputs from Lane Ω-W-V2, Lane J-NWC, Lane 20 Ballé to compute their evidence.

## 1. Existing scaffold audit

`src/tac/mdl_bayesian_codec.py` — DOES NOT EXIST (this lane creates it).
`src/tac/balle_hyperprior_renderer.py` — sibling module (Lane 20) that provides per-codec rate
calculations the MDL framework consumes.
`src/tac/arithmetic_qint_codec.py` — provides the canonical static-prior baseline `R_static`.

## 2. Math foundation

### 2.1 Two-part MDL code (Rissanen 1978; MacKay 2003 Ch. 28)

For a model M and dataset D (= the renderer qint stream / mask stream / pose stream), the
**total description length** is:

    L(D, M) = L(M) + L(D | M)                     [bits]

where `L(M)` = bits to encode the codec / model weights themselves and `L(D | M)` = bits to
encode the data given the model. **Minimize total**, not just `L(D|M)`.

This is the principled accounting that Lane 20 Ballé already does internally: hyperprior MLP
weights (~5 KB) must amortize against y-stream savings. Lane MDL **generalizes this across
codec families** and produces a single comparable score `L_total(family_k, D)`.

### 2.2 Bayesian evidence (MacKay 1992 "Bayesian interpolation")

Equivalent dual: model evidence `p(D|M_k) = ∫ p(D|θ, M_k) p(θ|M_k) dθ`. The evidence
ratio between two codec families is:

    BF_{12} = p(D|M_1) / p(D|M_2)                 [Bayes factor]

`log BF_{12}` is interpretable as "bits of evidence in favor of M_1." Per MacKay,
`log BF > 5` is "strong evidence" (~32:1 odds).

### 2.3 Laplace approximation for tractable evidence

Closed-form intractable; use Laplace at the MAP:

    log p(D|M) ≈ log p(D|θ_MAP, M) + log p(θ_MAP|M) + (k/2) log(2π) - (1/2) log |H|

where `H` = Hessian of negative log posterior at θ_MAP, `k` = parameter count. The `(1/2)log|H|`
is the **Occam's razor** term — penalizes models that need precise parameter tuning.

For our codec families, `θ` = (codec hyperparameters: block size, codebook size, MLP layers).

### 2.4 Variational MDL (Hinton & van Camp 1993)

For a stochastic encoder `q(θ|D)`, the variational free energy is:

    F = E_q[log p(D|θ, M)] - KL(q(θ|D) || p(θ|M))

This is the variational lower bound on `log p(D|M)` — same quantity Ballé 2018 minimizes for
his learned codec. **Lane MDL connects Lane 20 Ballé's training objective to the global codec
selection objective.**

### 2.5 Differential to Lane 20 Ballé hyperprior

Ballé hyperprior is a **single learned codec**: it minimizes `R_y + R_z` end-to-end via SGD.
Lane MDL is a **selection + weighting layer ABOVE** Ballé / NWC / Ω-W-V2: it takes their
final `L_total` numbers and produces:
- Best single codec to ship (decision)
- Posterior over codec families (uncertainty quantification)
- Bayesian model averaging weight if mixing (rare; usually one codec wins decisively)

## 3. Council deliberation

### MacKay (LEAD — channeled, MDL grandmaster)

> "Two-part MDL is the cleanest formulation. Compute `L(M_k) + L(D|M_k)` for each candidate
> codec family. The codec with smallest total wins. The Bayesian-evidence formulation is
> equivalent under uniform priors (or a stated non-uniform prior over codec families).
>
> **Critical:** the prior `p(M_k)` over codec families MUST be stated explicitly. A naive
> uniform prior over (Lane G v3 baseline, Lane Ω-W-V2, Lane J-NWC, Lane 20 Ballé, Lane MDL+X)
> is wrong because these are NOT independent — they share substructure (all use FP4
> quantization; all ship the same renderer architecture). The right prior is over
> *non-overlapping additions* to the canonical pipeline."
>
> **Verdict:** GREEN. The infrastructure that COMPUTES the L_total per family is the
> contribution; the actual numerical comparison is downstream.

### Shannon (LEAD — information theory)

> "MDL = Shannon entropy of the data under the model + entropy of the model. When
> `L(M)` is negligible (well-amortized), this reduces to Shannon's source-coding bound on
> `L(D|M)`. The interesting regime is when `L(M)` is non-negligible — that's exactly the
> Lane 20 amortization question (5 KB MLP weights vs ~5 KB y-stream savings).
>
> **Verifiable claim:** for renderer.bin = 64 KB total at Quantizr; FP4 quantization gives
> static cross-entropy ~3.6 bits/symbol; entropy lower bound ~3.4 bits/symbol; gap ~0.2
> bits/symbol × 100K symbols = ~2.5 KB max from breaking i.i.d. across blocks. Therefore
> any L(M) > 2.5 KB is NEVER worth shipping for THIS renderer.
>
> This is the kind of pre-registration the Lane MDL framework forces."
>
> **Verdict:** GREEN with kill-criterion: report MDL evidence with explicit prior; refuse
> to ship a codec whose `L(M)` exceeds the achievable y-stream savings ceiling.

### Hinton (channeled — variational MDL co-author 1993)

> "The variational MDL framework Hinton + van Camp 1993 is mathematically the same as the
> ELBO that modern neural codecs (Ballé) minimize. The connection is: the ELBO is a
> **variational lower bound** on `log p(D|M)` and therefore a variational UPPER bound on
> the description length `L(D|M)`. So when Ballé reports `R_total` for his trained codec,
> that R_total IS the variational MDL score for that codec family.
>
> Lane MDL should therefore consume Ballé's final R_total directly without re-running the
> variational inference — the work is already done."
>
> **Verdict:** GREEN. Use Lane 20 Ballé's reported R_total as one of the L_total inputs
> directly.

### Schmidhuber (advisory — compression-as-intelligence)

> "Compression IS the model. The model that compresses the data shortest IS the best
> understanding of the data. Lane MDL operationalizes this: pick the codec family with
> smallest L_total. This is the right framework.
>
> One sharper point: MDL handles the model-zoo case. When you have 5 candidate codecs and
> aren't sure which to ship, MDL gives you the principled answer. It also handles model
> averaging if no single codec dominates — but in our regime, one usually does."
>
> **Verdict:** GREEN.

### Quantizr (adversarial — leader at 0.33)

> "I shipped Brotli + FP4 + KL distill. My L_total for renderer.bin is 64 KB. Lane MDL
> ranks codec families by L_total. If your framework correctly identifies my Brotli+FP4
> stack as the optimal CHOICE in our regime, it shipped 0 BYTES of new code in MY archive.
>
> The framework's value is INFORMATIONAL. It tells you which sub-lane to invest in next.
> That has expected value but NOT direct byte savings. Tag predictions accordingly."
>
> **Verdict:** YELLOW. Direct byte savings = 0. Indirect strategic value real but harder
> to measure. Useful for Phase 2 / Phase 3 prioritization gates.

### Hotz (raw engineering)

> "OK so it's a meta-tool. Make it a SCRIPT that takes 5 codec result JSONs and prints a
> ranking table. Don't make it a 1500-LOC neural network. The ENTIRE module is:
>
>     def mdl_total_bits(codec_bytes_shipped, model_bits): ...
>     def laplace_evidence(...): ...   # if you must
>     def bayes_factor(L1, L2): ...
>     def rank_codecs(results): ...
>
> 200 lines. Done."
>
> **Verdict:** GREEN. Implementation should be tiny and audit-friendly.

### Selfcomp (block-FP author)

> "MDL framework is exactly what I needed when picking between block-FP-1.017bpw vs
> block-FP-1.5bpw vs FP4 on my renderer. I picked by hand on intuition. Having a
> framework that picks by evidence ratio is a better engineering practice."
>
> **Verdict:** GREEN.

## 4. Decision

**Adopt:** Implement Lane MDL as a thin **codec-comparison framework**, not a new codec.
Tiny module (~250 LOC) with:

1. `MDLCodecResult` dataclass — per-codec L_total accounting
2. `mdl_total_bits(model_bits, residual_bits, prior_log_p_model)` — two-part MDL
3. `laplace_evidence(log_likelihood_max, log_prior_max, hessian_logdet, n_params)` — Laplace
4. `bayes_factor(L_total_1, L_total_2)` — bits of evidence (returns log2 BF)
5. `rank_codecs(results: list[MDLCodecResult]) -> list[(codec_name, posterior_weight)]` —
   return softmax-over-negative-L_total
6. `bayesian_model_average(weights, predictions)` — weighted ensemble (rarely used)
7. `OccamCheck.kill_if_L_M_exceeds_ceiling(L_M, achievable_savings)` — refuses codecs whose
   model bits exceed achievable y-stream savings

**Wire format:** No new wire format. Lane MDL ships NOTHING in archive.zip. It produces
**reports/lane_mdl_bayesian_ranking.md** and **reports/lane_mdl_bayesian_ranking.json**.

**Kill criteria:**
- If MDL framework's recommendation contradicts the empirical contest-CUDA score on Lane G v3
  more than once: framework prior or evidence calculation has a bug; abandon until fixed.
- If posterior over codec families is < 60% on the actual winner: framework signal is too
  noisy to use for production gating; document as research curiosity only.

## 5. Phase ordering (operational)

1. **Phase A** (this doc) — DONE
2. **Phase B (Level 1)** — `src/tac/mdl_bayesian_codec.py` skeleton + 8-10 synthetic tests
3. **Phase C (Level 2 prep)** — wire to Lane Ω-W-V2 + Lane J-NWC + Lane 20 Ballé result JSONs
4. **Phase D (Level 2)** — produce ranking report on actual artifacts; cross-check against
   contest-CUDA winners
5. **Phase E (Level 3)** — STRICT preflight check: any new codec lane MUST report L_total + ship
   the MDL-ranked winner
6. **Phase F** — 3-clean-pass adversarial review

## 6. Cross-references

- CLAUDE.md "Council conduct — non-negotiable" (no conservative bias; math/empirical only)
- CLAUDE.md "MacKay's specific contributions" (the framework is direct expression of his role)
- CLAUDE.md "Auth eval EVERYWHERE" (MDL recommendation must agree with auth-eval winner)
- `feedback_production_hardened_standard_definition_20260430.md` (the Level 3 bar)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 16 Bayesian MDL"
- `feedback_codec_stacking_composition_canonical_orders_20260429.md` (codec stacking that MDL
  ranks)
- `src/tac/balle_hyperprior_renderer.py` (Lane 20; one of the codec families MDL compares)
- `src/tac/arithmetic_qint_codec.py` (Lane SH; the static-prior baseline)
- MacKay 2003 — *Information Theory, Inference, and Learning Algorithms* Ch. 28 ("MDL")
- Hinton & van Camp 1993 — "Keeping neural networks simple by minimizing the description
  length of the weights"
- Rissanen 1978 — "Modeling by shortest data description"
