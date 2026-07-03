# Gaussian Quant (GQ) — deep dive through the deterministic-repro / no-signal-loss north star

**Paper:** "Training-Free Vector Quantization via Gaussian VAEs" — Tongda Xu, Wendi Zheng, Jiajun He,
José Miguel Hernández-Lobato, Yan Wang, Ya-Qin Zhang, Jie Tang. arXiv:2512.06609 (v1 2025-12-07; v3
2026-05-26; **ICML 2026**, PMLR 306). Code: https://github.com/tongdaxu/VQ-VAE-from-Gaussian-VAE (PyTorch).
**Date:** 2026-07-03 · **Method:** full-PDF `pdftotext` + arXiv HTML render + GitHub README + related-work
web reads; adversarially mapped onto the level-set witness. **Pointer 0.19110 UNMOVED** — this is research
(MEANS); its worth is a concrete deterministic, no-signal-loss, byte-closeable rate path OR an honest
"not yet, because X."

---

## §0 — HEADLINE VERDICT (means/ends firewall, honest)

**WATCH-ITEM (rate-axis, north-star-aligned), NOT a near-term lever.** GQ is a beautiful, proof-carrying
instance of *exactly* our "deterministic generation as a legal score-mover" discipline — a **seeded random
Gaussian codebook** (regenerated from a shared RNG seed on both sides → **zero stored codebook bytes →
rule-118 FREE**, bit-identical, host-portable) plus a **closed-form "certify-or-block" condition**
(`log K ≥ bits-back rate ⇒ bounded quantization error`, with error decaying *doubly exponentially* above the
rate). That condition is the single genuinely-new thing GQ hands us. **But:**

1. GQ is a **quantizer / rate-term method**, and **d_seg is our binding wall** (need ~0.001; rate is
   comparatively understood). It cannot move the wall.
2. Its clean bound controls the **latent** error `|ẑ_i − μ_i|`; it gives **NO free d_seg guarantee**,
   because SegNet argmax is the discontinuous readout where a tiny latent nudge flips the partition — the
   surrogate-≠-authority wall. Any d_seg claim must be **measured through-R on n600**, not bounded.
3. The headline "seeded codebook = free bytes" is **already a registered canonical equation for us** —
   `procedural_codebook_from_seed_compression_savings_v1` (`ΔS = −25·(N−K)/37,545,489`). GQ does not add
   that; it adds the *sizing law* underneath it.
4. GQ requires **re-architecting the payload as a constrained Gaussian VAE** (a factorized-Gaussian
   posterior per coded quantity, KL-controlled) — which our softmax-of-SDF witness is not — and the
   byte-efficient operating point (`log K ≈ R`) is exactly where the bound is *loose*; the *comfortable*
   regime (`log K = R + t`, they train to `log 3K`) costs `t` nats/token of **counted** bytes we cannot
   spare at the frontier.
5. On our already-tiny near-entropy payload (hundreds of bytes → ~KB), the marginal win over an ordinary
   arithmetic coder is **sub-0.0003 in S** — below exact-eval noise.

It is **dominated / redundant** with the flat-minima/MDL rate lever already ranked **#1** in
`compression_as_intelligence_lineage_crossref_20260702.md` §9 — GQ is one *specific training-free codec*
that would *realize* that lever's bits-back rate, not a new source of rate reduction. **Reactivation
criterion is concrete (§7).** The durable prize is a **canonical-equation sister** (the K-sizing
certify-or-block bound) that hardens our existing procedural-codebook equation regardless of whether we
ever ship GQ.

---

## §1 — WHAT GQ IS (first-principles, the whole structure at once)

Classic VQ-VAE (van den Oord 2017) is hard to train because the discretization (nearest-codeword argmin +
straight-through / commitment / codebook losses) is unstable. GQ's move: **don't train the VQ at all.**
Two stages:

**Stage 1 — train a *constrained* Gaussian VAE.** A vanilla VAE with prior `N(0,I)` and a fully-factorized
Gaussian posterior `q(Z_i|X) = N(μ_i, σ_i²)`. Its (negative) ELBO is literally a rate–distortion functional
(paper Eq. 2):

```
L_VAE = λ · D_KL(q(Z|X) || N(0,1))  +  E[Δ(X, g(z))]
        └── bits-back coding bitrate ──┘   └── distortion ──┘
```

The per-dimension **bits-back coding bitrate** is `R_i = D_KL(q(Z_i|X) || N(0,1))` nats (Hinton & Van Camp
1993; Townsend et al. 2019) — the number of nats needed to *losslessly* communicate `z_i` to arbitrary
precision when you have a shared prior. `R_i^bits = R_i / log 2`.

**Stage 2 — convert to VQ, training-free.** Generate a **1-D codebook of Gaussian noise** `c_{1:K} ∼ N(0,1)`,
**shared across all dimensions**, **fixed once generated**. For each latent dim, snap the posterior mean to
the nearest codebook entry (paper Eq. 3):

```
ẑ_i = argmin_{c_j ∈ c_{1:K}} || μ_i − c_j || ,   c_{1:K} ∼ N(0,1)
```

The token is the **index** `j`. For a group of `m` dims → one big token, codebook `∼ N(0, I_m)`,
σ-normalized distance + an `ω·||c_j||` push-outward term to avoid codebook collapse at low bitrate (Eq. 8;
because `|μ_i| ≤ √(2R_i)` from Lemma A.1, far-from-0 codewords are never selected when `R_i` is small).

**The critical repro sentence (Appendix D.6, verbatim):** *"our codebook can be generated on the fly by
maintaining the same random number generator seed on both the encoder and decoder sides. Therefore, our GQ
model has the same parameter size as the vanilla Gaussian VAE."* → **the codebook is not stored; it is a
seed.** Table 8: GQ is **nearly invariant to the random seed** (robust across draws). This is the
deterministic-reproducibility crux, and it is *native* to the method, not bolted on.

---

## §2 — THE THEOREM (verbatim + proof intuition — the joyful part)

**Theorem 3.1 (upper bound).** With `|μ_i σ_i| ≤ c₁`, `|μ_i| + |σ_i| ≤ c₂`, fixed `R_i = D_KL(q(Z_i|X)||N(0,1))`,
the probability of a *large quantization error* `|ẑ_i − μ_i| ≥ σ_i` decays **doubly exponentially** in the
excess nats `t = log K − R_i`:

```
when log K = R_i + t :
    Pr{ |ẑ_i − μ_i| ≥ σ_i }  ≤  exp( −e^t · √(2/π) · e^(−c₁−0.5) )
```

**Theorem 3.2 (lower bound / converse).** Below the rate the error probability rushes to 1:

```
when log K = R_i − t :
    Pr{ |ẑ_i − μ_i| ≥ σ_i }  ≥  1 − e^(−t) · √(2/π) · e^(0.5 c₂² − 0.5)
```

**Guideline (their Table 7 confirms empirically):** set `⌈log₂ K⌋ = ⌈R_i^bits⌋`. Larger K buys nothing;
smaller K blows up.

**Proof intuition (elegant, ~10 lines).** The codebook draws `K` i.i.d. `N(0,1)` samples. A large error
means *none* of the `K` samples landed in the interval `[μ_i − σ_i, μ_i + σ_i]`:

```
Pr{large err} = (1 − [Φ(μ+σ) − Φ(μ−σ)])^K            (K independent Bernoulli misses)
             ≤ exp(−K · [Φ(μ+σ) − Φ(μ−σ)])           (1+y ≤ e^y, Bernoulli inequality)
             = exp(−K · ∫_{μ−σ}^{μ+σ} φ(x) dx)
             = exp(−K · 2σ · φ(x′))                    (integral mean-value theorem)
```

Then bound `φ(x′)` below by the worst-endpoint Gaussian density and complete the square:
`2σφ ≥ √(2/π)·e^(−R_i − |μσ| − 0.5)` (the `μ²+σ²−log σ²−1 = 2R_i` identity from Lemma A.1 substitutes the KL
straight in). Substituting `K = e^(R_i + t)` gives `exp(−e^t · √(2/π) · e^(−c₁−0.5))` — the `e^{R_i}` in K
**cancels** the `e^{−R_i}` in the density, leaving a clean `e^t` in the double exponent. That cancellation
IS the theorem: **codebook capacity `log K` must pay for the information content `R_i`; every surplus nat
`t` then buys doubly-exponential reliability.** Pure coding theory (a sphere-covering / coupon-collector
argument in disguise).

**Assumptions that matter for us:** factorized-Gaussian posterior, bounded `μ,σ` (auto-satisfied given
finite `R_i`), per-dimension independence. The bound is on the **latent**, and is agnostic to the
downstream task.

**TDC (Target Divergence Constraint).** A single shared `K` is only efficient if every dim has ~equal
`R_i` (else small-KL dims waste bits, big-KL dims exceed K and blow up). TDC extends the MIRACLE/HiFiC
single-λ heuristic to **per-dimension λ**: penalize `R_i` above `log K + α` hard (`λ_max`), below `log K − α`
soft (`λ_min`), else `λ_mean`; λ's updated multiplicatively (β=1.01), clipped to `[10⁻³,10³]`; **target set
to `log 3K`** (a few nats of headroom `t` above `log K`, exactly buying the safe side of Thm 3.1). Table 5:
vanilla VAE `R_i` spans 0.26–27.29 bits; TDC narrows to 2.93–5.63 bits. α=0.5, β=1.01; effect minimal for
α≤0.5. TDC also improves **TokenBridge** (Wang et al. ICCV'25 — a sister training-free PTQ conversion that
did *not* KL-constrain per-dim; GQ shows TokenBridge = the special case of an *evenly-partitioned* codebook,
and should also match `R_i`).

Practical numbers: `K ∈ {2¹⁴..2¹⁸}`, `m ∈ {1,4,8,16}`, `N ∈ {256..4096}`, latent-ch 16, UNet(SD3)/ViT,
ImageNet/COCO, metrics PSNR/LPIPS/SSIM/rFID/gFID. Beats VQGAN/FSQ/LFQ/BSQ; +TDC beats TokenBridge.
Prior-posterior mismatch measured at **0.00033 bpp (~0.1%)** → the bits-back rate is nearly tight in practice.

---

## §3 — THE DETERMINISTIC-REPRO / NO-SIGNAL-LOSS MAPPING (north star, front and center)

This is where GQ shines philosophically, and it is worth stating crisply because it is *our exact
discipline, proven in someone else's paper*:

| Our CLAUDE.md discipline | GQ instance |
|---|---|
| **rule-118: compile the generator; generated tables are FREE, only video-derived payload counted** | codebook `∼N(0,1)` from a **seed** → 0 archive bytes; only token **indices** counted |
| **deterministic decode: same archive → bit-identical output every host** | *"same RNG seed on both encoder and decoder sides"* → codebook bit-identical; Table 8 seed-robust |
| **"certify or block" — never move bytes without a machine-readable reproducibility proof** | Thm 3.1/3.2 IS a certify-or-block condition: `log K ≥ R_i` **guarantees** (doubly-exp) bounded error; `< R_i` **provably fails** |
| **no signal loss ever** | the bound *quantifies* the loss and its guarantee — you know exactly how many bits buy how much fidelity, closed-form |
| **S is a two-part description length under a task distortion** (compression-as-intelligence §MDL-S) | GQ's ELBO `= R_i (bits-back = L(model)) + distortion (L(data|model))`; the token stream is the MDL code |

GQ is thus a **published, proof-carrying validation** that "deterministic seeded generation with a
closed-form rate condition" is a legitimate, SOTA-competitive codec — precisely the class our north star
says is a legal score-mover. Even as a pure framing win, that is worth durably capturing (§8 canonical-eq
proposal).

---

## §4 — CROSS-REFERENCE AGAINST OUR PROJECT (the point)

### 4.1 The seeded-codebook insight is ALREADY OURS
`src/tac/canonical_equations/procedural_codebook_savings.py` registers
`procedural_codebook_from_seed_compression_savings_v1`: replacing N codebook bytes with K seed bytes
(K≪N) yields `ΔS = −25·(N−K)/37,545,489`, compliant via
`tac.procedural_codebook_generator.derive_codebook_from_seed`. **So GQ's headline (seeded codebook = free)
is not new to us.** What our equation *lacks* and GQ *supplies*: **how big must K be, and what quantization
error does that incur?** GQ's Thm 3.1/3.2 is the missing **sizing + certify-or-block law** underneath our
savings equation. That is the clean, non-redundant integration point (§8).

### 4.2 GQ = the training-free realization of our #1 rate lever (flat-minima / MDL weights)
`compression_as_intelligence_lineage_crossref_20260702.md` §9 ranks the **flat-minima / MDL weight
compression** regularizer (Hinton–van Camp variational weight-noise + Dziugaite–Roy PAC-Bayes flatness +
Wallace-Freeman per-parameter quantization) as **the one concrete, byte-closeable, $0 rate lever**. GQ is
the *same lineage's codec*: if the witness weights `θ` (or a per-pair latent) are trained with a
**variational Gaussian posterior** `q(θ_i)=N(μ_i,σ_i²)` and a KL to `N(0,1)` (= the flat-minima objective),
then GQ converts them to a discrete code **training-free**, with:
- **counted bytes = Σ_i log₂K = total bits-back rate = the KL** (Wallace-Freeman rate, realized without a
  learned entropy model or arithmetic coder),
- **codebook = seed = free**,
- **closed-form K per weight/group** (Thm guideline `⌈log₂K⌋ = ⌈R^bits⌋`),
- **O(log K) decode** per token via bisection for m=1 (Appendix D.7) → trivially inside the 30-min budget.

**Crucial honesty:** GQ does **not reduce** the description length below the KL — it *achieves* it. The
actual rate *reduction* is done by the flat-minima regularizer making the KL small (flat/noisy weights =
low `R_i`). GQ is the elegant, deterministic, entropy-model-free *codec on top*. So it is **downstream of,
and dominated in urgency by, the #1 lever** — you must build the variational-weight witness first; GQ is
then a *choice of coder* competing with an arithmetic coder against the same `N(0,1)` prior.

### 4.3 Where it could physically plug into the witness (candidate analysis)
The witness (per CLAUDE.md §WITNESS CAPSTONE) ships: (A) INR weights `θ` (Fourier features + FiLM-per-pair
mod + 5-class head); (B) a per-pair FiLM mod-code (the video-derived per-frame statistic, ~intrinsic 8-dim);
(C) the store-nothing pose sufficient statistic `ξ` (~1–2 KB).

- **(A) weights θ** — candidate, but requires a Bayesian/variational witness (see 4.2). Highest structural
  cost, and it *removes the deterministic softmax-of-SDF head that IS the d_seg vehicle* if done naively.
  The KL-total = counted bytes; only worth it if flat-minima proves the KL is genuinely small.
- **(B) per-pair mod-code** — the **best fit**. 600 pairs × a small code; if given a variational head with
  KL to `N(0,1)`, GQ codes it training-free with a seeded codebook, bounded per-token. Counted ≈ `600 ·
  Σ R` ≈ hundreds of bytes → matches our existing "AR-coded → hundreds of bytes" estimate. GQ's marginal
  gain over arithmetic coding here is the *fixed-vs-variable-length* gap (mitigated by TDC/grouping), i.e.
  small.
- **(C) pose ξ** — already solved by the store-nothing FiLM sidecar; pose contribution `√(10·d_pose)~0.018`
  is near-floor. Discretizing ξ via GQ is where the bytes *aren't*. Low value.

### 4.4 Bits-back ↔ MDL ↔ our theory spine
`R_i = D_KL(q||p)` **is** the MDL two-part description length (Hinton–van Camp; Shannon/MacKay on the inner
council; MacKay's *bits-back* is canonical). Our `S = 100·d_seg + √(10·d_pose) + 25·bytes/N` is literally
`L(task-data | model) + β·L(model)` with `25/N` = the IB/MDL Lagrangian β (per compression-as-intelligence
§0). GQ operationalizes the `L(model)` half as a *token stream whose length = the KL* and whose *dictionary
is free*. It is the **cleanest published bridge from "S is an MDL code" to "here is the deterministic,
seed-free-dictionary codec that realizes the model half."** That naming/lineage consistency is real
non-forgetting value even without a byte-closed row.

### 4.5 Reverse channel coding sisters (why GQ, not MRC)
GQ explicitly contrasts itself with **Minimal Random Code Learning (MRC; Havasi et al. 2018, "getting bits
back from compressed model parameters")** and reverse-channel-coding (Li–El Gamal SFRL, Flamich, Theis–Yosri,
He et al.): MRC *stochastically samples* to simulate `q`; a **VQ-VAE needs deterministic quantization**, and
GQ's nearest-mean snap *outperforms MRC by construction* (Eq. 3) while `m=1` bisection gives superior
asymptotic complexity. **This determinism is exactly why GQ (not MRC) fits our repro spine** — MRC's shared
random index is deterministic-given-seed but its *sampling* semantics are a worse fit for our
"same-bytes→bit-identical-output" contract. Good to know we'd reach for GQ over the RCC family.

---

## §5 — ADVERSARIAL / HONEST-LEVER CHECK (overturn the negative, or state it plainly)

Per discipline: negatives are suspect until adversarially overturned; a negative falsifies the *toy*, not
the *paradigm*. I steelmanned the LEVER case hard. It does not clear the bar **now**, for five measured/
derived reasons:

1. **Wrong axis.** GQ touches only `25·bytes/N`. The binding wall is **d_seg** (need ~0.001; the rate half
   is comparatively understood: indirect-RD sufficient statistic → hundreds of bytes). A rate codec cannot
   move the wall.
2. **The bound does not transfer through argmax.** Thm controls latent `|ẑ−μ|`. d_seg is `SegNet-argmax`
   through-R — the codim-1 separatrix where a sub-σ latent nudge *flips the partition*. So a "small
   quantization error guaranteed" is **not** a "small d_seg guaranteed." This is the surrogate-≠-authority
   wall; d_seg must be **measured through-R n600**, and GQ gives no shortcut. (The paper itself only reports
   PSNR/rFID reconstruction proxies — never a frozen-task argmax.)
3. **The byte-efficient regime is where the bound is loose.** Safety wants `log K = R + t` (they train to
   `log 3K`, ~1.1 nats headroom). Every headroom nat is `+1 nat/token` of **counted** bytes. At our frontier
   (rate term ~0.118) we run near `t≈0`, where Thm 3.1's exponent is `O(1)` → an O(1) fraction of tokens
   exceed σ. The *comfortable* regime costs bytes we do not have.
4. **Re-architecture cost + it kills our d_seg vehicle.** GQ needs the coded quantity to be a
   *KL-constrained factorized-Gaussian posterior*. Our witness is a **deterministic softmax-of-SDF coord-INR**
   — not a Gaussian VAE. Converting it re-introduces VAE training (GQ removes VQ-codebook training, **not**
   VAE training) and risks displacing the exact head that produces d_seg. Bad trade while d_seg is open.
5. **Marginal magnitude.** On our payload (hundreds of bytes → ~KB near-entropy), GQ-vs-arithmetic-coding
   differ by ~10–20% fixed-vs-variable-length: `25·(Δ~400 B)/37.5M ≈ 2.7e-4` in S — below exact-eval noise,
   ~700× below the d_seg gap. Even the steelman yields a sub-0.0003 move.

**Not a paradigm kill.** The *paradigm* (seeded deterministic dictionary + closed-form rate condition) is
correct and *already ours* (§4.1). GQ is a specific, dominated-in-urgency realization of the rate lineage
we already rank #1 — implementation-level "not yet," not paradigm-level "no."

---

## §6 — WHAT GQ GENUINELY ADDS (the durable prize)

Strip everything and one thing remains that we did **not** already have: **a closed-form, certify-or-block
codebook-sizing law** — `log K ≥ R_i` (with `R_i = D_KL(q‖p)`) ⇒ quantization error decays doubly
exponentially in the surplus; `log K < R_i` ⇒ error → 1. Our existing
`procedural_codebook_from_seed_compression_savings_v1` predicts the *savings* of a seeded codebook but is
**silent on how to size it and whether the result is faithful.** GQ supplies exactly that missing half. Even
if we never ship a Gaussian-VAE witness, this bound is the principled answer to *"how many bits must a
generated/procedural codebook carry to be lossless-enough?"* — a question our procedural-codebook and
per-tensor-quantization (PR95 L21/L29, Wallace-Freeman) surfaces all implicitly ask.

---

## §7 — EV VERDICT + CONCRETE NEXT STEP

**EV: LOW now, conditional-MEDIUM later. Verdict: WATCH-ITEM (rate-axis), dominated by flat-minima #1.**

Do **not** spend a unit building a Gaussian-VAE witness for GQ while d_seg is the wall. The correct ordering
(already in the lineage doc) is: **flat-minima/MDL lever first** (add variational weight-noise / SAM-flatness
to the *existing* witness loss; measure Δ(counted bytes) at fixed d_seg through-R, n600 — the $0 experiment
already queued as #242-adjacent). GQ is the *coder you reach for after* that lever proves the KL is small
AND the payload is a continuous variational bank.

**Concrete reactivation trigger (pin it):** *IF* (a) the witness payload becomes dominated by a large
continuous per-pair mod-code **or** a variational weight bank with a trained `N(0,1)`-KL posterior, **AND**
(b) d_seg is already at/near target so rate is the binding term, **THEN** run the $0 GQ experiment: size
`K = ⌈2^(R^bits)⌋` per Thm 3.1, snap (bisection, m=1), byte-close, and **measure Δ(archive bytes) and
d_seg/d_pose through-R on n600** vs the arithmetic-coding baseline. Promote only on a byte-closed exact row.
Until then: dominated.

---

## §8 — TRIALITY INTEGRATION PROPOSAL (propose only — DO NOT register without review)

The campaign is one object in three consistent views (DAG ↔ DSL ↔ equations). GQ, honestly scoped, touches
all three but should land as **one clean sister equation + one DSL lever stub + one DAG watch-node**, NOT a
new paradigm.

**(a) Canonical equation — SISTER to the existing procedural-codebook equation (RECOMMENDED, but gated).**
Propose `bits_back_codebook_size_quantization_error_bound_v1`, the sizing/certify-or-block law that hardens
`procedural_codebook_from_seed_compression_savings_v1`:

```
latex_form:  Pr{ |ẑ_i − μ_i| ≥ σ_i }  ≤  exp( −e^{(log K − R_i)} · sqrt(2/π) · e^{−c₁−0.5} ),
             R_i = D_KL(q(Z_i|X) ‖ N(0,1)),   guideline ⌈log₂ K⌋ = ⌈R_i / ln2⌋
one_line:    "A seeded/procedural codebook of size K faithfully quantizes a Gaussian-posterior latent iff
              log K ≥ its bits-back KL rate; error decays doubly-exponentially in the surplus nats."
units_in:    {R_i: nats, K: count, c1: dimensionless}
units_out:   {error_prob: probability}
producers:   [tac.procedural_codebook_generator.derive_codebook_from_seed  (the WHERE)]
consumers:   [procedural_codebook_from_seed_compression_savings_v1 (the HOW-MUCH-saved),
              any future variational-weight/mod-code quantizer]
domain_of_validity: {posterior: factorized_gaussian, coded_quantity: latent_not_argmax_task_metric}
```

**GATE (important, per canonical-equations discipline):** this has **no through-R EmpiricalAnchor from our
data** — it is a *borrowed theorem*. Registering it now would be tribal-knowledge-as-equation unless we
either (i) mark `predicted_vs_empirical_residual = 0 @ registration` with `source_artifact = arXiv:2512.06609`
(paper-as-anchor, like the PR95-family L-lessons) and `next_recalibration_trigger =
RECALIBRATE_ON_NEW_ANCHORS`, **or** (ii) carry `# FORMALIZATION_PENDING:borrowed-bound-no-through-R-row`.
**My recommendation: register it as a paper-anchored *design law* (i) ONLY if the operator wants the
sizing bound formally bound to the procedural-codebook equation now; otherwise hold as
FORMALIZATION_PENDING until the §7 $0 experiment produces the first through-R residual.** Do NOT invent a
d_seg version of the bound — it does not transfer (§5.2).

**(b) DSL lever stub (curriculum_dsl.py) — NOT yet.** A `SeededGaussianVQ(k_bits=..., group_m=..., omega=...)`
quantization *stage* is the natural DSL form (sits at the byte-close boundary, sibling to the existing
`gauge.py` L4-slots / L3-geometric-tolerance-quantize / L2-temporal-delta stack). **But it presupposes a
variational-latent witness that does not exist**, so wiring it now would be a dead lever (violates "no
scaffold without an imminent exact row"). **Propose: leave a one-line comment stub in `gauge.py` near the
L3 quantize slot** pointing at this memo + the reactivation trigger; do not add a callable.

**(c) DAG node — YES, a watch-node.** Add a FEED node: *"GQ (2512.06609) = training-free seeded-Gaussian-VQ;
bits-back K-sizing certify-or-block bound; WATCH-ITEM rate-axis, dominated by flat-minima #1; reactivation =
§7 trigger; sister to procedural_codebook eq."* This keeps the three legs consistent (equation proposed ↔
DAG watch ↔ DSL stub-comment) without drift.

---

## §9 — LEDGER ONE-LINER (for `reference_papers_checked_not_relevant_or_watch_item_ledger`)

> **arXiv 2512.06609** "Training-Free Vector Quantization via Gaussian VAEs" (Gaussian Quant / GQ; Xu et al.,
> ICML 2026; PyTorch OSS). Train a KL-constrained (TDC) Gaussian VAE, then convert to VQ **training-free** by
> snapping each posterior mean to the nearest entry of a **seeded random-Gaussian codebook** (regen from a
> shared RNG seed → 0 stored bytes, bit-identical both sides; only indices counted). Theorem: `log K ≥
> bits-back rate R_i=D_KL(q‖N(0,1))` ⇒ quantization error decays **doubly-exponentially** (certify-or-block).
> **VERDICT: WATCH-ITEM (rate-axis, north-star-aligned), NOT a lever now.** CRUX: (a) it is a *quantizer*;
> **d_seg is the wall**, not rate; (b) the clean bound is on the **latent**, does **NOT** transfer through
> SegNet-argmax to d_seg (surrogate≠authority — measure through-R); (c) the "seeded codebook = free"
> insight is **already ours** (`procedural_codebook_from_seed_compression_savings_v1`); GQ only adds the
> K-**sizing** bound; (d) needs re-architecting the witness as a Gaussian VAE (kills the softmax-of-SDF
> head) and the byte-efficient `log K≈R` regime is where the bound is loosest; (e) **dominated by the
> flat-minima/MDL #1 rate lever** — GQ is the training-free *codec* that would *realize* that lever's
> bits-back rate, not a new reduction; marginal S move ~2.7e-4. REACTIVATION: after flat-minima proves the
> KL small AND rate becomes the binding term, run the $0 GQ-vs-arithmetic byte-closed n600 experiment. Sister
> to Havasi MRC / reverse-channel-coding (GQ is deterministic-by-construction → better repro fit than MRC).
> Durable prize: the K-sizing certify-or-block bound as a canonical-eq sister to procedural_codebook.

---

## Sources
- arXiv:2512.06609 (full PDF `pdftotext` + arXiv HTML v3) — theorems, proof (Appendix A), TDC, Appendix D.6
  (seed-both-sides), Table 8 (seed robustness), Table 19 (0.00033 bpp prior-posterior mismatch).
- https://github.com/tongdaxu/VQ-VAE-from-Gaussian-VAE (README; `pit/quantization/gaussian.py`
  `GaussianQuantRegularizer`; `sd3unet_gq_0.25.yaml` K=2¹⁶, m=16; codebook regen from prior samples, PyTorch,
  optional CUDA kernel — **no numpy/MLX port**).
- Hinton & Van Camp 1993 (bits-back / MDL weights); Townsend et al. 2019 (BB-ANS); Havasi et al. 2018 (MRC —
  "getting bits back from compressed model parameters"); Wang et al. ICCV'25 (TokenBridge); Mentzer FSQ,
  Yu LFQ, Zhao BSQ, van den Oord VQ-VAE.
- OUR repo: `src/tac/canonical_equations/procedural_codebook_savings.py`
  (`procedural_codebook_from_seed_compression_savings_v1`); `.omx/research/
  compression_as_intelligence_lineage_crossref_20260702.md` §9 (flat-minima/MDL = #1 rate lever); CLAUDE.md
  §"inflate.py is a FREE interpreter — COMPILE the generator" (rule-118); §WITNESS CAPSTONE.
```
