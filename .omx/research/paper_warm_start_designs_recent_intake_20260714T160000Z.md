# Warm-start-from-divergence DESIGNS for recent paper intake (operator method 2026-07-14)

**Method (now contract-enforced, `PAPER_WARM_START_FROM_DIVERGENCE`):** for each paper NOT directly
applicable, trace upstream to where its assumptions diverge from ours, warm-start there with OUR premises
(v9·CGauge witness + level-set flow + frozen SegNet-argmax/PoseNet-YUV6/archive-bytes info space), and
carry ALL THE WAY to a design + implementation. These are the designs I owed and under-delivered as
"route/cross-ref" over the past several days. Each is a v9·CGauge design brief; $0-measurable ones are
queued (heavy/paid = operator-GO). Pointer 0.19108/0.18804 UNMOVED until one is byte-closed through R.

---
## 1. RIPO 2607.10169 → Fisher-isometric margin-natural-gradient seg-head preconditioner (→ #500 metric)
**Divergence fork:** RIPO assumes an RL policy π_θ over discrete actions optimized by PPO-Clip, whose
implicit Euclidean metric on the probability simplex over/under-steps in high/low-density regions.
**Warm-start with OUR premises:** OUR "policy" = the SegNet-softmax per-pixel class distribution the
witness must match; OUR manifold metric = Fisher = the MEASURED margin field (Pearson 0.978); OUR
optimizer = AdamW/Muon on witness weights (not PPO). RIPO's identified pathology — Euclidean steps are
over-aggressive where the softmax is confident (flat interior, DARK in Fisher) and over-conservative
where uncertain (the boundary ANNULUS, BRIGHT in Fisher) — is EXACTLY our measured Fisher geometry
(flat-interior + anisotropic-on-boundary). So a Fisher-Rao-ISOMETRIC (natural-gradient-style) update on
the seg-head systematically REALLOCATES step-size onto the ~4.7%-area annulus where ~97% of d_seg lives.
**Carry to implementation:** a seg-head gradient preconditioner `g' = margin_field ⊙ g` (or a diagonal
Fisher = margin-surrogate scaling, `G≈diag(margin)`), applied at the head only (cheap, θ-count small),
composed with the existing #423 Hessian-preconditioned head-offset. **$0-measurable:** ablate the
preconditioner ON/OFF through R at n600 on the current EMA-best (does margin-Fisher-scaled head-gradient
lower d_seg vs vanilla?). **Land as:** a `Lever` in witness_dsl for #500's metric arm.

## 2. TOP-D 2607.04751 → trust-region surrogate re-anchor schedule (→ #455/#485 throughput)
**Divergence fork:** TOP-D assumes a MOVING teacher and instability from on-policy teacher-student
mismatch; fix = dynamically construct a proximal teacher near the student.
**Warm-start with OUR premises:** OUR teacher (frozen SegNet) is FIXED — no moving-teacher instability.
BUT the DISTRIBUTION SHIFT is real and OURS: the distilled cheap SegNet surrogate (the 95%-kill forward)
was fit on EARLY-witness frames; as the witness trains, its output-frame distribution drifts → the
surrogate's argmax-parity to the frozen teacher DEGRADES on the new distribution (the exact surrogate-
staleness risk). TOP-D's proximal-teacher → OUR version = a trust-region RE-ANCHOR: periodically (event-
triggered on measured argmax-parity drop, not a fixed cadence) re-distill the surrogate on the CURRENT
witness output distribution, keeping the cheap forward valid without paying the full frozen forward each
step. **Carry to implementation:** a `surrogate_reanchor` controller in the throughput path — SENSE =
running argmax-parity of surrogate vs a cheap periodic exact-SegNet spot-check; ACT = re-fit surrogate
when parity < threshold. **$0-measurable:** replay a witness training log, measure surrogate parity drift
across epochs, verify the re-anchor recovers parity. **Land into #455/#485.**

## 3. log-Sobolev-on-cycle 2605.29035 → spectral-gap-bounded τ-anneal rate (→ #318/#500 curriculum)
**Divergence fork:** Frank-Ivanisvili assume a reversible Markov diffusion on the n-cycle; result = sharp
entropy-contraction rate = ½·spectral-gap.
**Warm-start with OUR premises:** OUR "diffusion" = the curriculum τ-anneal (level-set viscosity flow);
OUR "stationary manifold" = the current-temperature witness minimizer; OUR "spectral gap" = the Hessian
gap of the witness loss at the current stage (the EoS/critical-slowing we MEASURED, #316-#320). LSI's
sharp-rate principle → you cannot cool faster than the local mixing time without falling out of
equilibrium (= the eikonal/EoS instability we fought). This is a SECOND, independent derivation of the
optimal cooling rate — from the entropy-contraction side — complementing #318's von-Neumann/CFL DE
derivation. **Carry to implementation:** `dτ/dt ≤ c·λ_gap(t)` where λ_gap is estimated from the measured
loss-curvature/critical-slowing telemetry → a spectral-gap-bounded τ-schedule that the curriculum
derivation (#302/#500) consumes. **$0:** cross-check the LSI bound against #318's DE bound on the same
telemetry — do they agree on the safe anneal rate? **Land into the curriculum-derivation study.**

## 4. ERM 2607.10128 → energy-guided parallel-tempered K-candidate selection (already routed → #396)
Warm-start done in the prior ERM DAG FEED: parallel-tempering the MC-finisher's K-candidate draw + a
cheap Hopfield valid-partition-patch pre-rank. Design is in-hand for the #396/#400 arm to build+measure.

## 5. MorphoHDL (Paradigms of Intelligence) → recursive-subdivision partition generator (→ rule-118/#503)
**Divergence fork:** their SPLIT/CAT recursive-rewrite infers bus widths to grow boolean CIRCUITS.
**Warm-start with OUR premises:** substitute "circuit" → "argmax partition"; a size-agnostic recursive-
subdivision generator that infers RESOLUTION per region (fine on the boundary annulus, coarse in the flat
interior) is a concrete inflate.py generator STRUCTURE for the v8 Laguerre power-diagram / lane-raster —
maximal deterministic generic structure (FREE, rule-118) expanding a tiny video-derived seed. **Carry:**
a recursive-refinement decode structure for #503's fractal-composition (interior-coarse / boundary-fine
partition-of-unity). Design-level; folds into #503, not a standalone dispatch.

---
**Disposition:** Designs 1-3 are the strongest and $0-measurable now; #1 (RIPO margin-Fisher seg-head
preconditioner) is the single highest-value — it drops straight into the live #500 metric arm and targets
the annulus where d_seg lives. I recommend folding #1 into #500 (or a focused $0 dispatch under the cap)
and #2 into #455/#485. Pointer moves only via a byte-closed exact row through one of these.
