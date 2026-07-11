# Task #442 — WeightWatcher / WW-PGD port + backtest gate (Phase-1 $0 diagnostic)

**Date:** 2026-07-11 · **Axis:** `[macOS-CPU advisory] NON-PROMOTABLE` · **Pointer:** 0.19108282
[contest-CPU] **UNMOVED** (this is MEANS; no score claim) · **Scope:** all numbers MEASURED on
frozen #205 mod32cap checkpoints; verdicts on the verdict-scope ladder.

Script (durable): `experiments/ww_pgd_442_spectral_gate.py` · metrics JSON:
`experiments/reports/ww_pgd_442_phase1_metrics.json`.

## 1. The method (verified against weightwatcher.ai/ww_pgd.html + HTSR literature)

WW-PGD is a **spectral-projection wrapper for optimizers, weights-only inputs**. Per layer:
identify the ESD tail via `k_mid = floor((detX_num + num_pl_spikes)/2)` (SETOL predicts these two
converge as α→2); build a rank-ordered power-law template on the tail (**target α≈2, never α<2**);
apply a Cayley-style update in **log-eigenvalue space**; retract to the ERG trace-log condition
(**Σ log λ_tail = 0 ⇔ detX = 1 on the tail**); reconstruct W and blend back (`blend_eta=0.5`).
Theory: Heavy-Tailed Self-Regularization (Martin & Mahoney) + SETOL. The page is **explicitly
"Experimental"** and reports FashionMNIST "roughly accuracy-neutral" — WW's own claims are
honest-weak; the durable value is the **spectral diagnostics** (α, α̂, detX, stable rank).

The projection **modifies weights** → any port is a **training-physics LEVER** (lever discipline +
A/B), NOT a neutral/bit-identical speedup.

## 2. Estimators (no pip weightwatcher/powerlaw available — implemented directly)

- **α**: Clauset-Shalizi-Newman continuous power-law MLE (Hill form)
  `α = 1 + n_tail / Σ ln(λ_i/xmin)` on the ESD `λ = σ²(W)` (X=WᵀW); **xmin chosen by minimizing
  the KS distance** between empirical tail CDF and fitted PL CDF (CSN). Interp: Martin-Mahoney HTSR
  (α∈[2,4] well-trained; α→2 critical/ideal; α≫6 under-trained/random).
- **ks_D** = KS distance at chosen xmin = **fit quality** (honesty flag at width-96).
- **detX_num** (operational): # top eigenvalues with product ≥ 1 (Σ log λ ≥ 0). Documented as an
  operational ERG reimplementation, **not** a verified match to WW internals.
- stable_rank = ‖W‖_F²/‖W‖₂² = Σλ/λ_max · α̂ = α·log₁₀λ_max · spectral_entropy = −Σ pᵢ log pᵢ.
- **codability proxy** (H2): per-tensor int8 quant (PR95-family L21/L29 style) → **brotli q=11** →
  `bits_per_param = 8·bytes/n_params` (lower = cheaper counted bytes).

## 3. Data — #205 mod32cap curriculum trajectory (one parent run, KNOWN events)

`tau_crossover_trainflow_20260707/` frozen checkpoints, parent
`levelset_n600_witness_mod32cap_20260706T115554Z`. Trunk = 5 square 96×96 (in_proj + hidden.0-3).

| ckpt | epoch | event | τ | d_seg (n600) |
|---|---|---|---|---|
| ep299_CEend | 299 | CE stage end | 0.806 | 0.004594 |
| ep650_tauBest | 650 | τ-best | 0.310 | 0.003146 |
| ep726_MuonStart | 726 | **Muon start** | 0.217 | 0.003033 |
| ep925_liveEMA | 925 | mid-Muon | 0.216 | 0.003867 |
| final_ep1000 | 1000 | final | 0.216 | (n/a) |

## 4. FIT QUALITY (honesty first)

Square-trunk **median KS D = 0.053, max 0.063** at **n_eig = 96**. This is a *reasonable* PL fit
(below the ~0.10 noise line) — better than feared — but 96 eigenvalues still carry real sampling
variance in α. **Standalone measured fact:** the trunk α sits at **2.18–2.81 (mean ≈ 2.5)**, i.e.
squarely in the HTSR "well-trained" band [2,4] and **already close to the α→2 target** (in_proj
α=2.18 is essentially AT target). WW-PGD's projection pushes α→2 → **near-zero headroom on this
trunk.**

## 5. Per-hypothesis MEASURED verdicts

### H1 — REGIME/SENSE : **NO-GO** (INSTANCE/FORMULATION scope)
Trunk metric trajectory movement vs within-checkpoint layer spread (noise floor):

| metric | across-ckpt range | layer-spread noise | range/noise | reads events? |
|---|---|---|---|---|
| **α** | 0.063 | 0.180 | **0.35** | no — flat 2.48–2.55, no Muon step |
| stable_rank | 0.51 | 0.76 | 0.67 | no |
| detX_num | 0.8 | 4.4 | 0.18 | no |
| spectral_entropy | 0.023 | 0.098 | 0.23 | no |
| log_spectral_norm | 0.081 | 0.032 | **2.56** | **yes** — drops after Muon (ep726 1.658→ep1000 1.582) |

The **HTSR-specific** metrics (α, detX) are **flat and noise-dominated** across every curriculum
event. The only metric that moves is the **spectral norm**, which shrinks under Muon — a trivial
weight-norm effect readable without any WW machinery. **α does not sense the regime.**

### H2 — RATE/CODABILITY : **PARTIAL GREEN, but the useful predictor is NOT α**
Pooled homogeneous trunk (n=25, all 96×96; bits/param spans **6.24–7.20 = a real ~16% range**):

| predictor | Pearson vs bits/param | Spearman | read |
|---|---|---|---|
| **spectral_entropy** | **+0.891** | +0.592 | STRONG, clean (effective-rank ↑ → less compressible) |
| detX_num | +0.887 | +0.360 | strong (another effective-rank measure) |
| α | +0.856 | **+0.521** | Pearson strong but rank-corr weak → range/outlier-driven |
| stable_rank | +0.341 | +0.218 | weak |

**Codability IS spectrally predictable** — confirming the premise *spectral shape → rate*. BUT the
clean predictor is **spectral entropy / effective rank**, not the HTSR α specifically (α's Spearman
is only 0.52). This relation is **INSTANCE-scope (n=25, one lineage, advisory)** and **duplicates
the existing MDL / weight-entropy-rate direction** — it does not motivate a *new* WW lever.

**Bonus rate finding (the biggest counted item):** the `code` matrix (1200×32, ~20 KB brotli — the
actual video-derived payload) is **near-rank-1: stable_rank ≈ 1.02, spectral_entropy ≈ 0.17**, and
its codability tracks training (ep299 3.95 → ep726 4.31 → ep1000 3.95 bits/param). A **low-rank
`code` codec** (task #140 territory) is strongly motivated by this — independent of WW-PGD.

### H3 — QUALITY : **NO-GO** (INSTANCE scope, n=4)
mean-trunk-α vs d_seg: Pearson −0.82 but **Spearman −0.40, n=4**, and **non-monotone** (ep925 has
the HIGHEST α yet d_seg RISES to 0.00387). The whole α range (0.045) is within the α fit noise.
No trustworthy quality signal.

## 6. GATE VERDICT — Phase 2 (port WW-PGD as a lever): **NO-GO**

- H1 no regime signal · H3 no quality signal (both noise/INSTANCE).
- H2 has a real *spectral→rate* law, but (a) it is carried by **spectral entropy**, not α; (b) it
  **duplicates the existing MDL/effective-rank rate direction**; and (c) the trunk is **already
  near-critical (α≈2.5)** so WW-PGD's α→2 projection has **near-zero headroom** while still being a
  weight-modifying physics lever that would demand a full A/B for ~nil gain. **Dominated.**

Building the WW-PGD lever is not justified. **No lever, no stub.** The diagnostic is not wasted: it
**re-motivates two already-owned directions** (effective-rank/MDL weight regularizer #242; low-rank
`code` codec #140) with fresh MEASURED evidence — the "results become system intelligence" wire-in.

## 7. Triality legs

- **DAG:** FEED-442 appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL:** N/A — Phase-2 lever NOT built (gate NO-GO). No `Lever` factory added (correct per
  gate; a default-off stub with no measured value would be orphan/stub theater).
- **Equations:** N/A-with-reason — the measured H2 relation (spectral_entropy↔codability, Pearson
  0.89) is **INSTANCE-scope (n=25, single lineage, advisory)** and duplicates the MDL/effective-rank
  premise; promoting it to a canonical LAW (implies FAMILY+ scope) would violate verdict-scope
  discipline. Documented here as advisory evidence for #242/#140 instead.

## 8. What I did NOT do
- No training launches, no pid disturbance (CONTAINMENT); read-only linalg on frozen checkpoints.
- Did not build the Phase-2 WW-PGD lever (gate NO-GO); no DSL `Lever`, no MLX twin, no costate row.
- Did not register a canonical equation (INSTANCE-scope; would over-claim).
- Did not use pip weightwatcher (unavailable) — estimators implemented + cited; α at width-96 is
  labeled with its KS fit quality, not asserted.
