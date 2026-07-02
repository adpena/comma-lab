# Activation-axis resolution — hosc(β-anneal+siren-init) vs step_basis vs alternatives (MEASURED)

**UTC** 20260701 · **tag** `[macOS-MLX / CPU-torch advisory · $0 · NON-PROMOTABLE · pointer 0.19110 UNMOVED]`
**Vehicle** the level-set task-space witness (`experiments/train_LEVELSET_witness_realized_through_R_mlx.py`).
**Authority** frozen numpy/CPU-torch, MPS never touched. **means≠ends:** every number here is a MEANS; the
only END is a byte-closed n600 exact row < 0.19110. This memo answers the operator's activation directive
(2026-07-01) with a $0 measured A/B + deep-math + OSS synthesis + verdict.

---

## 0. TL;DR verdict

**Launch activation = `hosc` + `--siren-init` + `β-anneal 1.0→4.0` (linear), NOT fixed-β4, NOT step_basis, NOT siren/finer.**
- The drift is RESOLVED: the "fixed-β hosc DIVERGES" DAG measurements ran **without SIREN-init** (bc20 screen ω=1 + AdamW, and capstone-v1 random-init). The PROVEN baseline (`hosc β4 + siren-init`) is in the **healthy** regime and descended (level-set mod-32 → 0.001243 n96; n200 arm → 0.0037).
- β-anneal 1→4 is a **byte-neutral, already-wired, strictly-better-conditioned** refinement: my grad-health shows β=1 has **0% vanishing gradient** and the best-conditioned chain, sharpening to β=4 (step-native, cleanest Gibbs/R-survival) as the SDF partition pins. Independently DAG-MEASURED to fix untrainability (0.689→descending) and literature-standard.
- **Do NOT anneal to β8** (measured 65% saturation + ill-conditioned chain-ratio 35; sharper step but worse trainability — Pareto-dominated).
- **step_basis** = well-motivated FOLLOW-ON (not wired into the level-set MLX `_act`; needs a port + +12-param byte-close; on bc20 it never beat FINER and K is a speed-not-floor knob).
- **siren/finer** = REJECT (catastrophic ω=30 failure in BOTH my trunk and 1-D; FINER's n100 win is capacity-fragile, reverses at n600).

---

## 1. What the trainer exposes + what the baseline actually used

**LEVELSET trainer `_act` (the launch vehicle) supports ONLY:** `hosc` = `tanh(β·sin(ω·u))`, `wire` = `cos(w0·u)·exp(−(s0·u)²)`, `relu`. Flags: `--activation {wire,hosc,relu}` (default **hosc**), `--hosc-beta 4.0`, **`--hosc-beta-end` (default None → NO anneal → bit-identical fixed-β)**, `--hosc-beta-anneal {linear,cosine}`, `--hosc-omega 1.0`, `--wire-w0 20 --wire-s0 10`, **`--siren-init` (default True)** = SIREN *weight-init* (Sitzmann 2020) applied to in_proj+hidden for hosc/wire ONLY (NOT a sine activation).

**`step_basis`, `siren`, `finer`, `gauss` are NOT in the LEVELSET `_act`.** They exist in (a) the **RGB** witness trainer `train_witness_realized_through_R_mlx.py` `_act` (hosc/siren/finer/wire + LearnableStepBasis) and (b) the canonical torch `src/tac/substrates/siren/activation_family.py` (full menu + `LearnableStepBasis`/`FourierKANActivation`, NO-FAKE tested). **Using them in the 1B level-set launch requires a PORT into the level-set MLX `_act`.**

**BASELINE (n600_v2 launch design + proven n200 Muon arm):** `--activation hosc --hosc-beta 4.0 --hosc-omega 1.0 --siren-init` with **NO `--hosc-beta-end` → FIXED β=4 + SIREN weight-init.** This descended to realized d_seg **0.0037** (n200) / the level-set mod-32 predecessor reached **0.001243** (n96, advisory).

---

## 2. The drift, RESOLVED (from the DAG, verbatim-sourced)

The "fixed-β hosc diverges / b=8 worse than b=4" finding is REAL but was measured on configs **without SIREN-init**:
- **bc20 step-native screen (MEASURED n600, AdamW, ω=1, NO siren-init):** hosc(β4) 0.0119→0.0084→0.0072→**0.01357 RISING = DIVERGING**; hosc(β8) →**0.03996 (worse, β-monotone)**; step_basis(k8) →**0.00643 STABLE/monotone** (train_loss 0.65 healthiest). Mechanism: AdamW ÷√v̂ in flat near-zero-grad regions → full-size steps on noise → weight random-walk → d_seg rises. **Existence-proof: ω=30 does NOT rescue — the pathology is β-saturation, ω-independent.**
- **witness capstone-v1 (RANDOM init):** hosc → gnorm 460 → spike-guard all-skip → d_seg 0.689. "Needs SIREN-init."
- **THE FIX (both MEASURED):** SIREN-init → 0.689→0.221; β-anneal 1→4/8 → 0.689→descending (ep10 0.245). DAG DECISION: config-retire *fixed-β* hosc as an AdamW-saturation failure (NOT a paradigm kill); carry step-native via **step_basis (learnable slope)** and **hosc ONLY with siren-init and/or β-anneal**. Literature "uses HOSC carefully, with annealing."

**⇒ The baseline (fixed-β4 + siren-init) is NOT the diverging config.** It sits in the healthy band. β-anneal is the further hardening.

---

## 3. MEASURED A/B #1 — gradient health on the REAL level-set trunk (MLX, $0, 13s)

Faithful replica of the trainer's `LevelSetRGBWitness` trunk (real curvelet front-end in_feat=80, real SIREN-init, arch mod26/hidden120/n_hidden4), pluggable activation. Loss = per-coordinate CE(softmax(sdf), lstar) on REAL gt_n96 pair-0 SegNet argmax (the CE-curriculum-stage target). 8192 real coordinates. Seeded, MPS never.

| activation | preact_std L0→L4 | vanish_frac (deep) | chain_ratio in/out | gnorm_code | CE 1.70→ (9 pts, 200 steps) | argmax acc | verdict |
|---|---|---|---|---|---|---|---|
| **hosc β4 +siren** | 0.05→1.30 | **0.34** | 4.23 | 0.25 | →**0.0004** | **1.00** | ✅ healthy (baseline) |
| hosc β8 +siren | 0.05→1.35 | **0.65** | **34.99** (ill-cond) | 0.24 | →0.0006 (slow) | 1.00 | ⚠️ over-saturated |
| **hosc β1 +siren** | 0.05→0.38 | **0.00** | 0.57 | **0.38** | →0.0006 | 1.00 | ✅ best-conditioned (anneal start) |
| hosc β4 −siren | 0.41 flat | 0.03 | 0.28 | — | →0.0003 | 1.00 | CE-proxy trains* |
| hosc β8 −siren | 0.55 | 0.34 | 0.62 | — | →0.0002 | 1.00 | CE-proxy trains* |
| siren (ω30) +siren | 0.12 | 0.00 | 0.51 | 3.37 | STALL **1.147** | **0.49** | ❌ fails partition |
| finer (ω30) +siren | 0.11 | 0.00 | 1.02 | 2.27 | STALL **1.145** | **0.49** | ❌ fails partition |
| wire (w0=20) | 0.13 | 0.00 | 0.09 | 1.33 | 0.71 | 0.73 | ⚠️ partial/slow |
| gauss (s=10) | 0.28 | 0.28 | 3.40 | 0.25 | →0.036 | 0.99 | viable smooth |
| relu | 0.14 | 0.46 | **0.006** | 0.11 | →0.00 | 1.00 | worst grad-prop* |

\* **CAVEAT (necessary≠sufficient):** the CE-proxy at init measures *trainability of the partition fit*, NOT realized d_seg through R over long horizon. `hosc β4 −siren` descends in this proxy (MLX default init keeps preacts small → low saturation) yet the DAG shows it DIVERGES through-R at n600 over 200+ epochs (the AdamW random-walk). So this table is the **trainability necessary-condition**; §4 is the **R-survival sufficient-condition**; the through-R n600 convergence A/B is the launch confirmation.

**Reads:** β↑ monotonically raises deep-layer saturation (0%@β1 → 34%@β4 → 65%@β8) — the `d/du tanh(β sin u)=β cos u·sech²(β sin u)` law, sech²→0. β4+siren is healthy (chain 4.2, code-grad 0.25, CE→4e-4); β8 is ill-conditioned (chain 35, 65% sat, slower); β1 is the cleanest (0% vanish, chain 0.57, code-grad 0.38). **siren/finer at ω=30 STALL** (acc 0.49) — the gradient is huge (~19) but off-target; the high-ω sine can't form the smooth per-region SDF.

---

## 4. MEASURED A/B #2 — Gibbs overshoot + R-survival (1-D step, torch, $0)

Fit each activation's tiny INR to a 1-D step at c=0.13 (the codim-1 argmax-boundary prototype), Fourier front-end. Measure L∞ overshoot in the flat plateaus (the error that flips shallow-margin pixels) + argmax-flip after a low-pass round-trip (R proxy). Uses the canonical `activation_family` + `LearnableStepBasis`.

| activation | fit mse | **Gibbs overshoot** | R-survival flip | note |
|---|---|---|---|---|
| hosc β8 | 0.00013 | **0.0037** | 0.0 | sharpest step, least ring |
| gauss | 0.00023 | 0.0072 | 0.0 | lowest-ring smooth bump |
| **hosc β4** | 0.00025 | **0.009** | 0.0 | clean step (baseline) |
| fkan | 0.00015 | 0.0126 | 0.0 | learns harmonics, fits |
| step_basis (K4) | 0.00077 | 0.0275 | 0.0 | fits; higher ring than hosc at K4 |
| relu | 0.00015 | 0.039 | 0.0 | piecewise-linear |
| wire (ω30) | 0.00072 | **0.128** | 0.0 | rings most of the fitters |
| **siren (ω30)** | **0.247** | **0.734** | **0.44** | ❌ cannot fit the step |
| **finer (ω30)** | **0.246** | **0.656** | **0.44** | ❌ cannot fit the step |

**Reads:** step-native/localized families (hosc, gauss, step_basis) ring least (overshoot 0.004–0.027) → the Gibbs-9%-persists deep-math confirmed for the sine family (siren/finer overshoot 0.66–0.73, 44% flip). **β↑ lowers overshoot (0.009→0.0037)** = the *sharpness* half of the trainability↔sharpness Pareto (β8 wins R-survival, loses trainability in §3 → the anneal reconciles: train soft, deploy sharp — but β4 already has 0-flip R-survival, so annealing to β8 is unnecessary).

---

## 5. Deep math (why)

- **Topology match / L∞-optimality.** The target is a piecewise-constant function on a stratified domain (regions ∪ codim-1 boundaries); the natural basis is indicator/step, not sine. d_seg is a pointwise **argmax-at-edge = L∞-at-edge** criterion. A step encodes an edge in **O(1) params, zero L∞ overshoot**; a sine needs O(1/ε) harmonics and **never** removes the ~9% Gibbs overshoot. hosc (`tanh(β sin)`→square-wave as β→∞) and step_basis (Σ soft-Heaviside) are the topology-matched charts.
- **Gibbs → aliasing under R.** The uint8 round-trip (bicubic↑874 → bilinear↓384) is a low-pass; a sine's Gibbs ringing = high-freq content that aliases → flips shallow-margin argmax cells; a saturating step has no ringing → survives (confirmed §4: sine 44% flip, step 0%).
- **Saturation/vanishing gradient.** `d/du tanh(β sin u) = β cos u · sech²(β sin u)` → sech²→0 wherever β·sin u is large. Saturation fraction ↑ with β (measured 0%/34%/65% @ β1/4/8). Under **AdamW** (÷√v̂), tiny-but-noisy gradients in saturated regions → full-size random-walk steps → d_seg rises (the DAG bc20 divergence). Two independent cures: **SIREN-init** (controls the periodic-family weight scale so preacts stay in-band) and **β-anneal** (start β≈1: 0% saturation, near-linear, well-conditioned → sharpen late once the partition pins).
- **Capacity-vs-bandwidth regime.** At the capacity-limited n600 contest regime the binding constraint is **params-per-edge** (step = O(1)/edge), not spectral reach — so FINER's variable-frequency n100 win (−18.7%) is unusable under-budget and **reverses to −4.5%/1.03× at n600** (MEASURED). Step-native's advantage GROWS under capacity pressure.

## 6. OSS / literature (cited)

SIREN (2006.09661, control, band-pass NTK@ω, rings on steps) · FINER (2312.02434, variable-freq sine, n100 −18.7% but capacity-fragile) · WIRE (2301.05187, real-Gabor; NULL here — localized the wrong fixed-sine carrier, rings 0.128) · BACON (2112.04645, analytically band-limited/multiscale — an anti-alias tool, not a step-matcher) · Gauss/Ramasinghe-Lucey (2111.15135, smooth bump, no Gibbs, but low-pass under-fits the edge) · **HOSC/Serrano 2024 (2601.07870, `tanh(β sin`)→square-wave step-train; step-native; used *with annealing*)** · sinc/Saratchandran 2024 (brick-wall low-pass, rings) · FLAIR/rcgauss (2508.13544, band-localized) · FKAN (2409.09323) / SineKAN (2407.04149, learnable Fourier — still Gibbs). Both in-tree deep-math memos rank **hosc #1, step_basis #2** for the argmax-edge-through-R target; the pure-math audit reframes it as "step class beats spectral-bias sine, but the binding lever is capacity+routing — activation is a ~7–20% lever, not the floor."

---

## 7. VERDICT + launch config

**PRIMARY (recommended 1B launch): hosc + siren-init + β-anneal 1.0→4.0 linear** — byte-neutral, zero code change (already wired), best-conditioned early + step-native late, MEASURED-safe both ends.

```
--activation hosc --hosc-omega 1.0 --siren-init \
--hosc-beta 1.0 --hosc-beta-end 4.0 --hosc-beta-anneal linear
# ensure --anneal-epochs = schedule length so β reaches ~4 in phase with tau@300/l7@600 sharpening
```

**PAIRED PROVEN FALLBACK (the attribution anchor): fixed hosc β4 + siren-init** (`--hosc-beta 4.0`, no `--hosc-beta-end`) — the measured-descent lineage (0.001243 n96). Run as the A/B control so the anneal's benefit is attributable and the launch never regresses below the proven floor.

**REJECT for launch:** fixed-β without siren-init (diverges); β8 (Pareto-dominated); siren/finer at ω=30 (stall/ring); wire (cargo-culted, rings). **DEFER (follow-on):** step_basis (port to level-set MLX `_act` + +12-param byte-close, then warm-start A/B); gauss (cheap $0 A/B dark-horse if hosc-anneal underperforms).

## 8. Risks
1. **necessary≠sufficient:** §3/§4 are trainability + Gibbs (init + 1-D); the through-R realized-d_seg n600 convergence A/B (anneal vs fixed-β4) is the launch confirmation — the DAG's real divergence mechanism (long-horizon AdamW-through-R) is NOT reproduced by 200-step CE.
2. **anneal denominator:** `--anneal-epochs` must be the schedule length (not run length) so β reaches 4 in phase; a warm-start arm needs the same care as softmax-temp (trainer already extracts `_hosc_beta_for_epoch` with the `--anneal-epochs` denominator — verified).
3. **anneal is a refinement, not a converged win** on the level-set vehicle — hence the paired proven-β4 fallback.
4. **ω apples-to-oranges** (DAG FEED-24k): the siren/finer rejection is at canonical ω=30; ω-tuned variants are off-target + unproven and out of scope.

## 9. DSL gauge SPEC (for `tac.witness_dsl.gauge` — SPEC ONLY, do not edit gauge.py here)

Add an `ActivationGauge(Enum)` (the trunk-chart gauge cell) patterned on `WarpGauge`/`CarrierGauge`, with a `GaugeCost` row per chart (populated by the $0 probes above), hard-gate-rejecting the non-descending charts:

```
class ActivationGauge(Enum):
    HOSC_ANNEAL_SIREN = "hosc_anneal_siren"   # SELECTED: tanh(β sin), β:1→4 anneal + SIREN-init
    HOSC_FIXED4_SIREN = "hosc_fixed4_siren"   # PROVEN fallback / attribution control
    STEP_BASIS        = "step_basis"          # PENDING-PORT (learnable Σ tanh soft-steps; +12 B)
    GAUSS             = "gauss"               # PROBE dark-horse (no Gibbs, low-pass under-fit)
    SIREN             = "siren"               # HARD-GATE REJECT (ω30 stall, 44% R-flip)
    FINER             = "finer"               # HARD-GATE REJECT (capacity-fragile, ω30 stall)
    WIRE              = "wire"                 # REJECT (cargo-culted, rings 0.128)
    RELU              = "relu"                 # REJECT (not step-native, worst grad-prop)
```
GaugeCost fields to fill from §3/§4: `dseg_probe` (best measured through-R), `gibbs_overshoot`, `r_survival_flip`, `vanish_frac`, `chain_cond`, `byte_delta` (0 for hosc/gauss/siren-family; +12 step_basis), `compliant` (deterministic+byte-closeable), `status` (SELECTED/FALLBACK/PENDING_PORT/REJECT). `fix_gauge` drops PENDING_PORT (can't select an unwired chart) and rejects the HARD-GATE charts, selecting `HOSC_ANNEAL_SIREN` with `HOSC_FIXED4_SIREN` as the paired control.

## 10. Canonical equation
Registered `hosc_activation_saturation_trainability_v1` (see `.omx/state/canonical_equations_registry.jsonl`):
saturation ρ(β)=P(|β cos u·sech²(β sin u)|<ε) monotone↑ in β (measured 0/0.34/0.65 @ β1/4/8); trainable iff preacts held in-band by SIREN-init OR β-anneal; fixed-β≥4 without siren-init + AdamW ⇒ d_seg random-walk-rises.
