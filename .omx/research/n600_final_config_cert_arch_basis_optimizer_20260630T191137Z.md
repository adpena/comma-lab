# n600 v2 witness — FINAL-CONFIG CERTIFICATION: ARCHITECTURE · BASIS · OPTIMIZER

**UTC** 20260630T191137Z · `[macOS-MLX/numpy advisory · certification artifact · NON-PROMOTABLE]` · **pointer 0.19110 UNMOVED.**
Author = independent certification scientist (CPU-only, NO GPU, NO launch, READ-ONLY on caches/ckpts, not a score).
Certifies the ARCH+BASIS+OPTIMIZER axis of the GO-ready from-scratch n600 v2 config **before** a 1000-epoch one-GPU burn.

**means≠ends (binding):** every verdict below is a MEANS. The pointer 0.19110 moves ONLY when a **byte-closed n600
archive** is scored by `upstream/evaluate.py` (contest-CPU AND/OR CUDA, never MPS). A descending realized d_seg is a
surrogate, not a score. This memo certifies that the knobs are deep-math-justified or θ*-flagged; it does not claim a score.

---

## 0. THE HEADLINE (the certification result in one paragraph)

**The axis-to-certify is the PROVEN n200 Muon arm config VERBATIM — except ONE knob (`--mod-dim 32→26`).** I recalled the
live proven argv from `experiments/results/levelset_thetastar_muon_arm/run_muon.log` and its result JSON
(`levelset_train_result.json`, best realized **d_seg 0.0036976 @ ep1000**, the lowest measured on this carrier, **still
monotonically descending** at the final epoch — 0.003805→0.003718→0.003698 over the last 3 verdicts). Of the **19 knobs**
in the ARCH+BASIS+OPTIMIZER axis, **18 are byte-identical to that proven argv** (hidden-96 / n-hidden-4 / hosc / β4 / ω1 /
siren-init / self-orient / n-dir-freqs-2 / freq-across-32 / freq-along-4 / reorient-50 / max-bank-64 / chroma /
palette-anchor / muon-lr-0.002 / muon-momentum-0.95 / muon-ns-5 / grad-clip-1.0 / accum-8). The **one delta is `mod 32→26`**,
a rate refinement. This is the cleanest certification posture possible: the architecture/basis/optimizer axis IS the
existence-proof config, so "OPTIMUM-CONFIRMED" rests on the strongest evidence we have (a measured descent on this exact
carrier), and the single unproven knob is isolated, rate-positive, and capacity-safe.

**Provenance correction (NO-FAKE):** the launch-design doc (`n600_v2_launch_ready_design_*.md` line 83) called the proven
mod-dim "the n200/n96-era **21**." The actual proven argv used **`--mod-dim 32`** (verified in `run_muon.log` +
`front_end: in_feat 88`). So mod-26 is a delta off **32**, not a "correction of 21." This memo treats mod-26 honestly as an
UNPROVEN rate refinement vs the proven 32, not as a recall.

---

## 1. PER-KNOB CERTIFICATION TABLE

Verdict legend: **OPTIMUM-CONFIRMED** (proven-lineage value AND deep-math-justified; ship it) · **REFINE-TO-x** (deep-math
says a different value dominates) · **NEEDS-θ\*** (defensible but unproven; isolate as an A/B arm). Lens key: IT=information
theory · SB=spectral bias · RD=rate-distortion · OG=optimization geometry · SX=signal-basis/sampling.

### A. REPRESENTATION

| Knob | Value | Proven? | First-principles justification (5-lens) | Verdict |
|---|---|---|---|---|
| `--mod-dim` | 26 | proven=**32** | **The per-pair code is the video-derived COUNTED payload** (codes = n_pairs×mod floats ≈ rate-linear in mod; reviewer-measured int8+brotli code = 33553 B @ mod32 → ~27346 @ mod26 → mod32→26 = **ΔS_rate −0.0042**, my arithmetic). **Capacity is NOT binding:** the reviewer's TwoNN+MLE nonlinear intrinsic-dim on the *proven code that hit 0.003698* = **~8–11**, and on the GT partition manifold = **~9** — both ≪ 26 (the "26.33" is a LINEAR participation-ratio OVERCOUNT of a curved manifold). **Whitney embedding** (a smooth m-manifold embeds in 2m+1 linear dims; Whitney 1936): m≈9 ⇒ floor **19**, m≈13 (lane-orbit⊕screw) ⇒ **27** — so mod-26 sits safely in-band, mod-9 would under-embed/collapse. So mod-26 = a rate win at ~zero capacity risk; but the RD-optimum is likely **lower** (see Q1). | **NEEDS-θ\*** (ship 26 for launch; the RD-optimum is the open question; pure-control=32) |
| `--hidden-dim` | 96 | **proven=96** | **The trunk is the rate-costly knob** (params ≈ hidden²·n_hidden; reviewer-measured hidden 96→120 = +21KB base, **+0.010–0.014 ΔS_rate**). The trunk's job (OG/SB) is to *nonlinearly unfold* the low-linear-rank code (linear PR 1.32) into the ~9-dim nonlinear target manifold — and hidden-96 demonstrably does so (it produced 0.003698). Since nonlinear capacity is already adequate at 96, the RD gradient points **down** (toward 88), never up. RD-optimum among measured {96,120} is **96**. | **OPTIMUM-CONFIRMED** (down-only θ* note: 88 rate-curve) |
| `--n-hidden` | 4 | **proven=4** | **Depth = compositional unfolding capacity** for the code→manifold map (a depth-d periodic MLP composes d nonlinear folds; SIREN used 5, HOSC 3–5). 4 balances unfolding power against rate (each layer ≈ +9.2K params @ h96) and against optimization stability through periodic activations (deeper periodic nets are conditioning-sensitive — but Muon's spectral orthogonalization specifically conditions deep 2-D weights, OG). Proven adequate at the achieved descent. | **OPTIMUM-CONFIRMED** |

### B. ACTIVATION

| Knob | Value | Proven? | First-principles justification (5-lens) | Verdict |
|---|---|---|---|---|
| `--activation` | hosc | **proven=hosc** | **The target is a piecewise-constant argmax partition (a step field).** SB: ReLU has strong low-frequency spectral bias → blurs the codim-1 boundary; pure-sine (SIREN) carries the high frequencies but **rings (Gibbs)** at the step, and the ringing *causes annulus flips* (the residual). **HOSC = tanh(β·sin(ωu))** (Serrano et al. 2024, arXiv:2401.10967) interpolates sine→square-wave as β grows: it is the **topology-matched chart for a step target** — sharp-feature-preserving with saturation that suppresses Gibbs. Our in-repo A/B is the only descent evidence: **hosc 0.221 vs wire 0.265** (config-review #3). Matches the published HOSC>SIREN/FINER on sharp-feature tasks. | **OPTIMUM-CONFIRMED** |
| `--hosc-beta` | 4.0 | **proven=4.0** | β is the **sharpness/saturation control**: β→∞ ⇒ step-native (O(1) params/edge, L∞-at-edge optimal, no Gibbs); β→0 ⇒ linear. β=4 is the published/our default sweet spot (trainable from scratch with SIREN-init; too-large β at init kills gradients). **But β is a STATIC value of an UNSWEPT lever** — `--hosc-beta-end` anneals β4→8 (FEED-fb), the deep-math-predicted step-sharpening homotopy (sister of the softmax-temp anneal). | **OPTIMUM-CONFIRMED for static; NEEDS-θ\*** for β-anneal (top-3 refinement #2) |
| `--hosc-omega` | 1.0 | **proven=1.0** | ω is the **base angular frequency** of the periodic activation (SIREN's ω₀). ω=1 with SIREN-init keeps pre-activations ~unit-variance Gaussian (Sitzmann 2020, arXiv:2006.09661) → stable from-scratch training. The high-frequency content is carried by the **input Fourier/curvelet bank** (freq-across 32), not by ω, so ω=1 correctly avoids double-counting frequency and avoids the SIREN "spectral-bottleneck" misalignment failure. | **OPTIMUM-CONFIRMED** |
| `--siren-init` | on | **proven=on** | **The principled init** (Sitzmann 2020): pre-activations stay unit-variance Gaussian through periodic layers ⇒ deep periodic MLP trains from scratch. Existence proof: the proven arm used siren-init+hosc and did NOT stall (pretrain disagree 0.00313). Without it, periodic nets are init-fragile. | **OPTIMUM-CONFIRMED** |

### C. DIRECTIONAL FOURIER BASIS (+ chroma)

| Knob | Value | Proven? | First-principles justification (5-lens) | Verdict |
|---|---|---|---|---|
| `--self-orient` | on | **proven=on** | **Anisotropy matched to the boundary tangent field is THE #1 measured d_seg lever (−48% n600, ~0 byte).** SX: an argmax boundary is a codim-1 curve; the optimal atom to resolve it is **anisotropic** — high frequency ACROSS the edge (normal), low ALONG it (tangent) — exactly the **curvelet/parabolic-scaling** prior (Candès–Donoho). `--self-orient` orients the feature frame to the *measured* local boundary normal (per-pair argmax reorient). Caveat (Q4): help says "needs a roughly-learned base"; from-scratch the early reorients (ep<300) orient against a forming partition — `--structured-init` + the openpilot lane-prior give it a base, so this is de-risked but not zero-risk. | **OPTIMUM-CONFIRMED** (the decisive lever) |
| `--n-dir-freqs` | 2 | **proven=2** | Number of directional frequency rings per oriented atom. 2 (vs default 6) is the proven rate-lean choice: the normal-direction step needs few rings once the frame is oriented (the orientation does the work, not a dense ring bank). IT/RD: extra rings = bytes + over-Nyquist waste (LEVER-2). | **OPTIMUM-CONFIRMED** |
| `--freq-across` | 32 | **proven=32** | **Normal-direction bandwidth.** SX: a sharp edge wants maximal observable normal bandwidth. The hard ceiling is the SegNet stem Nyquist **64** (below), but the **through-R operator** (bicubic↑384→874→uint8→bilinear↓512) low-passes the witness before the scorer — so the *post-R effective* normal Nyquist is < 64, and freq-across 32 (half stem-Nyquist) is plausibly matched to it. This is the one basis knob where the proven value may be sub-RD-optimal in EITHER direction (more bandwidth if R preserves it; less if it aliases). | **OPTIMUM-CONFIRMED for launch; NEEDS-θ\*** (freq-across ∈ {24,32,48}, post-R-Nyquist arm — top-3 refinement #3) |
| `--freq-along` | 4 | **proven=4** | **Tangent-direction bandwidth.** Low along the edge = the edge is locally straight at atom scale (parabolic/curvelet width~length²). The across:along ratio **32:4 = 8:1** ≈ 3 octaves of anisotropy. SX: optimal anisotropy for a C² curve is curvature-bounded (atom must stay straight: L·κ ≲ w); 8:1 is a reasonable global compromise but a single ratio cannot be locally optimal for a curved boundary (true curvelets are multi-scale-anisotropic). | **OPTIMUM-CONFIRMED** (multi-scale anisotropy = a v2 build, not a knob) |
| `--reorient-every` | 50 | **proven=50** | Reorientation cadence (epochs). OG: the boundary frame is a slowly-varying field; 50-epoch reorient tracks it without thrashing the optimizer. Proven; low sensitivity. | **OPTIMUM-CONFIRMED** |
| `--max-bank-freq` | 64 | **proven=64** | **LEVER-2 = the scorer-matched anti-alias / rate ceiling.** IT (data-processing/sampling): SegNet's EfficientNet-B2 stem is stride-2 on a 512-wide resize ⇒ Nyquist = SEG_W/(4·stem_stride) = **64 cyc/unit**. Atoms above 64 are *invisible to the argmax* (aliased) → pure wasted bytes + aliasing noise. Capping at 64 = matched-filter to the scorer's observable band. With the current bank (max ~32) it is near-no-op but principled (a guard against future bank widening). | **OPTIMUM-CONFIRMED** |
| `--chroma` | on | **proven=on** | **SegNet reads RGB ⇒ its argmax depends on chroma ⇒ chroma is a genuine d_seg actuator** (operator "Chroma too"). The seg-frame has RGB-slack; routing capacity into chroma where it flips the boundary annulus is free d_seg. Secondary: PoseNet reads YUV6 (but pose rides the stored sidecar). | **OPTIMUM-CONFIRMED** |
| `--palette-anchor` | on | **proven=on** | **Diagnosed fix**: init the learnable palette to per-class mean GT RGB (logged means e.g. Road [45,45,37], Undriv [24,13,14]) → breaks the ~0.51 luma-ramp plateau (a generic ramp init lands the render in a SegNet-confusing basin). OG: a good basin init. | **OPTIMUM-CONFIRMED** |

### D. OPTIMIZER (Muon finisher + grad discipline)

| Knob | Value | Proven? | First-principles justification (5-lens) | Verdict |
|---|---|---|---|---|
| `--muon-lr` | 0.002 | **proven=0.002** | **Muon is the measured d_seg finisher** ("Muon is THE drop"). OG: Muon (Jordan et al. 2024; Newton-Schulz-orthogonalized momentum) replaces the update with the **nearest orthogonal matrix** to the momentum → every singular direction of the 2-D hidden weights advances equally → it **conditions** the late-stage descent that AdamW cannot (the spectral-norm-controlled step). Because Muon normalizes to ~unit spectral norm, `--muon-lr` IS a spectral step size; 0.002 ∈ the published optimal-form band (help: 1e-3–5e-3) and is the EXACT proven value (run_muon.log + `muon_finisher_switch` JSON). The omitted-default 1e-4 would be 20× too low (the review's CRITICAL catch). | **OPTIMUM-CONFIRMED** (the launch design's CRITICAL revision, now correct) |
| `--muon-momentum` | 0.95 | **proven=0.95** | Keller-Jordan default; high momentum (0.95) is standard for the orthogonalized buffer. Proven. | **OPTIMUM-CONFIRMED** |
| `--muon-ns-steps` | 5 | **proven=5** | Newton-Schulz iteration count = the orthogonalization accuracy/cost knob. 5 (Keller-Jordan default) reaches near-orthogonal with the tuned quintic coefficients. Lower = under-orthogonalized (worse conditioning); higher = wasted compute (score>time allows it, but no accuracy gain past ~5). | **OPTIMUM-CONFIRMED** |
| `--grad-clip` | 1.0 | **proven=1.0** | Stabilizes the stage-transition spikes (the spike-guard re-treats at every boundary). OG: a standard trust-region cap; proven through 4 stages. | **OPTIMUM-CONFIRMED** |
| `--accum-pairs` | 8 | **proven=8** | Gradient accumulation over 8 pairs = the effective batch for the per-pair-code + shared-decoder geometry. IT: larger batch = lower-variance gradient (helps the late critical-slowing tail); 8 is proven and K=1 per-pair scorer-forward is compute-bound, so 8-accum is the throughput/variance sweet spot. | **OPTIMUM-CONFIRMED** |

---

## 2. THE DECISIVE OPEN SUB-QUESTIONS (answered honestly)

### Q1. Is `--hidden-dim 96` the RD-optimum at n600 (vs 88/104/120)?
**Yes among measured, and the RD gradient points DOWN, not up.** Reviewer measured 96 vs 120: hidden-96 is a **+0.010–0.014 S
rate WIN** over 120, and the nonlinear-ID (~9) shows hidden-96 trunk capacity already covers the manifold (existence proof:
it hit 0.003698). 88/104 are unmeasured, but since capacity is non-binding the only RD-improving direction is **downward**
(88 is a candidate rate-curve arm; 104/120 are dominated — they buy trunk capacity the ~9-dim manifold does not need).
**Verdict: hidden-96 OPTIMUM-CONFIRMED; the only RD experiment worth running is 88 (rate-down), not 104/120.**

### Q2. Is `--mod-dim 26` vs ~19–21 a real rate win worth a θ\* arm? **YES — this is the #1 refinement.**
The code is the counted video-derived payload and is ~rate-linear in mod-dim. My arithmetic (extrapolating the reviewer's
MEASURED int8+brotli code bytes linearly — flagged ESTIMATE):

| mod-dim | est. archive B | est. S_rate | ΔS vs proven 32 | ΔS vs 26 | capacity status |
|---:|---:|---:|---:|---:|---|
| 32 (proven) | 96 828 | 0.06447 | — | +0.0042 | proven |
| **26 (GO)** | ~90 537 | ~0.06028 | **−0.0042** | — | safe (≫ ID 9; in Whitney band) |
| **21** | ~85 294 | ~0.05679 | −0.0077 | **−0.0035** | Whitney floor for m≈9–10 (2m+1≈19–21) |
| 19 | ~83 197 | ~0.05540 | −0.0091 | −0.0049 | aggressive Whitney floor (2·9+1) |
| 16 | ~80 052 | ~0.05330 | −0.0112 | −0.0070 | BELOW Whitney floor → under-embedding risk |

Because **RATE is the binding sub-0.15 lever** and capacity has ~3× headroom (ID 9 vs mod 26), **mod-19–21 is a defensible
~0.003–0.005 S rate win.** mod-26 is the *safe* first launch; **mod-21 (the Whitney floor for the measured m≈9–10) is the
RD-optimum candidate**, mod-19 the aggressive edge, mod-16 the under-embedding red line. **Verdict: run a mod ∈ {19,21,26}
rate-curve θ\* arm; mod-21 is my predicted RD-optimum.** This is the single highest-EV certification finding.

### Q3. Is the Fourier bank (freq-across 32 / freq-along 4 / max 64) optimal for the all-class boundary tangent field?
**Topologically yes; quantitatively the across-bandwidth is the one open value.** The *form* (anisotropic, oriented,
normal-high/tangent-low, Nyquist-capped) is the correct curvelet prior for a codim-1 argmax boundary and is the proven −48%
lever. The *magnitudes* are proven but only one is RD-sensitive: **freq-across 32**. Its true ceiling is not the stem Nyquist
64 but the **post-R effective Nyquist** (the bicubic/uint8/bilinear R chain low-passes the witness before SegNet). If R
preserves >32 normal cycles, freq-across could rise (sharper edge); if R aliases, 32 may already overshoot. **Verdict:
freq-across ∈ {24,32,48} is a worthwhile θ\* arm gated on a $0 post-R-bandwidth measurement; freq-along/max-bank/n-dir/
reorient are settled.** The genuinely better basis (locally-curvature-adaptive multi-scale anisotropy = true curvelets) is a
v2 BUILD, not a knob.

---

## 3. RANKED: SETTLED vs θ\*-WORTHY

**SETTLED (OPTIMUM-CONFIRMED — ship as-is; do not spend an arm):**
`--hidden-dim 96` · `--n-hidden 4` · `--activation hosc` · `--hosc-omega 1.0` · `--siren-init` · `--self-orient` ·
`--n-dir-freqs 2` · `--freq-along 4` · `--reorient-every 50` · `--max-bank-freq 64` · `--chroma` · `--palette-anchor` ·
`--muon-lr 0.002` · `--muon-momentum 0.95` · `--muon-ns-steps 5` · `--grad-clip 1.0` · `--accum-pairs 8`.
(17 knobs = the existence-proof config; their justification is "this exact value produced d_seg 0.003698" + the deep-math
above.)

**θ\*-WORTHY (deserve a rate/capacity A/B arm, ranked by EV = ΔS-likely / risk):**
1. **`--mod-dim` 26 → {21,19} rate-curve** — predicted RD-optimum mod-21 (Whitney floor m≈9), **~−0.0035 S** at ~zero
   capacity risk. *Highest EV; pure rate.*
2. **`--hosc-beta-end` (β4→8 anneal)** — the UNSWEPT step-native homotopy; targets the d_seg residual (annulus Gibbs), not
   rate. *High EV; byte-free; bit-identical when off.*
3. **`--freq-across` 32 → {24,48}** — gated on a $0 post-R normal-Nyquist measurement; could be ±d_seg or ±rate. *Medium EV.*
4. **`--hidden-dim` 96 → 88 rate-down** — only if mod-21 lands and more rate is wanted; small (~−0.005 S) but capacity-riskier
   than mod. *Lower EV; do AFTER mod.*

**NOT a knob (v2 BUILDS, flagged so they aren't mistaken for config):** multi-scale locally-anisotropic curvelets ·
the se(3) screw-warp temporal/rate factor · root-tracking anneal scheduler · LoRA/DoRA rank-~8 annulus re-treatment.

---

## 4. TOP-3 REFINEMENTS I WOULD DEFEND AT ICLR REVIEW

1. **mod-dim is the rate lever, and the RD-optimum is the Whitney floor of the MEASURED nonlinear intrinsic dimension, not
   the linear participation ratio.** The codebase chose mod-26 from a *linear* eff-dim (26.33); the capacity-relevant
   quantity is the *nonlinear* ID (TwoNN/MLE ~9; Facco et al. 2017; Levina–Bickel 2004), and a curved m-manifold needs only
   2m+1 linear coordinates to embed (Whitney 1936) ⇒ mod ≈ 2·9+1 = **19–21** is sufficient and **~0.003–0.005 S cheaper**.
   *Defense:* an INR per-pair code is a linear chart of a nonlinear manifold; sizing it by linear PR systematically
   over-pays rate. This converts "auto-config from eff-dim" into "auto-config from **intrinsic** dim + Whitney," a
   principled rate-optimal sizing rule.
2. **β-annealed HOSC realizes a step-native homotopy that the static-β config leaves on the table.** HOSC = tanh(β·sin)
   →square-wave as β→∞ (Serrano et al., arXiv:2401.10967; the Jan-2026 saturation-control follow-up arXiv:2601.07870 adds an
   explicit Lipschitz/saturation knob — direct external validation of the anneal direction). Annealing β4→8 in lock-step
   with the softmax-temp anneal sharpens the activation toward the piecewise-constant argmax target exactly as the partition
   pins, attacking the Gibbs-ringing component of the annulus residual with **zero added bytes** and bit-identical-when-off
   safety. *Defense:* the target is a step field; the activation should converge to a step basis on the same homotopy that
   sharpens the decision temperature.
3. **The directional basis should be band-matched to the SCORER's observable spectrum AFTER the round-trip operator, not to
   the raw stem Nyquist.** `--max-bank-freq 64` correctly matches the SegNet stem Nyquist (data-processing inequality: atoms
   above 64 cyc/unit cannot change the argmax), but the binding bandlimit for `--freq-across` is the **post-R effective
   Nyquist** (bicubic↑/uint8/bilinear↓). A $0 measurement of the witness's surviving normal-direction bandwidth through R
   converts freq-across from a guessed 32 into a matched-filter value — the inverse-steganalysis / coding-for-machines
   principle (spend bits only where the receiver can detect them) applied to the basis bandwidth.

---

## 5. HONEST PROVENANCE (measured vs estimated)

- **MEASURED (real artifacts):** the entire 18-knob proven axis (recalled verbatim from `run_muon.log` + verified against the
  116-flag argparse — no invented flag); best realized **d_seg 0.0036976 @ ep1000, still descending** (`levelset_train_result.json`);
  hosc>wire 0.221/0.265 (config-review #3); directional −48% (frontier lever #1); reviewer's nonlinear-ID ~9 (TwoNN/MLE) and
  byte-close 32/96=96828 / 26/120=111902.
- **ESTIMATED (flagged):** the mod-21/19/16 archive-byte rows (linear extrapolation of the reviewer's MEASURED mod-32 code
  bytes; the base bytes are mod-independent so the extrapolation is well-grounded but unmeasured at those mod values);
  post-R effective Nyquist (un-measured — the gating $0 experiment for refinement #3).
- **EXTERNAL CITATIONS:** HOSC — Serrano et al. 2024 [arXiv:2401.10967](https://arxiv.org/abs/2401.10967) + saturation-control
  follow-up [arXiv:2601.07870](https://arxiv.org/html/2601.07870v1); SIREN/init — Sitzmann et al. 2020
  [arXiv:2006.09661](https://arxiv.org/abs/2006.09661); Muon — Jordan et al. 2024 ([kellerjordan.github.io/posts/muon](https://kellerjordan.github.io/posts/muon/),
  Newton-Schulz orthogonalization, spectral-norm LR transfer); Whitney embedding (Whitney 1936, 2m+1); nonlinear intrinsic
  dimension — Facco et al. 2017 (TwoNN), Levina–Bickel 2004 (MLE); anisotropic/curvelet basis — Candès–Donoho (parabolic
  scaling width~length²).
- **NOT a score:** `[macOS-MLX/numpy advisory · NON-PROMOTABLE]`. **pointer 0.19110 UNMOVED.** Every verdict is a MEANS; the
  only END is a byte-closed n600 exact row < 0.19110 (CPU/CUDA, never MPS).

## 6. CERTIFICATION VERDICT
**The ARCH+BASIS+OPTIMIZER axis is CERTIFIED for the 1000-epoch n600 burn**, on the strongest available basis: 17 knobs are
the exact existence-proof values (d_seg 0.003698) with deep-math backing, `--mod-dim 26` is a capacity-safe rate refinement
(ship for launch; mod-21 is the predicted RD-optimum for a follow-up rate-curve arm), and the three open values
(mod-dim, β-anneal, freq-across) are isolated as ranked θ* arms that do NOT block the launch. No knob is cargo-culted; no flag
is invented; the one launch-blocking error the review caught (muon-lr) is fixed.
