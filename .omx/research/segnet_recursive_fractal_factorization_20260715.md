# SegNet recursive-fractal factorization — what the FROZEN WEIGHTS reveal that we didn't know

**Date:** 2026-07-15/16 UTC · **Operator P0:** *"Does flattening and factorization reveal anything about
the segnet implementation that we don't know yet? Need deep recursive fractal analysis."*
**Authority:** `[macOS-CPU advisory]` frozen CPU-torch fp32 on the REAL weights
(`upstream/models/segnet.safetensors`, sha256 `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6`),
exact upstream architecture (`smp.Unet('tu-efficientnet_b2', classes=5, activation=None)`, modules.py:103-113),
exact upstream preprocess (modules.py:108-109). **Authority anchor:** my rig reproduces the cached n96 GT
BIT-EXACTLY — 0 mismatched argmax pixels / 96 frames and max-abs margin error 0.0 vs
`gt_n96.npz` (stage_b1.json). `research_only=true`; `score_claim=false`; pointer 0.19108 UNMOVED (MEANS —
this is the white-box the c2 levers optimize against).

**Artifacts (all MEASURED numbers below come from these; dir gitignored, sha256 custody here):**
`experiments/results/segnet_fractal_20260715/` — stage_a.json `2e75209f…5615`, stage_b1.json
`ff03153f…67b2`, stage_b2.json `70829147…02ba`, stage_c.json `6d6478f5…25ac` + the four generating
scripts in the same dir (deterministic: rerun regenerates from the frozen weights + gt_n96 cache).
Scope of through-net rows: n96 real GT frame_1 (subsets stated per row).
**Canonical equation registered:** `segnet_head_rank4_linear_flipdist_v1`
(`tac.canonical_equations.segnet_head_rank4_flipdist_20260715`, 7 tests green).

## 0. ALREADY KNOWN (findings below are NET of this)

1. Stride-2 stem Nyquist / structurally HF-blind (argmax-control-clues memo + Yousfi WIFS 2020 surgery point).
2. 3.2× along-tangent spectral deficit (4-lens, L65); d_seg is boundary-flip not texture (spectral atlas).
3. ~4.7%-area annulus carries ~97% of d_seg (#333); "SegNet sees regions, not pixels" (qualitative).
4. comma10k class order [Road, Lane, Undrivable, Movable, MyCar] (measured, CLAUDE.md).
5. Exact scored-chain factorization treating N_seg as ONE opaque nonlinearity
   (`frozen_scorer_exact_factorization_20260715.md`: A_seg≡A_pose, frame_0 seg-free, blind subspaces B1-B7).
6. Pre-SE locus family kill for cheap single-source feature localization + upstream-SE non-tileability (#484).
7. 1-thread exact-forward 3× speedup; CPU fp16 flips 11.1% / bf16 0.45% of pixels (pair-0) (#456).
8. Tropical/Laguerre/power-diagram FORMULATION of the argmax (#284) — until now not grounded in the actual weights.

## 1. The recursive factorization (L0 → L4)

```
L0  d_seg = mean 1[argmax N(A·x) ≠ argmax N(A·x*)]           (opaque N — the PRIOR memo's stopping point)
L1  N = HEAD ∘ DECODER ∘ ENCODER
      HEAD    = Conv2d(16→5, k=3, activation=None)            ← LINEAR. The partition is a hyperplane
                                                                 arrangement in 144-dim patch space (§2)
      DECODER = 5 U-net blocks (256,128,64,32,16ch), skips from encoder at strides [16,8,4,2],
                **final stride-1 block has NO skip** (conv1_in=32 measured)   (§5)
      ENCODER = EfficientNet-B2: stem s2 → 7 stages, reductions [2,4,8,16,32], out ch [16,24,48,120,352]
L2  each stage = MBConv blocks = expand-1x1 ∘ BN ∘ depthwise ∘ BN ∘ SE ∘ project-1x1 ∘ BN
L3  depthwise kernels = per-channel FIR filters (14,784 of them — spectra measured §6)
      SE = 23 global gates with 4-24-dim bottlenecks (dynamics measured §4)
      BN = per-channel affine (fold: 0 dead channels of 33,368 — §7)
L4  argmax = tropical decision over 5 linear functionals of the penultimate map (§2-3)
```

## 2. NEW №1 — the head is EXACTLY rank-4 linear: the argmax partition is a fixed 4-dim hyperplane arrangement (MEASURED)

`segmentation_head.0.weight` is `(5,16,3,3)` + bias, `activation=None` ⟹ the 5 logits at a pixel are a
FIXED LINEAR functional of the 144-dim (16ch × 3×3) penultimate patch. Argmax depends only on differences ⟹
the centered head governs everything; its measured singular values are **[3.128, 2.154, 2.025, 1.796, 0]** —
**exactly rank 4** (rank-4 reconstruction max-abs err 5.96e-8 = fp32 floor), and WELL-CONDITIONED (σ1/σ4 = 1.74).

- **The entire 5-class partition at every pixel is decided by a 4-dim linear projection** of the penultimate
  patch. The tropical/Laguerre formulation (#284) is now GROUNDED in the actual frozen weights: 10 class-pair
  hyperplanes `⟨w_c−w_c', f⟩ + (b_c−b_c') = 0`, normals measured:

  | pair | ‖Δw‖ | pair | ‖Δw‖ |
  |---|---|---|---|
  | **Lane-Movable** | **4.007** | Undrivable-Movable | 2.946 |
  | **Road-Lane** | **3.953** | Road-Movable | 2.942 |
  | **Lane-MyCar** | **3.862** | Movable-MyCar | 2.910 |
  | **Lane-Undrivable** | **3.748** | Undrivable-MyCar | 2.869 |
  | Road-MyCar | 2.705 | Road-Undrivable | 2.602 |

  **ALL four Lane rows are the four largest normals (3.75-4.01 vs 2.60-2.95 for every non-Lane pair).** The
  frozen net AMPLIFIES the Lane direction: per unit penultimate perturbation, Lane boundaries move the most.
  The mutual angles of the 10 normals span 25.8°-90° (median 62°) — no two boundaries are near-parallel.
- Head spatial structure: 65.9% of centered-head energy is the DC (sum-over-3×3) part — the head is roughly a
  1×1 classifier on 3×3-box-filtered features; 24.2% sits on the center tap; ~34% is spatial-derivative-like.
- Penultimate features are ReLU outputs: 53.6% exactly zero, typical activation magnitude 0.47 — the
  arrangement lives in the non-negative orthant.

## 3. NEW №2 — closed-form minimal argmax-flip (MEASURED; canonical equation registered)

Because the head is linear, the minimal L2 perturbation of the local penultimate patch that flips the
(c,c') ordering at a pixel is EXACT (not first-order): **d_flip = |margin_cc'| / ‖w_c−w_c'‖**. Measured at
real boundary pixels (4-neighbor label disagreement, 8 frames of n96):

| pair | boundary px | median logit margin | median feature-space flip dist | p10 |
|---|---|---|---|---|
| Undrivable-Movable | 916 | 0.24 | **0.081** | 0.021 |
| Road-Movable | 1011 | 0.31 | 0.106 | 0.017 |
| Road-Lane | 8812 | 0.52 | 0.131 | 0.024 |
| Road-MyCar | 4033 | 0.46 | 0.171 | 0.033 |
| Road-Undrivable | 3289 | 0.46 | 0.177 | 0.031 |

All small vs typical activation 0.47: **a ~0.08-0.18 L2 nudge in one 144-dim patch flips a boundary pixel —
exactly computable from the frozen head, no surrogate, no training.** Pixel-space pullback (first-order,
45 boundary pixels, 3 frames): median minimal L2 = **8.8** in 0-255 input units spread over the whole
(384,512) frame — i.e. RMS ≈ 0.014/pixel, FAR below 1 uint8 LSB. **Boundary flips are not gradient-limited;
they are REALIZATION-limited** (uint8 lattice + through-R survival + spatial-locality constraints) — the
sharp statement of why d_seg optimization is a coding problem, not a sensitivity problem.
→ Registered: `segnet_head_rank4_linear_flipdist_v1`.

## 4. NEW №3 — SE global context: tiny, half-FROZEN, and a STABILIZER (hypothesis 4 REFUTED in the exploit direction) (MEASURED)

23 SE modules, bottlenecks only 4-24 dims (blocks.1.0: 96→4). Measured across ALL n96 real frames:

- **Every stage-ENTRY (downsample) SE is essentially FROZEN on this video distribution:** blocks 0.0/1.0/2.0/3.0
  cross-frame gate std_med 2.0e-4-9.5e-4 (range_max ≤ 0.034). Mid blocks drift moderately; only deep SEs
  (blocks 5.1-6.1) genuinely vary (std_max up to 0.23, range up to 0.87).
- **Freezing gates to their n96 means** (replace SE by a constant per-channel gain): blocks 0-2 frozen →
  flip rate **3.3e-4**; blocks 0-3 → 4.3e-4; ALL 23 frozen → 1.9e-3. So the shallow "global context" is
  worth only ~3e-4 of argmax; a shallow-SE-frozen surrogate is a certified-cheap local CNN approximation.
- **Brightness +4 (global, uint8): the TRUE net flips only 13-29 px/frame (~1e-4); with gates frozen at the
  base frame's values it flips 69-149 px/frame (4-6× MORE).** The SE pathway actively CANCELS global
  illumination shifts. The "cheap global multiplicative attack via SE" hypothesis is **REFUTED** —
  verdict_scope: formulation (global brightness/gain perturbations, this video's distribution); SE remains
  an amplifier candidate only for inputs far outside that distribution.
- Reconciliation with #484 (pre-SE locus kill): the 4 upstream SE reductions that killed block2-pre-SE strict
  tileability are exactly the near-frozen shallow gates measured here. **Constant-gate approximate tileability
  (certified ~3.3e-4 flip cost) is a REOPEN path** the #484 verdict explicitly left open ("explicitly
  cached/donated SE-gate broadcast") — now with the measured price tag.

## 5. NEW №4 — the stride-2 skip is the 16-channel Lane bottleneck; the last decoder block has NO skip (MEASURED)

Decoder in-channels measured from the weights: blocks get skips from encoder strides [16,8,4,2] and the
final stride-1 block takes **conv1_in=32 = decoder-only, NO skip** (the stride-1 identity is unused).
⟹ **Nothing in the net sees full-res (384,512) pixels after the stem: all sub-stride-4 boundary
localization flows through the ONE stride-2 skip — 16 channels at (192,256).**

Skip-ablation (destroy sub-stride-4 detail in a skip via down-up 2×, 16 frames):

| ablated | total flips | flip rate | top pair |
|---|---|---|---|
| stride-2 skip detail | 8,072 | 2.6e-3 | **Road-Lane 6,205 = 77%** |
| stride-4 skip detail | 8,682 | 2.8e-3 | Road-Lane 3,764 = 43% (spread) |

**Lane boundary precision rides on the stride-2 skip far out of proportion (77% of induced flips), and that
skip is only 16 channels.** This is the measured "boundary-sharpness ceiling per class pair" (hypothesis 7
confirmed): Lane is THE skip-limited pair; Road-Undrivable/MyCar horizon-scale boundaries barely care.

## 6. NEW №5 — depthwise spectra: 74% low-pass, and deep oriented kernels are AXIS-ALIGNED (diagonal bins empty) (MEASURED; causal link = HYPOTHESIS)

All 14,784 depthwise kernels FFT'd: **74.3% low-pass, 8.6% oriented, 4.4% high-pass, 12.8% mixed.**
Per-stage structure:

- High-pass/oriented mass lives in stages 0-2 (strides 2-8). Deep stages are almost purely low-pass
  (blocks.5.0: 99% LP; 6.0: 95% LP).
- **The oriented kernels' frequency-orientation histograms in stages 3-6 are AXIS-ALIGNED: horizontal/vertical
  bins dominate and the diagonal bins are near-EMPTY** (blocks.4.1: [21,0,9,0,20]; 5.1: [27,1,14,0,34];
  6.1: [80,0,44,0,81] over bins [-90,-67.5,-22.5,22.5,67.5,90]°). Diagonal-tuned kernels exist only in
  stages 1-2 (blocks.1.2: [8,11,9,12,9]; 2.x similar).
  **Candidate frozen-kernel mechanism for the 3.2× along-tangent deficit:** perspective lane edges are
  predominantly diagonal in the image; beyond stride 8 the net simply has (almost) no diagonal-tuned filters,
  so diagonal fine structure must be carried positionally by low-pass channels. Spectra = MEASURED;
  the deficit attribution = HYPOTHESIS (needs a through-R oriented-grating probe to close).
- **Anti-aliasing is inhomogeneous:** the deep downsamplers self-antialias (blocks.3.0 94% LP, 5.0 99% LP at
  stride 2) but **blocks.1.0 (stride → 4) is only 53% low-pass with 35% oriented + 10% high-pass** — the
  stride-4 stage aliases oriented content. Combined with the unsurgered stride-2 stem, sub-pixel boundary
  placement sensitivity (the #333 boundary-jitter phenomenon) has its sharpest injection point at strides 2-4.

## 7. Hypothesis scorecard (all 8 from the P0 prompt)

| # | hypothesis | verdict |
|---|---|---|
| 1 | head linear ⟹ hyperplane arrangement | **CONFIRMED + sharpened**: k=3 not 1×1; exactly rank-4; Lane normals largest (§2) |
| 2 | closed-form minimal flip | **CONFIRMED in feature space (exact), measured; pixel-space first-order sub-LSB ⟹ realization-limited (§3)** |
| 3 | BN-fold reveals dead channels | **REFUTED (pruning payoff): 0 dead / 33,368 channels** — the net is fully live; fold itself trivially valid |
| 4 | SE ⟹ cheap global perturbation lever | **REFUTED (formulation: global gain/brightness, this distribution)** — SE is a stabilizer (4-6× flip reduction); shallow SEs frozen (§4) |
| 5 | depthwise spectra → low-pass culprits | **CONFIRMED (74% LP) + NEW axis-alignment structure; 3.2× attribution = open HYPOTHESIS (§6)** |
| 6 | effective receptive field number | **MEASURED**: analytic encoder RF 1183px > frame; empirical margin-grad: r50 ≈ 50-160px (median ~85), r90 ≈ 206-424px, 46-74% of grad mass beyond 65px (45 boundary px, 3 frames). "Sees regions" ⟹ half the sensitivity comes from >85px away |
| 7 | skip resolution → boundary ceiling per pair | **CONFIRMED + quantified**: final block skipless; stride-2 skip = 16-ch bottleneck; Lane = 77% of skip-detail flips (§5) |
| 8 | per-pair separability, Lane = hard manifold | **MEASURED**: boundary-patch local dim is LOW for all pairs (dim90 = 2-6 of 144); Road-Lane is the HIGHEST (dim90=6, dim99=25, n=960); Undrivable-Movable lowest (dim90=2). Lane is the most feature-diverse boundary but still low-dim — a small parametric code steers it |

## 8. RANKED new-exploitable findings → c2 levers / solvers

1. **Closed-form flip solver at the head (§2-3)** → margin-residual #226 / MC-finisher #396: per-pixel exact
   targets `Δf = −(margin+ε)·(Δw/‖Δw‖²)` in the 4-dim head projection. No learned surrogate at the head layer,
   ever — the arrangement is known in closed form. (Equation `segnet_head_rank4_linear_flipdist_v1`.)
2. **Stride-2-skip supervision (§5)** → a 16-channel, (192,256) white-box target that owns 77% of Lane
   boundary precision: train/steer the witness render against its imprint on the stride-2 skip (16-dim per
   position, cheapest exact sub-target in the net) instead of full logits. Candidate new DSL lever.
3. **Shallow-SE-frozen certified surrogate (§4)** → training-gradient device: replace blocks 0-2 SEs with the
   constant mean gates (measured cost 3.3e-4 flip rate) ⟹ shallow encoder becomes purely LOCAL ⟹ tileable
   feature banking + partial #484 reopen (constant-gate variant), plus cheaper in-loop scorer forwards.
   NEVER a verdict device (verdicts stay exact).
4. **Axis-aligned-orientation gap (§6)** → sharpens the basis/boundary-placement discipline: the scorer cannot
   read diagonal fine structure beyond stride 8; witness capacity spent on diagonal HF along lanes is spent on
   a signal the deep net integrates only positionally. Feeds the curvelet/orientation lever ordering; owed
   probe: oriented-grating through-R response by angle.
5. **Realization-limited boundary control (§3)** → reframes d_seg optimization: gradients are enormous
   relative to margins (sub-LSB first-order flips); binding constraints are uint8 + R-survival + locality.
   Supports geometry/placement levers over amplitude levers, now with numbers.
6. **ERF budget (§7 row 6)** → any local finisher edit leaks: ~50-65% of a boundary pixel's margin
   sensitivity lies >65px away — MC-finisher acceptance tests must check nonlocal spillover at ≥r90 (~300px).

## 9. verdict_scope on negatives

- H3 (dead channels): negative at the BN-gamma pruning formulation only; other compression families untouched.
- H4 (SE exploit): negative for GLOBAL gain/brightness perturbations on this video's distribution;
  spatially-structured or out-of-distribution inputs unmeasured.
- All through-net rows: n96 real frames (subsets as stated), macOS-CPU fp32 advisory; NOT contest-CPU/CUDA
  authority; no score claims. n600 generalization owed where a lever consumes these numbers quantitatively.

## STORES CONSULTED

CLAUDE.md; AGENTS.md; docs/operating_manual_craft_handoff.md; frozen_scorer_exact_factorization_20260715.md;
pre_se_locus_20260713.md; codex_findings_frozen_segnet_exact_forward_20260713_codex.md;
segnet_argmax_control_clues_20260629T233214Z.md; codex_findings_rate_op2_tropical_boundary_20260518T232212Z_codex.md;
graph-memory recall ("segnet frozen weights factorization"); gt_n96 cache; upstream/modules.py (pinned, read-only).
