# Flowers warp mechanism — deep dive + assessment vs OUR 5 warp surfaces (2026-07-09)

**Task:** P0 deep-research — assess the "Flowers" warp mechanism (arXiv 2603.04430, Muser/Spitzer/Lassas/
de Hoop/Dokmanić — "Flowers: A Warp Drive for Neural PDE Solvers") as a concrete cheaper/sharper warp for
OUR stack. Read: abstract, HTML method (arxiv.org/html/2603.04430v1), project page (t-muser.github.io/flowers),
and the ACTUAL released code (`flower_standalone.py` 887 lines + `flowers/models/utils/grid_samples.py` +
repo tree via GitHub API). Labels: **MEASURED** (from our runs) / **DERIVED** (from paper+code, verifiable) /
**DESIGNED** (my proposal, unmeasured).

Pointer 0.19110 UNMOVED. Nothing heavy/paid launched. `experiments/results/levelset_n600_witness_*` untouched.

---

## 1. The PRECISE mechanism (DERIVED — read from the real code, not the abstract)

The entire "warp" is one module, `SelfWarp` (verbatim structure from `flower_standalone.py:426-489`):

```python
self.flow_head = nn.Sequential(          # LEARNED, pointwise
    Conv{2,3}d(in_channels, out_channels, kernel_size=1),
    nn.ReLU(inplace=True),
    Conv{2,3}d(out_channels, n_spatial_dims * num_heads, kernel_size=1),
)
self.value_head = Conv{2,3}d(in_channels, out_channels, kernel_size=1)   # LEARNED
# forward(u):
flow  = self.flow_head(u)                # per-head displacement field δ^(h)(x), pointwise
value = self.value_head(u)               # "mixed" features v = f[u]
grid  = base_grid + flow                 # base_grid = meshgrid in [-1,1]  (Id + δ)
u_warp = grid_sample(value, grid, mode='bilinear', align_corners=True)   # v(x + δ(x))
```

Precise answers to the four sub-questions:

- **How is the displacement field parametrized/predicted?** A *tiny 2-layer 1×1-conv MLP* (`flow_head`),
  applied **pointwise** — `δ^(h)(x)` depends only on `u(x)`, no spatial aggregation (paper: "ϱ(x) depends
  on u(x) but not on u(x′) for x′≠x"). `num_heads=40` default; each head emits an `n_spatial_dims`-vector
  displacement, so output is `(n_dims·num_heads)` channels. These are **LEARNED trainable conv weights.**
- **How does sparse source-coordinate sampling give linear-cost nonlocality?** Prediction is local (O(N)),
  but the *sample* `v(x+δ(x))` reaches an arbitrary faraway source coordinate — "one per head." So global
  interaction enters through *where you fetch from*, not through an all-pairs mix. Cost is **O(N·H)**, H fixed
  → linear in grid points N (paper's headline). No softmax attention, no Fourier multiplier, no conv mixing.
- **What makes it "sharp" — advect a template, or interpolate and re-blur?** The op is **`F.grid_sample(...,
  mode='bilinear')`** (their `custom_grid_sample2d/3d` wraps `F.grid_sample`, adding periodic-BC padding —
  `grid_samples.py`/standalone lines 135-391). So it **advects LEARNED FEATURES via BILINEAR interpolation**.
  The sharpness claim is **RELATIVE, not absolute**: vs FNO it applies **no global spectral truncation** (an
  edge is *moved*, not band-limited), so grid-resolvable discontinuities survive better than under Fourier
  multipliers. BUT each bilinear `grid_sample` is itself a **sub-grid ~1px low-pass**, and FlowerBlocks are
  *stacked* (U-Net, 4 levels, one warp each) → the smoothing **compounds**. It is emphatically *not* a
  sharpness-generating superpower; it is "less blurring than Fourier/attention on smooth PDE fields."
- **Differentiable + MLX/Metal-friendly?** Differentiable (`grid_sample` has a VJP). Framework = **PyTorch
  only** ("no dependencies beyond torch"), `@torch.compile` on the sampler. MLX has no native `grid_sample`;
  we would need a custom Metal kernel — **which we already own** (our `warp_frame0_native_mlx` + fused-R are
  exactly this bilinear-pullback operator class, bit-checked vs numpy oracle).

**Architecture wrapper (DERIVED):** `FlowerBlock = GELU∘Norm∘(SelfWarp + IdProj)` (Id = 1×1 conv skip),
scaffolded in a **U-Net** (`Flower`, `flower_standalone.py:638`). Default `flower.yaml`: lifting_dim 160,
n_levels 4, num_heads 40 → the model is a **17M-param (up to 150M) LEARNED neural operator.** The 17M variant
beats FNO/conv/attention baselines and the 156M beats Poseidon-L (628M) on smooth-PDE benchmarks (flows/waves).

---

## 2. Assessment vs OUR 5 warp surfaces (DERIVED)

| # | Our surface | Flowers relation | Verdict |
|---|---|---|---|
| 1 | Temporal-screw force (LIVE #205): warp 3 GROUND softmax channels by ego homography H(ξ) via `warp_frame0_native_mlx` | **SAME operator class** (bilinear pullback `v(x+δ)`). Difference: our δ = **analytic ego-homography** (generic, ~free); Flowers' δ = **learned conv-MLP** (counted). | AFFIRMS our design; adds nothing free |
| 2 | #365 Morse-Smale-stratified parallax warp (task-space pose, per-class) | Flowers' **multihead** δ (40 independent heads) is the one genuinely-new *idea*: per-stratum/per-class heads. But learned heads = counted bytes. | Idea-affinity only; learned = wrong side of rule-118 |
| 3 | The R operator (bicubic↑384→874 → uint8 → bilinear↓512×384) | R's bilinear↓ is the **same bilinear-`grid_sample` low-pass**. We already **MEASURED R erases sub-crossover lane dashes**. Flowers' warp is on the SAME side of that crossover. | Flowers ≠ Gibbs-antidote for OUR erasure |
| 4 | se(3)/SE(3) screw engine (`tac.lie`) — ξ warps partition (d_seg) AND is pose (d_pose), dual-use | Flowers has **no Lie/geometry structure**; its δ is a free-form learned vector field, not a twist. Our ξ is a *3-param* generic prior; Flowers' field is *N·H learned scalars*. | Ours is strictly more rate-optimal |
| 5 | v8 merge→diff→correct (warp composite by ego motion) | Same bilinear-pullback op; same free-vs-learned distinction. | AFFIRMS; adds nothing free |

**Synthesis:** Flowers' warp OPERATOR is *exactly the operator we already own and already ship* (bilinear
pullback via grid_sample, custom Metal kernel, periodic/boundary handling). What Flowers adds on top is a
**learned, multihead, pointwise displacement predictor as a backbone-scale replacement for conv/Fourier/attn
mixing.** That is an *architecture* choice whose weights are **video-derived learned content**.

---

## 3. Rule-118 accounting — THE DECISIVE cut (DERIVED)

CLAUDE.md rule 118 boundary: GENERIC ALGORITHM = FREE in inflate.py; VIDEO-DERIVED LEARNED content = COUNTED
in archive.zip.

- **The warp OPERATOR** `grid_sample(value, base_grid + δ)` = generic algorithm → **FREE** (we already run it
  free in inflate; the base_grid is deterministic).
- **The Flowers δ predictor** (`flow_head`) + `value_head` + the whole U-Net = **LEARNED conv weights,
  17M–150M params → COUNTED**, and enormous. This is the *opposite* of our capstone thesis (move maximal
  generic structure into free inflate.py; store ONLY the tiny video-derived sufficient statistic).
- **Our δ** on surfaces 1/3/4/5 is an **analytic function** (ego homography H(ξ), openpilot lane polynomial)
  whose only counted payload is the **~7.2 KB ξ** (MEASURED, R1 dxi) or a handful of lane coords. Rate-minimal.

**Conclusion:** adopting Flowers as a witness *backbone* would **ADD** a large counted weight blob and fight
the rate term — it walls exactly where our measured trilemma says bc36-class already sits (d_seg adequate,
**rate at frontier**). A learned warp backbone does not lower rate; at best it trades param-efficiency for
d_seg, and our #1 MEASURED lever (directional/curvelet Fourier basis, **−48% d_seg at ~0 byte**) + step-native
already attacks the Gibbs/thin-lane problem **rate-free**, which a counted backbone cannot beat on the rate axis.

---

## 4. Round-1 adversarial self-review — ATTACK my own "AFFIRMATION-only" verdict (DESIGNED)

**Steelman for "it IS a useful new lever":** (a) *Sharpness by advection is real* — the paper empirically
beats Fourier/attention and the interpretability figure shows velocity-aligned displacements; our #1 failure
is spectral-bias erasure, so a warp-native decoder could be more parameter-efficient at sharp argmax
boundaries than PR95's PixelShuffle+**sin** decoder (sin = Gibbs generator). (b) *Multihead per-stratum δ*
maps cleanly onto our #365 per-class parallax and v8 per-class carriers — 40 heads ≈ per-stratum flows.
(c) *We already own the Metal kernel*, so integration cost is low.

**Rebuttal (why it still collapses to AFFIRMATION-only):**
1. **The "sharp template" premise is only half-true, and the false half is exactly our problem.** Flowers
   warps *learned features* through **bilinear grid_sample** — a sub-grid low-pass that, *stacked*, blurs.
   We **MEASURED** that our R's bilinear pass **erases lane dashes below the crossover** (dash_erasure_
   homogenization_v1: dashes UNRECOVERABLE below crossover at ANY capacity). Flowers' warp is the *same
   interpolation class* → it inherits, not cures, our erasure. Its wins are on **smooth** PDE fields
   (flows/waves), a different regime from **piecewise-constant argmax with 1-px lane dashes**.
2. **Learned backbone = counted bytes = wrong axis.** Our binding wall at bc36 is RATE, not d_seg-capacity.
   A 17M–150M-param warp net is a rate catastrophe; even shrunk to bc36-size it does not beat a 0-byte
   directional-basis prior on the rate axis. (NO-FAKE #7: bolting a learned neural-operator backbone onto
   the capstone would also be a borrowed-substrate reskin, not our original rate-minimal task-space witness.)
3. **The operator is not new to us.** grid_sample-pullback is surfaces 1/3/4/5 already. The only genuinely
   novel *idea* (multihead learned δ) is precisely the counted-weight part we must NOT adopt.
4. **The one surviving DESIGNED sub-lever is off-path:** "warp-head decoder as a param-efficient sharp
   generator" only matters if we pursue a *learned bc36-class witness decoder* — a rate-dominated arm the
   capstone explicitly de-prioritizes vs the coord-INR + directional-basis. It is not a byte-closed-exact-row
   path to sub-0.19.

**Self-review verdict:** the steelman does not survive. Flowers **AFFIRMS** that warp/advection is the right
family (we're already in it) and **validates** our choice to advect by an *analytic* ξ rather than learn δ.
It does not hand us a new free lever, and its learned form is rate-antagonistic. **AFFIRMATION-only.**

---

## 5. $0 n600 probe DESIGN (DESIGNED — do NOT run concurrent with #205; a NEW through-R SegNet forward is memory-UNSAFE)

**Hypothesis to falsify (the steelman's core):** "a bilinear-warp / advection step *preserves* lane structure
in OUR data." If a single small bilinear warp already destroys lane-class IoU, Flowers-style advection cannot
be a lane-sharpness lever for us.

**Probe (read-only, memory-safe, NO SegNet forward):**
- Input: `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` key `'lstars'` (600×384×512 argmax; **$0
  read-only OK**). Extract the class-1 (Lane) binary mask per frame (comma10k canonical order — lane=1;
  self-detect, never hardcode).
- Compute frame-to-frame **lane pixel displacement magnitude** (centroid / nearest-match of lane components
  across consecutive frames) → is lane motion **sub-pixel (<1px, bilinear erases)** or **super-pixel
  (>1px, advection could help)**? MEASURED distribution decides.
- Cross-check: take frame-t lane mask, apply ONE CPU `torch.nn.functional.grid_sample(mask.float(),
  base_grid + δ, mode='bilinear')` with a small synthetic sub-pixel δ (pure CPU tensor op on the *gt mask*,
  **no scorer, no model**), threshold back, and measure **lane-IoU(warped, original)**. If IoU collapses
  (echoing the MEASURED lane IoU 0.263 instability), the bilinear-warp class erases lanes → Flowers dominated
  for surface-2/lane. Runtime: seconds on CPU, RSS ~ a few GB (one 384×512 float frame at a time), fully
  memory-safe alongside #205. **Falsification threshold:** median lane-IoU after one sub-pixel warp < 0.7 ⇒
  advection erases → Flowers NOT a lane lever (expected outcome, consistent with dash_erasure law).

This probe is **decisive at $0** and needs no paid GPU and no witness forward.

---

## 6. Triality framing (DERIVED)

- **DSL leg:** **NOT a lever candidate.** A learned Flowers backbone is a counted-weight architecture, not a
  swept `Lever` on the capstone's rate-minimal path; folding it as a DSL `Lever` would misrepresent an
  off-path arm as a live knob. The *warp operator* is already realized (surfaces 1/3/5) and needs no new DSL
  entry. If (and only if) we ever open a "learned warp-head witness decoder" arm, register it then.
- **equations leg:** **FORMALIZATION_PENDING** — no canonical equation until MEASURED. The one registrable
  claim would be a warp-erasure equation IF the §5 probe runs (it would *strengthen*, not replace,
  `dash_erasure_homogenization_v1` by tying bilinear-warp erasure to the same crossover). No equation is
  registered on a paper read.
- **DAG leg:** this memo + the FEED appended to `sub015_DAG_*`.

---

## Bottom line

Flowers is a beautiful, well-engineered learned neural-operator whose warp OPERATOR is **the exact bilinear
pullback we already own and ship free**, and whose novelty (multihead **learned** pointwise δ) is a
**counted-weight backbone** on the wrong side of rule-118 and the wrong axis (rate) of our measured trilemma.
Its "sharpness" is relative-to-Fourier on smooth PDE fields and does NOT cure our MEASURED sub-crossover lane
erasure (same bilinear class as our R). **VERDICT: AFFIRMATION-only** — it validates that we are in the right
(advection) family and that advecting by an *analytic ξ* (not a learned δ) is the rate-optimal choice; it
does not open a byte-closed-exact-row path to sub-0.19. Pointer 0.19110 UNMOVED.

<!-- APPEND-ONLY apparatus-hygiene footer (Catalog #344 re-baseline sweep 2026-07-09, per Catalog #110/#113 HISTORICAL_PROVENANCE; body unchanged). -->
# FORMALIZATION_PENDING: AFFIRMATION-only deep-dive — no NEW registrable canonical equation until the analytic-xi advection warp is MEASURED byte-closed on n600; the incidental empirical-narrative tokens are review NARRATIVE, not a new formalizable finding claim.
