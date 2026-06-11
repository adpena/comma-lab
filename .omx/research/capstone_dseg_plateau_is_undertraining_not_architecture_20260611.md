# Capstone d_seg plateau: UNDER-TRAINING, not architecture or the EMA artifact (2026-06-11)

**Authority discipline (binding).** Every number here is `[macOS-CPU advisory]` /
`[macOS-MLX research-signal]`, **NON-PROMOTABLE** (`promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`). torch-CPU `evaluate.py` (600-sample, Linux x86_64) is the
ONLY leaderboard authority. NO MPS. NO paid dispatch fired this tick. **Frontier pointer UNMOVED:
0.19109982 [contest-CPU], 177,169 B — ABOVE T_1 → GOAL UNSATISFIED.** This is a diagnostic synthesis,
not a pointer move. $0 spend, local only.

## The three-layer correction (this is the third peel of the same onion)

1. **Layer 1 (fixed `f771e6e00`):** the d_seg "wall at 0.505" was an **EMA-shadow-lag artifact** —
   constant decay froze the exported shadow near init on short runs. FIXED with warmup decay. Real,
   banked.
2. **Layer 2 (revealed by the fix):** removing the artifact exposed a **TRUE plateau at d_seg
   ~0.004–0.010** (n48), **7–18× above the frontier HNeRV basin of 5.6e-4**. The d_seg-crux memo's
   optimistic premise ("base_ch=20 reaches the basin, the wall was only the artifact") is **only
   half right** — the wall was an artifact AND there is a real plateau under it.
3. **Layer 3 (this memo): the plateau is UNDER-TRAINING, not architecture or capacity.**

## The evidence that isolates under-training

- **The decoder is PR95-bit-exact.** `vq_nerv_bundle.py:419–436` is the exact `_HNeRVUpsampleBlockMLX`
  forward: `identity = bilinear_resize2x(x); decoded = pixel_shuffle_2x(conv(x)); y = decoded +
  identity; FiLM; sin(y)`. All frontier **L18 ingredients present** (PixelShuffle ✓, bilinear-skip ✓,
  sin ✓). The "our decoder is architecturally weaker" hypothesis is **REFUTED** — it is not an
  ingredient gap.
- **The muon arm was still descending, not plateaued.** `bc20_p48` (muon_throughout, EMA-fixed,
  stored_latent): d_seg 0.50538 init → **0.00393 @ epoch 110, still falling ~6%/10ep**. It never
  reached a floor — it ran out of epochs (120). PR95's reference budget is **29,650 epochs**; we ran
  ~100.
- **The curriculum's higher plateau is a curriculum-implementation issue, not capacity.** `c1prime`
  (pr95_8stage, EMA-fixed, ~7.5h): d_seg **stuck ~0.0097–0.0100 across stages 2→5** (best 0.00968 @
  stage 2; no improvement through stage 5). The bounded-surrogate later stages (tau_softplus / smooth
  / c1a_l7) stop pushing once pixels are "won" — they plateau *higher* than plain muon CE (0.004),
  consistent with the d_seg-crux §1.3 finding that **CE already concentrates 99.4% of gradient on the
  boundary band**, so the surrogates do not help d_seg here. Stage-8 `muon_finetune` (not yet run in
  c1prime) is the curriculum's one remaining d_seg lever.
- **d_pose is solved.** Both runs hold d_pose ≤ 3e-4 (stored_latent carrier works). The binding wall
  is d_seg alone.

## The resource reality (the honest cost wall)

Reaching the 5.6e-4 basin appears to need **PR95-scale epochs** (~10³–10⁴, not 10²). Local throughput:
- n48 mlx_gpu ≈ 65 s/epoch → 3000 epochs ≈ **54 h** (multi-day, but the M5 Max can saturate it per
  the long-resumable-sweeps directive).
- **n600 (the real pointer target) at PR95-scale is NOT feasible locally** in reasonable wall-clock
  (months). It is feasible on a paid GPU (T4/4090/A100) in a few GPU-hours — the legitimate
  exact-row spend.

So the capstone pointer-mover is a **paid PR95-scale n600 training run**, and it must be **de-risked
first** by a FREE local **n48 deep-muon train** that validates the under-training hypothesis (does
muon_throughout break below ~0.002 with thousands of epochs, or plateau?). MVP-first phasing: free
local smoke before paid dispatch.

## What is running right now (do NOT kill — operator "leave long runs running" directive)

| run | pid | backend | what it gives | status |
|---|---|---|---|---|
| `bc20_p192` (2×2 scaling arm) | 24706 | mlx_gpu (GPU) | muon n48→n192 d_seg **scaling slope** | training, just past target-build |
| `c1prime` (pr95_8stage n48) | 72123 | torch_cpu (CPU) | full-curriculum verdict incl. stage-8 muon_finetune | stage 5/8, ~7.5h |

They are on **different devices** (GPU vs CPU) → no contention. A redundant mlx_gpu deep-muon
diagnostic launched this tick was **killed** (it would have contended with bc20_p192 on the single
Metal GPU).

## The queued decisive next steps (in order)

1. **Read bc20_p192 + c1prime verdicts** as they progress (scaling slope; does stage-8 break 0.010).
2. **When the GPU frees** (bc20_p192 finishes, ~8 h): launch a long **resumable** n48 deep-muon train
   (thousands of epochs, marker-on-exit) → does d_seg break below ~0.002? This is the de-risk gate.
   *Note: the campaign has NO resume flag — add minimal stage/epoch checkpoint+resume before the
   multi-day train so a death loses ≤ the in-flight checkpoint (the standing-directive requirement;
   the 2×2 ablation lost arms for exactly this gap).*
3. **If the deep-muon validates** (d_seg → basin with epochs): estimate the paid n600 PR95-scale GPU
   cost, then dispatch it as the pointer-mover → byte-close → dual CPU+CUDA exact eval.
4. **If the deep-muon plateaus** above ~0.002 even at thousands of epochs: base_ch=20 IS
   capacity-limited → run the bc24 capacity arm (the unrun half of the 2×2) before any paid n600.

## Honest bottom line

The capstone is **not broken** — faithful decoder, pose solved, EMA artifact removed. It is
**under-trained** relative to the PR95 reference budget, and the binding question (does it reach the
d_seg basin with PR95-scale epochs?) is answered cheapest by a free local n48 deep-train, gating a
paid n600 run. The pointer is UNMOVED at 0.191 and this tick did not move it — it sharpened the
critical path and refuted the architecture-gap red herring.

---

## ⚠️ SAME-DAY ADVERSARIAL-REVIEW CORRECTION (2026-06-11, post-review) — "under-training" is OVER-OPTIMISTIC

A 3-lens adversarial + fresh-eyes review (operator-requested) **partially refutes the "under-training,
not capacity" headline above.** The original reasoning is preserved (the EMA-fix layer is correct; the
decoder IS PR95-bit-exact); but the *conclusion* is corrected:

1. **The muon arm asymptotes ~0.0025–0.003 d_seg — a CAPACITY signature, not "ran out of epochs."**
   The consecutive d_seg delta-ratios are 0.68–0.84 (slow geometric decay); extrapolated asymptote
   ≈ 0.0025–0.003 = **~5× above the 5.6e-4 target.** "Still falling at ep110" was real but is converging
   to a floor ~5× too high, not heading to the basin.
2. **The 5.6e-4 basin was NEVER measured on our architecture.** It was measured only on A1/PR101/PR102/
   PR103 — **~178K-param UNTIED frontier HNeRV decoders.** Our capstone is **85K params with
   `tie_depth=2` weight-tying** (shared convs = a real capacity reduction). "base_ch=20 reaches the basin"
   is **unsupported**; "PR95-bit-exact forward ⇒ reaches PR95 d_seg" is a **non-sequitur** (bit-exact
   forward says nothing about the minimum a smaller/tied net reaches).
3. **The pr95_8stage curriculum is likely BUGGED, not just slow.** muon (0.004) BEATS the curriculum
   (0.010); in c1prime, stage-3 `smooth_disagreement` (the loss whose minimizer IS d_seg) made d_seg go
   **UP**, and `clip_would_fraction = 1.0 every step` (grad_clip_muon=1.0 throttling every update) vs
   0.03 in the muon arm. So "run the PR95 curriculum longer" may be the WRONG lever — the curriculum is
   a bug to fix, and muon-only may already be the better recipe.
4. **The REAL anchor (not extrapolation): bc20_p48 is byte-closed at advisory S = 0.46790** (seg 0.376 +
   pose 0.037 + rate 0.055; `capstone_capacity_ablation_2x2_20260611/bc20_p48/capstone_result.json`;
   `[macOS-CPU advisory]`, quant-gap d_seg 2.6e-5). At the *easy* n48 case the favorable arm is **2.45×
   above frontier, d_seg-dominated.** n600 makes d_seg worse → a base_ch=20 n600 train would NOT reach
   sub-0.15; firing it would be means-hoarding.
5. **Meta-signal (the strongest finding): the d_seg plateau has been re-diagnosed FOUR times in two days**
   (0.505 wall → EMA artifact → true plateau → "under-training" → now "capacity + broken curriculum").
   Each "wall" dissolved into an artifact/reframe. The discipline correction is **STOP DIAGNOSING, START
   MEASURING** — anchor the R-D curve with measured byte-closed S rows, not another interpretation.

### Corrected critical path (supersedes the "queued next steps" list above)

1. **Capacity verdict FIRST (the unrun half of the 2×2 + an UNTIED arm):** run bc24 (and ideally a
   frontier-class ~178K UNTIED) arm to n48 → does more/untied capacity lower the d_seg floor toward
   5.6e-4? This is the decisive capacity-vs-undertraining test. (bc20_p192, running, gives the
   pairs-scaling half.) GPU-gated behind the live bc20_p192 mlx_gpu run.
2. **$0 muon-resume-with-LIVE-d_seg** (Review 1's test): resume bc20_p48 muon +400ep logging
   `use_ema_for_eval=False` alongside the shadow → kills the residual-shadow-lag confound AND reads the
   true asymptote (<0.0015 ⇒ under-training; ≥0.0025 ⇒ capacity ceiling).
3. **Fix the curriculum bug** (grad_clip_muon=1.0 throttling; smooth_disagreement raising d_seg) — $0.
4. **Only after a capacity verdict**: a paid n600 PR95-scale run at the RIGHT capacity (frontier-class
   untied, NOT 85K tied) — local n600 is ~5–6 months on either backend (measured: torch-CPU 8.9 min/ep
   @ n600; mlx_gpu only ~1.2–1.5× faster because the FP32-exact arch-override forces the slow non-NAX
   kernel + depthwise convs are memory-bound), so the pointer-mover REQUIRES a paid GPU (few GPU-hours).

The original "under-training, fire n600 at base_ch=20" framing is **SUPERSEDED** by this correction.
The pointer remains UNMOVED at 0.191.
