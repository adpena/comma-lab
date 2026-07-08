# T5 CRUCIBLE — PURSUIT-CHAIN-A — spectrum → localize → solve-step (recursive chain log)

STORES CONSULTED: t5_crucible grounding packet · position_S3 (+ its harness
`experiments/t5_s3_hvp_lanczos_probe.py` + `spectrum_ep650_K8_s0.json`) · position_S2 (M-S2-1..5 +
RECESS-2 bar) · #341 `tools/quadratic_basin_finisher_probe.py` + its artifacts
(`verdict_baseline_n600` 0.0035103, `verdict_solved_head_n600` 0.0036878 = the measured +5.1% subset
NO-GO, `solve_log_head.jsonl`, `feats_state_main_gt1.npz`) · mod32cap run dir (ep650 EMA-best ckpt)
· coordinator caveats (self-orient reconstruction gap +4.3%; absorption-bug serializer flags) ·
memory L70 (MLX-GPU never bit-exact) / L77 (subset-solve gap eq) / n600-verdict-chunk law.
NOT consulted: other position_S*.md beyond S2/S3 (not in my charter), canonical-equations registry
beyond the L77 anchor, the islands treatment-arm memo (out of chain scope).

Chain agent: PURSUIT-CHAIN-A · 2026-07-07 · post-position pursuit (S3 spectrum + S2 recess-2 thread).
Axis: **[macOS-CPU/MLX advisory] NON-PROMOTABLE** on every number; pointer contest-CPU **0.19110 UNMOVED**;
everything here is MEANS. No training launches; foreground chunked only.

Artifacts: `experiments/results/t5_pursuit_chainA_20260707/` + reuse of
`experiments/results/t5_s3_hvp_lanczos_20260707/` and `experiments/results/basin_finisher_probe_20260707/`.

## Coordinator caveat honored (reconstruction gap)

The self-orient directional state is NOT persisted in checkpoints; the probe reconstructs it
(fixed-point map, `feats_state_main_gt1.npz`). MEASURED consequence already on disk (#341):
**baseline ep650-EMA re-verdict through THIS load path = d_seg 0.0035103 (n600)** vs the run-logged
0.0033662 → +4.3% reconstruction gap, matching the coordinator's number. All step comparisons in this
chain are apples-to-apples against **0.0035103** (same load path, same feats state, same int8-dequant
deploy verdict); 0.0033662 is reported alongside as the live-state bar.
Also on disk: #341 head-solve (K=8 subset) n600 verdict **0.0036878** = WORSE than baseline (the
subset-overfit anchor, +5.1%).

## LINK 0 — instrument validation (the un-skippable prequel the chain forced)

Before deepening the spectrum, validated the HVP operator itself ($0, one pair, both devices):

| measurement | CPU | GPU | reading |
|---|---|---|---|
| HVP vs central-FD of grad (ε=1e-3, along g), rel err / cos | 0.351 / 0.943 | 0.308 / 0.952 | analytic (STE-smoothed) curvature ≠ true local landscape by ~35% along g |
| symmetry asym uᵀHv vs vᵀHu (random u,v) | 3.0e-6 | 1.1e-3 | both devices internally consistent |
| same-process repeatability | 0.0 (bit) | 3.5e-6 | deterministic per device |
| CPU↔GPU operator gap (uᵀHv) | −0.03207 vs −0.02970 | | ~7.4% device gap (fp32 path), tolerable vs 10× kill band |
| HVP wall/pair | 11.0 s | 2.33 s | GPU = 4.7× throughput |

**Implications (each drove the next link):**
1. The FD mismatch is EXPECTED mechanistically: the through-R path carries live un-disableable
   uint8-STE (S2 audit #13; dossier §22(3)) — FD across ε=1e-3 integrates true rounding-jump
   curvature, the analytic HVP sees the straight-through-smoothed surface. ⇒ any solve step must be
   accepted by MEASURED loss/verdict (lm_accept-style), never by predicted quadratic reduction alone.
2. GPU admissible as THROUGHPUT device for the Lanczos ladder (research-signal; L70 — never bit-exact
   cross-process), with the K=8 CPU/GPU Ritz cross-check as the gate (Link 1).
3. Per-pair HVP noise ~7-8% between fp32 paths ⇒ prefer LARGER K (averaging 1/√K) — another reason the
   K-ladder is the right deepening, independent of subset-representativeness.

## LINK 1 — deepen the spectrum: K-ladder 8 → 32 → 128 at ep650-EMA (GPU throughput)

Kill band (pre-registered by S3 RECESS-R1, honored): at K=128, |λ₋|/λ_max < 0.1 KILLS the
"ep650 not 2nd-order exhausted" verdict → wall = capacity/basis, Arm A only mover.
Persist band: |λ₋|/λ_max > 0.5 at K=128 ⇒ indefiniteness is a full-P property (E[H_K]=H_600).

**K=8 GPU cross-check (10 iters, same subset seed 0):** Ritz `[−174.6, −105.8, −87.6, −43.2, −0.15,
+21.6, +53.1, +78.0, +89.0, +134.0]` — λ_max +134.0 (CPU +139.3: agrees ~4%), λ₋ −174.6 (CPU −369.7:
2.1× device gap on the extreme-negative magnitude). **Qualitative verdict device-robust** (strongly
indefinite, |λ₋|/λ_max = 1.3–2.7, n_neg ≥ 3); **extreme-negative MAGNITUDE is fp32-path-fragile** —
treat |λ₋| as order-of-magnitude only. Artifact `spectrum_ep650gpu_K8_s0.json`.

**K-ladder at ep650-EMA (GPU throughput, seed-0 subsets, 10-12 Lanczos iters):**

| K pairs | λ₋ | λ_max | |λ₋|/λ_max | n_neg | grad_norm |
|---|---|---|---|---|---|
| 8 (CPU, S3) | −369.7 | +139.3 | 2.65 | 3/7 | 0.787 |
| 8 (GPU) | −174.6 | +134.0 | 1.30 | 5/10 | 0.796 |
| 32 (GPU) | **−33.3** | +117.4 | **0.28** | 4/12 | 0.363 |
| 128 (GPU) | (running) | | kill band < 0.1 | | |

λ₋ collapses ~1/K (K8→K32: 5.2× shrink for 4× pairs) while λ_max is K-stable. The subset gradient
norm also halves — much of the K=8 gradient was pair-idiosyncratic. **Pre-registered K=128
prediction (before the run finished): λ₋ ∈ [−15, −5], ratio ∈ [0.04, 0.14] — i.e. AT or BELOW the
kill band.** Artifacts `spectrum_ep650gpu_K{8,32,128}_s0.json`.

## LINK 2b — independent falsification: directional-curvature TRANSFER test ($0, minutes)

Chained from Link 3's machinery smoke: measured the SECOND DIFFERENCE of the loss along u_min
(the K=8 λ₋=−175 direction) on independent holdout subsets, [L(θ+su)+L(θ−su)−2L(θ)]/s²:

| holdout | s | implied curvature along u_min |
|---|---|---|
| K=16 seed 11 | 0.01 | −1.2 |
| K=16 seed 11 | 0.02 | +1.3 |
| K=16 seed 11 | 0.05 | +1.2 |
| K=8 seed 7 | 0.10 | +1.2 |

**The −175 direction has |curvature| ≈ 1 on independent pairs — 150× smaller, sign-unstable
around 0.** Two independent probes (K-ladder + transfer test) agree: the K=8 indefiniteness is
SUBSET-IDIOSYNCRATIC, not a full-loss property. Artifacts
`experiments/results/t5_pursuit_chainA_20260707/krylov_step_screen_{smoke_K8,curvtransfer_K8}.json`.

Incidental measured fact (new): the int8-dequant round-trip alone costs **+5.2% surrogate loss**
(holdout fp32 0.2357 → deploy 0.2479) — the deploy-quantization gap at ep650 is of the same order
as the whole reconstruction gap. Every solve/step claim must be screened at deploy params.

## LINK 2c — curiosity extension (coordinator): input-structure decomposition of u_min's in_proj block

Question posed: does u_min's in_proj mass concentrate on the ALONG-TANGENT frequency columns (the
Candès–Donoho rebalance direction → mechanism-identified Arm A grounding) or spread isotropically?
Column layout rebuilt exactly from the ckpt cfg + `curvelet_directional_B` /
`directional_fourier_feats` source: 80 curvelet cols (40 atoms × sin/cos; scales s0–s3 + iso) +
16 dir cols (4 freqs × {sin,cos}×{along, across}); in_feat = 96 verified.

| state | curv total (iso-expected 0.833) | dir_along (exp 0.083) | dir_across (exp 0.083) | enrichment (curv/along/across) |
|---|---|---|---|---|
| CPU K=8 | 0.832 | 0.087 | 0.081 | 1.00 / 1.05 / 0.97 |
| GPU K=8 | 0.841 | 0.087 | 0.072 | 1.01 / 1.05 / 0.86 |
| GPU K=32 | 0.835 | 0.083 | 0.082 | 1.00 / 1.00 / 0.98 |

**ISOTROPIC — decisively.** Per-scale masses are proportional to atom counts; no along-vs-across
asymmetry; no orientation coherence. The u_min escape direction is statistically indistinguishable
from an isotropic random vector in the basis-coupling block — exactly the signature of per-pair
Hessian NOISE, and triple-consistent with Link 1 (λ₋ collapses under averaging) and Link 2b (the
direction is flat on holdout). The Hessian at ep650 is NOT pointing at DirectionalBasisRebalance —
it is not pointing anywhere: the reachable 2nd-order structure at the best point carries no
exploitable basis information. (Arm A's case rests on its own measured −48% anchor, unchanged;
what this kills is the hope of a mechanism-identified SHORTCUT from the Lanczos geometry.)

## LINK 3b — cross-subset GRADIENT descent probe (does ANY cheap 1st-order descent exist?)

The EMA shadow is not a stationary point (subset gnorm 0.36–0.80), so a gradient step is the last
cheap descent candidate. Protocol: gradient from a LARGE subset (K=128 seed 7), line search
η ∈ {0.003, 0.01, 0.03, 0.1} along −ĝ, measured on a mostly-DISJOINT holdout (K=64 seed 13,
overlap 7/64), fp32 AND int8-deploy.

**Gradient-noise decomposition first (MEASURED, the gnorm ladder):** ‖g_K‖ = 0.796 (K=8), 0.363
(K=32), 0.194 (K=128) — textbook 1/√K. Fitting ‖g_K‖² = ‖g_true‖² + σ²/K across the ladder:
**g_true ≈ 0.08, per-pair σ ≈ 2.0** → the ep650-EMA point is NEAR-STATIONARY on the full loss and
even the K=128 gradient direction is majority noise (coherent-fraction SNR ≈ 0.44).

**Line-search result (`gradstep_cross_subset.json`): NO descent at ANY step, monotone worsening:**

| η | fp32 Δ | int8-deploy Δ |
|---|---|---|
| 0.003 | +5.3e-5 | +1.2e-5 |
| 0.01 | +5.6e-4 | +2.6e-4 |
| 0.03 | +5.1e-3 | +5.0e-3 |
| 0.1 | +5.9e-2 | +6.9e-2 |

Implied curvature along −ĝ ≈ **+12 (positive)**. First-order descent from cheap subset information
does not exist at ep650-EMA at the frozen schedule point.

## LINK 3c — the K=32 negative direction + the winner's-curse close-out

Completeness: does the LESS-noisy K=32 u_min (λ₋=−33) transfer? Screen (holdout K=16 seed 11):
fp32 flat-to-worse at every step; one candidate (`negcurv+0.02`) showed int8-deploy **−6.1e-4** —
chased immediately on a DISJOINT larger holdout (K=64 seed 13): flips to **+5.8e-4 int8 /
+1.1e-4 fp32**. Verdict: winner's curse on a 16-pair piecewise-constant int8 surface (best-of-6
selection), not a real descent. The int8-dither hypothesis is dead too.
Artifacts `krylov_step_screen_curvtransfer_K32.json`, `rescreen_negcurvK32_p02.json`.

**Measurement-economy note:** no n600 verdict was spent on any stepped candidate — every candidate
failed (or flipped negative on) the cross-subset screens that gate it. The chunked n600 verdict
path stands ready (`stage verdict`, stepped ckpts saved) if any future candidate passes a screen.

## LINK 2 — localize: WHERE the extreme curvature lives (Ritz-vector block projection, $0)

Projected the extreme Ritz vectors (u = V·y from the saved Lanczos states) onto the 18 parameter
blocks. DEVICE-ROBUST to 3 decimal places on the negative direction:

| direction | top blocks (mass fraction) | CPU / GPU agreement |
|---|---|---|
| most-NEGATIVE (λ₋) | **in_proj.weight 0.86** · hidden.0.weight 0.07 · in_proj.bias 0.02 · film.bias 0.02 | 0.859 vs 0.857 |
| most-POSITIVE (λ_max) | **out_tex.weight 0.48–0.61** · in_proj.weight 0.16–0.28 · film.bias 0.06 | consistent |

**Implication (the chain's sharpest structural finding so far):** the reachable negative curvature at
the run's best point is concentrated in `in_proj.weight` — the projection of the (curvelet +
self-orient directional) FEATURES into the trunk, i.e. the **basis-coupling layer**. The 2nd-order
escape direction the optimizer never took is a re-mixing of the basis features — this is
mechanistically the same territory as Arm A `DirectionalBasisRebalance` (basis-match is PRIOR to
capacity). The positive (stiff) directions live in the texture head. Negative curvature = large
coherent block (9,216 params), NOT noise-dominated small blocks ⇒ exploitable in principle.

## LINK 3 — exploit: curvature-aware solve step from ep650 (design)

Protocol (subset-overfit-aware, per #341's measured +5.1% NO-GO on K=8 subset SOLVES):
1. Build step candidates from the K≥32 Lanczos state at ZERO extra HVPs: exact trust-region solution
   on the tridiagonal (Δ(r) = V·y*, y* = TR argmin of ½yᵀTy + ‖g‖e1ᵀy s.t. ‖y‖≤r) — a Krylov-TR
   step family over radii r; plus pure ±u_min line search (negative-curvature exploitation).
2. Screen candidates on a HOLDOUT subset loss (different seed, K=64) — never the solve subset.
3. Quantization-survival screen: re-evaluate the survivor at int8-dequant deploy params (the verdict
   codepath quantizes; a step must survive the int8 round-trip to be real).
4. n600 chunked verdict on the survivor (verdict-batch ≤32, same reconstruction load path);
   bar: < 0.0035103 apples-to-apples (< 0.0033662 = live-state bar, reported alongside).

(running — results appended)

