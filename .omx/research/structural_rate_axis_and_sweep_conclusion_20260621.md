# Structural rate axis (low-rank / weight-tie) + the UNIFYING conclusion of the 2026-06-21 deep-math sweep

**Operator ask (2026-06-21):** the structural rate axis — the one path the codec sweep left open. **THEORY + $0
MEASUREMENT** (decoder SVD, no rendering). Authority: `[contest-CPU advisory]`, NON-PROMOTABLE, pointer UNMOVED
0.19110. Closes the rate-axis investigation.

---

## 1. Measurement (live bc20 decoder, per-tensor SVD, $0)
83,422 params, 7 big tensors. Per-tensor effective rank (matrix-folded conv `(out, in·kh·kw)`):

| tensor | matrix | r90 | r99 | min-dim |
|---|---|---|---|---|
| stem | 768×28 | 24 | 28 | 28 (full) |
| blocks.0 | 64×144 | 43 | 60 | 64 |
| blocks.1 | 68×144 | 45 | 64 | 68 |
| blocks.2 | 76×153 | 49 | 70 | 76 |
| blocks.3 | 76×171 | 50 | 71 | 76 |
| blocks.4 | 56×171 | 40 | 54 | 56 |
| blocks.5 | 40×126 | 28 | 38 | 40 |

**r99 is 85–95% of the min-dim everywhere → NEAR-FULL-RANK.** The trained decoder USES its rank.

## 2. The structural levers, ranked
1. **Post-hoc low-rank factorization — small AND fragile.** Optimistic @r90 (drop 10% energy) = 4.3 KB
   (ΔS −0.0028), but dropping 10% energy perturbs the weights → the shallow d_seg boundary (66.5% of flips
   <0.5 logit) flips → hurts d_seg (same fragility as int5 quant). @r99 (safe): ~0 (small near-full-rank
   tensors don't compress: r·(m+n) ≈ m·n). ✗ mostly.
2. **Post-hoc weight-tie — N/A.** No two big tensors share an identical shape → can't tie a trained decoder.
   Must be DESIGNED-IN.
3. **DESIGNED-IN structure — the REAL rate lever.** Train a decoder with fewer effective params from the start
   (smaller base_ch, tied blocks, low-rank parameterization `W=UV`, grid-PE) → the boundary FORMS inside the
   compressed parameterization → co-adapts → d_seg-neutral BY CONSTRUCTION (unlike post-hoc perturbation). This
   is the capstone L1 weight-tie + base_ch choice.

## 3. THE UNIFYING CONCLUSION OF THE SWEEP (rate is param-count, and param-count is a TRAINING choice)
Every rate lever investigated 2026-06-21 reduces to one fact: **the decoder rate term = param_count × 6.9
bits/param (the entropy floor, already hit by brotli-q11).** Post-hoc compression of a TRAINED decoder is
exhausted or boundary-blocked across the board:
- entropy coding → at the floor (brotli ≈ H(W); `decoder_weight_rate_axis...`).
- uniform sub-8-bit quant → caps S~0.49 (shallow boundary quant-fragile; int5 anchor).
- post-hoc low-rank → near-full-rank + fragile (this memo).
- per-flip d_seg sidecar → break-even 1.273 B/flip (`dseg_boundary...` §6.3).
- latent dedup → small, near-full-rank (`latent_dedup...`).

**→ The ONLY rate lever is DESIGNED-IN: train fewer effective params from the start.** And the d_seg lever is
DESIGNED-IN too: the Muon κ-buster (free, training-time). **The score moves on TRAINING + ARCHITECTURE, never
on post-hoc codec.** Both confirm the capstone direction (fresh-init small-basis + L1 weight-tie + grid-PE,
finished with Muon) and the live run (Muon stage 8). The deep-math sweep's net contribution: it bounded FIVE
post-hoc detours with measured numbers and re-pointed all effort at the two designed-in levers already in play.

## 4. The deeper structural insight (a concrete NEXT capstone axis)
The decoder is near-full-rank because it's trained on RGB RECONSTRUCTION (recon-MSE). But the contest scores
ONLY d_seg (argmax) + d_pose — which need LESS than full RGB fidelity (the conditioning finding: NOT
capacity-limited for the score). So the decoder is **"full-rank for RGB, over-capacity for the SCORE."** A
recon-LIGHT, score-AWARE decoder — capacity allocated to the boundary-controlling + pose-controlling pixels,
sloppy elsewhere — could be genuinely SMALLER → the rate lever IS score-aware training (down-weight recon-MSE,
up-weight the d_seg surrogate + d_pose), letting the decoder shrink into the score-relevant subspace. This is
the task-space/VCM direction (code the task-sufficient statistic, not the RGB) realized as a TRAINING objective
re-weighting. **Concrete next capstone axis: a recon-light score-aware loss schedule + a smaller base_ch, gated
on the conditioning headroom.** (Caveat: the decoder must still emit RGB for the scorer; "recon-light" means
lower fidelity where the score doesn't look, not no-RGB.)

## NO-FAKE ledger
- MEASURED ($0): decoder per-tensor SVD (near-full-rank r99 85–95%); post-hoc low-rank @r90 4.3 KB optimistic;
  no shared shapes (tie N/A post-hoc).
- REASONED: post-hoc low-rank is fragile (shallow boundary); designed-in structure co-adapts; rate = param_count
  × entropy-floor; the full-rank-for-RGB-vs-over-capacity-for-score distinction.
- NOT claimed: no score moved; pointer UNMOVED 0.19110; the recon-light score-aware-smaller-decoder is a DESIGN
  hypothesis, not measured (its d_seg-neutrality must be trained + verified, not assumed).

## Cross-references (the full 2026-06-21 sweep)
- `dseg_boundary_hessian_conditioning_20260621.md` (d_seg = κ-conditioning; Muon κ-buster; shallow boundary).
- `decoder_weight_rate_axis_and_shallow_boundary_synthesis_20260621.md` (entropy floor; quant blocked; the synthesis).
- `latent_dedup_information_bound_20260621.md` (latent rate small).
- `capstone_batch_size_fixed_point_B64_launch_spec_20260621.md` (the B=64 + Muon-B-invariance spec).
- `optimal_capstone_vehicle_spec_20260611.md` (the canonical vehicle: small-basis + L1 weight-tie this confirms).
- `tilde_optimizer_survey_vs_dseg_conditioning_20260621.md` (the κ-buster optimizer survey, in flight).
