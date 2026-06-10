# Structural Q* compression of the learned HNeRV decoder (#71) — verdict

**Date:** 2026-06-10 · **Lane:** #71 (the convergent singular pointer-mover) ·
**Authority:** exact frozen CPU scorer (`upstream/modules.py` DistortionNet, GT via
`frame_utils.yuv420_to_rgb`, NEVER MPS) + exact codec-grammar brotli byte cost.
**Tag:** `[macOS-CPU advisory]` (candidate-generator; a pointer move requires the
confirming paired contest CPU+CUDA exact eval).

## FIRST LINE — did the exact pointer move?

**NO — the exact frontier pointer did NOT move (0.19109982 unchanged).** Every
structural-compression lever that reduces the decoder-blob bytes (the 162,127 B,
91.5% of the 177,169 B archive) breaks d_seg/d_pose far outside the Q* cell, by a
margin of **~70–370× the rate-gain it buys**. This is the *fourth* convergent
no-move (#64 lossless, #72 residual collateral, #73 generic-basis ≥625 KB, and now
#71 structural) — together they prove the 162 KB learned HNeRV basis is at its
**distortion-holding floor**: it is NOT over-parameterized for the SCORE (only for
pixel reconstruction), so it cannot be shrunk while holding both terms.

## The operating-point curve (the prize, and why it's unreachable)

Frontier: S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489 = rate 0.118 (177,169 B)
+ distortion 0.073 = **0.19110**. At HELD distortion, ΔS = −25·(decoder_blob_saved)/
37,545,489. So a score-equivalent renderer at a smaller blob would move the pointer
by RATE ALONE:

| keep frac | blob bytes | Δblob | rate-only ΔS (if distortion held) | hypothetical S | feasible? |
|---|---|---|---|---|---|
| 1.00 (frontier) | 162,127 | 0 | 0 | 0.19110 | — |
| 0.90 | 159,079 | −3,084 | −0.00205 | 0.18905 | NO |
| 0.70 | 141,422 | −20,741 | −0.01381 | 0.17729 | NO |
| 0.50 | 118,380 | −43,783 | −0.02916 | 0.16194 | NO |
| 0.30 | 93,506 | −68,657 | −0.04572 | **0.14538** | NO |

The byte side is REAL and large (keep=0.3 would be sub-0.15 by rate alone). But the
distortion side dies: even keep=0.90 (only 10% pruned) raises S by **+0.038** —
the rate gain (−0.002) is dwarfed ~19× by distortion damage.

## The decisive quantity: distortion cost-per-kB vs the feasibility threshold

For a structural prune to MOVE the pointer (net-negative ΔS) the distortion
score-cost it incurs must be **< 25·1000/37,545,489 = 0.000666 per kB saved**.
The exact-scorer per-tensor ablation (prune one tensor to keep=0.5, measure
exact d_seg/d_pose on 8 pairs) found:

| tensor | Δd_seg | Δd_pose | byte_save | Δscore (distortion) | **cost/kB** |
|---|---|---|---|---|---|
| stem.weight | +0.004793 | +3.20e-3 | 10,453 | +0.6436 | +0.0616 |
| blocks.0.weight | +0.004302 | +4.61e-3 | 11,154 | +0.6301 | +0.0565 |
| blocks.1.weight | +0.003869 | +3.09e-3 | 11,304 | +0.5482 | **+0.0485** (cheapest big) |
| blocks.2.weight | +0.002944 | +5.86e-3 | 9,042 | +0.5217 | +0.0577 |
| blocks.3.weight | +0.002169 | +3.67e-3 | 5,025 | +0.3938 | +0.0784 |
| blocks.4.weight | +0.001806 | +2.83e-3 | 3,485 | +0.3343 | +0.0959 |
| blocks.5.weight | +0.002138 | +1.12e-2 | 2,147 | +0.5342 | +0.2488 |
| stem.bias | +0.003465 | +1.45e-3 | 488 | +0.4528 | +0.9278 |
| refine.0.weight | +0.000394 | +2.14e-5 | 217 | +0.0453 | +0.2089 |
| refine.1.weight | +0.000193 | +3.86e-5 | 388 | +0.0290 | +0.0747 |

The cheapest big tensor (blocks.1) costs **+0.0485/kB ≈ 73× the feasibility threshold**;
the global magnitude prune is ~80–90× over; even the tiniest tensors (refine.1
+0.0747/kB) are 112× over. There is no mixed schedule that closes a 73× gap.

## Levers tested (each measured on the exact scorer / exact byte grammar)

1. **Low-rank factorization (SVD/Tucker on dense weights).** Re-quantizing a
   rank-r reconstruction *INCREASES* the brotli blob by **+9% to +14%** at every
   distortion-relevant fraction (the SVD reconstruction destroys the
   brotli-friendly int8 code structure; factored storage also costs more params
   because the break-even rank is ABOVE the 95%-energy rank for every dense block).
   This independently confirms #73's "generic basis can't beat 177 KB" finding from
   the renderer-weight side: the learned weights are dense (r95 ≈ 78% of full rank).
2. **Magnitude structured pruning.** Shrinks the blob hugely (keep=0.7 −12.8%,
   keep=0.5 −27%, keep=0.3 −42% — zeros are nearly free under brotli) but breaks
   distortion at ~80–90× the feasibility threshold (table above).
3. **Score-aware Taylor pruning (|grad·weight| via a differentiable pose-MSE +
   SegNet-logit-KL surrogate, yuv6 patched so PoseNet carries grad).** WORSE than
   magnitude. Apples-to-apples at keep=0.6 (8 pairs, exact scorer):
   - score-aware: Δd_seg +0.0066, Δd_pose +0.064, blob −35,500 B, **ΔS +1.423**
   - magnitude:   Δd_seg +0.0062, Δd_pose +0.053, blob −38,510 B, **ΔS +1.309**

   Choosing WHICH weights to prune by scorer-gradient relevance does NOT hold
   distortion better than magnitude — it is slightly worse. The decisive
   mechanism: the learned weights are JOINTLY ENTANGLED for the memorized
   reconstruction; there is no sparse "score-irrelevant" subset to drop. The
   soft differentiable surrogate (pre-argmax SegNet KL + un-rounded pose) also
   under-predicts the rounded receiver's argmax-flip + pose blowup. This closes
   the last post-hoc structural lever: the frozen frontier weights cannot be
   pruned/factored while holding Q*.
4. **Distillation to a smaller architecture** — out of the $0–$1 fast scope (needs a
   training run); structurally faces the SAME distortion floor (the score depends on
   boundary-fidelity + pose, which the per-tensor ablation shows requires the full
   weight budget). DEFER-pending-research with a funded KD campaign.

## Why this is the convergent floor (the 4-no-move meta-finding)

#64 (lossless exhausted) + #72 (residual codes cheap but receptive-field collateral
kills application) + #73 (generic basis needs ≥625 KB / pair to hold the pose tube) +
**#71 (the learned basis itself cannot be pruned/factored ~73× cheaper than the rate
it buys)** all confirm from four directions: **the 162 KB learned HNeRV nonlinear
basis IS the cheap-feasible representation for holding pose+seg simultaneously, and it
is at its distortion-holding floor.** It is minimal for the SCORE, not just an
arbitrary memorized point. The rate term cannot be lowered via the decoder blob
without an equal-or-larger distortion penalty.

## Reactivation criteria (DEFER, not KILL — per "Forbidden premature KILL")

The structural lever reopens only when the distortion floor itself moves:
1. **A score-aware RETRAINED smaller renderer** (distill/QAT a renderer at a smaller
   architecture with the score-domain Lagrangian `α·B + β·d_seg + γ·√d_pose` as the
   training objective, NOT post-hoc pruning of the frozen frontier weights) — funded
   KD campaign; the only path that could relocate the floor.
2. **Compose with #69 (score-aware requant):** if requant relaxes the per-tensor
   distortion budget enough that the cheapest-cost-per-kB tensors (blocks.1/3/4) cross
   below 0.000666/kB — currently 73–144× away, so this requires #69 to deliver a
   ≥70× per-tensor distortion-budget relaxation (implausible alone, but the composed
   operating curve is the right place to re-measure).
3. **Lever-C contiguous-residual base (#72 reactivation):** a structurally different
   carrier where the receptive-field collateral that killed #72 is absent.

## Wire-in (6 hooks per Catalog #125)

- **#1 sensitivity-map:** ACTIVE — the per-tensor exact-scorer cost/kB table is a
  canonical pruning-saliency prior (blocks.1 cheapest at +0.0485/kB; blocks.5/stem
  most critical). Feeds the bit-allocator.
- **#2 Pareto:** ACTIVE — adds the hard constraint "decoder-blob structural prune is
  rate-dominated by distortion at this operating point" (the 0.000666/kB feasibility
  threshold is a Pareto facet).
- **#3 bit-allocator:** ACTIVE — the per-tensor cost/kB ranking is the allocation
  order for any future mixed-rate schedule (compose with #69).
- **#4 cathedral autopilot dispatch:** N/A — no archive-deployable candidate (all
  operating points fail Q* membership; no paid dispatch warranted).
- **#5 continual-learning posterior:** ACTIVE — DEFER probe outcome + the convergent
  4-no-move meta-finding seed the planner (de-prioritize post-hoc structural
  compression of the frozen frontier weights; prioritize score-domain retraining).
- **#6 probe-disambiguator:** ACTIVE — this memo + the `scorer_quotient_candidate_row`
  IS the disambiguator between "structural compression unreachable on frozen weights"
  (proven) vs "reachable via score-domain retraining" (reactivation #1).

## Innovation status

The frontier base is a defensive bank (`defensive_bank=true`, borrowed PR#112). The
score-sensitivity structural-compression METHOD (exact codec-grammar byte re-encoder +
per-tensor exact-scorer cost/kB ablation + the 0.000666/kB feasibility threshold) is
ORIGINAL and feeds the lever-C / score-domain-retraining campaign. No pointer move ⇒
no submission, no innovation claim.
