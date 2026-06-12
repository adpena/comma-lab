# R1 — Lever-C cheap-frame1 carrier RE-MEASURED with LIVE d_seg + warmup-EMA

**Authority:** `[local CPU-torch advisory]` — exact frozen `modules.py` PoseNet/SegNet on CPU, GT decoded
via `frame_utils.yuv420_to_rgb` ONLY (verified in `render_and_score_lib`), S/terms recomputed from
components. `[macOS-MLX research-signal]` for the conv-decoder forward (numpy↔torch parity 1.0). **NOT**
the contest 600-sample harness → NON-PROMOTABLE per the authority ladder. `$0` spend, NO GPU, NO paid
dispatch, **NO MPS**. `promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`.

**Frontier (pointer, not hardcoded):** `0.19109982 [contest-CPU]`, 177,169 B — UNMOVED. GOAL UNSATISFIED.

## The question (R1 of `negative_results_resurrection_ledger_20260611.md`)

The lever-C verdict (`lever_c_viability_smoke_20260610T144739Z.md`, tasks #57/#61/#62) reported the small
<120KB conv per-pair frame1 carrier moved d_pose 114× but moved **d_seg by ZERO** — trained-decoder exact
d_seg `0.50732 == constant-frame floor 0.50692` ("moved by zero"). That verdict used the **constant-decay
EMA shadow** as the inference checkpoint on an 8-pair / 180-epoch SHORT run. Per the capstone EMA-shadow-LAG
finding (commit `f771e6e00`), a constant-decay shadow (τ_eff = 333 steps at 0.997) FREEZES near init on a
short run, so exact d_seg reads the near-init constant-frame floor EVEN IF the LIVE weights solved seg.

**R1 re-measures with R2's warmup EMA AND LIVE d_seg** to disambiguate: was the "moved by zero" a frozen
shadow (→ the cheap carrier un-falsifies), or is the carrier genuinely seg-blind (→ the verdict survives)?

## Method

- R2 fix landed: lever-C trainer switched from its inline constant-decay EMA to the canonical
  `tac.training.EMA` (now warmup `min(decay,(1+t)/(10+t))`). Added an `--eval-weights {ema,live}` switch
  and an `r1_disambiguator` block that ALWAYS reports BOTH the LIVE-weights and warmup-EMA exact d_seg/d_pose.
- Config A (the actual <120KB candidate): 8 pairs, seed 32, channels 32-24-16-12, latent 24, int8 →
  104,426 B. 40 epochs (the capstone descended by ep25; 40 is ample to see descent-or-not).
- Same authority chain, same SSD targets (`lever_b_score_native_argmax_smoke_20260610/targets_n600`).

## The LIVE training trajectory (the mechanism, captured live)

| epoch | seg_ce (live, through frozen SegNet) | pose_mse (live) | recon |
|---|---|---|---|
| 1  | 3.152 | 6.397 | 10078 |
| 10 | 2.381 | 0.116 | 5590 |
| 20 | 2.687 | 0.280 | 4449 |
| 30 | 2.438 | 0.081 | 4074 |

**seg_ce does NOT collapse toward zero** — it oscillates in the 2.4–2.7 band while pose_mse is crushed to
~0.08–0.28. This is the SAME antagonism signature the original memo described (seg term blocked while pose
solves), now observed on the LIVE weights — NOT a frozen shadow. Contrast the capstone case, where the LIVE
seg CE dropped 150× (the model WAS solving seg, only the shadow lagged). Here the LIVE seg objective itself
plateaus high.

## RESULT — LIVE vs warmup-EMA exact d_seg (config A, 40 epochs)

| eval weights | exact d_seg | exact d_pose | bytes | below 0.05 re-open threshold? |
|---|---|---|---|:--:|
| **LIVE**       | **0.50553** | 0.09961 | 104,426 | NO (10× above) |
| **warmup-EMA** | **0.41365** | 0.06659 | 104,426 | NO (8× above) |
| constant-frame floor | 0.50692 | 13.48 | — | — |
| ORIGINAL (constant-decay EMA shadow, FALSIFIED) | 0.50732 | 0.10540 | 104,426 | NO |

numpy↔torch parity 1.0; seg_loss=argmax_ce; joint_hold_under_120kb = **NO**.

## What the EMA fix REVEALED (the partial artifact)

1. **The original 0.50732 WAS partly the frozen-shadow artifact.** The warmup-EMA d_seg is **0.41365** —
   `−0.093` BELOW the constant-frame floor (0.50692) and below the original frozen reading. The
   constant-decay shadow froze the original reading AT the floor; the warmup shadow descends off it. So
   "d_seg moved by ZERO" (original) is FALSE — d_seg DID move, by ~0.09, once the shadow tracks.
2. **BUT d_seg does NOT descend to the re-open threshold.** Best d_seg (warmup-EMA) = **0.414**, still
   **~8× above** the ledger's 0.05 reactivation bar and **~740× above** the frontier's 5.6e-4. The LIVE
   weights are even worse (0.506, at the floor). The carrier moves d_seg a LITTLE, not to viability.
3. **The 114× pose win is real and survives** (d_pose 0.067–0.10 vs constant 13.5).
4. **The mechanism is the antagonism, NOT just lag.** LIVE seg_ce (through the frozen SegNet) plateaus in
   the 2.4–2.7 band across 40 epochs (3.15→2.38→2.69→2.44) — it does NOT descend toward ~0.08 (the CE that
   lever_b's LOGIT-space generator reached for d_seg 0.012). Contrast the capstone, where LIVE seg CE
   dropped 150× and ONLY the shadow lagged. Here the LIVE objective itself is stuck high → the RGB→frozen-
   SegNet path carries weak d_seg gradient at score-native capacity, exactly the original §3 mechanism.

## VERDICT — RE-MEASURED, NOT un-falsified (partial-artifact correction)

The original "moved by ZERO" claim is CORRECTED: with the warmup EMA, d_seg moves ~0.09 off the floor — the
"exactly the constant floor" reading was a frozen-shadow artifact. **But the carrier does NOT un-falsify:**
the best d_seg (0.414) is ~8× above the re-open threshold and ~740× above the frontier. The cheap <120KB
frame1 carrier still cannot hold d_seg (joint_hold = NO). The pivot in the original memo (rate-side levers
on the EXISTING frontier — lever D contour residual / R1+R2+R3 entropy bank) STANDS.

**Honest CAVEAT (per the ledger):** this is a 40-epoch smoke; a 120-epoch confirmation run is in flight to
verify the seg_ce plateau is not under-training. The seg_ce trajectory (stuck at 2.4–2.7) strongly predicts
a plateau, not slow descent. Config B (250k/214KB, the "second effect" config) was NOT re-run to completion
(already over the 120KB budget regardless; parallel CPU contention).
