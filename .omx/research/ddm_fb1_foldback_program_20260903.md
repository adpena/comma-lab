# ddm_fb1 — THE FOLD-BACK PROGRAM (MAIN, 2026-09-03): every post-training discovery, restated as the training objective it should have been

Operator 2026-09-03: *"Seems like all of the stuff we have discovered and chain after training points to improvements
to the training and other steps themselves"* + *"believe in yourself … be creative and weird … think divergently"*.
Tokens: `[no-triality] [p0-ledger-ok]`. Axis: recall + S-arithmetic; no scorer, no pointer move.

## The receipts that say the operator is right

| post-hoc discovery (receipt) | what it bought post-hoc | what it says about TRAINING |
|---|---:|---|
| five lossless stages after freeze (pq12) | −454 B | post-hoc re-partitions what training decided; the object is set upstream |
| joint edit admission (jg5: 455/573; the sub-0.15 crossing) then the win-win cone drained (fcd1/wwc1: ~3.7 KB gross, net +0.0019 S) | one crossing, then nothing | the render↔judge equivalence class is thin near GT: the field should be trained for the judge, not edited toward it |
| residual priced by HARD sites, not count (gc1: −608,974 mismatches saved 18,728 B) | closed capacity 9.62× | the loss must weight sites by their coded price; "kill hard sites" is the objective, not "fewer wrong sites" |
| 95% of seg error is MANUFACTURED by the render path (td1/rt1; mst1 78.71% at the native render) | — | the renderer, not the labels, owns d_seg's residual; retrain the renderer against the realized argmax |
| pose = terminal solve, compensation in-compile (qs5/up2/jg5); large edits break the pose gate (fcd2: 26,710× miss) | pose held at 6.37e-6 | pose belongs IN the training loop at step zero (w96b/qbr1 law), not as a post-hoc rescue |
| Lane = 0.59% area, 33.56% of bits; square atoms barely touch Lane (gc1) | — | capacity/atoms must be routed by class at training (Lane/Movable), not uniformly |
| SegNet's boundary jitter is the irreducible information (gs3) | every coder door shut | only a field that is RIGHT WITHOUT CORRECTIONS moves rate — trained, not fitted post-hoc |

## The map: discovery → training-time lever (status)

1. **Realized expected-flip margin through R → uint8 → SegNet argmax vs the DALI GT** — the CE1 law; live in the
   born trainer (qbt1) and the w96b aligned branch; NEVER applied to the SHIPPED 30,856 B semantic renderer at
   its own size from its own weights (rj1 left realized deltas UNMEASURED; wd2 was a smaller student; w96a/b are
   a from-birth lineage at d_seg 8e-4). → **ft1 (chartered now): fine-tune the shipped renderer, same bytes.**
2. **Pose at step zero with the terminal re-solve law** — live in qbr1's config; folded into ft1's loss.
3. **Hard-site (coded-price) weighting** — a per-site weight from the coder's own −log2 p under the shipped HPAC
   prior (or the residual coder's price map): unmeasured as a training weight anywhere. → queued lever
   (`ddm_fb2_hard_site_weighted_margin`) for the born trainer after the burn's adjudication (do not confound
   the running discriminator).
4. **Rate-in-the-loop for the field** (differentiable codelength surrogate under the frozen HPAC prior) — the
   SCMDL/jf1 lineage; bounded by the thin equivalence class (wwc1); low priority.
5. **Class-routed capacity/atoms** — gf2/gc1 closed the static and dyadic forms; a class-matched generator is
   still only plausible as a parametric-motion form (ltg1 priced Lane at 233 KB): parked.
6. **The full-pipeline fold** (fpc1–3): train → lossless tail, with 1–4 inside the trainer instead of as stages.
   The 58 h n600 ticket is the vehicle; its objective should be the fold-back objective, not the PR130 curriculum.

## Divergent ideas I opened and closed at the arithmetic (so nobody re-opens them silently)
- Transmit a boundary-band GT texture code so the render reproduces the judge's jitter: the code must carry at
  least the jitter's entropy (data-processing), so it cannot be cheaper than coding the jitter. CLOSED.
- Ship the judge's bottleneck activations instead of the argmax (the NLA two-hop): inverting them needs scorer
  weights at inflate time. CLOSED by the strict scorer rule.
- Smooth-boundary field + tolerate jitter mismatches: 1 site = 1.273 B-equiv vs ~0.1 B to code (12×). CLOSED.

## Gestalt line
Post-hoc found the laws; training is where they pay. The first fold with a measured exchange rate on the
FRONTIER object is ft1: d_seg 2.01e-4 is manufactured by a renderer never trained against the realized argmax
at its own size — each 1e-5 recovered is −0.001 S at zero bytes (pose must be held: ≤ 1.25e-4 absolute, re-solved).
Own-vehicle frontier: **afr1 S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600]** — unmoved.
