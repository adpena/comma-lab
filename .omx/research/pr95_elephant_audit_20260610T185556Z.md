# THE PR95 ELEPHANT — audit verdict (task #75)

UTC 2026-06-10 · claude (subagent `pr95_elephant_audit_75`) · `[contest-CPU advisory]` /
`exact_cpu_advisory` / `metric_family=exact_evaluate`. NON-promotable; `score_roadmap_update_eligible=False`,
`mechanism_update_eligible=True`. The decisive measurement is a real exact-evaluate number recomputed from
components; it directs the next experiment, it does not move the score roadmap.

---

## LEAD ANSWER

**Do WE reproduce ~0.193 on PR95's OWN released bytes? — YES, EXACTLY.**

Decoding PR95's RELEASED archive (`archive.zip` sha `e976acd5…`, the `0.bin` inside has sha `4b8013fb…`,
178,417 bytes — the PR #95 "hnerv_muon" 0.199 leaderboard submission) through PR95's own `hnerv_muon`
decoder, then running OUR exact upstream `evaluate.py` pipeline (CPU, GT via `frame_utils.yuv420_to_rgb`,
`seq_len=2` non-overlapping, SegNet argmax-disagreement, PoseNet pose-MSE), recomputed from components over
all 600 samples:

| term | OUR eval on released bytes | PR95 PR-body claim | match |
|---|---|---|---|
| d_seg  | **0.00061217** | 0.00061212 | bit-exact (6 dp) |
| d_pose | **0.00003494** | 0.00003494 | exact |
| rate   | 0.00475202 | 0.00475202 | exact |
| **final_score** | **0.19871** | 0.1987 | exact |

Score recomputed: `100*0.00061217 + sqrt(10*0.00003494) + 25*0.00475202 = 0.06122 + 0.01869 + 0.11880 = 0.19871`.
Artifact: `.omx/tmp/pr95_elephant/decisive_result.json`. Harness: `.omx/tmp/pr95_elephant/decisive_check.py`
(rebuildable; raw frames on `/Volumes/VertigoDataTier/pact/pr95_elephant_decisive/` with REBUILD_MANIFEST).

**THE ELEPHANT IS NOT MEASUREMENT (hypothesis (a) is FALSIFIED).** Our eval is correct. The d_seg≈0.50 wall
is a REAL training failure, not a measurement artifact.

---

## THE ELEPHANT: hypothesis (b) — a BROKEN/INERT score-aware training loop (implementation-level)

Our faithful B1 clean-PR95 reproduction (`/Volumes/VertigoDataTier/pact/b1_229k_clean_20260609T085348Z`,
lane `lane_hi_nerv_mlx_score_aware_local_20260602`, 229K decoder, 600 pairs, the full PR95 8-stage
score-aware curriculum scaled 3000ep) has its **exact** d_seg PINNED at ~0.5048 across the ENTIRE run, with
exact eval rows at every 250-epoch checkpoint:

| ep | exact d_seg | exact d_pose | bytes | score | stage / loss_form |
|---|---|---|---|---|---|
| 250  | 0.504823 | 155.75 | 256072 | 90.12 | stage1 ce_seg_loss |
| 500  | 0.504845 | 165.01 | 255192 | 91.28 | stage2 tau_softplus |
| 750  | 0.504824 | 157.90 | 254364 | 90.39 | stage2/3 |
| 1000 | 0.504824 | 157.90 | 254364 | 90.39 | stage3 smooth_disagreement |
| 1500 | 0.504100 | 167.33 | 254181 | 91.49 | stage5 l7_softplus |
| 2000 | 0.504100 | 167.33 | 254181 | 91.49 | stage5 |
| 2500 | 0.504499 | 167.58 | 254138 | 91.56 | stage7 |
| 2750 | 0.638556 | 166.91 | 254318 | 104.88 | stage8 (worse) |
| 3000 | 0.504824 | 157.70 | 254012 | 90.36 | stage8 muon |

**The metric is FLAT (and pose is ~4.5-million× worse than PR95's 0.0000349).** The curriculum runs to
completion — all 8 stages including the L7 hard-pixel stages and the Muon final stage — and d_seg never
leaves 0.50. This single fact disposes of TWO hypotheses:

- **(c) under-train is FALSIFIED as the mechanism.** A flat curve does not descend with more epochs. PR95's
  29,650 vs our 3000 (9.9× fewer) is real, but a curve pinned at 0.5048 from ep250→ep3000 would not reach
  0.0006 at ep29,650 — there is zero downward trend to extrapolate. (PR95's stage-1 CE alone ran 3000 ep =
  our ENTIRE run; even that stage drives PR95's decoder argmax-correct. Ours does not move in 3000 ep.)
- **(d) wrong-objective is FALSIFIED for the B1 clean run.** B1 DID run the score-aware curriculum
  (`ce_seg_loss → tau_softplus_seg_loss → smooth_disagreement_seg_loss → l7_softplus_seg_loss`,
  per its launch manifest + telemetry `loss_form`), NOT pure recon-MSE. (The separate F1 recon-ablation
  "21.74 dB plateau" probe DID use pure RGB-MSE — that is a different, correctly-diagnosed probe; the F1
  verdict already flagged MSE's mean-field minimizer. F1 is not the elephant; it was never the score-aware
  run.)

### The mechanistic smoking gun: the seg loss never descends

B1 telemetry (`telemetry.jsonl`, 3000 rows) — the SegNet seg loss across the run:

| ep | loss_seg | loss_pose | grad_norm | loss_form |
|---|---|---|---|---|
| 0    | 1.1590 | 157.23 | 53,425    | ce_seg |
| 280  | 1.1706 | 2.83   | 94,751    | ce_seg (stage-1 end) |
| 1000 | 1.4933 | 7.18   | 1,694,336 | smooth_disagreement |
| 2000 | 1.6136 | 5.20   | 1,566,377 | l7_softplus |
| 2999 | 1.6117 | 22.55  | 6,809,149 | l7_softplus (muon) |

**`loss_seg` NEVER decreases — it drifts UP from 1.16 to 1.61.** A working CE-against-GT-argmax objective
descends toward ~0 as frames become argmax-correct (that is exactly how PR95's stage-1 CE pulls the decoder
to argmax-correct frames). Ours is stuck. Meanwhile `grad_norm` is astronomical (5.3e4 → 6.8e6),
hard-clipped to 1.0 EVERY step (per the ep614 strict-scrutiny finding: "grad-clip fires 100% of steps") —
so the effective optimizer step is a tiny fraction of a near-random direction. The objective is COMPUTED
but the optimization is INERT. The checkpoint selector held `best_epoch000286` (stage-1 end) for the entire
3000-ep run: the EMA-best never improved past the very first stage.

This is the canonical **inactive-objective / Mistake-B** bug class the Vehicle OS already names
(`docs/vehicle_operating_system.md:81-83`: "the shared MLX harness silently defaulted the SegNet/PoseNet
distillation weights to 0.0 — so 'score-aware' runs trained recon-MSE-only (SNeRV ep22399 d_seg=0.71)").
Our B1 run is the SAME pattern at the implementation level: the loss has the PR95 names but the gradient
through the scorer is not productively reducing argmax disagreement.

---

## WHAT PR95 ACTUALLY DOES (the d_seg mechanism — exact line-cited)

Source: `…/source/submissions/hnerv_muon/src/{train.py, stages/common.py, losses.py, data.py, codec.py,
model.py, inflate.py}` (the PR #95 head, sha `9bdce26f…`).

1. **The objective IS the evaluator metric — directly, through the LIVE frozen scorer.** In
   `stages/common.py` (the shared training loop), every batch:
   ```python
   posenet_in, segnet_in = distortion_net.preprocess_input(decoded_bhwc)
   seg_out  = distortion_net.segnet(segnet_in)          # LIVE frozen SegNet forward on the rendered frame
   pose_out = distortion_net.posenet(posenet_in)        # LIVE frozen PoseNet forward
   seg_l  = cfg.seg_loss_fn(seg_out, seg_targets_hard[idx])   # CE/softplus/smooth ON SegNet logits
   pose_l = sqrt(10 * F.mse_loss(pose_out['pose'][:,:6], pose_targets[idx]))
   loss   = 100*seg_l + 1*pose_l + cat_lambda*entropy
   loss.backward()                                      # gradient flows render -> frozen SegNet -> render params
   ```
   There is **NO recon-MSE term anywhere** (`losses.py` confirms: every stage loss is a SegNet-logit margin
   surrogate + pose-MSE + optional C1a entropy). The seg loss is the SegNet argmax-disagreement, smoothed:
   - `ce_seg_loss` = `F.cross_entropy(seg_logits, targets_hard)` (stage 1)
   - `tau_softplus_seg_loss` / `smooth_disagreement_seg_loss` = sigmoid/softplus on the **target-minus-runnerup
     margin** — "Bell-curve gradient peaks at margin=0 — pushes boundary pixels across the line" (`losses.py`).
   - `l7_softplus_seg_loss` = the same margin loss with a 5× weight boost on hard pixels (`margin < 1.0`).

2. **The targets ARE the GT scorer's own outputs** (`data.py::precompute_targets`, lines 109-119):
   ```python
   f = yuv420_to_rgb(frame)                  # GT decode == eval decode
   po, so = distortion_net(pair)             # GT through the frozen scorer
   seg_targets_hard.append(so.argmax(dim=1)) # GT SegNet per-pixel argmax = the exact d_seg reference
   pose_targets.append(po['pose'][:, :6])    # GT PoseNet 6-dim output
   ```
   So PR95 minimizes "render's SegNet argmax disagrees with GT's SegNet argmax" = literally the d_seg the
   evaluator charges. The pairing (`prev, f; prev=None`) is non-overlapping consecutive pairs = `seq_len=2`.

3. **The eval roundtrip is in the inner loop** (`common.py`): render @ 384×512 → bicubic up to 874×1164 →
   bilinear down to 512×384 → clamp → round (straight-through) — the exact uint8 path the evaluator applies,
   so the optimizer sees the rounded frames the scorer will see.

**How PR95 gets argmax-correct frames:** it does NOT chase pixel fidelity (its 21-ish-dB-class recon is
blurry — that is irrelevant). It directly pushes every SegNet decision-boundary pixel across the class line
via the margin loss, and pins the 6 PoseNet output dims to the GT pose. The frames only need to be "good
enough that SegNet argmax matches and PoseNet output matches" — a far lower bar than visual fidelity, and a
bar that recon-MSE (mean-field blur) never targets. d_seg=0.0006 means 99.94% of SegNet argmax pixels match.

---

## THE ACTIONABLE FIX (feeds #63 / #74 / #71)

The fix is NOT "more epochs" (#c, flat curve) and NOT "add a recon term" (#d). It is: **make the score-aware
gradient actually descend the SegNet/PoseNet objective on OUR carrier.** Concretely, port PR95's loop
mechanism faithfully — the part our MLX harness is missing or has inert:

1. **Backprop the seg loss through the LIVE frozen scorer on each rendered frame**, with GT-scorer-argmax
   targets — PR95's `seg_out = distortion_net.segnet(render); ce(seg_out, gt_argmax)`. If our MLX path uses a
   precomputed-teacher KL-distillation with a learnable student head (per `mlx_score_aware/bundle.py`
   docstring) and the student-head/teacher coupling is what is inert, that is the defect: a distillation
   surrogate whose KL stays ~constant is not the same gradient signal as direct CE through the live SegNet.
   **Verify the SegNet/PoseNet objective weights are explicit + nonzero AND the loss actually descends**
   (per Vehicle OS objective-activation rule + `check_score_aware_run_has_nonzero_scorer_objective_weights`).
2. **Fix the optimization, not just the objective.** grad_norm 5e4→7e6 hard-clipped to 1.0 every step =
   the LR/clip regime is wrong for this loss surface. PR95 uses AdamW peak 1e-3 (stage 1), latent_lr 10×,
   cosine-to-5e-6, grad_clip 1.0 — but its grad norms are sane because the loss is well-conditioned. A loss
   that produces 1e6 grad norms is not the PR95 loss; re-derive the exact margin loss + STE-round + the
   exact preprocess (interpolate 512×384 bilinear, rgb_to_yuv6) so the gradient is well-scaled.
3. **The decisive re-probe (cheap, $0, mechanism):** on the fixed loop, score the LIVE render's EXACT d_seg
   (argmax disagreement) at ep50/ep250 — NOT a proxy/PSNR. If loss_seg descends AND exact d_seg drops below
   ~0.50 → the loop is fixed; continue. If loss_seg descends but exact d_seg stays 0.50 → the surrogate is
   not aligned with argmax (margin-loss-to-argmax mismatch). If loss_seg does not descend → still inert.
4. **Fallback per the F1/reference-carrier fork:** if our 388 KB-scaffolded MLX carrier cannot be made to
   descend the score-aware loss, vendor PR95's ~300-500 LOC HNeRV loop verbatim (it is in-repo at
   `…/hnerv_muon/src/`, proven to reach 0.193) rather than continuing to patch the sketch — the Vehicle OS
   "vendor-a-faithful-carrier over endlessly patching" branch.

---

## Authority + bookkeeping

- The 0.19871 number is `[contest-CPU advisory]` (macOS-CPU, `--device cpu`, recomputed from components on
  the exact released bytes). It is NOT 1:1 with the Linux-x86_64 `[contest-CPU]` leaderboard axis, but it
  matches PR95's published GHA-ubuntu CPU 0.1987 to 4 dp — strong evidence our CPU eval path is faithful.
- NO MPS used. GT decode ONLY via `upstream/frame_utils.yuv420_to_rgb` (the AVVideoDataset path).
- Hypotheses verdict: **(a) FALSIFIED** (eval correct), **(c) FALSIFIED-as-mechanism** (flat curve),
  **(d) FALSIFIED for B1** (B1 ran score-aware curriculum), **(b) CONFIRMED** (broken/inert score-aware loop:
  loss_seg never descends, grad_norm pathological, exact d_seg pinned at 0.50 through all 8 stages).
- Disk hygiene: 3.66 GB decoded raw frames on `/Volumes/VertigoDataTier/pact/pr95_elephant_decisive/` with
  `REBUILD_MANIFEST.json` (rebuildable from `archive.zip` + `decisive_check.py`; safe to delete).

## Cross-refs
`b1_clean_pr95_ep1000_verdict_psnr_is_not_d_seg_20260609.md` (the ep1000 read that this trajectory completes
to ep3000) · `b1_f1_recon_ablation_verdict_skip_recon_inert_under_mse_20260609.md` (the SEPARATE pure-MSE
probe — not the elephant) · `docs/vehicle_operating_system.md:81-83` (the inactive-objective / Mistake-B bug
class this confirms on HiNeRV) · `b1_229k_clean_20260609T085348Z/STRICT_SCRUTINY_ep614_finding.md` (the
grad-clip-100%-of-steps + "proxy loss_seg stable != d_seg good" finding this audit confirms at ep3000).
