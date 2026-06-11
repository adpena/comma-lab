# PR95 seg-convergence mechanism spec + recipe-gap audit of our reproduction (2026-06-11)

**Authority discipline (binding).** Every d_seg number here is `[macOS-CPU advisory]` /
`[macOS-MLX research-signal]`, **NON-PROMOTABLE** (`promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`). torch-CPU `evaluate.py` (600-sample, Linux x86_64) is the ONLY
leaderboard authority. NO MPS. NO paid dispatch fired. **Frontier pointer UNMOVED: 0.19109982
[contest-CPU] — ABOVE T_1.** This is a recipe-correctness diagnosis + a controlled $0 experiment, not a
pointer move.

---

## PART 1 — The PR95 seg-convergence mechanism (which ingredient converges d_seg)

### 1.1 The existence proof we hold
PR95's HNeRV/Muon decoder reaches **d_seg ≈ 5.6e-4** (the basin) and the frontier (sha b46897267d) is a
PR95-class decoder at the basin. So seg-convergence on this contest IS PROVABLY SOLVABLE. The 8-stage
curriculum is the proven path; the question this memo answers is *which stage/ingredient does the d_seg
heavy-lifting*, and whether our reproduction faithfully reproduces it.

### 1.2 The 8-stage 29,650-epoch curriculum (exact spec, verified against
`src/tac/mlx_pr95_port/curriculum.py::_PR95_8STAGE` ↔ `profile_pr95_hnerv_muon_intake.md`)

| # | name | epochs | seg_loss | sigma_wn | c1a_λ | qat | opt / lr |
|---|---|---:|---|---:|---:|---|---|
| 1 | stage1_v328_ce | 3000 | `ce_seg_loss` | 0.0 | 0.0 | no | AdamW 1e-3 |
| 2 | stage2_v331_softplus | 5650 | `tau_softplus(0.3)` | 0.0 | 0.0 | no | AdamW 1e-3 |
| 3 | stage3_v332_smooth | 1500 | `smooth_disagree(0.3)` | 0.0 | 0.0 | no | AdamW 1e-4 |
| 4 | stage4_v332_qat | 500 | `smooth_disagree(0.3)` | 0.0 | 0.0 | **YES** | AdamW 1e-4 |
| 5 | stage5_c1a_l7 | 9000 | `l7_softplus(0.3)` | 0.2 | 0.01 | YES | AdamW 3e-5 |
| 6 | stage6_lambda_sweep | 2000 | `l7_softplus(0.3)` | 0.2 | 0.02 | YES | AdamW 3e-5 |
| 7 | stage7_sigma_sweep | 3000 | `l7_softplus(0.3)` | 0.1 | 0.02 | YES | AdamW 3e-5 |
| 8 | stage8_muon_finetune | 5000 | `l7_softplus(0.3)` | 0.1 | 0.02 | YES | **Muon 2e-4** + AdamW 1e-5 |

### 1.3 The seg-loss functions (verified bit-faithful in `live_segnet_loss.py`)
All four operate on the per-pixel **margin** `m = target_logit − max_other_class_logit` through the LIVE
frozen SegNet (NOT extracted masks). `d_seg = mean(argmax(live) != gt_argmax) = mean(m < 0)`.

- **CE** (`ce_seg_loss`): `F.cross_entropy(logits, gt_argmax)`. Gradient is non-vanishing for ALL
  mis-classified pixels (even confidently-wrong ones, `m ≪ 0`), so it does the **bulk descent** from
  d_seg 0.5 → ~0.02. **This is the workhorse.**
- **tau_softplus** (`τ·softplus(−m/τ)`): smooth hinge; gradient `−sigmoid(−m/τ)` non-vanishing for
  `m < 0`, decaying for `m ≫ 0`. Refines the boundary band below CE's floor (in our run: 0.02 → 0.0097).
- **smooth_disagreement** (`sigmoid(−m/τ)`): the soft indicator whose minimizer literally IS d_seg.
  BUT its gradient `−(1/τ)·sigmoid(−m/τ)(1−sigmoid(−m/τ))` is a **bell curve peaking at m=0 and VANISHING
  for both `m ≪ 0` (confidently-wrong) AND `m ≫ 0` (confidently-right)**. So it can ONLY nudge pixels
  already near the boundary; it CANNOT fix a confidently-wrong pixel. This is the key to why it "raises
  d_seg" on a weak/throttled basis — see §2.3.
- **l7_softplus**: tau_softplus with a `(1+4·[m<1])` hard-pixel boost (mean-renormalized) — concentrates
  capacity on pixels still near/over the line.

### 1.4 The other ingredients
- **Muon (L15)** — orthogonalized SGD via Newton-Schulz on the hidden conv weights (177K of 229K
  params). PR95 uses it ONLY in stage 8. NS makes the update magnitude **grad-norm-INDEPENDENT** (≈ unit
  per matrix), so the **step size is dominated by `muon_lr`, not the gradient norm**. This is THE key
  fact for the recipe bug (§2.1): a tight grad-norm clip barely changes the muon DIRECTION (NS
  re-normalizes), but `muon_lr` directly scales the step.
- **C1a coder-aware reg (L16)** — entropy/brotli-friendliness regularizer on decoder weights, λ 0.01→0.02
  in stages 5-8. A RATE lever (brotli-friendly weights), NOT a d_seg lever.
- **sigma noise (L17)** — Gaussian weight-noise 0.2→0.1 in stages 5-8, simulating the uint8 quant
  roundtrip. A robustness/QAT lever, NOT the primary d_seg descent.
- **QAT (stage 4+)** — INT8 fake-quant in the forward so the int8 archive's d_seg ≈ the training d_seg.
- **EMA** — exports the averaged shadow (lower-variance, lower-d_seg point). MUST use warmup decay
  (`min(decay,(1+t)/(10+t))`) or the shadow lags init on short runs (the f771e6e00 fix; correct here).
- **eval_roundtrip** — the bridge applies bicubic-up → bilinear-down → uint8 STE in BOTH the loss and the
  d_seg/d_pose measurement (correct here: `TorchScorerBridge(..., eval_roundtrip=True)`).

### 1.5 VERDICT on which ingredient converges d_seg
**The d_seg heavy-lifting is done by (a) CE in stage 1 [0.5 → ~0.02] then (b) the
tau_softplus/l7 margin-surrogate refinement + the muon-finetune step-size in stage 8.** The C1a / sigma /
smooth_disagreement stages do NOT lower d_seg (the smooth_disagreement gradient vanishes off-boundary;
C1a/sigma are rate/robustness levers). The basin (5.6e-4) requires the *combination* of a non-vanishing
boundary-pushing surrogate AND **enough optimizer step-size for enough epochs** to push every flip-prone
boundary pixel across the line. **The dominant knobs are: the seg-loss gradient shape (CE/l7, not
smooth_disagreement-alone) and the muon step-size (muon_lr) × epoch budget.**

---

## PART 2 — Recipe-gap audit: is our reproduction faithful? (every bug/gap)

The reproduction is `experiments/run_capstone_campaign.py` → `CapstoneTrainer` → curriculum runner. I
audited it against the §1 spec on the REAL `modules.py` SegNet (the loss IS through the live frozen
scorer — confirmed, the #76 fix is **REAL**). The bugs are NOT in the loss; they are in the
**optimizer-LR / clip wiring through `configure_stage`**.

### 2.1 BUG-A (decisive): the curriculum silently DROPS the working muon_lr (150× too small).
`run_capstone_campaign.py` defaults `--muon-lr 3e-2 --grad-clip 50 --grad-clip-muon 50` (the values the
muon-only arm used to reach 0.0037). These flow into `CapstoneTrainConfig`. BUT when the curriculum runs,
`CapstoneTrainer.configure_stage(spec, ...)` **rebuilds `self.opt_config` from the StageSpec**, which
hardcodes `muon_lr=2e-4` (PR95's torch stage-8 value) and `grad_clip_muon=1.0`. So:
- the CLI `--muon-lr 0.03` is **ignored** for the curriculum; every stage trains Muon at **2e-4 = 150×
  smaller** than the value that empirically works on this small MLX basis.
- `grad_clip_muon=1.0` ⇒ **`clip_would_fraction = 1.000` every epoch, every stage** (the conv-weight
  gradient norm is always > 1).

**This is the confound the operator named.** PR95's 2e-4 muon_lr was tuned for a 178K-param net over 5000
stage-8 epochs; on the 85K tied basis at compressed epochs, it crawls. The muon-only arm (`curriculum=none`,
`muon_throughout`, muon_lr=0.03, grad_clip=50) reached **d_seg 0.0037**; the curriculum (muon_lr=2e-4,
clip=1.0) plateaued at **0.0101** — *the curriculum is WORSE than no curriculum*, which is backwards and is
the signature of a recipe bug, not a curriculum benefit.

### 2.2 BUG-B: the cosine `eta_min_ratio` floors the LR at 0.1667 against the WRONG denominator.
`pr95_cosine_lr_scale` computes `eta_min_ratio = max(lr_floor_ratio/base_lr, 1e-3)`. The runner passes
`base_lr = adamw_lr` (3e-5 in stages 5-8), so `eta_min_ratio = max(5e-6/3e-5, 1e-3) = 0.1667`. This SAME
cosine scales the **muon_lr** too (line 2578). So even the muon-only arm has its LR floored at
0.1667×muon_lr from ~ep90, prematurely halting the d_seg descent (the descent ratio is still +0.05/10ep at
the floor — it was LR-starved, NOT a capacity asymptote). The "muon asymptotes ~0.0025" claim in the
adversarial-correction memo is **partly an artifact of this LR floor**, not a clean capacity reading.

### 2.3 BUG-C (the "smooth_disagreement raises d_seg" symptom): a CONSEQUENCE of A+B, not a standalone bug.
In c1prime stage 3, d_seg went 0.00968 → 0.01016 under `smooth_disagreement`. Mechanism: its gradient
**vanishes off-boundary** (§1.3), so it can only refine pixels near m=0. When the optimizer step is already
throttled (muon 2e-4 + clip 1.0 + cosine floor), and stage 3 switches from a non-vanishing surrogate
(tau_softplus) to a vanishing one, the only updates that DO fire are weak boundary nudges that perturb
already-won pixels slightly the wrong way → d_seg drifts up. So smooth_disagreement is **mis-staged for the
small-basis muon_throughout regime** (in PR95 it runs at AdamW 1e-4 with a fully-trained stage-2 basis, not
a throttled muon basis). Not a code bug in the loss — a curriculum-fit gap.

### 2.4 What is CORRECT (NOT bugs) — the #76 fix IS real.
- **The loss is margin-through-LIVE-SegNet, NOT extracted masks.** `live_segnet_loss.py` is a verified
  bit-faithful port; the gradient flows `render → frozen SegNet → render params`. **#76 ("FIX THE INERT
  SCORE-AWARE LOOP") is genuinely done** — the inert KL-student-head surrogate is gone; the exact d_seg
  DESCENDS (0.5 → 0.0097 in c1prime stage 1-2). #76's *claim* is honest; what #76 did NOT do is tune the
  curriculum's muon_lr/clip for the small MLX basis (BUG-A/B), which is why the wall persisted.
- **EMA warmup decay** is correct (f771e6e00; `tac.ema_warmup.warmup_ema_decay`).
- **eval_roundtrip** is on in the bridge for both loss and measurement.
- **The decoder is PR95-bit-exact** (PixelShuffle + bilinear-skip + sin; verified prior memo).
- **QAT / C1a / sigma mechanisms** are wired per spec (`StageMechanisms`).
- **Live vs EMA d_seg agree** post-warmup-fix (the shadow tracks live; no shadow-lag confound).

### 2.5 Is the c1prime "stuck 0.010" a real plateau or an artifact?
**An artifact of BUG-A+B+C**, not a capacity plateau. The min (0.00968) is reached at stage-2 end (the only
stages with a non-vanishing surrogate at a usable AdamW LR), then BUG-C drifts it up and BUG-A/B prevent any
muon stage from recovering. The c1prime run also DIED before stage 8 (the one stage with the muon-finetune
lever under the PR95 schedule), so the curriculum's own d_seg lever was never exercised.

---

## PART 3 — The fixes (see commit)
1. **Fix BUG-A**: under `muon_throughout`, `configure_stage` uses the config's `muon_lr` / `grad_clip_muon`
   (the working 0.03 / 50) instead of the StageSpec's torch-tuned 2e-4 / 1.0. The seg-loss family /
   QAT / C1a / sigma / AdamW-LR schedule (the actual curriculum STRUCTURE) is unchanged — only the muon
   step-size + clip that the `muon_throughout` deviation already owns. Parity-tested: `pr95_adamw_then_muon`
   (the faithful schedule) is byte-unchanged.
2. (BUG-B left as a documented knob, not changed by default — fixing it requires re-pointing the cosine
   denominator and would change the faithful schedule; the controlled test instead runs ENOUGH epochs that
   the floor is reached late.)

---

## PART 4 — The decisive controlled $0 smoke (see PART 5 results)
Hold architecture FIXED (base_ch=20, tie_depth=2, n=48, stored_latent, int8 — identical to bc20_p48), vary
ONLY recipe correctness: run the CORRECTED curriculum (muon_throughout + muon_lr 0.03 + clip 50) and read
whether d_seg breaks below the muon-only ~0.0037 toward the basin.
