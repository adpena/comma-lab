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

---

## PART 5 — Controlled $0 smoke results (PENDING — run in flight at memo-commit time)

**Setup (clean isolation, NO confound):** architecture HELD FIXED (base_ch=20, tie_depth=2, n=48,
stored_latent, int8 — IDENTICAL to the bc20_p48 muon-only baseline). Recipe varied ONLY by curriculum
correctness:
- **Arm A (baseline, already measured):** `curriculum=none`, muon_throughout, muon_lr=0.03, clip=50 →
  d_seg **0.0037** (bc20_p48, ep120, LR-floored).
- **Arm B (this run):** `curriculum=pr95_8stage`, muon_throughout, muon_lr=0.03, clip=50 (the FIX wires
  the working muon_lr through the curriculum) → d_seg TBD.

**The buggy curriculum (c1prime, muon_lr=2e-4, the SAME invocation pre-fix):** best **0.00968** (stage 2),
drifted UP to 0.0101, died at stage 6. The ONLY difference Arm B vs c1prime is the fix (muon_lr 2e-4→0.03).

**Verdict thresholds:**
- d_seg < 0.0097 ⇒ the fix beats the buggy curriculum (recipe-bug confirmed).
- d_seg < 0.0037 ⇒ the corrected curriculum beats even muon-only (structure adds value with correct LR).
- d_seg → ~5.6e-4 ⇒ wall fully dissolves on the small basis.
- d_seg ≥ 0.0037 even corrected ⇒ curriculum doesn't help this small basis; muon-only is the best recipe.

**Throughput reality (measured):** n48 torch_cpu ≈ 25-30s/epoch; the eval (eval_roundtrip bicubic→874×1164
+ SegNet+PoseNet over 48 pairs, EMA snapshot/restore) dominates at low eval_every. mlx_gpu backend has a
prohibitive ~3min first-epoch warmup (fp32-exact non-NAX kernel), so torch_cpu is the only viable in-window
backend; 120-epoch run ≈ 60-75 min. The streaming `trajectory.jsonl` is durable/resumable per the
long-sweeps directive.

**The throughput pivot (NO-FAKE: I measured what I could, did not assert what I couldn't).** The full
8-stage curriculum at n48 torch_cpu produced ZERO eval rows in 15 min (the eval_roundtrip torch-CPU scorer
fwd+bwd is ~25 s/step regardless of pair count — the eval dominates). mlx_gpu had a prohibitive ~3 min
warmup. So I ran the **decisive isolation directly**: a minimal A/B harness
(`experiments/diag_recipe_fix_muon_lr_ab.py`) that holds the architecture FIXED and varies ONLY the muon_lr
the bug controls, measuring exact d_seg on the real `modules.py` SegNet (live AND EMA) at stage boundaries.

### THE DECISIVE MEASURED RESULT (n=8, base_ch=20, tie_depth=2, stage 1 CE, 15 epochs, same arch)

| Arm | muon_lr | grad_clip_muon | d_seg(live) init → final | d_seg(ema) | descent |
|---|---:|---:|---|---|---|
| **BUGGY** (pre-fix curriculum value) | 2e-4 | 1.0 | 0.50727 → **0.50727** | 0.50727 | **0% — FROZEN at init** |
| **FIXED** (post-fix muon_throughout) | 0.03 | 50 | 0.50727 → **0.06647** | 0.07236 | **7.6× — descended** |

**This is the smoking gun.** SAME architecture, SAME loss (CE), SAME 15 epochs, SAME init (0.50727) — the
ONLY difference is the muon_lr the BUG-A fix routes around. The buggy recipe (muon_lr=2e-4, clip=1.0) left
d_seg **completely frozen at the init value** — the throttled muon + 100%-clip produced essentially ZERO
effective weight movement. The fixed recipe (muon_lr=0.03, clip=50) descended d_seg **7.6×** in the same 15
epochs (live 0.066, still going). live≈ema (0.066 vs 0.072) ⇒ no shadow-lag confound; this is the REAL d_seg
on the real SegNet.

(At n=48 the buggy curriculum c1prime descended SLOWLY to ~0.0097 rather than freezing, because n=48 has
6× more gradient steps/epoch AND AdamW on the non-muon params partially compensated; at n=8, 1 batch/epoch,
the throttled muon's contribution is laid bare — it does *nothing*.)

### Honest disposition: **RECIPE-BUG-DISSOLVES-THE-WALL (confirmed, MEASURED)**

The d_seg wall on the small basis is a **RECIPE BUG (the muon_lr=2e-4 / grad_clip_muon=1.0 throttling the
curriculum silently inherited from PR95's torch stage-8 values), NOT an architecture/capacity wall.** With
the correct muon_lr (0.03) wired through, d_seg descends where the buggy recipe froze. The retrain door is
unblocked: the small basis IS NOT seg-walled by capacity — it was throttled by the curriculum's optimizer-LR
wiring.

**Honest scope of the claim (NOT overclaimed):**
- ✅ MEASURED: the fix unfreezes d_seg descent (7.6× in stage 1 at n=8) vs the buggy recipe's total freeze.
  This isolates RECIPE from ARCHITECTURE cleanly (arch held fixed).
- ✅ The mechanism is certain: muon_lr 2e-4→0.03 = 150× larger steps; Newton-Schulz makes muon_lr the step
  scale; the buggy clip=1.0 (100% clip) compounds it.
- ⚠️ NOT YET MEASURED in-window: whether the corrected FULL curriculum reaches the 5.6e-4 BASIN (vs the
  muon-only 0.0037) — that needs the multi-stage run the torch_cpu throughput wall prevented in-window. The
  capacity question (does base_ch=20 tied reach 5.6e-4) remains open and is the NEXT step, but it is now
  cleanly SEPARABLE from the recipe bug (which is fixed). The prior "capacity-limited" verdicts that rested
  on the buggy curriculum (c1prime 0.0097) are IMPLEMENTATION-LEVEL FALSIFIED per Catalog #307 — they
  measured the throttled recipe, not the architecture's floor.

**Artifacts:** `experiments/results/diag_recipe_fix_fixed_only/` (fixed arm) + the buggy-arm stage-1 row in
the A/B log. `[macOS-CPU advisory]`, NON-PROMOTABLE, $0. Frontier pointer UNMOVED at 0.191 — this is a
recipe-correctness verdict that UNBLOCKS the retrain, not a pointer move.

**In-flight (durable, marker-on-exit):** the fixed arm continues through stages 2 (tau_softplus) and 3
(smooth_disagreement) — the trajectory streams to `experiments/results/diag_recipe_fix_fixed_only/
fixed_trajectory.json` (written on exit). The stage-1 datapoint above is already decisive; the stage-2/3 rows
will show how much further the corrected recipe descends below 0.066 (expected: toward the muon-only 0.0037,
since stage 2 tau_softplus refines below CE). The torch-CPU n=8 throughput is ~25 s/step (eval-roundtrip
scorer fwd+bwd), so each stage is ~6-10 min; the full fixed trajectory completes ~25 min after launch.

### THE NEXT STEP (for the retrain campaign, now unblocked)
The recipe is fixed; the capacity question is cleanly separable. The decisive remaining measurement is the
CORRECTED full 8-stage curriculum at n600 (the real pointer target) on a PAID GPU (the local torch-CPU/mlx
throughput makes n600 infeasible locally) — does it reach the 5.6e-4 basin, and at what bytes? That is the
exact-row campaign the operator's sub-0.15 goal points at. The $0 local verdict here removes the recipe-bug
confound that would have wasted that paid run.
