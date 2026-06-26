# DAG FEED-bs — MARGIN-WEIGHTED LIVE d_seg loss WIRED (actuates FEED-bp NEW-B)

**Trigger:** actuate the BRIDGE finding (FEED-bp, `deepmath_multiscale_bridge_hunt_20260626.md`,
commit 49ee33647): the d_seg gap is a capacity-ALLOCATION problem, not a shortage. MEASURED:
the decision-boundary annulus is **2.26%** of pixels; **~89% of d_seg lives in the bottom-5%
margin**; witness-hard pixels margin **0.42 vs 5.79 global (14x)**. The margin map is a per-pixel
bit-allocator with a **~20x effective-capacity multiplier**. Re-route the SAME bytes onto the annulus.

**Authority:** `[macOS-CPU advisory] NON-PROMOTABLE` (CPU-torch realized smoke, NOT the 600-sample
harness). Pointer UNMOVED **contest-CPU 0.19110** — this is a TRAINING-LOSS lever wired + $0-smoke
proven; it does NOT move the score until a margin-weighted real-RGB witness byte-closes + exact-evals.
NO-FAKE: the d_seg VERDICT stays the **unweighted realized argmax-disagreement** through R; ONLY the
training loss is re-weighted. Memory-safe (104 GB free >> 40 GB floor); NO GPU arm (iso baseline pid
37977 owns the single slot, untouched; fleet HARD-BLOCKED #173).

## WIRED (experiments/train_witness_realized_through_R_mlx.py)
- `_live_margin_weight(seg_logits, fn, temp)` (new module-level helper): LIVE margin = top1-top2 logit
  gap of the FROZEN-SegNet logits of the witness frame1 **rendered through R** (the SAME tensor the seg
  loss already scores — no extra forward). Returns a (...,H,W) weight map, **mean-1 normalized**
  (re-allocates, does NOT add budget) + **mx.stop_gradient** (NO-FAKE: an allocation PRIOR, not a
  differentiable knob — the witness must FIX the flip, not game its own margin).
- Allocators: `inverse` = 1/(1+m/temp); `exp` = exp(-m/temp); `bottom-k` = 1[m<=quantile(m,temp)]
  (temp = bottom-fraction, scale-invariant).
- Wired into `make_loss_fn` (per-pair, the live training path) CE **and** margin_hinge branches, and into
  `make_loss_fn_batch` CE branch. Flags: `--margin-weighted-loss` (BooleanOptionalAction, **default OFF**),
  `--margin-weight-fn {inverse,exp,bottom-k}`, `--margin-weight-temp`. Logged in the `loss_mode` line + result config.

## $0 CPU SMOKE (real SegNet, real GT pair, realized render THROUGH R; mlx-device cpu)
Fraction of total loss-WEIGHT landing on the bottom-5%-LIVE-margin annulus:
- uniform baseline: **5%** (definitional); WITHOUT-flag default CE×GT-margin weight: **1.3–6.2%**
  (random-init CE barely correlates with the live margin).
- WITH flag: `exp` t0.5 **8%** -> t0.2 **36%** -> t0.05 **100%**; `bottom-k` 0.05 / 0.02 **100%**;
  `inverse` weak on random-init (margins ≫ temp). => a clean monotone re-routing of the SAME budget to
  the annulus: **5% -> 100%** (the ~20x multiplier), scale-invariant for `bottom-k`.
- loss RUNS finite ON (400.04) and OFF (332.64). DEFAULT byte-identical: OFF == manual original formula
  (Δ 2.7e-6 = numpy-reimpl roundoff; OFF path structurally identical via the `if margin_weighted:` gate).
- STOP-GRADIENT (NO-FAKE): max|grad| through the weight = **0.0**.
- VERDICT unchanged: `cpu_verdict_d_seg` == manual unweighted argmax-disagreement (0.50692…); verdict fn
  has NO loss flag.

## CAPACITY-ROUTING (#9 boundary_routing.py) — FOLLOW-ON
`BoundaryFiLM` KKT-waterfill is a **torch nn.Module** on signed-distance maps + needs a SEG-margin
saliency field (the composition-map #10 gap is that saliency is currently POSE-side). Porting it into the
MLX witness is a cross-framework build. The margin-weighted loss already captures most of the ~20x
(soft capacity-routing on the loss side, recomputed live each step → self-correcting on newly-born flips).
Routing = the next build once the margin-weighted witness descends.

## CAPSTONE LAUNCH (the moment the iso baseline hits GATE1; do NOT launch into the single slot now)
```
.venv/bin/python tools/safe_run.py --rss-mb 28000 --timeout 100000 -- \
  .venv/bin/python -u experiments/train_witness_realized_through_R_mlx.py \
  --out-dir experiments/results/witness_margin_weighted_n600 \
  --num-pairs 600 --epochs 300 --eval-every 10 --render-h 384 --render-w 512 \
  --hidden-dim 112 --n-hidden 3 --mod-dim 32 --n-fourier 24 \
  --basis isotropic --activation hosc --hosc-beta-start 1.0 --hosc-beta-end 6.0 \
  --chroma --seg-loss ce \
  --margin-weighted-loss --margin-weight-fn bottom-k --margin-weight-temp 0.05 \
  --score-domain-loss --w-seg 100 --w-pose 1 --pose-eps 1e-2 \
  --grad-clip 1.0 --accum-pairs 8 --n-restarts 2 --lr 1e-3 --lr-schedule \
  --warmup-epochs 1 --lr-end 1e-4 --ema-decay 0.997 --int8-verdict \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --mlx-device gpu --seed 0
```
Real-RGB realized witness (render 384x512, ~89KB base_ch20-class, rate ~0.06) + chroma + hosc step-native
basis + bottom-5% live-margin re-allocation. Stored-pose sidecar (`src/tac/scorer_targets.py`) composes
at byte-close (pose → 0.017 SOLVED, frees all capacity to d_seg). TARGET d_seg < 7.3e-4 at rate 0.0594 →
**S ≈ 0.149** (corridor A). `exp --margin-weight-temp 0.1` is the smooth alternative (keeps an interior
floor) if the hard bottom-k mask lets the interior drift.

**Cross:** FEED-bp (bridge) · FEED-bo (synergy map: lever #9 routing, #10 saliency gap, #6/#7 hosc/chroma) ·
`capstone_synergy_composition_map_20260626.md` top-5 #2.
