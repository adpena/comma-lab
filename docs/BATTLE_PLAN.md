# Battle Plan: April 15 - May 3 (18 days)

Status: Contest-compliant auth=0.87, Unlimited-compute auth=0.37, Quantizr leads at 0.33.

## LATEST BREAKTHROUGHS

### Pose-Space TTO (April 15)
- **seg_weight=0**: Pure PoseNet optimization via FiLM conditioning vectors.
- **Result**: -94.7% PoseNet distortion (0.031 -> 0.0016) in 500 steps.
- **Auth eval (ep300)**: 0.61 (contest-compliant, no scorers at inflate time).
- **Archive cost**: 14.4 KB (600 poses x 6 x float32). Rate: 0.0004.
- Per-pair: 6 values optimized instead of 707M pixel values.

### Distillation Trajectory
- **proxy 0.375 at ep550** (from 0.807 at ep0). Still converging.
- Config: pose_weight=10, seg_weight=100, hinge loss, eval roundtrip, FiLM pose_dim=6.
- Warm restart from Phase 2 checkpoint.

### Five Moonshots (implemented April 15)
1. **Embedding-Space TTO** (`experiments/optimize_embedding.py`): Optimize the renderer's
   shared nn.Embedding(5,6) -- 30 values that control class appearance. GLOBAL optimization
   (one pass over all pairs), compounds with pose TTO. Archive cost: 120 bytes.
2. **Pre-Computed Gradient Corrections** (`experiments/precompute_gradient_corrections.py`):
   Compute full d(score)/d(pixel) at compress time, sparsify top 5%, quantize to int8,
   compress with zlib. ONE-STEP TTO at inflate time without any scorer. Expected ~50-100 KB.
3. **Embedding + Pose TTO pipeline**: First optimize embedding (global), then optimize
   poses (per-pair) with the improved embedding. Compounds both gains.
4. **Constrained Generation from Noise** (existing): No renderer needed, direct pixel
   optimization from class-mean initialization.
5. **Distillation**: Train renderer to reproduce TTO frames in single forward pass.

### Council Binding Decisions
- Hinge loss is mandatory for SegNet (25% better than xent at 500 steps).
- seg_weight=0 for pose-space TTO (pure PoseNet lane).
- Embedding optimization is GLOBAL, must precede per-pair pose optimization.
- Gradient corrections at inflate time are contest-compliant (no neural weights).

### Revised Timeline
| Days | Action | Target |
|------|--------|--------|
| 1-2 | Embedding TTO smoke test + full run | Validate 30-value optimization |
| 2-3 | Gradient corrections full run on 4090 | Measure archive size and score delta |
| 3-5 | Combined pipeline: embedding + pose + corrections | Target sub-0.30 proxy |
| 5-7 | Distillation convergence + auth eval | Contest-compliant sub-0.50 |
| 7-9 | FiLM integration into renderer training (v6) | Renderer conditioned on pose |
| 9-11 | DSConv + capacity sweep | Rate improvement |
| 11-13 | Archive compression, final auth eval | Final score |
| 13-18 | Integration, submission PR, paper | May 3 deadline |

## NEXT DEPLOYMENT: Distillation (Lane 1 priority)

**Goal:** Train renderer to reproduce v7 TTO frames (auth 0.37) in a single forward pass.
This eliminates TTO compute at inflate time, making the 0.37 result contest-compliant.

**Script:** `experiments/train_distill.py`
**Vast.ai registry:** `distill_full` (6h timeout, RTX 4090)

**Prerequisites:**
1. Extract GT poses: `PYTHONPATH=src:upstream python -m tac.pose_extraction --upstream upstream/ --output experiments/results/gt_poses.pt --device mps`
2. Verify TTO frames exist: `experiments/results/tto_v7_hinge_500/tto_frames.pt`
3. Verify renderer checkpoint: `experiments/results/v5_lagrangian_renderer/renderer_best.pt`

**Deploy command:**
```bash
python scripts/check_vastai.py run <id> distill_full
```

**Expected cost:** $1.50 (6h x $0.25/hr)
**Expected result:** auth 0.45-0.55 (gap between pure distillation and TTO)

---

## The Three Lanes

---

### Lane 1: Contest-Compliant (Goal: auth < 0.40)

**What must change:** The renderer itself must produce better frames. TTO cannot help here
(exceeds 30-min inflate budget). We need the same techniques Quantizr used:
FiLM on pose vectors, eval-matched resize in training, depthwise separable convs.

**Day-by-day schedule:**

| Days | Action | Expected Result |
|------|--------|-----------------|
| 1-2 | FiLM integration into renderer training | Renderer conditioned on pose vectors |
| 2-3 | Add simulate_eval_roundtrip to training loss | Compensates for quantization blur |
| 3-4 | Train renderer v6: FiLM + eval roundtrip (Vast.ai 4090, 4h) | Proxy < 0.6 |
| 4-5 | Auth eval v6 | Target auth < 0.60 |
| 5-7 | Latent code optimization at compress time (9.6KB) | Auth < 0.50 |
| 7-9 | Depthwise separable + capacity increase sweep | Auth < 0.45 |
| 9-11 | Archive compression: prune + entropy code | Rate improvement |
| 11-13 | Integration, final auth eval, submission PR | Final score |

**Training config (v6):**
```bash
python3 -u experiments/train_tac.py \
  --profile proven_baseline \
  --pair-mode asymmetric \
  --renderer-pose-dim 6 \
  --simulate-eval-roundtrip \
  --lr 0.001 \
  --epochs 5000 \
  --save-every 500 \
  --resume experiments/results/v5_lagrangian_renderer/renderer_best.pt
```

**Kill criteria:**
- If proxy > 1.0 after 2000 epochs: FiLM integration is broken, debug
- If auth > 0.80 after full training: technique isn't helping, pivot to latent codes only
- If rate > 0.15 after latent codes: archive too large, reduce latent dim

---

### Lane 2: Unlimited Compute (Goal: auth < 0.20)

**Purpose:** Paper scalability story. Shows what's achievable with more compute.

**Config: TTO v7 (all breakthroughs combined):**
```bash
PYTHONPATH=src:upstream python3 -u experiments/renderer_tto.py \
  --checkpoint renderer_best.pt \
  --device cuda \
  --n-frames 1200 \
  --tto-steps 500 \
  --tto-lr 0.005 \
  --batch-pairs 10 \
  --seg-weight 100 --pose-weight 10 --compress-weight 0.5 \
  --use-embedding-loss --seg-odd-only \
  --early-stop-patience 500 \
  --simulate-resize --eval-roundtrip \
  --segnet-loss-mode hinge --hinge-margin 0.5
```

**Expected cost:** $0.75 (3h on Vast.ai 4090)
**Expected result:** auth 0.25-0.35

**If TTO v7 < 0.30:** Add two-phase TTO:
```bash
  --tto-phase2-segnet-only --phase2-steps 200
```

**If TTO v7 > 0.35:** Try cosine LR schedule:
```bash
  --lr-schedule cosine
```

**Kill criteria:**
- Proxy > 0.50: something fundamentally wrong, check gradients
- PoseNet contribution > 0.20: embedding loss not helping, check dimensions
- SegNet contribution > 0.10 with hinge: hinge loss broken, fall back to xent

---

### Lane 3: Research Maximum (Goal: auth < 0.25)

**Purpose:** Prove theoretical floor for the paper.

**Stack everything:**
1. FiLM-conditioned renderer (from Lane 1) as warm-start
2. Latent codes optimized at compress time (16-dim per frame)
3. TTO with hinge + eval roundtrip + embedding + cosine LR
4. Per-pair difficulty-adaptive step counts
5. Archive re-quantization with entropy coding

**Architecture sweep (if time permits):**
- Depthwise separable convs (88K params like Quantizr)
- Deeper MaskRenderer (depth=2)
- Larger embed_dim (8 or 12)

**Run order:**
1. Run Lane 2 experiment first (cheapest, highest signal)
2. Start Lane 1 renderer retraining in parallel
3. Lane 3 only if Lanes 1+2 show clear promise by Day 7

---

## Deployment Checklist (Per Experiment)

1. **Build fresh bundle:**
   ```bash
   bash scripts/build_deploy_bundle.sh
   ```
   Verify: pyproject.toml, src/tac/checkpoint.py, experiments/*.py all present.

2. **Verify checkpoint:**
   ```python
   from tac.checkpoint import verify_checkpoint_identity
   verify_checkpoint_identity("experiments/results/v5_lagrangian_renderer/renderer_best.pt")
   # Must return "cff8dca4"
   ```

3. **Create instance:**
   ```bash
   python scripts/check_vastai.py create
   ```

4. **Deploy:**
   ```bash
   python scripts/check_vastai.py deploy <id>
   ```

5. **Verify remote:** SSH in and check:
   ```bash
   python scripts/check_vastai.py ssh <id> "python3 -c 'import torch; print(torch.cuda.get_device_name(0))'"
   python scripts/check_vastai.py ssh <id> "which ffmpeg"
   python scripts/check_vastai.py ssh <id> "ls /workspace/upstream/models/"
   python scripts/check_vastai.py ssh <id> "echo \$PYTHONPATH"
   ```

6. **Upload checkpoint:**
   ```bash
   # rsync the renderer_best.pt separately (excluded from deploy by *.pt rule)
   rsync -avz -e "ssh -p <port> -o StrictHostKeyChecking=no" \
     experiments/results/v5_lagrangian_renderer/renderer_best.pt \
     root@<host>:/workspace/renderer_best.pt
   ```

7. **Run experiment:**
   ```bash
   python scripts/check_vastai.py run <id> tto_v7_hinge_roundtrip
   ```

8. **Monitor:** Check logs every 5 min:
   ```bash
   python scripts/check_vastai.py ssh <id> "tail -20 /workspace/experiments/results/*/results.json 2>/dev/null || echo 'still running'"
   ```

9. **Download results BEFORE destroying:**
   ```bash
   python scripts/check_vastai.py download <id> /workspace/experiments/results ./experiments/results/vastai_run_N
   ```

10. **Destroy instance:**
    ```bash
    python scripts/check_vastai.py destroy <id>
    ```

11. **Commit results with full provenance:**
    ```bash
    git add experiments/results/vastai_run_N/
    git commit -m "vastai: tto_v7_hinge_roundtrip results (auth=X.XX)"
    ```

---

## Red Flags and Expected Results

| Symptom | Diagnosis | Action |
|---------|-----------|--------|
| Baseline PoseNet > 1.0 | WRONG CHECKPOINT | Abort. Verify MD5 = cff8dca4 |
| Baseline SegNet > 0.01 | Wrong model or masks | Check mask codec, verify 5 classes |
| Score > 2.0 | Catastrophic failure | Abort immediately, check device |
| Batch time > 60s | Potential OOM | Reduce --batch-pairs to 5 |
| Process dies mid-batch | Hard pair or OOM | Resume from checkpoint, skip batch |
| NaN in loss | Usually all-zero gradients | Check scorer differentiability |
| GPU util < 5% after start | Setup stalled | SSH in, check .setup_done file |
| Cost > $6 for single run | Timeout or stuck | Destroy instance immediately |

**Healthy run indicators:**
- Baseline proxy: 0.5-0.7 (with simulate_resize)
- TTO progress: PoseNet should drop 5-10x by step 200
- Batch time: 30-45s on 4090 (10 pairs, 500 steps)
- Total time: 2-3h for full 1200-frame run

---

## Budget Allocation

| Lane | Platform | Estimated Cost | Max |
|------|----------|---------------|-----|
| Lane 1 (contest) | Vast.ai 4090 | $2.00 (8h training) | $4.00 |
| Lane 2 (unlimited) | Vast.ai 4090 | $0.75 (3h TTO) | $2.00 |
| Lane 3 (research) | Vast.ai 4090 | $2.00 (combined) | $4.00 |
| Auth evals (all) | Modal T4 or local | $0 (free) | $2.00 |
| Contingency | - | - | $4.00 |
| **Total** | | **$4.75** | **$16.00** |

Remaining budget: ~$24 (full Vast.ai cap). Plenty of room.

---

## Critical Reminders

1. **NEVER conflate lanes.** Label every score [contest-compliant] or [unlimited-compute].
2. **Always pass --simulate-resize** in TTO experiments. Without it, proxy is 5000x wrong.
3. **Always verify checkpoint MD5** before any experiment. Wrong checkpoint wastes hours.
4. **Destroy instances immediately** when done. Idle 4090 = $0.25/hr wasted.
5. **Commit everything.** Git history IS the research timeline.
6. **Council approval** required for any design tradeoff. Bug fixes can be immediate.
