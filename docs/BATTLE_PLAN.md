# Battle Plan: April 15 - May 3 (12 days remaining)

Status: Contest-compliant auth=0.61 (pose TTO + distillation), Unlimited-compute auth=0.37, Quantizr leads at 0.33.

## LATEST RESULTS (as of 2026-04-15)

### Score Scoreboard
| Lane | Score | Proxy | Notes |
|------|-------|-------|-------|
| Unlimited-compute | **0.37** | 0.195 | TTO v7, hinge, 500 steps |
| Contest-compliant | **0.61** | 0.446 | Distillation ep300 (auth eval done) |
| Contest-compliant | **0.61** | — | Pose-space TTO ep300 |
| Contest-compliant baseline | 0.87 | 0.807 | Renderer only |
| Distillation (running) | ~0.47* | 0.338 | Epoch 900, still converging |
| Quantizr (threat) | 0.33 | — | PR#55, FiLM+DSConv+eval resize |

*Projected from proxy trajectory

### Pose-Space TTO (April 15 — CONFIRMED)
- **seg_weight=0**: Pure PoseNet optimization via FiLM conditioning vectors.
- **Result**: -94.7% PoseNet distortion (0.031 -> 0.0016) in 500 steps.
- **Auth eval (ep300)**: 0.61 [contest-compliant]. No scorers at inflate time.
- **Archive cost**: 14.4 KB (600 poses x 6 x float32). Rate: 0.0004.
- **Insight**: 196K:1 compression of optimization space. PoseNet-SegNet orthogonality in FiLM space.

### Distillation v2 Trajectory (RUNNING — Vast.ai RTX 4090)
- **proxy 0.338 at ep900** (from 0.807 at ep0). Still converging, no plateau.
- **Auth at ep300**: 0.61 [contest-compliant].
- Config: pose_weight=10, seg_weight=100, hinge loss, eval roundtrip, FiLM pose_dim=6.
- Warm restart from Phase 2 checkpoint. pose_weight=10 was THE critical fix.

### FP4 Export (DONE — 0.085 free rate points)
- Renderer FP32 ZIP: 297 KB → FP4: 170 KB. Saves 0.085 score points.
- FP4 + CRF30 masks: 215 KB total archive. Saves 0.113 vs FP32 baseline.
- No training changes required.

### Gradient Corrections (DEPRIORITIZED — too large)
- Measured: 743 KB for 20 frames → projected ~44 MB for 1200 frames.
- Rate cost ~1.19 (catastrophic). Not viable for production.
- Fundamental issue: gradient signal too dense to sparsify at 5%.

### Mini-Scorer Results (DONE)
- **MiniSegNet (h=32)**: 98.7% fidelity PASSES. Archive: 87 KB (FP16).
- **MiniPoseNet**: FAILS. R²=0.002. Architecture bottleneck, not hyperparameters.
- Workaround: store GT pose targets (14.4 KB) instead.

### Five Moonshots (implemented April 15 — status updated)
1. **Embedding-Space TTO** (`experiments/optimize_embedding.py`): 30 values. Archive: 120 bytes. PENDING validation.
2. **Pre-Computed Gradient Corrections**: DEPRIORITIZED — 743KB for 20 frames, too large.
3. **Embedding + Pose TTO pipeline**: PENDING — embedding first, then per-pair poses.
4. **Constrained Generation from Noise**: PENDING — gated behind Lane 2.
5. **Distillation**: RUNNING — proxy 0.338 at ep900, auth 0.61 at ep300.

### Council Binding Decisions (confirmed)
- Hinge loss is mandatory for SegNet (25% better than xent at 500 steps).
- seg_weight=0 for pose-space TTO (pure PoseNet lane).
- Gradient corrections: DEAD. Archive too large (43MB projected).
- FP4 export: APPROVED. 170KB renderer, 215KB total archive.
- MiniSegNet (87KB) for inflate-time TTO — viable.
- MiniPoseNet: DEAD. Use stored pose targets instead.

### Updated Timeline (12 days to deadline)
| Days | Action | Target | Status |
|------|--------|--------|--------|
| 0 | Distillation ep900 auth eval | auth ~0.47 | RUNNING |
| 1-2 | Distillation ep1000+ auth eval | auth ~0.45 | NEXT |
| 2-3 | Pose TTO on distilled renderer | auth ~0.35 | PLANNED |
| 3-4 | FP4 archive + auth eval | Rate -0.085 | PLANNED |
| 4-6 | MiniSegNet inflate TTO | auth ~0.30 | PLANNED |
| 6-8 | Combined: distilled + pose TTO + FP4 | auth ~0.28 | PLANNED |
| 8-10 | Embedding TTO validation | Compound gain | PLANNED |
| 10-12 | Integration, final auth eval, PR | Final score | DEADLINE |

**Budget status**: ~$15 remaining (of $24 cap).

## NEXT DEPLOYMENT: Distillation auth eval at ep900+

**Goal:** Auth eval of current distillation run (proxy 0.338 at ep900).

**Expected:** auth ~0.47 (proxy 0.338 × 1.4x ratio).

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
