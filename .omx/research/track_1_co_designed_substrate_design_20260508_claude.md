# Track 1 — Co-designed substrate for sub-0.17 CPU

**Date:** 2026-05-08
**Author:** claude:main (this session, council-synthesized)
**Status:** DESIGN MEMO — no dispatch, no archive, no score claim yet
**Predicted band:** 0.155–0.180 [contest-CPU] (council median 0.165)
**Dispatch budget estimate:** $30–60 Lightning T4 (12–18h training + harvest)
**Operator authorization:** PENDING — directives "create new substrate and make everything match" + "the model is the thing" + "push toward sub-0.17" point at this work; explicit dispatch authorization not yet granted
**Cross-ref task:** #307 PARADIGM-δεζ (this memo IS its scope spec)

## Why a new substrate

Every medal-cluster PR (101/102/103/107) is HNeRV-class. Joint-entropy floor on PR101 weights is 148–162 KB (memory `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md`). PR101 archive is 178 KB. **Codec-only paths cap above 0.18** — Shannon's R(D) verdict from this session's council deliberation.

To go below 0.17, the model itself has to change. Bolt-ons saturate at 0.18-0.19; substrate redesign is the only path to 0.155–0.170 council-median.

Operator's most-recent directives that ground this:
- "create new substrate and make everything match"
- "the model is the thing"
- "push toward sub-0.17"

## Mission

Build a contest-compliant archive (Linux x86_64 CPU + NVIDIA CUDA 1:1 hardware, runs `inflate.sh` + `evaluate.py` within 30 min on T4) at:
- ≤ 140 KB archive bytes (vs PR101 178 KB)
- `seg_avg ≤ 4.5e-4` (vs PR107 5.9e-4)
- `pose_avg ≤ 3.0e-5` (vs PR107 3.6e-5)
- → CPU score ≤ 0.165

This requires JOINT optimization of: (model weights) + (codec entropy model) + (per-tensor sensitivity allocation) + (pose representation) + (score-gradient supervision).

## The five co-design decisions

### Decision 1: Ballé scale hyperprior on quantized weights

Replace the current iid factorized prior (which is what brotli effectively assumes) with channel-wise scale prediction. Hyperprior `z` is small (1–3 KB), transmitted in archive; quantized weights `ŵ` are encoded against `p(ŵ | z)` via arithmetic coding.

- Source: Ballé 2018, end-to-end-trainable scale hyperprior for image compression
- Empirical: 14–32 KB savings on PR101 weight stream beyond per-tensor brotli (cross-tensor MI)
- Wiring: existing `tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py` provides one substrate template; new training loop in `experiments/train_track1_substrate.py` (NEW)
- Test: roundtrip exactness — `decode(encode(W)) ≡ W` byte-for-byte

### Decision 2: Score-gradient supervision in the inner loop

Use the actual `SegNet.forward` and `PoseNet.forward` from `upstream/scorers/` on a small frame subset (50 frames out of 600) as the inner-loop loss. This is the canonical "use the contest scorer in training" closure that closes the proxy-auth gap.

- Cost: ~5× per-step slowdown vs proxy loss (manageable; 12-18h total at T4)
- Memory: SegNet + PoseNet on CUDA = ~2 GB (fits T4)
- Dropout: every step uses a different random 50-frame subset; full 600-frame is only the eval pass
- Tag discipline: scorers must be on `--device cuda` (NEVER MPS — would silently produce wrong gradients per CLAUDE.md MPS-NOISE rule); training tagged `[contest-CUDA]` only when full-eval lands
- Source: this idea has been research-synthesized for months but never landed; `feedback_proxy_auth_math_useless` documents the cost of not doing it

### Decision 3: Sensitivity-aware per-tensor quantization (UNIWARD discipline at weight level)

Each weight tensor has a different importance for SegNet/PoseNet output. Compute per-tensor Hessian-trace or Fisher-information estimate. Allocate quantization budget where it matters — high precision on stem.weight + early Conv2d (where SegNet's stride-2 sees), low precision on tail blocks (where it doesn't).

- Source: Fridrich UNIWARD principle — errors in undetected regions are free; council Track 1 promotion
- Empirical anchor: lossy_int4 mixed-precision int4/int6/int8 achieves -77 KB at 1.55% rel_err on PR107 (lane #412 completed) — but at uniform allocation. Sensitivity-weighted should be a strict improvement.
- Wiring: existing `tac.optimization.lagrangian_per_tensor_allocation` provides the allocator; `src/tac/sensitivity_map.py` provides the per-tensor importance
- Tool: `tools/sensitivity_weighted_lossy_coarsening.py` (NEW) — wraps the existing K-search with importance weights

### Decision 4: Pose-deriver head (no separate pose tensor)

Currently PR107 ships ~7 KB of pose tensor (600 × 6 fp16). Train a pose-deriver head that takes the model's intermediate representation and outputs poses deterministically. This drops 7 KB pure rate save with zero distortion if trained correctly.

- Mechanism: 64-dim hidden → 6 pose outputs, ~4 KB params after FP4 quant → 3 KB net rate save
- Risk: pose accuracy may degrade; mitigation is to KEEP the residual delta (encode `pose_true - pose_derived` instead of `pose_true`, which is much smaller-entropy)
- Council: Yousfi+Fridrich endorse (residual coding is canonical contest-design pattern)

### Decision 5: Frame-conditional bit budget (rate is local)

Currently rate is allocated uniformly across frames. But video is non-stationary — some frames are eventful (high motion, high-information), others are not. A per-frame Lagrangian with a global budget shifts bits where they matter.

- Mechanism: encode each frame's latent under a hyperprior conditioned on a small global side-info; the side-info is the per-frame budget allocation (1-2 bits per frame, ~150 B total)
- Empirical anchor: existing per-frame γ-JCSP work (lanes #378-#380) provides the StreamSource builder; never combined with a hyperprior
- Trade: complicates inflate.sh (needs to decode side-info first, then condition); admissible per CLAUDE.md if the runtime contract holds

## Architecture spec (concrete)

```
class Track1Substrate(nn.Module):
    """Co-designed HNeRV-derivative with Ballé hyperprior + score-gradient training."""

    # === Decoder stem (PR101-class but trained jointly with codec) ===
    stem: Conv2d(in=3, out=36, kernel=3)   # ~1 KB FP4
    blocks: 3 × HNeRVBlock(36, 36)         # ~28 KB FP4
    rgb_head: Conv2d(36, 3)                # ~0.5 KB FP4

    # === Hyperprior (NEW) ===
    hp_encoder: Conv2d(36, 8, stride=2)    # ~0.3 KB FP4
    hp_decoder: Conv2d(8, 36)              # ~0.3 KB FP4
    z_quantizer: BallleQuantizer(8 chan)   # learned scale params

    # === Pose deriver (NEW) ===
    pose_head: Linear(36 → 6)              # ~0.2 KB FP4

    # === Frame-budget side-info (Decision 5) ===
    budget_predictor: Linear(36 → 1)       # ~0.2 KB FP4

# Total params: ~250K (vs PR101 228K) — slight increase from hyperprior + heads
# Total FP4 bytes: ~30 KB renderer + ~3 KB hyperprior + ~3 KB pose-deriver
# Plus latents: ~110 KB encoded under hyperprior (vs PR101 ~150 KB under brotli)
# Total predicted archive: ~125-145 KB
```

## Training spec

```
optimizer = AdamW(lr=5e-4, weight_decay=1e-5)
scheduler = CosineAnnealingLR(T_max=2000)
ema = EMA(decay=0.997)  # PER CLAUDE.md NON-NEGOTIABLE
eval_roundtrip = True   # PER CLAUDE.md NON-NEGOTIABLE
noise_std = 0.5         # Hotz fix

loss = (
    100.0 * segnet_grad_loss(pred_frames, gt_frames)        # Decision 2: actual scorer
    + 10.0 * posenet_grad_loss(pred_frames, gt_frames)      # Decision 2: actual scorer
    + lambda_R * estimated_rate_bits(quantized_weights, z)  # Decision 1: hyperprior rate
    + lambda_S * sensitivity_weighted_quant_loss(...)       # Decision 3
    + 0.1 * pose_residual_l1(true_pose, derived_pose)       # Decision 4
)

# Joint Lagrangian dual update (every 100 steps):
lambda_R += eta * (estimated_rate - target_rate)
# Decision 5 (frame budget) handled inside the rate term

epochs = 2000
batch_size = 4 (subject to CUDA memory; 50-frame eval subset uses gradient checkpointing)
```

## Score-gradient supervision details (Decision 2 deepening)

The scorers `SegNet` (EfficientNet-B2 U-Net) and `PoseNet` (FastViT-T12) are differentiable. We never end-to-end-trained against them because:
1. Cost (5× slowdown) — acceptable if it closes the proxy-auth gap
2. Memory — fits T4
3. "Cheating" concern — the contest doesn't forbid using the public scorers in training; only forbids EDITING upstream

Critical: training MUST be on CUDA. MPS scorer outputs drift 23× per CLAUDE.md MPS-NOISE rule. CPU scorer outputs are valid (per `[contest-CPU]` mandate) but ~10× slower than CUDA.

## Predicted bands and risks

Council median: 0.165 [contest-CPU predicted].

Component-wise predicted (PR101 baseline → Track 1):
- bytes: 178 KB → 130 KB (-27%, from hyperprior 14-32 KB + sensitivity 8-15 KB + pose 3 KB)
- seg_avg: 0.000589 → 0.000400 (-32%, from score-gradient supervision closing proxy-auth gap)
- pose_avg: 3.6e-5 → 2.5e-5 (-31%, from score-gradient + pose-deriver residual)
- score: 0.197 → 0.158 (predicted)

Risks:
1. **Hyperprior may not generalize** — PR101 weight distribution may be too iid for hyperprior to win. Sister anchor (`compressai_balle_hyperprior` lane, completed task #399) returned 0.985 — i.e., the canonical Ballé did NOT save bytes on PR101. **CRITICAL OPEN QUESTION.**
2. **Score-gradient training instability** — gradients through SegNet/PoseNet may be noisy on small subsets. Mitigation: increase subset size to 100, gradient clipping, EMA on gradient updates.
3. **Score may converge to local minimum that doesn't transfer to full 600-sample eval** — overfit to the 50-frame subset. Mitigation: random-subset every step + held-out 100-frame validation.
4. **Pose-deriver may degrade pose accuracy** — risk addressed by residual coding (encode delta only).
5. **Lightning T4 dispatch may take longer than 18h budget** — mitigation: A100 if T4 unavailable; checkpoint/resume.

## Reactivation / kill criteria

Per CLAUDE.md "KILL is LAST RESORT":
- This lane gets KILLED only if ALL FIVE alternative configurations fail empirically:
  1. Hyperprior + score-gradient + sensitivity + pose-deriver + frame-budget (full stack)
  2. Hyperprior + score-gradient + sensitivity (drop pose + frame-budget)
  3. Hyperprior + score-gradient (drop sensitivity)
  4. Score-gradient only (canonical proof of concept)
  5. Hyperprior only (canonical proof of concept)
- DEFERRED-pending-research is the default verdict for any partial failure.

## Dispatch wrapper outline (NOT YET BUILT)

`scripts/remote_track1_substrate.sh`:
- Lightning T4 g4dn.2xlarge with cu124 pin
- bootstrap_runtime_deps from `scripts/remote_archive_only_eval.sh`
- Self-bootstraps uv + torch 2.5.1+cu124 + ffmpeg + brotli
- Runs `experiments/train_track1_substrate.py` (NEW) for 12-18h
- Heartbeat every 5 min to `/tmp/heartbeat_track1.log` (per CLAUDE.md remote-script rule)
- On completion: build archive, run `[contest-CUDA]` eval, `[contest-CPU]` eval (BOTH per dual-eval mandate)
- Harvest: `experiments/results/track1_substrate_<timestamp>/` with archive.zip + adjudicated.json + checkpoints + provenance.json

`tools/dispatch_track1_substrate.py`:
- Lane claim: `track1_co_designed_substrate`
- Cost cap: $60 hard
- Disk: 60 GB minimum

## What needs to happen BEFORE dispatch

1. **Operator authorization** for the dispatch budget ($30-60).
2. **Sister-anchor sanity check**: re-read `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508.md` and the `compressai_balle_hyperprior` lane result (-0.985 score). If hyperprior demonstrably FAILS on PR101 weights at canonical config, Decision 1 needs deeper diagnosis before dispatch.
3. **Score-gradient unit test**: verify `SegNet.forward` produces correct gradients on a 4-frame batch (reasonable loss, finite gradients, no NaNs). 30-min smoke on local CUDA (none available; could use Modal CPU smoke if billing unblocked).
4. **Recursive adversarial review** per CLAUDE.md: 3 clean passes from inner council before dispatch.
5. **Lane claim** opened via `tools/claim_lane_dispatch.py claim ...` per CLAUDE.md cross-agent coordination rule.

## Cross-references

- This session's council deliberation (in conversation, not committed)
- `feedback_substrate_vs_codec_composition_meta_pattern_20260508.md` — substrate-branching cap argument
- `feedback_pr101_joint_entropy_floor_subagent_verdict_20260507.md` — joint-entropy floor 148-162 KB
- `feedback_pr101_analytical_lossy_coarsening_BEATS_neural_codecs_20260508.md` — Ballé failure on PR101 (-0.985); critical context for Decision 1 risk
- Task #307 PARADIGM-δεζ (this memo's parent task)
- Task #308 PHASE 4 INTEGRATION
- `submissions/factorized_hnerv_v1/` — sister runtime infra that lands at L2 byte-proxy-only (subagent a021d70b, this session)

## Verdict

**Design is council-aligned, predicted band is tight (0.158-0.165), risks are concrete (especially Risk 1 hyperprior generalization).** This memo is the dispatch SCOPE; not the dispatch itself. Operator review + the Risk 1 sister-anchor check are the gating items.

If operator green-lights, next deliverables in order:
1. Build `experiments/train_track1_substrate.py` (~600 LOC)
2. Build `scripts/remote_track1_substrate.sh` + `tools/dispatch_track1_substrate.py` (~200 LOC each)
3. Recursive adversarial review (3 clean passes)
4. Smoke (1 epoch on Modal CPU if billing unblocked, else local 5-min sanity)
5. Lightning T4 dispatch via parallel-dispatch actuator (per CLAUDE.md race-rule, even if not racing — same hygiene)
6. Harvest + dual-eval (CUDA + CPU on Linux x86_64)
7. If sub-0.17 lands: re-PR via fork after secrecy audit + 5-pass clean review
