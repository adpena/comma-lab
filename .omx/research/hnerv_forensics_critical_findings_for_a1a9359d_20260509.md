# HNeRV forensics CRITICAL findings for subagent a1a9359d (2026-05-09)

<!-- generated_at: 2026-05-09T05:10:00Z, from_state_hash: hnerv_binary_forensics_pass_complete -->
<!-- HISTORICAL_PROVENANCE — append-only -->

This is a per-subagent handoff. Full forensic dossier:
`/Users/adpena/Projects/pact/.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`

## TL;DR for a1a9359d's lesson catalog

Cite the dossier. The "Thing" is concrete and small.

### THE single load-bearing fact

**PR100, PR101, PR102, PR103 all share BYTE-IDENTICAL DECODER WEIGHTS**, sourced
from PR #95 `hnerv_muon` by AaronLeslie138. I verified this by dequantizing each
PR's archive with each PR's own decoder and comparing per-tensor `absmax`/`std`
to ≥6 decimal places — every tensor across these four PRs matches exactly.

[empirical: `/Users/adpena/Projects/pact/.omx/tmp/hnerv_forensics/extract_weights.py` output]

So the "training secret" they have that we don't is **not in PR100/101/102/103.
It is in PR #95**. And PR #95 was published 2026-05-04T07:47:15Z with the FULL
8-stage training pipeline, the Muon optimizer, the C1a entropy regularizer,
the L7 hard-pixel weighting, the QAT integration, every loss function, every
hyperparameter, and a verified-runs `train.py`. **It was open-source from
07:47 UTC on race day.** Our PR #107 (apogee, 0.229) shipped 4h 23min later
without consuming this work.

### THE concrete training stack (PR #95 / re-emitted in PR #106 src/)

`/Users/adpena/Projects/pact/experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/`

8-stage curriculum (Aaron Leslie's `hnerv_muon`):
1. **Stage 1 (3000 ep, AdamW lr=1e-3, latent_lr=1e-2)**: random init, F.cross_entropy seg + sqrt(10·MSE) pose, 100×seg + 1×pose weighting, EMA 0.999, batch_size=8.
2. **Stage 2 (5650 ep)**: switch to τ-Softplus margin loss (tau=0.3), continue cosine.
3. **Stage 3 (1500 ep, lr=1e-4 fresh cosine)**: switch to smooth-disagreement (sigmoid bell-curve at margin=0).
4. **Stage 4 (500 ep, lr=1e-4)**: same loss + add **QAT** (per-tensor sym INT8 fake-quant, n_levels=127, STE).
5. **Stage 5 (9000 ep, lr=3e-5)**: switch seg loss to L7-weighted Softplus (margin<1 pixels get ×5 boost) + add **C1a entropy regularizer** (cat_entropy_v2 lambda=0.01, sigma=0.2).
6. **Stage 6 (2000 ep)**: bump C1a lambda to 0.02.
7. **Stage 7 (3000 ep)**: sharpen C1a sigma 0.2 → 0.1.
8. **Stage 8 (5000 ep, adamw_lr=1e-5, muon_lr=2e-4)**: switch hidden-conv optimizer to **Muon** (Newton-Schulz orthogonalized momentum, Keller Jordan 2024) with **WD=5e-4** (Chen-Li-Liu arXiv:2506.15054 spectral-norm KKT mechanism).

**Total: ~30,650 epochs at batch_size=8 = 2.3M steps over ~50 hours on one GPU.**

### THE per-secret breakdown (cite these in lesson catalog)

| # | Secret | File:line | Score impact | Confidence |
|---|--------|-----------|---|---|
| 1 | **C1a entropy regularizer** sharpens INT8 weight distribution toward integer grid → collapses brotli entropy floor | `losses.py:79-113 cat_entropy_v2` | Drives the 178 KB → ~155 KB brotli compression curve | HIGH (Aaron's own writeup) |
| 2 | **Muon + WD=5e-4** spectral-norm KKT finetune | `optim.py + stage8_muon_finetune.py:37` | Stage 7 0.2042 → Stage 8 0.2009 = -0.0033 | HIGH (Aaron's stages numbered with scores) |
| 3 | **L7 hard-pixel weighting** | `losses.py:49-62 l7_softplus_seg_loss` | Stage 4 → Stage 5 transition | HIGH |
| 4 | **Smooth-disagreement seg loss** (sigmoid bell at margin=0) instead of CE | `losses.py:39-46` | Stage 2-3 transition (~10 KB rate equivalent) | HIGH |
| 5 | **eval_roundtrip baked into training**: every step does `up=bicubic→down=bilinear→clamp→round STE→YUV→SegNet/PoseNet` | `common.py:179-194` | Closes proxy-auth gap that wrecked our internal NeRV | CRITICAL |
| 6 | **rgb_to_yuv6 differentiability fix** — challenge's `frame_utils.rgb_to_yuv6` is wrapped in `@torch.no_grad()` and uses in-place clamp; pose loss never reaches the decoder. Aaron monkey-patches both `frame_utils.rgb_to_yuv6` AND `modules.rgb_to_yuv6` at import time. | `data.py:51-81` | Without this, **pose plateaus at 142 across 2500+ epochs** (his own bug-class note) | CRITICAL |
| 7 | **Per-pair latent correction sidecar** with hand-built fixed delta vocabulary {±0.01, ±0.02, ±0.03, ±0.04, ±0.05, ±0.06, ±0.08, ±0.10} (PR99/100/101/102/103) chosen post-hoc to minimize joint SegNet+PoseNet distortion. **Adds NO training cost — pure inflate-time perturbation table.** | PR101 `codec.py:68-71`, PR100 `sidecar.py:6-14` | -0.001 to -0.002 score (BradyMeighan PR99 derivation) | HIGH |
| 8 | **Inflate-time channel-bias correction** | PR101 `inflate.py:49-51`: `up[:,0,0].sub_(1.0); up[:,0,2].sub_(1.0); up[:,1,1].sub_(1.0)`; PR102 `inflate.py:118`: `up[:,0,0].add_(1.0)` | Zero-byte runtime-bias hypothesis; PR102 proves runtime constants can move score on a byte-identical archive, PR101-vs-PR103 still needs same-archive offset ablation | HIGH priority / causality pending |
| 9 | **DELTA_SCALE retune 0.0100 → 0.0095** (PR102 only change vs PR100) | PR102 `sidecar.py:14` | PR100 0.1954 → PR102 0.19499 (0.00041 improvement, **third prize**) | EMPIRICAL |
| 10 | **EMA decay 0.999 from epoch 0**, EMA shadow used at eval time + saved as inference checkpoint | `common.py:124-125, 212, 254-255` | Aligned with our CLAUDE.md "EMA non-negotiable" rule | CONFIRMED-WE-DO-THIS |
| 11 | **Best-checkpoint selection by exact archive build at every eval**: every eval_every=25 epochs, the EMA snapshot is run through `build_archive` → `parse_archive` round-trip → real `evaluate_decoder` on the actual video, and the lowest-score archive is saved | `common.py:227-261` | Bridges proxy-vs-auth — **this IS auth eval inside training** | CRITICAL (we explicitly fail this in our 12+ NeRV attempts) |
| 12 | **No noise injection, no dropout, no augmentation** — pure single-video memorization at native (384×512), batch_size=8, 600 latents are `nn.Parameter` learned jointly with weights | `common.py:118-122 + train_stage` | Confirms the architecture is the upper bound; any synthetic data hurts | HIGH |
| 13 | **Per-tensor symmetric INT8, n_levels=127, scale = absmax/127** — uniform quantization across all 28 tensors with no per-channel scaling | `losses.py:120-125 fake_quantize` + `codec.py:146-155 quantize_state_dict` | Simplest possible quant; works because C1a regularizer pre-shapes weights | HIGH |
| 14 | **Decoder architecture: 28 latent dim, 36 base channels, 6 PixelShuffle stages, sin() activation everywhere, dilated-conv refine residual at 0.1× scale, separate rgb_0/rgb_1 heads (frame 0 / frame 1 weights NOT shared)** | `model.py:14-54` (29 lines total) | Sin-activated PixelShuffle is the canonical NeRV cell from Chen 2023 | HIGH |
| 15 | **Sub-byte codec polymorphism (PR101 only)**: 6 sidecar layouts (HUFF_ENUM/HUFF_COMB/HUFF/SPLIT/PACKED/RAW) + per-tensor brotli-stream count + per-tensor zigzag/twos-complement/offset byte mapping + per-tensor 4D conv transpose-permutation hand-tuned | PR101 `codec.py:32-87` | -94 B archive (PR100 178873 → PR101 178158) | HIGH |
| 16 | **Constriction-based arithmetic coding (PR103 only)** for 8 heaviest tensors via per-tensor empirical histograms; latent hi/lo split with hi byte through AC | PR103 `inflate.py:51-58, 123-136` | -750 B vs PR100 (178873 → 178123) at small AC overhead | HIGH |

### Lesson #5 and #6 are the catastrophic ones

These two were the bugs that pinned every internal NeRV/HNeRV attempt:

- **#5 eval_roundtrip baked into the training inner loop** (NOT just at eval): every gradient step rounds and clips the decoded tensor in the same way the contest scorer does. `decoded_clamped = decoded_bhwc.clamp(0, 255); decoded_rounded = decoded_clamped.round(); decoded_bhwc = decoded_clamped + (decoded_rounded - decoded_clamped).detach()` (STE-rounded). Our internal HNeRV/NeRV experiments train against floating-point reconstruction targets; the proxy-auth gap there is the same 2-11× we get on every other path that violates the eval_roundtrip rule.
- **#6 rgb_to_yuv6 monkey-patch** is a SUBTLE foot-gun: the contest's own `frame_utils.rgb_to_yuv6` and `modules.rgb_to_yuv6` are decorated `@torch.no_grad()` and use in-place clamp. If you import them naively, **PoseNet's preprocess_input severs autograd** and pose gradient never reaches the decoder. Aaron explicitly says "pose plateaued at 142 across 2500+ epochs" — that's the same plateau pattern we've seen in our internal NeRV/HNeRV runs.

These are NOT cited anywhere in our internal NeRV literature. They are the SINGLE biggest "thing they did right that we didn't."

### Lessons that map to our existing CLAUDE.md non-negotiables

- **#10 EMA**: covered. ✓
- **#11 best-archive-by-exact-eval inside training**: covered by "auth eval EVERYWHERE" but not implemented in our NeRV path. ✗
- **#5 eval_roundtrip**: covered by CLAUDE.md non-negotiable. We did NOT extend it to NeRV/HNeRV training paths. ✗
- **#6 rgb_to_yuv6 differentiability**: NOT in CLAUDE.md, NOT in any code. NEW non-negotiable. ✗

### What was NOT recoverable from artifacts

- Stage 1's actual training corpus split (we know it's "single video memorization" but not how prev was set up — Aaron's own readme says the script is the ground truth and it doesn't deliberately split anything; the stage1 config uses ALL 600 frame pairs).
- Random seed used for stage1 random init (would let us reproduce the exact weights). Likely Aaron just used seed=1234 (the contest eval default).
- Whether Aaron tried wider/deeper variants and rejected them, or whether 28×36 was first try (no comment in code).
- The exact validation-vs-training split used during training (likely none — overfit is the goal).

### What we should ABSOLUTELY do next

1. **Land lessons #5 and #6 as non-negotiables in CLAUDE.md** before any new NeRV/HNeRV training dispatch.
2. **Run Aaron's `train.py` end-to-end on 4090 ($12.50 estimated)** to establish a reproduced 0.198 baseline. Then we have a substrate to bolt anything onto.
3. **Apply T1 (Ballé end-to-end Lagrangian-ADMM) on top of #2's verified substrate.**
4. Track lessons #7/#8/#9 — they are inflate-time-only constants (1-3 lines of code) that bought medal positions for PR99-PR103.

## Cross-references

- Codex representation-integration gap audit: `.omx/research/representation_integration_gap_audit_20260508_codex.md`
- Substrate-vs-codec meta: `~/.claude/projects/.../feedback_substrate_vs_codec_composition_meta_pattern_20260508.md`
- Operator clarification: `.omx/research/hnerv_retrospective_user_clarification_20260509.md`
- HNeRV cluster CUDA-CPU drift: `~/.claude/projects/.../feedback_cuda_cpu_axis_profile_learning_layer_20260508.md` (R_pose=5.04, R_seg=1.17 — these calibrations all derive from Aaron's substrate too)
