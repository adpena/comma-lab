# PR95 hnerv_muon — 8-stage curriculum forensic recovery (F1, 2026-05-13)

**Lane**: `lane_f1_pr95_8stage_reproduce_20260513` (registered L0 → working to L1+).
**lane_class**: substrate_engineering — PR95 curriculum primitive/forensics; archive grammar belongs to the downstream substrate packet.
**Source clone**: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/` (also mirror in `kaggle_mirror/` and `public_pr_intake_full/`). The `public_pr95_intake_20260504_codex/pr95_src/` clone only contains `.pyc` files; the `_auto` clones carry the canonical `.py` source.
**Council G provenance**: `896f1d79` (council: first-principles original score-lowering verdict TRIPLET phi).
**Predicted score**: 0.20-0.21 [contest-CUDA] if reproduced byte-faithfully (matches PR95 public T4 anchor 0.21 [contest-CUDA] / 0.197 [contest-CPU]). PR101 (gold, 0.193) and PR100 (silver, 0.195) layer on top of PR95's substrate; PR101 has **NO training script** of its own — it is codec-only on top of PR95-shaped weights.

## Operator-facing TL;DR

PR95 is the **only checked-in HNeRV-family training script in the corpus**, and we have not yet reproduced it internally. It is the missing canary anchor that every downstream HNeRV-family lane (PR97/98/100/101/102/103) presumes.

The 8 stages live at `submissions/hnerv_muon/src/stages/stage{1..8}_*.py` (each declares a `make_config(...) -> StageConfig`); the shared training loop is `src/stages/common.py` (275 LOC; one canonical `train_stage(cfg, device, video_path, shared_state)` function). The orchestrator is `src/train.py` (78 LOC) which builds 8 configs in series, threads `prev_stage_output_dir` through, and emits a single `submission_archive/0.bin` at the end via the codec stage.

## Architectural baseline (the substrate the stages train on)

- **Decoder**: `HNeRVDecoder` (`src/model.py`, 55 LOC). 229K params. Latent_dim=28, base_channels=36, eval_size=(384,512). Stem Linear(28→C·6·8), 7-stage channel taper `[C, C, C, int(0.75C), int(0.58C), int(0.5C), int(0.5C)]`, 6 upsample blocks (Conv→`PixelShuffle(2)` + bilinear-skip + `sin(x + identity)`), dilated `refine` block (`Conv3x3 dilation=2 → Conv3x3`), separate `rgb_0` / `rgb_1` Conv heads (`sigmoid * 255`). Output is `(B, 2, 3, 384, 512)` in `[0, 255]`.
- **Codec** (`src/codec.py`, 181 LOC; brotli q=11):
  - Decoder weights: per-tensor symmetric INT8 (n_quant=127), zigzag-encoded, concat with shape+scale metadata, brotli q=11.
  - Latents: per-dim min/max scale to `[0,254]` uint8, 1st-order temporal delta, zigzag uint16, lo/hi byte split, brotli q=11.
  - Archive layout: `[meta_brotli_len:u32][meta_brotli][decoder_blob_len:u32][decoder_blob][latents_brotli_len:u32][latents_brotli]`.
  - Round-trip bit-exact on INT8 weights (asserts in `codec_stage.py`).
- **Inflate** (`submissions/hnerv_muon/inflate.py`, 72 LOC; `inflate.sh`, 24 LOC; 3-positional-arg contract `$1=data_dir $2=output_dir $3=file_list`; no scorer imports; per-video loop over `file_list`).

## The 8 stages

Total ≈ 29,650 epochs at batch 8. Stage 1 ran 3000 epochs (canonical PR95 ran 10K, resumed mid-cosine for Stage 2 from ep3000). Total wall-clock ≈ 50 hr on a single GPU per `src/train.py` docstring.

| Stage | File | Epochs | Seg loss | Optimizer | LR | C1a λ | C1a σ | QAT | Resume from |
|------:|:-----|-------:|:--------|:---------|:-----|-------:|------:|:----|:------------|
| 1 | `stage1_v328_ce.py` | 3000 | `ce_seg_loss` (`F.cross_entropy(logits, hard_targets)`) | AdamW only | 1e-3 → 5e-6 cosine | 0.0 | 0.2 | no | random init |
| 2 | `stage2_v331_softplus.py` | 5650 | `tau_softplus_seg_loss(tau=0.3)` | AdamW only | continues Stage 1 cosine | 0.0 | 0.2 | no | Stage 1 final |
| 3 | `stage3_v332_smooth.py` | 1500 | `smooth_disagreement_seg_loss(tau=0.3)` (sigmoid bell on neg margin) | AdamW only | **fresh** cosine 1e-4 → 5e-6 | 0.0 | 0.2 | no | Stage 2 final |
| 4 | `stage4_v332_qat.py` | 500 | `smooth_disagreement_seg_loss(tau=0.3)` | AdamW only | continues Stage 3 cosine | 0.0 | 0.2 | **yes** | Stage 3 final |
| 5 | `stage5_c1a_l7.py` | 9000 (extension; canonical 6000) | `l7_softplus_seg_loss(tau=0.3, l7_threshold=1.0, l7_mult=4.0)` | AdamW only | 3e-5 cosine | **0.01** | 0.2 | yes | Stage 4 final |
| 6 | `stage6_lambda_sweep.py` | 2000 (extension; canonical 1000) | `l7_softplus_seg_loss` (same) | AdamW only | 3e-5 cosine | **0.02** | 0.2 | yes | Stage 5 final |
| 7 | `stage7_sigma_sweep.py` | 3000 (extension; canonical 2000) | `l7_softplus_seg_loss` (same) | AdamW only | 3e-5 cosine | 0.02 | **0.1** | yes | Stage 6 final |
| 8 | `stage8_muon_finetune.py` | 5000 (extension; canonical 3000) | `l7_softplus_seg_loss` (same) | **Muon hidden convs + AdamW stem/heads/biases/latents** | adamw_lr=1e-5 + muon_lr=2e-4 | 0.02 | 0.1 | yes | Stage 7 final |

Every stage uses batch_size=8, ema_decay=0.999, grad_clip=1.0, seg_weight=100.0, pose_weight=1.0, latent_lr_mult=10.0 (latents trained at 10× decoder LR), eval_every=25 epochs.

## Per-batch forward and loss (shared by all 8 stages)

`src/stages/common.py:164-219` — the heart of the training loop.

```
# Per-batch (B pairs from a shuffled permutation of n_pairs):
if cfg.use_qat:
    originals = apply_qat(decoder)            # in-place INT8 fake-quant on Conv2d/Linear weights
decoded_pair = decoder(latents[idx])          # (B, 2, 3, 384, 512) in [0, 255]
if cfg.use_qat:
    restore_qat(decoder, originals)           # restore live weights

# Eval-roundtrip simulation (the 384 -> 874 -> 384 chain matches the contest scorer pre-processing exactly)
flat = decoded_pair.reshape(B*2, 3, 384, 512)
up = F.interpolate(flat, size=(874, 1164), mode='bicubic', align_corners=False)
down = F.interpolate(up, size=(384, 512), mode='bilinear', align_corners=False)
decoded_bhwc = down.reshape(B, 2, 3, 384, 512).permute(0, 1, 3, 4, 2)

# Differentiable round-to-uint8 (straight-through estimator)
decoded_clamped = decoded_bhwc.clamp(0, 255)
decoded_rounded = decoded_clamped.round()
decoded_bhwc = decoded_clamped + (decoded_rounded - decoded_clamped).detach()

# Scorer forward through differentiable yuv6
posenet_in, segnet_in = distortion_net.preprocess_input(decoded_bhwc)
seg_out  = distortion_net.segnet(segnet_in)
pose_out = distortion_net.posenet(posenet_in)

# Stage-specific seg loss + the canonical pose loss
seg_l    = cfg.seg_loss_fn(seg_out, seg_targets_hard[idx])
pose_mse = F.mse_loss(pose_out['pose'][:, :6], pose_targets[idx])
pose_l   = torch.sqrt(10.0 * pose_mse + 1e-12)

# Aggregate loss
loss = cfg.seg_weight * seg_l + cfg.pose_weight * pose_l
if cfg.cat_lambda > 0:
    ent  = cat_entropy_v2(decoder, sigma=cfg.cat_sigma, sample_size=2000, device=device)
    loss = loss + cfg.cat_lambda * ent
```

Two optimizers (Stage 8 only):
- Muon (`src/optim.py`, 100 LOC; Keller Jordan Newton-Schulz orthogonalized momentum, BF16 NS step, decoupled weight decay): drives `nn.Conv2d` hidden weights with `ndim >= 2` AND name does NOT contain `stem` / `rgb` / `.rgb_`. 11 tensors, ~177K params. `momentum=0.95, nesterov=True, ns_steps=5, weight_decay=5e-4` (researcher #24 tweak; not in canonical PR95 but in our extension Stage 8). `lr=2e-4`.
- AdamW: stem Linear, both Conv RGB heads, all biases, all 1D params, AND the per-pair latents (at 10× lr_mult). `lr=1e-5` (Stage 8 only); Stages 1-7 use AdamW only.

EMA update (decay=0.999) after every `optimizer.step()`. Inference is from EMA shadow — `ema_decoder.state_dict()` is what gets quantized + packed into the archive.

Eval every 25 epochs: build a fresh archive from `ema_decoder + ema_latents`, parse it back, run the parsed archive forward, score via `compute_score(seg_d, pose_d, archive_bytes, total_video_bytes)` (the canonical contest metric `100*seg + sqrt(10*pose) + 25*rate`). If the new score beats `best_score`, save `decoder_f32.pt` + `latents_f32.pt` + `best_archive.bin` + `best_meta.json`. Final epoch always writes `final_decoder.pt` + `final_latents.pt` for the next stage to resume from.

## Critical PR95-specific primitives

1. **Differentiable rgb_to_yuv6** (`src/data.py:51-81`). Upstream `frame_utils.rgb_to_yuv6` is `@torch.no_grad()` + in-place `clamp_()`, which severs the autograd graph through PoseNet. PR95 patches BOTH `frame_utils.rgb_to_yuv6` AND `modules.rgb_to_yuv6` (because `modules.py` already imported the original) at module-import time. **Without this patch, pose loss does not reach the decoder; pose stays pinned at random-init value through training.** This is PR95's named-and-fixed `v1/v2 pose plateau bug` and is exactly the eval-roundtrip-into-training non-negotiable in CLAUDE.md.
2. **`cat_entropy_v2`** (`src/losses.py:79-113`). Size-weighted soft histogram entropy: per Conv2d/Linear weight tensor, normalize to `[-127, 127]`, soft-assign via Gaussian bandwidth `sigma` over the 255 integer bins, compute categorical entropy in bits, weight by tensor `numel`. Returns `weighted_entropy / total_numel` — the post-INT8 average bits/weight. Lower σ + higher λ → sharper post-INT8 distribution at integer grid points → smaller brotli-compressed bytes. This is **the soft-MDL term** that drives Stage 5+ archive shrinkage (and is roughly what every HNeRV-family rate primitive ports forward).
3. **`apply_qat`/`restore_qat`** (`src/losses.py:128-148`). Per-tensor symmetric INT8 fake-quant with straight-through estimator — `apply_qat(decoder)` mutates Conv2d/Linear weights in place to the fake-quantized value (returning a `dict` of originals for `restore_qat`), and the surrounding loop pattern is `originals = apply_qat(decoder); decoded = decoder(...); restore_qat(decoder, originals)`. The STE is `(q*scale - tensor).detach() + tensor` so gradients flow but the forward sees the quantized value. Applies from Stage 4 onward.
4. **Muon** (`src/optim.py`, 100 LOC). Keller Jordan 2024 Newton-Schulz orthogonalization, used in Stage 8 only. The pivotal detail per `src/optim.py:55-58`: decoupled weight decay (`p.mul_(1.0 - lr * wd)`) applied BEFORE the orthogonalized update — matches AdamW convention. The Chen-Li-Liu arXiv:2506.15054 argument is that Muon's spectral-norm KKT story requires WD to be active. PR95 canonical used `wd=0.0`; our extension (and the council G memo) recommend `wd=5e-4`.
5. **L7-weighted Softplus** (`src/losses.py:49-62`). Stage 5+ concentration mechanism: per-pixel `tau * softplus(-margin/tau)` weighted by `(1 + 4 * 1[margin < 1])`, renormalized to mean-1 weights. Concentrates the seg gradient on hard-to-classify pixels (margin < 1 are "near the boundary" pixels the scorer will flip).
6. **Bilinear-skip + `sin(x + identity)` decoder block** (`src/model.py:42-54`). Each upsample block forms `identity = bilinear_upsample_2x(x); identity = skip(identity); x = PixelShuffle(2)(block(x)); x = sin(x + identity)`. The bilinear skip carries the previous-resolution signal across the upsample; the `sin` activation gives SIREN-style positional encoding. This is the substrate's expressivity engine.

## Comparison to internal `tac.substrates.sane_hnerv`

Our internal `sane_hnerv` architecture is **structurally similar but NOT byte-faithful** to PR95:

| Aspect | PR95 hnerv_muon | Internal sane_hnerv |
|:-------|:----------------|:--------------------|
| Latent dim | 28 | 28 ✓ |
| Base channels | 36 | embed_dim=48 ✗ |
| Channel taper | `[C, C, C, 0.75C, 0.58C, 0.5C, 0.5C]` (7-stage) | `(40, 32, 24, 20, 16, 12, 8)` (7-stage) ✗ |
| Upsample blocks | 6 (3×4 → 384×512 via 2^6 + bilinear interpolate to 384×512) | 7 (3×4 → 256×384 → interpolate to 384×512) ✗ |
| Block formula | `sin(PixelShuffle(Conv(x)) + bilinear_skip(x))` | `PixelShuffle(sin(Conv(x)))` (NO bilinear skip) ✗ |
| Refine block | `Conv 3x3 dilation=2 → Conv 3x3` then `x + 0.1 * sin(refine(x))` | absent ✗ |
| RGB heads | separate `rgb_0` / `rgb_1` Conv2d, `sigmoid * 255` | separate `head_rgb_0` / `head_rgb_1` Conv2d, `sigmoid` (no *255) ✗ |
| Param count | 229K | ~216K |
| Output range | `[0, 255]` directly | `[0, 1]` (trainer multiplies by 255 at the boundary) ✗ |
| Training | 8 stages, 29,650 epochs, AdamW→Muon, CE→Softplus→Smooth→Smooth+QAT→L7+C1a→λ-sweep→σ-sweep→Muon | 1 stage, 2000 epochs default, AdamW only, score-aware Lagrangian (no curriculum, no C1a, no QAT, no Muon) |

**Conclusion**: byte-faithful reproduction of PR95 requires a NEW substrate variant `pr95_hnerv_muon` mirroring `src/model.py` exactly, NOT a tweak to `sane_hnerv`. The new substrate is a substrate-engineering landing (Catalog #124 opt-out `lane_class=substrate_engineering`).

## Per-stage epoch budget on Modal A100 (extrapolated; UNVALIDATED)

PR95 docs say ~50 hr on a single GPU (unspecified — Keller Jordan's own runs use H100). Using PR101's empirical T4 wall-clock numbers as a lower-bound proxy + the substrate-design grand council 2026-05-12 cost-band table:

- Per-batch wall-clock on A100 (FP32, no autocast, batch 8): ~0.3-0.5 s (dominated by the SegNet+PoseNet scorer forwards, NOT the 229K-param decoder)
- 600 pairs ÷ batch 8 = 75 batches per epoch
- 1 epoch on A100 ≈ 25-40 s
- 29,650 epochs ≈ 200-330 hr on A100 — **NOT feasible inside a $4-15 budget** (A100 at ~$3.40/hr = $680-1100)
- 5000 epochs (Stage 8 only, the cheapest budget arm) ≈ 35-55 hr — also infeasible

**Cost-feasibility decision** (operator-grade): the full 8-stage 29,650-epoch reproduction does NOT fit the F1 $4-15 envelope on Modal A100. Two viable scaled-down arms:

A. **Short-burn proof-of-concept** (target $4-8): Train Stage 1 only at 100-200 epochs as a SMOKE proof that (i) the differentiable yuv6 patch wires correctly, (ii) the scorer-domain seg+pose loss decreases monotonically, (iii) the eval-roundtrip simulation matches the contest pre-process. NO score claim; this is wire-verification only.

B. **Stage-8-only finetune** (target $10-15): If the PR95 published `decoder_f32.pt` + `latents_f32.pt` checkpoints are available in the PR95 intake clone, load them and run Stage 8 (Muon + L7+C1a@σ=0.1,λ=0.02 + QAT) for 1000-2000 epochs. This is the "single-stage finetune layered on PR95's hard-won 7-stage trained substrate" pattern, mirroring how PR101 layered on PR95 codec-only.

PR95 intake at `experiments/results/public_pr95_intake_20260504_codex/pr95_src/models/{segnet,posenet}.safetensors` carries scorer weights but **not the trained decoder weights** — the published archive's `0.bin` is the only embodiment of the trained substrate. Parsing `0.bin` via `parse_archive()` reconstructs the post-INT8 decoder + latents, which is the natural Stage-8 starting point.

**Recommendation**: Arm B (Stage-8-only finetune from the parsed `0.bin`). This costs $10-15, takes 4-6 hr on A100, and tests the most novel PR95 primitive (Muon). It is the cheapest possible empirical anchor on the PR95 curriculum and is directly compatible with the Council G "Stage-8 Muon polish" hypothesis at memo `896f1d79`.

## Outstanding HNeRV-parity audit (pre-dispatch)

The 13 lessons from CLAUDE.md "HNeRV / leaderboard-implementation parity discipline":

1. ✓ Score-aware loss against `upstream/videos/0.mkv` (PR95's `data.py:104-132` is exactly this).
2. ✓ Export-first design — PR95's codec.py is the archive grammar; `0.bin` is the monolithic packet.
3. ✓ Archive grammar = monolithic `0.bin` with fixed offsets `[meta_len:u32][meta][dec_len:u32][dec][lat_len:u32][lat]`.
4. ✓ `inflate.py` is 72 LOC, `inflate.sh` is 24 LOC — both ≤ 100 LOC.
5. ✓ Architecture is the FULL RGB renderer (not mask-only) — `HNeRVDecoder` outputs `(B, 2, 3, 384, 512)`.
6. ✓ Score-domain Lagrangian: `loss = 100*seg + sqrt(10*pose)` (matches contest metric coefficients exactly; the `+25*rate` term enters via `cat_entropy_v2` regularizing the post-INT8 distribution).
7. ✓ Substrate engineering excursion — Stage 1-8 is one canonical curriculum, NOT 8 separate research artifacts. Bolt-ons go on top.
8. ✓ Eval-roundtrip-aware: the 384→874→384 bicubic+bilinear chain inside the per-batch loop simulates the contest scorer pre-processing exactly; the differentiable yuv6 patch is the missing PoseNet-gradient-reachability piece.
9. ⚠ Runtime closure — to be verified by smoke dispatch (the canonical `inflate.sh` signature lands deterministically).
10. ⚠ Mask/pose coupling gate — not applicable to renderer substrate (no mask file).
11. ⚠ No-op detector — to be verified post-dispatch.
12. ✓ Reviewable in 30 seconds — the 8-stage configs each occupy 1.2-1.8 KB.
13. ✓ KILL is last resort — this lane is DEFERRED-pending-empirical-dispatch.

## Path forward (Phase 2 + Phase 3)

1. **Port HNeRVDecoder + losses + Muon to `tac.substrates.pr95_hnerv_muon`** — byte-faithful Python port of PR95's `model.py`, `losses.py`, `optim.py`, `codec.py`, `data.py` (the yuv6 patch). Substrate-engineering landing per Catalog #124 opt-out.
2. **Build `experiments/train_substrate_pr95_hnerv_muon.py`** — single-stage CLI (`--curriculum-stage stage8`, `--resume-from-archive <path>` to seed from a parsed `0.bin`); per-stage configs registered as profile keys.
3. **Recipe + wrapper + smoke-before-full** — Modal A100 dispatch with `tools/run_modal_smoke_before_full.py` $0.30-0.80 T4 smoke (Stage 1 @ 50 epochs, batch 8, 4 pairs) → full A100 Stage 8 @ 1000-2000 epochs (target $10-15).
4. **CUDA + CPU paired auth eval per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA"** — both `[contest-CUDA]` Modal A100 and `[contest-CPU]` Linux x86_64 (Lightning CPU studio or Modal CPU container).
5. **Lane registry**: this lane lands at L2 if smoke succeeds and full A100 produces a `[contest-CUDA]` anchor; L1 if smoke succeeds but A100 dispatch cannot be afforded within the F1 budget envelope.

## Wire-in (6-hook coherence per Catalog #125)

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): N/A for this forensic phase; the empirical anchor from Phase 3 will feed downstream sensitivity-of-archive-bytes-to-Muon-WD into `cat_entropy_v2`-style soft-MDL surrogates. To be wired in the landing memo if Phase 3 produces a measurable score.
2. **Pareto constraint** (`tac.pareto_*`): the Stage 8 Muon ablation arm is a feasibility constraint on the `(seg, pose, rate)` Pareto frontier — adding a Muon polish to an INT8-QAT-trained decoder cannot *increase* archive bytes (the codec output is byte-identical post-INT8 quantization) but can *decrease* seg+pose distortion. The constraint is `rate ≤ rate(stage7) ∧ d_seg ≤ d_seg(stage7) ∧ d_pose ≤ d_pose(stage7)` (Dykstra-feasibility test for Stage 8 improvement claim).
3. **Bit-allocator hook**: N/A — per-tensor INT8 symmetric quant is the canonical bit-allocator for this lane; no per-tensor bit-budget tuning.
4. **Cathedral autopilot dispatch hook**: this lane is registered in the autopilot via the `lane_f1_pr95_8stage_reproduce_20260513` registry entry; the Stage-8 finetune dispatch is the actuator.
5. **Continual-learning posterior update**: the Phase 3 auth-eval anchor (CUDA + CPU paired) will append via `posterior_update_locked(ContestResult(...))` per Catalog #128.
6. **Probe-disambiguator**: N/A — there is one canonical curriculum (PR95's), not two interpretations.

## File-by-file forensic line count

| File | LOC | Reviewable in 30s? |
|:-----|---:|:------------------|
| `model.py` | 55 | ✓ |
| `losses.py` | 162 | ✓ (one function per stage's seg loss; plus `cat_entropy_v2`, QAT pair, EMA) |
| `optim.py` | 100 | ✓ (Muon + `partition_params_for_muon`) |
| `codec.py` | 181 | ✓ (quant + zigzag + brotli helpers + archive layout) |
| `data.py` | 160 | ✓ (yuv6 patch + `precompute_targets` + video path resolver) |
| `score.py` | 132 | ✓ (`evaluate_decoder` + `compute_score`) |
| `train.py` | 78 | ✓ (orchestrator) |
| `stages/common.py` | 275 | ✓ (the shared training loop) |
| `stages/stage1..stage8` (each) | 40-50 | ✓ |
| `stages/codec_stage.py` | 72 | ✓ |
| **Total** | **~1,400** | ✓ |

## References

- PR95 source: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/`
- Council G memo (Council 7-3 binding TRIPLET phi): commit `896f1d79`
- PR95 retrospective: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_why_leaderboard_hnerv_worked_when_ours_didnt_PERMANENT_KNOWLEDGE_20260509.md`
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" non-negotiable
- Lane registry: `lane_f1_pr95_8stage_reproduce_20260513` (L0 at registration, target L2)
