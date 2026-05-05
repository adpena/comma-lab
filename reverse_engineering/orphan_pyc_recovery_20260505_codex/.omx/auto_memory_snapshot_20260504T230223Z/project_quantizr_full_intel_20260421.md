---
name: Quantizr full deobfuscated architecture and training pipeline
description: Complete competitive intelligence on Quantizr PR#55 (0.33). 88K params, 300KB archive, 5-stage training, FiLM+DSConv, KL distill T=2.0, half-frame masks. Says sub-0.30 possible, stopped optimizing.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Quantizr PR#55 — Complete Architecture (2026-04-21)

**Person**: Jimmy, UCLA CSE/Neuroscience background. GitHub: Quantizr.

**Score**: 0.33. Archive: 299,970 bytes (293KB).

**Model**: FiLMSepResBlock — FiLM-conditioned depthwise-separable conv, 88K params
- SepConvGNAct: depthwise-separable convs + GroupNorm + SiLU
- FiLM conditioning on 6D pose vectors (replaced optical flow from PR#53)
- FP4 quantization with custom nibble-packed codebook + Brotli compression → ~64KB

**Training** (5-stage):
1. Anchor: baseline training with standard loss
2. Finetune: scorer-guided loss (PoseNet + SegNet)
3. Joint: combined pixel + scorer loss
4. QAT: insert FP4 fake quantize, fine-tune at lower LR
5. Final: export + eval

**Key techniques**:
- diff_round(): differentiable rounding matching eval chain
- diff_rgb_to_yuv6(): independently found same gradient bug as us
- kl_on_logits(T=2.0): SegNet KL distillation during training
- EMA on model weights
- Encodes ONLY frame2 masks (600 frames, not 1200) — frame1 is warped

**Archive breakdown**:
- renderer.bin: ~64KB (FP4 + Brotli)
- masks.mkv: ~220KB (AV1, 600 odd-frame masks at higher CRF)
- poses.pt: ~8KB (fp16 poses)

**His own words**: "sub 0.30 is possible just by sweeping conv dims" and "he is done working on it"

**How to apply**: Our architecture is the same paradigm (AsymmetricPairGenerator, FiLM, warp).
We need: smaller model (use_dsconv=True, hidden_ch=32), 5-stage training, QAT, half-frame masks.

---

## 2026-04-28 ARCHITECTURAL GROUND TRUTH (Sherlock audit correction)

**The "same paradigm" line above is WRONG.** Detailed re-read of `/tmp/quantizr_inflate.py` (PR #55 head SHA `e0b643b0a7c21f62cc93b5d920bcf3fc0d5a33d9`, 323 lines) — full audit at `.omx/research/quantizr_replica_audit_20260428.md`:

- **No motion module. No warp. No optical flow.** PR body verbatim: *"dropping optical flow and using Feature-wise Linear Modulation on pose vectors instead of using both masks."*
- **Single mask input.** `JointFrameGenerator.forward(mask2, pose6) → (frame1, frame2)`. Both frames derived from the SAME odd-frame mask via two parallel heads sharing a trunk.
- `frame2_head` is **UNCONDITIONAL** (`Frame2StaticHead`).
- `frame1_head` is **FiLM-conditioned on pose6** (`FrameHead`/`FiLMFrameHead`).
- `pose_mlp = Linear(6, 48) → SiLU → Linear(48, 48)` — **cond_dim=48**, NOT pose_dim=6 fed directly.
- `SharedMaskDecoder(emb_dim=6, c1=56, c2=64, depth_mult=1)`.
- DSConv (depthwise-separable) trunk via `SepConvGNAct` / `SepConv` / `SepResBlock`.

The actual archive contains 3 brotli files: `model.pt.br` (FP4), `mask.obu.br` (libaom-av1), `pose.npy.br`. Inflate just runs `(f1, f2) = generator(mask2, pose6)` per pair — no warp at inflate.

Lane V/V2/K all kept the wrong architectural family (warp + dual-mask + MotionPredictor) and were therefore not faithful replicas — even though they matched param count + DSConv flag. **Param-count matching is a red herring; the structure matters.**

**The corrected rebuild is Lane Q-FAITHFUL** (`src/tac/quantizr_faithful_renderer.py`, profile `q_faithful_dilated_88k`, deploy `scripts/remote_lane_q_faithful_jointgen.sh`). 87,836 params, NO motion, single-mask + FiLM-on-pose dual-head — see `project_lane_q_faithful_design_20260428.md`.
