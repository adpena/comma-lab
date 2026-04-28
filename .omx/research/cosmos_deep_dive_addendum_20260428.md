# NVIDIA Cosmos — Deep Dive Addendum (corrective pass)

**Date:** 2026-04-28
**Author:** Cosmos deep-dive subagent (corrective pass; previous synthesis dismissed Cosmos as "wrong scale" without reading the runnable code)
**Status:** Council-ready; lane proposals included with predicted bands.
**Companion to:** `/Users/adpena/Projects/pact/.omx/research/cosmos_mae_2604_telescope_synthesis.md` Section 1 (lines 22-79). Read that section's TL;DR first; this addendum supersedes its "Council verdict on Cosmos: park" line.

---

## Headline correction

The prior synthesis was wrong to park Cosmos. The PARAMETER COUNTS (2B–14B) are wrong scale — but that was never the transferable artifact. The transferable artifacts are **architectural patterns**, **conditioning fusion mechanisms**, **the FP4/FP8 protected-ops recipe**, and a **per-clip outlier scoring method that drops directly into Lane W**. Reading the runnable code (cosmos-transfer1, cosmos-rl, cosmos-predict2.5, cosmos-cookbook) line-by-line surfaces 6 lane candidates, of which 3 are net-new and the top one (Lane F-V5) directly attacks the documented FP4 failure mode that killed Lane F.

---

## 1. Code-walked

Files read line-by-line (URLs + local cache paths):

| Source | URL | Local cache | LOC | What it gave us |
|---|---|---|---|---|
| `transfer1/inference-x-mobility/cosmos_dataset_to_xmob.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/scripts/examples/transfer1/inference-x-mobility/cosmos_dataset_to_xmob.py) | `/tmp/cosmos_xmob_pre.py` | 434 | Data wrangler only — confirms x-mobility uses (camera_image bytes + perspective_semantic_image_shape) parquet schema. NOT model code. |
| `transfer1/inference-x-mobility/xmob_dataset_to_videos.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/scripts/examples/transfer1/inference-x-mobility/xmob_dataset_to_videos.py) | `/tmp/cosmos_xmob_post.py` | 305 | Pure parquet→mp4 utility. 7-class semantic segmentation: [Navigable, Forklift, Cone, Sign, Pallet, Fence, Background]. |
| `transfer1/inference-x-mobility/inference.md` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/recipes/inference/transfer1/inference-x-mobility/inference.md) | `/tmp/xmob_inference.md` | 211 | The KEY READ. ControlNet controlnet_specs JSON: `seg.control_weight=1.0` "preserves the original geometry exactly, allowing direct reuse of all ground-truth labels." |
| `predict2.5/gr00t-dreams/inference.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/scripts/examples/predict2.5/gr00t-dreams/inference.py) | `/tmp/gr00t_inference.py` | 205 | Wraps `Video2WorldInference`. 3 modes: TEXT2WORLD/IMAGE2WORLD/VIDEO2WORLD (0/1/2 input frames). Autoregressive sliding window with `chunk_size=77, chunk_overlap=1` for long-form. |
| `predict2.5/gr00t-dreams/config.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/scripts/examples/predict2.5/gr00t-dreams/config.py) | `/tmp/gr00t_config.py` | 539 | Variants: `BASE`, `AUTO_MULTIVIEW`, `ROBOT_ACTION_COND`, `ROBOT_MULTIVIEW_AGIBOT`. `num_steps=35, guidance=7` (range 0-7). `num_input_frames` = {0,1,2}. |
| `cosmos-transfer1/diffusion/networks/general_dit_ctrl_enc.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/cosmos_transfer1/diffusion/networks/general_dit_ctrl_enc.py) | `/tmp/transfer_ctrl_enc.py` | 344 | **GOLD MINE.** Multi-control encoder, hint encoder, `zero_module` init pattern, `dropout_ctrl_branch=0.5`, per-modality `hint_encoders` ModuleList, spatial-temporal weight maps. Lane TFR proposal source. |
| `cosmos-transfer1/diffusion/networks/general_dit.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/cosmos_transfer1/diffusion/networks/general_dit.py) | `/tmp/transfer_general_dit.py` | 611 | Base GeneralDIT: `block_config="FA-CA-MLP"`, RoPE 3D, `extra_per_block_abs_pos_emb`, adaLN_modulation init to ZERO. |
| `cosmos-transfer1/diffusion/inference/world_generation_pipeline.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/cosmos_transfer1/diffusion/inference/world_generation_pipeline.py) | `/tmp/transfer_pipeline.py` | 1455 | Per-modality `hint_encoders=ModuleList([...])` loaded each from its own ckpt. **Partial-denoising via `sigma_max=70`**: encode input video latents, add noise of σ_max, sample from there (img2img in latent space). |
| `cosmos-transfer1/diffusion/inference/transfer.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-transfer1/blob/main/cosmos_transfer1/diffusion/inference/transfer.py) | `/tmp/transfer_main.py` | 433 | CLI defaults: `--num_steps=35, --guidance=5, --sigma_max=70.0, --num_input_frames=1, fps=24`. `--upsample_prompt` uses Pixtral upsampler. |
| `cosmos-rl/utils/fp4/config.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/fp4/config.py) | `/tmp/cosmos_config.py` | 208 | **3 FP4 recipes**: TENSORWISE / ROWWISE / **ROWWISE_WITH_GW_HP** (the killer feature). |
| `cosmos-rl/utils/fp4/float4_linear.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/fp4/float4_linear.py) | `/tmp/cosmos_float4_linear.py` | 439 | Asymmetric-precision `matmul_with_hp_or_float4_args` autograd.Function. Forward + grad_input + grad_weight gemms each get independent quantization config. |
| `cosmos-rl/utils/fp4/fp4_util.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/fp4/fp4_util.py) | `/tmp/cosmos_fp4_util.py` | 140 | **Hardware: NVFP4 needs CC 10.0 (Blackwell B100/B200), NOT 4090.** Filter: `dims_multiples_of_16 AND not is_filtered_fqn`. `torch._inductor.config.emulate_precision_casts=True` for ROWWISE+RMSNorm+compile. |
| `cosmos-rl/utils/fp8/fp8_util.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/fp8/fp8_util.py) | `/tmp/cosmos_fp8_util.py` | 166 | **FP8 IS supported on RTX 4090 (CC ≥ 8.9)**. Uses `torchao.float8.convert_to_float8_training`. Same FQN filter. |
| `cosmos-rl/utils/mxfp4/quantizer.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/utils/mxfp4/quantizer.py) | `/tmp/cosmos_quantizer.py` | 98 | MXFP4 = block-32 microscaling. ~5 bits/param effective. |
| `cosmos-cookbook/data_curation/outlier_detection.md` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/recipes/data_curation/outlier_detection/outlier_detection.md) | `/tmp/curator_outlier.md` | 479 | Cosmos-Embed1-336p video embedding → PCA-3 → Time Series K-Means + soft-DTW barycenter distance → Q95 outlier threshold. **Drops into Lane W.** |
| `cosmos-cookbook/control_modalities/overview.md` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/core_concepts/control_modalities/overview.md) | `/tmp/control_overview.md` | 249 | Multi-control normalization rule (Σweights ≤ 1.0 → as-is, > 1.0 → re-normalize to 1.0). 4 modalities: Edge / Seg / Vis / Depth, each with task decomposition. |
| `cosmos-cookbook/distillation/distilling_predict2.5.md` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/core_concepts/distillation/distilling_predict2.5.md) | `/tmp/distill_predict.md` | 94 | DMD2 36→4 steps with critic, `student_update_freq=5` (4 critic + 1 student per macro-step). 1500 iters = 300 student + 1200 critic. CFG distilled into student weights. |
| `cosmos-cookbook/distillation/distilling_transfer1.md` | [GitHub](https://github.com/nvidia-cosmos/cosmos-cookbook/blob/main/docs/core_concepts/distillation/distilling_transfer1.md) | `/tmp/distill_transfer.md` | 111 | 2-stage: KD warm-up on synthetic teacher data (10K noise-video pairs), then DMD2 with real data. Batch size 64 critical. |
| `cosmos-predict2.5/predict2/action/configs/action_conditioned/conditioner.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/action/configs/action_conditioned/conditioner.py) | `/tmp/p25_action_conditioner.py` | 304 | **CFG dropout: text=0.2, use_video_condition=0.2, action=0.0** (action never dropped). `Video2WorldConditionV2` zeroes conditional frames in CFG-uncond branch. |
| `cosmos-predict2.5/predict2/action/configs/action_conditioned/net.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/action/configs/action_conditioned/net.py) | `/tmp/p25_action_net.py` | 139 | Architecture numbers (see §2.4). Mini-debug net is `model_channels=1024, num_heads=8, num_blocks=2`. RoPE 3D + adaln_lora_dim=256. |
| `cosmos-predict2.5/predict2/action/models/action_conditioned_video2world_model.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/action/models/action_conditioned_video2world_model.py) | `/tmp/p25_action_model.py` | 318 | **`HighSigmaStrategy` — 5% of training samples get sigma redrawn from [80, 2000] or log-uniform [200, 100000].** `low_sigma_ratio=0.05` for the near-zero regime. `sigma_conditional=0.0001` for clean conditioning frames. CFG formula `raw_x0 = cond_x0 + guidance*(cond_x0 - uncond_x0)`. Inpainting `guided_image*guided_mask + (1-guided_mask)*x0`. |
| `cosmos-predict2.5/interactive/configs/registry_experiment/experiments_dmd2_predict2p5.py` | [GitHub](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/interactive/configs/registry_experiment/experiments_dmd2_predict2p5.py) | `/tmp/p25_dmd2_exp.py` | 463 | DMD2 numerical hparams: lr=2e-7 critic, teacher_guidance=3, timestep_shift=5, sde p_mean=-0.8 p_std=1.6 (heavily low-sigma-skewed). |
| (web) Cosmos Predict 2.5 paper [arXiv:2511.00062](https://arxiv.org/abs/2511.00062), [HF model card](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B), [research page](https://research.nvidia.com/labs/cosmos-lab/cosmos-predict2.5/) | — | — | — | Wan2.1 VAE, Cosmos-Reason1 (7B Qwen-based) as text encoder, flow-based, 2B/14B sizes, 1280×704 / 832×480 resolutions. |
| (web) Cosmos World Foundation Models [arXiv:2501.03575](https://arxiv.org/abs/2501.03575) | — | — | — | **3D Haar wavelet** in tokenizer, **FSQ (Finite Scalar Quantization)** for discrete variant, CV8x8x8 = 8× spatial × 8× temporal continuous tokenizer. |

---

## 2. Architecture extraction

### 2.1 Cosmos Transfer 2.5 / Transfer 1 — multi-control DiT

**Block-level architecture (verbatim from `general_dit_ctrl_enc.py`):**

```
Input video frames (B,C,T,H,W) → encode (Wan VAE 8× spatial × 4× temporal) → latent (B,16,T/4,H/8,W/8)
                                                                              │
For each control modality m ∈ {edge, seg, depth, vis, hdmap, lidar, kpts}:    │
  hint_m: (B,Cm,T,H,W) ─→ PatchEmbed3D → embedded tokens (T,H,W,B,D)         │
                       │                                                       │
                       └─→ input_hint_block: 7-layer MLP                       │
                            channels: [16, 16, 32, 32, 96, 96, 256]            │
                            ↓ zero_module(Linear(256, model_channels))         │
                            (final layer initialized to ZERO — ControlNet trick)│
                                                                              │
For each block i in [0..num_blocks):                                          │
  noise tokens x_i ─→ DiTBlock_i (FA-CA-MLP) → x_{i+1}                       │
                                  ↓ if i < num_control_blocks:                │
                                     x_{i+1} += guided_hint_m (only at block 0)│
                                     control_feat_i = zero_blocks[m][i](x_{i+1}) │
                                                       × control_weight[m]    │
                                                       × control_gate[i]      │
                                     outs[i] += control_feat_i  (sum across m)│
                                                                              │
Base model forward: x = base.net(x, ..., x_ctrl=outs)                        │
  → at each block i, base block adds outs[i] to its residual stream          │
```

**Key parameters (Cosmos Transfer 2.5 base 2B):**
- `model_channels=2048, num_heads=16, num_blocks=28, in_channels=16` (latent), `patch_spatial=2, patch_temporal=1`
- `pos_emb_cls="rope3d"` with `rope_h_extrapolation_ratio=1.0` (no extrapolation at small scale)
- `block_config="FA-CA-MLP"` (Full self-attention + Cross-attention to text → MLP)
- `use_adaln_lora=True, adaln_lora_dim=256` — adaLN modulation is itself rank-256 LoRA-decomposed
- **All adaLN modulation Linear layers initialized to ZERO** (`nn.init.constant_(block.adaLN_modulation[-1].weight, 0)`) — block starts as identity, learns modulation from scratch
- **`hint_channels=16`** per modality, **`hint_nf=[16,16,32,32,96,96,256]`** in the small MLP encoder
- **`dropout_ctrl_branch=0.5`** — half of training batches drop the control branch (CFG trick on conditioning)

**Multi-control fusion mechanism (the part the prior agent missed):**
- Per-modality CHECKPOINT — each control modality (seg, edge, depth, vis) ships its OWN ckpt loaded into `hint_encoders=nn.ModuleList([...])`.
- Per-block summation: `outs[block_i] = Σ_m (zero_blocks[m,i](x) × control_weight[m] × gate[i])`
- **Per-pixel control_weight maps**: `control_weight` can be `[B, 1, T, H, W]` → modality strength varies by pixel and frame. (Lane HFM motivation.)
- **Layer gating**: `control_gate_per_layer[i] = (i < num_control_blocks)` — only the FIRST `num_control_blocks` blocks inject conditioning; deeper blocks generate without. **This is a "rate budget" decision in the architecture itself.**
- **Weight normalization rule** (from Transfer 2.5 docs): if Σ_m control_weight[m] ≤ 1.0 → as-is; if > 1.0 → re-normalized so sum = 1.0. Hard cap on total conditioning influence.

**At training time:**
- 50% probability the control branch is dropped (`dropout_ctrl_branch=0.5`).
- 20% probability the text condition is dropped (`text.dropout_rate=0.2` in the SHARED conditioner config).
- 20% probability `use_video_condition` is False (`BooleanFlag.dropout_rate=0.2`); when False, conditional frames are zeroed in V2 conditioner.
- These three independent dropouts let inference compose arbitrary CFG: text-only, control-only, both, or unconditional.

### 2.2 Cosmos Predict 2.5 — base video2world DiT (action-conditioned variant)

**Architecture sizes (verbatim from `p25_action_net.py`):**

| Size | model_channels | num_heads | num_blocks | extra_per_block_abs_pos_emb | rope_t_extrap |
|---|---|---|---|---|---|
| 2B | 2048 | 16 | 28 | False | 1.0 |
| 7B | 4096 | 32 | 28 | True | 2.0 |
| 14B | 5120 | 40 | 36 | False | 1.0 |
| **mini_net (debug)** | **1024** | **8** | **2** | True | 1.0 |

Even their debug network (mini_net) is 2 blocks × 1024 channels — about 25M params. Our 80–100K params is 250× smaller than their smallest config. **This is exactly the scale issue the prior agent flagged — but the architectural patterns transfer.**

Common settings: `in_channels=16` (latent), `patch_spatial=2, patch_temporal=1`, `pos_emb_cls="rope3d"`, `pos_emb_learnable=True`, `use_adaln_lora=True, adaln_lora_dim=256`, `atten_backend="minimal_a2a"`.

**Conditioning (action-conditioned model):**
- Action enters via the conditioner as a raw `ReMapkey(input_key="action", dropout_rate=0.0)` — never dropped during training. **Action conditioning is a HARD signal**, unlike text/video which are soft (CFG-droppable).
- `action_dim = 10 * 8 = 80` for GR00T (10 timesteps × 8-DOF action vector).
- `Video2WorldCondition` carries `gt_frames`, `condition_video_input_mask_B_C_T_H_W` (binary [B,1,T,H,W] mask of which frames are conditioning), and `num_conditional_frames_B`.
- **Frame-replace conditioning strategy**: at the latent level, the first N frames of `xt` are REPLACED by the GT latents (with sigma_conditional=0.0001 noise), and during loss, the predicted x0 is REPLACED back to GT for those frames. **Net effect: the model never has to predict the conditioning frames; it only predicts the future.**

**Training noise schedule (HighSigmaStrategy):**
The 2B model defaults to `UNIFORM80_2000`:
- 95% of training samples: standard sigma drawn from `lognormal(p_mean=-0.8, p_std=1.6)` (heavily low-sigma-skewed: median sigma ≈ 0.45).
- **5% of training samples: sigma redrawn UNIFORMLY from [80, 2000]** — wildly out-of-distribution noise. Forces the model to be robust at extreme noise levels even though those aren't where standard inference operates.
- Variant `BALANCED_TWO_HEADS_V1` adds another 5% with sigma drawn UNIFORMLY from [0.00001, 2.0] — the very-low-noise regime. **Forces the model to also handle near-clean inputs.**

**This is the critical insight for closing the proxy/auth gap.** Cosmos doesn't just self-augment the conditioning (Lyra-style); they also **augment the noise schedule itself** — train the model to be robust to noise levels it normally wouldn't see, so deploy-time surprises don't break it.

**CFG inference formula:**
```python
def x0_fn(noise_x, sigma):
    cond_x0 = denoise(noise_x, sigma, condition).x0
    uncond_x0 = denoise(noise_x, sigma, uncondition).x0
    raw_x0 = cond_x0 + guidance * (cond_x0 - uncond_x0)
    if "guided_image" in batch:
        raw_x0 = guided_mask * guided_image + (1-guided_mask) * raw_x0
    return raw_x0
```
- `guidance=1.5` default. Standard CFG extrapolation.
- **Inpainting hook built in**: `guided_image + guided_mask` lock specific pixels at inference (any pixel where guided_mask=1 is forced to GT, others denoised). **This means the model always supports "lock these regions, regenerate those" without retraining.**

### 2.3 Cosmos RL — FP4/FP8 protected-ops recipe (Lane F-V5 source)

**Three FP4 recipes (verbatim from `cosmos_config.py`):**

1. **`TENSORWISE`** — single scaling factor for the entire tensor. Default. Simple. For us: most lossy.
2. **`ROWWISE`** — per-row (per-output-channel) scaling factor. CUTLASS kernel. Better preservation of weight magnitude per channel.
3. **`ROWWISE_WITH_GW_HP`** — **THE KILLER RECIPE.** GW = grad_weight, HP = high precision.
   - Forward gemm: rowwise FP4 (`cc_i, cc_w` axiswise quantized).
   - grad_input gemm: rowwise FP4 (`cc_go` axiswise, `cc_w_gi` tensorwise — re-quantize weight per-tensor for the backward).
   - **grad_weight gemm: DISABLED scaling = FP32/BF16** (`cc_i_gw, cc_go_gw` both `ScalingType.DISABLED`).
   - Rationale: weight gradients are tiny accumulated signals. FP4-quantizing them destroys the QAT update direction. Forward and activation-grad can tolerate FP4 noise; the weight gradient cannot.

**Hardware compatibility (CRUCIAL for our budget):**
- **NVFP4 hardware: requires CUDA compute capability 10.0** = Blackwell (B100, B200). **RTX 4090 is CC 8.9 → NO HARDWARE NVFP4.** Our previous Lane F was using `FakeQuantFP4` simulation, which is what we'd still need.
- **FP8 hardware: requires CC ≥ 8.9** = Ada Lovelace (4090, L40, L40s). **WE HAVE FP8 HARDWARE on Vast.ai 4090s.** Uses `torchao.float8.convert_to_float8_training` (PyTorch's official path).
- **MXFP4 = block-32 microscaling** (every 32 elements share a scale factor). At ~5 bits/param effective, slightly worse rate-density than pure FP4 but easier to support without Blackwell.

**Module filter pattern:**
```python
def module_filter_fn(mod, fqn, filter_fqns):
    if not isinstance(mod, nn.Linear): return False
    dims_multiples_of_16 = (mod.weight.shape[0] % 16 == 0) and (mod.weight.shape[1] % 16 == 0)
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)
    return dims_multiples_of_16 and not is_filtered_fqn
```

- Linear layers with weight shapes not divisible by 16 are AUTOMATICALLY skipped (hardware constraint).
- The model itself declares its protected list via `model.fqn_filter_for_quantization()`. This is the CANONICAL "protected ops" mechanism — encoded in the model class, not the training script.
- Conv layers are NEVER quantized by this codepath. Only Linears. (Our renderer's conv weights would need separate quantization; this codepath is Linear-only.)

**Compile gotcha:**
```python
if quant_recipe == "rowwise":
    torch._inductor.config.emulate_precision_casts = True
```
**ROWWISE FP4/FP8 + RMSNorm + torch.compile = NaN** without this flag. This is the kind of hidden landmine that destroys a run silently.

### 2.4 Cosmos World Foundation Model — tokenizer architecture

From [arXiv:2501.03575](https://arxiv.org/abs/2501.03575) (Cosmos WFM Platform paper) and [GitHub Cosmos-Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer):

- Tokenizer architecture: **3D Haar wavelet** (front-end) → **causal residual blocks** → **causal downsampling** → **causal spatio-temporal attention**. Decoder mirrors with causal upsampling.
- Two variants: continuous (CV) and discrete (DV).
- CV8x8x8 = 8× spatial × 8× spatial × 8× temporal compression, latent C=16 channels.
- Discrete variant uses **Finite Scalar Quantization (FSQ)** instead of VQ-VAE (no codebook collapse, fully differentiable).
- "Temporally length-agnostic at inference" — tokenizer trained on shorter clips can tokenize longer ones via the causal structure.
- **The 3D Haar wavelet is parameter-free, analytical, invertible.** This matches our "analytical knowledge" frame from `project_arbitrary_vs_learnable_taxonomy` — the Haar wavelet is the canonical "free" multi-scale decomposition. Our renderer currently does no explicit multi-scale decomposition (besides whatever the conv stack learns). A Haar wavelet front-end on the masks could give the renderer free scale-decomposition.

### 2.5 Cosmos Curator — outlier/typicality scoring (the missing Lane W input)

From `outlier_detection.md`:

- Use **Cosmos-Embed1-336p** (a learned video embedding model) to compute per-clip embeddings.
- Per-frame embeddings → per-trajectory time series.
- **Interpolate trajectories to a fixed length** (e.g. 20 frames). Reduces variable-length problem.
- **PCA-3 dimensionality reduction** (faster than UMAP, deterministic, sufficient for clustering).
- **Time Series K-Means with Soft-DTW barycenter** (`tslearn.clustering.TimeSeriesKMeans(metric="softdtw", gamma=1.0)`).
- For each trajectory, compute `soft_dtw(traj_i, barycenter[cluster(i)], gamma=1.0)`.
- **Outlier threshold: Q95 quantile of distances** = top 5% are outliers within their cluster.

**For us**: at compress time we have 1199 forward-driving pairs. Compute per-pair embeddings (using SegNet's encoder, or a tiny pretrained backbone like DINOv2-small), cluster into K=4-6 groups (e.g. straight, gentle-curve, sharp-curve, lane-change), then identify per-cluster outliers. **Outliers are the pairs the renderer is most likely to fail on at deploy time.** They get extra weight in Lane W's hard-pair loss, AND extra rate budget in Lane FP-style variable-resolution mask coding.

This is the principled, council-defensible answer to "which pairs are hard?" Currently Lane W uses the proxy SegNet/PoseNet loss as the difficulty signal; that's circular (hard to optimize against itself). **Soft-DTW outlier scoring is independent of the renderer.**

---

## 3. Lane proposals

EV = (predicted-band-midpoint reduction below current Lane A 1.15 baseline) × confidence ÷ cost. All bands relative to 1.15 contest-CUDA. Council non-conservative.

### Lane TFR — Multi-Control Conditioning Encoder

**Premise**: replace our current "concatenate masks + pose vector" conditioning with a multi-modal Cosmos-Transfer-style encoder that injects (Seg + Edge + Vis) modalities each via its own small zero-init MLP, summed per-block in the renderer's residual stream, with per-pixel weight maps learned at compress time.

**Architecture**: At our scale, this is a 3-modality version of the hint encoder pattern in `general_dit_ctrl_enc.py`:
- Seg: 5-class one-hot (5 ch) → `hint_nf=[5, 8, 16, 32]` MLP → zero-Linear(32, renderer_channels)
- Edge: Canny on the GT video at compress (2 bytes/frame at low res = ~2KB total) → `hint_nf=[1, 8, 16, 32]` → zero-Linear
- Vis: 4× downsample of seg-class probabilities (low-pass appearance) → `hint_nf=[5, 8, 16]` → zero-Linear
- Total conditioning encoder: ~3 × 2K = ~6K params extra. Edge can be derived AT INFLATE from masks (free), Vis is downsampled-mask (free).
- Per-block fusion: `outs[i] = Σ_m zero_blocks[m,i](renderer_act_i) × control_weight[m]`
- Spatially varying `control_weight[m]` for each modality lets us upweight Seg in lane-marking regions, Vis in sky regions.

**Predicted band**: [0.85, 1.10]
**Mid Δ**: -0.18
**Confidence**: 0.40 (moderate — this is a refactor of conditioning rather than a new mechanism)
**Cost**: $5 + 18h on 4090 (full retrain with new conditioning fusion)
**Composability**: Stacks with Lane W (hard-pair weighting on the conditioning loss), Lane MAE-V (MAE half-frame is a special case of the Vis modality being the warped even frame). Conflicts with Lane S (self-compress) — both attack the conditioning encoder; need to pick.
**Risk**: We don't have separate per-modality checkpoints to load. We'd train all three conditioning encoders jointly from scratch. The dropout_ctrl_branch=0.5 trick makes this stable: half the time the model gets ALL conditioning, half the time NONE; the model learns to do both.
**Council pre-check**: Edge and Vis are derivable from mask at inflate time → ZERO archive cost for those modalities. Only Seg costs rate. **This is a free architecture change** — we just trade the current monolithic conditioning for 3 specialized streams.

### Lane F-V5 — FP8 (NOT FP4) on dilated-h64 ASYM

**Premise**: Cosmos RL exposes that **FP8 is the right precision for our hardware (RTX 4090 is CC 8.9 = FP8-native)**. NVFP4 needs Blackwell (CC 10.0) which we don't have. Our prior Lane F used `FakeQuantFP4` simulation and got destroyed (20× PoseNet penalty). **Switch to TRUE FP8 via `torchao.float8.convert_to_float8_training` with the protected-ops list** (FiLM, motion.head, decoder head, any layer with shape not divisible by 16 — automatic).

**Implementation sketch**:
```python
from torchao.float8 import convert_to_float8_training, Float8LinearConfig

# Same FQN-filter pattern as Cosmos-RL
PROTECTED_FQNS = ["film", "motion.head", "decoder_head", "pose_emb"]
def filter_fn(mod, fqn):
    if not isinstance(mod, nn.Linear): return False
    if mod.weight.shape[0] % 16 != 0 or mod.weight.shape[1] % 16 != 0: return False
    return not any(p in fqn for p in PROTECTED_FQNS)

convert_to_float8_training(model, config=Float8LinearConfig(...), module_filter_fn=filter_fn)
```

**Predicted band**: [0.95, 1.20]
**Mid Δ**: -0.075
**Confidence**: 0.50 (HIGH — FP8 is well-understood, hardware-supported, and we KNOW from FP4 failure that FP8's 8 bits/weight is the sweet spot)
**Cost**: $2 + 8h (smaller retrain since FP8 needs less QAT than FP4)
**Composability**: Stacks with EVERY other lane (it's a pure quantization change). Replaces Lane F (FP4) which is dead.
**Rate impact**: 8 bits/param × 100K params = 100KB renderer (vs 50KB FP4). **This is +0.034 rate cost** (50KB × 25 / 37.5MB scoring formula). For Lane F-V5 to be net-positive, distortion must improve by > 0.034. Since FP4 caused +0.44 score increase, FP8 needs only ~0.4 of that distortion damage to break even. Expected.
**Risk**: 100KB renderer is bigger than Quantizr's. Need to confirm we still hit the 300KB total. Current Lane A: 100KB renderer + 180KB masks + 7KB poses = 287KB. With FP8: 100KB → fits. With MXFP4 (~62KB): fits comfortably.
**Council pre-check**: Cosmos RL's recipe handles ALL the gotchas: dims_divisible_by_16 filter, RMSNorm+compile NaN flag, asymmetric grad-weight precision. **This is the lowest-risk lane in this addendum.** Should run regardless of what else.

### Lane F-V6 — MXFP4 with block-32 microscaling

**Premise**: If Lane F-V5 (FP8 at 100KB) still pushes our archive over 300KB after Lane TFR or Lane MAE-V add their bytes, MXFP4 is the fallback. ~5 bits/param effective via block-32 scaling, supported on 4090 via emulation (slower but functional). At 100K params → ~62KB. Better rate than FP8, less destructive than pure FP4.

**Predicted band**: [0.95, 1.18]
**Mid Δ**: -0.085
**Confidence**: 0.30 (lower — MXFP4 less battle-tested than FP8)
**Cost**: $3 + 12h
**Composability**: Replaces Lane F-V5 if rate budget pinched. Otherwise skip.
**Risk**: vLLM's MXFP4 path is for large MoE models; our small dense Conv2d/Linear stack may not hit the scaling factor's design assumptions. Block-32 means our smallest Linears (pose_dim=12) only have 3 blocks — high relative scale overhead.

### Lane SAUG-V2 — Noise-schedule augmentation (extends Lane SAUG)

**Premise**: The prior synthesis proposed Lane SAUG (Lyra-style self-augmentation of conditioning frames). Cosmos `HighSigmaStrategy` adds a SECOND axis of augmentation: **the noise level itself**. Train at OOD noise levels — 5% of batches at sigma redrawn from [80, 2000], 5% at sigma redrawn from [0.00001, 2.0]. This forces the model to be robust at noise levels it normally wouldn't see at inference, which closes the proxy/auth gap from a different angle than self-augmentation alone.

For our renderer (which doesn't have noise schedule per se since it's a single-pass U-Net), the analog is:
- Train on AV1-encoded masks at variable CRF (5% at extreme CRF=51, 5% at CRF=0 = lossless), not just CRF=50.
- Train on TTO-noise-corrupted poses (5% with noise variance = 10× the typical TTO step size, 5% with zero noise = clean).
- Train with `noise_std` for the renderer activations drawn from a distribution (not fixed).
- Combined with eval_roundtrip, this is "dropout on the entire forward path."

**Predicted band**: [0.70, 1.00]
**Mid Δ**: -0.30
**Confidence**: 0.45 (high if the proxy/auth gap is THE blocker; the gap is well-documented as 100-350×)
**Cost**: $4 + 14h (smoke test 50 epochs first)
**Composability**: STACKS WITH EVERYTHING. Schedule augmentation is a training-only change. No architecture change. No archive change.
**Risk**: Exotic noise levels can destabilize training. Mitigation: ramp the augmentation probability from 0 → 5% over the first 1000 epochs.
**Pre-flight gate**: smoke 50 epochs, measure proxy/auth ratio reduction. If ratio drops below 30× from current 100-350×, full retrain. If not, kill.

### Lane WC — Cosmos-Curator outlier weighting for Lane W (drop-in replacement for the difficulty signal)

**Premise**: Lane W (in MEMORY.md as `project_lane_w_hard_pair_self_compress_premise_20260427`) currently weights the top-K hardest pairs by their per-pair PoseNet/SegNet proxy loss. The proxy is circular (hard to optimize against itself; the "hardness" signal IS what we're trying to fix). Cosmos Curator gives us an **independent** typicality score: per-pair embedding via SegNet's encoder → cluster → soft-DTW distance to barycenter → Q95 outlier flag.

**Implementation**:
```python
# At compress time, ONCE
import tslearn
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import soft_dtw

# Compute per-frame SegNet encoder embedding (B,16,T,H,W → B,T,d after pool)
emb = segnet.encoder(frames).mean(dim=[3,4])  # (B, T, d)
# Subdivide each pair-trajectory to 20 timesteps (same as Curator)
emb_interp = subdivide(emb, n_points=20)  # (1199, 20, d)
# PCA to 3D
flat = emb_interp.reshape(-1, d)
flat_reduced = PCA(n_components=3).fit_transform(flat)
trajs = flat_reduced.reshape(1199, 20, 3)
# Soft-DTW K-Means
model = TimeSeriesKMeans(n_clusters=6, metric="softdtw", random_state=42)
labels = model.fit_predict(trajs)
centers = model.cluster_centers_
distances = np.array([soft_dtw(trajs[i], centers[labels[i]], gamma=1.0) for i in range(1199)])
# Q95 threshold per-cluster (not global!) — atypicality is relative to cluster
hard_mask = np.zeros(1199, dtype=bool)
for c in range(6):
    cluster_dists = distances[labels==c]
    threshold = np.quantile(cluster_dists, 0.95)
    hard_mask[labels==c] = distances[labels==c] > threshold
# Lane W now uses hard_mask × 5.0 weighting on the loss for outlier pairs
```

**Predicted band**: [0.78, 1.05]
**Mid Δ**: -0.235
**Confidence**: 0.40 (moderate — relies on Lane W concept being correct; if Lane W itself doesn't help, this just makes a non-helpful lane non-helpful with better signal)
**Cost**: $3 + 12h (smoke 30 epochs first to verify cluster-based weighting actually picks hard pairs differently from proxy-based)
**Composability**: REPLACES Lane W's difficulty signal. Stacks with Lane FP (FramePack — variable rate per frame) where the outlier flag determines rate budget. Stacks with Lane SAUG-V2.
**Risk**: 6 clusters is arbitrary; needs sweep K∈{3,4,6,8,12}. If clusters don't separate driving regimes meaningfully, the outlier signal degenerates to "noisy frames are outliers" (which is what proxy loss already gives us).
**Pre-flight**: visualize cluster assignments overlaid on the 1199 pair indices. Should see structured groupings (e.g. all "lane change" pairs in one cluster). If random-looking, abort.

### Lane V-DMD — DMD2 distillation for the renderer (replaces our 5-stage QAT final stage)

**Premise**: Cosmos Predict 2.5 distills 36-step diffusion → 4-step student via DMD2 with `student_update_freq=5` (4 critic updates per 1 student update). The student bakes the CFG conditioning effect into its weights. **For our renderer (a single-pass U-Net, not multi-step diffusion), the "step reduction" doesn't apply, but the adversarial-critic training pattern IS transferable**: train a critic ("fake score net") that learns to discriminate between (real GT scorer outputs) and (renderer-predicted scorer outputs). The renderer is updated less often than the critic; the critic catches up between renderer updates. The renderer eventually matches the GT distribution, not just the GT mean (which is what L2 / MSE achieves).

**Architecture**:
- Critic = small CNN, ~10K params, takes (5-class mask logits | 6-DOF pose) as input, outputs 1 logit (real/fake).
- Stage 5 of QAT: alternate 4 critic updates, 1 renderer update. Renderer loss = L2_to_GT + 0.001 × adversarial_loss.
- Use existing teacher checkpoint (the un-quantized renderer) as the "real" distribution generator if needed.

**Predicted band**: [0.95, 1.20]
**Mid Δ**: -0.075
**Confidence**: 0.25 (low — DMD2 was designed for distribution matching in diffusion, our renderer isn't a sampler; the analogy might not transfer)
**Cost**: $4 + 14h
**Composability**: Replaces Lane A's QAT stage 5 only. Stacks with everything else.
**Risk**: Adversarial training is unstable at small scales. The DMD2 paper recommends batch_size ≥ 32 for stability; our single-clip task naturally has batch_size=1 (1199 pairs in a clip). Mitigation: gradient accumulate to effective batch 32.
**Council verdict**: LOWER PRIORITY than Lane F-V5 / Lane SAUG-V2 / Lane WC. Run only if those three land and we have budget.

### Lane HW — Haar wavelet front-end on the masks (free architectural change)

**Premise**: Cosmos tokenizer uses **3D Haar wavelet** as its first stage. Haar is parameter-free, analytical, invertible. Our renderer currently feeds raw 5-class one-hot masks; the conv stack has to learn its own multi-scale representation. **Insert a 3D Haar wavelet decomposition (1 spatial level + 1 temporal level) as a free pre-processor on the mask input**. The renderer sees the masks at 4 spatial sub-bands (LL, LH, HL, HH) × 2 temporal sub-bands (L, H) = 8 channels of multi-scale signal at 1/2 spatial × 1/2 temporal resolution.

**Architecture**: Haar-3D = 3 successive 1D Haar transforms along T, H, W. Inverse exists. Total parameter cost: ZERO. Renderer's first conv goes from 5-channels-input to 8-channels-input (5 classes × 8 sub-bands → 40 channels actually).

**Predicted band**: [0.95, 1.15]
**Mid Δ**: -0.10
**Confidence**: 0.30 (low-moderate — pure architectural change, no clear evidence from our codebase whether the renderer needs this)
**Cost**: $2 + 8h (smoke at 50 epochs to confirm convergence not destabilized)
**Composability**: Free stack with everything. No archive cost. Pure inflate-time compute (~1ms/frame on T4).
**Risk**: Our renderer might already learn the equivalent decomposition. Wavelet front-end may then just slow training without changing the optimum. Mitigation: ablate at smoke time — train 30 epochs with vs without; pick whichever has lower proxy loss.
**Council pre-check**: Haar inverse is exact; loss can flow back. Check our current renderer's first-conv kernel activations — if they look anything like Haar basis functions, this is dead. If they look chaotic, this lane wins.

### Lane GUIDE — Inference-time inpainting hook (FORBIDDEN by strict-scorer-rule for inflate, but useful at compress)

**Premise**: Cosmos Predict 2.5 has built-in `guided_image + guided_mask` inpainting at inference: `raw_x0 = guided_mask * guided_image + (1 - guided_mask) * x0`. Lock specific pixels to GT, regenerate the rest. **For us at INFLATE time this is forbidden** (would require shipping the guide image — adds rate). But **at COMPRESS time during pose TTO** we can use this trick to hold known-good regions stable while letting other regions converge. Currently our pose TTO is unconstrained — every pixel can change.

**Predicted band**: [0.95, 1.15]
**Mid Δ**: -0.10
**Confidence**: 0.20 (very low — this is a refinement of compress-time TTO, which we know hurts on dilated-h64; we documented this in `project_lane_b_pose_tto_proxy_auth_gap`)
**Cost**: $2 + 8h
**Composability**: Compress-time only. Replaces or supplements our existing pose TTO. Could be combined with Lane CCW (canonical-coord warp) where the warp defines which pixels to lock.
**Risk**: Pose TTO is a known landmine on this checkpoint. **DEPRIORITIZE** unless other lanes succeed and we want to push further.
**Council verdict**: park for now. Revisit if we get below 1.0 and need an extra 0.05.

### Lane DD — Dropout-based CFG on conditioning (free training change)

**Premise**: Cosmos uses `dropout_ctrl_branch=0.5, text.dropout_rate=0.2, use_video_condition.dropout_rate=0.2` — three independent dropouts on conditioning during training. **Our renderer's training currently always feeds masks + poses** (no dropout). At inference, the conditioning is what it is — there's no CFG at inflate (single pass). But **at training time, randomly zeroing the conditioning forces the renderer to learn a "fallback" distribution that doesn't depend on conditioning**. This is the inverse of self-augmentation — instead of training on noisy conditioning, train on absent conditioning.

For us: with `p=0.2`, replace mask input with all-zeros; with `p=0.2`, replace pose with all-zeros. The renderer must learn a "prior" mode that's good even when conditioning is unhelpful.

**Predicted band**: [0.85, 1.10]
**Mid Δ**: -0.175
**Confidence**: 0.30 (we have no direct evidence this helps a single-pass renderer; CFG-style benefits typically show up at multi-step samplers)
**Cost**: $3 + 10h
**Composability**: Pure training change. Stacks with everything.
**Risk**: For single-pass renderer, training without conditioning may just teach an "average mask" prior that hurts conditioned outputs. Counter-evidence: Cosmos's `Video2WorldConditionV2` zeros conditioning during CFG-uncond — this works empirically.

---

## 4. Council ranking by EV

EV = mid Δ × confidence ÷ cost ($).

| Rank | Lane | Mid Δ | Conf | Cost ($) | EV | Notes |
|---|---|---|---|---|---|---|
| 1 | **Lane F-V5** (FP8 on dilated-h64) | -0.075 | 0.50 | 2 | **0.0188** | Lowest risk, highest confidence, immediate. |
| 2 | **Lane SAUG-V2** (noise-schedule + conditioning self-aug) | -0.30 | 0.45 | 4 | **0.0338** | Highest mid Δ. Smoke first. |
| 3 | **Lane WC** (Curator outlier weighting for Lane W) | -0.235 | 0.40 | 3 | **0.0313** | Independent of renderer; replaces circular signal. |
| 4 | **Lane TFR** (multi-control conditioning encoder) | -0.18 | 0.40 | 5 | **0.0144** | Architectural. Conflicts with Lane S. |
| 5 | **Lane HW** (3D Haar wavelet front-end) | -0.10 | 0.30 | 2 | **0.0150** | Free, pure architectural. |
| 6 | **Lane DD** (dropout CFG on conditioning) | -0.175 | 0.30 | 3 | **0.0175** | Pure training change. |
| 7 | **Lane F-V6** (MXFP4) | -0.085 | 0.30 | 3 | **0.0085** | Backup if Lane F-V5 doesn't fit budget. |
| 8 | **Lane V-DMD** (adversarial QAT) | -0.075 | 0.25 | 4 | **0.0047** | Speculative; small batch instability risk. |
| — | **Lane GUIDE** (compress-time inpainting at TTO) | -0.10 | 0.20 | 2 | 0.0100 | Park; pose TTO is a landmine on this checkpoint. |

### Re-ranked by EV including the prior synthesis's top 3:

| Rank | Lane | Source | Mid Δ | Conf | Cost ($) | EV |
|---|---|---|---|---|---|---|
| 1 | **Lane SAUG-V2** | Cosmos addendum (THIS doc) | -0.30 | 0.45 | 4 | **0.0338** |
| 2 | **Lane WC** | Cosmos addendum (THIS doc) | -0.235 | 0.40 | 3 | **0.0313** |
| 3 | **Lane SAUG** (orig Lyra) | Prior synthesis | -0.25 | 0.55 | 5 | **0.0275** |
| 4 | **Lane MAE-V** | Prior synthesis | -0.20 | 0.50 | 4 | **0.0250** |
| 5 | **Lane F-V5** (FP8) | Cosmos addendum (THIS doc) | -0.075 | 0.50 | 2 | **0.0188** |
| 6 | **Lane HF** (Telescope foveation) | Prior synthesis | -0.18 | 0.40 | 4 | **0.0180** |
| 7 | **Lane DD** | Cosmos addendum (THIS doc) | -0.175 | 0.30 | 3 | **0.0175** |
| 8 | **Lane HW** (Haar wavelet) | Cosmos addendum (THIS doc) | -0.10 | 0.30 | 2 | **0.0150** |
| 9 | **Lane TFR** | Cosmos addendum (THIS doc) | -0.18 | 0.40 | 5 | **0.0144** |

### Recommended cycle (3 in parallel per CLAUDE.md operating rule)

**Cycle 1 — best-EV cosmos lanes that don't overlap with prior in-flight work:**
1. **Lane F-V5 (FP8)** — $2, lowest-risk, ships immediately. Run at smoke first.
2. **Lane SAUG-V2** (noise-schedule augmentation + conditioning self-aug) — $4, smoke 50 epochs, full retrain if ratio drops below 30×. **Subsumes Lane SAUG** (Lyra original).
3. **Lane WC** (Curator outlier weighting) — $3, smoke 30 epochs to verify cluster signal differs from proxy.

**Total Cycle 1 cost**: $9 + 12h wallclock on 3× 4090 (parallel).

**Cycle 2 (composition, only if Cycle 1 produces ≥1 winner):**
- Best of Cycle 1 + **Lane HW (Haar wavelet, free)** + **Lane TFR or Lane MAE-V (whichever has architectural conflict) — pick one**
- Predicted composed band: [0.55, 0.85] = path to first sub-1.0.

**Cycle 3 (architectural moonshot):**
- Lane DD + Lane V-DMD + Lane HF (foveation) — 3 architectural changes, $11 total
- Reserved for after sub-1.0 is reached.

---

## 5. Dispositioned proposals (looked at, not transferable)

So we don't ask again later:

- **Cosmos Predict 2/2.5/14B base models** — 2-14B params, 720p/93 frames. Wrong scale. Confirmed by reading actual configs (`p25_action_net.py`): even their `mini_net` debug config is 25M params (250× our budget).
- **Cosmos Reason1 7B VLM as text encoder** — we have no text input; nothing to encode.
- **Wan 2.1 VAE (CV8x8x8)** — the encoder/decoder ALONE is bigger than our entire archive (estimated 50-200MB depending on variant). Skip.
- **DMD2 multi-step → few-step distillation** — our renderer is already 1 forward pass. No multi-step to distill from. The CRITIC pattern transfers (Lane V-DMD); the step reduction doesn't.
- **Cosmos LoRA recipe (rank=32, alpha=32, q/k/v/o/mlp1/mlp2 targets)** — already covered in prior synthesis as "Lane LR-V3 — useless single-clip." Confirmed.
- **Beamr CABR (NVENC h264 RC-MAXQ)** — already covered. We use libsvtav1 for masks; CABR is hardware NVENC tuning only.
- **Cosmos Curator's Ray-based pipeline** — we have one 60s clip; Ray distribution gains nothing. The Curator OUTLIER ANALYSIS algorithm transfers (Lane WC); the infrastructure does not.
- **Pixtral prompt upsampler** — text-prompt augmentation. We have no text prompts.
- **SAM3 backbone in Telescope** — already covered in prior synthesis (Lane HF) as too large; SAM2 used for mask generation in Cosmos Transfer is also too large for us (~100M params).
- **Cosmos guardrails** (text safety + face blur) — irrelevant compliance layer for our task.
- **`is_av_sample` post-trained checkpoint** in cosmos-transfer1 — there's an "AV sample" variant suggesting they have a driving-specific post-train, but it requires loading 7B base + AV LoRA. Wrong scale.
- **Multi-view variants (`AUTO_MULTIVIEW`, `ROBOT_MULTIVIEW_AGIBOT`)** — for cross-camera consistency. We have 1 EON camera. No multi-view.
- **Autoregressive sliding-window inference (`chunk_size=77, chunk_overlap=1`)** — for >77-frame videos. Our 1199 pairs already fit in our renderer's batch. Not architecturally interesting at our scale.
- **Cosmos Tokenizer's CAUSAL temporal blocks** — for streaming/real-time. We have offline access to all 1200 frames at compress time; non-causal access is a strict superset. Skip causal.

---

## 6. Implementation notes (for whoever picks up the work)

### Lane F-V5 dispatch:
- Add `--use-fp8` flag to `experiments/pipeline.py`. Verify with `grep "add_argument" experiments/pipeline.py` BEFORE wiring (per CLAUDE.md non-negotiable).
- Before adding flag, `grep "fqn_filter_for_quantization" src/tac/architectures.py` — it likely doesn't exist; need to add per-architecture protected-FQN list.
- Test on Vast.ai 4090 (CC 8.9 = FP8 native). NVFP4 path is gated behind CC 10.0 check; will raise `RuntimeError` on 4090.
- Test the `torch._inductor.config.emulate_precision_casts = True` flag interaction with our `torch.compile` usage in `src/tac/training.py`.
- Subagent prompt: "Implement FP8 training for the dilated-h64 renderer using `torchao.float8.convert_to_float8_training` with FQN filter `['film', 'motion.head', 'decoder_head', 'pose_emb']`. Add CLI flag `--use-fp8` to `experiments/pipeline.py` (verify against argparse FIRST). Smoke 30 epochs measuring proxy SegNet/PoseNet vs FP32 baseline. If proxy degradation < 5%, full retrain + auth eval."

### Lane SAUG-V2 dispatch:
- Modify `src/tac/training.py` to add `noise_schedule_aug_prob`, `low_sigma_aug_prob`, `crf_aug_prob`, `pose_noise_aug_prob` flags.
- For each batch: with prob `noise_schedule_aug_prob=0.05`, redraw noise_std from log-uniform [10×default, 100×default]. With prob `low_sigma_aug_prob=0.05`, set noise_std=0. With prob `crf_aug_prob=0.05`, re-encode masks at CRF=∈{0, 51} (extreme). With prob `pose_noise_aug_prob=0.05`, perturb poses by N(0, 10×TTO_step_size).
- Subagent prompt: "Add `cosmos-style noise schedule augmentation' to src/tac/training.py: 4 independent dropouts at p=0.05 each on (noise_std out-of-distribution, noise_std≈0, mask CRF∈{0,51}, pose noise 10× normal). NO ad-hoc; profile in `src/tac/profiles.py` as `dilated_h64_saug_v2`. Verify proxy/auth ratio drops below 30× in 50-epoch smoke before full retrain."

### Lane WC dispatch:
- New file `src/tac/curator.py` implementing the outlier-detection pipeline (PCA-3 + Time Series K-Means soft-DTW + Q95 per-cluster).
- Wire into Lane W's hard-pair loss as the difficulty signal source.
- Subagent prompt: "Implement Cosmos-Curator-style per-pair outlier scoring in src/tac/curator.py. Use SegNet's encoder pooled features as embedding (we don't have Cosmos-Embed1-336p; SegNet encoder is the closest analog and is already in our archive). PCA-3, K=6 Time Series K-Means with `tslearn` soft-DTW gamma=1.0, Q95 per-cluster outlier flag. Surface as `outlier_mask` to Lane W's loss. Verify cluster assignments are non-random by visualizing 1199 pairs colored by cluster (should show structured driving-regime groupings)."

---

## 7. Ratio-of-rigor: what the prior synthesis got right vs missed

**Got right**: Cosmos at our parameter scale is wrong. Lane LR-V3 (LoRA delta per clip) is useless single-clip. CABR is NVENC tuning, not a learned codec.

**Missed**:
1. **The conditioning ENCODER architecture** in `general_dit_ctrl_enc.py` is a transferable pattern at OUR scale (small MLP + zero-init projections + per-modality summation). Lane TFR.
2. **Cosmos RL has a complete FP4/FP8 protected-ops recipe** with hardware compatibility checks, FQN filtering, asymmetric grad-precision, RMSNorm+compile gotchas. Directly addresses Lane F failure. Lane F-V5.
3. **Cosmos Predict 2.5's `HighSigmaStrategy`** is a SECOND axis of self-augmentation orthogonal to Lyra's. Lane SAUG-V2.
4. **Cosmos Curator's outlier detection** is a drop-in independent difficulty signal for Lane W. Lane WC.
5. **3D Haar wavelet** in Cosmos Tokenizer is a free, parameter-free, analytical multi-scale front-end. Lane HW.
6. **CFG-style triple-dropout** (text=0.2, video=0.2, ctrl=0.5) on conditioning during training is a proven training trick. Lane DD.
7. **Built-in inference inpainting** (`guided_image + guided_mask`) shows the architectural primitive for "lock these pixels, regenerate those." Lane GUIDE for compress-time.
8. **DMD2's critic pattern** (4:1 critic-to-student updates, distribution matching not mean matching) is transferable to QAT stage 5. Lane V-DMD.

The "wrong scale" framing is correct for the WEIGHTS. It's wrong for the MECHANISMS. Mechanisms scale down; the published recipes survive shrinkage by 2-3 orders of magnitude.

— end addendum —
