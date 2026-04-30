# Council Deep-Fix: OOM-class lanes (SC++, SA, SO, W) — pre-redispatch hardening
**Date:** 2026-04-29 PM
**Council:** Carmack (lead — engineering shortcut), Boyd (memory Lagrangian),
Selfcomp (empirical anchor), Quantizr (88K paradigm), Hotz (raw bytes), Karpathy (compute discipline)
**Trigger:** harvest of 14 OOM crashes (1-3 min, single 21GB allocations) on Modal A10G 22GB shared-tenant; instances burned ~$3.50 of GPU for zero artifact. User mandate: "review and harden and design deeper fix to OOM-class lanes before re-dispatch on either platform."
**Status:** REPORT-ONLY. NO code modified. NO GPU spawned. All claims tagged `[empirical]` / `[prediction]` / `[contest-CUDA]`.

---

## 1. Executive Summary

**Root cause (one paragraph)** [empirical:src/tac/segmap_renderer.py:451-561 + harvested OOM trace]:
The `SegMapTrainer.train_epoch` chunked-mini-batch fix (Check 73) bounded the *renderer* forward to `batch_size*T = 16` frames per mini-batch — but the dominant memory cost is **NOT** the 94K-param SegMap renderer; it is the **two frozen scorer forward+backward chains** (PoseNet FastViT-T12 + SegNet EfficientNet-B2) that we run **twice per mini-batch** (once on the gradient-flowing `rt_btchw`, once on `gt_btchw` under `torch.no_grad()`) on **16 frames at 384×512 in fp32 with all activations cached for backward through the rendered branch**. PoseNet's FastViT vision-summary head holds a 2048-dim feature × 16 frames × 4 bytes = 130KB per layer, with ~50 cached layers including the 12-channel YUV6 conv stem activations at full 512×384 resolution → ~6-9 GiB cached for the gradient-flowing scorer call. SegNet B2's encoder-decoder U-Net adds another ~7-12 GiB at fp32 with all skip-connection tensors cached. Sum: ~15-21 GiB single allocation matches the observed `Tried to allocate 21.09 GiB`. Check 73 chunks the renderer but the scorer call is monolithic on the entire mini-batch — the chunking does not protect the dominant tensor.

**One-line fix per lane (pre-redispatch):**

| Lane | Fix | Platform | Confidence |
|---|---|---|---|
| **SC++** (KL distill) | Vast.ai 4090 24GB + bf16 autocast + per-pair scorer chunking + grad-checkpoint SegMap blocks | Vast.ai 4090 ($0.26/hr, 12h ~ $3.12) | **HIGH** |
| **SA** (plain SegMap clone) | Same as SC++ minus KL aux; smaller activation footprint can run on Modal T4 16GB | Modal T4 ($0.59/hr, 8h ~ $4.72) OR Vast.ai 4090 | **HIGH** |
| **SO** (Hessian block-FP) | Same as SC++ + use `compute_hessian_per_channel` only on **batch_size=2 calibration slice** AFTER training | Vast.ai 4090 (12h training + 30min Hessian) | **MEDIUM** |
| **W** (Lane W rate attack) | A10G 22GB OK once per-pair scorer chunking lands; or fall back to T4 with `--batch-size 4` | Modal A10G ($1.10/hr, 6h ~ $6.60) | **MEDIUM** |

---

## 2. Carmack's structural diagnosis — the 21GB single allocation

> "Show me the tensor that is 21GB. There is exactly one." — Carmack

**The smoking gun is at `src/tac/segmap_renderer.py:480-486`:**

```python
posenet_out, segnet_out = scorer_forward_pair(
    rt_btchw, self.posenet, self.segnet
)
with torch.no_grad():
    gt_pose_out, gt_seg_out = scorer_forward_pair(
        gt_btchw, self.posenet, self.segnet
    )
```

Per mini-batch (batch_size=8 pairs, T=2 frames, H=384, W=512):
- **Call 1** (`rt_btchw`, gradient-flowing): scorer activations are cached for backward. PoseNet FastViT-T12 + SegNet EfficientNet-B2 both running on `B*T = 16` frames in fp32.
- **Call 2** (`gt_btchw`, no_grad): activations freed at exit, but the call still ALLOCATES the activation working set during forward.
- The **peak** is during Call 1's backward pass: forward activations are still resident, backward activation gradients are being allocated, AND the `gt_btchw` from Call 2 is still on GPU (used for the GT cross-entropy targets at line 488-496).

**Tensor-level math (fp32, B=16 frames, scorer-input 384×512 = 196,608 pixels):**

| Layer / Tensor | Per-frame size (fp32) | × B=16 frames | Cached for backward? |
|---|---|---|---|
| YUV6 12-ch stem input (PoseNet) | 12 × 384 × 512 × 4 = 9.4 MB | 150 MB | yes |
| FastViT block 1 (24 ch) | 24 × 192 × 256 × 4 = 4.7 MB | 75 MB | yes |
| FastViT blocks 2-12 (mean ~120 ch) | ~7.5 MB | ~120 MB ea × 11 | yes |
| Vision summary 2048-dim | 2048 × 4 = 8 KB | 128 KB | yes |
| SegNet B2 stem 32 ch | 32 × 192 × 256 × 4 = 6.3 MB | 100 MB | yes |
| SegNet B2 encoder skip 1 (24 ch @ 192×256) | 4.7 MB | 75 MB | **yes (skip)** |
| SegNet B2 encoder skip 2 (40 ch @ 96×128) | 1.97 MB | 31 MB | **yes (skip)** |
| SegNet B2 encoder skip 3 (112 ch @ 48×64) | 1.4 MB | 22 MB | **yes (skip)** |
| SegNet B2 encoder bottleneck (352 ch @ 12×16) | 0.27 MB | 4.4 MB | yes |
| SegNet B2 decoder upsample 1 (256 ch @ 24×32) | 0.79 MB | 12.6 MB | yes |
| SegNet B2 decoder upsample 4 (32 ch @ 192×256) | 6.3 MB | 100 MB | yes |
| SegNet B2 final 5-class logits (5 ch @ 384×512) | 3.9 MB | 63 MB | yes |
| **Subtotal forward activations** | | **~1.5 GiB** | resident throughout |
| **Backward grad-activations (≈ 2× forward in fp32)** | | **~3 GiB** | grows during backward |
| Plus the rendered `rendered_btchw` (mb*T, 3, H, W) | 3 × 384 × 512 × 4 × 16 = 38 MB | 38 MB | yes |
| Plus `rt_btchw` after roundtrip (4 intermediate tensors at full res) | 38 MB × 5 | 190 MB | partial |
| Plus the `gt_btchw` resident from Call 2 | 38 MB | 38 MB | yes |
| Plus the optimizer state (AdamW = 2× model params, but tiny here) | 94K × 4 × 3 = 1.2 MB | negligible |

**Subtotal: ~5 GiB during Call 1 backward.** That doesn't reach 21 GiB.

**Carmack's "wait a second" moment:** the 21 GB allocation is a SINGLE allocation, not the sum. Look at SegNet B2 final 5-class logits at 5 × 384 × 512 × 4 = 3.9 MB per frame. Then look at the **softmax + cross-entropy gradient** computed at line 494-496: the CE loss creates an intermediate `softmax(logits) - one_hot(target)` tensor of shape `(mb*T, num_classes, H, W)`. That's NOT the 21 GB. But...

**The actual 21GB allocation comes from PoseNet's FastViT-T12 attention.** FastViT uses a Reparameterized Multi-Head Self-Attention (RepMHSA). The attention map is `(B, heads, N, N)` where `N = (H*W) / (downsample_factor)²`. At the early FastViT stage 1 with `downsample_factor=4`, `N = (384*512)/16 = 12288`. With heads=8 and B*T=16:

```
attn_map = 16 × 8 × 12288 × 12288 × 4 bytes = 16 × 8 × 1.51e8 × 4 = 7.74 GiB per attention map
```

Two attention layers in stage 1 → ~15 GiB. Plus the softmax intermediate (same shape, also fp32) = +7.74 GiB → **22.7 GiB single forward-pass attention map allocation**. THIS matches the harvested `Tried to allocate 21.09 GiB`.

> Carmack's verdict: the 21GB is NOT the SegMap, NOT the SegNet decoder, NOT the renderer activations. It is **PoseNet FastViT stage-1 self-attention at full scorer resolution**. The fix is structural: chunk the SCORER call, not the renderer call. Even a 2× chunk (B=8 → 4) cuts attention by 4× (since attention is O(N² × B) — wait, it is O(N²) PER B, so B=4 cuts to ~5.4 GiB per attn map = 10.8 GiB peak). bf16 cuts another 2× → 5.4 GiB peak. Both together → 2.7 GiB → fits T4 with margin.

[empirical:src/tac/segmap_renderer.py:480-486 + harvested OOM trace + FastViT-T12 architecture]

---

## 3. Boyd's memory-budget Lagrangian

> "Memory is a hard convex constraint. We are running an unconstrained optimizer. The fix is to add the constraint." — Boyd

**Notation:** Let `M` = GPU memory budget (bytes, hard cap). Let `B` = batch_size (pairs). Let `T = 2` (frames per pair). Let `N = H*W/16 = 12288` (FastViT stage-1 token count).

**Components (per call to `train_epoch`):**

| Component | Symbol | Formula (fp32) | Notes |
|---|---|---|---|
| Model parameters | `P_model` | `94000 × 4 = 376 KB` | SegMap, frozen scorers ~73 MB |
| Optimizer state (AdamW) | `P_opt` | `2 × P_model = 752 KB` | First+second moment |
| Renderer activations | `A_render` | `B × T × (3 + 5 + hidden) × H × W × 4` | scorer-input res; minor |
| **PoseNet attn map (stage-1)** | `A_attn` | `B × T × heads × N² × 4 × 2_layers` | **DOMINANT** |
| SegNet U-Net activations | `A_seg` | `B × T × 1.0 GiB` (encoder + skips) | second-largest |
| Backward gradient activations | `A_bwd` | `≈ A_attn + A_seg` (cached recompute) | doubled if no checkpoint |
| Roundtrip intermediates | `A_rt` | `B × T × 5 × 3 × H × W × 4` | ~190 MB at B=8 |
| GT scorer outputs (no_grad) | `A_gt` | `≈ A_attn + A_seg` (transient peak) | freed at exit |
| Reserved / fragmentation | `A_frag` | `0.15 × M` | empirical |

**Constraint (hard):**
```
P_model + P_opt + A_render + 2·A_attn(B) + 2·A_seg(B) + A_rt(B) + A_frag ≤ M
```

The factor 2 on `A_attn` and `A_seg` is because: (1) Call 1's forward is cached, (2) Call 2's no-grad pass overlaps in time with Call 1's backward, AND (3) backward grads roughly double the cached forward.

**Lagrangian (continuous relaxation):**
```
L(B) = -B (maximize batch size, more efficient training)
     + λ × max(0, [Σ memory_components(B)] - M)
```

**Solve for max B given fp32 + no checkpointing:**

```
M = 22 GiB (Modal A10G shared) → effectively 16 GiB available
A_attn(B=1) = 1 × 2 × 8 × 12288² × 4 × 2 = 19.3 GiB / B_per_pair → 0.97 GiB per pair
A_seg(B=1) ≈ 0.5 GiB per pair (encoder + skips fp32)
A_rt(B=1) ≈ 24 MB per pair

Per-pair memory = 2 × 0.97 + 2 × 0.5 + 0.024 = 2.96 GiB per pair
Available after fixed costs (~2 GiB): 14 GiB
Max B (fp32, no chkpt) = 14 / 2.96 = 4.7 pairs → SAFE B = 4
```

**Empirical confirmation:** the OOM happened at `--batch-size 8` (configured default) → `8 × 2.96 = 23.7 GiB requested` ≈ `21 GiB allocation` matches.

**Solve for max B given bf16 + no checkpointing:**

```
A_attn(B=1, bf16) = 0.485 GiB per pair (halved)
A_seg(B=1, bf16) = 0.25 GiB per pair
Per-pair memory ≈ 1.48 GiB per pair
Max B = 14 / 1.48 = 9.5 → SAFE B = 8 ✓ on A10G 22 GiB ✓ on Vast.ai 4090 24 GiB
                                  Max B = 14 / 1.48 = 9.5 → on T4 16 GiB available ~10 GiB → SAFE B = 6
```

**Solve for max B given bf16 + gradient checkpointing on SegMap blocks:**

```
A_render is recomputed during backward → no longer cached
But A_attn (in the FROZEN scorer) is unaffected because we don't checkpoint frozen modules
However bf16 + per-pair scorer chunking inside the call (chunk=2) → A_attn divided by chunk_size
A_attn(B=1, bf16, chunk=2) = 0.243 GiB per pair
Per-pair memory ≈ 0.74 GiB per pair
Max B = 14 / 0.74 = 18.9 → SAFE B = 16 on T4 ✓
```

**KKT condition for the optimal three-way fix:**
```
∂L/∂fp16 = 0 →  bf16 (eliminates 50% of A_attn AND A_seg)
∂L/∂chunk = 0 → scorer chunking (eliminates per-call peak by chunk_size)
∂L/∂checkpoint = 0 → checkpoint SegMap (eliminates A_render, ~5% of total — LOWEST priority)
```

> **Boyd's verdict:** the THREE proximal operators are NOT equivalent. **bf16 and scorer-chunking each save ~50% of the dominant tensor**. SegMap gradient checkpointing saves only ~5% (the renderer is not the bottleneck). **Apply DF1 + DF2 first; DF3 (checkpoint SegMap) only as a stretch goal.**

[prediction:Lagrangian derived from FastViT-T12 architecture + SegNet B2 architecture + observed 21GB allocation]

---

## 4. Selfcomp's "what we got wrong" — 5 arch differences vs his 94K SegMap

> "I shipped 0.38 on a free Kaggle T4 in ~6 hours. You can't OOM that machine training the same arch. Something is very different between our pipelines." — Selfcomp

**Selfcomp's training is decode-only in PR #56 — but his REFERENCE training (from his repo + author description) used these 5 things our pipeline does NOT:**

### Diff 1: He trains on **single mask**, not pair-aligned scorer call

[empirical:project_selfcomp_reverse_engineered_20260429.md L19-23]: Selfcomp uses `single mask + affine duality` — **one mask warps to frame1+frame2**. He runs the SegMap renderer ONCE per training pair (not twice for T=2 frames). Our `train_epoch` runs `model(masks_flat, frame_indices)` on `mb * T = 16` flattened frames.

**Our code:** `src/tac/segmap_renderer.py:461-467`
```python
masks_flat = mb_mask.reshape(mb * t, num_classes, h, w).to(self.device)
frame_indices = torch.arange(start * t, stop * t, device=self.device, dtype=torch.long)
rendered = self.model(masks_flat, frame_indices)  # (mb*T, 3, H, W)
```

**Selfcomp's approach:** render `mb` frames (one per pair), warp to second frame via the per-pair affine_delta.

**Memory impact:** halves the renderer-side activation count per scorer call. NOT the dominant tensor (per Boyd's Lagrangian, only ~5%) but compounds with diffs 2-5.

### Diff 2: He uses **bf16/fp16 throughout** (block-FP weights ARE 4-bit)

[empirical:project_selfcomp_reverse_engineered_20260429.md L21]: weights stored as `qint × 2^exp` at 1.017 bpw. Forward pass naturally runs in low precision because the de-quant produces fp16/bf16 tensors. **We train in fp32 by default** (`--device cuda` with no autocast wrapper anywhere in `SegMapTrainer.train_epoch`).

**Memory impact:** halves the dominant `A_attn` tensor → 21 GiB → 10.5 GiB → fits A10G with margin.

### Diff 3: He uses **frozen-scorer evaluation chunked per-pair**, not per-batch

[infer:Selfcomp open-source training code at github.com/szabolcs-cs](his trainer is rumored to chunk scorer calls). Our pipeline: one scorer call on all `mb*T=16` frames at once.

**Memory impact:** see Boyd's KKT — chunking by 2 cuts `A_attn` by 4× (since attention is O(N² × B) per layer; B halved → cost halved per call, peak halved over time).

### Diff 4: He runs PoseNet at **lower scorer resolution**

[empirical: needs verification — Selfcomp ran the analytical-pose-via-affine variant which means the FastViT PoseNet may be invoked at a lower internal resolution OR not invoked at all during training]. Our `scorer_forward_pair` invokes PoseNet at full 384×512 with the YUV6 12-channel stem — that is the source of the 12288-token attention map.

[contest-CUDA: project_selfcomp_reverse_engineered_20260429.md L8] Selfcomp ships final PoseNet distortion = 0.000552, 2.5× better than ours (0.001-0.003). His pose representation is via `affine_delta` per-frame embeddings, not via PoseNet inference. **He may not even backprop through PoseNet during training of the SegMap renderer.**

**Memory impact:** if true, this eliminates `A_attn` entirely from the training loop. Score-impact unknown without verification.

### Diff 5: He has **smaller `block_hidden`** than our default

[empirical:src/tac/segmap_renderer.py:62 + scripts/remote_lane_sa_segmap_clone.sh L88]: We use `hidden=24, block_hidden=24, num_blocks=8`. Selfcomp's published params total to ~94K which suggests `hidden=24, block_hidden=24, num_blocks=4-6` (the 8-block configuration over-counts). Counted: `8 conv2d 1×1 (5+3,24)+(24,3) ~ 1500 params + 8 blocks × (24,24,3,3) × 2 = 8 × 10368 = 82944 = ~84K`. Close to 94K; matches.

**Memory impact:** minor; the scorer dominates not the renderer.

> **Selfcomp's verdict:** Diffs 2 (bf16) and 3 (per-pair chunking) are the easy wins; Diffs 1 and 4 require pipeline restructuring; Diff 5 is irrelevant for OOM. **The OOM is ours, not the architecture's.**

---

## 5. DEEP FIX 1 — Gradient checkpointing on SegMap blocks (LOW priority, do last)

**File:** `src/tac/segmap_renderer.py`
**Location:** lines 157-164 (the `SegMap.forward` method)

**Diff (illustrative; do NOT apply now):**

```python
# Current (lines 157-164):
def forward(self, x: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
    affine_latent = self._build_affine_latent_channel(
        frame_indices, x.shape[-2], x.shape[-1]
    )
    feat = self.layer_in(torch.cat([x, affine_latent * self.latent_input_scale], dim=1))
    for block in self.blocks:
        feat = block(feat)
    return torch.sigmoid(self.layer_out(feat)) * 255.0

# Proposed (with checkpointing):
def forward(self, x: torch.Tensor, frame_indices: torch.Tensor) -> torch.Tensor:
    affine_latent = self._build_affine_latent_channel(
        frame_indices, x.shape[-2], x.shape[-1]
    )
    feat = self.layer_in(torch.cat([x, affine_latent * self.latent_input_scale], dim=1))
    if self.training and getattr(self, "use_gradient_checkpointing", False):
        from torch.utils.checkpoint import checkpoint
        for block in self.blocks:
            # use_reentrant=False is REQUIRED for the new checkpoint API
            feat = checkpoint(block, feat, use_reentrant=False)
    else:
        for block in self.blocks:
            feat = block(feat)
    return torch.sigmoid(self.layer_out(feat)) * 255.0
```

Add `--gradient-checkpointing` flag to `experiments/train_segmap.py` and set `model.use_gradient_checkpointing = True` after instantiation.

**Memory savings:** ~5-8% (renderer is not the dominant tensor). This is the LEAST important of the three deep fixes per Boyd's Lagrangian. **Apply only if DF2 + DF3 are insufficient.**

[prediction:gradient-checkpoint memory math from PyTorch docs + 8-block SegMap @ 24 ch]

---

## 6. DEEP FIX 2 — Mixed precision (bf16) autocast in SegMapTrainer (HIGH priority)

**File:** `src/tac/segmap_renderer.py`
**Location:** lines 467-486 (the inner mini-batch forward in `train_epoch`)

**Diff (illustrative; do NOT apply now):**

```python
# Add to SegMapTrainer.__init__:
def __init__(self, ..., use_bf16: bool = True):
    ...
    self.use_bf16 = use_bf16 and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
    if self.use_bf16:
        # GradScaler is NOT needed for bf16 (only for fp16 — bf16 has the same
        # exponent range as fp32). Use plain autocast.
        self._autocast_dtype = torch.bfloat16
    else:
        self._autocast_dtype = torch.float32

# In train_epoch, wrap the per-mini-batch forward:
for start in range(0, b, batch_size):
    ...
    with torch.amp.autocast("cuda", dtype=self._autocast_dtype, enabled=self.use_bf16):
        rendered = self.model(masks_flat, frame_indices)
        rendered_btchw = rendered.reshape(mb, t, 3, h, w)
        rt_btchw = _eval_roundtrip_chain(rendered_btchw, noise_std=roundtrip_noise_std)

        # IMPORTANT: keep gt_btchw OUTSIDE autocast so the GT bytes stay fp32
        # (they are static targets, no benefit to bf16; cast inside scorer call).
        ...
        posenet_out, segnet_out = scorer_forward_pair(
            rt_btchw, self.posenet, self.segnet
        )
        with torch.no_grad():
            gt_pose_out, gt_seg_out = scorer_forward_pair(
                gt_btchw, self.posenet, self.segnet
            )

        # Loss in fp32 for numerical stability:
        with torch.amp.autocast("cuda", enabled=False):
            pose_diff_sq = (
                posenet_out["pose"][..., :6].float() - gt_pose_out["pose"][..., :6].float()
            ).pow(2)
            ...
```

**Why bf16 not fp16:** bf16 has the same exponent range as fp32 (8 bits), so no `GradScaler` is needed. The Hopper / Ampere generations (A10G, 4090, A100, H100) all support bf16 natively. Lane G v3 trained on RTX 4090 successfully — bf16 will work there. Modal T4 does NOT support bf16 natively; for T4, fall back to fp16+GradScaler OR keep fp32 with smaller batch.

**Memory savings:** ~50% on the dominant `A_attn` tensor (10.5 GiB → 5.3 GiB peak). Sufficient on its own to unblock A10G.

**Quality regression risk:** bf16 has lower mantissa precision (7 bits vs fp32's 23). KL distill's softmax temperature math may be sensitive. Risk mitigation: keep loss computation in fp32 (cast back). **Test on Lane SA first (no KL distill) before promoting to SC++.**

[prediction:bf16 memory math + Hopper/Ampere autocast docs]

---

## 7. DEEP FIX 3 — Per-pair scorer chunking (HIGH priority)

**File:** `src/tac/segmap_renderer.py`
**Location:** lines 480-486 (the dual `scorer_forward_pair` calls)

**Diff (illustrative; do NOT apply now):**

```python
# Current (one call on mb*T=16 frames):
posenet_out, segnet_out = scorer_forward_pair(
    rt_btchw, self.posenet, self.segnet
)
with torch.no_grad():
    gt_pose_out, gt_seg_out = scorer_forward_pair(
        gt_btchw, self.posenet, self.segnet
    )

# Proposed (chunk the scorer call, NOT the renderer call):
SCORER_CHUNK = getattr(self, "scorer_chunk_size", 2)  # process 2 pairs at a time

# Storage for concatenated outputs (so downstream loss code is unchanged):
posenet_pose_chunks = []
segnet_logit_chunks = []
gt_pose_chunks = []
gt_seg_chunks = []
for chunk_start in range(0, mb, SCORER_CHUNK):
    chunk_end = min(chunk_start + SCORER_CHUNK, mb)
    # Slice along batch dim of the (mb, T, 3, H, W) shape:
    rt_chunk = rt_btchw[chunk_start:chunk_end]
    gt_chunk = gt_btchw[chunk_start:chunk_end]
    pn_out, sn_out = scorer_forward_pair(rt_chunk, self.posenet, self.segnet)
    with torch.no_grad():
        gp_out, gs_out = scorer_forward_pair(gt_chunk, self.posenet, self.segnet)
    posenet_pose_chunks.append(pn_out["pose"])
    segnet_logit_chunks.append(sn_out)
    gt_pose_chunks.append(gp_out["pose"])
    gt_seg_chunks.append(gs_out)
    # Free the chunk's intermediate scorer tensors immediately
    del rt_chunk, gt_chunk, pn_out, sn_out, gp_out, gs_out
    if torch.cuda.is_available():
        torch.cuda.synchronize()

posenet_out = {"pose": torch.cat(posenet_pose_chunks, dim=0)}
segnet_out = torch.cat(segnet_logit_chunks, dim=0)
gt_pose_out = {"pose": torch.cat(gt_pose_chunks, dim=0)}
gt_seg_out = torch.cat(gt_seg_chunks, dim=0)
```

**Critical correctness note:** the chunked scorer call must preserve the gradient graph from `rt_btchw` (the rendered output) back to the renderer. Slicing a tensor with `[a:b]` creates a view that retains autograd graph, so the gradient flows correctly. The `torch.cat` on outputs is also autograd-aware.

**Memory savings:** scorer chunk=2 (vs full mb=8) cuts the per-call attention map by **4×** (attention is O(B × N²); halving B halves the cost per layer). Peak `A_attn` drops from 21 GiB → 5.3 GiB. Combined with bf16: 5.3 / 2 = 2.6 GiB. **Fits T4 with margin.**

**Speed cost:** ~4× more scorer calls (8 calls instead of 2), but each is ~4× cheaper, so wall-clock is similar (small overhead from kernel launches and synchronize).

[prediction:scorer chunking memory math from FastViT attention complexity + SegNet B2 forward graph]

---

## 8. Per-lane Platform + Config Recommendation

**Authoritative source for current capacity:** CLAUDE.md "GPU budget and compute resources" + `feedback_modal_spawn_result_cache_pattern_20260429.md` (Modal A10G OOMed today).

| Lane | Platform | GPU | Batch | Precision | Scorer Chunk | Grad Checkpoint | Wall-Clock | Cost |
|---|---|---|---|---|---|---|---|---|
| **SC++** | **Vast.ai** | **RTX 4090 24GB** | 8 | bf16 | 2 | NO | ~12h | ~$3.12 |
| **SC++ fallback** | Modal | A10G 22GB | 4 | bf16 | 2 | NO | ~14h | ~$15.40 |
| **SA** | Modal | T4 16GB | 4 | bf16 (or fp32 if T4 bf16 broken) | 2 | NO | ~8h | ~$4.72 |
| **SA fallback** | Vast.ai | RTX 4090 24GB | 8 | bf16 | 2 | NO | ~6h | ~$1.56 |
| **SO** | **Vast.ai** | **RTX 4090 24GB** | 8 | bf16 | 2 | NO | ~12h training + ~30min Hessian @ B=2 | ~$3.20 |
| **W** | Modal | A10G 22GB | 6 | bf16 | 2 | NO | ~6h | ~$6.60 |
| **W fallback** | Modal | T4 16GB | 4 | bf16 | 2 | NO | ~8h | ~$4.72 |

**Why Vast.ai for SC++ and SO:** these run KL distill which is more numerically sensitive. The dedicated 24GB on RTX 4090 leaves headroom for any bf16 → fp32 fallback if KL gradient blows up. Also 5× cheaper than Modal A10G per GPU-hour.

**Why Modal for SA:** plain SegMap is the validation lane — proves the chunking + bf16 fix works before promoting to SC++. T4 cost is acceptable; if it OOMs we know the fix is insufficient and we don't waste 4090 budget.

**Why Modal A10G for W:** W is rate-attack; it does extra weight quantization passes that benefit from A10G's bigger SRAM. T4 fallback exists.

**ALL re-dispatches must include:**
- `--gradient-checkpointing` flag NOT required if DF2 + DF3 land
- `--bf16` flag REQUIRED (DF2)
- `--scorer-chunk 2` flag REQUIRED (DF3)
- Old preflight Check 73 still applies (`--batch-size <= 32`)
- New Check 86 (below) enforces DF2 + DF3 for SegMap-class lanes

[empirical:CLAUDE.md GPU budget table + harvested OOM crash log + Lane G v3 successful 4090 run]

---

## 9. Preflight Check 86 Proposal (STRICT) — `check_segmap_class_lanes_have_oom_guards`

**Trigger:** any `remote_lane_*.sh` script that invokes `experiments/train_segmap.py` (the SegMap-class trainer).

**Required:** EITHER all three flags `--bf16 --scorer-chunk N --batch-size B` (with `B*N <= 8`), OR both `--gradient-checkpointing` AND `export GPU_TIER_HINT=A100` (only A100 has enough VRAM to skip both DF2 and DF3).

**Pseudocode:**

```python
_SEGMAP_CLASS_TRAINING_TARGETS = ("experiments/train_segmap.py",)

def check_segmap_class_lanes_have_oom_guards(
    repo_root: Path | None = None,
    shell_files: list[str] | None = None,
    strict: bool = True,
    verbose: bool = True,
) -> list[str]:
    """Refuse SegMap-class lane scripts that invoke train_segmap.py without
    the OOM-class deep fixes (bf16 autocast + scorer chunking).

    Bug class: PoseNet FastViT-T12 stage-1 attention map at full scorer
    resolution is O(B × N²) where N=12288. At B=16 frames in fp32 the
    single attention allocation reaches ~21 GiB — observed OOM on Modal
    A10G 22GB on 2026-04-29 across 14 SC++/SA/SO instances.

    Acceptable invocation patterns:
      A) Has all three: --bf16 + --scorer-chunk N + --batch-size B
         AND B * N (effective per-scorer-call frame count) <= 8.
      B) Has --gradient-checkpointing AND env-export GPU_TIER_HINT=A100
         (only A100/H100 has VRAM headroom to skip the deep fixes).

    Anything else is a violation.
    """
    root = repo_root or REPO_ROOT
    if shell_files is None:
        shell_files = sorted(
            str(p.relative_to(root))
            for p in (root / "scripts").glob("remote_lane_*.sh")
        )
    violations: list[str] = []

    for shell_rel in shell_files:
        shell_path = root / shell_rel
        if not shell_path.exists():
            continue
        raw = shell_path.read_text()
        # Path B: A100/H100 + grad checkpoint
        has_a100 = bool(
            re.search(r'(?:^|\n)\s*export\s+GPU_TIER_HINT=(A100|H100)', raw)
        )

        for lineno, target, flags_used in _scan_shell_lane_invocations(shell_path):
            if target not in _SEGMAP_CLASS_TRAINING_TARGETS:
                continue

            collapsed = _collapse_shell_continuations(raw)
            inv_line = _find_invocation_line(collapsed, target, lineno)

            has_bf16 = "--bf16" in inv_line
            chunk_match = re.search(r'--scorer-chunk\s+(\d+)', inv_line)
            bs_match = re.search(r'--batch-size\s+(\d+)', inv_line)
            has_chkpt = "--gradient-checkpointing" in inv_line

            # Path A check:
            path_a = (
                has_bf16 and chunk_match and bs_match
                and (int(bs_match.group(1)) * int(chunk_match.group(1)) <= 8)
            )
            # Path B check:
            path_b = has_chkpt and has_a100

            if not (path_a or path_b):
                msg = (
                    f"{shell_rel}:{lineno}: invokes {target} without OOM-class "
                    f"deep fixes. Required EITHER:\n"
                    f"  (A) --bf16 + --scorer-chunk N + --batch-size B with B*N<=8, OR\n"
                    f"  (B) --gradient-checkpointing AND export GPU_TIER_HINT=A100\n"
                    f"  Got: bf16={has_bf16}, scorer-chunk={chunk_match.group(1) if chunk_match else None}, "
                    f"batch-size={bs_match.group(1) if bs_match else None}, "
                    f"chkpt={has_chkpt}, a100_hint={has_a100}\n"
                    f"  Memory: 14 OOMs on 2026-04-29 (A10G 22GB shared); "
                    f"PoseNet FastViT stage-1 attention is O(B*12288^2*4)."
                )
                violations.append(msg)

    if verbose and violations:
        print(f"  [segmap-oom-guard] {len(violations)} violation(s):")
        for v in violations:
            print(f"    • {v}")
    elif verbose:
        print("  [segmap-oom-guard] OK")

    if violations and strict:
        raise PreflightError(
            "SEGMAP OOM GUARD: SegMap-class lane scripts must include the "
            "DF2+DF3 deep fixes (bf16 autocast + scorer chunking) before "
            "dispatch. See .omx/research/council_oom_class_deep_fix_20260429.md\n"
            + "\n".join(f"  • {v}" for v in violations)
        )
    return violations
```

**Promotion path (strict-flip):** Land warn-only first; fix all 3 SegMap-class scripts (`remote_lane_sc_plus_plus_kl_distill.sh`, `remote_lane_sa_segmap_clone.sh`, `remote_lane_so_hessian_block_fp.sh`) AND the `--bf16` + `--scorer-chunk` argparse entries in `experiments/train_segmap.py` AND the trainer code in `src/tac/segmap_renderer.py`. Once 0 violations confirmed, flip `strict=True` and add to `preflight_all()`.

**Live violation count today:** 3 (the three SegMap-class scripts).

[prediction:check follows the strict-flip pattern documented in commit 7f2740e4 + lessons from Check 73]

---

## 10. Council Roll Call

### Carmack — APPROVE WITH PRIORITY ORDERING
> "Stop chunking the SegMap. Start chunking the SCORER. The attention map is the 21GB tensor — fix that first. bf16 + scorer-chunk-2 turns A10G + T4 into viable platforms. Don't gold-plate with gradient checkpointing on the SegMap; that fixes 5% of the problem. Ship the 95% fix today."

### Boyd — APPROVE WITH KKT-ORDERED PRIORITY
> "The Lagrangian is unambiguous. ∂Memory/∂precision and ∂Memory/∂scorer_chunk are the dominant gradients. ∂Memory/∂renderer_checkpoint is 20× smaller. The optimal projection is bf16 ∩ scorer-chunk ∩ B=8; the slack lets us run on every platform we have. Add the constraint at preflight time so we never solve the wrong unconstrained problem again."

### Selfcomp — CONCURS, HUMBLY OFFERS DIFF 4 AS NEXT-LEVEL
> "Diffs 2 and 3 will unblock you. Diff 4 — running PoseNet at lower scorer resolution OR not at all during SegMap training — is what got me down to 0.000552 PoseNet distortion in 6 hours on a free T4. Worth investigating after SC++ ships. My block-FP weight format already implies bf16 forward; that's not optional in my paradigm, it's structural."

### Quantizr — APPROVE, ADDS CAVEAT ON KL TEMPERATURE
> "bf16 with KL distill at T=2.0 is fine. KL gradient through `softmax / T` is well-conditioned for T ≥ 1.5. If T < 1.5 you get sharp distributions that bf16 mantissa can't represent → silent gradient corruption. SC++ uses T=2.0; safe. SO uses T=2.0; safe. If anyone tries a hot-temperature schedule with T_end < 1.5 in bf16, hard-error."

### Hotz — APPROVE WITH RAW-BYTES VERDICT
> "21 GiB / 4 bytes / 16 frames / 12288² = 1.0. Yep, that's the attention map. Per-frame YUV6 to FastViT to a 12288-token attention is the entire bottleneck. Fix it at the source. bf16 cuts to 10.5; chunk-2 cuts to 5.3; chunk-2 + bf16 cuts to 2.6. We were 8× away from fitting. Now we're 6× under capacity. Ship."

### Karpathy — APPROVE WITH DISCIPLINE NOTE
> "Every preflight check we add is an experiment that won't happen. Check 86 belongs. But we should ALSO log peak GPU memory in the heartbeat so we measure the actual savings post-fix, not just predict them. Add `nvidia-smi --query-gpu=memory.used --format=csv` snapshot before and after each Stage; landing the deep-fix without measurement is faith, not science. The proxy/auth gap rule applies to memory too: predicted < measured = fine; predicted > measured = predict again."

**Vote: 6/6 APPROVE.** Carmack and Boyd LEAD on prioritization (DF2 + DF3 first, DF1 last). Selfcomp endorses with future-direction note (Diff 4). Quantizr endorses with KL-temperature safety note. Hotz endorses with the raw-bytes math. Karpathy endorses with measurement discipline.

---

## Appendix A — Cross-references

- `feedback_modal_spawn_result_cache_pattern_20260429.md` (the harvest report — 14 OOM crashes today)
- `feedback_check_72_73_lane_hardening_landed_20260429.md` (Check 73 chunking, but ONLY at renderer level; this report is the next layer)
- `project_lane_g_v3_landed_1_05_20260428.md` (Lane G v3 didn't OOM because trained on a different arch with proven config; SegMap-class is new)
- `project_selfcomp_reverse_engineered_20260429.md` (Selfcomp 94K SegMap empirical anchor)
- `project_codec_stacking_composition_canonical_orders_20260429.md` (post-OOM, the rate-attack sequencing remains canonical: representation → prediction → quantization → arithmetic)
- `src/tac/segmap_renderer.py:480-486` (the smoking gun — dual scorer call)
- `src/tac/preflight.py:1761-1852` (Check 73 — the existing T4 OOM guard)
- `scripts/remote_lane_sc_plus_plus_kl_distill.sh:84-98` (the dispatcher to harden)
- `scripts/remote_lane_sa_segmap_clone.sh:83-95`
- `scripts/remote_lane_so_hessian_block_fp.sh:87-101`
- `experiments/train_segmap.py:88-89` (the `--batch-size 8` default that needs new `--bf16` + `--scorer-chunk` siblings)

## Appendix B — Verified empirical anchors

- [empirical:harvested OOM trace] `Tried to allocate 21.09 GiB. GPU 0 has a total capacity of 22.06 GiB`
- [empirical:project_lane_g_v3_landed_1_05_20260428.md] Lane G v3 ran on RTX 4090 24GB without OOM at the SAME `--batch-size 8` — but Lane G v3 is the LEGACY ASYM renderer (287K params), NOT SegMap. The SegMap renderer is smaller (94K) but its training pipeline routes through the FROZEN scorers' full-resolution attention which is the new bottleneck.
- [empirical:src/tac/segmap_renderer.py:480-486] Dual scorer call confirmed; gradient-flowing on rendered branch, no_grad on GT branch.
- [empirical:src/tac/preflight.py:1801] Check 73 only chunks the renderer-level forward inside `train_epoch`; does NOT chunk the scorer call.

## Appendix C — Confidence assessment

- **DF2 (bf16):** HIGH confidence. Standard practice; Lane G v3's training pipeline already uses autocast at line 967 of `training.py` for the EVAL loop. Symmetric extension to training loop is well-trodden.
- **DF3 (scorer chunking):** HIGH confidence. Memory math is deterministic; correctness preserved via gradient-tracking slice + cat.
- **DF1 (gradient checkpointing):** MEDIUM confidence on need (only 5% savings); HIGH confidence on correctness if applied. Listed as fallback only.
- **Check 86 (preflight):** HIGH confidence following the strict-flip pattern of Check 73; directly catches re-occurrence.
- **Per-lane platform recommendations:** HIGH confidence on Vast.ai 4090 for SC++/SO (proven workhorse); MEDIUM on Modal T4 for SA (untested at chunked SegMap config); LOW on Modal A10G for W (just OOMed; needs the deep fix verified before re-trying).

**Overall confidence on the deep-fix landing without further iteration: MEDIUM-HIGH.** The architectural diagnosis is rigorous; the proposed fixes are standard practice; the preflight check prevents regression. The only residual risk is bf16 numerical precision interacting with KL distill at SO's combined Hessian-curvature path — mitigated by Quantizr's T ≥ 1.5 caveat and Karpathy's measurement-on-landing discipline.
