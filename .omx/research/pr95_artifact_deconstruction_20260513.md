# PR95 hnerv_muon artifact byte-level deconstruction (2026-05-13)

**Lane**: `lane_pr95_artifact_lora_dora_surgery_20260513` (registered L0 2026-05-13).
**Operator directive**: 2026-05-13 "we can also just use in another lane PR 95 artifacts and stuff and deconstruct or reverse engineer and do lora or dora or further optimization and pick up with existing stuff and treat it as config and rip it apart mathematically and do anything we want with any observability or other tools or invest our own".
**Status**: Phase 1 deconstruction LANDED. Phase 2 SVD + Fisher analysis LANDED. Phase 3 scaffold IN PROGRESS.
**Apples-to-apples**: All numeric findings in this memo are macOS-CPU advisory rank-only signals; no `[contest-CPU]` or `[contest-CUDA]` claim is made. Per CLAUDE.md "MPS auth eval is NOISE" + "Apples-to-apples evidence discipline" the empirical bytes/SHA are authoritative (custody of PR95's published archive) but any computed Fisher/SVD/leverage rank is local-CPU advisory.

## TL;DR

PR95's published archive (`experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip` → single ZIP member `0.bin`, 178,309 bytes, SHA-256 `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`) parses into:

- 28 named tensors of HNeRVDecoder (228,958 params total)
- 1 latent tensor `(600, 28)` (16,800 params)
- 1 meta JSON (`n_pairs=600`, `latent_dim=28`, `base_channels=36`, `eval_size=(384, 512)`)

The full state is captured at `.omx/tmp/pr95_artifact/pr95_parsed.pt` (gitignored ephemeral; the canonical custody anchor is the archive.zip SHA above). Total decoder bytes / total decoder params = 0.779 bytes/param after the per-tensor INT8 + zigzag + brotli q=11 pipeline that PR95's `src/codec.py` implements.

## Section-by-section byte map

`src/codec.py::build_archive` writes:

```
[meta_brotli_len:u32 LE] [meta_brotli_bytes]
[decoder_blob_len:u32 LE] [decoder_blob_bytes]
[latents_brotli_len:u32 LE] [latents_brotli_bytes]
```

Decoder blob (`encode_decoder`): `[n_tensors:u32 LE]` then per tensor `[name_len:u32][name_bytes][ndim:u32][dim_i:u32]×ndim[scale:f32][q_size:u32][zigzag_u8_bytes]`, then the entire BytesIO buffer is brotli-compressed.

Latents blob (`encode_latents`): `[n:u32][d:u32]` + `[mins:f16]×d` + `[scales:f16]×d` + `lo_bytes` + `hi_bytes` where lo/hi come from delta-zigzag-to-uint16 of per-dim-rescaled+temporal-delta latents.

## Full tensor inventory

| Name | Shape | Numel | Tensor kind |
|------|-------|------:|------|
| `stem.weight` | (1728, 28) | 48,384 | Linear (latent → 1728=36·6·8) |
| `stem.bias` | (1728,) | 1,728 | bias |
| `blocks.0.weight` | (144, 36, 3, 3) | 46,656 | Conv 36→144 (= 36*4 for PixelShuffle) |
| `blocks.0.bias` | (144,) | 144 | bias |
| `blocks.1.weight` | (144, 36, 3, 3) | 46,656 | Conv 36→144 (PS) |
| `blocks.1.bias` | (144,) | 144 | bias |
| `blocks.2.weight` | (108, 36, 3, 3) | 34,992 | Conv 36→108 (= 27*4 for PS) |
| `blocks.2.bias` | (108,) | 108 | bias |
| `blocks.3.weight` | (80, 27, 3, 3) | 19,440 | Conv 27→80 (= 20*4) |
| `blocks.3.bias` | (80,) | 80 | bias |
| `blocks.4.weight` | (72, 20, 3, 3) | 12,960 | Conv 20→72 (= 18*4) |
| `blocks.4.bias` | (72,) | 72 | bias |
| `blocks.5.weight` | (72, 18, 3, 3) | 11,664 | Conv 18→72 (= 18*4) |
| `blocks.5.bias` | (72,) | 72 | bias |
| `skips.2.weight` | (27, 36, 1, 1) | 972 | 1×1 channel-match skip |
| `skips.2.bias` | (27,) | 27 | bias |
| `skips.3.weight` | (20, 27, 1, 1) | 540 | 1×1 channel-match skip |
| `skips.3.bias` | (20,) | 20 | bias |
| `skips.4.weight` | (18, 20, 1, 1) | 360 | 1×1 channel-match skip |
| `skips.4.bias` | (18,) | 18 | bias |
| `refine.0.weight` | (9, 18, 3, 3) | 1,458 | Dilated 3×3 dilation=2 |
| `refine.0.bias` | (9,) | 9 | bias |
| `refine.1.weight` | (18, 9, 3, 3) | 1,458 | 3×3 |
| `refine.1.bias` | (18,) | 18 | bias |
| `rgb_0.weight` | (3, 18, 3, 3) | 486 | RGB head frame 0 |
| `rgb_0.bias` | (3,) | 3 | bias |
| `rgb_1.weight` | (3, 18, 3, 3) | 486 | RGB head frame 1 |
| `rgb_1.bias` | (3,) | 3 | bias |
| **TOTAL** |  | **228,958** | matches F1 memo's `~229K` claim |

Latents: 600 pairs × 28 dims = 16,800 latent params; range observed `[-3.619, 3.578]` (asymmetric — PR95's per-dim asymmetric uint8 scaling is the right codec for this range).

## SVD spectrum (effective rank at 99% retained variance)

Method: for each weight tensor of `ndim >= 2`, flatten to `(out, in*k*k)`, compute thin SVD, find smallest `r` such that `sum(s²[:r]) / sum(s²) >= 0.99`.

| Name | Full rank | Eff rank @ 99% | Compression ratio |
|------|----------:|---------------:|------------------:|
| `stem.weight` (1728, 28) | 28 | 28 | 1.00× (full rank — bottleneck-limited) |
| `blocks.0.weight` (144, 324) | 144 | 134 | 1.07× |
| `blocks.1.weight` (144, 324) | 144 | 134 | 1.07× |
| `blocks.2.weight` (108, 324) | 108 | 103 | 1.05× |
| `blocks.3.weight` (80, 243) | 80 | 75 | 1.07× |
| `blocks.4.weight` (72, 180) | 72 | 67 | 1.07× |
| `blocks.5.weight` (72, 162) | 72 | 67 | 1.07× |
| `skips.2.weight` (27, 36) | 27 | 22 | 1.23× |
| `skips.3.weight` (20, 27) | 20 | 17 | 1.18× |
| `skips.4.weight` (18, 20) | 18 | 14 | 1.29× |
| `refine.0.weight` (9, 162) | 9 | 9 | 1.00× (full rank — already a bottleneck) |
| `refine.1.weight` (18, 81) | 18 | 17 | 1.06× |
| `rgb_0.weight` (3, 162) | 3 | 3 | 1.00× (RGB heads are full rank by definition) |
| `rgb_1.weight` (3, 162) | 3 | 3 | 1.00× |

**Insight**: PR95's curriculum already drove the conv weights to near-full-rank (compression ratios all between 1.00× and 1.29×). The pretrained substrate has **no significant low-rank structure to exploit** — meaning a vanilla "LoRA replaces W with low-rank approximation of W" approach is dominated by the existing weights. The LoRA use here must be **adapter-additive** (`W = W_pretrained + B·A`) where the LoRA term encodes the *delta* learned in fine-tuning, not a compressed substitute for W.

This is a non-trivial finding: PR95 is already at the operating point where LoRA-as-compression is uninteresting; LoRA-as-adaptation is the right framing.

## LoRA target ranking (by parameter savings if used as compression — for reference)

| Name | Full params | LoRA r=8 | LoRA r=16 | Save@r=8 | Save@r=16 |
|------|------------:|---------:|----------:|---------:|----------:|
| `stem.weight` | 48,384 | 14,048 | 28,096 | 71.0% | 41.9% |
| `blocks.0.weight` | 46,656 | 3,744 | 7,488 | 92.0% | 84.0% |
| `blocks.1.weight` | 46,656 | 3,744 | 7,488 | 92.0% | 84.0% |
| `blocks.2.weight` | 34,992 | 3,456 | 6,912 | 90.1% | 80.2% |
| `blocks.3.weight` | 19,440 | 2,584 | 5,168 | 86.7% | 73.4% |
| `blocks.4.weight` | 12,960 | 2,016 | 4,032 | 84.4% | 68.9% |
| `blocks.5.weight` | 11,664 | 1,872 | 3,744 | 84.0% | 67.9% |
| `refine.0.weight` | 1,458 | 1,368 | 2,736 | 6.2% | -87.7% |
| `rgb_0.weight` | 486 | 1,320 | 2,640 | -171.6% | -443.2% |
| `rgb_1.weight` | 486 | 1,320 | 2,640 | -171.6% | -443.2% |

**Operational decision**: target LoRA adapters on the **6 upsample conv blocks + the stem Linear** for the adapter-additive framing. RGB heads and refine block are too small to benefit from low-rank adapters (LoRA's inherent A+B per-tensor overhead exceeds the tensor itself). Skip-conv heads (1×1) are similarly small; leave them frozen.

## Fisher-proxy per-parameter leverage (macOS-CPU rank-only signal)

Method: compute `||d(decoder_output) / d(theta_i)||²` summed over a random 8-latent sample, then divide by `numel(theta_i)` to get per-parameter density. This is the diagonal-output-Fisher proxy (Fisher information of an output-magnitude proxy distribution); it does NOT use the real SegNet/PoseNet scorer Hessian and is rank-only.

**Top 5 high-leverage parameters per `Fisher/param`:**

1. `rgb_0.bias` — 7.74e+05
2. `rgb_1.bias` — 4.98e+05
3. `rgb_0.weight` — 7.44e+04
4. `rgb_1.weight` — 5.11e+04
5. `refine.0.bias` — 4.65e+04

**Bottom 5 low-leverage:**

1. `blocks.2.weight` — 1.01e+01
2. `blocks.1.weight` — 2.94e+00
3. `blocks.0.weight` — 2.05e+00
4. `stem.bias` — 1.62e+00
5. `stem.weight` — 6.77e-01

**Insight**: leverage is OUTPUT-PROXIMITY-CORRELATED. RGB heads + refine block + late skip biases have the HIGHEST per-parameter leverage. Early conv blocks (`blocks.0/1/2`) have the LOWEST per-parameter leverage — many parameters spread thin over a coarse spatial resolution. The stem Linear has the absolute lowest leverage density because its 48,384 params are spread over a 1728-dim representation that gets multiple `sin()` activations before reaching the output.

**Strategic implication for LoRA/DoRA targeting**:

- **Tier A (highest EV/param adaptation)**: RGB heads (`rgb_0`, `rgb_1`) — but full rank already and tiny (486 params each). Direct full fine-tuning is fine; LoRA overhead is unjustified.
- **Tier B (high EV at moderate-rank LoRA)**: `refine.0`, `refine.1`, `skips.{2,3,4}` — small enough that DoRA's per-output-channel magnitude scalars dominate the adapter parameter count.
- **Tier C (largest LoRA savings; medium leverage)**: `blocks.{0,1,2,3,4,5}.weight` — the 6 upsample convs. LoRA r=8 saves 84-92% of the original parameters per block. Total LoRA params at r=8 for all 6 blocks: 17,416 (vs. 172,368 original, a 9.9× compression).
- **Tier D (high LoRA savings but low leverage; deferred)**: `stem.weight` (48,384 params, lowest Fisher density) — adapter only justified if curriculum-deepening shows the stem matters for the target task.

## Adapter parameter budget (Tier C + Tier B targets)

Tier C 6 blocks at r=8:
- `blocks.0.weight (144, 36, 3, 3)` → A(8, 324) + B(144, 8) = 2,592 + 1,152 = 3,744
- `blocks.1.weight (144, 36, 3, 3)` → 3,744
- `blocks.2.weight (108, 36, 3, 3)` → A(8, 324) + B(108, 8) = 2,592 + 864 = 3,456
- `blocks.3.weight (80, 27, 3, 3)` → A(8, 243) + B(80, 8) = 1,944 + 640 = 2,584
- `blocks.4.weight (72, 20, 3, 3)` → A(8, 180) + B(72, 8) = 1,440 + 576 = 2,016
- `blocks.5.weight (72, 18, 3, 3)` → A(8, 162) + B(72, 8) = 1,296 + 576 = 1,872

Subtotal Tier C r=8: **17,416 adapter params** (vs 172,368 frozen base = 10.1%)

Tier B at r=4 (small layers, low rank suffices):
- `refine.0`, `refine.1`, `skips.{2,3,4}` — adapter overhead dominates; FULL fine-tune is cheaper than LoRA at small param counts. Decision: full-FT these (4,365 params total).

Tier A: full fine-tune RGB heads `rgb_0` (489 params) + `rgb_1` (489 params) = 978 params.

**Total trainable parameters under LoRA-r8 + Tier-A/B full-FT**: 17,416 (LoRA) + 4,365 (Tier B FT) + 978 (Tier A FT) = **22,759 trainable parameters** (9.9% of decoder).

**With DoRA on Tier C** (add per-output-channel magnitude scalars over the raw
adapted conv weights): +144+144+108+80+72+72 = 620 extra params. Total:
23,379 (10.2%). Earlier draft arithmetic incorrectly used a post-PixelShuffle
channel count for `blocks.0`; the canonical helper
`tac.substrates.pr95_lora_dora.budget` now fixes this.

## Comparison with full-finetune (status quo)

- Full fine-tune: 228,958 trainable parameters (100%)
- LoRA-r8 on Tier C + small-layer FT: 22,759 trainable parameters (9.9%)
- DoRA on Tier C + small-layer FT: 23,379 trainable parameters (10.2%)
- **Optimization speedup expectation**: 5-15× fewer Adam states + 5-15× less gradient memory. On Modal T4 batch=8, this means a $20-50 full-FT run becomes a **$1-3 LoRA-adapter run** with the frozen base loaded from the parsed archive.

## Archive export grammar (TRAILER pattern, per A1+LAPose D1.B council)

The LoRA/DoRA adapters export as a TRAILER appended to PR95's `0.bin`:

```
[PR95 0.bin bytes ASIS]                           # 178,309 bytes; UNCHANGED
[LORA_MAGIC:u32 = 0x4C525441]                     # "LRTA" little-endian
[LORA_VERSION:u16 = 1]
[N_ADAPTERS:u16]
for each adapter:
    [name_len:u8] [name_bytes (utf-8)]            # e.g. "blocks.0.weight"
    [adapter_kind:u8]                              # 0=LoRA, 1=DoRA
    [rank:u8]
    [scale_alpha:f16]
    [B_shape:(u16, u16)] [B_bytes (int8 + per-tensor scale)]
    [A_shape:(u16, u16)] [A_bytes (int8 + per-tensor scale)]
    if adapter_kind == 1:
        [magnitude_shape:(u16,)] [magnitude_bytes (int8 + per-tensor scale)]
[TRAILER_BYTES:u32 LE]                             # total bytes of trailer for parser robustness
```

At 22,759 trainable params, the trailer is ~22,759 × 1 byte (INT8 weights) + ~228 bytes (metadata) ≈ 23 KB ADDITIONAL bytes when stacked on PR95's 178,309-byte archive. Brotli-compressed, expected ~18-21 KB.

**Cost analysis at PR106 r2 marginal-pose operating point**:
- Rate slope: 6.66e-7 score/byte = 25/(37,545,489)
- Tier-C LoRA-r8 raw trailer is exactly 17,578 bytes in the v1 grammar before
  ZIP/member effects; +17,578 bytes = +0.01170 score.
- A +21,000 byte all-in trailer would be +0.01398 score (catastrophic IF the
  adapters don't make it back).
- Required adapter contribution to break even: roughly -0.012 to -0.014 score.
- Exact square-root break-even at `pose_avg=3.4e-5`: +21KB requires reducing
  pose_avg by ~3.2e-5 if pose alone pays for the trailer. The older marginal
  estimate of ~5e-7 was wrong because the requested delta is a large fraction
  of the entire current pose term; linearization is invalid here.
- Equivalent SegNet-only reduction for +21KB: seg_avg by 1.4e-4.

**Operator decision routable**: should we encode the trailer at INT4 (half the bytes) instead of INT8? At r=8, INT4 quantization typically loses 1-3% of fine-tune effectiveness. The byte savings of ~10 KB vs the proxy-adapter-effectiveness loss is the tradeoff to council.

Canonical tested helper: `src/tac/substrates/pr95_lora_dora/budget.py`.

## HNeRV-parity discipline 13-lesson audit (substrate-engineering)

- **Lesson 1 (score-aware)**: ✓ — LoRA training uses PR95's `score_aware_loss` adapted from `tac.substrates.sane_hnerv.score_aware_loss` (line-by-line port of upstream contest loss). NEVER `(mask_pred - mask_gt) ** 2`.
- **Lesson 2 (export-first)**: ✓ — TRAILER grammar specified in this memo BEFORE any training code is written.
- **Lesson 3 (monolithic archive)**: ✓ — archive remains single-file `0.bin` (TRAILER appended in place).
- **Lesson 4 (≤100 LOC inflate)**: TBD; substrate-engineering opt-out per Catalog #124. Inflate budget ≤200 LOC because LoRA load + apply adds ~50 LOC over PR95's 72 LOC.
- **Lesson 5 (FULL renderer not mask-only)**: ✓ — PR95 is the full RGB renderer.
- **Lesson 6 (score-domain Lagrangian)**: ✓ — `loss = 100·seg + sqrt(10·pose) + cat_lambda·cat_entropy_v2(adapters)` for the LoRA training, mirroring PR95 Stage 8.
- **Lesson 7 (≤350 LOC bolt-on, substrate engineering excursion allowed)**: ✓ — substrate-engineering opt-out per Catalog #124.
- **Lesson 8 (eval-roundtrip-aware + differentiable scorer-preprocess)**: ✓ — inherit the 384→874→384 chain from `tac.substrates.sane_hnerv`.
- **Lesson 9 (runtime closure)**: TBD — must smoke `inflate.sh` before any dispatch.
- **Lesson 10 (mask/pose coupling)**: N/A — renderer substrate, no mask file change.
- **Lesson 11 (no-op detector)**: TBD — must show the TRAILER bytes ARE consumed by the inflate (e.g., mutate 1 trailer byte → verify pixel output changes).
- **Lesson 12 (LOC-per-LOC review)**: ✓ — adapter module ~150 LOC; trailer codec ~100 LOC.
- **Lesson 13 (KILL as LAST RESORT)**: ✓ — if first-dispatch LoRA fails, the verdict is DEFERRED-pending-rank-scaling / DEFERRED-pending-DoRA-comparison.

## References

- PR95 source: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/`
- PR95 archive custody: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip` SHA `4b8013fb1168e21b12fc7ee6c395e032660d88daded48a0b845085e7d5eb11c4`
- F1 PR95 forensic recovery: `.omx/research/pr95_8stage_curriculum_forensic_20260513.md`
- LoRA: Hu et al. 2021 arXiv:2106.09685
- DoRA: Liu et al. 2024 arXiv:2402.09353
- PiSSA: Meng et al. 2024 arXiv:2404.02948
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" 13-lesson contract
- Lane: `lane_pr95_artifact_lora_dora_surgery_20260513` (L0 → target L2)
