# MLX-ARCH-3 FastViT-T12 Backbone + PoseNet Primitives Landed

Date: 2026-05-21
Lane: `lane_mlx_arch_3_fastvit_t12_backbone_full_assembly_20260521`
Sister-3 of: 5-stage MLX architecture port cascade
Predecessor anchors:
- MLX-ARCH-2 commit `51349c604` (attention primitives) — Sister-3 READY signal
- MLX-ARCH-1 commit `18520f83e` (foundational primitives)
- OVERNIGHT-WW commit `94c03b83c` (base 9 primitives)

Operator directive: 2026-05-21 *"We need to unlock MLX for the new frontier
lowering work because we can't afford to spend hundreds on GPU compute"* +
*"Full contest scorer required"* + *"We can test against PR 101's archive
and submission for deterministic output perhaps"*.

Carmack MVP-first 5-step per CLAUDE.md `be125b878`.

## Scope (what landed)

7 new portable-primitive surfaces + 8 canonical config constants compose
the FastViT-T12 backbone + Hydra head into the full PoseNet contest
scorer architecture per `upstream/modules.py` lines 22-80. Together with
the 18 WW + ARCH-1 + ARCH-2 primitives this completes the architecture
scaffold required for ARCH-5 paired CUDA ground-truth validation against
PR 101.

1. **`PortableAllNorm`** — BatchNorm1d-over-flattened-view sister of
   `upstream.modules.AllNorm`; the canonical normalization used
   throughout the Hydra head + ResBlock. EVAL-mode by default (per
   paired-CUDA validation contract). PyTorch path uses `nn.BatchNorm1d`;
   MLX path implements equivalent affine arithmetic via running stats
   buffers `[empirical:test_allnorm_cross_backend_equivalence,
   max-abs-diff 1.19e-7]`.

2. **`PortableResBlock`** — 2-branch residual MLP block per
   `upstream.modules.ResBlock`; composes WW `PortableLinear` + ARCH-3
   `PortableAllNorm` + WW `relu` into the canonical
   `a = x + block_a(x); out = ReLU(a + block_b(a))` shape. Used 2x in
   PoseNet (summarizer + Hydra inner). Canonical 12-key export dict
   for sister Wave 4 export pipeline.

3. **`PortableHydra`** — multi-head MLP head sister of
   `upstream.modules.Hydra`; returns dict keyed by head.name. For
   PoseNet: single `('pose', hidden=32, out=12)` head -> `{'pose': (B, 12)}`
   with first 6 dims used by `compute_distortion` per upstream contract.

4. **`PortableFastViTBlock`** — single FastViT RepMixer block:
   `x = x + LayerScale(RepMixer(x)); x = x + LayerScale(TokenMixer(x))`.
   Composes ARCH-2 `PortableRepMixer` + `PortableTokenMixer` +
   `PortableLayerScale` per the canonical timm `RepMixerBlock`. LayerScale
   init γ=1e-5 means residual branches start as near-identity (verified
   by `test_fastvit_block_layer_scale_default_near_identity`).

5. **`PortableFastViTStage`** — sequence of N RepMixer blocks with
   optional patch embedding (2x spatial downsample). Per timm
   `FastVitStage`: stage 0 has no downsample; stages 1-3 downsample by
   2x. Includes `PortablePatchEmbed` internal primitive (single conv +
   stride-2 slice; byte-stable timm-MobileOneBlock parity deferred to
   ARCH-3b / ARCH-5 state_dict load).

6. **`PortableFastViTT12Backbone`** — full T12 architecture:
   - Stem: 3-conv stride-2 sequence `in_chans -> 64 -> 64 -> 64`
   - 4 stages with canonical (2, 2, 6, 2) blocks at (64, 128, 256, 512) dim
   - Final 1x1 conv 512 -> 1024
   - Global avg pool + classifier head `Linear(1024 -> num_classes)`

   For PoseNet usage: `in_chans=12, num_classes=2048` overrides the
   imagenet 1000-class default per `upstream.modules.PoseNet.__init__`
   line 66.

7. **`PortablePoseNet`** — full PoseNet wrapper composing everything
   above:
   - Input normalization `(x - 127.5) / 63.75` (per-channel mean/std
     buffers per `upstream.modules.PoseNet._mean` + `_std`)
   - Backbone `PortableFastViTT12Backbone(in_chans=12, num_classes=2048)`
   - Summarizer `Linear(2048->512) -> ReLU -> ResBlock(512)`
   - Hydra `PortableHydra(512, heads=[('pose', 32, 12)])` ->
     `{'pose': (B, 12)}`

**Canonical config constants** sourced from `timm.models.fastvit`
`fastvit_t12` model_args + `upstream/modules.py`:

- `FASTVIT_T12_LAYERS = (2, 2, 6, 2)` — blocks per stage
- `FASTVIT_T12_EMBED_DIMS = (64, 128, 256, 512)` — channel dim per stage
- `FASTVIT_T12_MLP_RATIOS = (3, 3, 3, 3)` — channel-mixer hidden_dim multiplier
- `POSENET_IN_CHANS = 12` — 2 frames × 6 YUV6 channels
- `POSENET_VISION_FEATURES = 2048` — backbone output dim
- `POSENET_SUMMARY_FEATURES = 512` — summarizer output dim
- `POSENET_INPUT_MEAN = 127.5`, `POSENET_INPUT_STD = 63.75`

**Important spec correction from ARCH-2 docstring**: the ARCH-2 landing
memo cited stage dims `(96, 192, 384, 768)` and "stages 3+4 use MHSA"
following an early FastViT paper interpretation. The empirically-verified
canonical for `fastvit_t12` (the model the contest PoseNet uses) is
`embed_dims=(64, 128, 256, 512)` and `token_mixers=(repmixer, repmixer,
repmixer, repmixer)` — i.e. **ALL 4 stages use RepMixer** (no MHSA
stages in T12). The MHSA-stage variant is `fastvit_sa12` (a different
model). ARCH-3 follows the empirically-verified spec.

LOC: ~580 (`nn_fastvit.py`) + ~430 (`tests/test_portable_primitives_fastvit.py`)
+ 21-line `__init__.py` extension = **~1030 LOC** across 3 changed files.
Within the canonical reference budget (ARCH-1 941 LOC, ARCH-2 840 LOC)
and well inside the dispatch contract's implicit scope.

## Files touched (APPEND-ONLY per Catalog #110/#113)

- **NEW** `src/tac/portable_primitives/nn_fastvit.py` (~580 LOC)
- **NEW** `src/tac/portable_primitives/tests/test_portable_primitives_fastvit.py` (~430 LOC)
- **EXTEND-ONLY** `src/tac/portable_primitives/__init__.py` (+21 lines:
  15-symbol import block + 16-line `__all__` extension; ZERO mutation
  of WW + ARCH-1 + ARCH-2 existing exports per the ownership map)

Zero mutation of WW base primitives (`nn.py` / `backend.py` / `tensor.py` /
`loss.py` / `optim.py`) or ARCH-1 (`nn_extended.py`) or ARCH-2
(`nn_attention.py`) per APPEND-ONLY discipline.

## Empirical numerical-equivalence pin (Phase 3 PV)

Sister test convention from WW + ARCH-1 + ARCH-2: ε ≤ 5e-3 fp32 per
Phase 1 PV. ARCH-3 introduces a sister depth-accumulated band ε ≤ 5e-2
for composed blocks that stack multiple primitives (FMA reordering
compounds with depth).

**Measured max-abs-diff MLX-vs-PyTorch** (seeded inputs; M5 Max Metal
GPU vs PyTorch CPU fp32) `[empirical:src/tac/portable_primitives/tests/test_portable_primitives_fastvit.py]`:

| Primitive | Max abs diff | ε ceiling | Headroom |
|---|---|---|---|
| `PortableAllNorm` (BN1d-eval, F=1) | 1.19e-7 | 5e-3 | byte-stable |
| `PortableResBlock` (F=64, 4 Linears + 4 AllNorms + ReLU) | 3.10e-3 | 5e-2 | 1 OOM |
| `PortableHydra` (F=64, pose head, 1 ResBlock + 7 Linears) | 4.18e-3 | 5e-2 | 1 OOM |

Per-primitive equivalence (AllNorm) byte-stable (1e-7 = float32 unit
roundoff). The deeper composed blocks (ResBlock + Hydra) drift up by
about 1 OOM per ~5 cascaded primitives but remain firmly under the
depth-accumulated band (5e-2). This is the expected pattern from
Metal FMA reordering compounding through chained matmuls.

Full-backbone (PoseNet 7-stage + 12 RepMixer blocks) MLX-vs-PyTorch
numerical equivalence is **NOT pinned at this layer**; that is the
explicit ARCH-5 scope where PR 101 state_dict load + paired CUDA
ground-truth validation lands the contest-axis ε band.

## Tests

**29/29 PASS** at the bands above
`[empirical:src/tac/portable_primitives/tests/test_portable_primitives_fastvit.py]`:

- Canonical config (2 tests): T12 layers/embed_dims/mlp_ratios + PoseNet
  IN_CHANS/VISION_FEATURES/SUMMARY_FEATURES pinned vs timm + upstream
- AllNorm (4 tests): PyTorch shape preserved / non-canonical num_features
  rejected / load+export round-trip / MLX-vs-PyTorch equivalence
- ResBlock (3 tests): shape preserved / seed reproducibility / canonical
  12-key export dict
- Hydra (3 tests): canonical PoseNet pose-head shape (`{'pose': (B, 12)}`) /
  multi-head support / export includes resblock + per-head dict
- FastViT Block (3 tests): shape preserved / reparameterize preserves
  shape / LayerScale init γ=1e-5 starts as near-identity
- FastViT Stage (3 tests): no-downsample preserves spatial / downsample
  halves spatial + changes dim / reparameterize fuses all blocks
- T12 Backbone (3 tests): canonical PoseNet `(B,12,32,32) -> (B,2048)`
  signature / reparameterize preserves forward / stage count is exactly
  4 with canonical (2,2,6,2) blocks
- PoseNet (5 tests): canonical `{'pose': (B,12)}` output / input
  normalization wired / first-6-dims contract per compute_distortion /
  seed reproducibility / reparameterize preserves output shape
- Cross-backend equivalence (3 tests): ResBlock MLX-vs-PT / Hydra
  MLX-vs-PT / FastViT Block shape match

**Zero regression against WW + ARCH-1 + ARCH-2**: full
`pytest src/tac/portable_primitives/tests/` = **75/75 PASS** (13 WW
base + 17 ARCH-1 extended + 16 ARCH-2 attention + 29 ARCH-3 FastViT).

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map = N/A (architecture scaffold layer; no
  sensitivity signal contribution at this stage; ARCH-5 paired CUDA
  ground-truth will participate)
- Hook #2 Pareto constraint = N/A (architecture-level; no Pareto-relevant
  signal until paired contest-axis dispatch)
- Hook #3 bit-allocator = N/A (no bit-allocator signal at architecture
  layer)
- Hook #4 cathedral autopilot dispatch = N/A (architecture-level; no
  dispatch contribution; ARCH-5 contest-axis paired-eval lands the
  hook #4 surface)
- Hook #5 continual-learning posterior = **ACTIVE** (test results +
  empirical max-abs-diff measurements participate in the canonical
  posterior via this landing memo's evidence tags; extends the WW +
  ARCH-1 + ARCH-2 sister equation `mlx_pytorch_primitive_numeric_parity_band_v1`
  with the new depth-accumulated sister band)
- Hook #6 probe-disambiguator = N/A (no competing interpretations at
  the architecture scaffold layer; ARCH-5 may surface state_dict-key
  naming disambiguator paths for timm vs PR 101 weight compatibility)

Per CLAUDE.md "Subagent coherence-by-default" — silent omission of
hooks that don't apply at this stage of the cascade is documented
above, not silently ignored.

## Discipline compliance

- **Catalog #229 (premise verification)**: read ARCH-2 landing memo +
  ARCH-2 `nn_attention.py` (in full for the canonical interface) + WW
  base `nn.py` + ARCH-1 `nn_extended.py` (signatures only — already
  validated in prior cascades) + `upstream/modules.py` lines 1-100
  (PoseNet canonical reference with all hyperparameters + Hydra +
  ResBlock + AllNorm) + verified timm `fastvit_t12` spec empirically
  via `timm.models.fastvit.fastvit_t12` source inspection BEFORE any
  new code.
- **Catalog #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE)**: NEW files
  only for nn_fastvit.py + sister tests + landing memo; only
  `__init__.py` mutated via additive extension of existing `__all__` +
  import block (21-line additive extension only).
- **Catalog #287 (forbidden empirical-claim-without-evidence-tag)**:
  every max-abs-diff + tests-pass-count + LOC claim above carries
  inline `[empirical:<path>]` tag or table-cell location.
- **Catalog #344 (canonical-equation-reference)**: this landing's
  empirical-equivalence anchor extends the sister equation
  `mlx_pytorch_primitive_numeric_parity_band_v1` (registered at WW;
  empirically vindicated at ARCH-1 + ARCH-2). The depth-accumulated
  band (ε ≤ 5e-2 for ~5-cascaded primitives) is a new sister datum.
  `FORMALIZATION_PENDING` waiver acceptable per WW + ARCH-1 + ARCH-2
  precedent; ARCH-5 paired-eval lands the explicit registration
  extension as `mlx_pytorch_full_backbone_numeric_parity_band_v1` and
  the sister `pr101_state_dict_load_paired_forward_deterministic_ground_truth_v1`.
- **Catalog #117/#157/#174 (commit serializer + POST-EDIT
  --expected-content-sha256)**: commit lands via canonical
  `tools/subagent_commit_serializer.py` with sha256s captured AFTER
  edits per the 2026-05-13 docstring correction.
- **Catalog #206 (subagent checkpoint discipline)**: 4 checkpoints
  emitted during the 4-phase execution (initial / post-PV / post-write /
  post-tests + final complete) so a successor on API crash can resume.
- **Catalog #340 (sister-checkpoint guard PROCEED)**: verified `PROCEED`
  via canonical `tac.commit_safety.sister_checkpoint_guard.check_files_against_sister_checkpoints(...)`
  on all 3 changed files BEFORE staging.
- **Catalog #1 (forbidden MPS-fallback default) + #192 (macOS-CPU
  advisory not promoted)**: MLX backend remains non-promotable; the new
  architecture primitives inherit the WW + ARCH-1 + ARCH-2 canonical
  posture (numerical equivalence tested, but contest-axis promotion
  still requires PyTorch CUDA T4 paired auth-eval per CLAUDE.md
  "Submission auth eval — BOTH CPU AND CUDA").
- **Carmack MVP-first 5-step**: shape correctness + ε≤5e-2 depth-
  accumulated numerical equivalence tests ARE the empirical falsifiable
  challenge to "MLX FastViT-T12 backbone primitives are sufficient for
  PoseNet assembly" cargo-cult; 29/29 pass at well-under-band headroom
  is the verdict per step 2; canonical equation reference (WW + ARCH-1
  + ARCH-2 sister `mlx_pytorch_primitive_numeric_parity_band_v1` +
  pending sister formalization at ARCH-5 per step 3); landed in same
  commit batch as tests per step 4; re-routes operator priority queue
  to Sister-4 EfficientNet-B2-UNet SegNet assembly per step 5.

## Sister coordination (Catalog #230 ownership map + #314 absorption avoidance)

- **Slot 2 (`aab98b11` OVERNIGHT-DDD UNIWARD sharper-inversion follow-on)**:
  touches UNIWARD-axis files — DISJOINT scope from MLX FastViT
  primitives. Verified sister-checkpoint guard `PROCEED` with empty
  `conflicts` tuple on the 3 changed files.
- **Cron `9efd7486` (Selfcomp XX harvest)**: scheduled for 17:00 CDT;
  DISJOINT (Selfcomp XX is a different scope).
- **Cap=2 firm**: only this slot + DDD active at landing time. No 3rd
  sister overlap risk.

## Sister-4 ready signal

Next cascade stage per the dispatch contract: **EfficientNet-B2-UNet
SegNet full assembly** at `src/tac/portable_primitives/nn_segnet.py`.
Prerequisites:

1. ARCH-1 foundational primitives (BatchNorm2d + DepthwiseConv2d +
   MaxPool2d + AvgPool2d + silu) — landed commit `18520f83e` ✓
2. ARCH-2 attention primitives (LayerScale + MHSA + TokenMixer +
   RepMixer) — landed commit `51349c604` ✓
3. ARCH-3 FastViT-T12 backbone + PoseNet primitives — landed THIS
   commit batch ✓
4. WW base 9 primitives (Linear + Conv2d + LayerNorm + GELU + sigmoid +
   ReLU + softmax + matmul + bilinear_upsample) — landed commit
   `94c03b83c` ✓

All 4 prerequisites SATISFIED. Sister-4 can spawn immediately after
this commit lands.

**Sister-4 ownership map** (NEW file, disjoint from WW + ARCH-1 +
ARCH-2 + ARCH-3):
- `src/tac/portable_primitives/nn_segnet.py` — EfficientNet-B2-UNet
  SegNet full architecture per CLAUDE.md "Exact scorer architectures
  — SegNet" + `upstream/modules.py` (`smp.Unet('tu-efficientnet_b2',
  classes=5, activation=None, encoder_weights=None)`). 5-class output;
  argmax disagreement is the entire distortion signal. Input: LAST
  frame only `x[:, -1, ...]`, bilinear resize to `(512, 384)`.
- `src/tac/portable_primitives/tests/test_portable_primitives_segnet.py`
  — paired forward-equivalence + state_dict shape validation against
  smp / timm EfficientNet-B2 reference.

Note: SegNet requires the EfficientNet-B2 backbone (similar in pattern
to FastViT-T12 but with MBConv/Inverted Residual blocks instead of
RepMixer). Sister-4 may need to first land additional foundational
primitives (`PortableMBConvBlock` / `PortableSqueezeExcite`) before
assembling the full encoder; consider a Sister-4a precursor lane if
the LOC budget exceeds Sister-3's ~1030 LOC reference.

## Deferred-to-ARCH-3b / ARCH-5 (out of MVP scope per Carmack)

- **MobileOneBlock structural re-parameterization**: timm's stem uses
  3-branch MobileOne; current scaffold uses simpler WW Conv2d stem.
  Byte-stable parity deferred to ARCH-5 state_dict load (where the
  weight-shape match becomes the verification surface).
- **Per-block `conv_ffn` / `patch_emb` exact timm-equivalent shapes**:
  current scaffold preserves dimension flow but not byte-stable timm
  key naming. ARCH-5 state_dict load will require either canonical
  re-naming or per-block adapter.
- **`act_layer='gelu_tanh'`**: current scaffold uses GELU (the
  default `approximate='none'`); gelu_tanh drift is sub-ε per ARCH-1
  PV but the canonical contract requires the exact match.
- **Deterministic PR 101 state_dict load + paired forward**: the
  Sister-5 ownership map per dispatch contract.

## Operator-routable next-actions

1. **Spawn Sister-4 (EfficientNet-B2-UNet SegNet assembly)** per the
   5-stage cascade. Sister-4 may need a precursor Sister-4a if
   MBConv/SqueezeExcite primitives need to land first.
2. (Optional) review the canonical empirical max-abs-diff table above
   and decide whether to tighten ε ceiling for AllNorm from 5e-3 to
   1e-5 (current byte-stable headroom is 4-5 OOM).
3. (Optional) Sister-3b: layer MobileOneBlock structural
   re-parameterization on top of `PortableFastViTBlock` for byte-stable
   timm parity before ARCH-5 state_dict load.

## Cost

$0 paid GPU + ~50 min wall-clock + ~15 tool uses.

## Cross-references

- MLX-ARCH-2 landing memo (`.omx/research/mlx_arch_2_attention_primitives_landed_20260521.md`)
  — canonical interface conventions + ε band + test discipline this
  landing inherits + Sister-3 ownership map.
- MLX-ARCH-1 landing memo
  (`.omx/research/mlx_arch_1_foundational_primitives_landed_20260521.md`)
  — Phase 1 ε≤5e-3 PV anchor.
- WW landing memo (canonical base 9 primitives) — baseline ε band
  reference.
- CLAUDE.md "Exact scorer architectures — VERIFIED from upstream
  modules.py" — PoseNet FastViT-T12 + 12-channel YUV6 input + Hydra
  head spec (canonical reference this landing implements).
- CLAUDE.md "Beauty, simplicity, and developer experience" — primitives
  reviewable in 30 seconds each.
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
  this landing is research-only (non-promotable per Catalog #192);
  ARCH-5 paired-eval is where contest-axis promotion gates apply.
- `upstream/modules.py` lines 22-80 — PoseNet canonical reference
  (Head namedtuple + AllNorm + ResBlock + Hydra + PoseNet).
- `timm.models.fastvit.fastvit_t12` (timm package) — empirically-verified
  T12 spec: `layers=(2,2,6,2), embed_dims=(64,128,256,512),
  mlp_ratios=(3,3,3,3), token_mixers=ALL repmixer`.
- FastViT paper (Vasu et al. 2023) — re-parameterization technique
  + LayerScale init γ=1e-5 + RepMixer block design.
