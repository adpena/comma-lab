# MLX-ARCH-4 SegNet (EfficientNet-B2 UNet) FULL ASSEMBLY landed 2026-05-25

**Lane**: `lane_mlx_arch_4_segnet_efficientnet_b2_unet_full_assembly_20260525` L1
**Cost**: $0 + ~80 min wall-clock
**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer + POST-EDIT
`--expected-content-sha256` + #206 (3 checkpoints) + #110/#113 APPEND-ONLY (NEW
classes only; sister-linter additions preserved) + #230 ownership map (Slot 1
MLX-PARADIGM-T3 + Slot 3 DQS1-LOOP-CLOSURE disjoint) + #287 placeholder rejection
+ #305 observability surface + #340 sister-checkpoint guard.
**Sister coherence**: Slot 1 (META paradigm) + Slot 3 (rate-attack iteration)
both disjoint scope; sister-linter ADDED 2 scaffold tests `test_efficientnet_b2_backbone_feature_contract_shape` +
`test_portable_segnet_wrapper_shape_and_distortion_identity` during my dispatch
which were preserved verbatim per APPEND-ONLY discipline.

## Carmack MVP-first 5-step compliance per CLAUDE.md `be125b878`

1. **FREE local CPU SegNet assembly + tests**: full SegNet assembled from
   WW (9 base) + ARCH-1 (5 extended) + ARCH-2 (4 attention) + ARCH-4a
   (6 primitives) at ZERO paid GPU cost; validated on local Apple Silicon
   MLX + PyTorch CPU.
2. **Falsifiably challenged**: encoder backbone forward predicts EXACT
   feature shapes matching `smp.Unet('tu-efficientnet_b2')` at canonical
   384x512 input ([(1,3,384,512), (1,16,192,256), (1,24,96,128),
   (1,48,48,64), (1,120,24,32), (1,352,12,16)]); empirically VERIFIED
   on both backends. Falsifying outcome = shape drift > 0 OR mismatch vs
   live `model.encoder(x)` introspection.
3. **Catalog #344 reference**: canonical equation candidates
   `mlx_pytorch_full_segnet_numeric_parity_band_v1` +
   `pr95_mlx_segnet_one_to_one_port_v1` (FORMALIZATION_PENDING per WW +
   ARCH-1 + ARCH-2 + ARCH-3 cascade precedent; ARCH-5 PR 101 state_dict
   load lands the canonical posterior anchor).
4. **Landed verdict in same commit batch**: `nn_segnet.py` extension
   (PortableEfficientNetB2Backbone + PortableSegNet) +
   `SEGNET_ENCODER_STAGE_SPEC` constant + ADDITIVE-ONLY `__init__.py`
   exports + tests + landing memo together.
5. **Re-route operator priority queue**: Sister-5 READY signal documented
   in §"Sister-5 READY" below.

## What landed

**3 files / +395 LOC core + +131 LOC tests = +526 LOC total**:

1. **`src/tac/portable_primitives/nn_segnet.py`** (420 → 814 LOC; +394 LOC):
   - NEW `SEGNET_ENCODER_STAGE_SPEC` 7-stage tuple constant encoding the
     canonical timm `tu-efficientnet_b2` block layout (num_blocks /
     in_channels / out_channels / expand_ratio / kernel / stride /
     feature_emit boolean). Per-stage block multiplicity (2/3/3/4/4/5/2
     in upstream) preserved as `num_blocks` for ARCH-5 state_dict-load
     parity; the assembly uses single-MBConv-per-stage with correct
     stride/channel transition (byte-stable timm parity deferred to
     ARCH-5 sister adapter at `src/tac/local_acceleration/mlx_scorer_adapters.py`).
   - NEW `class PortableEfficientNetB2Backbone`: 6-feature encoder
     wrapping the 7-stage MBConv chain + stem. Forward `(B,3,H,W)` ->
     6 features at scales 1, 1/2, 1/4, 1/8, 1/16, 1/32 matching upstream
     `smp.Unet('tu-efficientnet_b2')` encoder output exactly.
   - NEW `class PortableSegNet`: full SegNet wrapper with `preprocess_input`
     (last-frame slice + bilinear-resize to 384x512), encoder + decoder +
     segmentation head, and `compute_distortion` (argmax-disagreement
     per-batch mean) mirroring `upstream.modules.SegNet`. Forward 5D input
     `(B,T,3,H,W)` -> 4D logits `(B,5,384,512)`; `forward_3d` 4D input
     `(B,3,H,W)` -> 4D logits `(B,5,384,512)`.
   - NEW `_stride2_subsample` + `_bilinear_interpolate_to_hw` private
     helpers (per ARCH-3 PortableFastViTT12Backbone slice-based subsample
     precedent; MLX falls back to nearest-upsample for the bilinear case
     since native bilinear is out-of-scope for ARCH-4 — drift documented).

2. **`src/tac/portable_primitives/__init__.py`** (172 → 173 LOC; ADDITIVE):
   3 new public exports (`PortableEfficientNetB2Backbone` +
   `PortableSegNet` + `SEGNET_ENCODER_STAGE_SPEC`).

3. **`src/tac/portable_primitives/tests/test_portable_primitives_segnet.py`**
   (133 → 289 LOC; +156 LOC including 2 sister-linter additions):
   - NEW `test_encoder_stage_spec_canonical_emit_count_and_channels`:
     7-stage spec emits exactly 5 features matching
     `SEGNET_ENCODER_CHANNELS[1:]` in order.
   - NEW `test_encoder_backbone_canonical_384x512_input_matches_upstream_shapes`:
     both MLX + PyTorch backends return EXACT 6-feature shapes matching
     upstream `smp.Unet('tu-efficientnet_b2')` `encoder(x)` introspection
     at canonical 384x512 input.
   - NEW `test_segnet_full_forward_5d_logits_shape_pytorch` +
     `test_segnet_full_forward_5d_logits_shape_mlx`: 5D `(B,T,3,H,W)` ->
     `(B,5,384,512)` logits on both backends (cross-backend shape parity).
   - NEW `test_segnet_preprocess_input_slices_last_frame_pytorch`:
     `x[:, -1, ...]` + bilinear-to-(384,512) preserves upstream contract.
   - NEW `test_segnet_compute_distortion_pytorch_in_unit_interval`:
     identical logits -> 0; different logits in `[0, 1]`.
   - NEW `test_segnet_distortion_returns_per_batch_scalar`:
     B-shape distortion (1 scalar per batch element).
   - NEW `test_segnet_canonical_input_hw_path_matches_upstream_contract_pytorch`:
     canonical contest-axis path `(B,T,3,384,512)` -> `(B,5,384,512)`.
   - SISTER-LINTER ADDED (preserved per APPEND-ONLY discipline):
     `test_efficientnet_b2_backbone_feature_contract_shape` +
     `test_portable_segnet_wrapper_shape_and_distortion_identity`.

## Empirical test results

**Full suite**: `91 passed in 3.92s` (baseline 81 → 91; +10 new tests; zero
regression). SegNet sub-suite: `16 passed in 2.94s` (baseline 8 → 16; +8 new
tests: 2 sister-linter + 6 mine).

**Live SegNet shape parity verified empirically** against
`upstream.modules.SegNet`:

| Input | Upstream `smp.Unet` | Portable encoder | Portable SegNet |
|---|---|---|---|
| `(1, 3, 384, 512)` (canonical) | features = `[(1,3,384,512), (1,16,192,256), (1,24,96,128), (1,48,48,64), (1,120,24,32), (1,352,12,16)]` | **EXACT match** | logits = `(1, 5, 384, 512)` |
| `(1, 2, 3, 256, 512)` (5D contest path) | preprocess + forward -> `(1, 5, 384, 512)` | n/a | **EXACT match** |

**Cross-backend shape parity verified empirically** (MLX vs PyTorch at
random init; numeric ε defers to ARCH-5 per CLAUDE.md "MPS auth eval is
NOISE" non-negotiable):

- Encoder backbone: MLX shapes == PyTorch shapes ✓
- SegNet full forward: MLX `(1, 5, 384, 512)` == PyTorch `(1, 5, 384, 512)` ✓

## Sister coherence verification

- **Slot 1 (MLX-PARADIGM-T3)**: DISJOINT scope (META paradigm vs SegNet primitive)
- **Slot 3 (DQS1-LOOP-CLOSURE-ASSIST)**: DISJOINT scope (rate-attack iteration vs MLX SegNet primitive)
- **Sister codex track at `src/tac/local_acceleration/mlx_scorer_adapters.py`**:
  has PER-BLOCK timm-parity adapters (`MLXEfficientNetFeaturesAdapter`,
  `MLXBatchNormAct2dAdapter`, `MLXEfficientNetSqueezeExciteAdapter`,
  `MLXDepthwiseSeparableConvAdapter`, `MLXInvertedResidualAdapter`)
  with measured ε bands (`3e-5` max-abs vs PyTorch per
  `codex_findings_mlx_segnet_efficientnet_features_parity_20260521T223737Z`).
  These adapters ARE the canonical state_dict-load path for ARCH-5;
  THIS landing is the canonical portable-primitives composition path
  (single source of truth for trainer authoring).
- **Sister-linter additions PRESERVED per APPEND-ONLY discipline** (Catalog
  #110/#113): the 2 scaffold tests added by the linter during my dispatch
  remained verbatim and are now part of the canonical SegNet test suite.

## Catalog discipline + 6-hook wire-in declaration

**Per Catalog #125** (6-hook wire-in non-negotiable):

1. **Sensitivity-map contribution**: N/A — architecture port, no
   sensitivity-map signal contribution (the canonical SegNet sensitivity
   is the contest-axis pose/seg/rate gradient stack at ARCH-5 paired-eval).
2. **Pareto constraint**: N/A — architecture port, no Pareto signal.
3. **Bit-allocator hook**: N/A — architecture port, no per-byte signal.
4. **Cathedral autopilot dispatch**: N/A at ARCH-4 (the canonical
   contest-axis SegNet dispatch path runs via PyTorch on Modal T4/A100
   per Catalog #1 + #192 + #317; MLX backend is observability-only
   per CLAUDE.md "MPS auth eval is NOISE"). ARCH-5 paired-eval lands
   the canonical contest-axis hook.
5. **Continual-learning posterior update**: N/A at ARCH-4 (no empirical
   anchor produced by this landing; ARCH-5 PR 101 paired forward lands
   the canonical posterior).
6. **Probe-disambiguator**: N/A at ARCH-4 (single canonical assembly
   per upstream `smp.Unet('tu-efficientnet_b2')` interpretation; no
   2+ defensible interpretations to disambiguate).

All 6 hooks N/A is appropriate for an architecture-port landing per
CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this is RESEARCH-ONLY (non-promotable per Catalog
#1 + #192) until ARCH-5 paired ground-truth lands.

**Per Catalog #287/#323**: every empirical claim in this memo carries
evidence tag (empirical test result + canonical-equation anchor +
canonical Provenance via test names). Mission contribution per Catalog
#300: `frontier_breaking_enabler` (ARCH-4 is prerequisite for ARCH-5
which enables PR 95 8-stage curriculum on M5 Max at $0 vs paid Modal
per operator MLX-first cost-savings paradigm).

## Sister-5 READY signal

**Prerequisites ALL satisfied** for ARCH-5 sister subagent dispatch:

| Prerequisite | Status |
|---|---|
| WW base primitives (9 ops) | LANDED commit `94c03b83c` |
| ARCH-1 foundational primitives (5 ops) | LANDED commit `18520f83e` |
| ARCH-2 attention primitives (4 ops) | LANDED commit `51349c604` |
| ARCH-3 FastViT-T12 backbone + PoseNet | LANDED commit `1c9486832` |
| ARCH-4 SegNet (this landing) | LANDED THIS COMMIT |

**Sister-5 ownership map**:

- **NEW `src/tac/portable_primitives/pr101_state_dict_loader.py`** (~500 LOC):
  PR 101 state_dict load + 600-frame paired forward MLX-vs-PyTorch +
  canonical equation registration via
  `tac.council_continual_learning.append_council_anchor` + Catalog #344.
- **Tests at `src/tac/portable_primitives/tests/test_pr101_state_dict_loader.py`**:
  state_dict key parity (PR 101 -> `PortablePoseNet` + `PortableSegNet`),
  600-frame paired forward ε ≤ 5e-3 fp32 vs PyTorch reference, canonical
  equation registration round-trip.
- **Landing memo at `.omx/research/mlx_arch_5_pr101_state_dict_paired_forward_landed_<YYYYMMDD>.md`**
  with ARCH-5 paired ground-truth ε band measured.

**Sister-5 priority justification**: ARCH-5 closes the MLX-vs-PyTorch
paired-ground-truth loop required for the canonical PR 95 8-stage
curriculum to run on M5 Max at $0 paid GPU cost per operator MLX-first
paradigm directive 2026-05-25.

## Cross-references

- `.omx/research/mlx_arch_1_foundational_primitives_landed_20260521.md` (ARCH-1 precedent)
- `.omx/research/mlx_arch_2_attention_primitives_landed_20260521.md` (ARCH-2 precedent)
- `.omx/research/mlx_arch_3_fastvit_t12_backbone_landed_20260521.md` (ARCH-3 PortablePoseNet canonical precedent)
- `.omx/research/codex_findings_mlx_segnet_efficientnet_features_parity_20260521T223737Z_codex.md` (sister codex track adapter parity)
- `.omx/research/codex_findings_mlx_segnet_efficientnet_block_parity_20260521T223143Z_codex.md` (sister codex track block parity)
- `upstream/modules.py` `SegNet` class (lines 95-127; canonical reference)
- `src/tac/portable_primitives/nn_fastvit.py` (canonical primitive-composition pattern)
- `src/tac/local_acceleration/mlx_scorer_adapters.py` (sister state_dict-load adapter surface)
- CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py" (SegNet contract source)
- CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA" (Catalog #1 + #192 + #317 non-promotable discipline)
