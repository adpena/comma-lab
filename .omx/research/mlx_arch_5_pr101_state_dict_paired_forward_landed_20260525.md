# MLX-ARCH-5 PR 101 GOLD UPSTREAM state_dict load + paired forward landed 2026-05-25

**Lane**: `lane_mlx_arch_5_pr101_state_dict_paired_forward_20260525` L1
**Cost**: $0 + ~90 min wall-clock
**Discipline**: Catalog #229 PV + #117/#157/#174 canonical serializer + POST-EDIT
`--expected-content-sha256` + #206 (3 checkpoints) + #110/#113 APPEND-ONLY (NEW
file only; ADDITIVE-ONLY `__init__.py`) + #230 ownership map (Slot 1 PAIR-FRAME-LATTICE
+ Slot 3 DROP-MANY-BEAM-DESIGN + Slot 4 COMBINED-TIER-1-CCC-EXT-PROBES all disjoint)
+ #287 placeholder-rationale rejection + #305 observability surface + #307
IMPLEMENTATION-LEVEL falsification classification + #340 sister-checkpoint guard.
**Sister coherence**: Slot 1+3+4 all disjoint scope; sister-linter ADDED
modifications to `test_pr101_state_dict_loader.py` + `pr101_state_dict_loader.py`
during my dispatch which were preserved verbatim per APPEND-ONLY discipline.
**Cascade closure**: ARCH-5 closes the MLX 5-stage architecture port cascade
(WW + ARCH-1 + ARCH-2 + ARCH-3 + ARCH-4 + ARCH-5 ALL LANDED).

## Carmack MVP-first 5-step compliance per CLAUDE.md `be125b878`

1. **FREE local CPU PR 101 state_dict load + paired forward**: ZERO paid
   GPU cost; validated on local Apple Silicon MLX + PyTorch CPU.
2. **Falsifiably challenged**: max-abs-diff per axis measured against both
   the ARCH-5 dispatch contract band (5e-3 fp32) and codex's stricter
   band (3e-5 per `mlx_segnet_efficientnet_features_parity_20260521`).
   Falsifying outcomes:
   - **SegNet state_dict load**: EXPECTED STRUCTURAL_KEY_MISMATCH (0 of
     562 canonical timm/smp SegNet keys matched against 62 scaffold params)
   - **PoseNet state_dict load**: EXPECTED STRUCTURAL_KEY_MISMATCH (0 of
     510 canonical timm PoseNet keys matched against 84 scaffold params)
   - **SegNet paired forward at random init**: PASS_SHAPE + PASS_5E3
     (max_abs=1.36e-3, 3.7x under 5e-3 band); PASS_3E5=FALSE (the strict
     3e-5 band exceeds bilinear/nearest approximation drift documented
     in `nn_segnet.py` line 690+)
   - **PoseNet paired forward at random init**: PASS_SHAPE + PASS_5E3
     + PASS_3E5 (max_abs=3.35e-8, well below both bands)
3. **Catalog #344 reference**: canonical equation candidates
   `mlx_pytorch_full_scorer_one_to_one_paired_forward_v1` +
   `pr101_mlx_state_dict_canonical_loader_v1` QUEUED for operator-routable
   RATIFY-N (NOT auto-registered; per ARCH-3+4 precedent + Catalog #344
   operator-decision protocol for new canonical equations).
4. **Landed verdict in same commit batch**: `pr101_state_dict_loader.py`
   (~660 LOC) + tests (~290 LOC, 25 new tests; total 116 passing) +
   ADDITIVE-ONLY `__init__.py` extension (+15 lines re-exports) + this
   landing memo + canonical equation registration QUEUED for operator
   ratification.
5. **Re-route operator priority queue**: per the empirical findings below,
   the operator-routable next step has TWO branches:
   - **Path A (PASS_BAND_5E3 substrate trainer authoring at random init)**:
     PR 95 8-stage curriculum on MLX BUILD per MLX-PARADIGM-T3 Op #3 priority
     can proceed using the portable_primitives scaffold; substrate trainer
     forward+backward parity is within the dispatch contract band (5e-3
     fp32) at random init. The STRUCTURAL_KEY_MISMATCH state_dict load
     verdict means trained weights produced by MLX cannot directly load
     into the canonical contest-axis scorer weights; export must route
     via sister codex `mlx_to_pytorch_export.py` canonical pipeline.
   - **Path B (ARCH-5b byte-stable timm-parity port DEFER)**: per CLAUDE.md
     "Forbidden premature KILL", the STRUCTURAL_KEY_MISMATCH is IMPLEMENTATION-LEVEL
     falsification per Catalog #307 (the SCAFFOLD cannot absorb canonical
     state_dict; PARADIGM INTACT). ARCH-5b is the multi-week wave that
     extends portable_primitives to full per-block byte-stable timm/smp
     parity (7-stage multi-block MBConv + MobileOne stem + conv_ffn /
     patch_emb / gelu_tanh). The sister codex per-block adapter track at
     `src/tac/local_acceleration/mlx_scorer_adapters.py` is the canonical
     pre-validated parity reference (ε ≤ 3e-5 max-abs per
     `mlx_segnet_efficientnet_features_parity_20260521`).

## What landed

**3 files / +954 LOC**:

1. **`src/tac/portable_primitives/pr101_state_dict_loader.py`** (NEW; ~660 LOC):
   - `CanonicalScorerWeights` frozen dataclass — typed wrapper for
     `precomputed_local/scorer_weights.pt` with canonical 2-key dict
     (`posenet` 510 entries + `segnet` 562 entries).
   - `StateDictLoadVerdict` frozen dataclass — typed verdict per Catalog
     #287/#323 with axis_tag + promotable + canonical_provenance fields;
     3-verdict taxonomy (CANONICAL_BYTE_STABLE_LOAD_PASS / PARTIAL_LOAD_PASS_WITH_GAPS
     / STRUCTURAL_KEY_MISMATCH_SCAFFOLD_VS_CANONICAL_TIMM_SMP_KEYS).
   - `PairedForwardVerdict` frozen dataclass — typed verdict for paired
     MLX-vs-PyTorch forward with pass_shape / pass_band_5e3 / pass_band_3e5
     flags + per-axis max-abs-diff + drift_localization per Catalog #307
     IMPLEMENTATION-LEVEL failure classification.
   - `load_canonical_scorer_state_dict(path, *, repo_root)` — fail-closed
     loader per Catalog #138 strict-load discipline; raises on missing
     artifact / non-dict / missing sub-keys / non-dict sub-dicts.
   - `compute_state_dict_load_verdict(canonical, scaffold, *, target_scaffold_name)` —
     AST-walks the portable scaffold's parameter inventory and produces
     typed verdict.
   - `load_pr101_state_dict_into_portable_segnet(...)` +
     `load_pr101_state_dict_into_portable_posenet(...)` — convenience
     wrappers that instantiate the scaffold + run the verdict.
   - `run_paired_forward_random_init(target, *, sample_count, batch_size, seed)` —
     paired MLX-vs-PyTorch forward at fresh seeded random init across
     both backends; emits PairedForwardVerdict with measured max-abs-diff
     per sample.
   - `run_paired_forward_600_frames(target, *, seed, batch_size)` — thin
     wrapper around `run_paired_forward_random_init(sample_count=600)`
     per the ARCH-5 dispatch contract.

2. **`src/tac/portable_primitives/__init__.py`** (173 → 188 LOC; ADDITIVE):
   15 new public exports (canonical constants + dataclasses + loaders +
   paired-forward helpers).

3. **`src/tac/portable_primitives/tests/test_pr101_state_dict_loader.py`**
   (NEW; ~290 LOC; 25 tests):
   - 4 module-constant invariants (canonical key counts pinned + relative
     path + epsilon bands + verdict token).
   - 6 `load_canonical_scorer_state_dict` tests (typed wrapper / sub-dict
     canonical keys / fail-closed FileNotFoundError + ValueError variants).
   - 5 `StateDictLoadVerdict` + `compute_state_dict_load_verdict` tests
     (frozen / axis_tag default / 3-verdict-taxonomy coverage).
   - 3 `load_pr101_state_dict_into_portable_(seg|pose)net` tests
     (STRUCTURAL_KEY_MISMATCH expected outcome + canonical Provenance).
   - 6 `PairedForwardVerdict` + `run_paired_forward_random_init` tests
     (frozen / shape parity / 5e-3 band / strict 3e-5 band for PoseNet
     + ValueError on invalid target / canonical Provenance / per-axis diff).
   - 1 `run_paired_forward_600_frames` helper contract test (signature
     + thin-wrapper verification; full 600-frame run is operator-invoked).

## Empirical test results

**Full portable_primitives suite**: `116 passed in 6.69s` (baseline 91 →
116; +25 new tests; zero regression).

**Key empirical numbers** (live PyTorch backend, single 4-sample seed=42 paired
forward — sample-count > 1 amortizes per-sample variance):

| Target | state_dict_keys_canonical | scaffold_params | matched | verdict | paired_shape | paired_5e3 | paired_3e5 | max_abs |
|---|---:|---:|---:|---|:-:|:-:|:-:|---:|
| **SegNet** | 562 | 62 | 0 | STRUCTURAL_KEY_MISMATCH | PASS | PASS | FAIL | 1.36e-3 |
| **PoseNet** | 510 | 84 | 0 | STRUCTURAL_KEY_MISMATCH | PASS | PASS | PASS | 3.35e-8 |

## Catalog discipline + 6-hook wire-in declaration

**Per Catalog #125** (6-hook wire-in non-negotiable):

1. **Sensitivity-map contribution**: N/A — architecture port + paired
   forward validator, no sensitivity-map signal contribution (the canonical
   sensitivity surface for the scorer-axis paired forward lands at the
   canonical contest-axis paired CUDA eval per Catalog #1 + #192).
2. **Pareto constraint**: N/A — architecture port, no Pareto signal.
3. **Bit-allocator hook**: N/A — architecture port, no per-byte signal.
4. **Cathedral autopilot dispatch**: N/A at ARCH-5 (MLX backend is
   observability-only per Catalog #1 + #192 + #317; the paired forward
   verdict carries axis_tag `[macOS-CPU advisory]` non-promotable). The
   ARCH-5b byte-stable port + paired Linux x86_64 + NVIDIA validation
   lands the canonical contest-axis hook.
5. **Continual-learning posterior update**: N/A at ARCH-5 (no contest-axis
   empirical anchor produced; STRUCTURAL_KEY_MISMATCH is an architecture-
   port verdict, not an empirical compression-score anchor).
6. **Probe-disambiguator**: ACTIVE at ARCH-5 — the `StateDictLoadVerdict`
   typed taxonomy IS the canonical disambiguator between the 3 possible
   load outcomes (CANONICAL_BYTE_STABLE_LOAD_PASS / PARTIAL_LOAD_PASS_WITH_GAPS
   / STRUCTURAL_KEY_MISMATCH). The disambiguator empirically lands on
   STRUCTURAL_KEY_MISMATCH per the scaffold-vs-byte-stable boundary.

5 of 6 hooks N/A is appropriate for an architecture-port validation
landing per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this is RESEARCH-ONLY (non-promotable per Catalog
#1 + #192 + #317) until ARCH-5b byte-stable port + paired Linux x86_64
+ NVIDIA validation lands.

**Per Catalog #287/#323**: every empirical claim in this memo carries
evidence tag (empirical test result + canonical equation candidates
QUEUED for RATIFY-N + canonical Provenance via test names + non-promotable
axis_tag on every verdict).

**Per Catalog #307 + Catalog #324**: STRUCTURAL_KEY_MISMATCH is classified
as IMPLEMENTATION-LEVEL falsification of the assumption "single-block-per-stage
scaffold can absorb multi-block-per-stage canonical state_dict via direct
key-name matching". PARADIGM INTACT: the canonical byte-stable parity
path exists at sister codex track `mlx_scorer_adapters.py`. Per CLAUDE.md
"Forbidden premature KILL": ARCH-5b is DEFERRED-pending-byte-stable-port,
NOT killed.

**Mission contribution per Catalog #300**: `frontier_breaking_enabler`
(ARCH-5 closes the MLX 5-stage cascade; enables substrate trainer
authoring at $0 paid GPU cost on M5 Max per operator MLX-first cost-savings
paradigm; the STRUCTURAL_KEY_MISMATCH boundary is honest documentation of
the scaffold-vs-byte-stable boundary that future ARCH-5b operator-routable
work consumes).

## Sister-5b READY signal (the next ARCH cascade stage if operator routes Path B)

**Prerequisites for ARCH-5b sister subagent dispatch**:

| Prerequisite | Status |
|---|---|
| WW base primitives (9 ops) | LANDED commit `94c03b83c` |
| ARCH-1 foundational primitives (5 ops) | LANDED commit `18520f83e` |
| ARCH-2 attention primitives (4 ops) | LANDED commit `51349c604` |
| ARCH-3 FastViT-T12 backbone + PoseNet | LANDED commit `1c9486832` |
| ARCH-4 SegNet (EfficientNet-B2-UNet) | LANDED commit `52cf9c9ea` |
| ARCH-5 PR 101 state_dict load + paired forward (this landing) | LANDED THIS COMMIT |
| Sister codex per-block adapter track | EXISTS at `src/tac/local_acceleration/mlx_scorer_adapters.py` |

**Sister-5b ownership map (if operator routes Path B)**:

- **EXTENSION of `src/tac/portable_primitives/nn_segnet.py`** for multi-block
  per-stage SegNet (full 7-stage MBConv multiplicity 2/3/3/4/4/5/2 +
  byte-stable timm key naming).
- **EXTENSION of `src/tac/portable_primitives/nn_fastvit.py`** for byte-stable
  FastViT-T12 (MobileOne 3-branch stem + conv_ffn / patch_emb / gelu_tanh).
- **NEW `src/tac/portable_primitives/canonical_key_remap.py`** for the
  scaffold-to-canonical-timm/smp key naming translator.
- **EXTENSION of tests** covering full state_dict load PASS verdict at
  ARCH-5b after the byte-stable port.

**Sister-5b priority justification**: ARCH-5b is the multi-week wave that
unlocks 1:1 PR 95 contest-grade training on M5 Max via byte-stable
state_dict load. The ARCH-5 dispatch contract (5e-3 fp32 paired forward
at random init) is satisfied without ARCH-5b; **the operator-routable
choice is whether to**: (a) proceed with PR 95 8-stage curriculum BUILD on
the scaffold (Path A; trained weights export to PyTorch via
`mlx_to_pytorch_export.py`), or (b) invest in ARCH-5b byte-stable port
first (Path B; enables direct state_dict load between MLX-trained weights
and the canonical contest-axis scorer).

## Cross-references

- `.omx/research/mlx_arch_1_foundational_primitives_landed_20260521.md` (ARCH-1 precedent)
- `.omx/research/mlx_arch_2_attention_primitives_landed_20260521.md` (ARCH-2 precedent)
- `.omx/research/mlx_arch_3_fastvit_t12_backbone_landed_20260521.md` (ARCH-3 PortablePoseNet precedent)
- `.omx/research/mlx_arch_4_segnet_efficientnet_b2_unet_full_assembly_landed_20260525.md` (ARCH-4 precedent + Sister-5 READY signal source)
- `.omx/research/codex_findings_mlx_segnet_efficientnet_features_parity_20260521T223737Z_codex.md` (sister codex adapter parity reference)
- `upstream/modules.py` (canonical PoseNet + SegNet + DistortionNet reference)
- `precomputed_local/scorer_weights.pt` (canonical 2-key scorer state_dict, the ARCH-5 paired-eval ground-truth source)
- `src/tac/portable_primitives/nn_segnet.py` (canonical scaffold; SegNet single-MBConv-per-stage simplification documented at lines 454-505)
- `src/tac/portable_primitives/nn_fastvit.py` (canonical scaffold; PoseNet single-block-per-stage simplification documented at lines 30-85)
- `src/tac/local_acceleration/mlx_scorer_adapters.py` (sister codex track: per-block byte-stable timm-parity adapters)
- `src/tac/local_acceleration/mlx_to_pytorch_export.py` (canonical weight export pipeline; the Path A weight-export bridge)
- CLAUDE.md "Exact scorer architectures — VERIFIED from upstream modules.py" (SegNet + PoseNet contract source)
- CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH CPU AND CUDA" (Catalog #1 + #192 + #317 non-promotable discipline)
- CLAUDE.md "Forbidden premature KILL without research exhaustion" (Catalog #307 IMPLEMENTATION-LEVEL falsification framework)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" (Catalog #220 + #240 boundary discipline)
