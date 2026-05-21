# MLX-ARCH-1 Foundational Primitives Landed (BatchNorm2d + DepthwiseConv2d + MaxPool2d + AvgPool2d + silu)

Date: 2026-05-21
Lane: `lane_mlx_arch_1_foundational_primitives_batchnorm_depthwiseconv_maxpool_avgpool_silu_20260521`
Sister-1 of: 5-stage MLX architecture port cascade
Predecessor anchor: OVERNIGHT-ZZ commit `330a58a60` op-routable #1
Operator directive: 2026-05-21 *"Full contest scorer required"*
Carmack MVP-first 5-step per CLAUDE.md `be125b878`.

## Scope (what landed)

5 new portable-primitive surfaces extending OVERNIGHT-WW's 9 base ops, sister
MLX + PyTorch implementations, contract-faithful to WW's `Primitive(...,
backend=...)` interface convention:

1. **`PortableBatchNorm2d`** — 2D BatchNorm with running stats; eval-mode
   per CLAUDE.md "QAT pipeline" + PR101 frozen-BN-at-inference contract.
   `weight` + `bias` + `running_mean` + `running_var` buffers (all shape
   `(num_features,)`); `train()` / `eval()` togglable; canonical
   `load_weights(...)` / `export_weights() -> dict` for sister Wave 4
   weight-export pipeline.

2. **`PortableDepthwiseConv2d`** — Conv2d `groups=in_channels` variant per
   FastViT RepMixer + EfficientNet MBConv depthwise stages. PyTorch
   weight layout `(in_channels, 1, kH, kW)` canonical for export;
   internally re-laid to MLX `(out_channels, kH, kW, in_channels_per_group)`
   NHWC at forward time, mirroring `PortableConv2d` convention.

3. **`PortableMaxPool2d`** — 2D max pooling. Defaults `stride=None` per
   PyTorch convention (stride=kernel_size). FastViT stem pattern
   (`kernel=3, stride=2, padding=1`) validated.

4. **`PortableAvgPool2d`** — 2D average pooling. Supports global pooling
   per EfficientNet squeeze-excite + final global-pool by setting
   kernel to match spatial dims. Spatial + same-size smoothing patterns
   validated.

5. **`silu`** — Swish/SiLU activation `x * sigmoid(x)` per EfficientNet
   (Tan & Le 2019). Sister MLX `mlx.nn.silu` + PyTorch `F.silu`.

LOC: 515 (`nn_extended.py`) + 426 (`tests/test_portable_primitives_extended.py`)
= **941 LOC** across 3 changed files. Well inside the 1500-LOC / 5-file scope
limit per the dispatch contract.

## Files touched (APPEND-ONLY per Catalog #110/#113)

- **NEW** `src/tac/portable_primitives/nn_extended.py` (515 LOC)
- **NEW** `src/tac/portable_primitives/tests/test_portable_primitives_extended.py` (426 LOC)
- **EXTEND-ONLY** `src/tac/portable_primitives/__init__.py` (+8 lines re-exporting
  the 5 new symbols in `__all__`; ZERO mutation of WW's existing exports)

Zero mutation of WW base primitives (`nn.py` / `backend.py` / `tensor.py` /
`loss.py` / `optim.py`) per APPEND-ONLY discipline.

## Empirical numerical-equivalence pin (Phase 3 PV)

Sister test convention from
`test_portable_primitives_numerical_equivalence.py`: ε ≤ 5e-3 fp32 per
Phase 1 PV (Metal FMA reordering).

**Measured max-abs-diff MLX-vs-PyTorch** (seeded inputs; M5 Max Metal GPU
vs PyTorch CPU fp32) `[empirical:src/tac/portable_primitives/tests/test_portable_primitives_extended.py]`:

| Primitive | Max abs diff | ε ceiling | Headroom |
|---|---|---|---|
| `PortableBatchNorm2d` (eval) | 4.77e-7 | 5e-3 | 4 orders of magnitude |
| `PortableDepthwiseConv2d` | 1.79e-7 | 5e-3 | 4 orders of magnitude |
| `PortableMaxPool2d` | 0.00e+0 | 5e-3 | byte-stable |
| `PortableAvgPool2d` | 0.00e+0 | 5e-3 | byte-stable |
| `silu` | 1.19e-7 | 5e-3 | 4 orders of magnitude |

Max-abs-diff observed is 4 orders of magnitude below the documented ε
band, confirming the canonical contract is robust under all tested
shapes / strides / kernel sizes.

## Tests

**17/17 PASS** at ε=5e-3 `[empirical:src/tac/portable_primitives/tests/test_portable_primitives_extended.py]`:

- BatchNorm2d (4 tests): train forward / eval forward / export round-trip /
  train-eval toggle identity check
- DepthwiseConv2d (4 tests): default forward / stride=2 (MBConv pattern) /
  no-bias / export round-trip
- MaxPool2d (3 tests): kernel=2 stride=2 / kernel=3 stride=2 padding=1
  (FastViT stem) / default-stride convention
- AvgPool2d (3 tests): spatial downsample / global pool (4x4→1x1) /
  kernel=3 stride=1 padding=1 (same-size smoothing)
- silu (3 tests): basic 2D / 4D post-conv / matches `x * sigmoid(x)`
  canonical definition

**Zero regression against WW base**: full
`pytest src/tac/portable_primitives/tests/` = **30/30 PASS** (13 WW base
+ 17 ARCH-1 extended).

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map = N/A (primitive-level layer; no sensitivity
  signal contribution at this stage; sister-3+ architecture-level port
  will participate)
- Hook #2 Pareto constraint = N/A (primitive-level; no Pareto-relevant signal)
- Hook #3 bit-allocator = N/A (primitive-level; no bit-allocator signal)
- Hook #4 cathedral autopilot dispatch = N/A (primitive-level; no dispatch
  contribution; sister-5 contest-axis paired-eval lands the hook #4 surface)
- Hook #5 continual-learning posterior = **ACTIVE** (test results +
  empirical max-abs-diff measurements participate in the canonical
  posterior via this landing memo's evidence tags)
- Hook #6 probe-disambiguator = N/A (no competing interpretations at
  primitive level; sister-3+ architecture-level port may surface
  disambiguator paths)

Per CLAUDE.md "Subagent coherence-by-default" — silent omission of hooks
that don't apply at this stage of the cascade is documented above, not
silently ignored.

## Discipline compliance

- **Catalog #229 (premise verification)**: read WW canonical interface
  files in full (`nn.py` / `backend.py` / `tensor.py` / sister test
  file) BEFORE any new code; verified MLX + PyTorch backends available;
  verified MLX exposes native BatchNorm / Conv2d-with-groups / MaxPool2d /
  AvgPool2d / silu primitives via empirical import + smoke before
  implementation.
- **Catalog #110/#113 (APPEND-ONLY HISTORICAL_PROVENANCE)**: NEW files
  only for nn_extended.py + sister tests; only `__init__.py` mutated
  via additive extension of existing `__all__` + import block.
- **Catalog #287 (forbidden empirical-claim-without-evidence-tag)**:
  every max-abs-diff + tests-pass-count + LOC claim above carries
  inline `[empirical:<path>]` tag or table-cell location.
- **Catalog #117/#157/#174 (commit serializer + POST-EDIT
  --expected-content-sha256)**: commit lands via canonical
  `tools/subagent_commit_serializer.py` with sha256s captured AFTER
  edits per the 2026-05-13 docstring correction.
- **Catalog #206 (subagent checkpoint discipline)**: 3 checkpoints emitted
  during the 4-phase execution (in_progress at phase 1 / phase 3 /
  complete at landing) so a successor on API crash can resume.
- **Catalog #340 (sister-checkpoint guard PROCEED)**: verified `PROCEED`
  via canonical `tac.commit_safety.sister_checkpoint_guard.check_files_against_sister_checkpoints(...)`
  on all 3 changed files BEFORE staging.
- **Catalog #1 (forbidden MPS-fallback default) + #192 (macOS-CPU advisory
  not promoted)**: MLX backend remains non-promotable; the new
  primitives inherit the WW canonical posture (numerical equivalence
  tested, but contest-axis promotion still requires PyTorch CUDA T4
  paired auth-eval per CLAUDE.md "Submission auth eval — BOTH CPU AND
  CUDA").
- **Carmack MVP-first 5-step**: ε=5e-3 numerical equivalence tests ARE
  the empirical falsifiable challenge to "MLX is sufficient for sister
  primitives" cargo-cult; 17/17 pass at 4-orders-of-magnitude headroom
  is the verdict per step 2; canonical equation reference (WW Phase 1
  PV ε band) per step 3; landed in same commit batch as smoke per step 4;
  re-routes operator priority queue to sister-2 attention-primitive cascade
  per step 5.

## Sister coordination (Catalog #230 ownership map + #314 absorption avoidance)

- **Slot 2 (`affcec3e` OVERNIGHT-BBB NSCS06 v8 rc=1 diagnosis)**: touches
  NSCS06 v8 substrate + landing memo — DISJOINT substrate. Verified
  via sister-checkpoint guard `PROCEED` with empty `conflicts` tuple.
- **Cap=2 firm**: only this slot + BBB active at landing time. No
  3rd sister overlap risk.

## Sister-2 ready signal

Next cascade stage per ZZ scope analysis: **attention primitives**
(LayerScale / multi-head self-attention block / token mixer / repmixer
block for FastViT-T12 stages). Foundation primitives in THIS landing
+ WW base (Linear / Conv2d / LayerNorm / GELU) are sufficient for
sister-2 to implement attention blocks without further base extensions.

Recommended sister-2 dispatch can spawn immediately after this commit
lands; ownership map: `src/tac/portable_primitives/nn_attention.py` (NEW;
disjoint from THIS landing's `nn_extended.py`).

## Operator-routable next-actions

1. **Spawn sister-2 (attention primitives)** per the 5-stage cascade.
2. (Optional) review the canonical empirical max-abs-diff table above
   and decide whether to tighten ε ceiling from 5e-3 to 1e-5 for
   ARCH-1 primitives specifically (the 4-OOM headroom would support
   a tighter pin).

## Cost

$0 paid GPU + ~50 min wall-clock + 16 tool uses.

## Cross-references

- WW landing memo (canonical base 9 primitives) — `.omx/research/` —
  baseline ε band reference.
- ZZ commit `330a58a60` op-routable #1 — primitive-level scope
  analysis listing 15 missing primitives; ARCH-1 covers 5 of them.
- CLAUDE.md "QAT pipeline — non-negotiable for FP4 deployment" —
  BatchNorm2d eval-mode contract step 2.
- CLAUDE.md "Beauty, simplicity, and developer experience" —
  primitives reviewable in 30 seconds each.
