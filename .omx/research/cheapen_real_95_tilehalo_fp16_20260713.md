# CHEAPEN-THE-REAL-95 — exact tile-halo + MLX training precision — 2026-07-13

`research_only=true` · `score_claim=false` · `pointer_moved=false` · `$0 LOCAL` · no training launch · no trainer edit

## Outcome first

| Surface | Outcome | Evidence label |
|---|---|---|
| Current n600 training wall | **295.352 s/epoch**, 0.492253 s/pair visit, 3.938027 s/optimizer chunk. This is an aggregate, not the requested component split. | **DERIVED** from MEASURED log timestamps/service durations over ep25–225 |
| Requested split `{MLX fwd, MLX bwd, render, R, losses, CPU verdict}` | `{NOT_MEASURED, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED, NOT_MEASURED, 89.532 s/epoch service; 0 s/epoch observed critical-path wait}`. The full 295.352 s remains unallocated. | **BLOCKED_NOT_MEASURED** for the five MLX components; CPU service **MEASURED**, zero wait **DERIVED from completion order** |
| Lever A | **NO-GO**. Exact-on-tiles: **NO**. Measured boundary coverage: **4.7365977%**. Safe local halo: **685 px** plus 23 global SE reductions, hence exact source coverage **100%**. Exact ideal upper bound **1.00x**, not a timing. | coverage **MEASURED n600**; topology/halo/speed ceiling **DERIVED** |
| Lever B | **NO-VERDICT-BLOCKED**. fp16/bf16 symbols exist, but this process has no Metal device. Cosine **N/A**; speedup **N/A**. | **BLOCKED_NOT_MEASURED**, environment-scoped only |
| A × B composition | **NOT DERIVABLE**. Numeric speedup is `null`; isolated overlapping factors are not multiplied. | canonical equation **REFUSED_INCOMPLETE_MEASUREMENTS** |

This unit does **not** satisfy the requested fresh measured component split or Lever-B measurement. It converts both misses into executable, fail-closed receipts instead of substituting the stale 78/22 row or a synthetic profiler.

## STORES CONSULTED

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`; `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` including §8; `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`; `.omx/research/cheapen_real95_directive_stratified_sensitivity_20260713.md`; `.omx/research/whole_step_megakernel_356_20260711.md`; `.omx/research/kernel_stack_sweep_443_20260711.md`; `.omx/research/micro_batch_v9_unlock_20260712.md`; `.omx/research/microbatch_bit_identity_smoke_n600_20260710.md`; `.omx/research/mlx_custom_grouped_backward_kernel_makes_mlx_gpu_fast_20260612.md`; `.omx/research/cheaper_exact_forward_transfer_95kill_20260713.md`; `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`; `reports/latest.md`; the live run's `launch.sh`, `daemon.log`, resume state, and EMA checkpoint; the n600 boundary-share receipt; frozen upstream scorer source; current MLX scorer adapters, R implementation, micro-batch twin, and trainer source.

Lane: `lane_cheapen_real95_tilehalo_fp16_20260713`, registered L0/phase 7. Canonical run directory was read only.

## Step 0 — current wall, not stale wall

### Settled facts recalled but not reused as current measurements

- #449's approximately 78/22 forward/backward split is a pre-current-stack CPU/older formulation anchor. **STALE FOR THIS DECISION**.
- #356 whole-step megakernel was already scoped NO-GO at at most 1.21x on its representative formulation with changed numerics. It was not reopened.
- `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (approximately 17x on its isolated grouped-backward gate), the custom persistence pool, fused R, safe HOSC compile region, and async CPU verdict are ON in the audited launch.
- Micro-batch is implemented and has prior parity/throughput receipts, but `--micro-batch-pairs` is **OFF** in this trajectory. Therefore “all requested speed levers ON” is false for the only live n600 wall evidence.
- The one-thread CPU law is a settled approximately 2.995x exact SegNet-forward subcomponent result. Here CPU verdict work completed asynchronously before the next verdict boundary, so its measured service demand is not a 30.3% critical-path share.

### Current canonical-equation-ready row

The log has no `profile_timing` rows. The live loss is a nested `value_and_grad` closure, and the execution sandbox reports no Metal device. No permitted short probe can recover disjoint fused-graph component times after the fact.

| Component | s/epoch | Status | Use in composition |
|---|---:|---|---|
| MLX scorer forward | `null` | `BLOCKED_NOT_MEASURED` | forbidden |
| MLX scorer backward | `null` | `BLOCKED_NOT_MEASURED` | forbidden |
| render | `null` | `BLOCKED_NOT_MEASURED` | forbidden |
| R | `null` | `BLOCKED_NOT_MEASURED` | forbidden |
| loss terms | `null` | `BLOCKED_NOT_MEASURED` | forbidden |
| CPU-torch verdict service | 89.532 | MEASURED: median 2238.3 s/call / cadence 25 | report separately; asynchronous service is not additive wall |
| CPU-torch verdict critical path | 0.000 observed wait | DERIVED from every completion preceding its next verdict boundary | zero in this window |
| unallocated training critical path | 295.352 | DERIVED residual | **not assignable** |

The 295.352 s/epoch aggregate is the median of eight corrected ep25 intervals. Each boundary-to-boundary completion delta is corrected by the change in async service duration: `(C_b-C_a) - (V_b-V_a)`. It covers 600 real pair visits and 75 optimizer chunks per epoch. Timestamp granularity and the correction make it DERIVED from measured logs, not a fresh instrumented component measurement.

Receipt: `experiments/results/cheapen_real95_tilehalo_fp16_20260713/current_wall_receipt.json`, SHA-256 `c9ec6b2d7154a69b98dddd5c8a6a47455187fcdd3c0f4ea6afbff28554ac3614`.

## Lever A — exact tile-halo and sensitivity waterfill

### Architecture-derived exact halo, including skips

Authority scorer: `smp.Unet("tu-efficientnet_b2")`, input `384×512`, nearest-neighbor U-Net decoder, two 3×3 convolutions per decoder block, final 3×3 head. For convolution, `r' = r + (k-1)j`, `j'=sj`. Decoder support was propagated with the exact integer recurrence `i -> floor(i/2)` and unioned with every skip; the deep support dominates every merge.

| Stage | jump | local RF width | safe local halo |
|---|---:|---:|---:|
| stem | 2 | 3 | 1 |
| encoder 0 | 2 | 11 | 5 |
| encoder 1 | 4 | 31 | 15 |
| encoder 2 | 8 | 111 | 55 |
| encoder 3 | 16 | 223 | 111 |
| encoder 4 | 16 | 479 | 239 |
| encoder 5 | 32 | 1055 | 527 |
| encoder 6 | 32 | 1183 | 591 |
| decoder 0 | 16 | 1247 | 639 |
| decoder 1 | 8 | 1279 | 663 |
| decoder 2 | 4 | 1311 | 675 |
| decoder 3 | 2 | 1311 | 681 |
| decoder 4 | 1 | 1311 | 684 |
| segmentation head | 1 | 1311 | **685** |

The final 32-phase support is not a symmetric radius-654 approximation:

```text
p=0:       [i-655,     i+623]      width 1279
p=1..30:   [i-(655+p), i+(655-p)]  width 1311
p=31:      [i-654,     i+624]      width 1279
max reach: 685 left, 654 right; safe symmetric halo = 685
```

That support clips to the entire 384×512 frame for every selected output. Stronger still, all **23** EfficientNet MBConv blocks contain squeeze-excite. Each spatial mean makes the forward and its VJP globally dependent even if the local RF were smaller.

R must also stay global: render → bicubic 874×1164 → camera uint8 STE → bilinear 384×512. Independently resizing a crop changes half-pixel phase; arbitrary origins also change stride-32 and nearest-decoder phase.

### n600 coverage and exactness gate

- MEASURED real-state `bulk_boundary.px_share = 0.04736597696940104` over exact pair IDs 0…599.
- That artifact's schema reports `bulk_boundary.share_of_flips = 0.26803822820970963`. The operator-routed “approximately 97% of d_seg” uses a different annulus/island accounting. They are not silently equated.
- Exact source area after the 685-pixel/global-SE dependency closure: **1.0**.
- Exact ideal speedup ceiling: `1/1.0 = 1.00x` before scheduling/copy overhead.
- A GO bar requires exact selected logits/argmax plus a **MEASURED >=2x** timing. Neither passes.

`n600_logit_bitcompare = STRUCTURALLY_REFUSED_BEFORE_EXECUTION`: there is no smaller exact operator to compare. Calling the full frame a “tile” and bit-comparing it would be tautological, not evidence.

**Lever-A verdict: NO-GO.** Exact-on-tiles **NO**; measured speedup `null`; DERIVED ideal upper bound **1.00x** at measured 4.7366% annulus coverage.

**Verdict scope:** finite input-crop tile-with-halo exact sparse forward for the frozen `tu-efficientnet_b2` SMP U-Net SegNet at 384×512. This is not a verdict on sparse decoder work after one dense encoder/SE pass, cached-SE approximate training, a local student, or dense-forward/sparse-cotangent formulations.

### Waterfill arm and what approximation loses

The operator-routed three-tier/class-pair waterfill is preserved as a default-OFF proposal, but it does not rescue exact frozen-scorer convolution:

- Tier 0 blind camera coordinates (22.7%) can remove render/R work before the scorer; they do not remove exact globally coupled scorer work. The 25.6% static MyCar core overlaps blind coordinates by an unresolved amount, so the Tier-0 union is only bounded `[25.6%, 48.3%]`.
- Tier 2 remains every-step/full-precision, weighted by margin-saliency × class flip density (Road 2.2x, Lane 32x, Undrivable 0.26x, hood approximately 0; Movable numeric weight absent, so no guess).
- With the pre-registered freshness law `max K: IoU^K >= 0.90`, class cadences are Lane 1, Movable 1, Road 2, MyCar nonstatic remainder 17, Undrivable 21; pair cadence is the minimum of its classes. Static hood is a never-compute proposal.

A low-cadence full-frame refresh makes Tier-1 a training-tolerance formulation only. It does **not** restore current-step exactness. Between refreshes it loses the global SE VJP outside selected tiles, interior CE/bulk cotangent, class-interior margin/calibration forces, and current-step Tier-1 changes. No expected waterfill speedup is derived because the Tier-0 overlap and continuous Tier-1 saliency/cost histogram needed for FLOP integration are not measured in one common scorer-compute geometry.

Reformulation queue: (1) sparse decoder after measured dense encoder/SE; (2) cached SE with explicit n600 gradient gate; (3) local student with n600 Jacobian/argmax gate; (4) loss/cotangent sparsification after a dense forward.

Receipt: `experiments/results/cheapen_real95_tilehalo_fp16_20260713/tile_halo_receipt.json`, SHA-256 `b9f264166fea40224966c1902065eebd3fb34949750f87d7fd020e963bb99465`.

## Lever B — fp16/bf16 MLX training path

### Pre-registered gate

Training signal only; NumPy-fp32 byte-close and exact CPU/CUDA evaluation remain untouched.

```text
GO(dtype) := speedup_fwd+bwd >= 1.5
          AND global cosine(dL/d post-R RGB, fp32) >= 0.99
          AND min pair cosine over exact n600 >= 0.99
          AND quality_pairs = 600
```

The probe restores the preserved EMA checkpoint, consumes the real `gt_n600.npz`, renders fixed pair states without an optimizer step, casts scorer weights **and** activations, retains fp32 loss reductions, records adapter/custom-kernel dtype receipts, times forward and `value_and_grad`, and atomically checkpoints each quality pair. It also records argmax flips, pose drift, relative L2, and per-pair/global render-pixel cotangent cosine. A dtype that silently falls back or fails a custom grouped-convolution backend remains blocked.

### Execution receipt

Host identity selected without serial/UUID fields: Apple M5 Max, 128 GB, MLX 0.31.2. `mx.float16`, `mx.bfloat16`, and `mx.float32` symbols exist. The actual GPU preflight failed:

```text
[metal::load_device] No Metal device available. This typically occurs in
headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

Therefore fp16 cosine = `null`, bf16 cosine = `null`, fp16/bf16 speed = `null`. **Do not infer that low precision is slow or inaccurate on M5 Max.**

**Lever-B verdict: NO_VERDICT_BLOCKED.** Verdict scope is this execution environment only; the fp16/bf16 training-path family remains queued.

Metal-enabled forward/grad probe command (not a training run):

```bash
TAC_MLX_CUSTOM_GROUPED_BACKWARD=1 .venv/bin/python \
  tools/probe_mlx_real_n600_precision.py \
  --run-dir experiments/results/levelset_v752_baseline_20260710T185913Z \
  --checkpoint levelset_witness_ema_mlx.npz \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --candidate-dtypes float16 bfloat16 \
  --timing-pairs 8 --quality-pairs 600 --warmup 1 --repeats 3 \
  --out experiments/results/cheapen_real95_tilehalo_fp16_20260713/mlx_precision_receipt.json
```

Receipt: `experiments/results/cheapen_real95_tilehalo_fp16_20260713/mlx_precision_receipt.json`, SHA-256 `78982aa3223d9a327c326b9ceb95cb4231bcf2ba3ba4f49e576e55d1e10b2240`.

## Canonical equation and composition refusal

For a completely measured split,

```text
T0 = t_mlx_fwd + t_mlx_bwd + t_render + t_R + t_loss + t_cpu_critical
T' = T0 - Σ_i (t_i,before - t_i,after)       [MEASURED disjoint components only]
S_total = T0 / T'
```

Async CPU service is reported separately from `t_cpu_critical`. Overlapping levers cannot enter `Σ` twice. Tile-halo and mixed precision both touch scorer forward, so they require one joint measured A+B before/after component; multiplying isolated `x_A * x_B` is forbidden.

Current substitution is impossible: the five MLX components are null, A has no measured timing and fails exactness, B is blocked, and no joint A+B receipt exists. Thus:

```text
S_total = null
status  = REFUSED_INCOMPLETE_MEASUREMENTS
```

Composition receipt: `experiments/results/cheapen_real95_tilehalo_fp16_20260713/composition_receipt.json`, SHA-256 `8115f8292c975f2f97e5a50f6da25f8e86a7fdc9858be0ff2c3a0d682837f99c`.

## Triality and system wiring

- **Equation:** `src/tac/canonical_equations/amdahl_measured_wall_split_20260713.py`; measured-only, disjoint-component law; overlapping components fail closed.
- **DSL:** `src/tac/witness_dsl/tile_halo_mixed_precision_proposal.py`; both arms default `OFF_UNWIRED`, emit no invented argv, require a measured anchor before enablement, and carry the pre-registered GO bars.
- **DAG:** `.omx/research/sub015_DAG_cheapen_real95_tilehalo_fp16_20260713.md`; isolated FEED because shared DAG files are hot/sibling-owned.

Six-hook disposition: sensitivity map **ACTIVE** via margin × class-pair waterfill; Pareto constraint **ACTIVE** via measured seconds/cosine/exactness gates; bit allocator **NO DIRECT BYTE ACTUATOR** but consumes the same sensitivity object; cathedral/autopilot **BLOCKED until receipt GO**; continual learning **this memo + receipts + FEED**; probe-disambiguator **binary annulus exact control vs three-tier approximate waterfill, plus fp16/bf16 modes**.

## Verification and custody

- Focused pure CPU suite: **18 passed**.
- `py_compile`/source tests do not require Metal. The blocked Metal preflight itself is a durable receipt.
- New evidence is 21,981 bytes of JSON plus resumable small per-pair quality rows only; no bulky artifact was generated. GT cache is pre-existing and read only.
- `experiments/results/levelset_v752_baseline_20260710T185913Z` was not mutated. No optimizer step, training run, paid dispatch, exact-eval dispatch, archive mutation, or trainer edit occurred.
- Pointer delta: **ZERO**. All evidence is `[macOS-MLX research-signal / macOS-CPU advisory; NON-PROMOTABLE]`.
- Files remain **uncommitted for main review**, as requested.
- Integration note: the sibling-owned uncommitted `steps_dimension_95kill_20260713{,_SPEC}.md` and `sub015_DAG_steps_dimension_95kill_20260713.md` captured interim hashes of these receipts while this lane was still hardening source custody. They were not edited here; main must refresh those citations to the final hashes above before joint landing.
