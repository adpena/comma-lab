# PP2 static Metal-port audit: PR130 pose carrier

**Verdict: `portable-with-named-work`.** This is a static source verdict, not a Metal execution receipt. The audited trainer has a dense forward/backward path whose operator families are covered by current MPS registrations, but the exact pinned runtime still has three UNKNOWN rows out of 60 inventoried execution families. The highest-risk family is `nn.Embedding(..., sparse=True)` backward through sparse COO coalescing and the custom row-local optimizer on PyTorch 2.10.0. One unconditional `torch.cuda.empty_cache()` call is a certain port edit, not an unknown.

No MPS/MLX device was accessed. No trainer, scorer, evaluator, archive, launch, dispatch, or paid job ran. No score was measured. Nothing here is promotion-eligible, and the frontier pointer is unchanged.

## Scope and custody

- Read-only trainer: `/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code/train_pose_carrier_full.py`, 467 lines, SHA-256 `684a4906edecb7653572db77c11a03a4e445eb256a8dc7b665e8fa0f78cab649`.
- Direct intake imports audited: `learned_pose_carrier_oracle.py`, `pose_basis_oracle.py`, `semantic_renderer_oracle.py`, and the conditionally imported `evaluate_semantic_quantization.py`. Together with the trainer: 5 files, 1,415 lines.
- Challenge import audited: `upstream/modules.py`, lines 28-80 for `AllNorm`, `ResBlock`, `Hydra`, and `PoseNet`, plus the `timm` FastViT-T12 composition instantiated by line 66.
- Runtime lock: `upstream/uv.lock` pins PyTorch 2.10.0 and timm 1.0.22. The static dispatcher available in the current repository environment was PyTorch 2.12.1/timm 1.0.27. Therefore the current dispatcher is corroboration, not proof of the pinned port.
- Charter SHA-256: `65b35999eea7ea2fa3c2f1209e1040e06028a916a728a511d0606982ac969754`. Common-contract SHA-256: `eeae9e0035582e6bdd65fd837e4aa35a65e064fd09900b9c212d41ac02086771`.
- Operator denominator: one row per distinct execution family needed by setup, forward, backward, optimizer, evaluation, or device-bound loading on the active trainer/import path. Repeated call sites are folded into one row and all source sites are listed. Result: **60 rows; 3 UNKNOWN (5.0%)**. Rows 36-37 are two surfaces of the same sparse-gradient risk; row 33 is the external direct-to-MPS safetensors loader. See `PP2_OPS.jsonl`.

The MPS classification uses static source plus the local dispatcher and is bounded by the official [PyTorch MPS backend note](https://docs.pytorch.org/docs/stable/notes/mps.html), [MPS environment-variable reference](https://docs.pytorch.org/docs/stable/mps_environment_variables.html), and [PyTorch's operator-coverage tracking issue](https://github.com/pytorch/pytorch/issues/141287). `PYTORCH_ENABLE_MPS_FALLBACK=1` is useful as a detector, not permission to call a port clean: the acceptance probe must emit zero fallback warnings.

## What the trainer actually does

The trainer renders or loads all 600 uint8 master frames, instantiates the official frozen PoseNet, learns a low-resolution basis plus one sparse coefficient row per pair, and differentiates PoseNet MSE through the exact carrier render. The carrier path is bicubic/bilinear resize, uint8 STE, low-rank `einsum`, RGB-to-YUV6, FastViT-T12, and the official pose head (`train_pose_carrier_full.py:26-177,237-290,305-360`; `learned_pose_carrier_oracle.py:64-125`; `pose_basis_oracle.py:104-149`; `upstream/modules.py:28-80`).

The PoseNet weights are frozen at `train_pose_carrier_full.py:268-271`, but backward still traverses the entire FastViT and pose head to reach the carrier pixels, basis, and coefficients. A forward-only PoseNet smoke is therefore insufficient.

Static FastViT/PoseNet module census in the current matching model implementation found 78 `Conv2d`, 28 `BatchNorm2d`, 19 GELUTanh, 13 `Dropout`, 1 `AdaptiveAvgPool2d`, 14 `Linear`, 8 `BatchNorm1d`, and 12 `ReLU` modules. PoseNet is always `.eval()`, so dropout/drop-path are identity paths, while batch norm remains a real frozen-statistics forward plus input-gradient backward.

## Operator verdict

Dense operations are not the blocker. Bilinear and bicubic interpolation forward/backward, convolution, batch/group norm, linear/addmm, GELU, ReLU, adaptive average pool, `einsum`, reductions, indexing, `index_select`, `index_copy_`, and `index_add_` all have native or composite coverage in the inspected PyTorch 2.12.1 dispatcher. Basis Adam requests neither fused nor foreach execution, so MPS takes the ordinary single-tensor path. `clip_grad_norm_` also falls to its ordinary vector-norm path on MPS; this may be slower but is not statically unsupported.

The three UNKNOWN rows are:

1. `PP2_OPS.jsonl` row 36: sparse `nn.Embedding` backward on the exact PyTorch 2.10.0 MPS build. Current 2.12.1 exposes `SparseMPS` registrations, but no exact-version real-Metal receipt was found.
2. Row 37: sparse COO `coalesce()`, `indices()`, and `values()` flowing into `RowLocalSparseAdam`. It is coupled to row 36 but is a distinct failure surface.
3. Row 33: `safetensors.torch.load_file(..., device='mps')`. This external device-loader behavior is avoidable: load on CPU, call `load_state_dict`, then move the module to MPS.

The certain edit is row 34: `train_pose_carrier_full.py:264` unconditionally calls `torch.cuda.empty_cache()`. It must dispatch by `device.type`, using `torch.mps.empty_cache()` on MPS, CUDA cache clearing on CUDA, and no-op on CPU.

The sparse path is the highest-risk operator because failure there blocks every training step and because replacing it naively with ordinary dense Adam changes the mechanism. `RowLocalSparseAdam` maintains an independent bias-correction clock for each coefficient row (`train_pose_carrier_full.py:128-176`). Any dense fallback must gather only selected rows and preserve those row-local clocks and untouched-row identity; substituting stock Adam over the full table is not an equivalent port.

One-line decisive probe:

```text
PyTorch 2.10.0 on real MPS: Embedding(600,12,sparse=True), repeated row ids, squared loss, backward, coalesce/indices/values, gradient clip, and two RowLocalSparseAdam steps; require zero CPU-fallback warnings, exact selected-row set, untouched rows bit-identical, and CPU/MPS state/update parity within a predeclared fp32 tolerance.
```

## Exhaustive device-pin and AMP census

Scope: the five intake files above (1,415 lines) plus the active `upstream/modules.py` PoseNet definition.

- Tensor/module `.cuda()` calls: **0**.
- Literal tensor allocations with `device='cuda'`: **0**.
- CUDA-specific API calls on the active trainer path: **1**, `torch.cuda.empty_cache()` at trainer line 264.
- Trainer CLI default: `--device cuda` at line 220. This is a default, not a physical pin; every model and active tensor placement otherwise uses the parsed `device`.
- Conditional CUDA defaults exist in standalone imported scripts (`learned_pose_carrier_oracle.py:56`, `pose_basis_oracle.py:43`, `semantic_renderer_oracle.py:47`) but their `main()` functions are not invoked by this trainer.
- AMP/autocast in the trainer: **0**. There is no `--amp` argument. `semantic_renderer_oracle.py:246` has autocast only in that file's standalone training `main()`, which is not reached by importing `SemanticTokenRenderer`.
- `pin_memory`, DALI, NCCL, fused optimizers, custom CUDA extensions/kernels on the active path: **0**.
- Master caching is explicitly controlled by `--cache-masters-on-device`; the E2E pose helper always passes it (`scripts/e2e.py:165-170`).

Therefore the executable Metal policy is fp32 with AMP off because there is no AMP path to disable. Do not invent an `--amp` flag for this trainer.

## Exact argparse surface

The trainer defines 34 arguments at lines 182-222:

- Required paths: `--challenge-root`, `--target-cache`, `--master-checkpoint`, `--init-carrier`, `--out`, `--save`.
- Optional cache controls: `--master-cache=None`, `--reuse-master-cache` false, `--cache-masters-on-device` false.
- Schedule/batches: `--steps=20000`, `--stop-after-step=None`, `--batch-size=12`, `--eval-batch-size=12`, `--render-batch-size=4`, `--eval-every=1000`.
- Optimization: `--lr-basis=0.003`, `--lr-coeff=0.03`, `--basis-freeze-fraction=0.30`, `--basis-train-until-fraction=1.0`, `--qat-fraction=0.65`, `--coeff-qat-fraction=None`.
- Loss/mining: `--metric-loss-after-basis` false, `--always-metric-loss` false, `--metric-normalized-weight=0.0`, `--hard-mining-power=0.0`, `--hard-mining-max=8.0`.
- Representation: `--basis-bits=8`, `--coeff-bits=8` with choices `{8,10,12,16}`, `--amplitude=32.0`, `--master-carrier-amplitude=0.0`, `--carrier-base=gray` with choices `{gray,master}`, `--zero-init-coeff` false.
- Determinism/device: `--seed=20260715`, `--device=cuda`.

Validation only checks positive `--steps` and `1 <= stop_after_step <= steps` at lines 224-228. The port must not add invented runtime flags; it should use the existing typed surface or the already-landed lifted wrapper named below.

## Stage 09-32 graph and cache crosswalk

Stage 01 explicitly builds `run_dir/cache/official_targets.pt` with `--dataset dali` (`scripts/e2e.py:294-310`). Every stage 09-32 declares that same target-cache path as an input. What is **not** declared in the stage graph is the cache's content hash, decoder/hardware axis, or producer environment. That omission is load-bearing: the retained official-Ada DALI cache and the same-T4 DALI cache are different byte objects, while the same-T4 AV cache represents the contest-CPU decoder axis. A future run must bind one named cache SHA-256 and label its axis; neither local cache is evaluator authority.

The master caches are path-declared and include a `source_checkpoint` string that the trainer validates at lines 47-62. Their byte hashes are not declared. Stage 11 creates `masters_qat12k.pt` from stage 07; stages 12-20 reuse it. Stage 21 creates `masters_final_semantic.pt` from stage 08; stages 22-32 reuse it.

| Stage | Script | Device declaration | Init / upstream state | Master input | Target cache |
|---|---|---|---|---|---|
| 09 `pose_pilot4` | `learned_pose_carrier_oracle.py` | pipeline device | random basis | stage 02 semantic seed checkpoint | explicit stage-01 path; content implicit |
| 10 `pose_pilot12` | `learned_pose_carrier_oracle.py` | pipeline device | stage 09 basis | stage 02 semantic seed checkpoint | same |
| 11 `pose_joint_qat` | audited trainer | pipeline device | stage 10 | stage 07; **creates** `masters_qat12k.pt` | same |
| 12 `pose_raw7500` | audited trainer | pipeline device | stage 11 latest | reuse `masters_qat12k.pt` | same |
| 13 `pose_hard750` | audited trainer | pipeline device | stage 12 latest | reuse `masters_qat12k.pt` | same |
| 14 `pose_resident` | audited trainer | pipeline device | stage 13 latest | reuse `masters_qat12k.pt` | same |
| 15 `pose_uniform` | audited trainer | pipeline device | stage 14 best | reuse `masters_qat12k.pt` | same |
| 16 `pose_tail64` | audited trainer | pipeline device | stage 15 best | reuse `masters_qat12k.pt` | same |
| 17 `pose_cpu_coeff100` | audited trainer | **CPU forced** | stage 16 final | reuse `masters_qat12k.pt` | same |
| 18 `pose_search32` | `search_pose_coeff_cpu.py` | **CPU forced** | stage 17 | reuse `masters_qat12k.pt` | same |
| 19 `pose_search256` | `search_pose_coeff_cpu.py` | **CPU forced** | stage 18 | reuse `masters_qat12k.pt` | same |
| 20 `pose_cpu_fullqat` | audited trainer | **CPU forced** | stage 19 | reuse `masters_qat12k.pt` | same |
| 21 `pose_retarget_coeff1000` | audited trainer | pipeline device | stage 20 | stage 08; **creates** `masters_final_semantic.pt` | same |
| 22 `pose_basis_adapt250` | audited trainer | pipeline device | stage 21 latest | reuse `masters_final_semantic.pt` | same |
| 23 `pose_basis_adapt3000` | audited trainer | pipeline device | stage 22 best | reuse `masters_final_semantic.pt` | same |
| 24 `pose_final_cpu100` | audited trainer | **CPU forced** | stage 23 final | reuse `masters_final_semantic.pt` | same |
| 25 `pose_official_coeff` | audited trainer | pipeline device | stage 24 best | reuse `masters_final_semantic.pt` | same |
| 26 `pose_grid128x3` | `search_pose_coeff_cpu.py` | pipeline device despite filename | stage 25 best | reuse `masters_final_semantic.pt` | same |
| 27 `pose_grid64x12` | `search_pose_coeff_cpu.py` | pipeline device despite filename | stage 26 | reuse `masters_final_semantic.pt` | same |
| 28 `pose_refine_pass1` | `refine_pose_coeff_codes.py` | pipeline device | stage 27 | reuse `masters_final_semantic.pt` | same |
| 29 `pose_refine_pass2` | `refine_pose_coeff_codes.py` | pipeline device | stage 28 | reuse `masters_final_semantic.pt` | same |
| 30 `pose_anchor` | `refine_pose_coeff_codes.py` | pipeline device | stage 29 | reuse `masters_final_semantic.pt` | same |
| 31 `pose_int6_stable8k` | audited trainer | pipeline device | stage 30 | reuse `masters_final_semantic.pt` | same |
| 32 `pose_int6_coefftail4k` | audited trainer | pipeline device | stage 31 final | reuse `masters_final_semantic.pt` | same |

Denominators: **24/24** pose stages read the explicit target-cache path. **15/24** use the audited trainer; of those, 12 inherit the pipeline device and 3 are intentionally CPU. The other **9/24** use three sibling script families and are outside this trainer verdict. Therefore “the entire pose leg is Metal-portable” is not established.

## Ordered executable port plan

1. **Bind the immutable inputs before code work.** MAIN selects `contest-CPU/AV` or `contest-CUDA/DALI`, records the exact target-cache SHA-256 and producer axis, pins PyTorch 2.10.0/timm 1.0.22 (or explicitly governs a newer runtime), and names the exact master/initial checkpoints. Do not mix cache axes.
2. **Work in the existing lifted, resumable vehicle.** Reuse `src/tac/pr130_lift/pose/train_pose_carrier_full_resumable.py` and its prior MX2B resume apparatus; do not modify the read-only intake or recreate checkpoint semantics. Preserve per-stage, atomic, stage-named checkpoints and RNG/optimizer/scheduler state.
3. **Land the deterministic device edits.** Replace unconditional CUDA cache clearing with device dispatch. Load PoseNet safetensors on CPU, load the state dict, then move the module to the selected device. Keep fp32; there is no AMP flag or autocast path to change.
4. **Fire the sparse micro-probe first.** Run rows 36-37 plus gradient clipping and two optimizer steps on real MPS under the governed runtime. Require zero fallback warnings and CPU/MPS row/state parity. If it passes, retain the sparse mechanism. If it fails, implement a selected-row dense fallback that preserves row-local bias-correction clocks and untouched-row identity; do not substitute full-table Adam.
5. **Fire a full one-batch graph probe.** Use exact PoseNet weights, real basis/master shapes, both carrier-base branches that the stage graph uses, QAT on, loss backward, both optimizers, and scheduler. Assert raw-basis grad, sparse/dense coefficient grad, no PoseNet weight grads, no CPU fallback, finite values, and a CPU reference tolerance.
6. **Exercise setup branches.** Cover quantized and non-quantized semantic checkpoints, both master-cache states, `--cache-masters-on-device` on/off, direct cache source-checkpoint validation, latest/best checkpoint writes, and resume parity through a stage boundary.
7. **Audit the nine remaining stage scripts.** Apply the same operator/device/argparse/cache method to `learned_pose_carrier_oracle.py` as an executable, `search_pose_coeff_cpu.py`, and `refine_pose_coeff_codes.py`. Only after all three are closed may the 24-stage pose leg receive a portability verdict.
8. **Only then authorize a bounded governed training receipt.** MAIN owns real-Metal execution because this arm has no Metal device. Storage preflight, safe-run governor, distinct stage checkpoints, and durable SSD outputs are mandatory. This port unit itself authorizes no launch.

## Recall evidence

Original recall searched the canonical research index, hot state, task ledger, research corpus, DAG surfaces, and equation registry before deciding. The most decision-changing prior artifacts were:

- `.omx/research/ddm_mp2_20260809T105302Z/MP2_FINDINGS.md`: proves this arm has no Metal device and that prior “whole directory has no CUDA pin” wording was too broad; it did not establish the pose sparse path.
- `.omx/research/ddm_rm1_20260808/ADDENDUM_gt_provenance_and_compute_split.md`: establishes DALI/AV and DALI-hardware cache ambiguity, so cache-axis binding must precede a port receipt.
- `.omx/research/ddm_mx2b_20260806/RECEIPT.md`: identifies the already-built resumable lifted pose wrapper; the port must reuse it.
- `.omx/research/pr130_eureka_intake_acquisition_20260806.md`, `.omx/research/pr130_lift_wave_element_audit_20260806.md`, and `.omx/research/pr130_lift_wave_round1_adversarial_review_20260806.md`: establish intake custody and the pose rail's role without proving MPS support.
- Canonical equation registry: no prior equation or measured receipt closed the exact sparse-MPS trainer question. No score equation was evaluated in PP2.

## Triality and boundaries

- DSL leg: exact 34-argument trainer surface and E2E-provided device values are recorded above; no invented flag was admitted.
- DAG leg: all 24 stages and both master-cache transitions are crosswalked above with explicit/implicit inputs.
- Equation leg: no contest score term was measured. Portability is reported as typed operator counts, not translated into score units.
- Measured here: source bytes/hashes, line counts, argument/stage/operator censuses, and local dispatcher registrations.
- Not measured here: any MPS kernel execution, CPU fallback count, seconds per step, memory, training quality, PoseNet parity, score, archive bytes, or evaluator result.
- Authority boundary: static source audit only. Even a future successful MPS training receipt would be `[macOS-MPS research-signal]`, not contest-CPU/CUDA authority.

## LIVE-HYPOTHESES

- The exact sparse path may work unchanged on PyTorch 2.10.0 because PyTorch 2.12.1 exposes native `SparseMPS` registrations for the required COO and indexing surfaces; this remains plausible but untested on the governed version.
- CPU-load-then-`Module.to('mps')` should eliminate the safetensors unknown without changing model bytes because state-dict loading is device-agnostic and the current direct-device load is setup-only.
- If sparse COO fails, a selected-row dense optimizer can preserve the original row-local mechanism at small cost because the coefficient table is only 600 by the basis dimension and each step already has explicit `batch_ids`.
- The dense PoseNet backward is likely portable because every constituent FastViT/PoseNet family has current MPS coverage and MAIN already closed the semantic renderer's dense MPS path; only an exact-weight real-shape backward can turn that into evidence.

## DEAD-ENDS

- Declaring the pose leg portable from the prior semantic-renderer MPS receipt is closed: the pose trainer adds sparse embedding/COO, custom row-local Adam, safetensors direct-device loading, and a full PoseNet backward.
- Treating all stages 09-32 as this trainer is closed: 9/24 stages use three other script families.
- Calling stage 01 portable is closed: DALI target extraction is physically CUDA-bound; the reusable output cache must instead be named by hash and axis.
- Enabling AMP is closed for this trainer: no AMP flag or active autocast path exists.
- Replacing sparse row-local Adam with stock dense Adam is closed because it changes per-row bias-correction clocks and updates semantics.
- Running a CPU substitute in this arm is closed as a Metal-port test: it cannot measure MPS coverage or CPU fallback.

Own-vehicle frontier: unchanged at `0.7534578126155775` for `357,837` bytes, `[macOS-CPU advisory]`; contest pointer unchanged and this static audit produced no score row.
