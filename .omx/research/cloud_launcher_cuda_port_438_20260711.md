# Task 438 — cloud launcher + V9 CGauge Torch/CUDA port receipt (2026-07-11)

Status: **BUILD LANDED; CUDA 1:1 PARITY BLOCKED, FAIL-CLOSED**.  Authority:
`[macOS-CPU/Torch advisory] NON-PROMOTABLE`.  Pointer: **0.19108282
[contest-CPU] UNMOVED**.  No provider was contacted and no paid resource was
dispatched.  The active local arm at
`experiments/results/v9_cgauge_432_coherent_arm_20260711` was not read, written,
signalled, or used for verification.

## STORES CONSULTED

- `CLAUDE.md`, `AGENTS.md`, `docs/operating_manual_craft_handoff.md`.
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md` and
  `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`.
- `src/tac/witness_dsl/spec_v9_cgauge.py` (the typed scientific configuration),
  `experiments/train_levelset_witness_realized_through_R_mlx.py` (MLX semantic
  authority), and the canonical equation
  `witness_fp_reorder_transform_bit_identity_wall_v1`.
- `tools/launch_witness_run.py`, `experiments/modal_train_lane.py`,
  `scripts/remote_*.sh`, `tools/parallel_dispatch_top_k.py`, provider contracts,
  Modal call-ID ledger/harvester, and the canonical run-artifact contract.
- Canonical lane/task/subagent ledgers.  Existing Task 438 and lane
  `lane_cloud_launcher_v9_cgauge_cuda_438_20260711` were reused; no duplicate
  lane was created.

## Landed apparatus

### Torch/CUDA mathematical substrate

`src/tac/cuda_levelset_training.py` provides a deterministic Torch twin of the
level-set FiLM/HOSC forward, contest resize/round operator R, unified-tau action,
Eikonal/length, chroma annulus, realized signed margins, island-birth hinge,
Chan-Vese area counterforce, clDice/persistence recall, weight-entropy rate,
phase tie coordinate/advection support, ground-homography warp, temporal screw,
and analytic lane receiver band.  Parameters can be exported to the independent
NumPy-fp32 forward oracle; Torch acceptance is measured from identical weights,
never inferred from MLX.

`experiments/train_levelset_witness_realized_through_R_torch.py` derives every
scientific value from `compile_v9_cgauge_432_launch_config` (no invented trainer
flags), seeds Python/NumPy/Torch/CUDA, forces deterministic Torch algorithms,
uses frozen real SegNet/PoseNet through R, emits the backend-neutral `loss_terms`
JSONL schema, maintains EMA, and atomically preserves rolling resume plus a
distinct checkpoint at every stage boundary.  Resume state contains live model,
EMA, optimizer, all RNG streams, exact DSL argv, and typed-config hash; config
drift refuses restore.

The CUDA backend-local compile policy is **eager unless an exact forward +
gradient identity probe passes on that backend**.  This directly consumes
`witness_fp_reorder_transform_bit_identity_wall_v1`; MLX's verdict is never
transferred to CUDA.

### Provider-neutral plan and custody

`src/tac/deploy/witness_cloud_launcher.py` defines a deterministic provider plan.
Modal is the first implemented lifecycle; AWS/GCP remain explicit scaffold-only
contracts with no invented actuator.  `tools/launch_witness_cloud.py` is plan-only
by default.  Mutation requires `--execute`, exact `GO-CLOUD-381`, an executable
provider, a complete CUDA parity receipt, and exact local GT-cache SHA-256.

The Modal command reuses `experiments/modal_train_lane.py`, hence detached
`.spawn()` plus canonical call-ID ledger custody and `harvest_modal_calls.py`
harvest-or-lose semantics.  The driver exports `DALI_DISABLE_NVML=1` (Catalog
#244), explicit `WITNESS_TRAINER_MODE=full` (Catalog #326), deterministic CUDA
environment, durable `/modal_results` paths, storage preflight, CUDA import/device
probe, staged-asset bytes+SHA receipt, resume path, and per-stage output custody.
No dispatch command was executed in this task.

## MEASURED local evidence

CPU-light `--verify-only --compile-probe`, identical exported weights:

- Torch-vs-NumPy RGB max absolute delta: **3.0517578125e-05**.
- Phi max absolute delta: **4.172325134277344e-07**.
- Argmax equality: **true**; phi cosine: **1.0** (threshold >= 0.9997).
- CPU eager-vs-`torch.compile`: loss max delta **0.0**, gradient max delta
  **0.0029296875**; exact wall failed, so compiled training was not adopted.
- Focused contract/parity suite: **16 passed**.
- Modal and AWS plan smokes returned plan-only receipts; provider contact: **none**.

CUDA's fp-reorder wall is **UNMEASURED under hard containment**: there is no local
CUDA measurement and paid dispatch was forbidden.  The remote driver is wired to
measure it before a future run, but that is not evidence until harvested.

## NO-FAKE semantic coverage verdict

Review against the *compiled active V9 argv* found that the mathematical loss
substrate is substantial but the path is **not yet a 1:1 active-run twin**.  The
machine-readable `cuda_v9_port_coverage.v1` receipt therefore blocks full training
and makes every cloud plan non-executable.  Unclosed score-affecting surfaces:

1. generated/table pose-carrier frame0 dispatch and learnable dxi;
2. structured-init scorer-SDF prefit;
3. accum-pairs=8 gradient accumulation/acceptance semantics;
4. event-triggered sensor latches and resume state;
5. LADDER eased targets/per-class-lambda refresh;
6. seeded-island and birth-completion classwise ramp;
7. AdamW-to-Muon transition, rewarmup, and optimizer persistence;
8. sigma-min pose-finish gate;
9. resumable Polyak candidate export;
10. dseg-aware taper/per-group clipping; and
11. governed tail-cycle stop controller.

This is a **FORMULATION-scoped blocker for the current Torch implementation**, not
a verdict against CUDA or the representation family.  Removing a receipt entry
without a semantic twin + parity test is forbidden.  Consequently Task 438 cannot
truthfully be marked complete, and no launch should be authorized from this
landing yet.

## Triality

- **DSL:** `compile_v9_cgauge_432_launch_config` remains the sole scientific
  config owner; the provider plan owns only hardware/custody.  The coverage
  receipt compares the emitted active surfaces against the port.
- **DAG:** `FEED-438-cloud-cuda-port` in the canonical Top-AI/ML pursuit DAG.
- **Equations:** the port consumes the registered level-set/Chan-Vese,
  persistence, phase/warp, and
  `witness_fp_reorder_transform_bit_identity_wall_v1` laws.  No new empirical
  CUDA law was registered because CUDA was unmeasured.

Pointer delta: **none**.  Durable delta: provider/custody apparatus, portable
Torch/NumPy mathematical core, canonical telemetry/checkpoint surfaces, and a
strict blocker that prevents a partial port from masquerading as 1:1.

## 2026-07-12 operator-override supersession (append-only)

The blocker verdict above is the truthful historical state of the 2026-07-11
landing, but it is no longer the current state. The follow-on receipt is
`.omx/research/cuda_v9_throughput_optimization_438_20260712.md`.

All eleven missing score-affecting surfaces now have real Torch twins and
`cuda_v9_port_coverage.v1` reports `COMPLETE_1_TO_1` with no blockers. Under the
operator's 2026-07-12 training-only override, backend fp bit identity is advisory;
compiled/fused training adoption requires functional oracle parity
(`argmax_equal=true`, `cosine_phi>=0.9997`). Score authority is unchanged and
still requires a byte-closed archive through `upstream/evaluate.py`.

No provider was contacted. CUDA throughput remains
**UNMEASURED-pending-CUDA-dispatch**. Pointer delta remains **none**.
