# ANE unlock follow-up — direct runtime, concurrency, and W8 batch tiers (2026-07-13)

`lane_id: lane_ane_unlock_followup_20260713`  
`research_only: true` · `training_launched: false` · `score_claim: false`  
`axis: [macOS ANE/CoreML/MLX local advisory] NON-PROMOTABLE` · `pointer_moved: false`

## Outcome first — MEANS, never the pointer

**Headline: ANE remains a verdict-advisory-only MEANS; this follow-up did not unlock an ANE
training-loop lever.** The exposed direct private inference API has forward execution and weight-update
surfaces but no backward/gradient/VJP surface; W8 shrinks the stored weight blob but worsens both
per-pair batch throughput and n600 QoI; T4 head-fp32 selection does not repair the matched fp16 drift;
and the representative MLX-GPU concurrency A/B is `BLOCKED_NOT_MEASURED` because this sandbox cannot
acquire Metal. Only a byte-closed exact evaluator row can move the score. No archive or evaluator ran.

The settled full-float32 CoreML route remains `UNLOCKED_LOCAL_ONLY` for verdict advisory. Nothing here
upgrades it to gradient or label authority. The current canonical `[contest-CPU]` pointer remains
`0.1880443979880752`; pointer delta is exactly zero.

## 1. Direct private-ANE route — built/probed, residency not executed

**MEASURED:** the native Objective-C probe compiled and ran on `macOS-26.4-arm64`. Both
`AppleNeuralEngine.framework` and `ANECompiler.framework` loaded from the dyld shared cache. The
filtered runtime inventory contains 35 ANE-prefixed classes and 22 forward/evaluate/enqueue candidates,
including `_ANEClient doEvaluateDirectWithModel...`, `_ANEInMemoryModel evaluateWithQoS...`, and
`_ANEVirtualClient evaluateWithModel...`. `_ANEWeight updateWeightURL:` and `_ANERequest weightsBuffer`
are consistent with delta-weight/update plumbing, but no entitled model was invoked.

**MEASURED:** zero selector candidates matched eleven backward-family tokens: `backward`, `gradient`,
`vjp`, `adjoint`, `autodiff`, `train`, `derivative`, `differentiate`, `costate`, `reverse`, and
`backprop`. No direct forward, direct residency, backward, or cotangent parity was executed.

**Decisive scoped answer:** backward/VJP reachable on ANE through the exposed direct private inference
API = **NO**. This caps this formulation at forward-only. It does not prove that no Apple firmware or
future API can implement a backward.

- `verdict_scope`: private-runtime classes/selectors visible on this exact OS build and the safe
  introspection formulation.
- `req-R`: a signed/entitled direct-ANE model plus a documented gradient/VJP execution selector and
  measured cotangent parity against NumPy-fp32 authority.
- Fragility: all class names/selectors are private, OS-version-sensitive, entitlement-sensitive, and
  unsuitable as a promotion dependency without a pinned runtime and parse-back proof.

Receipt: `experiments/results/ane_unlock_followup_20260713/direct_ane_probe.json`, SHA-256
`5421f6ea4b3929337696586e88e53f32bd98fb5d66319a55da3b0f323909bf0c`.

## 2. Representative ANE teacher ∥ MLX-GPU concurrency A/B

The representative MLX load is four 64-channel 3×3 convolutions plus reverse-mode differentiation at
`384×512`, deliberately exercising both compute and memory traffic. It is not the governed trainer.

**MEASURED:** frozen teacher solo median was `55.383667 ms/forward` over three calls with
`CPU_AND_NE` requested. **PLACEMENT NOT PROVED.** The matched MLX step failed before its first sample:

`[metal::load_device] No Metal device available ... sandboxed or virtualized macOS session`.

Therefore teacher degradation = `NOT_MEASURED`; MLX degradation = `NOT_MEASURED`; strict architecture
admission = **REJECT**. There are no invented zeroes. `powermetrics` also returned rc=1 with
`powermetrics must be invoked as the superuser`; ANE/GPU power is `BLOCKED_NOT_MEASURED`.

The frozen-teacher sidecar itself is built, resumable, and smoke-tested: three forwards, median
`59.141291 ms`, 30-second atomic checkpoint protocol, placement unproved. It is ready to serve as the
treatment load once an unsandboxed MLX process is available.

- Acceptance law: `D = T_concurrent/T_solo - 1`; both `D_teacher < 0.05` and `D_MLX < 0.05` strictly,
  plus independent ANE placement proof.
- `verdict_scope`: this sandboxed local process and representative MLX workload only.
- `req-R`: an unsandboxed local process in which `mlx.core.metal` acquires the Apple GPU, followed by
  the exact ABBA protocol with CPU_AND_NE placement telemetry.

## 3. Corrected W8 question — storage fit, batch timing, n600 QoI

This is **weight-only per-channel symmetric int8**. Activations remain unquantized. It is not the dead
W8A8 rung, and no int8-MAC speedup is claimed: the intended mechanism is stored-weight/SRAM headroom
and batch/dispatch amortization.

### Footprint

- Prior survey estimate: approximately `31 MB` fp16 working weight set (**INFERRED**, not remeasured
  resident SRAM here).
- Exact source state-dict payload: `38,442,892 bytes` fp32 (**MEASURED file tensors**).
- Same tensor count at fp16: `19,221,290 bytes` (**DERIVED**).
- W8 multifunction shared weight blob: `9,685,520 bytes = 9.686 MB` (**MEASURED on-disk package blob**).
- Whole three-function package: `10,354,194 bytes = 10.354 MB` (**MEASURED**), leaving
  `23,200,238 bytes` below a 32 MiB comparison bound (**DERIVED**).
- Actual ANE SRAM residency: `UNKNOWN_NOT_MEASURED`; package bytes are not silently relabeled SRAM.

Thus the requested shorthand is `31 MB (INFERRED survey working set) → 9.686 MB (MEASURED W8 package
weight blob)`, with the custody mismatch stated explicitly.

### Batch tiers

The multifunction package with `fwd_b1`, `fwd_b8`, and `fwd_b32` compiled and deduplicated to one weight
blob. CoreMLtools 9.0 then rejected `CompiledMLModel(functionName=...)` because the compiled container
was not identified as an ML Program. Timings therefore use the three exact fixed-shape source ML
Programs placed into that package; named-function dispatch itself remains blocked.

| fixed shape | batch median | ms/pair | vs b1 | verdict |
|---|---:|---:|---:|---|
| b1 | 59.476 ms | 59.476 | control | — |
| b8 | 525.987 ms | 65.748 | +10.545% | reject |
| b32 | 2,099.657 ms | 65.614 | +10.320% | reject |

All rows requested `CPU_AND_NE`; placement is unproved and E5RT cache creation was permission-blocked.
Within this formulation, shrinking weights does not unlock positive batch amortization.

- `verdict_scope`: these individually compiled fixed-shape W8 ML Programs and this host/SDK/cache state.
- `req-R`: execute all three named functions from one compiled container with proved ANE placement and
  repeat the matched timing protocol.

### n600 real-state QoI gate

Authority surface: exact frozen SegNet weights, real `gt_n600.npz['gt_f1']`, 600/600 states,
`384×512`, one-thread Torch-fp32 reference, NumPy argmax accounting. These are **n600 MEASURED**, not
n24 extrapolations. CoreML placement remains unproved, so the rows are local-advisory only.

| candidate | aggregate flips | aggregate flip | worst pair | worst-pair flip | advisory gate |
|---|---:|---:|---:|---:|---|
| dense fp16 control | 10,420,581 / 117,964,800 | 8.833636% | 433 | 63.347371% | control only |
| T4 fp32 logit head | 10,420,553 / 117,964,800 | 8.833612% | 433 | 63.340251% | no repair |
| W8 weight-only | 45,170,348 / 117,964,800 | 38.291378% | 419 | 88.775126% | **NO-GO** |

The matched dense control is slower and driftier than the prior clean n24 `9.098 ms / 2.474552%` anchor,
consistent with the recorded E5RT placement/cache failure. It is not substituted for that settled row.
W8 nevertheless adds approximately 29.458 percentage points aggregate drift versus its matched control,
so it fails even the tolerant verdict-advisory gate. T4 changes only 28 pixels over n600 versus the dense
control and does not collapse the drift. LUT4/LUT6 were not needed to answer the operator-specified W8
gate and were not measured; no verdict is assigned to those distinct formulations.

Measurement receipt: `experiments/results/ane_unlock_followup_20260713/measurement_receipt.json`,
SHA-256 `0990b6bfb0152d900858af054c1c16744dc816851b5e20449974c2a2ed14c739`.

## Consumer-tier correction-ladder update

| consumer tier | prior | follow-up update | current verdict |
|---|---|---|---|
| verdict-advisory | `UNLOCKED_LOCAL_ONLY` via settled full-float32 CoreML | W8 and T4 rejected; no new route. Existing full-float32 route unaffected. | `UNLOCKED_LOCAL_ONLY` unchanged |
| training-gradient | `BLOCKED_NOT_MEASURED` | exposed private inference API has no backward/VJP surface; no cotangent parity; concurrency not measured | `BLOCKED_NOT_MEASURED`; direct-inference formulation capped at forward |
| label-grade | `NOT_UNLOCKED` | W8/T4 fail n600 QoI; no placement or exact label-grade joint bar | `NOT_UNLOCKED` unchanged |

Overall scoped verdict: **ANE is a verdict-advisory-only forward MEANS.** No direct backward, positive
batch law, concurrency admission, training-gradient authority, or label-grade path was unlocked.

## Full-trainer concurrency A/B queue — prepped, but not one-GO-ready

The held typed ticket, exact launcher argv, resumable sidecar, strict gates, and sibling custody hashes
are materialized in
`experiments/results/ane_full_trainer_concurrency_ab_20260713/go_packet.json` (SHA-256
`feac1a8f1f085771dfab4b51fdff6aa9311749bfc94ac229851df1d1b3e12a27`).

The governed launcher dry-run caught a live sibling-owned blocker after an earlier pure compile had
succeeded: `LADDER↔Muon STAGGER VIOLATION`, because lane/movable windows `340/260` are not strictly
before `--muon-start-epoch 4`. Current queue status is `BLOCKED_TIMER_DSL_NOT_GO_READY`, not falsely
GO-ready. This lane did not edit the sibling timer or bypass the validator.
Sibling commit `61569fa752` fixes the separate inherited out-of-budget L7/Muon boot defect but leaves
this contradiction: Muon is now in-budget at epoch 4 while inherited LADDER support remains 340/260
epochs. Re-derivation after that commit still refuses.

- `verdict_scope`: sibling-owned four-epoch component-timer DSL on the current main worktree.
- `req-R`: sibling owner restores a typed four-epoch schedule whose LADDER windows end strictly before
  Muon; then the preparer and both governed `--dry-run` arms pass. Only then can one operator GO fire the
  bounded solo arm followed by the sidecar arm.
- DERIVED payoff: the settled full-float32 forward plus prior assumed teacher share gives the existing
  `2.293×` upper-case teacher speedup. Actual training-loop payoff remains blocked pending the
  sibling-owned in-loop forward/backward timer and the concurrency A/B; no contested split is promoted.

## Triality and verification

- Equation: `src/tac/canonical_equations/ane_unlock_followup_20260713.py`, equation id
  `ane_residency_concurrency_batch_sram_laws_v1`.
- DSL: `src/tac/witness_dsl/ane_unlock_followup_policy_20260713.py`; prepare-only, n24/four epochs,
  explicit operator-GO refusal.
- DAG: `.omx/research/ane_unlock_followup_DAG_FEED_20260713.md`; isolated because the canonical DAG is hot.
- Tests: 8 focused tests pass. Native probe builds/runs. W8 multifunction compiles. Sidecar smoke passes.
  Governed trainer dry-run correctly refuses the sibling schedule blocker. No heavy trainer ran.
- Storage: 8 GiB minimum preflight passed through `/private/tmp` because the preferred SSD is visible but
  not writable in this sandbox. All model packages lived in context-managed scratch and were deleted on
  exit; durable evidence is under `experiments/results`, never `/tmp`.
- Git custody: the canonical serializer was invoked on `main` with only this lane's isolated files. It
  failed before staging because the managed filesystem forbids creating the temporary git index:
  `error: unable to create temporary file: Operation not permitted`. The cached index remained empty;
  no sibling file was staged or committed. req-R is a permissions-enabled git process rerunning the same
  serializer file list.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; `PROGRAM.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/ane_ecosystem_survey_20260713.md`
- `.omx/research/ane_unlock_correction_20260713.md`
- `.omx/research/GO_PACKET_inloop_component_timer_20260713.md`
- `.omx/research/throughput_fresh_eyes_measurements_20260713.json`
- latest sister Codex findings/session summary, T3/design/directive memos
- `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`
- `.omx/state/canonical_frontier_pointer.json`; `.omx/state/canonical_task_status.jsonl`
- `reports/latest.md`; canonical equation, posterior, probe-outcome, and cost-band surfaces
