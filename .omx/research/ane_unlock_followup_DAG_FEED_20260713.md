# ANE unlock follow-up — isolated DAG FEED (2026-07-13)

`research_only=true` · `[macOS ANE/CoreML/MLX local advisory] NON-PROMOTABLE` · `pointer_moved=false`

This FEED is isolated because the shared canonical DAG is live/hot. It records a MEANS-only
compute lane. It grants no score, archive, contest-CPU, contest-CUDA, training-gradient, or
label authority. Only a byte-closed exact evaluator row can move the pointer.

## Nodes and edges

1. `frozen_b2_fp32` → `coreml_dense_fp16_b1` (`convert`, advisory forward).
2. `coreml_dense_fp16_b{1,8,32}` → `w8_weight_only_b{1,8,32}` (`offline per-channel W8`,
   activations unquantized).
3. `w8_weight_only_b{1,8,32}` → `w8_multifunction_package` (`merge functions`, one deduplicated
   weight blob).
4. `w8_multifunction_package` → `batch_timing_receipt` (`coremlcompiler` compile succeeded;
   compiled-container `functionName` dispatch blocked; timing uses its three exact source ML Programs).
5. `frozen_b2_fp32` → `t4_head_fp32` (`FP16ComputePrecision.op_selector`, SegNet head retained fp32).
6. `{coreml_dense_fp16_b1,w8_weight_only_b1,t4_head_fp32}` → `n600_qoi_receipt`
   (frozen fp32 Torch argmax reference, aggregate and worst-pair flips).
7. `private_ANE_runtime` → `forward_selector_inventory` (`MEASURED`, dlopen + Objective-C runtime).
8. `private_ANE_runtime` → `backward_vjp_surface` (`MEASURED` zero selectors across eleven
   gradient/VJP tokens; `INFERRED` exposed-inference-API cap, no model executed).
9. `{frozen_teacher_sidecar,representative_mlx_gpu_load}` → `concurrency_AB`
   (`BLOCKED_NOT_MEASURED` locally because Metal device acquisition and powermetrics authority fail).
10. `concurrency_AB` → `held_full_trainer_n24_AB` (`HELD_OPERATOR_GO_REQUIRED`; sibling component
    timer retained as the owner of the true forward/backward split).

## Triality

- DSL leg: `src/tac/witness_dsl/ane_unlock_followup_policy_20260713.py` seals the n24/four-epoch
  held A/B and refuses actuation.
- DAG leg: this FEED owns the nodes/edges without touching the hot shared DAG.
- Equation leg: `src/tac/canonical_equations/ane_unlock_followup_20260713.py` owns strict
  concurrency degradation, batch seconds/pair, payload-vs-cliff accounting, and parameterized
  forward-only Amdahl laws under equation id `ane_residency_concurrency_batch_sram_laws_v1`.

## Six-hook wire-in disposition

1. Sensitivity map: no score sensitivity is registered. The lane measures compute only; a forward
   approximation with no cotangent parity cannot become a gradient prior.
2. Pareto constraint: strict local timing/fidelity/placement gates are recorded; no score-rate
   Pareto admission exists.
3. Bit allocator: non-binding. Neither CoreML packages nor teacher weights ship in `archive.zip`.
4. Cathedral/autopilot: consumes only the held GO packet at
   `experiments/results/ane_full_trainer_concurrency_ab_20260713/go_packet.json`; it cannot fire
   without operator GO and a fresh lane-claim check.
5. Continual learning: the durable measurement receipt plus the dated follow-up memo carry every
   positive and negative; no chat-only result is authoritative.
6. Probe disambiguator: `tools/bench_ane_unlock_followup_20260713.py` ships both W8 and T4 modes and
   admits neither from proxy loss. LUT4/LUT6 remain separately reported if their compile runs.

## Negative verdict ladder

- Direct backward/VJP: `verdict_scope=private-runtime selectors visible on this exact macOS build
  and the safe introspection formulation`; `req-R=signed/entitled direct-ANE model plus documented
  gradient/VJP selector and cotangent parity receipt`.
- Concurrency: `verdict_scope=this sandboxed host process and representative MLX load formulation`;
  `req-R=unsandboxed local MLX Metal acquisition plus ANE placement telemetry and exact ABBA timing`.
- Multifunction dispatch: `verdict_scope=coremltools 9.0 CompiledMLModel loading of this compiled
  multifunction container`; `req-R=runtime path that identifies the package as ML Program and
  executes all three named functions from the same container`.

## STORES CONSULTED

- `CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`
- `.omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md`
- `.omx/research/SPEC_v8_perclass_decomposition_20260708.md`
- `.omx/research/ane_ecosystem_survey_20260713.md`
- `.omx/research/ane_unlock_correction_20260713.md`
- `.omx/research/GO_PACKET_inloop_component_timer_20260713.md`
- `.omx/research/throughput_fresh_eyes_measurements_20260713.json`
- `.omx/state/lane_registry.json`; `.omx/state/subagent_progress.jsonl`
- `.omx/state/canonical_frontier_pointer.json`; `reports/latest.md`

