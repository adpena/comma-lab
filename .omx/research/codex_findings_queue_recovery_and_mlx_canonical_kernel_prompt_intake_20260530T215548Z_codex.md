# Codex Findings: Queue Recovery + MLX Canonical Kernel Prompt Intake

- UTC: 2026-05-30T21:55:48Z
- Agent: Codex
- Scope: `/Users/adpena/Downloads/comprehensive_research_prompt_math_steg_mlx_genius_20260530.md` review follow-through, queue-owned final-rate attack recovery, MLX canonicalization hardening.
- Score authority: none. All local queue/MLX artifacts remain `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.

## Findings

1. Exact-readiness handoff artifacts were being placed under rerunnable materializer output directories. Because those materializers intentionally use overwrite semantics, a later rerun could erase downstream handoff proof files and leave succeeded queue steps with vanished postcondition artifacts. Fixed by moving materializer exact-readiness handoff directories to siblings of the materializer output directory.

2. `tools/queue_control.py recover` reconciled stale running steps but did not automatically apply the canonical recovery-plan action for succeeded steps whose required artifacts had vanished. Fixed by auto-rewinding only typed `rewind_succeeded_step_with_artifact_failure` actions with cascade, preserving all other recovery decisions as explicit.

3. The recovered queue path `.omx/research/frontier_final_rate_attack_feca_default_exec5_20260528Tlocal/experiment_queue.json` was repaired and supervised to terminal success: 20/20 succeeded, no score/promotion authority.

4. Bounded fleet drain `.omx/research/queue_fleet_local_drain_exec_20260530T2146Z/` completed 3 cycles with 0 failed children. Final bounded fleet health: `INVALID_QUEUE=0`, `NEEDS_RECOVERY=0`, `READY_TO_SUPERVISE=0`, `TERMINAL=55`, `PAUSED_WITH_QUEUED_WORK=12`, `PAUSED_EXACT_DISPATCH_GATE=4`, `NON_EXECUTABLE_QUEUE_ARTIFACT=184`.

5. MLX prompt intake exposed a real source-level canonicalization bug: `tac.framework_agnostic.canonical_kernels` documented canonical NHWC pixel-shuffle and bilinear-resize helpers but did not implement/export them. Fixed by adding `pixel_shuffle_2x_nhwc_canonical` and `bilinear_resize_nhwc_canonical` with numpy, MLX, PyTorch, and tinygrad routes. MLX routes through the PR95 canonical native helpers to preserve gradients.

6. `src/tac/substrates/time_traveler_l5_z7_mamba2/mlx_native.py` still carried local numpy/PIL fallback helpers for pixel shuffle / bilinear resize. Fixed by routing the decoder through the canonical kernel surface and deleting the local duplicate helpers.

7. The MLX review-required surface had drifted into many subtly different `import mlx.*` and `mx.eval` idioms. Added `tac.framework_agnostic.mlx_runtime` as the canonical MLX acquisition/eval boundary and routed the remaining production/tool audit blockers through it. The strict canonicalization audit now reports 47/47 MLX files routed or canonical, 0 waivers, 0 review-required rows.

8. The queue-fleet drain produced transient cycle directories that were useful for replay but not durable signal. `.gitignore` now excludes `.omx/research/queue_fleet_local_drain_exec_*/cycle_*/` so future queue drains preserve top-level ledgers without adding scratch supervisor state to git.

## Current Blockers

- No remaining MLX canonicalization audit blockers in the current production/tool scan. New MLX code should use `tac.framework_agnostic.mlx_runtime` or another canonical helper surface by default.
- Frontier score is unchanged because this slice repaired automation and canonical kernels; it did not dispatch exact auth eval or produce a new byte-closed candidate.

## Verification

- `ruff check` passed on touched queue, MLX runtime/kernel, and routed tool files.
- Focused queue/materializer/fleet/runtime/kernel pytest passed: 99 passed, 1 skipped.
- Canonical kernel pytest passed as part of the same run.
- MLX canonicalization audit artifacts:
  - `.omx/research/queue_fleet_local_drain_exec_20260530T2146Z/mlx_canonicalization_audit_runtime_helpers_final5.json`
  - `.omx/research/queue_fleet_local_drain_exec_20260530T2146Z/mlx_canonicalization_audit_runtime_helpers_final5.md`

## Next Action

Turn the now-green MLX helper surface into more frontier work: run queue-owned chained rate/distortion materializers against the current CPU frontier, route new inverse-steg MLX acquisition work through the same runtime boundary, and keep exact-auth dispatch gated on byte-closed receiver proof.
