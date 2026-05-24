# Codex Findings - MLX Learned Sweep CPU-Advisory Surface

UTC: 2026-05-24T14:34:39Z

## Scope

This pass continued the learned-sweep queue/autopilot tranche after local MLX
rows became exhaustible. The immediate failure mode was that the planner emitted
`macos_cpu_advisory` rows as ready local sweep work, but the actuator and queue
validator only admitted `mlx_local_response`.

## Landed Fix

- `macos_cpu_advisory` is now a supported local learned-sweep config, but only
  as an explicit local advisory artifact harvest. It never synthesizes CPU
  advisory signal from MLX cache tensors.
- The actuator requires candidate and baseline advisory JSON paths on the source
  selection row, validates `score_axis=cpu_advisory`,
  `evidence_semantics=non_contest_cpu_auth_eval_advisory`, false-authority
  fields, archive/runtime/raw hashes, and component contributions, then appends
  a `macos_cpu_advisory` observation row.
- The autopilot queue builder can now target `--sweep-config-id
  macos_cpu_advisory --device cpu`, writes matching queue postconditions, and
  refuses GPU execution for that config.
- The queue builder now fails before dispatch if the selected rows for a
  macOS-CPU advisory sweep do not carry both candidate and baseline advisory
  artifact paths. This moves the known-doomed failure from runtime actuation
  into queue construction.
- The auto-batch root planner forwards `sweep_config_id` into run specs so row
  roots can compile CPU-advisory queues rather than silently defaulting back to
  MLX rows.
- A next-surface report helper now routes exhausted plans to either local MLX,
  CPU-advisory artifact harvest, exact-calibration blockers, or candidate
  regeneration, with false-authority metadata throughout.
- A handoff helper and guarded CLI now stamp existing validated macOS-CPU
  advisory candidate/baseline artifact paths onto learned-sweep selection rows.
  It refuses non-advisory score axes, missing runtime identity, missing inflated
  output identity, and truthy authority fields before a CPU-advisory queue can
  consume the rows.

## Authority Boundary

All new rows and reports keep:

- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`
- `dispatch_attempted=false`
- `gpu_launched=false`

The CPU-advisory observation is planning/replanning signal only. It is not a
contest CPU score, not a CUDA score, and not exact-eval readiness.

## Verification

Focused tests:

- `pytest src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py
  src/tac/tests/test_mlx_learned_sweep_next_surface.py
  src/tac/tests/test_mlx_learned_sweep_advisory_handoff.py
  src/tac/tests/test_scorer_loss_convergence_detector.py`
- Result after scheduler pre-dispatch path validation and scorer-convergence
  export repair: `73 passed`

Focused lint:

- `ruff check` over touched implementation, tools, and tests
- Result: passed

Additional live-current check:

- Attempted to compile the latest cumulative replan's top
  `macos_cpu_advisory` row into a queue. It failed before writing a queue:
  `candidate macOS-CPU advisory path is required`.
- That failure is now intentional and strict: the current selection artifact
  does not yet stamp `local_cpu_advisory_source_path` or
  `window_baseline_local_cpu_advisory_source_path`.
- The next-surface artifact for the latest cumulative replan is:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_next_surface_report_20260524T143323Z.json`.
- `git diff --check`: clean.
- `tools/lane_maturity.py validate`: `OK — 1243 lane(s) validated cleanly`.
- `tools/review_gate_hook.py`: passed.

## Next Integration

The next high-EV bridge is wiring materializer/local-CPU eval producers to emit
the path-map artifact consumed by
`tools/stamp_mlx_learned_sweep_advisory_handoff.py`. That lets the queue run
MLX surface learning first, stamp validated CPU-advisory confirmations second,
and exact-auth anchor only candidates that survive both local axes.
