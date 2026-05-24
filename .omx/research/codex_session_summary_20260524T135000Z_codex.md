# Codex Session Summary - MLX Learned-Sweep Queue/Replan Closure

UTC: 2026-05-24T13:50:00Z
Evidence grade: `[macOS-MLX research-signal][queue-owned][planning-only]`

## Scope

Continued the MLX learned-sweep queue/DAG tranche by closing the observation
feedback loop from two named-policy local MLX queue generations into a
cumulative planning replan. This session did not claim a score, promotion,
rank/kill verdict, or exact-eval dispatch readiness.

## Landed Artifacts

- `tools/replan_mlx_dynamic_learned_sweep_from_observations.py`
  - Merges one or more MLX observation JSONL ledgers into a learned-sweep
    replan.
  - Deduplicates observations across all input ledgers by the canonical
    observation identity key before replanning, while reporting raw,
    deduplicated, and duplicate row counts.
  - Writes JSON, Markdown, and summary JSON artifacts.
  - Preserves fail-closed authority fields:
    `score_claim=false`, `promotion_eligible=false`,
    `rank_or_kill_eligible=false`, and
    `ready_for_exact_eval_dispatch=false`.
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T133520Z.json`
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T133520Z.md`
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_summary_20260524T133520Z.json`
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T135429Z.json`
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_20260524T135429Z.md`
- `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_cumulative_observation_replan_summary_20260524T135429Z.json`
- `src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py`
  - Added CLI coverage for merging parallel observation ledgers and proving
    cross-ledger duplicate accounting.
- `tools/build_mlx_learned_sweep_autopilot_queue.py`
  - Hardened runtime-telemetry discovery policy handling so an explicit
    discovery policy survives even when no compatible prior telemetry states
    are selected.
  - Hardened auto-discovered SQLite state telemetry to require an MLX
    learned-sweep queue id prefix, local MLX resource kind, MLX autopilot step
    id/tool identity, and positive artifact telemetry before balancing batch
    roots from that state.
- `src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py`
  - Added regression coverage for empty-compatible-state discovery policy
    preservation.
  - Added regression coverage excluding compatible-looking ad hoc/wrong-resource
    queue states from auto-discovered runtime telemetry.

## Executed Queue Evidence

First queue:

- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_adaptive_row_group_batch_queue_20260524T132520Z.json`
- State:
  `.omx/state/experiment_queue_mlx_learned_sweep_named_policy_adaptive_row_group_batch_live_20260524T132520Z.sqlite`
- Worker result: `success_count=2`, `failure_count=0`, no failed
  postconditions, no timeouts.
- Status after execution: `status_counts={"succeeded": 2}`, `ready_steps=[]`,
  `orphaned_step_count=0`.
- Performance summary: schema `experiment_queue_performance_summary.v1`,
  `telemetry_only=true`, `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.
- Local MLX timing: mean `45.177701708482346s`, sum `90.35540341696469s`,
  success count `2`.

Second queue:

- Queue:
  `experiments/results/mlx_effective_spend_triage_learned_sweep_live_20260524T100114Z/mlx_autopilot_named_policy_post_observation_batch_queue_20260524T133410Z.json`
- State:
  `.omx/state/experiment_queue_mlx_learned_sweep_named_policy_post_observation_batch_live_20260524T133410Z.sqlite`
- Worker result: `success_count=2`, `failure_count=0`, no failed
  postconditions, no timeouts.
- Status after execution: `status_counts={"succeeded": 2}`, `ready_steps=[]`,
  `orphaned_step_count=0`.
- Performance summary: schema `experiment_queue_performance_summary.v1`,
  `telemetry_only=true`, `score_claim=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.
- Local MLX timing: mean `28.133610292017693s`, sum `56.267220584035385s`,
  success count `2`.

## Cumulative Observation Replan

Merged four observation ledgers:

- `mlx_autopilot_named_policy_adaptive_row_group_batch_observations_20260524T132520Z_pass_0001_mlx_scorer_response_window_544_545__mlx_local_response__micro.jsonl`
- `mlx_autopilot_named_policy_adaptive_row_group_batch_observations_20260524T132520Z_pass_0002_mlx_scorer_response_window_496_497__mlx_local_response__intermediate.jsonl`
- `mlx_autopilot_named_policy_post_observation_batch_observations_20260524T133410Z_pass_0001_mlx_scorer_response_window_496_497__mlx_local_response__smoke.jsonl`
- `mlx_autopilot_named_policy_post_observation_batch_observations_20260524T133410Z_pass_0002_mlx_scorer_response_window_98_99__mlx_local_response__smoke.jsonl`

Each ledger has 3 JSONL rows, for 12 total observation rows.

Guarded overwrite note: the first cumulative artifact set at `20260524T133520Z`
was written before the dedupe-accounting hardening. The repo artifact writer
correctly refused an in-place overwrite without an expected existing SHA, so the
dedupe-accounting version was written as the append-only `20260524T135429Z`
artifact set.

Replan summary for `20260524T135429Z`:

- schema: `mlx_dynamic_learned_sweep_replan_from_observations.v1`
- observation JSONL count: `4`
- raw observation row count: `12`
- deduplicated observation row count: `12`
- duplicate observation row count: `0`
- replan schema: `mlx_dynamic_learned_sweep_plan.v1`
- ranked row count: `116`
- local-ready row count: `52`
- suppressed observed row count: `12`
- authority flags remain fail-closed:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`,
  `ready_for_exact_eval_dispatch=false`.

Top cumulative local-ready rows include:

- `mlx_scorer_response:window:109:110` / `mlx_local_response` / `smoke`
- `mlx_scorer_response:window:544:545` / `mlx_local_response` / `smoke`
- `mlx_scorer_response:window:496:497` / `macos_cpu_advisory` / `smoke`
- `mlx_scorer_response:window:98:99` / `macos_cpu_advisory` / `smoke`

These are planning rows only. Local MLX/CPU rows may drive more local sweeps.
Contest CPU/CUDA rows still require materialization, controls, lane claim,
canonical auth eval, and harvest before any score claim.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_experiment_queue.py \
  src/tac/tests/test_staircase_dag.py \
  src/tac/tests/test_mlx_effective_spend_triage_selection.py \
  src/tac/tests/test_mlx_effective_spend_triage_learned_sweep_adapter.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_observation_harvest.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py
```

Result: `162 passed in 11.10s`.

Post-audit focused regression after hardening cross-ledger dedupe and stricter
state discovery:

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py \
  src/tac/tests/test_mlx_dynamic_sweep_observations.py
```

Result: `62 passed in 2.34s`.

```bash
ruff check \
  src/comma_lab/scheduler/experiment_queue.py \
  src/comma_lab/scheduler/mlx_learned_sweep_autopilot_queue.py \
  src/comma_lab/scheduler/staircase_dag.py \
  src/tac/optimization/mlx_effective_spend_triage_selection.py \
  src/tac/optimization/mlx_effective_spend_triage_learned_sweep_adapter.py \
  src/tac/optimization/mlx_dynamic_learned_sweep_observation_harvest.py \
  src/tac/optimization/mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/optimization/mlx_dynamic_learned_sweep_local_autopilot.py \
  src/tac/optimization/mlx_learned_sweep_batch_roots.py \
  tools/build_mlx_learned_sweep_autopilot_queue.py \
  tools/run_mlx_dynamic_learned_sweep_local.py \
  tools/run_mlx_dynamic_learned_sweep_autopilot.py \
  tools/harvest_mlx_dynamic_learned_sweep_observations.py \
  tools/adapt_mlx_effective_spend_triage_to_learned_sweep.py \
  tools/replan_mlx_dynamic_learned_sweep_from_observations.py \
  src/tac/tests/test_mlx_learned_sweep_autopilot_queue.py \
  src/tac/tests/test_mlx_dynamic_learned_sweep_local_actuator.py \
  src/tac/tests/test_staircase_dag.py
```

Result: `All checks passed!`.

```bash
git diff --check
```

Result: clean.

## PixelShuffle Routing Note

Wrote a separate name-disambiguation refresh memo:

- `.omx/research/codex_findings_pixelshuffle_name_disambiguation_refresh_20260524T134358Z_codex.md`

Verdict: `pixelshuffle_h64_long1000` and `psd_h64_long1000` are historical
internal post-filter lanes, while `PixelShuffle(2)` is a decoder primitive in
HNeRV/NeRV/BoostNeRV-style substrates. Current high-EV use is primitive
lowering/fusion/parity and tensor-layout search, not resurrecting the old
post-filter lane without fresh exact or component-response evidence.

## Remaining Work

- Build the next queue generation from the cumulative replan if local MLX
  capacity is still the best next action.
- Feed cumulative observation statistics into any runtime telemetry or root
  selection policy that can consume them without granting score authority.
- Continue profiling and lowering actual measured hotspots for HNeRV/NeRV
  family local MLX/Rust/Metal paths, including PixelShuffle primitive lowering
  where profiles justify it.
