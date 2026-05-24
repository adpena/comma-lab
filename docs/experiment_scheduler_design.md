# experiment scheduler design

## Overview

A CLI tool that orchestrates post-filter training experiments across a fleet
of heterogeneous compute platforms (local MLX/MPS advisory compute, bat00 CUDA,
tertiary CPU-only edge work, Colab T4, Kaggle T4) with automatic scheduling,
telemetry, budgeting, and recovery.

## CLI interface

```bash
# Submit an experiment
pact-sched submit \
    --arch psd_h64 \
    --epochs 1000 \
    --alpha 20 \
    --platform auto \
    --priority high \
    --budget-gpu-hours 8

# Check fleet status
pact-sched status

# View telemetry dashboard
pact-sched dashboard

# List all results ranked by score
pact-sched results --sort scorer

# Proxy-score a candidate
pact-sched proxy <weights.pt>

# Promote a candidate to the submission
pact-sched promote <weights.pt>

# Fire the endgame checklist
pact-sched endgame --max-parallel 3

# Check platform budgets
pact-sched budget
```

## Platform registry

```yaml
platforms:
  local_mps:
    type: mps
    memory_gb: 64
    max_concurrent: 3
    connect: local
    budget: unlimited
    
  bat00_cuda:
    type: cuda
    gpu: "RTX 2070 Super"
    vram_gb: 8
    wsl_memory_gb: 12
    connect: "ssh bat00 'wsl --exec bash'"
    budget: unlimited
    keepalive: "schtasks WSLKeepalive"
    
  tertiary_cpu:
    type: cpu
    memory_gb: 8
    max_concurrent: 2
    max_artifact_gb: 1
    connect: "ssh adpena@tertiary"
    verified_hostname: "Tertiary.local"
    executor: "ssh_experiment_queue"
    policy: "light CPU-only work through queue-owned SSH executor"
    budget: unlimited
    
  colab_free:
    type: cuda
    gpu: "T4"
    vram_gb: 16
    connect: "gcloud notebooks"
    budget_hours_per_week: 12
    session_max_hours: 12
    setup: "pip install torch av safetensors timm einops smp numpy"
    
  kaggle_free:
    type: cuda
    gpu: "T4 or P100"
    vram_gb: 16
    connect: "kaggle api"
    budget_hours_per_week: 30
    session_max_hours: 12
    setup: "pip install av safetensors timm einops smp"
```

## Scheduler logic

1. On submit: check experiment against platform constraints (memory, VRAM,
   hidden width limits, time budget)
2. Auto-select the best available platform:
   - If GPU needed and local MPS is busy → bat00 CUDA
   - If bat00 WSL OOMs → Colab/Kaggle
   - If all GPU busy → queue
3. Launch with `nohup`/`tmux`/Colab API
4. Monitor via periodic SSH/API poll
5. On completion: auto-triage against promoted floor
6. On crash: detect via PID check, auto-restart with saved checkpoint

## Telemetry

Every 60 seconds, poll each active experiment and write:
```json
{
  "timestamp": "2026-04-09T12:00:00Z",
  "experiment": "psd_h64_long1000",
  "platform": "local_mps",
  "epoch": 50,
  "scorer": 4.25,
  "pose": 0.073,
  "seg": 0.036,
  "lr": 0.0005,
  "best_scorer": 4.16,
  "best_epoch": 26,
  "gpu_util_pct": null,
  "mem_used_gb": 5.2
}
```

Dashboard reads this file and renders a live Markdown table or simple HTML.

## Budget management

```python
class BudgetTracker:
    def __init__(self):
        self.platforms = load_platform_registry()
    
    def remaining_hours(self, platform: str) -> float:
        used = self.query_usage(platform)
        limit = self.platforms[platform].budget_hours_per_week
        return limit - used
    
    def can_afford(self, platform: str, estimated_hours: float) -> bool:
        return self.remaining_hours(platform) >= estimated_hours
```

## Recovery

- Every experiment saves best checkpoints via `save_best_checkpoint`
- On crash, the scheduler detects dead PID within 5 minutes
- If checkpoint exists: restart from checkpoint epoch
- If no checkpoint: restart from scratch
- Rate limit restarts: max 3 per experiment

## Queue-owned SSH execution

`tools/run_staircase_ssh_executor.py` consumes a `staircase_dispatch_plan.v1`
plus its source `experiment_queue.v1`. The DAG remains planning-only; the
executor re-reads the queue, checks the plan source hash, selects only ready
steps with explicit `queue_state_writeback`, claims the step in SQLite before
launch, and records a terminal `succeeded` or `failed` event after local
postconditions are evaluated.

Remote hosts must advertise `executor=ssh_experiment_queue`, `ssh_target`, and
an absolute `remote_repo_root`; queued SSH steps must have local-visible
postconditions because v1 terminal writeback is decided from local artifacts
after the remote command exits. `--execute` requires the canonical queue state
unless an explicit `--noncanonical-state-rationale` is supplied, orphaned active
state requires `--orphaned-state-rationale`, and dirty remote git requires both
`--allow-dirty-remote-git` and `--dirty-remote-git-rationale`. Dry runs are
false-authority artifacts and never grant score, promotion, rank/kill, or
exact-dispatch authority.

SSH execution that can create artifacts outside the local filesystem must use
the artifact mobility contract. Pass `--require-artifact-mobility` with either
one or more `--artifact-path-map LOCAL_PREFIX=REMOTE_PREFIX` mappings for input
rsync push plus output rsync pullback, or a specific
`--artifact-shared-path-rationale` when the remote host and local machine are
operating on the same mounted storage. The executor
claims the queue step locally, pushes every declared
`telemetry.input_artifact_paths` entry through the same path map, runs the
remote command, pulls back every mapped local-visible postcondition artifact,
and only then evaluates terminal postconditions in the local authority state. A
remote zero exit code without successful local artifact visibility is a failed
queue step, not a succeeded remote result; a missing or unmapped declared input
artifact is blocked before launch.

Declared SSH input artifacts are custody-bearing inputs, not advisory path
strings. `experiment_queue.v1` normalizes `telemetry.input_artifact_paths`, the
SSH selector rejects missing, unmapped, or symlinked inputs, directory pushes
use `rsync --delete`, remote artifact paths reject shell/glob metacharacters,
and terminal input-mobility events record per-file or recursive content
manifests before the remote command can run.

The materializer campaign runner exposes the same contract through
`--staircase-ssh-execute`, `--staircase-ssh-artifact-path-map`,
`--staircase-ssh-artifact-shared-path-rationale`, and
`--staircase-ssh-require-artifact-mobility`. `--staircase-ssh-execute` is
mutually exclusive with top-level local `--execute`, takes its own
`--staircase-ssh-max-steps` bound, and requires a mobility contract so bounded
fleet work can be launched from the generated queue without turning a detached
SSH filesystem into implicit source-of-truth state.

`tools/smoke_staircase_ssh_input_custody.py` is the no-network operator smoke
for that contract. It builds a tiny `experiment_queue.v1`, emits a staircase
dispatch plan, runs the production SSH executor with a fake transport, records
the directory input manifest, verifies the directory push used `rsync --delete`,
pulls back a local output artifact, and forces the resulting report through the
proxy false-authority boundary. Recursive output pullbacks use content-sync
semantics with `rsync --delete`, emit post-pull recursive manifests, and fail
closed if the requested manifest cap truncates the pulled tree. Passing this
smoke proves the custody plumbing; it does not prove SSH reachability, paid
dispatch readiness, or score authority.

The preferred production entry point is not a leaf materializer. Run
`tools/run_byte_shaving_materializer_campaign.py` directly from high-level
sources such as `--scorer-response`, `--inverse-scorer-surface`,
`--mlx-effective-spend-triage-selection`, or `--atom`. When `--plan` is omitted
the runner first builds an inverse-steganalysis action functional, then a
bounded byte-shaving campaign plan, then the queue/DAG/SSH artifacts. This keeps
inverse-cell materialization as one actuator underneath the planner instead of
making it the planning surface.

For local MLX acquisition, `tools/build_mlx_scorer_response_execution_queue.py`
can append that same runner as a follow-up step with
`--include-acquisition-followup` plus at least one
`--acquisition-baseline-response`. Each experiment then runs MLX scorer response
first, normalizes the fresh response against same-window MLX baseline responses
as `scorer_response_dataset.v1`, and only then builds the inverse action
functional, byte-shaving campaign plan, materializer execution queue, and
optional staircase plan under `.omx/research/mlx_acquisition_batches/`. This is
the queue-owned path for broad MLX acquisition batches: MLX remains local
research signal, the follow-up artifacts stay false-authority, and exact
contest CPU/CUDA remains the only score/promotion authority.

For PR95/HNeRV reproduction, `tools/build_pr95_mlx_optimizer_matrix_queue.py`
is the queue-owned fan-out entry point. It emits one plan-only
`tools/run_pr95_mlx_timing_smoke.py` packet per selected stage, optimizer
descriptor, and seed, then compiles those packets into `experiment_queue.v1`
with local-MLX concurrency preserved in `controls.max_concurrency.local_mlx`.
Descriptor-only or unsupported optimizer rows are recorded as false-authority
refusals instead of being queued. The generated worker commands are local
training probes only: they can measure MLX throughput and produce runtime
manifests, but they cannot claim score, rank/kill candidates, promote archives,
or dispatch exact eval until byte-closed export, receiver proof, runtime
consumption proof, PyTorch parity, and contest CPU/CUDA auth anchors exist.

## Exact-ready consumer

`tools/build_materializer_exact_eval_consumer.py` bridges materializer exact
readiness into queue-owned dry-run dispatch preparation. It consumes
`materializer_chain_exact_readiness_bridge_report.v1` and/or
`optimizer_candidate_exact_eval_ready_queue_v1` artifacts, dedupes candidates
by archive SHA plus runtime content/tree SHA plus explicit score axis, re-runs
the exact-ready audit and `exact_dispatch_authority` preclaim gate, and emits a paused
`experiment_queue.v1`.

The consumer is intentionally not a paid dispatcher. Generated claim steps use
`tools/claim_lane_dispatch.py --dry-run`, generated dispatch steps call
`tools/parallel_dispatch_top_k.py --dry-run`, and the consumer report is forced
through the planning/proxy false-authority boundary. A row can be selected for
the paused dry-run queue only after the exact-dispatch authority path says it
is ready; score, promotion, rank/kill, and contest authority still require the
later contest CPU/CUDA auth result artifact.

Materializer exact-eval dispatch plans dedupe by the score-affecting stable
identity: archive SHA, runtime-content SHA, runtime-tree SHA, and score axis.
Repeated `candidate_id` values are not dispatch authority and are not enough to
collapse rows. When multiple stable identities share a lane id, the plan keeps
the later rows visible but blocks them with
`same_lane_dispatch_claim_serialization_required:*` until the first dispatch has
a terminal claim. The plan also tightens its CUDA score floor from
`tac.frontier_scan.best_per_axis.contest_cuda` when canonical state has a
stricter frontier than the static fallback.

## Exact-ready score-axis repair

Legacy exact-ready queues that predate explicit `score_axis` metadata are not
hand-edited in place. `tools/repair_exact_ready_score_axis.py` audits each
source queue, copies only rows whose sole blocker is `score_axis_missing`, and
then reruns the exact-ready audit on each copy. The source queue remains
unchanged, the report keeps score/promotion/rank/dispatch authority false, and
copy writing requires `--write-repaired-queues`.

This repair is metadata hygiene, not dispatch authority. Repaired queues still
flow through `tools/build_materializer_exact_eval_consumer.py` or the normal
exact-dispatch authority path, where runtime-consumption proof, runtime-content
custody, active claims, and terminal result-review blockers can still reject
the row.

## Implementation plan

1. Phase 1 (now): Shell script orchestrator (`run_endgame.sh`) — DONE
2. Phase 2 (now): `experiment_queue.v1` local worker + queue-owned SSH executor
   for light CPU hosts
3. Phase 3 (later): Colab/Kaggle API integration
4. Phase 4 (stretch): Web dashboard via Cloudflare Pages

## What this enables for the writeup

"We trained our task-aware post-filter across a fleet of heterogeneous
compute platforms — Apple Silicon MPS (local Mac), NVIDIA RTX 2070 Super
(Windows WSL2), and free-tier Google Colab T4 GPUs — with automatic
experiment scheduling, crash recovery, and budget-aware platform selection.
The entire pipeline, from architecture search to final submission, used
zero paid cloud compute."
