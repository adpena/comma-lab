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

The materializer campaign runner exposes the same contract through
`--staircase-ssh-execute`, `--staircase-ssh-artifact-path-map`,
`--staircase-ssh-artifact-shared-path-rationale`, and
`--staircase-ssh-require-artifact-mobility`. `--staircase-ssh-execute` is
mutually exclusive with top-level local `--execute`, takes its own
`--staircase-ssh-max-steps` bound, and requires a mobility contract so bounded
fleet work can be launched from the generated queue without turning a detached
SSH filesystem into implicit source-of-truth state.

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
