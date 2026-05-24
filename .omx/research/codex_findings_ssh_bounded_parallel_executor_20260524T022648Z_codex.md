# Codex Findings: SSH Bounded Parallel Executor

UTC: 2026-05-24T02:26:48Z

## Scope

This pass hardened the queue-owned SSH executor from serial execution toward
bounded parallel execution for materializer and staircase queue work. The goal
is higher wall-clock throughput without letting the Dask/staircase plan, remote
processes, or remote artifacts become score or state authority.

## Findings

1. The executor could fan out only serially, so queue/DAG machinery was still a
   bottleneck after MLX/local scoring and materializer steps became cheap enough
   to parallelize.
2. A naive thread-pool fanout would have introduced two high-risk race classes:
   stale remote finalizers could overwrite a terminal or rewound queue row, and
   concurrent pullbacks could write to the same local artifact destination.
3. Stale-running recovery needed a local parent PID marker for SSH-external
   work. Without it, a live SSH worker row with no child PID could be classified
   as recoverable after the grace window.

## Fixes Landed

- `run_staircase_ssh_executor` now executes selected SSH tasks through a bounded
  `ThreadPoolExecutor` while preserving result order by plan index.
- Each worker uses short-lived SQLite connections for claim/finalize only. No
  transaction is held across remote preflight, SSH execution, or rsync pullback.
- `finalize_claimed_step_execution` now refuses stale finalization unless the
  row is still `running`, hashes still match, resource kind still matches, and
  the `worker_run_id` matches the running claim event.
- SSH running events now record `parent_pid`, allowing stale-running recovery to
  skip live SSH executor work.
- The executor now blocks duplicate or overlapping pullback destinations before
  remote execution.

## Self-Protection

New regression coverage asserts:

- parallel execution actually runs two selected tasks concurrently;
- every non-preflight remote command starts only after its local row is claimed
  `running`;
- two executor instances cannot duplicate the same remote command;
- duplicate pullback destinations are blocked before SSH or rsync starts;
- stale finalizers cannot overwrite terminal rows or mismatched worker claims;
- stale-running recovery skips a live SSH worker via the recorded parent PID.

## Authority Boundary

All outputs remain false-authority:

- no score claim;
- no promotion eligibility;
- no rank/kill eligibility;
- no exact-eval dispatch readiness.

This is throughput and custody infrastructure only. It may select local/SSH
follow-up work, but contest authority still requires byte-closed archive/runtime
packets on a contest CPU/CUDA auth axis.
