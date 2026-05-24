---
schema: codex_findings_v1
author: codex
created_at_utc: 2026-05-24T00:03:35Z
lane_id: codex_exact_ready_consumer_tranche_20260523
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
---

# Materializer Campaign Control Plane

## Findings

The materializer stack had the right primitives but still required too much
manual stitching after proof-chain execution. The queue builder now appends
per-materializer follow-up steps that harvest the produced chain manifest,
run the exact-readiness bridge, and write a paused dry-run exact-eval dispatch
queue. This keeps the high-throughput path inside `experiment_queue.v1` and
avoids shared aggregate handoff paths that would serialize or race.

The normal campaign entrypoint now exposes the follow-up flags, and
`tools/run_byte_shaving_materializer_campaign.py` provides a one-command
build/init/run/observe wrapper for local materializer campaigns. Paid dispatch
remains separate and explicit; generated dispatch queues are paused and dry-run
by default.

The storage preflight path had a bug class where `/Volumes/<name>` paths could
be fabricated when an external SSD was missing. Storage tier planning now fails
closed on missing or non-mounted external volume anchors before creating or
probing workload roots. DQS1 scheduler-preflight queues also now require real
cleanup execution and bind generated heavy output roots to the declared
workload root.

## Landed Surfaces

- `comma_lab.scheduler.byte_shaving_campaign_queue`: per-row
  materializer -> harvest/exact-readiness -> dispatch-plan DAG steps.
- `tools/build_byte_shaving_campaign_queue.py`: CLI flags for materializer
  exact-readiness follow-ups and dry-run dispatch-plan controls.
- `tools/run_byte_shaving_materializer_campaign.py`: one-command local
  materializer campaign runner.
- `comma_lab.scheduler.materializer_exact_eval_dispatch_plan`: strict
  exact-ready queue to paused claim/dispatch queue builder.
- `comma_lab.storage_tiers`: external volume mount truth guard.
- `comma_lab.scheduler.dqs1_local_first_queue` and
  `tools/run_dqs1_local_first_tranche.py`: real cleanup gate and storage-root
  binding for generated DQS1 queues.
- `comma_lab.scheduler.staircase_dag`: verified `tertiary` over SSH and added
  it as a low-memory, CPU-only Tailscale worker preset until a remote
  experiment-queue writeback executor lands.
- `.gitignore`: generated materializer campaign and handoff artifacts remain
  local custody unless explicitly promoted.

## Verification

- `PYTHONPATH=. .venv/bin/python -m ruff check ...` on touched scheduler,
  storage, tool, and test files.
- `PYTHONPATH=. .venv/bin/python tools/experiment_queue.py --queue configs/experiment_queues/dqs1_pairset_local_first.yaml validate`
- Focused tests: `75 passed`, then `128 passed`.
- Broad queue/control-plane regression: `400 passed in 23.16s`.
- `git diff --check`
- `git check-ignore` for generated materializer campaign, exact-readiness, and
  temp handoff artifacts.

## Frontier Status

No new exact auth eval was dispatched in this tranche. `reports/latest.md`
still lists `[contest-CPU Linux x86_64]` best at `0.1920282830` and
`[contest-CUDA T4]` best at `0.2053300290`.

## Next Tranche

The next high-EV tranche is a real queue-claiming executor for the staircase
/ Dask planner surface: Dask may schedule work, but `experiment_queue.v1` must
remain the authority for claims, state, postconditions, pause/rewind, and
promotion. Once that executor exists, materializer campaigns can fan out across
local CPU/MLX and peer machines without losing custody or duplicating truth.
