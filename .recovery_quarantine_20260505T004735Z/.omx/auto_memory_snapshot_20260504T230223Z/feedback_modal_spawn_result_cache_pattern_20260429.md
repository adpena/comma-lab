---
name: Modal spawn() returns artifacts in result cache (24h TTL) — must HARVEST or LOSE
description: 2026-04-29 PM CRITICAL. Modal .spawn() puts artifacts in the FunctionCall return value (result cache, ~24h TTL). modal_train_lane.py uses spawn() exclusively. Calls that complete (rc=0 or rc>0 with traceback) STILL hold their artifacts in the cache; if you never call .get(), they're GC'd at TTL expiry. Today's incident: 31 of 37 calls had artifacts sitting in the cache for hours; 5 ran successfully (lane_gp_v2, lane_gp_v3, lane_mm_v2, uniward_v7, uniward_v8) and we'd have lost everything if not harvested before 24h.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## The bug class — "spawn artifacts orphaned in result cache"

`modal_train_lane.py` uses `fn.spawn(...)` (fire-and-forget) to dispatch lanes. The `_run_lane_inner` function:
1. Runs the lane script inside the Modal worker.
2. Scans `$WORKSPACE` for `.bin/.zip/.pt/.mkv/.json/.log/.safetensors` files.
3. Packs them into a dict and returns it as the function's return value.
4. The dict sits in Modal's `FunctionCall` result cache for ~24h.

The local-side dispatcher writes `modal_call_id.txt` + `modal_metadata.json` and exits. It does NOT poll for the result. NOTHING is written to a Modal volume.

**If you don't call `modal.functions.FunctionCall.from_id(call_id).get()` within 24h, the artifacts are GC'd.**

This means:
- `modal app list` shows the FUNCTION as still queued/running, but if it actually completed you'd see results in the cache only via `.get()`.
- `modal volume ls` shows NOTHING because the artifacts were never written there.
- `modal app logs` only shows the most recent log buffer — earlier successful runs may have aged out.

## What I missed earlier today

I claimed ALL Modal apps had wasted ~$40 with zero output. I was wrong because:
1. I read `modal app list` (only showed currently-queued/running, not completed).
2. I read `modal app logs <app>` for visible apps (only 4-6 lines of "waiting for GPU").
3. I read `modal volume ls` (no lane outputs because spawn() doesn't write volumes).

I forgot to check `modal.functions.FunctionCall.from_id(call_id).get()` for the 37 dispatched call_ids tracked in `experiments/results/lane_*_modal/modal_metadata.json`.

When I ran the harvester (`/tmp/harvest_modal_calls.py`), here's what came back from the 31 fetched calls:
- **5 OK runs** (rc=0): lane_gp_v2 (824s), lane_gp_v3 (793s), lane_mm_v2 (832s), uniward_v7 (848s), uniward_v8 (779s). Each ~13-14 min, real artifacts.
- **1 long timeout**: lane_w_v2 (28800s = 8h, hit max_seconds cap)
- **~14 quick crashes** (rc=1, 1-3 min): mostly OOM on A10G (Lane SC++ tried 21+GB allocation; A10G has 22GB shared with other tenants → CUDA OOM)
- **~6 still NOT_READY**: sa_v4, sc_plus_plus_v4, mae_v_v2, q_faithful_v3, stc_cuda, sz_phase2_v2 (still queued — waiting hours for GPU capacity)
- **0 EXPIRED**: harvested in time

Modal billing dashboard confirms $38.80 spent on lane training (T4 $27.25 + A10G $11.55). The runs DID consume real GPU time. They DID produce artifacts. We DID nearly lose them all.

## Permanent fix

**Harvest pattern**: any time `experiments/results/lane_*_modal/modal_metadata.json` is created, schedule a harvester to run within 24h. Reference impl: `/tmp/harvest_modal_calls.py` (move into `tools/harvest_modal_calls.py` next session).

**Better pattern**: change `modal_train_lane.py` so `_run_lane_inner` writes artifacts to a Modal Volume instead of (or in addition to) returning them in the dict. Then volumes persist indefinitely and `modal volume get` works.

**Best pattern**: change to `fn.remote()` blocking + write-to-volume + log-streaming, not `.spawn()`. The `.spawn()` pattern was added for "detached" runs, but the price is orphaned artifacts. Detached + persistent storage requires a Volume, not the result cache.

## Modal scheduling truths I should have known

1. Modal T4 and A10G are GPU-shortage prone. "Function is waiting to be scheduled" can mean hours, not minutes.
2. Preempted runs (`Runner interrupted due to worker preemption`) restart from scratch with the same input — no charge for the preempted attempt, but you pay full price on retry.
3. `modal app list` only shows currently-active apps. Terminated ephemeral apps disappear from this list quickly.
4. `modal app logs` shows the most recent log buffer; earlier runs aged out aren't reachable via CLI.
5. The actual billing source-of-truth is the Modal web dashboard at modal.com/usage (NOT any CLI command).
6. Result cache TTL on spawn() is documented at https://modal.com/docs/guide/spawn — confirm exact TTL each session (likely 1-7 days depending on plan).
7. Volume writes persist indefinitely. Volume reads via CLI: `modal volume get <vol> <path>`.

## CLI commands to NEVER skip when investigating "what happened"

```bash
# Per call_id (the truth source for spawn() runs):
.venv/bin/python -c "import modal; r = modal.functions.FunctionCall.from_id('fc-...').get(timeout=2); print(r.get('returncode'), r.get('elapsed_seconds'), len(r.get('artifacts', {})))"

# Modal billing dashboard (open in browser, NOT CLI):
open https://modal.com/usage

# Volume contents:
.venv/bin/modal volume ls <volume_name>
.venv/bin/modal volume get <volume_name> <path>
```

## Cross-refs

- Code: `experiments/modal_train_lane.py` (the spawn-using dispatcher)
- Tracker: `experiments/results/lane_*_modal/modal_metadata.json` (call_ids per dispatch)
- Harvest summary: `experiments/results/_modal_harvest_summary.json`
- Modal docs: https://modal.com/docs/guide/spawn (verify TTL each session)
- CLAUDE.md "Tooling — non-negotiable" section gets the harvest-within-24h rule
