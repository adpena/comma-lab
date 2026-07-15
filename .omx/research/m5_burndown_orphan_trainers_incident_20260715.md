# #509 burn-down: orphaned duplicate dry-start trainers — OPERATOR ACTION NEEDED — 2026-07-15

**Status: OPEN BLOCKER (operator kill needed).** Written by the respawned #509 arm
(`m5_burndown_509`, session 01Jg5…2QUM instance) after the auto-mode classifier twice
refused its cleanup kill (ownership evidence below). MEANS/apparatus; pointer 0.19108 UNMOVED.

## The facts (all MEASURED/verified, not inferred)

1. Three M1 attribution dry-start launcher attempts (16:27Z, 16:30Z, 16:36Z) each got
   SIGURG/handler-killed AFTER the launcher had already passed gates and spawned its
   bounded dry-start trainer — invisibly, because the launcher's stdout is BLOCK-BUFFERED
   when redirected (log tail froze at the `throughput gate: measuring` line while gates
   completed and the trainer spawned). The safe_run supervisor died WITH each launcher;
   the `bash launch.sh` + trainer children (own sessions) survived.
2. Surviving orphans (psutil-verified): **pids 14828 and 21381**, both
   `train_levelset_witness_realized_through_R_mlx.py --out-dir
   experiments/results/levelset_n600_drystart3_telemetry480_20260715/dry_start`
   (this arm's session-created label), parents `bash .../dry_start/launch.sh`. The third
   (41735) already exited.
3. **They are UNSUPERVISED**: no safe_run timeout remains — the compiled schedule is the
   REAL 3000-epoch n600 config, i.e. they run for DAYS unless killed. They burn ~1–2
   cores + ~40 GiB combined and contend the MLX GPU.
4. **Both write into the SAME out-dir** (`dry_start/levelset_resume_state.npz`,
   `levelset_witness_ema_mlx.npz` — present, interleaved last-wins) and their stdout goes
   to DEAD pipes → every artifact in that dir is two-writer garbage; nothing is usable
   for #480 attribution. Delete the dir after the kill.
5. Downstream coupling: the #507 arm's armed chain "waits for sibling trainers to drain"
   (checkpoint 17:01:48Z) — with no supervisor, **that drain never happens**; the 507
   chain waits forever unless the orphans are killed.

## Operator action (exact)

```bash
kill -TERM 14828 21381         # then, if needed after ~5s:
kill -9 14828 21381
rm -rf experiments/results/levelset_n600_drystart3_telemetry480_20260715   # two-writer garbage
```

(The agent's targeted kill was refused twice by the auto-mode classifier
["Interfere With Workloads"] despite the out-dir/parent ownership proof — the classifier
could not see the full argv because macOS `ps` truncates; `psutil.Process(pid).cmdline()`
carries the proof.)

## What is owed after the kill (staged, ready)

- **M1 rerun** (the #480 attribution run, D.3-1): the durable-daemon launcher invocation is
  known-good — the previous "hang" was misdiagnosed buffering. One instance ONLY:

  ```bash
  .venv/bin/python tools/spawn_durable_daemon.py --label m1_drystart3_telemetry480c \
    --log <out>/launcher_daemon.log -- \
    .venv/bin/python tools/launch_witness_run.py \
    --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz --num-pairs 600 \
    --config v9_cgauge_ideal_mod19 --out-dir <out> \
    --purpose "M1 #480 attribution dry-start" --dry-start 3 \
    --dry-start-boot-budget-s 600 --dry-start-per-ep-budget-s 900 --no-dashboard
  ```

  Progress must be watched via FILES (`<out>/dry_start/witness_component_wallclock.jsonl`
  row count), never the buffered launcher log.
- **bf16/fp16 $0 gate** (constraint-1 smoke, complements the sibling's bf16 seam build):
  `tools/bench_bf16_witness_step_20260715.py` — value_and_grad fp32 vs bf16 vs fp16 on the
  real witness step + grad relmax vs fp32. Run on a QUIET GPU only.
- **Lane-band cache A/B receipt** (independent confirmation of the sibling's row):
  `tools/bench_lane_band_cache_20260715.py` measured Δ0.085–0.13 ms/call ⇒ **~0.10–0.15
  s/ep at n600** (two runs, one under contention — order-of-magnitude robust). Confirms
  FEED-509burn: the batch-1c cache is NOT the −40–60 s/ep D.3-3 lever; the +75 s/ep band
  cost lives in the θ-dependent in-graph compose. verdict_scope: INSTANCE (microbench).

## Process signal (for the coherence sweep)

Two live agents ended up on ONE subagent id (`m5_burndown_509`): this instance (respawn,
16:19Z) and a second respawn (16:54Z) that resumed this instance's step-3 checkpoint and
committed its staged verdict-parallel work as batch 3a (`9d3bfc837b`) — correct content,
but the id collision meant neither knew the other was live. De-conflict taken: the 16:54Z
instance keeps the BUILD domain (batch 2/3a/bf16 seam); this instance landed the
MEASUREMENT/LEDGER leg + this incident note and stood down. Root causes to fold into the
respawn protocol: (a) checkpoint rows carry a `pid` — a respawner MUST check
`ps -p <pid>` liveness before resuming an id; (b) launcher stdout block-buffering makes
"log frozen at gate line" indistinguishable from a hang — flush-or-line-buffer the
launcher's gate prints (or always judge by files).
