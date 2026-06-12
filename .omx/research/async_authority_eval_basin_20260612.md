# Async authority eval for the torch-vehicle basin — design, no-regression proof, overlap measurement, relaunch

**Date:** 2026-06-12
**Subagent:** async-eval-basin-20260612
**Scope:** non-blocking CPU authority eval for the full-MPS base_ch=20 n600 basin
(`torch_vehicle`). $0 local, no GPU, no PR. `[macOS-CPU advisory]` NON-PROMOTABLE
(the basin GATES toward S≈0.131; it is not itself a contest score). Frontier
UNMOVED (0.19109982) — this is an enabler (throughput), not a score row.

## The problem (measured, not asserted)

The full-MPS basin (`experiments/results/torch_vehicle_full_mps_basin_bc20_n600`,
pid 59649) trains on the Apple GPU at **~13.5 s/epoch** but its torch-CPU authority
eval every 10 epochs (byte-close an archive + exact d_seg/d_pose/rate/score over
600 pairs on CPU) **BLOCKS the training loop**. From the live trajectory
(`dt` = wall-clock delta per epoch):

| epoch | kind | dt |
|------:|------|---:|
| 8 | train | 14.9 s |
| 9 | train | 15.1 s |
| **10** | **EVAL** | **698.9 s** |
| 11 | train | 14.9 s |
| 19 | train | 14.4 s |
| **20** | **EVAL** | **672.3 s** |
| 21 | train | 12.7 s |

Each eval epoch is **~11–11.6 min** — ~50× a normal epoch — and it freezes the
MPS GPU the whole time (CPU work serialized into the GPU loop). With eval every 10
epochs that amortizes to ~+67 s/epoch, inflating ETA to ~28 h. The eval is CPU,
the training is GPU/MPS: they should OVERLAP, not serialize.

## The implementation (async authority eval)

In `src/tac/torch_vehicle/driver.py`. The eval was refactored so the SAME eval
runs sync OR async on a point-in-time snapshot:

1. **`_snapshot_ema(rt)`** — main-thread, cheap. Deep-copies the EMA shadow
   (`ema_decoder.state_dict()` + `ema_latents`) to CPU tensors. This is an
   immutable `_EvalSnapshot`. After it returns, training may keep mutating the
   live (MPS) EMA shadow without racing the eval.
2. **`_eval_snapshot(snap, spec, stage_index, snapshot_epoch)`** — the heavy CPU
   eval, IDENTICAL math for sync and async: `build_archive` (int8 quant) →
   `parse_archive` → `exact_eval` on the CPU AUTHORITY (never MPS). Best-tracking
   and the `best/` dir write happen under `_eval_lock`. Pure-eval reads only the
   immutable snapshot, so it is thread-safe by construction.
3. **`_schedule_async_eval(...)`** — snapshots NOW (main thread) and spawns ONE
   daemon `threading.Thread` worker that runs `_eval_snapshot` + records the eval
   telemetry row (tagged with the snapshot epoch) + logs. Returns immediately;
   training continues.
4. **One-in-flight throttle** — if a worker is still alive when a new eval epoch
   arrives, SKIP (log + `_skipped_evals += 1`). At ~150 s between eval epochs and
   an ~11-min eval, early evals overrun and self-throttle; this is expected and
   correct (no pile-up).
5. **`_join_async_eval()`** — at run completion (before the DONE marker) the
   in-flight worker is JOINED so the final BEST + last eval row land.

The launcher gains `--async-eval` (`experiments/launch_split_by_head_basin.py`),
threaded into `TorchVehicleConfig.async_eval`. **Default OFF** — the sync path is
byte-identical to the legacy run (one combined train+eval row, `evaluated=True`).

### Sync vs async telemetry rows
- **Sync** (default): the eval row IS the train row (loss/lr + d_seg/d_pose/score),
  unchanged from before.
- **Async**: the train row is recorded immediately (train-only, `evaluated=False`);
  the eval row lands LATER from the worker, `evaluated=True`, `loss=NaN`,
  `extra={"async_eval_row": true, "snapshot_epoch": N}`. The dashboard reads both.

## No-regression proof (sync == async, full precision)

`src/tac/torch_vehicle/tests/test_async_eval.py` — 7 tests, all pass (6.6 s):

- **`test_async_eval_numbers_equal_sync_eval`** (THE proof): train a real EMA
  shadow, take ONE snapshot, eval it via the sync inline path and via the async
  path's exact call. Every authority field is **bit-for-bit equal**:
  `d_seg == d_seg`, `d_pose == d_pose`, `rate == rate`, `score == score`,
  `archive_bytes == archive_bytes`. It's the IDENTICAL computation on the
  IDENTICAL snapshot — only the thread differs.
- **`test_async_full_thread_equals_inline`**: the eval routed through the ACTUAL
  background-thread scheduler + join writes the SAME numbers to telemetry as the
  inline call (proves the worker wiring is faithful, not a different code path).
- **`test_snapshot_decouples_from_live_weights`**: violently mutating the live
  EMA shadow AFTER the snapshot does NOT change the snapshot's eval (point-in-time
  copy — the background eval cannot race the training loop). A fresh snapshot of
  the mutated weights gives a different eval (the snapshot really captures live
  weights, not a constant).
- **`test_one_eval_in_flight_skips_overcadence`**: a second eval scheduled while
  the first is alive is SKIPPED + counted; after join, scheduling is allowed again.
- **`test_join_writes_final_eval_row`**: a full async run joins the worker so the
  eval row (snapshot-epoch-tagged) + best land before the DONE marker.
- **`test_sync_mode_is_byte_identical`**: with `--async-eval` OFF the run produces
  the legacy combined train+eval rows (no `async_eval_row` marker, real loss).
- **`test_best_tracking_uses_snapshot_epoch`**: BEST + the eval row are tagged
  with the SNAPSHOT epoch, not the (later) epoch training is on when the eval ends.

Full torch_vehicle suite: **40 passed** (no regression in the refactored eval).
Ruff: clean.

## Thread vs subprocess decision (MEASURED — thread overlaps)

The eval's dominant cost is the 600-pair CPU EfficientNet forward, which runs in
torch's MKL/C++ kernels that **release the GIL**. `build_archive` / `parse_archive`
are seconds. So a thread should overlap. I MEASURED it (`/tmp/overlap_bench3.py`)
with faithful proxies: MPS matmul "training" + CPU conv-heavy "eval" forward
(EfficientNet-class). Over a 25 s concurrent window:

| metric | baseline | under concurrent bg eval |
|---|---:|---:|
| training rate | 55.2 iter/s | **49.3 iter/s (−11%)** |
| eval throughput | 0.15 fwd/s (solo) | **0.12 fwd/s (−20%)**, 3 forwards completed |

**Verdict: the thread OVERLAPS.** Both proceed concurrently — training keeps
running at ~89% of full rate while the eval makes real progress (~80% of solo
throughput). The CPU eval and the MPS-dispatched training share the machine; the
~11% training cost is the bg eval competing for CPU cores during its (few-second)
non-MKL Python portions + memory bandwidth. **No subprocess fallback needed** —
the simpler thread is correct here. (Had inflation been ≥50% / eval starved, the
plan was a multiprocessing worker serializing the snapshot to a temp file; the
measurement makes that unnecessary.)

## New s/epoch + ETA

- **Before (sync):** non-eval epochs ~13.5 s; eval epochs ~673–699 s (blocking).
  Amortized over the eval-every-10 cadence: ~+67 s/epoch → ETA ~28 h.
- **After (async):** the eval epoch is no longer blocking — it becomes a normal
  ~13.5 s training epoch that ALSO snapshots (negligible) and spawns the worker.
  During the ~11-min eval the ~44 concurrent training epochs run at the measured
  ~+11% inflation → **~15 s/epoch effective** (vs 13.5 s clean). Eval rows land
  asynchronously ~44 epochs after their snapshot (acceptable — the trajectory is
  snapshot-epoch-tagged so attribution is correct, and BEST is still driven by the
  authority eval, just a few epochs late).
- **Net: effective ~15 s/epoch vs the sync ~80 s/epoch amortized → ETA collapses
  from ~28 h to roughly ~5–6 h** for the same epoch budget (a >4× wall-clock cut),
  with the authority numbers byte-for-byte unchanged.

## Relaunch (resumes from epoch 29 checkpoint)

The sync basin (pid 59649) was killed and relaunched on the SAME out-dir with
`--async-eval` (resumes from the durable epoch-29 checkpoint — minimal loss,
preserves dashboard + trajectory continuity), detached + durable:

```bash
nohup .venv/bin/python -u experiments/launch_split_by_head_basin.py \
  --no-split-by-head --train-device mps --device cpu \
  --base-channels 20 --n-pairs 600 --eval-every 10 \
  --checkpoint-every-epochs 1 --ema-decay 0.999 \
  --out-dir experiments/results/torch_vehicle_full_mps_basin_bc20_n600 \
  --seed 0 --async-eval \
  >> experiments/results/torch_vehicle_full_mps_basin_bc20_n600/launch_nohup.log 2>&1 < /dev/null & disown
```

Dashboard (pid 66519, port 8765) + capstone cross-check daemon (pid 42035) were
NOT disturbed; the dashboard displays the async eval rows.

## 6-hook wire-in (Catalog #125)
1. sensitivity-map — N/A (throughput infra, no per-axis byte savings).
2. Pareto — N/A (no new constraint).
3. bit-allocator — N/A.
4. cathedral autopilot — N/A (not archive-deployable; a training-loop optimization).
5. continual-learning posterior — N/A (no empirical score anchor; advisory).
6. probe-disambiguator — N/A (no competing interpretations; the thread-vs-subprocess
   question was resolved by direct measurement, recorded above).

`research_only` framing: this is a throughput enabler for an in-flight advisory
basin; it does not move the exact frontier. mission=frontier_breaking_enabler.

## Authority discipline
The async eval is the SAME torch-CPU authority eval — MPS is NEVER the eval
(`device='cpu'`, `__post_init__` rejects MPS as authority). The eval numbers stay
`[contest-CPU advisory]` NON-PROMOTABLE until the byte-closed archive is run
through `upstream/evaluate.py`. The basin GATES toward S≈0.131; a sub-frontier
basin is not itself a paired contest-CPU+CUDA exact eval.
