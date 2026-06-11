# Scorer spectral-sensitivity atlas: CELL-LEVEL PARALLELISM landed; exhaustive daemon relaunched at 12 workers (2026-06-11)

**Authority:** `[macOS-CPU advisory]` / `exact_pair_scorer` -> `mechanism_update_eligible` ONLY
(inherited from the v2 physics; UNCHANGED — the scorer, device, and every cell's value are byte-identical
to the serial run). NOT a score row; does NOT move the canonical frontier pointer (UNMOVED at
0.19109982). Commit: `24e36c469`. Builds on the resumable landing `ad07cb838`
(`scorer_atlas_resumable_multitier_landed_20260611.md`). Implements operator directive 2026-06-11
("SATURATE the M5 Max — the scorer atlas is using ~2.2 of 18 cores").

## The defect this fixes (single-process saturation)

The resumable atlas (`run_resumable_atlas`) was SINGLE-PROCESS. Each cell scores a band-limited
perturbation through the EXACT torch `DistortionNet` on CPU (~1-2s/cell at the smoke density; ~85s/cell at
the exhaustive `n_pairs=12, phases=2` density) and the cells are INDEPENDENT (embarrassingly parallel), but
the serial loop pinned only ~2 of the M5 Max's 18 cores. Exhaustive (6400 cells) projected ~6 days serial.
14 cores sat idle. The fix is cell-level parallelism that leaves headroom for the coexisting capstone daemon.

## What changed (extend, not rewrite — SEARCH-AND-FAMILIARIZE)

Reused the ENTIRE committed physics (band synthesis, `cell_seed_for`, `_aggregate_cell`, `_cell_record`,
the fcntl-locked JSONL append, `load_completed_cells`, `aggregate_atlas_from_cells`, the tier presets, the
progress sidecar, the DONE.marker, the lowering analysis). Only ADDED a worker pool over cells.

1. **`src/tac/analysis/scorer_spectral_atlas_parallel.py`** (NEW) — the worker-pool driver:
   - `run_resumable_atlas_parallel(...)` — same signature surface as the serial runner plus `workers` +
     `raw_path` + `torch_threads_per_worker`. `workers <= 1` transparently delegates to the committed
     `run_resumable_atlas` (so `--workers 1` IS the serial path, the bit-identity reference).
   - **Parent = single writer.** The parent reads the resume skip-set ONCE, enumerates the REMAINING cell
     keys (`enumerate_cell_keys` minus the skip-set), feeds them to a shared `multiprocessing` task queue,
     and is the ONLY process that appends to the JSONL (reusing `append_cell_jsonl` under its existing fcntl
     lock). Single writer => no concurrent-writer contention, and each remaining key is enqueued exactly
     once => no cell computed twice and none skipped.
   - **Workers = compute only.** N worker PROCESSES (`spawn` context — torch CPU + `fork` can deadlock;
     spawn is mandatory) each build their OWN `FrozenScorer` once, memmap the source pairs from the shared
     `source.raw` on the SSD (no 73MB pickle per worker), and pull cell keys, measuring each via
     `measure_cell_by_key` — which rebuilds the SAME `BandSpec`, derives the SAME intrinsic
     `cell_seed_for(...)` seed, and calls the SAME `_aggregate_cell`. A failed cell pushes a `None` so the
     parent fails loud (the JSONL stays valid; re-running RESUMES and retries the failed keys).
   - **Thread caps.** Each worker calls `torch.set_num_threads(torch_threads_per_worker)` (default 1) +
     sets `OMP/MKL/VECLIB/NUMEXPR_NUM_THREADS` so `workers x threads` does not oversubscribe the cores and
     starve the capstone daemon.
   - **`auto_worker_count()`** = `min(12, max(1, physical_cores - 4))`. On the M5 Max (16 physical / 18
     logical) = **12**. Reserves 4 cores for the capstone daemon (~2.5 cores) + the OS.

2. **`tools/measure_scorer_spectral_sensitivity.py`** — `v2-resume` gains `--workers {auto|N|1}` (default
   `auto`) and `--torch-threads-per-worker` (default 1), dispatches to `run_resumable_atlas_parallel`, and
   the ETA print divides by the worker count. NO change to the cell math, the device (`cpu`), or the JSONL.

## The headline NO-FAKE guard: parallel == serial, BIT-IDENTICAL (with a real finding)

`src/tac/tests/test_scorer_spectral_atlas_parallel.py` — 10 tests (9 fast + 1 torch-gated):

- **`test_real_scorer_parallel_bit_identical_to_serial`** (torch-gated, PASSES in 35s) — runs a serial
  (`workers=1`) and a parallel (`workers=2`, REAL worker subprocesses) sweep through the EXACT
  `DistortionNet`, sorts both JSONLs by cell-key, and asserts EVERY cell value is bit-identical. This is the
  authority-surface proof that distributing cells across processes cannot change a value.
- **`test_parallel_jsonl_is_bit_identical_to_serial`** — the same claim through the spawn/queue/single-writer
  machinery on a deterministic content-derived stub scorer (no torch), so the parallel CONTRACT is tested
  in <2s.
- **`test_parallel_writes_every_remaining_cell_exactly_once`** — the JSONL has exactly `total_cells` UNIQUE
  keys (no double-write, none skipped — partition correctness).
- **`test_parallel_resume_then_parallel_is_bit_identical`** — kill a parallel run mid-way, resume in
  parallel, still bit-identical to uninterrupted.
- `test_measure_cell_by_key_matches_serial_iter_cell`, `test_auto_worker_count_leaves_headroom`,
  the serial-delegation + raw-bytes-contract tests.

Full pair (parallel + the existing runner suite) = **25 tests pass** (92s incl. both slow real-scorer
tests). Ruff clean.

### THE REAL FINDING (a genuine NO-FAKE crux, not a constant test)

The first real-scorer parallel-vs-serial comparison FAILED at the 7th decimal: `logit_margin_drop_p10`
came back `-0.39463281` (serial) vs `-0.39463329` (parallel) — a ~5e-7 float32 difference. A 2x2 probe
isolated the cause: **torch CPU float32 reductions are reduction-ORDER-dependent on the thread count**
(the `kthvalue`/sort path behind `logit_margin_drop_p10`). Same-thread runs are bit-identical
(`1-thread A == B` exactly); 1-thread vs 8-thread differ (`-0.39463281` vs `-0.39463138`). The serial
delegation ran unpinned (many threads) while workers ran at 1 thread => divergence.

**Fix (the canonical contract):** the atlas runs at a FIXED torch thread count. `_pin_torch_threads(...)`
pins BOTH the parent baseline measurement AND the serial delegation to the same value the workers use, so
serial == parallel is provably bit-identical. (The drift is ~5e-7, far below contest reporting precision —
each cell is independently a valid exact-scorer measurement either way — but the bit-identical NO-FAKE
claim required the pin.) This finding is itself a mechanism fact: the exact scorer's logit-margin tail
statistics are thread-count-sensitive at the 7th decimal; the resumable atlas now standardizes on a fixed
thread count for reproducible bytes.

## Auto worker count chosen + saturation (measured)

- **`--workers auto` -> 12** workers (`min(12, 16 physical - 4)`), 1 torch thread each.
- **System CPU at steady state: 1427% / 1800%** (~14.3 of 18 cores busy = the saturation target:
  ~12 atlas workers + ~2.5 capstone, ~4 reserved for the OS). Before: ~4.5 of 18 cores (atlas ~2 + capstone
  ~2.5), 13+ idle.

## Measured cells/min — before vs after (exhaustive density, real CLI, real DistortionNet)

| run | cells/min | per-cell | source |
|---|---|---|---|
| serial daemon (old, pid 26486) | ~0.7 | ~85s | 17 cells in ~24 min at `n_pairs=12, phases=2` |
| parallel daemon (12 workers) post-warmup | **9.6** | ~6.3s amortized | 24 cells in 150s (steady state) |

That is a **~13x throughput gain** (12 workers, near-linear minus startup + the capstone's ~2.5-core share
+ memory bandwidth). A small-grid CLI A/B (8 cells, 6 workers, both daemons live) measured serial 6.95
cells/min vs parallel 16.66 cells/min (**2.4x** — startup-dominated at 8 cells, where 6 model-loads aren't
amortized); the 6400-cell exhaustive sweep amortizes the 12 model-loads and reaches the ~13x above.

## Exhaustive daemon RELAUNCHED (detached, parallel, resumed from the JSONL skip-set)

- **Atlas daemon PID:** `54478` (parent ppid `54476` = the `nohup ... ; OUTER_DONE_PARALLEL.marker` bash
  launcher; fully detached from the spawning shell). 12 worker child processes under it (`pgrep -P 54478`).
- **Killed** the old single-process serial daemon (pid `26486`) first — it is resumable, so killing lost
  NOTHING: the JSONL stayed at 17 cells across the kill (verified before/after).
- **RESUMED from the existing JSONL skip-set:** the relaunch log shows `already_done=17` and the parallel
  loop enqueued only the remaining 6383 cells — the 17 already-computed cells (incl. the original 6 + the 11
  the serial daemon added) are skipped, not recomputed. Zero compute lost.
- **Tier:** `exhaustive` (6400 cells), `--workers auto` (=12), device `cpu`, `--progress-every 1`.
- **Work dir (SSD):** `/Volumes/VertigoDataTier/pact/scorer_spectral_atlas_exhaustive_20260611T080857Z/`
  (SAME dir as the killed run — that is what makes it resume). JSONL `atlas_cells.jsonl`; sidecars
  `atlas_progress.json` (heartbeat), `DONE.marker` (inner exit), `OUTER_DONE_PARALLEL.marker` (launcher),
  `daemon_parallel.log`, final `atlas.json`.
- **New ETA: ~11 hours** for the remaining ~6347 cells at 9.6 cells/min (down from ~6 days serial). The
  progress sidecar's `eta_seconds` refines live.
- **Orphaning is APPROVED + safe:** a kill/crash/reboot loses at most the in-flight cells per worker (one
  per worker = at most 12 cells); re-launching the exact `v2-resume --tier exhaustive --workers auto
  --work-dir <same dir>` RESUMES from the JSONL. The thread count is fixed so resumed cells stay
  bit-identical.

To check progress without parsing the JSONL:
```bash
cat /Volumes/VertigoDataTier/pact/scorer_spectral_atlas_exhaustive_20260611T080857Z/atlas_progress.json
```

## Coexistence (no starvation)

- The decisive **capstone daemon (pid `72123`)** is UNTOUCHED and healthy (2:44+ elapsed, ~218% CPU at
  relaunch). The auto worker count's `RESERVED_CORES=4` leaves it its ~2.5 cores; measured combined CPU
  (~14.3/18) confirms neither daemon is starved.

## Disk hygiene

The timing-smoke probe dir (`atlas_parallel_smoke_20260611`) was rebuildable scratch and was removed after
measurement. The exhaustive work dir is the durable resume store (kept). No `/tmp` evidence paths.

## 6-hook wire-in (Catalog #125)

Same as the resumable landing (the atlas is unchanged in content): #1 sensitivity-map ACTIVE (the atlas IS
the per-cell scorer sensitivity map); #3 bit-allocator ACTIVE (design — `analyze_lowering_opportunities`
spend/shed feeds); #6 probe-disambiguator ACTIVE (the w-question + shed-bytes). #2 Pareto / #4 cathedral /
#5 continual-learning N/A (advisory, non-promotable). This landing adds NO new score signal — it makes the
SAME measurement ~13x faster by saturating idle cores.

## Authority / NO-FAKE

Every artifact carries `authority_tier=exact_cpu_advisory`, `promotable=false`,
`mechanism_update_eligible=true`. Parallel produces BIT-IDENTICAL results to serial at the pinned torch
thread count (proven through both the stub and the EXACT `DistortionNet` across real worker subprocesses).
NO MPS. CPU only. The scorer, device, and per-cell math are unchanged; only the orchestration is parallel.
