# 2026-04-03 bat00 side-lane experiment design

## purpose

Use bat00 as a speculative side lane for **runtime/profiling and environment scouting**, while keeping all claimed competition scores on the local official CPU scorer path.

## experiment question

Can bat00 provide useful throughput, runtime, or CUDA-adjacent insight that helps the mainline without becoming a source of non-authoritative score claims?

## constraints

- Do not replace the local authoritative CPU scoring path.
- Any bat00 scorer number must be clearly labeled as side-lane / non-authoritative unless the exact official path is reproduced and compared carefully.
- Prefer reproducible setup via WSL + uv.

## phase 1 — environment validation

1. Verify WSL Ubuntu toolchain:
   - `python3`
   - `git`
   - `uv`
   - `ffmpeg`
   - `git-lfs`
2. Record GPU visibility:
   - `nvidia-smi`
3. Capture baseline runtime metadata:
   - CPU model
   - GPU model
   - driver / CUDA version

## phase 2 — non-authoritative runtime experiment

Run a **single promoted config** on bat00 using the same repo snapshot and compare:

- encode wall-clock time
- inflate wall-clock time
- total scorer wall-clock time
- archive bytes
- score (recorded, but labeled side-lane)

Use the currently promoted honest config as the test subject:
- `448x336 / medium / 23 / keyint=32 / bframes=4 / ref=4`

## phase 3 — decision rule

Promote bat00 from "infrastructure only" to "active side lane" only if it gives one of:

- materially faster experiment throughput
- useful CUDA/runtime evidence for a future mainline lane
- reproducible remote benchmark data that strengthens the writeup

Do **not** promote based on convenience alone.

## artifacts to retain

- remote setup log
- runtime timing log
- side-lane result JSON
- note in `reports/timeline.jsonl`
- note in `reports/writeup_working.md`

## explicit non-goal

bat00 is **not** the source of authoritative claimed scores for the competition leaderboard path.

## approved sequence

1. Runtime profiling benchmark suite
2. JAX/CUDA surrogate setup

The runtime suite comes first because it strengthens the writeup immediately and helps decide whether deeper systems work is justified.
