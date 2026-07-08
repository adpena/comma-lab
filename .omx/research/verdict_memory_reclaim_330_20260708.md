# #330 VERDICT-MEMORY-RECLAIM — subprocess vs cheap-trim, MEASURED [no-triality]

**Date:** 2026-07-08. **Task:** the periodic CPU-torch verdict's ~5-6 GiB fp32/activation transient
ratchets parent RSS instead of returning to the OS. Build the reclaim path; try the CHEAP fix first;
build whichever the measurement supports. Advisory / NON-PROMOTABLE (never a score).

## STORES CONSULTED
- CLAUDE.md §"Local Disk / SSD spill" + §"scale measured+safeguarded"; the #205 OOM forbidden-pattern
  ("Forbidden full-P batched CPU-scorer verdict") — the transient this fix targets.
- `tools/spawn_durable_daemon.py` — the #167 killpg / `start_new_session=True` no-orphan discipline
  (child == session leader ⟹ pid==pgid ⟹ group kill reaches all descendants).
- Trainer `experiments/train_levelset_witness_realized_through_R_mlx.py` — `_verdict_v` /
  `_verdict_dseg_dpose_chunked` / async-verdict thread / `--verdict-batch` (#205) / `--verdict-device`
  gpu HYBRID (compose, don't break).
- Base `experiments/train_witness_realized_through_R_mlx.py` — `cpu_verdict_d_seg_batch` /
  `cpu_verdict_d_pose_batch` / `load_gt_from_cache` (the SAME primitives the child re-uses).
- MEMORY.md L42/L47 (scale-safeguarded; spawned long-runners die ~5min — the parent-death watch).

## THE MEASURED COMPARISON (`tools/measure_verdict_memory_reclaim.py`, real gt_n24, vbatch 8, macOS M5 Max)
Governor-gated (normal pressure). Reduced-n (RSS-return is n-independent in mechanism); each row is a
REAL SegNet+PoseNet forward.

| path | parent ratchet vs baseline | child peak | reclaimed | note |
|---|---|---|---|---|
| baseline | — (0.639 GiB) | — | — | pre-verdict |
| **C subprocess** (killpg child) | **+0.0 GiB** | 5.30 GiB | n/a | parent stays at baseline; child holds the transient, OS reclaims on exit |
| A in-process, no reclaim | **+4.63 GiB** | — | — | the ratchet, confirmed |
| B in-process, cheap reclaim | +4.6 (unchanged) | — | **0.0 GiB** | `gc.collect` + `malloc_zone_pressure_relief` + torch-cache release returns NOTHING on macOS |

**Bit-identity:** subprocess d_seg/d_pose == in-process, per-pair `array_equal` AND mean **bit-equal**
(0.0 abs diff) — same frozen scorers, same preprocess→forward→argmax/MSE, same `--verdict-batch`.

## VERDICT: build the subprocess (cheap fix is INSUFFICIENT on macOS)
The honest simplest-fix (`malloc_trim`/`malloc_zone_pressure_relief`) reclaimed **0.0 GiB** — macOS
libmalloc does not return the torch/numpy transient pages to the OS under pressure-relief, so the
in-process ratchet persists. The subprocess keeps the parent at **exactly baseline** because the child
process (its own session; killpg-reclaimed) holds the whole spike and the OS reclaims it on exit. Both
`reclaim_process_memory()` (cheap, kept for the CUDA/Linux case where the allocator DOES honor trim)
and `run_verdict_in_subprocess()` are shipped; the trainer uses the subprocess.

## WHAT LANDED
- `src/tac/witness_control/verdict_reclaim.py` — `reclaim_process_memory()` (gc + platform allocator
  trim + torch CUDA cache) and `run_verdict_in_subprocess()` (killpg child, tmpfile I/O, fallback).
- `src/tac/witness_control/_verdict_subprocess_worker.py` — child entrypoint; re-loads frozen scorers,
  runs the SAME chunked `cpu_verdict_*`, atomic-writes result json; parent-death watch (getppid poll →
  `os._exit`) so a SIGKILLed parent leaves NO orphan.
- `tools/measure_verdict_memory_reclaim.py` — the 3-way governor-gated measurement harness.
- Trainer: `--verdict-subprocess` (BooleanOptionalAction, **DEFAULT OFF ⟹ byte-identical** to the
  sealed #205 verdict). Wired ONLY into the PLAIN cpu-device path (nucleus/annulus stay in-process —
  they need the realized argmax maps the subprocess boundary does not return; gpu-verdict/anchor
  untouched). Fail-open to the in-process chunked verdict.
- Tests: `src/tac/witness_control/tests/test_verdict_reclaim.py` — 11 tests (cheap-reclaim shape +
  score-neutrality, platform trim method, ragged/empty input raises, **killpg reaps whole session**,
  worker error→rc4 no-fabrication, parent raises-not-fabricates, **slow: bit-identity + parent RSS
  returns to baseline**). All pass; ruff F clean; trainer py_compile OK.

## INTERACTIONS VERIFIED
- **EMA-shadow serialization cost = ZERO:** the boundary is at the ALREADY-RENDERED uint8 frames (the
  parent renders from the shadow as before); the shadow never crosses the process boundary.
- **async-verdict × subprocess:** the async daemon thread stays the coordinator; the child replaces the
  thread's scorer compute (thread blocks on `proc.wait()`, training GPU never blocks). Valid combo is
  async+cpu+subprocess (gpu is refused with async anyway).
- **verdict_anchor / gpu-verdict:** untouched — `_verdict_subprocess_on` is False when device==gpu.
- **crash-safety:** child is session leader (pid==pgid); parent `killpg`s on every exit/timeout; child
  self-exits on parent death.

## HONEST CAVEATS
- **Disk I/O:** the boundary serializes the rendered frames to an npz (~7 GiB at n600 per verdict,
  removed in `finally`). Trades RSS ratchet for transient SSD write at eval cadence — acceptable given
  the SSD tier + the multi-GiB RSS it saves; a future pipe/shm boundary could remove the disk hop.
- **Scope:** wired for the plain path only. Extending nucleus/annulus needs the child to also return the
  realized argmax maps (bigger payload) — deferred until a run needs subprocess + those telemetries.

## DSL / DEFAULT (observability-vs-lever) — council decides inclusion
Score-neutral by bit-identity proof ⟹ this is an INFRASTRUCTURE knob, not a score-lever, so it is a
plain argparse `typed_config` flag, not a swept DSL `Lever`. It COULD default-ON (the measurement
supports it on macOS), but recommend keeping DEFAULT OFF until a governed n600 run confirms the disk-hop
cost is acceptable at cadence; council decides promotion to default-ON.
