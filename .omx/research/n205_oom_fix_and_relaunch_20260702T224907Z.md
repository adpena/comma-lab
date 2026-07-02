# #205 OOM: root cause (MEASURED), fix, memory-preflight self-protection — LAUNCH HELD

**UTC:** 2026-07-02T22:49:07Z · **git (pre-commit):** 3fabbb6092e5 · **axis:** `[macOS-MLX research-signal]`
NON-PROMOTABLE (infra; pointer 0.19110 UNMOVED) · **evidence:** `experiments/results/n205_oom_probe_20260702T214441Z/`

> **⛔ LAUNCH STATUS — #205 RELAUNCH IS HELD (operator directive 2026-07-02).** Do NOT spawn the
> multi-day run. The operator is waiting for (1) a REAL, optimal, deep-math-grounded + tested +
> measured + calibrated POSE implementation (NOT the current naive warp-real-luma carrier with the
> ~0.0304-rate real keyframes) AND (2) byte-close PROPERLY engineered + confirmed (the levelset
> inflate/eval can't yet reproduce the pose-carrier decode or count the keyframe payload → any
> pose-carrier row today is pose-blind + under-counts rate). This ledger's OOM fix + memory-preflight
> STAND READY for when the launch is un-held; the launch itself waits. No GPU spend fired.

## TL;DR
The prompt's assumed diagnosis (accum-loop MLX graph not `mx.eval`'d; store a "7.3 GB fp32 pose
keyframe table" as uint8) is a **MISDIAGNOSIS** — I confirmed this by measurement and did NOT ship a
fake fix. The **real, MEASURED** OOM driver is the **advisory verdict inference**: `realized_verdict()`
runs the CPU SegNet + PoseNet over **ONE 600-wide torch batch**, whose fp32 cast +
EfficientNet-B2/FastViT-T12 activations spike **+66 GiB**, on top of the resident **~41 GiB** self-orient
`cf_mx_cache`. That is the 90 GiB that tripped the `safe_run --rss-cap 90000` guard before the first
checkpoint. **Fix = chunk the verdict inference** (`--verdict-batch 32`, default-on) → verdict transient
**+66 → +5.6 GiB** (n600 peak ~90 → ~68 GiB projected), **score-neutral** (eval-mode BN is batch-size
independent; the verdict is advisory + never read into training). Plus a **launcher memory-preflight**
that projects peak RSS from the emitted flags and REFUSES a known-OOM config (this is the structural
self-protection the launcher lacked — it only gated COMPUTE at B=8, never MEMORY at n600).

## 1. Root cause — CONFIRMED BY MEASUREMENT (not the prompt's assumption)

Failed run `experiments/results/levelset_n600_witness_20260702T210653Z`:
`SAFE_RUN status=oom exit=137 peak_rss=90300MiB elapsed=612s limit_rss=90000MiB`. Last printed line
was `island_amplify`; **neither** the baseline-verdict row **nor** any epoch row printed → the OOM was
BETWEEN `island_amplify` (init) and the first verdict, i.e. during the resident cf-cache build +
the synchronous baseline `v0 = realized_verdict()` — **not** the accum loop.

### 1a. The accum loop is NOT the leak (falsifies the prompt's assumption)
`experiments/train_levelset_witness_realized_through_R_mlx.py:3079` already `mx.eval(loss, grads)` +
`mx.eval(accum)` **per pair** and `mx.clear_cache()` per epoch. **Measured (n64, `TAC_MEM_PROBE=1`,
`--mlx-cache-clear-accum 0`):** RSS **flat at 11.51 GiB** (active 4.62 / cache 7.95) across all 8
accum-batches × 2 epochs — the buffer pool is already reused batch-to-batch; **zero** per-accum growth.
So the leak is **n-dependent**, not per-batch.

### 1b. Two n-dependent carriers (measured)
- **`cf_mx_cache`** (`--self-orient`): a **per-pair** MLX coord-feats tensor of size P at render
  resolution. Measured active: n64 → 4.62 GiB (0.072/pair), n300 → 21.04 GiB (0.070/pair) → **~41 GiB @
  n600**. Resident floor (in-place rebuild holds it steady; `lever_b_levelset_generator.py:671`).
- **verdict inference spike** — the killer. `realized_verdict()`
  (`train_levelset_witness_realized_through_R_mlx.py`) renders all P pairs then calls
  `cpu_verdict_d_seg_batch` / `cpu_verdict_d_pose_batch` (`train_witness_realized_through_R_mlx.py:539/555`)
  with **all P at once**. Inside: `torch.from_numpy(arr).float()` on `(N,2,3,874,1164)` = **14.6 GiB fp32
  @ N=600**, then EfficientNet-B2 / FastViT-T12 forward at batch 600 → tens of GiB of activations.
  (`--verdict-pairs 0` in the sealed config → vpairs = **all 600**.)

### 1c. Isolated micro-probe — the smoking gun (`verdict_mem_microprobe.py`, `resource.getrusage` peak, N=600)
| verdict path | peak RSS | Δ over baseline | d_seg | d_pose |
|---|---:|---:|---:|---:|
| **UNCHUNKED (single 600-wide batch)** | **70.65 GiB** | **+66.18** | 0.994764 | 0.096572 |
| CHUNKED vbatch=32 (the fix) | 10.03 GiB | +5.56 | 0.994764 | 0.096572 |
| CHUNKED vbatch=8 | 10.13 GiB | +5.66 | 0.994764 | 0.096573 |

+66 GiB verdict spike **+ 41 GiB resident cf_mx_cache ⇒ >90 GiB ⇒ the OOM.** Fix bounds the spike to
~+6 GiB (**−60 GiB**). d_seg **bit-identical**; d_pose identical to 6 dp @ vbatch=32 (BLAS batch-tiling
noise ~1e-6 at vbatch=8, far inside the 0.9997 parity bar; advisory-only).

### 1d. The prompt's "uint8 keyframe table" premise does NOT apply (no fake fix)
`gt.gt_f0/gt_f1` are **already uint8** (`train_witness_realized_through_R_mlx.py:470-471`); the f0
keyframe is converted fp32 **transiently per-call** (`...:1737`, "no P-length fp32 cache"). The
`--pose-carrier-residual-mode table` residual is a per-pair **(P,6)** twist (~14 KB), NOT keyframes.
There is no 7.3 GB fp32 keyframe table to convert → I did **not** ship a no-op "fix" here (NO-FAKE).

## 2. The fix (all score-neutral)

1. **Verdict chunking (THE fix).** New module-level `_verdict_dseg_dpose_chunked(...)` runs the CPU
   scorers in `--verdict-batch` (default **32**) pair chunks; wired into BOTH `realized_verdict()` and
   the async `_verdict_from_snapshot()`. `--verdict-batch 0` restores the pre-fix single-batch path
   (for the A/B). **Score-neutral:** eval-mode BN uses RUNNING stats (batch-size independent); argmax
   per-pixel; MSE per-pair; the verdict is ADVISORY (never read into training).
2. **Accum-loop `mx.clear_cache()` (secondary hygiene, NOT the OOM driver — labeled honestly).**
   `--mlx-cache-clear-accum` (default 1) returns the Metal buffer pool to the OS every N accum-batches
   (also on spike-skips). The pool was already flat (1a), so this is a bounded-pathology guard, not the
   fix. `clear_cache` frees only pooled (already-freed) buffers → cannot change compute.
3. **Instrumentation (default-off).** `TAC_MEM_PROBE=1` logs per-accum-batch + cf-build + v0-verdict
   RSS / MLX active·cache·peak (`_rss_gib`, `_mlx_mem_gib`). Pure observability → bit-identical.

## 3. Score-neutrality proof + a KEY determinism finding

**Both changes live OUTSIDE `value_and_grad`** (verdict = advisory readout; clear_cache = pool hygiene)
→ training loss is bit-identical **by construction**. Empirical corroboration + a crucial control:

- n32 loss A/B, `--mlx-cache-clear-accum` **0 vs 1**: ep1 ep_loss 225.81 vs 225.985 — a small diff.
- **CONTROL — SAME config twice** (clear-accum=1 run c1 vs c1b): ep_loss **225.985 vs 226.02**;
  d_seg 0.079575 vs 0.079582. **Identical configs diverge by the same magnitude.**

⇒ The A/B divergence is the **pre-existing MLX-GPU run-to-run nondeterminism** (memory
`mlx_gpu_not_bit_identical_crossprocess_bitexact_proof_cpu_locked_20260702`), NOT the fix. On MLX-GPU
you **cannot** prove bit-identity by re-running (the substrate isn't bit-reproducible cross-process);
neutrality is established by the code-path argument + the control showing the fix stays within the
GPU-nondeterminism floor. (Determinism authority remains the CPU numpy-fp32 byte-close verdict.)

## 4. Self-protection — launcher memory-preflight (the structural fix)

`tools/witness_memory_preflight.py` (pure, tested) projects peak RSS from the EMITTED `launch.sh`
using the MEASURED constants above:
`peak ≈ 15 (fixed) + cf_mx_cache(P, self_orient) + gt(P) + verdict(verdict_batch)`.
Wired into `tools/launch_witness_run.py` as step (b1): **REFUSE** (rc=4) if projected peak > a
control-plane-safe RAM fraction (default **0.70**; `--skip-mem-preflight` overrides). This closes the
gap the throughput gate left (it measures COMPUTE at B=8; it never projected MEMORY at n600).

Projection validation (128 GiB box):
- sealed n600 **chunked** (verdict-batch 32, the fix): **67.6 GiB → SAFE** (ceiling 89.6).
- sealed n600 **unchunked** (verdict-batch 0, the #205-original OOM config): **127.6 GiB → REFUSE (rc=3)**.

Conservative over-estimate (fail-closed); `safe_run --rss-cap-mb` stays the runtime backstop.

## 5. Tests + gates
- `src/tac/tests/test_witness_memory_preflight.py` — 13 tests (refuse unchunked / pass chunked /
  self-orient off / scaling / launch.sh parse / CLI rc / constants-lock-to-ledger).
- `src/tac/tests/test_levelset_verdict_chunking.py` — 9 tests (chunked == unchunked mean-exact across
  vbatch ∈ {1,7,8,32,64,599,600,601}; call-width ≤ vbatch; every pair scored once).
- `src/tac/tests/test_launch_witness_run.py` — 11 existing, still green.
- ruff F821 clean on all edited files; end-to-end `--dry-run` prints the SAFE projection.

## 6. Memory budget WITH the fix (n600, projected)
`~15 fixed + 41 cf_mx_cache + 3.4 gt + 6 verdict ≈ 68 GiB` peak. Coexists with the parallel Track B
jobs (~13 GiB) on the 128 GiB box (~81 GiB total). The **~41 GiB cf_mx_cache is the inherent
self-orient floor**; if tighter coexistence is later needed, levers (future, unmeasured): fp16 cf-cache
(halves it, needs a forward-parity check), or a smaller `--verdict-pairs` subset. Not needed now (launch held).

## 7. Files
- `experiments/train_levelset_witness_realized_through_R_mlx.py` — `_verdict_dseg_dpose_chunked` +
  `--verdict-batch` (default 32) + `--mlx-cache-clear-accum` (default 1) + `TAC_MEM_PROBE` instrumentation.
- `tools/witness_memory_preflight.py` — projection + CLI (new).
- `tools/launch_witness_run.py` — step (b1) memory preflight wire-in + `--mem-preflight-safe-frac` /
  `--skip-mem-preflight`.
- `src/tac/tests/test_witness_memory_preflight.py`, `src/tac/tests/test_levelset_verdict_chunking.py` (new).
- Evidence: `experiments/results/n205_oom_probe_20260702T214441Z/` (microprobe + A/B logs).

## 8. When the launch is UN-HELD (checklist for the future agent)
1. Confirm (1) real optimal pose + (2) byte-close reproduce+count are DONE (operator gates).
2. `.venv/bin/python tools/witness_memory_preflight.py --launch-sh <launch.sh> --strict` → expect SAFE.
3. Launch via `tools/launch_witness_run.py --config sealed_205 ... --epochs 1000` (mem-preflight now
   auto-refuses an OOM config); keep `--rss-cap-mb ~100000` (leave ~28 GiB headroom), `--ckpt-every 25`,
   `--stage-checkpoints`, seed 0. VERIFY a checkpoint on disk before declaring "launched".
