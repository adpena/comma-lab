# D1 GPU-vs-CPU verdict AGREEMENT probe — n600 (device-forward drift)

**STORES CONSULTED (proactive recall before building):**
CLAUDE.md non-negotiables — MPS/MLX NEVER a score authority (this probe is ADVISORY-tier;
CPU-torch is the advisory authority, `upstream/evaluate.py` the only score) · n600-or-not-evidence
· governed-launch P0 (`tac.admission_guard.assert_governed_admission` + `tools/safe_run.py`) ·
`docs/operating_manual_craft_handoff.md` (do LESS but REAL; verdict-scope ladder; label
MEASURED/INFERRED). MEMORY.md L70 (MLX-GPU bit-identity localized to dup-index atomic scatter =
R UP-backward; fused-R cures it — but the VERDICT is forward-only so the scatter-VJP wall does
NOT apply), L53 (MPS never a score; CPU/CUDA authority), L67 (#205 CE-floor d_seg ~0.005).
Equation module `src/tac/canonical_equations/safe_compile_device_bitidentity_20260708.py` (the
per-{chip,os,mlx,device} bit-identity law this probe extends from compiled elementwise regions
to the full scorer FORWARD). `src/tac/witness_control/gpu_verdict.py` (the device-free anchor/
drift schema; operator 2026-07-08 hybrid design). `experiments/train_witness_realized_through_R_mlx.py`
verdict primitives (`cpu_verdict_d_seg_argmax_batch` / `gpu_verdict_d_seg_argmax_batch` /
`cpu_verdict_d_pose_batch` / `gpu_verdict_d_pose_batch` / `paired_anchor_verdict` /
`load_gt_from_cache`). Live run-1 `experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/launch.sh`
(uses `--async-verdict` ⇒ `gpu_verdict_conflicts` FORBIDS `--verdict-device gpu` ⇒ run-1 runs CPU
verdict only, emits NO paired rows ⇒ this controlled probe is genuinely needed). GT cache
`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` (gt_f0/gt_f1/lstars/margins/gt_poses, n=600).

**Axis:** `[macOS-MLX research-signal]` / `[macOS-CPU advisory]` — NON-PROMOTABLE. **Pointer
contest-CPU 0.19110 UNMOVED** (this is a MEANS: it decides whether v7 may use the GPU verdict as
the in-training advisory SENSOR; it does not move the score).

## The question (operator crux-engineering, 2026-07-08)
Can the MLX-GPU verdict serve as the IN-TRAINING ADVISORY sensor, replacing the ~80-min-stale
CPU-torch anchor cadence with a minutes cadence? **Fit-for-advisory IFF the GPU-vs-CPU-induced
error in the sensor readings (d_seg, d_pose, and the low-margin annulus statistics the
part_frac / within_flip / plateau triggers read) is FAR BELOW the sensor's decision granularity.**

## Method (device-forward-drift formulation — MEASURED substrate)
The trainer's verdict scores WITNESS-rendered frames; the DIFFERENCE between the CPU and GPU
verdict flavours is NOT the witness — it is the frozen-scorer FORWARD numerics (torch-CPU vs
MLX-GPU) on the SAME input frames. The preprocess (resize + last-frame select for SegNet;
interpolate + rgb_to_yuv6 for PoseNet) is the SAME torch `preprocess_input` in BOTH flavours
(see `gpu_verdict_d_seg_argmax_batch` / `gpu_verdict_d_pose_batch`); ONLY the forward kernel
differs. That drift is checkpoint-independent — a property of (scorer weights, input frames,
device), governed at the argmax by LOW-MARGIN pixels.

We measure it on the **n600 GT-reference frames** (`gt_n600.npz`) — the most realistic possible
frames, sitting EXACTLY at the reference separatrix, with cached per-pixel margins so the
disagreement is resolved **by margin bin** (the sensor granularity). On GT frames the CPU
verdict is ~0 by construction (`lstars` / `gt_poses` ARE the frozen CPU-torch argmax / pose), so
**every reported DELTA is the pure CPU→GPU device drift**, and the low-margin-bin disagreement
RATE bounds the induced sensor error at ANY operating point INCLUDING the witness annulus.

Harness: `tools/d1_gpu_verdict_agreement_probe_n600.py` — reuses the EXACT trainer verdict
primitives (no trainer modification), sets the MLX default device to GPU (replicating
`--mlx-device gpu`), chunked-resumable (VBATCH=8), footprint-bounded, routed through
`tools/safe_run.py` (memory governor P0 gate) with `assert_governed_admission`.

## Scope decision (honest — verdict_scope = FORMULATION)
This is the **device-forward-drift formulation**. It does NOT re-render the two frozen witness
checkpoints (mod32cap EMA-BEST + run-1 EMA-BEST). Rationale, stated up front:
1. **No reusable checkpoint→render→verdict driver exists in-tree** (no `tools/*probe_n600.py`
   builds the witness + renders; only GT-frame probes exist). Reconstructing the full compose
   (self-orient + chroma + lane-render-band + structured-init) faithfully is a real
   render-INFIDELITY risk.
2. **Beside the untouchable live run** (pid 63069, MLX-GPU): re-rendering 600 pairs adds heavy
   GPU contention for no gain to the device-drift answer, which is **checkpoint-independent**.
3. The margin-resolved GT measurement is arguably **stronger**: it bounds the sensor error at
   EVERY operating point, not just two witness snapshots.
The witness-frame sensor deltas at the actual witness operating point are the **OWED
confirmation** (reactivation: build the checkpoint→render driver, then re-run this probe on
witness-rendered f0/f1). Also note: the requested "mod32cap ep650" checkpoint does not exist —
that lineage has stage ckpts ep299/726/1000 + EMA-BEST; the frozen EMA-BEST is the valid
coverage point (moot under this formulation, which is checkpoint-independent).

## Pre-registered fit-for-advisory thresholds (WRITTEN BEFORE MEASUREMENT)
Sensor operating point: witness verdict d_seg ~0.005; the sensor reads d_seg trends near ~1e-4;
the annulus part_frac / within_flip are low-margin (margin<1) statistics.
- **mean per-pair |Δd_seg| < 5e-5** (device d_seg drift < ~1% of the 0.005 operating point)
- **low-margin (margin<1) argmax disagreement RATE < 1e-3** (< 0.1% of annulus pixels flip
  between devices — the part_frac / within_flip granularity)
- **mean per-pair |Δd_pose| < 1e-6** AND **max per-pair |Δd_pose| < 1e-4**
- **GPU double-forward bit-identical** (determinism floor) — else the GPU verdict is
  non-reproducible and unfit regardless.
FIT ⟺ all four hold. A FAIL is a real verdict (verdict_scope = formulation: the current MLX-GPU
verdict path on THIS chip); the failing metric is the reactivation target.

Instrument-validity gates: (a) GPU double-forward bit-identity; (b) CPU verdict ~0 vs the cache
(confirms the GT-reference construction / no cache drift).

## Results — BLOCKED by the governed P0 memory gate (honest outcome; NO raw bypass)
The harness is built, ruff-clean, reviewed, committed, admission-wired, and pre-registered — but
the governed launch **could not run**. `tools/safe_run.py` routed the probe to the system memory
governor, which **REFUSED admission** (verbatim):

> `REFUSED (system admission gate — SUM-over-RAM crash guard): projected system-used 143.5 GiB`
> `EXCEEDS adaptive ceiling 66.1 GiB by 77.4 GiB — launching would risk a SYSTEM OOM/jetsam`
> `cascade (current used 75.5 + active-growth 60.1 + new 8.0). REFUSE. [projected=8.0GiB]`

**This is the P0 gate working as designed, NOT a phantom.** The `active-growth 60.1 GiB` is the
governor legitimately reserving the LIVE #205 run's projected peak (`projected_peak_gib ~67.6` —
the n600 self-orient `cf_mx_cache`; current RSS is lower, but the governor reserves the growth
headroom so the live run can grow without a system OOM). No stale reservation files exist; the
number is the real live-run reservation. The adaptive ceiling (66.1 GiB) is already exceeded by
`current used + active-growth` ALONE (135.5 GiB), so **no probe footprint — not even 0 GiB —
would be admissible** while the live run holds the ceiling. Per the task ("if admission refuses,
STOP and report") and the machine-crash P0 non-negotiable (raw-python bypass FORBIDDEN), I
STOPPED. I did NOT set `TAC_ADMISSION_BYPASS_OK` (that would be the forbidden bypass).

Consequently: **the n600 agreement numbers were not measured this session** — there is no
d_seg/d_pose/per-class/flip-disagreement data, and therefore **no fit-for-advisory verdict**.
The honest status is BLOCKED-pending-memory-headroom, verdict_scope = INSTANCE (this launch
attempt), not a finding about the GPU verdict itself.

## Verdict + proposed v7 hybrid cadence
**Verdict: DEFERRED — measurement BLOCKED by the P0 memory governor while the live #205 run
reserves the RAM ceiling.** No claim about GPU-vs-CPU agreement can be made (no data). The
proposed v7 hybrid cadence (documented in the harness for when data lands): IF a future run
measures FIT → GPU verdict at the FAST cadence (`--verdict-device gpu`) + a CPU-torch ANCHOR
(`paired_anchor_verdict`) at the SLOW cadence (checkpoint epochs, `--verdict-anchor-every N`) as
the positive-control sentinel + comparability baseline, with the controllers (nucleus-guard /
ladder-homotopy) kept on CPU authority per `gpu_verdict_conflicts`.

**Reactivation (the harness is ready — one governed command):**
```
.venv/bin/python tools/safe_run.py --label d1_gpu_verdict_probe --projected-gib 8 \
  --rss-mb 9500 --timeout 540 -- \
  .venv/bin/python tools/d1_gpu_verdict_agreement_probe_n600.py --chunk-seconds 460
```
Re-invoke (chunked-resumable) when EITHER (a) the live #205 run completes / frees the ceiling, OR
(b) the operator authorizes a governed slot (e.g. a brief coordinated pause at a stage boundary,
or lowering the live run's reserved projected peak). The probe resumes from
`experiments/results/d1_gpu_verdict_agreement_probe_20260708/probe_state.ckpt.npz`.

## Equations leg
**DEFERRED — no MEASURED drift to anchor yet.** The triality equations leg is required for a
MEASURED durable finding; this session produced NO measurement (admission-blocked), so
registering an `EmpiricalAnchor` now would be a fake empirical claim (NO-FAKE #8: surrogate/
absent-authority as a finding). When the probe runs, the scorer-FORWARD device-drift row extends
`safe_compile_hosc_device_bitidentity_v1` (compiled-region bit-identity → full-forward drift) OR
registers a sibling equation following that module's pattern, in the same commit as the results.
