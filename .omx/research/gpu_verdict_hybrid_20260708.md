# GPU (MLX) verdict device + CPU-torch positive-control ANCHOR (HYBRID) — 2026-07-08

Operator directive: *"Shouldn't the verdict be run on gpu now that it's deterministic and way
faster? Even if launched in a separate process temporarily the contention is probably worth it."*
Main-orchestrator design law (binding): **HYBRID** — GPU verdict at the FAST cadence + CPU-torch
ANCHOR at a SLOW cadence, because the CPU-torch verdict is the independent positive-control
sentinel (training gradients already flow through the MLX scorer; a GPU-only verdict would fold
the instrument into what it measures) AND every prior trajectory baseline is a CPU-verdict number
(comparability). `[no-triality]` (DSL + equations legs untouched; DSL lever/cadence ARE the
triality treatment for this build).

## STORES CONSULTED
- CLAUDE.md: "MPS/MLX NEVER a score authority" · "MLX portable-local-substrate authority" (advisory,
  NON-PROMOTABLE) · the #205 n600-verdict-OOM law (`_verdict_dseg_dpose_chunked`, verdict-batch 32
  MANDATORY) · "'Off' is a tracked queue" (observability additive-safe) · the confound
  immune-system (positive-control sentinel) · "Substrate MUST be at OPTIMAL FORM / measured-scored".
- MEMORY.md L70 (`mlx_gpu_crossprocess_nondeterminism_v1`): fused-R cures the dup-index atomic-scatter
  wall; the wall is **ONE op class = reference-R UP-BACKWARD** (dup-index scatter VJP); per-config
  probe owed. L53 (MPS never a score; CPU/CUDA authority, local MLX-GPU good). L38 (n600-or-not-evidence).
- Trainer verdict path: `experiments/train_witness_realized_through_R_mlx.py::cpu_verdict_*` +
  the levelset `_verdict_v` / `realized_verdict` closures.
- `tools/mlx_gpu_determinism_probe.py` (OPS list + `--composite`); `tools/safe_run.py` admission gate.

## What was built (default-OFF, byte-identical when cpu)
- **`--verdict-device {cpu,gpu}`** (default `cpu`) + **`--verdict-anchor-every N`** (default `0`) in the
  levelset trainer. `cpu`+`0` = today's byte-identical CPU-torch authority.
- **GPU primitives** (base trainer): `gpu_verdict_d_seg_argmax_batch`, `gpu_verdict_d_pose_batch`,
  `gpu_verdict_dseg_dpose_chunked`, `paired_anchor_verdict`. CONFOUND CONTROL: the preprocess
  (resize / last-frame / rgb→yuv6) is done by the SAME torch `preprocess_input` the CPU verdict uses
  ⇒ bit-identical; the ONLY CPU↔GPU difference is the forward kernel numerics — exactly the drift the
  anchor measures. Chunked (verdict-batch 32) — the #205 OOM law applies to the GPU path too.
- **Anchor**: every Nth gpu verdict ALSO runs the CPU-torch verdict on the SAME rendered frames and
  emits a paired `{stage:verdict_anchor}` DRIFT row `{d_seg_gpu/cpu, d_pose_gpu/cpu, d_seg/pose_delta,
  argmax_flip_disagreement_count, max_abs_dpose_delta}`, both axes labelled `[macOS-MLX advisory]` /
  `[macOS-CPU advisory]`, `promotable=False`.
- **FAIL-CLOSED guard** (`gpu_verdict_conflicts`): `--verdict-device gpu` is REFUSED with
  `--async-verdict` (MLX off the main thread races the training GPU stream) and with
  `--curriculum-nucleus-guard` / `--ladder-island-homotopy` (both feed the verdict argmax INTO
  training — they must stay on CPU authority). The gpu path never feeds a training decision on MLX
  numbers; best-ckpt selection under gpu is advisory (cross-checked by the anchor).
- **Telemetry**: every verdict row gains an additive `verdict_device` field (presence-gated).
- **DSL** (`curriculum_dsl`): `VerdictCadence` gains `verdict_device`/`verdict_anchor_every`
  (flags + validate, incl. gpu+async refusal); `VerdictDevice(anchor_every)` Lever factory (A/B
  activation surface). Registry `completeness().unmapped` holds neither flag (covered).
- **typed_config**: `TypedWitnessConfig.verdict_device`/`verdict_anchor_every` (validated; compiled
  into `base` in `to_program`; double-set + bad-device guards).
- Pure helpers in `src/tac/witness_control/gpu_verdict.py` (cadence / flip-counter / row-schema /
  conflict guard) — device-free, unit-testable at $0. 25 tests pass; ruff F clean; existing verdict
  + typed_config suites green.

## Measurement 1 — GPU forward determinism (MEASURED, decisive) ✅
`tools/mlx_gpu_determinism_probe.py --device gpu --n 5` over the **forward / verdict-relevant** ops
(the scorer verdict is inference-only): `conv2d_s2, conv2d_grouped_s2, matmul_square, matmul_bigK,
gemv_bigK, softmax_big, sum_reduce, mean_axis, elementwise`.

**Result: 9/9 ops `cross_process_identical=True` (unique_hashes=1), N=5 processes each.**

The GPU verdict FORWARD is cross-process deterministic. The L70 nondeterminism wall is BACKWARD-only
(dup-index atomic-scatter in reference-R UP-backward), which the forward verdict never touches — so
GPU determinism holds for the verdict WITHOUT needing `--fused-r-kernel` (that lever cures the
backward/training path, not the inference verdict).

## Measurement 2 — n600 GPU-vs-CPU verdict agreement (DEFERRED, governor REFUSE) ⛔
Target: on the mod32cap ep650-best EMA shadow
(`experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z/levelset_witness_ema_BEST.npz`,
d_seg 0.003366), run the n600 verdict on BOTH devices on the SAME rendered frames (a single
`--verdict-device gpu --verdict-anchor-every 1` resume-verdict process yields it directly, and
exercises the new code end-to-end) — reporting d_seg/d_pose both devices + argmax flip-disagreement +
wall-clock.

**Blocked, honestly: `tools/safe_run.py` REFUSED (system-admission SUM-over-RAM crash guard):**
`projected system-used 154.2 GiB EXCEEDS adaptive ceiling 117.8 GiB by 36.4 GiB (current 71.7 +
active-growth 30.5 + new 52.0). REFUSE.` The live run-1 (pid 63069, ~40–68 GiB) plus a ~52 GiB verdict
process would risk the P0 machine-crash class. Per the "NEVER risk the live run / respect a REFUSE"
non-negotiable, measurement 2 is DEFERRED — it must run when run-1 is idle or on a second box (n600 is
non-negotiable; no subset). First action when run-1 idles: the resume-verdict process above (drop
`--async-verdict`; it conflicts with the gpu guard).

## Default recommendation
**Ship BUILT + DEFAULT `--verdict-device cpu` (byte-identical). Recommend `gpu` + anchor as the v7
default = `council_pending`, GATED on measurement 2.** Rationale: determinism is confirmed (measure 1)
and the design makes preprocessing bit-identical so the CPU↔GPU gap is purely forward-kernel numerics —
a sound fast monitor IN PRINCIPLE. But promoting `gpu` to the v7 DEFAULT is a decision that needs the
measured n600 agreement number (flip-disagreement magnitude + Δd_pose + speedup); asserting it now would
be a surrogate-not-authority claim. So: default stays cpu; the lever + anchor are wired and duty-to-
measure-queued (`VerdictDevice`), and the first idle-window action measures the agreement, after which
the council flips the default if the drift is small/stable. NON-PROMOTABLE regardless of device — only
a byte-closed `upstream/evaluate.py` exact row moves the pointer.
