# Compute-facet MLX/Metal profile + Wave-D Metal-kernel build plan (MEASURED)

`[macOS-MLX/Metal engineering — advisory]` **means≠ends:** this is throughput
tooling/design for the #205 witness run + the #212 Metal-kernel campaign. The
canonical frontier pointer **0.19110 is UNMOVED** and moves only on a byte-closed
exact-eval n600 row. MLX/Metal is a gradient/compute substrate, **never** a score
authority (numpy-fp32 CPU / contest-CUDA only). Every timing below is measured on
this host; every speedup is labeled MEASURED (ran it) vs ESTIMATED (derived).

- **Date (UTC):** 2026-07-02T10:52Z · **HEAD:** 03e86edfe
- **Host:** Apple **M5 Max**, 137 GB, Metal, MLX **0.31.2**, python 3.13
- **Scope:** READ + PROFILE only. No trainer edits. No heavy n600 launch. Small-scale
  MLX-GPU profiling (a few steps) only. No live GPU run was contending (checked: only
  a 0%-CPU checkpoint-archiver was running).
- **Config profiled:** `witness_autoconfig._proven_base` (the live #205 launch config):
  render 384×512, accum_pairs=8, n_hidden=4, hosc, self-orient, eikonal+length,
  lane-prior-phi1, **w_pose=0 → the training step is SEG-ONLY** (SegNet reads the last
  frame only; pose is monitoring-only on the async CPU-authority thread, off the step).

---

## TL;DR — the five deliverables in one paragraph

The **one Metal kernel that matters is already built and active**: the custom
grouped-conv backward delivers a **MEASURED 16.9×** on the SegNet fwd+bwd (6713 → 396
ms @ B=8) and it is **ON by default**. It carries the full step: with the kernel ON the
representative seg-only step is **577 ms** (SegNet 69%, INR trunk 26%, R-op 5%); with it
OFF the step is ≈ **6.6 s** (≈11× slower — the OFF run literally timed out my 2-min
budget). The other two "fast paths" from the campaign — **`mx.compile` and the fused-R
Metal kernel — are NOT wired into either trainer** (no `--compile-step` / `--fused-r-kernel`
flag exists), so #205 today runs **without** them. Their measured upside is small at the
live render res (compile 1.22× on the trunk ≈ −27 ms/step; fused-R forward 10.8× but R is
only 5% of the step ≈ −15 ms/step) **and the fused-R VJP is currently broken on MLX 0.31.2**.
**Build plan: nothing new needs building before #205** (the high-impact kernel is banked);
Wave-D should (a) optionally wire `mx.compile` (cheap, bit-identical gate exists), (b) fix
the stale scorer bench, and (c) run the fused-R / AA-SDF kernels as a **parallel #212
campaign** that only becomes #205-critical **if a lever raises render resolution**
(supersample). The launch gate must be upgraded from a **flag check** to a **throughput
assertion** (a one-shot SegNet fwd+bwd micro-bench with a hard ms ceiling).

---

## 1. Fast-path verification (MEASURED — active? speedup delivered?)

Harness: forced `TAC_MLX_CUSTOM_GROUPED_BACKWARD` explicitly (0 vs 1) in separate
processes, real EfficientNet-B2 SegNet + FastViT-T12 PoseNet (frozen safetensors), the
real reference scorer-input cache, B=8, GPU, fp32.

| stage (B=8) | reference (env=0) | custom Metal (env=1) | **speedup** |
|---|---:|---:|---:|
| segnet_fwd | 304.2 ms | 119.6 ms | 2.5× |
| **segnet_fwd_bwd** | **6713.4 ms** | **396.4 ms** | **16.9×** |
| posenet_fwd | 255.7 ms | 17.7 ms | 14.5× |
| **posenet_fwd_bwd** | **1039.1 ms** | **43.0 ms** | **24.2×** |

Verdict per fast path:

- **Custom grouped-conv backward Metal kernel** (`metal_grouped_conv_backward.py`,
  `mx.fast.metal_kernel` grad_input + grad_weight): **ACTIVE by default** (env default
  `"1"`; Metal backend available) and **delivering the ~17× as advertised** (16.9×
  MEASURED). This is the single load-bearing lever. It exists because native MLX
  strided-grouped-conv backward is *numerically wrong* (cosine ~0.025), not merely slow —
  so it is the rare custom kernel worth building. **Applies to only the 4 strided-grouped
  convs in SegNet / 8 in PoseNet**, but those dominate the backward.
- **`mx.compile`** (`mlx_compile_step.py`): **NOT WIRED.** No `--compile-step` flag exists
  in either trainer; the module is built + CPU-tested but orphan. #205 runs uncompiled.
  MEASURED upside (below): 1.22× on the trunk slice.
- **Fused-R Metal kernel** (`metal_fused_r_operator.py`): **NOT WIRED** (no
  `--fused-r-kernel` flag). Forward works (10.8× MEASURED, below) but its
  `@mx.custom_function` **VJP is BROKEN on MLX 0.31.2** — both `mx.grad` and `mx.vjp`
  raise `ValueError: too many values to unpack (expected 1)` (the `(x,) = primals` unpack
  in `_fn_vjp`). Forward-only usable; **cannot be put in a training step until the VJP is
  fixed.** (Prior campaign validated the VJP via `mx.vjp` of the *oracle*, not via the
  custom_function path — that gap is why this wasn't caught.)
- **fp32:** confirmed (the scorer adapter is float32; matches the MPS/MLX guidance that
  fp32 is the sweet spot, half-precision is slower + worse-gradient).

### Bench bug found (fix in Wave-D)
`experiments/bench_mlx_scorer_stage_breakdown.py` reports **default ≈ custom (no
speedup)** — MISLEADING. Root cause: its `custom=False` column only `pop`s the env, but
the adapter default flipped to `"1"` (ON), so `get(env, "1")` still returns ON → **both
columns run the custom path.** It must set `TAC_MLX_CUSTOM_GROUPED_BACKWARD="0"` to force
the reference. (This is exactly the launch-gate lesson: a flag "present" ≠ throughput
delivered.)

---

## 2. Profiled hot-op breakdown (MEASURED, live seg-only step, B=8, kernel ON)

End-to-end representative step (trunk INR → compose RGB → R roundtrip → SegNet(last
frame) → CE loss → `value_and_grad` backward), attributed by ablation (drop-R, drop-SegNet):

| op | ms/step (B=8) | % of step | notes |
|---|---:|---:|---|
| **SegNet fwd+bwd** | **399.1** | **69%** | grouped-backward kernel already applied here |
| **INR trunk render** (compose RGB, 8 pairs) | **150.2** | **26%** | compute/mem-bound (see below) |
| **R operator** (eval roundtrip, f1 only, B=8) | **27.9** | **5%** | seg-only ⇒ R on last frame only |
| **full step** | **577.2** | 100% | kernel ON |
| full step, **kernel OFF** | **≈ 6600** | — | ≈11× slower (OFF run timed out @ 2 min) |

Sanity: 577 ms/step × 75 chunks (n600 / accum_pairs 8) ≈ **43 s/epoch** MLX-GPU seg-only,
vs the prior torch-CPU anchor of 18.7 min/epoch — consistent order of magnitude.

Sub-findings on the two non-SegNet slices:

- **INR trunk (150 ms, 26%)** — the surprise #2 slice. It is **NOT launch-overhead-bound**:
  batching the 8-pair Python loop into one `(B,P,·)` matmul gave **0 speedup** (148.6 vs
  150.2 ms). It is genuine compute/memory traffic (4× `Linear(96,96)` + FiLM affine + relu
  + softmax + sigmoid over 8×196 608 = 1.57 M points). `mx.compile` fuses the elementwise
  chain for **1.22×** (150 → 123 ms). A custom fused-MLP Metal kernel could push further but
  the ceiling on this slice is small (~150→~90 ms).
- **R operator (28 ms, 5%)** — small in the seg-only step. Production MLX R (5 separable
  passes + round) fwd+bwd = 57.5 ms at B=16 (both frames); ~28 ms for the seg-only single
  frame. (If pose were ever turned on, R doubles to both frames and PoseNet adds ~43 ms.)

---

## 3. Ranked Metal-kernel table (by MEASURED #205 wall-clock impact)

| rank | kernel / fusion | current state | targets (measured) | est. #205 win | #205-critical? |
|---|---|---|---|---|---|
| **1** | **grouped-conv backward** | **BUILT + ACTIVE + parity-gated** | SegNet/PoseNet strided-grouped conv bwd | **already banked: 16.9× (step ≈11× vs OFF)** | **YES — the lever; already done** |
| 2 | `mx.compile` (fusion, not a kernel) | built, tested, **NOT wired** | INR trunk elementwise chain (26% slice) | 1.22× on trunk ⇒ **≈ −27 ms/step (~5%)** | Low-med — cheap to wire |
| 3 | fused-R forward kernel | built, CPU-parity OK, **VJP BROKEN**, not wired | R forward (5% slice): **10.8× fwd (35.4→3.3 ms) MEASURED-GPU** | fwd-only ≈ −15 ms/step (~2.6%) | **No at 384×512**; rises if render res grows |
| 4 | fused-R transpose VJP (P2b) | not built | R backward (~14–22 ms) | ≈ −10 ms/step | No |
| 5 | persistence-pool (3×3 soft morph) | signature-spec only; MLX-native min/max pool | NOT in live config | ~0 (cheap term; native pool fine) | No |
| 6 | island-birth term | MLX-native + `mx.compile`'d variant; signature-spec | NOT in live config | ~0 (elementwise term) | No |
| 7 | AA-SDF line/area raster | boundary_math module, not in live step | render (analytic-lane / supersample) | unmeasured (lever off) | No now; **campaign** |
| 8 | warp grid-sample / margin-map / curvelet-dir bank | modules exist; reorient is every-50-ep (amortized ~0) | render/feats | unmeasured (lever off) | No now; **campaign** |

Honest note (NO-FAKE): ranks 3–8 are **NOT** worth building/wiring for the #205
pointer-mover run purely to "use Metal." The step is 95% SegNet+trunk; the SegNet lever is
already banked. Ranks 3–8 only earn their build cost when (a) a lever raises render
resolution (supersample makes trunk+R dominate), or (b) they feed repeated future runs /
the production memory-tiers.

---

## 4. Wave-D build plan (pre-#205 vs parallel campaign)

**PRE-#205 (only if it helps the pointer-mover run):**
1. **Nothing new needs building.** The high-impact kernel (grouped-backward) is built +
   active + delivering 16.9× MEASURED. Do NOT block #205 on new kernels.
2. **(Optional, cheap) Wire `mx.compile`** as `--compile-step` on the SEG-ONLY
   loss-and-grad closure (pose excluded — already the design). Gate it with the existing
   `assert_compile_bit_identical` (max|Δ| ≤ 1e-6, determinism exact) at launch. Win ≈ 5%
   of step. Low risk, but not required for the run.
3. **Fix the stale bench** (`bench_mlx_scorer_stage_breakdown.py`): force env `"0"` in the
   reference column so it reports the true 16.9× (currently reports ~1× — a false negative
   that could hide a real regression).

**PARALLEL #212 CAMPAIGN (feeds future runs, supersample, production — off the #205 path):**
1. **Fix the fused-R custom_function VJP** (`_fn_vjp` primals-unpack) on MLX 0.31.2 +
   validate through `value_and_grad` (not just `mx.vjp` of the oracle), then wire
   `--fused-r-kernel`. Add the on-GPU parity gate (`assert_metal_matches_cpu_oracle`).
   NEW measured GPU data point this session: fused-R **forward is bit-parity-expected and
   10.8× faster** — the campaign's gated GPU claim is now partially confirmed on GPU.
2. **fused-R transpose VJP (P2b)** — only after #1, only if R becomes a bigger slice.
3. **AA-SDF raster kernel** — build alongside the analytic-lane/supersample lever wire-in
   (#224); measure its step share THEN and re-rank. Do not build blind.

Every kernel's contract (unchanged, enforced): **bit-identical to the numpy-fp32
reference + a parity gate + a benchmark.** Rank by MEASURED impact after the lever is wired.

---

## 5. Launch-gate throughput assertion spec (the memory's binding fix)

Per `feedback_launch_gate_must_verify_perf_env_not_just_flags`: the current gate
(`tools/launch_witness_run.py::verify_perf_env`) only greps the
`{"stage":"custom_grouped_backward","active":true}` LINE — a **flag/backend** check, not a
throughput check. A silently-degraded path (Metal "available" but slow, a regression, or
the stale-bench-style default flip) passes today. Upgrade to a THROUGHPUT gate:

1. **One-shot SegNet fwd+bwd micro-bench at launch** (add to the launcher, ~1 s):
   time `mx.value_and_grad(seg_loss)(seg_nhwc)` at B=8 on GPU; **FAIL if median > 700 ms.**
   Rationale: MEASURED ON=396 ms, OFF=6713 ms → a 700 ms ceiling cleanly separates the fast
   path from the ground-down path with wide margin (and catches any future 2× regression).
2. **First-N-steps median step-time ceiling** from the trainer's own per-step telemetry:
   assert median(step_ms over first ~10 steps) ≤ **1.5× the config baseline** (baseline ≈
   577 ms for proven_base seg-only B=8; scale by render-res / accum_pairs if changed).
   WARN→FAIL if exceeded (a degraded kernel, OOM-thrash, or an accidental non-fast path).
3. **If `--compile-step` is wired:** require `assert_compile_bit_identical` PASS before the
   run trusts the compiled step (bit-identity + determinism), else run uncompiled.
4. Keep the `active=true` line as a **fast pre-filter**, but the **micro-bench (item 1) is
   the real gate** — it verifies THROUGHPUT DELIVERED, not merely a flag SET.

This makes the launch gate honor the memory: **verify the ~17× is actually happening**, not
just that the env is set.

---

## Provenance / reproduce
- grouped-backward: scratchpad `prof_grouped_bwd.py <0|1> <B> <iters> <warmup>` (forces env).
- R op: inline harness (production `apply_contest_faithful_roundtrip_nhwc` vs
  `make_fused_r_roundtrip`), B=16.
- e2e step + ablation: scratchpad `prof_e2e_step.py` (env-controlled grouped-backward).
- trunk compile/batch: inline harnesses.
- fused-R VJP bug: `mx.grad`/`mx.vjp` of `make_fused_r_roundtrip` → `ValueError` on MLX 0.31.2.
- All numbers `[macOS-MLX advisory]`; numpy-fp32 CPU / contest-CUDA remain the only score
  authority. Pointer 0.19110 UNMOVED.
