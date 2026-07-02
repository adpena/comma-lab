# #205 pre-launch gate — MLX/Metal NEW-kernel plan + EXISTING-CODE low-hanging fruit (ranked)

`[macOS-MLX/Metal engineering — advisory]` **means ≠ ends.** This is the compute-facet
(facet #6) phase of the #205 pre-launch gate. The canonical frontier pointer **0.19110 is
UNMOVED** and moves ONLY on a byte-closed exact-eval n600 row (numpy-fp32 CPU / contest-CUDA;
MPS never). Every kernel's acceptance bar is **bit-identical to the numpy-fp32 reference +
a parity gate + a benchmark** (CLAUDE.md compute-facet non-negotiable).

- **Date (UTC):** 2026-07-02 · **HEAD:** `78a623d27` · **Host of the profile:** Apple M5 Max, 137 GB, MLX 0.31.2.
- **ANALYSIS/DESIGN ONLY** — read + estimate + rank. NO live GPU profile (trainer mid-edit by
  another agent), NO GPU launch, NO code edits. Timing tagged **MEASURED** (from the
  companion profile ledger `compute_facet_mlx_metal_profile_and_wave_d_build_plan_20260702T105220Z.md`,
  ran on this host earlier today) vs **ESTIMATE** (derived, not run this session). Code-state
  claims are FACTS from reading the trainer + kernel modules at HEAD `78a623d27`.
- **Companion:** the profile ledger above measured the hot path; THIS ledger answers the
  operator's standing question — *"which NEW MLX/Metal kernels to build now/parallel/future"* —
  and adds the coordinator's **EXISTING-CODE low-hanging fruit** tier (highest ROI, no build).

---

## TL;DR — the honest one-paragraph answer

**Build NOTHING new before #205.** The single load-bearing Metal kernel (custom grouped-conv
backward) is **already built, active-by-default, parity-gated, and delivering 16.9× MEASURED**;
it carries the step (95% of #205 wall-clock is SegNet+trunk, and the SegNet backward is exactly
what that kernel accelerates). The **highest-ROI action is NOT a new kernel — it is one
existing-code fix**: the entire training step runs **UN-compiled** (`mx.compile` appears
**nowhere** in either trainer's hot path), so wiring `mx.compile` on the seg-loss/trunk closure
is a ~5% step win (**ESTIMATE −27 ms/step** from the MEASURED 1.22× trunk fusion), zero new
kernel to verify, and the bit-identity gate (`assert_compile_bit_identical`, max|Δ|=1e-6 +
determinism) already exists. Everything else in the existing MLX/Metal impl is **already
optimized** (cf_mx_cache in-place rebuild avoids the 2× 41 GB transient; Fourier features are
cached not recomputed; the launch-gate is already a THROUGHPUT assertion not a flag check; fp32
is correct-and-faster here). The genuine **NEW-kernel campaign (#212) is PARALLEL/FUTURE**, not
#205-critical: fused-R (VJP is BROKEN on MLX 0.31.2 — must fix before it can enter a step) and
AA-SDF raster only earn their build cost **after** a lever raises render resolution or the
analytic-lane band is trained render-time. Deep-math: contest S has no time term, but the
**campaign** objective is lowest-S × max-synergy × **shortest-train** on ONE GPU under a
multi-day budget — so per-epoch wall-clock IS an indirect score lever (faster epoch → more
epochs / deeper d_seg descent / more levers within the fixed burn). Correctness kernels
(grouped-backward: native MLX strided-grouped backward is **numerically wrong**, cos ~0.025)
are must-haves regardless of speed.

---

## Deep-math hook — why a faster epoch lowers S (and why correctness kernels are unconditional)

`S = 100·d_seg + √(10·d_pose) + 25·bytes/37_545_489`. There is **no explicit time term**
(README.md:92). So a kernel is NOT a *direct* score lever. It is an **indirect** one, two ways:

1. **Shortest-train IS in the campaign objective.** #205 is a multi-day burn on ONE M5 Max.
   The achieved d_seg at the wall-clock deadline is a monotone-decreasing function of epochs
   completed (the descent has not plateaued at the config's floor). A kernel that cuts
   per-epoch ms lets the SAME budget reach MORE epochs → deeper into the CE→τ→l7→Muon
   curriculum → lower achieved d_seg → **lower S**. It also lets more per-lever A/B fit in
   the budget. This is the honest coupling; every ms/step estimate below is a proxy for
   "epochs bought per fixed budget."
2. **Correctness kernels are unconditional.** The grouped-conv backward exists because native
   MLX strided-grouped-conv backward is **numerically WRONG** (cosine ~0.025 vs numpy-fp32),
   not merely slow. A wrong gradient corrupts the descent → wrong d_seg → the run is a FAKE
   measurement. So correctness kernels are must-haves **regardless of speed**; speed kernels
   are ranked by (hot-path ms saved) × (epochs-bought value) × (1/build-cost·risk).

---

## 0. Existing-kernel inventory — built / active / measured (FACTS at HEAD 78a623d27)

| kernel / fusion | module | built? | active in #205? | measured speedup | parity-gated? |
|---|---|---|---|---|---|
| **grouped-conv backward** (grad_input + grad_weight, 2× `mx.fast.metal_kernel`) | `local_acceleration/metal_grouped_conv_backward.py` | **YES** | **YES, default-ON** (env default `"1"`) | **16.9× SegNet fwd+bwd, 24.2× PoseNet** (MEASURED) | **YES** (`test_metal_grouped_conv_backward.py`) — **correctness-critical** (native MLX cos ~0.025) |
| fused-R forward (`apply_contest_faithful_roundtrip_nhwc` Metal variant) | `local_acceleration/metal_fused_r_operator.py` | fwd YES | **NO** (no `--fused-r-kernel` flag) | 10.8× fwd (35.4→3.3 ms, MEASURED-GPU) | fwd parity OK; **VJP BROKEN on MLX 0.31.2** (`_fn_vjp` `(x,)=primals` → `ValueError`) |
| `mx.compile` step fusion (not a kernel) | `local_acceleration/mlx_compile_step.py` | YES | **NO** (no `--compile-step` flag; grep: mx.compile absent from BOTH trainers' hot path) | 1.22× on the 26% trunk slice (MEASURED) | YES (`assert_compile_bit_identical`, atol 1e-6 + determinism) |
| persistence-pool (3×3 soft min/max morph) | `boundary_math/persistence_topology_loss.py` | **SPEC only** (`metal_pool_kernel_signature()`; runs MLX-native pool) | NO (lever off in seg-only) | — (MLX-native fwd+bwd 24.9 ms; term is cheap) | parity 0.99999941 (numpy↔MLX) |
| island-birth term | `boundary_math/island_protection.py` | **SPEC only** (`metal_island_birth_kernel_signature()`; MLX-native + `mx.compile` variant) | NO (lever off) | MLX-compiled 0.86 ms/pair (n600 0.51 s); parity 9.3e-8 | numpy authority |
| warp grid-sample + R | (no module) | **SPEC only** (DAG signature `TAC_MLX_CUSTOM_WARP_GRID_SAMPLE`) | NO (pose off, w_pose=0) | MLX-GPU warp+R 2.5 ms/pair vs numpy 118 ms (47×) | — |
| AA-SDF lane coverage raster | `boundary_math/aa_sdf_observation_render.py` | MLX-native + numpy ref built; **no Metal kernel** (`aa_sdf_lane_coverage` flagged) | NO (`--render-aa none` in autoconfig; analytic-lane is the AA path, not supersample) | AA-SDF raster 0.34 ms/frame numpy-vectorized; composite MLX 13.1 ms vs numpy 842 ms (64×, bit-identical) | numpy authority (parity ≥0.9997) |
| fused `aa_render_through_R` (supersample→box→bicubic-up, one strided pass) | (spec in DAG) | **not built** | NO | box-downsample MLX 8 ms vs numpy 210 ms (26×) | numpy authority |
| margin/saliency map (`∂margin/∂input`) | `margin_saliency_map.py` | built (**torch**, not MLX) | NO (LEVER-4 off in seg-only) | — (diagnostic) | torch autograd of frozen SegNet |

**One kernel matters, it is done.** Ranks below the grouped-backward are OFF the live #205 step
(seg-only, `w_pose=0`, `--render-aa none`, LEVER-4/persistence/island/warp all default-off).

---

## 1. ⭐ EXISTING-CODE low-hanging fruit tier (coordinator's ask — HIGHEST ROI, no new kernel)

Audit of the CURRENT MLX/Metal impl at HEAD `78a623d27` (facts from reading the code; wins
tagged ESTIMATE — no live profile this session). Ranked by (wall-clock win) × (low effort/risk):

| # | item | finding (FACT) | est. win | effort / risk | verdict |
|---|---|---|---|---|---|
| **F1** | **(a) `mx.compile` coverage** | `mx.compile` appears in **ZERO** places in either trainer; the trunk INR render + seg-loss `value_and_grad` closure (26%+69% of the step) runs **un-compiled**. Shapes are STATIC (accum_pairs=8, render 384×512 fixed) → no dynamic-shape graph breaks. Module + bit-identity gate already built. | **ESTIMATE −27 ms/step (~5%)** (MEASURED 1.22× on the 26% trunk) | LOW / LOW (gate exists) | **DO — the one real fruit.** Wire `--compile-step` on the SEG-ONLY closure, gate with `assert_compile_bit_identical` at launch, run uncompiled on gate FAIL. Optional pre-#205 (not required). |
| **F2** | **(f→bench) stale scorer bench** | `experiments/bench_mlx_scorer_stage_breakdown.py` `custom=False` column only `pop`s the env, but the adapter default is now `"1"` → BOTH columns run the fast path → reports ~1× (false negative that would HIDE a real 17× regression). | 0 ms (measurement hygiene) | LOW / LOW | **DO.** Force `TAC_MLX_CUSTOM_GROUPED_BACKWARD="0"` in the reference column. Protects the throughput gate's credibility. |
| F3 | **(f) launch-gate throughput** | **ALREADY DONE.** `tools/launch_witness_run.py::_run_throughput_gate` (line 156) runs a SegNet fwd+bwd micro-bench and **REFUSES if median > threshold** (`--throughput-threshold-ms`, `--skip-throughput-gate`). This is the memory-binding "verify throughput not just flags" fix — implemented. | — | — | **VERIFY only** (confirm it fires in the #205 launch path; it is wired at line 279–280). No work. |
| F4 | **(e) cf_mx_cache 41 GB memory** | **ALREADY DONE.** `rebuild_per_pair_feats_in_place` (line 2424) rebuilds the per-pair feature cache in place — the naive list-comp held old+new = **2× ~41 GB** at n600 → OOM; the in-place path holds it STEADY (peak ~63 GB). | — | — | **DON'T touch.** Note: 41 GB resident + supersample would hit ~86 GB — a FUTURE memory-budget constraint (see F8), not a wall-clock fruit. |
| F5 | **(d) redundant recompute** | **ALREADY GOOD.** Directional Fourier features are **cached** in `cf_mx_cache`, rebuilt only on reorient (every 50 ep), NOT per-step. R op is per-pair (inherent — each pair renders separately). Margin map + basis are NOT recomputed in the seg-only step. | — | — | **DON'T touch.** No stale recompute on the hot path. |
| F6 | **(g) eval / reorient cadence** | Async verdict runs on a **daemon CPU-authority thread OFF the GPU step**, self-throttling (one in flight; skips if prior still running, line 2383). eval-every=25, reorient-every=50 (41 GB rebuild amortized). Periodic telemetry `mx.eval`s fire only every 25 ep. | ~0 (already off-step) | — | **DON'T touch.** Cadence does not cost GPU wall-clock. (Minor: `_join_async_verdict` before re-schedule can block if a verdict outlasts 25 ep, but it self-throttles — leave it.) |
| F7 | **(b) grouped-conv kernel tuning** | Kernel uses a flat 1-D dispatch: `grid=(x.size,1,1)`, `threadgroup=(256,1,1)`, one thread/output-elem; grad_input + grad_weight are **two separate kernels**. Tile/coalescing tuning or fusing the two is *possible* but this is the ONE correctness-critical, already-16.9× kernel. | ESTIMATE uncertain (maybe 1.1–1.3× on a 69% slice) | MED–HIGH / **HIGH** (touch the load-bearing correctness kernel) | **DEFER to FUTURE.** Not worth the regression risk before/around #205. Needs a dedicated parity+bench cycle. |
| F8 | **(c) dtype fp16/bf16** | fp32 confirmed on the scorer adapter + gradient path. Per CLAUDE.md + the profile: fp16/bf16 on Apple GPU is **SLOWER *and* worse-gradient** — the fp32 sweet spot. No measured-safe fp16 spot exists on the gradient path. | negative (slower) | — / — | **DON'T.** fp32 is both the authority AND the fast path here. No fruit. |

**Honest negatives (measured, do NOT "optimize"):** (i) per-pair `mx.eval(loss, grads)` in the
accum loop (line 2969) — the profile MEASURED that batching the 8-pair loop gives **0 speedup**
(compute/mem-bound, not launch-overhead-bound); the per-pair eval also intentionally bounds the
lazy graph to avoid the 2× memory blowup. Leave it. (ii) fp16 (F8). (iii) batching the trunk
(MEASURED 148.6 vs 150.2 ms = 0×).

**Fruit-tier bottom line:** exactly **ONE** genuine wall-clock win (F1, ~5%, optional) + **ONE**
measurement-hygiene fix (F2). The rest of the existing impl is already optimized — the highest
ROI here is confirming that, not churning it.

---

## 2. Ranked NEW-kernel plan — NOW / IN-PARALLEL / FUTURE

### NOW (build/wire before the #205 multi-day burn)
**NONE required. The correctness-critical, high-impact kernel (grouped-backward) is already
banked.** Do **not** block #205 on a new kernel — the step is 95% SegNet+trunk and the SegNet
backward is already 16.9×. Optional-cheap: **F1 (`mx.compile` wire-in)** from the fruit tier —
it is a *fusion*, not a new kernel, so it needs no new parity kernel, only the existing
`assert_compile_bit_identical` gate. Treat F1 as the ONLY "NOW" candidate and it is optional.

### IN-PARALLEL (build while #205 trains / during the gate — medium win, off the #205 critical path, feeds future runs + #212)

| kernel | why parallel | build spec | expected win | parity requirement | risk |
|---|---|---|---|---|---|
| **P1. Fix fused-R custom_function VJP** (`metal_fused_r_operator.py`) | Forward is 10.8× MEASURED but the `@mx.custom_function` VJP raises `ValueError: too many values to unpack` on MLX 0.31.2 (`(x,)=primals` unpack in `_fn_vjp`). **Until the VJP works it CANNOT enter a training step.** R is 5% of the seg-only step now, but rises if render-res grows (P3/F8) and doubles when pose turns on. | Fix the `_fn_vjp` primals-unpack for the MLX 0.31.2 `(primals, cotangent, output)` signature; validate through `value_and_grad` (NOT just `mx.vjp` of the oracle — that gap hid this bug), then wire `--fused-r-kernel`. | ESTIMATE fwd-only −15 ms/step (~2.6%) at 384×512; grows with render-res / pose-on | on-GPU `assert_metal_matches_cpu_oracle` (fwd) **and** VJP bit-parity vs numpy-fp32 through `value_and_grad` | MED (VJP correctness on a contest-exact R is load-bearing) |
| **P2. AA-SDF lane-coverage Metal raster** (`aa_sdf_lane_coverage`) | The #1 MEASURED d_seg **representation** lever (point-sample erases class-1 dashes; footprint-integrated recovers +0.38 recall). MLX-native + numpy ref already built; a Metal kernel only earns its cost **when the analytic-lane band is trained render-time or supersample is on** (`--render-aa none` today). Build the kernel in parallel so it is READY when lever #224/#213 wires it in. | `aa_sdf_lane_coverage(u_center[L,H], hw[L,H], gate[L,H], col_grid[W], softness) -> coverage[H,W]` per the DAG signature; coverage-integrated (no python pixel loop). | unmeasured until the lever is wired (build-then-measure, do NOT build blind of the wire-in) | numpy-fp32 `aa_sdf_observation_footprint_render_dseg_v1` reference, parity ≥0.9997 | LOW–MED |

### FUTURE (speculative / small / needs the run's own profile or a lever first)

| kernel | gate to promote | expected win | note |
|---|---|---|---|
| F1b. fused-R transpose VJP (P2b) | only after P1, only if R becomes a bigger slice | ESTIMATE −10 ms/step | not built |
| Fused `aa_render_through_R` (supersample→box→bicubic-up, one strided pass) | only if a lever raises render res / turns on supersample (makes trunk+R dominate) | avoids materializing `(M, ss·H, ss·W, 3)`; MEASURED box-down 26× | the supersample path is DISQUALIFIED for the contest witness anyway (autoconfig picks `render-aa none`/`ipe`); this kernel is for research/production render, not the #205 archive |
| persistence-pool + island-birth Metal kernels | only if those levers enter the live config AND profile shows the term is hot | ~0 (terms are cheap; MLX-native pool/elementwise already fast) | SPEC-only; MLX-native is fine per measured 24.9 ms / 0.86 ms |
| warp grid-sample + R fused kernel | only if pose turns on (`w_pose>0`) — it is a pose/warp op, dead in seg-only | 47× vs numpy (MEASURED) but off the seg-only path entirely | no module yet |
| grouped-conv backward tile/coalescing tune + grad_input/grad_weight fusion (F7) | dedicated parity+bench cycle, NOT near a launch | ESTIMATE 1.1–1.3× on the 69% slice, uncertain | HIGH risk — the one correctness-critical kernel; only touch with a full regression harness |
| margin-saliency map MLX port (currently torch) | only when LEVER-4 (margin-saliency) enters the live loss AND is per-step | unmeasured | today it is a diagnostic, off the seg-only step |

---

## 3. Honest hot-path analysis (MEASURED vs ESTIMATE)

From the companion profile ledger (**MEASURED** on M5 Max, live `_proven_base` seg-only step,
B=8, grouped-backward kernel ON):

| op | ms/step | % | on a candidate kernel's path? |
|---|---:|---:|---|
| **SegNet fwd+bwd** | 399.1 | **69%** | grouped-backward (DONE) · F7 tune (FUTURE, risky) |
| **INR trunk render** (compose RGB, 8 pairs) | 150.2 | **26%** | **F1 `mx.compile`** (−27 ms ESTIMATE); genuine compute/mem-bound (batching = 0×) |
| **R operator** (roundtrip, f1 only) | 27.9 | **5%** | P1 fused-R (−15 ms ESTIMATE, needs VJP fix) |
| **full step** | **577.2** | 100% | kernel ON |
| full step, kernel OFF | ≈6600 | — | ≈11× slower (grouped-backward is load-bearing) |

Epoch math: 577 ms × 75 chunks ≈ **43 s/epoch** MLX-GPU seg-only (vs 18.7 min/epoch torch-CPU).
**Sum of every remaining lever (F1 + P1) ≈ −42 ms/step ≈ 7%** of the step → ~40 s/epoch. That
is the ENTIRE realistic near-term wall-clock headroom without a render-res change: the step is
95% SegNet+trunk and the SegNet lever is banked. **This is why "build nothing new before #205"
is the honest verdict**, and why the fruit-tier F1 (a fusion, not a kernel) is the only
pre-#205 win worth considering.

Estimates are ESTIMATE (derived from the MEASURED per-slice 1.22× / 10.8× applied to the slice
share); the real profile is the run's own per-step telemetry (which the throughput gate F3
already asserts against a ceiling).

---

## 4. Build-before-#205 vs during/after — the verdict

- **BEFORE #205:** build **nothing new**. Optionally wire **F1 (`mx.compile`)** — a fusion with
  an existing bit-identity gate, ~5% step, LOW risk — and do **F2** (bench fix, measurement
  hygiene). Neither is required to launch; the grouped-backward kernel + the already-live
  throughput gate (F3) are the launch-critical compute pieces and both are DONE.
- **DURING #205 / at the gate (PARALLEL):** **P1 (fix fused-R VJP)** and **P2 (AA-SDF Metal
  raster)** — the real #212 campaign. They feed future runs, the supersample/analytic-lane
  lever, and production render; they become #205-relevant only if a lever raises render res.
- **AFTER / FUTURE:** the transpose VJP, the fused AA-render-through-R, the persistence/island/
  warp kernels, the grouped-conv tune (F7), and the margin-saliency MLX port — each promoted
  ONLY when its lever enters the live loss and its slice is profiled hot (build-then-measure,
  never build blind).

**Every promotion is measured after the lever is wired; every kernel is bit-identical to
numpy-fp32 + parity-gated + benched. Pointer 0.19110 UNMOVED — this is means, not the end.**

---

## Provenance / reproduce
- Code-state facts read at HEAD `78a623d27`: `experiments/train_levelset_witness_realized_through_R_mlx.py`
  (mx.compile absent from hot path; `mx.eval` per-pair line 2969; `rebuild_per_pair_feats_in_place`
  line 2424; async-verdict daemon thread line 2406), `experiments/train_witness_realized_through_R_mlx.py`
  (`render_through_R_mlx` line 343, `make_loss_fn` line 752), `tools/launch_witness_run.py`
  (`_run_throughput_gate` line 156, wired line 279), `src/tac/local_acceleration/{metal_grouped_conv_backward,metal_fused_r_operator,mlx_compile_step}.py`,
  `src/tac/boundary_math/{aa_sdf_observation_render,persistence_topology_loss,island_protection}.py`,
  `src/tac/witness_autoconfig.py` (`render_aa: none`), the #205 launch config
  `experiments/results/levelset_n600_witness_20260702T115211Z/launch.sh`.
- All MEASURED timings sourced from `compute_facet_mlx_metal_profile_and_wave_d_build_plan_20260702T105220Z.md`
  (M5 Max, MLX 0.31.2, this-host profile). `[macOS-MLX advisory]`; numpy-fp32 CPU / contest-CUDA
  are the only score authorities. NO live GPU run this session (trainer mid-edit; no launch).
