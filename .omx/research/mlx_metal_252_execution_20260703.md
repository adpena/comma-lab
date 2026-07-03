# Compute-facet #252 execution — fused-R wiring + mx.compile gate + timing + drift attribution (2026-07-03)

**Standing program:** MLX + custom Metal is a first-class, intrinsic-value optimization program
(`[[mlx-metal-standing-first-class-optimization-program-regardless-of-task]]`), held to the SAME bar as
every unit: automated · world-class · recursively-adversarial-reviewed · deep-math-solved · CONFIRMED
bit-identical to the numpy-fp32 authority. **NO-FAKE:** every speedup/parity number below is MEASURED (or
explicitly labelled an estimate), never asserted. MLX/MPS is NEVER a score — this buys SPEED +
training-gradient fidelity; the d_seg/d_pose verdict + byte-closed archive stay numpy/torch-CPU authority.
**Pointer 0.19110 UNMOVED** (this is MEANS).

**Machine safety:** #205 witness training was LIVE on the Apple GPU throughout (PID 29129, RSS ~54.6 GB,
under `safe_run.py`; out-dir `levelset_n600_witness_20260703T120444Z`). All GPU touches here were SMALL +
BRIEF (tiny-input parity + micro-bench, ≤2 s each). **The full uncontended whole-run benchmark is DEFERRED
to after #205** (exact command in §5). Every trainer addition is DEFAULT-OFF and **byte-identical when
off** (verified) so the running #205 and any `--resume-from` are completely unaffected.

## What was wired (all DEFAULT-OFF, byte-identical when off)

Files: `experiments/train_witness_realized_through_R_mlx.py` (base; owns the render/R path),
`experiments/train_levelset_witness_realized_through_R_mlx.py` (LIVE launch entry — flags + startup gates +
timing), `src/tac/tests/test_witness_r_op_dispatch_wiring.py` (new; 8 tests).

Exact flags (levelset trainer):
- **`--fused-r-kernel`** (default OFF) — swaps the pure-MLX R roundtrip
  (`apply_contest_faithful_roundtrip_nhwc`) for the fused Metal kernel
  (`metal_fused_r_operator.make_fused_r_roundtrip`) in the two base render functions
  (`render_through_R_mlx` + `render_batch_through_R_mlx`, via a new module-level `_apply_R` dispatch).
  Startup runs the per-chip parity gate `assert_metal_matches_cpu_oracle()` which **fails CLOSED** if the
  kernel is not bit-identical on this GPU.
- **`--mx-compile`** (default OFF) — installs an `mx.compile`'d reference R only if a startup bit-identity
  gate passes; else **RAISES (fail closed)** — see §2.
- **`--profile-timing`** (default OFF) — per-epoch phase split + isolated in-situ R micro-bench (§3).

Bit-identical-when-off is **structural**: with no flags, `_apply_R(rgb)` is exactly
`apply_contest_faithful_roundtrip_nhwc(rgb, output_hw=(SEG_H, SEG_W), ste_round=True)` — the pre-existing
inline call, op-for-op.

## 1. Fused-R — MEASURED bit-identity + speed (small-scale, on this host)

Existing authority suite `src/tac/tests/test_metal_fused_r_operator.py` (25 tests) **passed in 1.36 s** —
it already proves numpy oracle == MLX production R (bit-identical), metal forward == numpy oracle
(bit-identical incl. real 874×1164→384×512), and fused-VJP == non-fused MLX-path VJP (~1 ULP). My new
wiring suite (8 tests, 1.03 s) proves the **specific swap**:

| Check | Result |
|---|---|
| OFF `_apply_R` == `apply_contest_faithful_roundtrip_nhwc` | **byte-identical (max\|Δ\|=0)** — #205 resume unaffected |
| FUSED forward vs reference | **bit-identical (max\|Δ\|=0)** |
| FUSED VJP vs reference VJP | max\|Δ\|=9.54e-6 (~1 ULP, the shared GPU-reduction floor both paths carry) |

**Isolated R micro-bench @ real render 384×512 (contended with #205 → conservative):**

| | ms/frame (ref, pure-MLX) | ms/frame (fused Metal) | speedup |
|---|---|---|---|
| forward | 6.056 | 0.453 | **13.4×** |
| forward+backward | 6.168 | 1.315 | **4.69×** |

The fused-R op is **4.69× on fwd+bwd** (the training-relevant number). Whole-run speedup is Amdahl-bounded
by R's fraction `f` of the step (§4) — the frozen scorer fwd+bwd dominates the step, so the realized
whole-run gain is smaller than 4.69× and must be MEASURED (§3/§5), not asserted.

## 2. mx.compile — MEASURED not-safe on the score-critical path (a real negative)

MEASURED on this host (M5 Max, current MLX), `mx.compile` of the R op at real 384×512:
- **forward NOT bit-identical: max\|Δ\|=4.82e-3** — `mx.compile` reintroduces fp-contraction (fma) that the
  fused kernel deliberately disables (`#pragma clang fp contract(off)`); this shifts camera pixels across
  the **uint8 STE round boundary → flips d_seg argmax pixels** = a NO-FAKE score risk.
- VJP not bit-identical (9.5e-6); speed only **1.11×** on R.

Therefore `--mx-compile` is wired as a **fail-closed gate** (`maybe_enable_mx_compile_r`): enabling it runs
the startup bit-identity check and **RAISES** (never silently drifts d_seg) unless the compile is genuinely
bit-identical. On this host it correctly fails closed (verified by test). **Verdict: do NOT use mx.compile
on the R/loss path; the contraction-off fused Metal kernel is the correct fast R.**

## 3. Timing instrumentation (`--profile-timing`)

Per-epoch (emitted at `eval_every` cadence to avoid spam) `{"stage":"profile_timing", ...}`:
`t_epoch_s`, `t_step_fwd_bwd_opt_ema_s` (the fused value_and_grad step + opt + ema — INR+R+scorer+loss+
backward are fused inside `value_and_grad` and are NOT cheaply separable without perturbing barriers),
`t_verdict_s`, `t_overhead_s`, plus an **isolated in-situ R micro-bench** (`R_isolated`: ref vs fused, fwd
and fwd+bwd at the real render res) and `R_fraction_of_step_est = R_fwdbwd_ms/1e3 * frames_per_epoch /
t_step_s`. This measures **R's fraction `f` directly** (honest — an isolated R micro-bench, not fake
per-op barriers) → the realized whole-run speedup follows by Amdahl. Zero added work + byte-identical when
off.

## 4. Realized whole-run speedup — the Amdahl frame (estimate; profile run confirms)

`whole_run_speedup = 1 / ((1 - f) + f / su_R)`, with `su_R = 4.69` (fused fwd+bwd). R fwd+bwd ≈ 6.17
ms/frame (ref). The per-frame step is **scorer-dominated** (frozen SegNet EfficientNet-B2 + PoseNet
FastViT-T12 fwd+bwd), so `f` is modest:

| assumed `f` (R share of step) | realized whole-run speedup |
|---|---|
| 0.15 | ~1.13× |
| 0.20 | ~1.19× |
| 0.25 | ~1.24× |

**Best MEASURED-so-far estimate: ~1.1–1.25× whole-run** (small-scale/contended — the R op speedup 4.69× is
firm; `f` is the unknown the `--profile-timing` run measures). The bigger compute lever is the scorer
fwd+bwd itself (grouped-backward ~17× is already active; the matmul-drift Kahan kernel in §4b is the next).

## 4b. Drift-narrowing (#89 open half) — MEASURED per-op attribution + fix

Per-op MLX-GPU vs torch-CPU fp32 authority (max\|Δ\|), measured on this host (synthetic real-ish inputs):

| op | max\|Δ\| |
|---|---|
| conv 1×1 (96→96) | **0.0** (bit-identical) |
| conv 3×3 s1 (32→64) | 1.67e-6 |
| conv 3×3 s2 grouped-dw (96) | 2.38e-7 |
| **dense matmul (K=512)** | **1.96e-3** ← DOMINANT (~1000× everything else) |
| sigmoid / silu / gelu | 1.2e-7 / 9.5e-7 / 7.2e-7 |
| batchnorm (eval) | 1.91e-6 |

**The single dominant drift op is the dense MATMUL.** Mechanism CONFIRMED (K=512, unit-variance probe vs
fp64-true): torch-CPU fp32 = 7.1e-5, **MLX-CPU fp32 = 7.1e-5 and bit-identical to torch-CPU (0.0)**, but
**MLX-GPU fp32 = 7.3e-2** (~1000× worse). So the drift is **pure GPU fp32 parallel/split-K reduction-order
non-associativity** — NOT a precision downcast, NOT a `fast::` transcendental, NOT norms (this **REFUTES
the prompt's hypothesis with measurement**: convs/transcendentals/norms are all ≤2e-6). In the scorers this
lands on **PoseNet's dense Hydra head** (2048→512→…→12) — consistent with "PoseNet drifts more than the
conv-dominated SegNet."

**Fix (MEASURED expected gain):** a **Kahan/Neumaier-compensated fp32 K-reduction** matmul. Emulated the
fix in numpy: Kahan-fp32 vs fp64-true = **4.9e-6** — ~15,000× better than naive MLX-GPU matmul and better
than torch-CPU's own blocked sgemm. Deep-math: naive fp32 dot of length K has error O(K·u·|x||w|)
(u=2⁻²⁴); Kahan reduces it to ≈O(u), matching the sequential-sum authority to ~1 ULP.
- **Fix-spec (follow-up #212 kernel, default-off + bit-checked):** `metal_fixed_order_matmul` — a custom
  Metal matmul with compensated (Kahan) fp32 accumulation in the K-reduction, `#pragma clang fp
  contract(off)` (prevent fma re-association), bit-checked against the numpy-fp32 authority via the
  `assert_metal_matches_cpu_oracle`-style harness; route the PoseNet-head Linear adapters through it.
- **Expected gain:** matmul GPU drift 1.96e-3 → ~1e-6 (≈1000×), landing PoseNet-head MLX-GPU forward well
  inside the #89 ~1e-3–1e-4 target. Convs/transcendentals/norms already ≤2e-6 → no work.
- NOT applied this unit (a full Kahan matmul kernel needs careful GPU validation I must not run heavily
  while #205 is live); the Kahan gain is already CONFIRMED by the numpy emulation above.

**Bonus finding (test-vs-reality drift to reconcile — follow-up, NOT my regression):** the layer-ladder
tests `test_grouped_strided_conv_uses_reference_path_for_metal_backward_crux` +
`test_grouped_strided_reference_path_is_also_forward_d_seg_exactness_crux` are STALE under the active
`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` (the ~17× path): they assert the OLD `MLXReferenceConv2dAdapter`
routing, but production now returns `MLXCustomKernelStridedGroupedConvAdapter` — **measured bit-faithful to
torch-CPU at 2.38e-7** (the old 5e-2 near-tie ceiling is gone on this host/config). Reconcile in a
follow-up (update the type assertion + the stale 5e-2 residual claim). Not touched here (separate subsystem,
env-dependent, pre-existing — not caused by this unit's edits).

## 5. DEFERRED uncontended benchmark (run AFTER #205 finishes)

R's fraction of the per-frame step is scale-invariant in `--num-pairs`, so a small-pairs run measures `f`
faithfully + safely (no n600 OOM). After `pgrep -f train_levelset` returns nothing:

```bash
cd /Users/adpena/Projects/pact
for FUSED in --no-fused-r-kernel --fused-r-kernel; do
  .venv/bin/python experiments/train_levelset_witness_realized_through_R_mlx.py \
    --out-dir experiments/results/_252_rbench${FUSED} \
    --num-pairs 32 --render-h 384 --render-w 512 --epochs 6 --eval-every 2 --accum-pairs 8 \
    --mlx-device gpu --profile-timing $FUSED \
    2>&1 | grep '"stage": "profile_timing"'
done
```

Read `R_fraction_of_step_est` + `R_isolated` (uncontended fused speedup) + compare `t_epoch_s` fused-vs-ref
→ the **MEASURED realized whole-run speedup** (replaces the ~1.1–1.25× estimate in §4). For the full n600
confirmation, use the memory-safe `tools/launch_witness_run.py` path (governor + `--verdict-batch 32`) with
`--profile-timing --fused-r-kernel` and diff a few `profile_timing` rows against a `--no-fused-r-kernel`
control.

## Provenance
Host M5 Max, MLX default device GPU, `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` active. All numbers measured
2026-07-03 with #205 contending on the GPU (absolute times conservative; ratios ~fair). Tests:
`test_metal_fused_r_operator.py` 25/25 · `test_witness_r_op_dispatch_wiring.py` 8/8. Ruff F821 clean;
py_compile clean.
