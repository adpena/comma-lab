# MLX-GPU scorer throughput plan — ranked, safe-first, each gated on BOTH terms at the real n (2026-06-12)

**Authority:** planning memo. No MLX-GPU was run for this. Every predicted speedup is a **PREDICTION** with
its basis stated; none is reported as measured. Frontier UNMOVED 0.19109982 `[contest-CPU]`. The job of this
plan is to make the *later* GPU run a sure thing — by ordering levers safe-first and binding every one to the
canonical BOTH-TERMS acceptance gate at the REAL n.

**The gate this plan depends on:** `tac.mlx_pr95_port.speedup_acceptance_gate.evaluate_descent_equivalence`
(+ `gradient_cosine_precheck_verdict`), driven by `experiments/measure_descent_equivalence.py` (now wired to
adjudicate through the gate) and the thin CLI `tools/run_speedup_acceptance_gate.py`. The gate REJECTS any
candidate whose d_pose diverges even when d_seg is perfect, REFUSES a d_seg-only trajectory, and FLAGS any
PASS obtained at n < 600 as provisional. **No lever below is admitted to a real n600 run without a PASS at
n600 (not n8) on BOTH d_seg AND d_pose.**

---

## (a) WHERE the FP32-exact override is, and WHY it forces the slow path

The "FP32-exact override" is **not** a global dtype flag. It is a **per-layer fallback to a Python-loop,
fixed-accumulation-order reference convolution** for exactly the layers where MLX's native Metal kernel was
found to be either numerically drift-prone or gradient-wrong. Three surfaces, all in
`src/tac/local_acceleration/mlx_scorer_adapters.py`:

1. **`torch_conv2d_to_mlx` (L1146)** — the production conv dispatcher. For a **strided grouped/depthwise
   Conv2d** (the EfficientNet-B2 downsample blocks) it does NOT use native `mx.nn.Conv2d`. Instead:
   - if `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` → `MLXCustomKernelStridedGroupedConvAdapter` (native forward +
     custom Metal backward — the kernel that **diverged at n600 on pose**; OUT until fixed);
   - else (default) → `MLXReferenceConv2dAdapter(accumulation_mode="fixed_fp32")` — the **Python-loop,
     element-wise multiply-accumulate, fixed-order fp32** conv (`_reference_conv2d_nhwc`, L457). Its own
     docstring (L1071) calls it the **"~13–35× slower Python-loop"** path.
   All **dense / non-strided** convs DO go through native fast `mx.nn.Conv2d`.
2. **`MLXExplicitSpatialConv2dAdapter` (L406)** — an explicit-spatial Python-loop conv used for the
   **SegNet head logits** (L400) and the **SE-block fc1/fc2** (L180–181), kept fp32-explicit for
   decision-boundary numerical sensitivity (argmax ties).
3. **`accumulation_mode` (L494)** — `fixed_fp64` (MLX-CPU) or `fixed_fp32` (Metal); these reference paths cast
   inputs+weights to fp32/fp64 and accumulate in a deterministic loop order to preserve CPU↔GPU gradient
   parity.

**Why this is the handbrake:** these reference paths were introduced to guarantee CPU/GPU forward+gradient
parity for the numerically sensitive layers (the 2026-06-11 drift audit). They run a Python `for kh: for kw:
for c:` loop with one MLX op per tap instead of one fused Metal conv — so the GPU spends most of its time in
op-dispatch overhead on tiny tensors, which is why MLX-GPU measured **26.6 s/step SLOWER than torch-CPU's
19.4 s/step**: the GPU is running the bandwidth-bound depthwise convs through a Python loop, not a kernel.
The fast native `mx.conv2d` exists; it is deliberately bypassed for these layers for *correctness*, not
because the GPU is incapable.

**Is bf16/fp16 fwd+bwd plausibly descent-equivalent, given exact eval is always torch-CPU?** Yes,
*plausibly* — and that "plausibly" is precisely what the gate is for. The REPORTED d_seg/d_pose are always
recomputed on the torch-CPU authority (the trainer's `exact_d_seg`/`mean_d_pose` use the torch bridge
regardless of gradient backend), so reduced precision can only affect the **gradient DIRECTION**, never the
score we trust. Reduced-precision gradients are a standard, well-understood training tool. BUT — the n600
incident proves a "good cosine" gradient can still diverge over many steps, and the failure was on **pose**,
exactly the axis where the frontier value (~3.4e-5) is small enough that fp16 gradient **underflow** is a
real concern. So bf16/fp16 is a *candidate*, not a free win; it is admitted only by a PASS at n600 on BOTH
terms.

---

## The ranked levers (safe-first)

### Rank 1 — `mx.compile` op-fusion of the frozen scorer graph  ·  SAFEST FIRST WIN
- **Mechanism:** wrap the frozen-scorer forward (and the `mx.value_and_grad` closure) in `mx.compile`. MLX's
  JIT fuses elementwise/pointwise ops and reduces Metal kernel-launch + intermediate-round-trip overhead —
  the same role CUDA graphs play, achieved by compilation. The scorer weights are FROZEN, so the graph is
  static and an ideal `mx.compile` target (shapes constant across steps at fixed n).
- **Precision change:** NONE. Fusion is correct-by-construction — it changes scheduling, not arithmetic. The
  numerics are bit-for-bit (modulo op-reordering that MLX already considers safe).
- **Predicted speedup:** **PREDICTION 1.2–1.8× on the scorer fwd+bwd → ~1.2–1.7× total step.** Basis: MLX
  vendor docs + WWDC25 "Get started with MLX" describe `mx.compile` fusion as the standard way to cut
  launch/bandwidth overhead; the depthwise-conv graph here is launch-overhead-heavy (especially if any
  Python-loop reference layers remain), so fusion has real headroom. Lower bound 1.2× is conservative; this
  is NOT a precision win so it stacks UNDER the Amdahl precision cap, not against it.
- **Risk:** LOW. No precision change ⇒ the gradient direction is unchanged ⇒ descent-equivalence is
  near-certain. The only failure modes are (i) `mx.compile` recompiling on shape change (avoid by fixing the
  batch/n) and (ii) a compile-time error on an unsupported op (caught immediately, not a silent drift).
- **Validation it needs:** the **cosine pre-check should be ~1.0 by construction** (no precision change);
  still run the **bounded n600 A/B through the gate** to confirm zero regression and to bank the timing.
  Because it cannot change direction, a PASS here is high-confidence — but the gate run is cheap insurance and
  the canonical record. Do Rank 1 FIRST: it de-risks the whole program and gives a clean fused baseline that
  the precision levers then build on.

### Rank 2 — Replace the Python-loop reference convs with native `mx.conv2d` (fp32, where parity holds)
- **Mechanism:** for the strided-grouped + SE + head layers currently on the Python-loop reference path,
  switch to native `mx.nn.Conv2d` **at the same fp32 precision** — i.e. recover throughput WITHOUT changing
  precision, by proving the native kernel's *forward+gradient* now matches the reference closely enough on the
  real scorer activations. (This is the layer-level sibling of "the sibling agent is fixing the custom Metal
  backward".)
- **Precision change:** NONE (stays fp32). This is the cleanest possible un-handbraking.
- **Predicted speedup:** **PREDICTION 3–13× on JUST the affected layers** (the reference path is documented
  as 13–35× slower than native for these layers; recovering even part of that is large). Total-step effect
  depends on what fraction of the 98.36% those specific layers are — bounded by the Amdahl table in the
  profile memo.
- **Risk:** MEDIUM. This is exactly where the custom-backward kernel got the **pose gradient wrong** — the
  native strided-grouped VJP had pose cosine ~0.025 (the documented blow-up). The reference path exists
  *because* of that. So native-conv recovery is admissible ONLY per-layer, ONLY where the
  forward+gradient parity is re-proven. **This lever is gated on the gradient-cosine pre-check (BOTH paths
  ≥ 0.999) AND the full n600 BOTH-terms A/B.** It is Rank 2 not Rank 1 because it touches the exact surface
  that already failed once. Coordinate with the sibling fixing `metal_grouped_conv_backward.py` — do not
  collide.
- **Validation it needs:** cosine pre-check on BOTH SegNet-path and PoseNet-path cotangents per layer
  (catches the ~0.025 pose-cosine signature instantly) → THEN the bounded n600 gate. A small-n PASS is
  explicitly insufficient (the incident's lesson).

### Rank 3 — fp16 scorer fwd+bwd on MLX-GPU  ·  PRECISION TIER, fully gated
- **Mechanism:** cast the frozen-scorer forward+backward to **fp16** (keep the render leaf + loss reduction in
  fp32; apply loss-scaling on the backward to avoid pose-gradient underflow). Halves bytes-moved for the
  bandwidth-bound convs.
- **Precision change:** YES (fp16). The whole BOTH-terms gate exists for this tier.
- **Predicted speedup:** **PREDICTION ~2× on scorer fwd+bwd → ~1.97× total** (first-order bandwidth halving;
  the Amdahl cap from the profile). Stacks multiplicatively with Rank 1 fusion → combined **~2.9–3.5× total**
  is the realistic optimistic target. Basis: bandwidth-bound depthwise convs + the ~½ byte traffic of fp16;
  web sources confirm fp16/bf16 beat fp32 on Apple Silicon precisely because of reduced memory-access
  bandwidth.
- **Risk:** HIGH on pose, the documented failure axis. **fp16 has a narrow exponent range** → near the
  frontier d_pose ≈ 3.4e-5, fp16 gradients can **underflow to zero** (the classic mixed-precision failure),
  which is a pose-DIRECTION corruption — the exact n600 class. Mitigation: loss-scaling (standard) + the gate.
  fp16's 10-bit mantissa is a BETTER fit for the tiny pose values than bf16's 7-bit (see Rank 4), so prefer
  fp16-with-loss-scaling over bf16 for the pose-sensitive path.
- **Validation it needs:** the FULL ceremony — (1) cosine pre-check BOTH paths; (2) bounded **n600** BOTH-terms
  A/B with `generalization_warning` cleared (n8 is explicitly NOT enough — a ~0.07%/step error compounds over
  n600's ~75× more steps/epoch into divergence); (3) tighten the gate's `pose_abs_tol`/`pose_rel_tol` for this
  tier since pose is where it breaks. Only on a clean n600 PASS does fp16 drive a real basin run.

### Rank 4 — bf16 scorer fwd+bwd on MLX-GPU  ·  PRECISION TIER, gated, prefer fp16 over this
- **Mechanism:** same as Rank 3 but bf16. bf16 has FP32's exponent range ⇒ **no loss-scaling needed, no
  gradient underflow** — attractive for the small pose values.
- **Precision change:** YES (bf16, 7-bit mantissa).
- **Predicted speedup:** **PREDICTION ~2× scorer fwd+bwd** (same bandwidth basis as fp16). BUT web sources
  report **fp16 is consistently faster than bf16 on Apple Silicon** (M1/M2 lack native bf16 → emulated with a
  slight rounding deviation; on M5 Max bf16 may be native — UNVERIFIED, must be measured). So bf16's
  throughput may *underperform* fp16 here.
- **Risk:** MEDIUM-HIGH. No underflow (range = fp32) but the **7-bit mantissa is a coarse quantization of the
  tiny near-frontier pose values** — it trades underflow risk for resolution risk on exactly the axis that
  matters. Whether that coarseness corrupts the pose DIRECTION over n600 is unknown ⇒ gate it.
- **Validation it needs:** identical to Rank 3 (cosine pre-check BOTH paths → bounded n600 BOTH-terms A/B).
  Run fp16 and bf16 as TWO candidates through the SAME gate and pick the one that both PASSES and is faster on
  the M5 Max — do not assume; the fp16-vs-bf16 throughput on this hardware is a measurement, not a given.

### Rank 5 — NAX / fast-matmul-kernel eligibility  ·  RESEARCH / LAST
- **Mechanism:** Apple's NAX (Neural Accelerator) / the fast matmul path accelerates **matmul-shaped** ops.
  EfficientNet-B2's bottleneck is **depthwise (grouped) conv**, which is NOT matmul-shaped — depthwise conv has
  ~1 MAC/weight and no GEMM structure, so it is the *least* NAX-eligible op in the graph. The 1×1 pointwise
  convs and the PoseNet FastViT attention/MLP ARE matmul-shaped and could benefit, but those are the cheap
  part of the profile.
- **Predicted speedup:** **PREDICTION small on this workload** — the dominant cost (depthwise convs) is not
  NAX-eligible; NAX would accelerate the already-cheap 1×1 + attention. Likely < 1.2× total here. Worth a
  measurement only AFTER Ranks 1–3 land, and only if a profiler shows the 1×1/attention share has grown.
- **Risk:** LOW correctness (matmul kernels are well-trodden) / LOW expected value (wrong op shape for the
  bottleneck). De-prioritized on EV grounds, not risk grounds.
- **Validation it needs:** if pursued, same gate — but the honest prior is that this is not where the win is.

---

## The recommended sequence (de-risked)

1. **Rank 1 (`mx.compile` fusion)** — no precision change, near-certain descent-equivalence, gives a clean
   fused baseline. Land it first; bank the timing through the gate.
2. **Rank 2 (native conv recovery, fp32)** — coordinate with the sibling fixing the Metal backward; per-layer,
   gradient-cosine-gated, n600 A/B. This is the biggest *correct* throughput lever if parity re-proves.
3. **Rank 3 (fp16 + loss-scaling)** — the precision tier; full n600 BOTH-terms ceremony, tightened pose tol.
4. **Rank 4 (bf16)** — only if fp16 underperforms on throughput AND passes the gate; pick the gate-PASSing,
   faster of {fp16, bf16}.
5. **Rank 5 (NAX)** — measure only if the profile shifts toward matmul-shaped ops after 1–3.

**The honest cap (from the profile memo):** the scorer fwd+bwd is 98.36% of the step, so even a perfect
scorer caps total speedup at ~61×, and the realistic precision+fusion program caps at **~3–5× total step**.
A combined Rank-1 fusion (~1.2–1.8×) × Rank-3 fp16 (~2×) realistically lands **~2.9–3.5× total** *if both pass
the gate at n600*. Past ~5× you are removing the scorer from the per-step loop (distilled surrogate — a
different program with its own large fidelity risk), not optimizing the kernel.

## Cross-references
- Profile: `.omx/research/scorer_step_profile_20260612.md` (the 98.36% / Amdahl ceiling this plan obeys).
- The lesson: `.omx/research/mlx_custom_backward_DIVERGES_at_n600_pose_gradient_20260612.md` (why every lever
  is gated on BOTH terms at the real n).
- The gate: `src/tac/mlx_pr95_port/speedup_acceptance_gate.py` + tests + `tools/run_speedup_acceptance_gate.py`.
- FP32 handbrake surfaces: `src/tac/local_acceleration/mlx_scorer_adapters.py`
  (`torch_conv2d_to_mlx` L1146, `MLXReferenceConv2dAdapter` L592, `MLXExplicitSpatialConv2dAdapter` L406,
  `_reference_conv2d_nhwc` L457, `_custom_metal_backward_enabled` L1124).

## Web research (folded into the risk column above)
- `mx.compile` = JIT op-fusion, same role as CUDA graphs (launch-overhead reduction by compilation);
  correct-by-construction, no precision change → Rank 1 is the safe first win.
  ([MLX/WWDC25](https://developer.apple.com/videos/play/wwdc2025/315/),
  [sglang #19145](https://github.com/sgl-project/sglang/issues/19145))
- fp16/bf16 beat fp32 on Apple Silicon because depthwise/conv is **memory-bandwidth-bound** (half the bytes);
  **fp16 is consistently faster than bf16** on Apple Silicon, and M1/M2 lack native bf16 (emulated, slight
  rounding deviation) → prefer fp16, verify bf16-native on M5 Max by measurement.
  ([fp16 vs bf16 on Apple Silicon](https://news.ycombinator.com/item?id=36575443),
  [MLX bf16 emulation issue](https://github.com/jundot/omlx/issues/604))
- fp16 narrow exponent range ⇒ gradient **underflow** for small loss values (the near-frontier d_pose ≈ 3.4e-5
  regime) ⇒ **loss-scaling required**; bf16 has fp32 range (no underflow) but 7-bit mantissa (coarse on tiny
  pose values). This is exactly why the precision tier is GATED on pose, not assumed.
  ([mixed-precision / loss-scaling](https://mbrenndoerfer.com/writing/mixed-precision-training-fp16-bf16-loss-scaling),
  [runpod mixed precision](https://www.runpod.io/articles/guides/fp16-bf16-fp8-mixed-precision-speed-up-my-model-training))
