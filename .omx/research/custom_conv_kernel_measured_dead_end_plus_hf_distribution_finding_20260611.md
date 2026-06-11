# Custom conv kernel — MEASURED dead end (any language) + HF-distribution finding (2026-06-11)

**Authority:** `[macOS-MLX research-signal]`, NON-PROMOTABLE, $0, **no MPS** used. Frontier pointer
UNMOVED 0.19109982 [contest-CPU], 177,169 B. This is a measured negative + a corrected prior — not a
pointer move. Sources are this session's two completed kernel agents (headroom prototype `ab9c4d`,
contribution-path `a4f777bb`) + two HF web searches.

## The question this closes

Operator asked: build our own grouped/depthwise-conv kernel ("any language … Mojo, JAX, assembly,
Swift, Rust, anything") to unblock the local-30k-epochs-on-MLX-GPU dream; "look at hf kernels too";
"add our port and kernels work to hf." This memo records the MEASURED answer so it is not re-opened
without new evidence.

## Finding 1 — a custom FORWARD depthwise kernel = ~0× headroom, in ANY language (genuine negative)

- Depthwise conv on SegNet's EfficientNet-B2 is **memory-bound**: arithmetic intensity 3.07 FLOP/byte;
  forward DW traffic 138 MB → bandwidth floor ≈ 314 µs/image at ~450 GB/s (M5 Max).
- MLX-default is **already near that ceiling once amortized**: a 5×5 720ch layer is 194 µs at b=1 →
  **30.5 µs/img at b=32 = only ~3× off the bandwidth floor.** The "20–34× off floor" at b=1 is
  **kernel-launch/dispatch overhead, not bad compute.**
- The hand-written `mx.fast.metal_kernel` prototype (naive + coalesced tiled) was **bit-exact vs
  `mx.conv2d` (max_abs_err 0.0)** but **slower-or-equal**: 1.10× at b=1 (a hair, from lower fixed
  dispatch), **0.44× at b=32** (loses 2× — MLX's tuned grouped-conv kernel is genuinely better).
- **Therefore language is MOOT for the forward kernel.** A bandwidth ceiling cannot be beaten by
  Mojo / JAX / Rust / Swift / assembly / MSL — they all hit the same ~450 GB/s wall MLX already hits.
  Classified **genuine-paradigm negative** (Catalog #307): bandwidth-bound, not a janky prototype and
  not a wrong-operating-point artifact (measured across b=1…32 and across the 3×3 and 5×5 regimes).
- **Backward is the hard, already-worked-around part.** `mlx_scorer_adapters.py:1070-1076` already
  forces strided depthwise *backward* onto a Python-loop reference because "MLX Metal reverse-mode
  produces extreme gradients." A forward-only kernel gives MLX no autodiff backward; hand-writing
  grad-w + grad-x kernels (≈2× the work) to chase a forward that is already 0× is not justified.

## Finding 2 — MPS is OFF the table (operator standing directive, reaffirmed 2026-06-11)

The contribution agent floated "benchmark MPSGraph depthwise as a zero-kernel local convenience."
**STRUCK.** Operator: "remember MPS has been trash and we are looking for MLX" (and the standing
MPS-NEVER authority rule + the 95.5%-argmax-corruption receipt). Stay MLX-native. It would not have
helped anyway — there is no forward headroom to capture (Finding 1).

## Finding 3 — the ONLY language-relevant kernel win is a FUSED MBConv megakernel (big bet, NOT on the score path)

The full 23-depthwise stack stays **7–9× off the bandwidth floor even at b=32** because each of 23
layers is a separate kernel launch and the small late stages (16×12) are pure launch-bound. The only
custom kernel that could win is a **fused multi-layer / whole-MBConv megakernel** that collapses
launches — and it would have to beat MLX's already-good per-op kernels *end-to-end*. That is a large,
uncertain effort, it is where a fast language (MSL via the HF Metal kernel-builder) would actually
matter, and it is **NOT on the score critical path** (the pointer-mover is paid n600, below). Recorded
as a real-but-deferred local-throughput lever, not a current build.

## Finding 4 — HF-kernels prior CORRECTED: HF HAS a live Metal path + Hub distribution

My strong prior ("HF `kernels` = CUDA/ROCm/Triton only") was **WRONG**:
- `huggingface/kernel-builder` ships a **Metal/Apple-Silicon builder** (`docs/metal.md`; macOS 26.0+
  ARM64, Xcode 26.x + Metal Toolchain).
- The HF Hub **already hosts community Metal kernels** — `kernels-community/mlx-quantization-metal-kernels`
  — **auto-downloaded from the Hub on first use, no manual compilation** (the exact "ship a kernel as
  a Hub artifact" pattern the operator asked about).
- HF has a first-class **MLX-on-Hub integration** (`huggingface.co/docs/hub/en/mlx`) — MLX model ports
  are a supported Hub artifact class.

**So the operator's "add our port and kernels work to hf" is a REAL, live destination** — for (a) our
MLX scorer PORTS (the bit-exact numpy-portable + MLX scorer bridges) and (b) any future fused
megakernel — distributed as auto-downloadable Hub artifacts. **This is an OSS-distribution / portability
contribution, DRAFT-gated pending operator approval per CLAUDE.md "Public Disclosure Hygiene" (keep
private infra / local paths / credentials out; public release is intentional, not automatic). It is NOT
a speed win and NOT a score move** — it is a way to share the portable ports we already built.

## Finding 5 — MLX upstream contribution path (if ever pursued): the NAMED gap only

Premise "MLX has no fast depthwise" is only partly true. MLX already ships `depthwise_conv_2D_gpu`
under a tight guard (`C_per_group==1 && O_per_group==1 && k≤7 && stride≤2 && C%16==0 && C==O`). The
verified remaining gaps: **channel-multiplier > 1** (EfficientNet/MobileNet — and EfficientNet-B2 IS
the contest SegNet backbone), k>7 / stride>2, and **general grouped conv** (issue #1409: 10–150×
slower than PyTorch MPS). Upstreamable form = **MSL kernel in `conv.metal` + C++ dispatch in `conv.cpp`
+ VJP + tests + benchmark** (forward-only is not mergeable; Mojo/Rust/JAX are not upstreamable into
MLX). The maintainers are *actively* working this area (the lead opened #2369 himself, fixed via
#2567), so only a **scoped, named-gap PR with VJP** is worth it — not a from-scratch kernel.

## The redirect (the actual critical path — unchanged)

1. **Local-30k is NOT unblocked by a kernel (any language) — measured.** The real local-throughput
   levers are software: **scorer batching** (amortize the 7–9× launch overhead) + **op-fusion**. Even
   those leave n600 at months, not the 10–50× a paid CUDA GPU gives.
2. **The pointer-mover is the PAID Modal n600 PR95-scale score-aware retraining run**, gated on the
   latent-heavy capacity de-risk (agent `a435737`, still running) reaching the corrected bar
   (d_seg ~0.001 + pose collapsed + sub-frontier bytes). $100 Modal grant stands ready to BUY that
   exact row the moment the candidate is measured-ready.
3. **HF-distribution of the portable ports** = a parallel, DRAFT-gated OSS contribution; it compounds
   the portability dream (numpy-portable inflate is already bit-exact) but does not move the score.

## Bottom line (non-sycophantic)

The custom conv kernel — in Mojo, JAX, Rust, Swift, assembly, or MSL — is a **measured dead end for
both speed and score**: depthwise is bandwidth-bound and MLX is already near the M5's ceiling. The one
real surprise is the *opposite* of the kernel question: **HF genuinely has a Metal kernel-builder + Hub
distribution** (but see the same-day correction below — it targets torch-MPS, not MLX). Frontier UNMOVED
0.19110; the pointer-mover stays the capacity-de-risked paid n600.

## ⚠️ SAME-DAY CORRECTION (2026-06-11, from the MLX-surface agent) — Findings 3 & 4 overstated

Two claims above are corrected by a later measurement (NO-FAKE catch-and-fix; original text preserved
above for provenance):

1. **Finding 4 (HF as a distribution destination for OUR kernel) was OVERSTATED.** HF `kernel-builder`'s
   Metal path produces **PyTorch/MPS extensions, NOT MLX kernels** (precedent
   `kernels-community/mlx-quantization-metal-kernels` is MLX-*derived* but packaged + auto-loaded as a
   **torch-MPS** artifact). So HF kernel-builder is **NOT** a distribution channel for our MLX scorer
   kernel, and it rides **MPS — which is forbidden as authority in this lab**. Distribution of an *MLX*
   artifact is via the MLX-on-Hub model integration / a normal repo, not kernel-builder. The honest
   residual: HF can host our MLX *ports* as MLX-on-Hub models (DRAFT-gated), but the "ship our kernel as
   an HF Metal artifact" framing only applies to a torch-MPS kernel we are not building.
2. **Finding 3 (the megakernel attacks the launch-overhead gap) is an UNVALIDATED HYPOTHESIS.** MLX's
   lazy graph **already batches the whole forward into ONE Metal command buffer** at `mx.eval` (the
   per-dispatch ~1ms overhead is already collapsed; ZMLX prior art shows the residual launch lever is
   single-digit %, not 7–9×). So the 7–9×-off-floor gap is **more likely slow MLX depthwise `conv_general`
   COMPUTE** (issue #1409: 10–150× slower than MPS) + HBM re-reads — which a fusion megakernel does NOT
   fix. **Two $0 checks GATE any megakernel build:** (a) run the full scorer forward as one lazy graph
   with a single terminal `mx.eval` (no interior `.item()`/`np.array()` syncs); (b) wrap in `mx.compile`
   and re-measure. If those close the gap, there is **no megakernel to build** — and building one anyway
   would be a fake optimization (optimizing a non-bottleneck). The design/profile agent runs this audit.

## ✅ MEGAKERNEL RESOLVED (all 3 agents converge) — DON'T build it; the real win is "g0" (a Python-loop fix)

The design/profile agent ran the audit. **The 7–9×-off-floor "launch gap" is ~98% a Python-loop
reference-conv FALLBACK artifact, not intrinsic launch overhead or a fusion prize.** The repo routes the
4 strided-depthwise convs through `mlx_reference_conv2d_nhwc` (a Python triple-loop, `mlx_scorer_adapters.py:456/591/1064`)
which emits **~22,464 elementwise MLX ops = ~98% of the encoder's op-launch count** (vs ~452 native
launches for the entire rest of the net). All three agents converge:

- **g0 (the MVP, measured 5.7× @ b=32 / 11.2× @ b=1): swap the 4 strided-dw convs to native `mx.conv2d`
  on the forward/eval path.** A ONE-FUNCTION change, no kernel authoring, forward-parity already tested.
  This is the real, cheap win — not a megakernel.
- **g0 is FORWARD/EVAL-tail SAFE only.** The 4 convs are on the reference path *because native MLX Metal
  reverse-mode produces extreme gradients for strided grouped Conv2d* (a real upstream MLX bug, worth
  filing). So g0 accelerates eval/monitoring + the exploit-tail VJP forward, NOT the training backward
  through those 4 layers.
- **`mx.compile`** (already in the repo, 30 sites; composes with `value_and_grad` for fwd+bwd) is MLX's
  CUDA-Graph analog and the other cheap lever — a measurement (is the conv step inside a compile
  boundary?), not a build.
- **A hand megakernel (g1 fuse-one-MBConv … g5 whole-auth-eval) is NOT worth building now:** every
  dedicated-megakernel result (Mirage/Hazy/ThunderKittens) is CUDA-only + inference-only + batch-1-decode
  mechanism = a mismatch to batch-32 CNN *training*; the one architecturally-matched paper (FCM, DW+PW+BN
  fusion, 3.7× memory-bound) is CUDA-only + forward-only = a design template, not portable code; g1's
  hard part is the 4-layer strided-dw *backward* (the same instability g0 sidesteps); g5 gains are capped
  by g0+g1 and sacrifice the swappable-tail modularity.

**Disposition (DAG-first, NO-FAKE):** g0 is a REAL, low-risk local-throughput win — but a SECONDARY MEANS.
Its training use-case (faster local 30k of a smaller basis) was just REFUTED by the capacity verdict
(`capacity_verdict_smaller_basis_by_rate_REFUTED_pivot_to_waterfiller_20260611.md`). Its remaining real
value is accelerating the **waterfiller pivot's exact-ΔS forward loop + the atlas margin-field/Jacobian
VJP** — so g0 should be landed WHEN the waterfiller pivot (#30) is activated (serving an imminent
exact-row loop), not as standalone infra now. The megakernel (g1+) is SHELVED with a clear reactivation
criterion: only if the reframed frontier-class #90 paid retrain is pursued AND its forward loop is the
measured bottleneck. **No megakernel is built this turn; the frontier is UNMOVED 0.19110; this work
refused a fake build and identified a real one-function eval-loop accelerator for the pivot.**
