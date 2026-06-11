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
distribution**, so the portable work we ALREADY built is shareable there (DRAFT-gated). Frontier UNMOVED
0.19110; the pointer-mover stays the capacity-de-risked paid n600.
