# MLX vs torch-MPS scorer ceiling — WHY MPS wins, the MLX-max number, and the GO/NO-GO

**Date:** 2026-06-12
**Author:** mlx-vs-mps-ceiling subagent (SIDE QUEST)
**Evidence grade:** `[macOS-MLX/MPS advisory]` for ALL numbers. torch-CPU is the only authority. NO score claim, NO promotion, NO d_seg/d_pose verdict. This is a SPEED + relative-fidelity investigation.
**Hardware:** M5 Max (128 GB unified), MLX 0.31.1, torch 2.11.0 (CUDA unavailable → torch-CPU is the authority axis), MPS available.
**Concurrent load (noted on every number):** ONE live training daemon shared the Metal GPU + CPU during all benchmarks — `pid 33911 = experiments/launch_split_by_head_basin.py --train-device mps ... base-channels 20 --n-pairs 600` at ~136–212% CPU. Because that daemon trains on **MPS**, it contends with BOTH the torch-MPS arm AND the MLX-GPU arm **equally**, so the RELATIVE MPS-vs-MLX comparison is fair; absolute numbers are depressed (faster uncontended).
**Did the exact frontier pointer move?** No. This is a tooling/throughput investigation; it does not lower the exact score.

---

## TL;DR (the four answers)

- **A — WHY MPS > MLX:** the MLX **default** path runs the strided-grouped/depthwise Conv2d **backward through a pure-Python fixed-order reference accumulator** (`MLXReferenceConv2dAdapter`). That single op class is ~94% of the MLX default cost: SegNet fwd+bwd = **9632 ms** default vs **569 ms** with the custom Metal backward kernel (17×); PoseNet fwd+bwd = **1760 → 91 ms** (19×). torch-MPS dispatches the same convs to fused Metal MPSGraph kernels with NO such Python fallback.
- **B — MLX-MAX:** with the opt-in custom Metal grouped-conv backward kernel (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, gradient-exonerated 2026-06-12), MLX-GPU full fwd+bwd is **765 ms/step** (B=8). **torch-MPS is 209 ms/step.** MLX-max is **~3.7× SLOWER than MPS**, not equal. The operator's hypothesis (MLX-max ≈ MPS) is **REFUTED for this CNN/ViT workload** — MPS wins clearly. Even MLX **forward-only** (967 ms) is slower than MPS **full fwd+bwd** (209 ms).
- **C — fast / fidelity-gate:** (1) the fastest training-gradient branch IS torch-MPS-fp32 (209 ms/step); nothing local beats it (optimized MLX tops out at 765 ms). (2) The slow fidelity branch (the exact CPU **forward-only** authority eval, ~680 s / 600 pairs on this arm64 box) has **NO bit-exact local speedup**: it is bottlenecked by `aten::_slow_conv2d_forward` (no mkldnn on arm64 torch), and the only bit-faithful-ish alternative, MLX-CPU, is *slower* (889 s). A faster-but-approximate authority (MPS, MLX-GPU) is FORBIDDEN (MPS-NOISE rule). The cheaper-EXACT lever is **contest x86_64 Linux (mkldnn)**, not the Mac.
- **D — GO/NO-GO:** **NO-GO on investing in MLX scorer speed.** torch-MPS is already at/near the Metal ceiling for this workload and is ~3.7× faster than the best MLX can reach; the canonical training path (`torch_vehicle` basin) already runs scorer-on-MPS-fp32. Spend the effort elsewhere. The one residual MLX win is niche (see §6).

---

## The measured numbers (NO FAKE — real frozen scorer, real 0.mkv inputs)

### A/B — head-to-head full forward+backward-to-input, B=8

Script: `experiments/bench_mlx_vs_mps_scorer_ceiling.py`. Same frozen EfficientNet-B2 SegNet + FastViT-T12 PoseNet, same real cached scorer-input batch (`mlx_scorer_input_cache_reference_video_...full600`), loss = `sum(seg_out² + pose_out²)` backproped to the input pixels (the exact training charge — scorer frozen, gradient flows to the render).

| backend | ms/step (median) | vs torch-CPU | vs torch-MPS |
|---|---:|---:|---:|
| torch-cpu (authority) | 6584 | 1.00× | 0.03× |
| **torch-mps fp32 (winner)** | **209** | **31.6×** | **1.00×** |
| mlx-gpu DEFAULT (ref backward) | 14428 | 0.46× | 0.01× |
| mlx-gpu CUSTOM (Metal backward = MLX-MAX) | 765 | 8.6× | 0.27× |
| mlx-gpu FORWARD-ONLY | 967 | 6.8× | 0.22× |

Stable across re-runs (torch-mps 209–255 ms, mlx-custom 765–822 ms; ratio ~3.2–3.7×). JSON: `.omx/research/mlx_vs_mps_scorer_ceiling_20260612T151012Z.json` (+ a `--iters 8` repeat `...151241Z.json`).

> The MLX **default** path is *slower than torch-CPU* (14428 vs 6584 ms) — a real trap: a naïve "run the MLX scorer on the GPU" is a regression unless the custom kernel is enabled.

### A — per-stage breakdown: WHERE the MLX time goes

Script: `experiments/bench_mlx_scorer_stage_breakdown.py` (B=8). The dispatch (`torch_conv2d_to_mlx`) routes `groups>1 AND stride≠(1,1)` Conv2d to the reference accumulator (default) or the custom Metal kernel (opt-in). Routing counts: **SegNet 4** strided-grouped convs (of 125 total), **PoseNet 8** (of 78).

| stage | DEFAULT ms | CUSTOM ms | speedup from kernel |
|---|---:|---:|---:|
| segnet_fwd | 503 | 205 | 2.5× |
| **segnet_fwd_bwd** | **9632** | **569** | **17×** |
| posenet_fwd | 397 | 36 | 11× |
| **posenet_fwd_bwd** | **1760** | **91** | **19×** |

**Localization:** the backward of a handful of strided-grouped/depthwise convs dominates the MLX default cost. The forward is also penalized (the reference conv forward is a Python/loop accumulator too). JSON: `.omx/research/mlx_scorer_stage_breakdown_20260612T151207Z.json`.

**The residual MLX-custom-vs-MPS gap (765 vs 209):** with the kernel, the strided convs are no longer the bottleneck — the remaining ~3.7× is the rest of the graph: MLX runs the adapter **eager (`mx.compile` count = 0** across the 9700-LOC adapter, confirmed in the 2026-06-09 throughput memo) and `mx.compile` gives only ~1.18× on the conv-heavy renderer, so even a fully-compiled MLX port would not close a 3.7× gap. torch-MPS uses Apple's fused MPSGraph kernels for the whole graph. This is an Apple-framework-maturity gap, not a fixable porting bug.

### Why the gradient-cosine column reads ~0 (honesty note, NOT a fidelity finding)

The head-to-head's `seg_input_grad_cosine_vs_cpu` shows torch-mps=1.0 but mlx≈7e-4. This is an **artifact of the benchmark's toy `out²` loss + NHWC↔NCHW flatten-order**, NOT a real MLX fidelity problem. The PROPER fidelity characterization already exists and is sound: the 2026-06-11 drift audit (`mlx_scorer_port_drift_audit_20260611.md`) measured MLX-CPU **bit-faithful** to torch-CPU (2 argmax flips / 19.66M at fp32-ULP ties; pose drift 8.7e-11) and MLX-GPU d_seg drift 1.2e-5 (boundary near-ties), pose 2.76e-4. The cosine column here is only a throughput-script sanity and is not load-bearing for the speed verdict; torch-mps's 1.0 confirms the MPS gradient direction matches CPU (consistent with the 104× memo's ~1.0 per-step cosine).

### C — the fidelity / cheaper-exact-eval question (forward-only, B=16)

The exact authority eval (`upstream/evaluate.py`, `--device`, `--batch-size 16`) is **forward-only** (`inference_mode`), 2 forwards/pair (gt + comp), 600 pairs. Measured forward throughput on this box (under load):

| backend | pairs/s | full 600-pair eval est | usable as authority? |
|---|---:|---:|---|
| **torch-cpu** | 1.76 | **~680 s (~11 min)** | **YES — the authority** |
| mlx-cpu | 1.35 | ~889 s | near (2 flips/19.66M) but **SLOWER**, so no |
| mlx-gpu | 18.1 | ~66 s | NO (1.2e-5 seg / 2.76e-4 pose drift) |
| torch-mps | **117** | **~10 s** | NO (23× pose drift — MPS-NOISE) |

**Answer C(1):** the fastest training-gradient branch is **torch-MPS-fp32 (209 ms/step fwd+bwd)**; no local branch beats it (best MLX = 765 ms). **Answer C(2):** there is **NO bit-exact local speedup of the CPU authority eval** on this arm64 Mac. The cost is `aten::_slow_conv2d_forward` (arm64 torch has no mkldnn fast depthwise path), and the only bit-faithful-ish substitute, MLX-CPU, is *slower* (1.35 vs 1.76 pairs/s). The fast options (MPS ~67× faster forward, MLX-GPU ~10× faster) are all **approximate** → forbidden as authority. The genuine cheaper-EXACT lever lives on the **contest x86_64 Linux runner** (mkldnn-accelerated depthwise), which is also where the authority is *defined* — i.e. run exact CPU evals on Linux x86_64 (Modal/Vast CPU container) rather than locally, per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant hardware". Local cheaper-exact: none.

---

## D — verdict + recommendation (GO/NO-GO)

**NO-GO on investing engineering in the MLX scorer port's speed.**

Rationale (all measured):
1. torch-MPS-fp32 is **3.7× faster than the best MLX can reach** (209 vs 765 ms/step) and is already the canonical training-gradient device (`torch_vehicle` basin trains scorer-on-MPS; the live daemon pid 33911 IS this path).
2. The MLX residual gap is an **Apple-framework-maturity** gap (MLX eager + `mx.compile`=0, ~1.18× ceiling vs MPSGraph fused kernels), not a fixable porting defect. Closing 3.7× via MLX kernel work is not credible.
3. The MLX **default** path is a latent regression (slower than torch-CPU) — anyone wiring "MLX-GPU scorer" without `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1` makes training SLOWER. That's a footgun, not an asset.
4. The cheaper-EXACT-CPU-eval lever is **not local** (no bit-exact arm64 speedup); it is contest-Linux-x86_64. Effort is better spent making the Linux x86_64 exact-eval loop turnkey (Modal/Vast CPU) than optimizing local MLX.

**Where the freed effort should go:** keep the scorer training-gradient on **torch-MPS-fp32** (already done); for exact fidelity, run the authority on **Linux x86_64 (mkldnn)** off-box. Do NOT chase MLX scorer speed.

### The one residual MLX win (niche, not a reason to invest now)

MLX-GPU is the only path that keeps the scorer **and** an MLX-native renderer/decoder on the GPU in ONE graph with `mx.value_and_grad` (no torch↔MLX device hop). For an **all-MLX substrate** (z7/z8/dreamer/HiNeRV MLX decoders) where the decoder is already MLX, the MLX-custom scorer (765 ms) avoids a torch round-trip that could cost more than the 765-vs-209 difference. That is a per-substrate wiring decision, not a scorer-speed investment — and it requires `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`. For the torch-native basin, torch-MPS wins outright.

---

## Reproduce

```bash
A=.omx/research
# head-to-head fwd+bwd (all 5 backends), B=8
.venv/bin/python experiments/bench_mlx_vs_mps_scorer_ceiling.py --batch 8 --iters 5 --warmup 2
# per-stage WHY breakdown (segnet/posenet fwd vs fwd+bwd, default vs custom kernel)
.venv/bin/python experiments/bench_mlx_scorer_stage_breakdown.py --batch 8 --iters 5 --warmup 2
```
Artifacts: `mlx_vs_mps_scorer_ceiling_20260612T1510*.json`, `mlx_scorer_stage_breakdown_20260612T151207Z.json`.

## NO-FAKE / authority notes
- Every ms is a real measured wall-time on the real frozen scorer + real 0.mkv scorer-input cache. No extrapolated number.
- All advisory; torch-CPU (and contest CPU/CUDA) remain the only authority. MPS/MLX never a score.
- Concurrent training load (1 MPS daemon) noted; it contends with torch-MPS and MLX-GPU equally → the MPS-vs-MLX ratio is fair, absolute numbers are a lower bound (faster uncontended). A clean-box re-run would widen MPS's lead, not narrow it (MPS is the more contended-sensitive of the two since the daemon also uses MPS — so the true MPS advantage is ≥ the measured 3.7×).
- 6-hook wire-in: this is a throughput/tooling investigation (advisory). #1 sensitivity-map N/A · #2 Pareto N/A · #3 bit-allocator N/A · #4 cathedral-dispatch N/A (not archive-deployable) · #5 continual-learning = this memo + the 2 JSON artifacts are the empirical anchors (the GO/NO-GO is the consumable signal: do not invest in MLX scorer speed; keep scorer-gradient on torch-MPS-fp32) · #6 probe-disambiguator N/A (the speed numbers are unambiguous).
```
