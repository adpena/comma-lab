# MPS "upstream-but-macOS" base_ch=20 basin — split-device wire-in + descent-equivalence gate → REJECT (pose), basin NOT launched

> **Bottom line:** the 90-104× MPS scorer speedup is REAL and the SegNet-path gradient is bit-identical to the CPU authority — but the PoseNet-path MPS gradient FAILED the both-terms descent-equivalence gate (final d_pose |gap| 7.02 > tol 1.12). The base_ch=20 MPS basin run is **NOT launched**; the torch-CPU basin (pid 42035) / Modal CUDA remain the trustworthy paths. Frontier UNMOVED. This is an honest negative that PREVENTED a fake (a pose-corrupted basin).

**Date:** 2026-06-12 (UTC)
**Subagent:** mps-basin-wirein-20260612
**Lane:** `lane_torch_vehicle_mps_gradient_basin_20260612` (L1) — sibling of `lane_torch_vehicle_pr95_readiness_20260611`
**Authority:** `[macOS-CPU advisory]` NON-PROMOTABLE. The exact d_seg/d_pose are torch-CPU for every decision; MPS is a GRADIENT backend only. A sub-frontier MPS-trained result GATES — never IS — a paired contest-CPU+CUDA exact eval (the sole pointer-mover). Frontier UNMOVED at this landing: 177,169 B `[contest-CPU]` (S=0.19109982).

## 1. The 104× MPS speedup — confirmed LIVE on this machine (2026-06-12)

`experiments/bench_scorer_mps_vs_cpu.py --batch-size 8 --iters 4` (re-run today, not the cached commit):

| backend | ms/step (fwd+bwd, B=8, full 874×1164) | speedup vs CPU | grad cosine vs CPU |
|---|---:|---:|---:|
| torch-CPU fp32 | 15311.0 | 1.0× | (authority) |
| **torch-MPS fp32** | **168.3** | **90.95×** | **1.0075** |

The committed anchor (`8374cf231`) reported 104× (17.3→0.166 s/step); the 90.95× today is the same regime (B/iter/thermal variance). The single enabling patch is `tac.torch_mps_compat.patch_batchnorm_contiguous_for_mps` (BN input `.contiguous()` on MPS — numerics-preserving, memory-layout only) — auto-applied by `load_frozen_distortion_net(device="mps")`. fp16/bf16 autocast are both SLOWER on MPS and have worse gradients; **fp32 is the sweet spot**. The ~1.0 per-step cosine is a FIRST descent-direction sanity, NOT descent-equivalence (the n600 lesson: a ~1.0-cosine gradient can still compound into a POSE blow-up a d_seg-only check never sees).

## 2. Vehicle choice + rationale — split-device P2 torch_vehicle

**Chosen: the P2 torch_vehicle (`src/tac/torch_vehicle/`) on a SPLIT device** — pure torch, decoder + frozen scorer both run on the Apple GPU for the per-step forward/backward (the gradient), while the exact-eval that picks BEST + seeds telemetry runs on torch-CPU. This is the most upstream-faithful "upstream but macOS" (no MLX↔torch boundary; the literal upstream `DistortionNet` modules on the device upstream's own `modules.py` selects: `cuda→mps→cpu`).

**Why a SPLIT, not `device=mps` everywhere:** the authority discipline is non-negotiable — the exact d_seg/d_pose that pick the BEST checkpoint MUST be CPU-TRUSTED (CLAUDE.md "MPS auth eval is NOISE": 23× pose drift). A single `device=mps` would route the BEST tracker through MPS = forbidden. So the split is mandatory, not optional.

### What landed (the wire-in)

* `TorchVehicleConfig`: new `train_device` (gradient backend; MPS allowed). `device` stays the AUTHORITY/eval device and still RAISES on `mps` (the ban is preserved on the axis that matters). `train_device=None` → equals `device` (legacy single-device, byte-identical behavior; all 17 prior tests still pass).
* `TorchVehicleDriver`: training decoder/latents + per-step forward on `train_device` (MPS); the parse-back EVAL decoder is built on `device` and `exact_eval` always runs on the CPU authority net. randperm + random latents are drawn on CPU then moved (MPS has its own RNG stream — a CPU draw keeps the init reproducible vs the CPU-arm A/B and keeps resume bit-identical). EMA shadow moved to CPU for the int8 archive build.
* `RealScorerContext`: holds TWO frozen scorer instances — authority (CPU) for `exact_eval`, train (`train_device`) for `seg_pose_forward`. Identical frozen weights; only the device differs. New `max_pairs` + `targets_cache`: a cheap-n A/B / smoke caps the per-step target precompute via the CAPPED, cached `build_gt_targets` (CPU authority) instead of the uncapped vendored `precompute_targets` (which runs all 600 pairs × ~15 s ≈ 2.5 h on CPU). Targets are ALWAYS CPU-computed (the n600 lesson: the d_seg reference may never be MPS).
* `run.py`: `--train-device {cpu,cuda,mps}` (eval stays on `--device`, CPU authority).
* Tests: `src/tac/torch_vehicle/tests/test_split_device_mps.py` (9 NO-FAKE tests — authority-MPS-ban preserved, split flags wired, synthetic split-logic run completes on the authority eval, MPS-hardware train-decoder-on-MPS / resumable-on-MPS). 26/26 torch_vehicle tests green; `check_no_mps_fallback_default` clean on my files (explicit opt-in, not a silent fallback).

## 3. Descent-equivalence A/B (the gate) — BOTH terms, at a feasible n

Harness: `experiments/measure_torch_vehicle_mps_descent_equivalence.py`. Two `TorchVehicleDriver` arms from the SAME seed/init/permutation, single-stage Muon basin curriculum (the stage-8 recipe that drives the basin):

* **Arm A** `train_device=cpu` — the AUTHORITY gradient (baseline).
* **Arm B** `train_device=mps` — the FAST gradient (candidate).
* BOTH arms eval d_seg AND d_pose on the torch-CPU authority every eval epoch (the driver always evals on `device=cpu` regardless of train_device) — same metric, only the gradient differs.

Adjudicated through the canonical reusable `tac.mlx_pr95_port.speedup_acceptance_gate.evaluate_descent_equivalence` — the gate that REFUSES a d_seg-only pass and detects the pose-divergence signature (it encodes the n600 incident structurally).

**n2/2-epoch end-to-end smoke (gate plumbing):** PASS — arms near-identical (pose gap 3.0e-3 ≤ tol 8.8e-3, d_seg gap 0). Confirms the harness + gate work end-to-end with the real frozen scorer.

**n48/30-epoch BOTH-terms verdict: REJECT (on the POSE axis).**

| eval ep | cpu d_seg | mps d_seg | cpu d_pose | mps d_pose | pose gap |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.505381 | 0.505381 | 169.137 | 169.079 | 0.058 |
| 10 | 0.505381 | 0.505381 | 167.733 | 168.042 | 0.309 |
| 15 | 0.505381 | 0.505381 | 169.037 | 168.671 | 0.366 |
| 20 | 0.505381 | 0.505381 | 172.643 | 169.341 | 3.303 |
| 25 | 0.505381 | 0.505381 | 174.883 | 169.540 | 5.343 |
| 30 | 0.505381 | 0.505381 | 173.602 | 166.579 | **7.023** |

* **d_seg: PASS** — bit-identical at every eval (final |gap| = 0.000 ≤ tol 5.0e-3). The SegNet-path MPS gradient IS descent-equivalent.
* **d_pose: REJECT** — final |gap| = **7.023** > tol 1.116 (the gate's pose tolerance = 0.25 × |baseline pose descent 4.465|). The PoseNet-path MPS gradient is NOT descent-equivalent: the gap GROWS monotonically (0.06 → 7.02) as the two arms accumulate different pose updates. No catastrophic divergence-signature (the arms don't blow up toward random), but the tracking gap exceeds tolerance.
* **This is the EXACT n600 failure class the gate exists to catch:** a speedup that is **seg-correct but pose-wrong**. The ~1.0 per-step gradient cosine + bit-identical d_seg would have wrongly admitted it under a d_seg-only check; the BOTH-terms gate (per the n600 lesson, encoded in `speedup_acceptance_gate`) correctly REJECTED it on the pose axis. (Verdict JSON: `experiments/results/torch_vehicle_mps_descent_ab/verdict.json`.)
* Both arms reached near-identical best_score (CPU 91.550 vs MPS 91.408) — because the score is seg-dominated at this operating point, the seg-correct MPS arm scores similarly, which is exactly why a SCORE-only or seg-only check would have missed the pose divergence.

### Throughput (measured, n48, contended machine)

* CPU-grad arm: 1797.0 s / 30 epochs = **59.90 s/epoch**.
* MPS-grad arm: 384.9 s / 30 epochs = **12.83 s/epoch** → **4.67× per-EPOCH** speedup.
* The per-epoch number is heavily diluted: training steps are ~90× faster on MPS (the bench), but the 6 shared CPU authority evals (full GT-decode + scorer forward on 48 pairs) are identical on both arms and were the dominant cost under heavy CPU contention from 3 sibling daemons. The pure-training step speedup is the ~90-104× from the bench; the eval-inclusive per-epoch speedup is 4.67×.

## 4. The basin run — NOT LAUNCHED (gate REJECTED)

Thesis: frontier is RATE-DOMINATED. base_ch=20 byte-closes to ~100 KB (vs 177,169 B) → **S≈0.131 (SUB-0.15) IF the d_seg 5.6e-4 basin holds** at ~30% of PR95's params.

**The base_ch=20 MPS basin run is NOT launched.** Per the operator's binding "VALIDATE BEFORE TRUSTING" directive and the NO-FAKE "surrogate-optimized-but-not-exact-authority-verified" rule, a basin run on a gradient that FAILED the descent-equivalence gate would manufacture a fake "the architecture can/can't reach the basin" verdict that is really a broken pose gradient. The MPS pose gradient is NOT trustworthy for an n600 basin run.

### Recommended fallback (honest)

1. **Primary: the torch-CPU basin daemon (pid 42035)** — the trustworthy gradient. It was NOT disturbed and remains the correct vehicle for the base_ch=20 basin on this machine, slow but pose-correct. It is the path to the S≈0.131 thesis.
2. **Or: Modal CUDA** — the $40-50 spend the MPS path hoped to replace. CUDA is a contest-authority axis (no pose drift), so a CUDA-gradient basin is trustworthy AND fast.
3. **The MPS path is NOT dead — it is SEG-ONLY descent-equivalent.** The SegNet-path gradient is bit-identical. A reactivation path: **split the gradient backend by head** — run the SegNet-path forward/backward on MPS (the 90× lever, validated) and the PoseNet-path forward/backward on torch-CPU (the small, pose-correct part). Whether that hybrid is faster-net depends on the PoseNet-vs-SegNet CPU cost split; it is the next probe if the MPS lever is to be salvaged. Alternatively, diagnose the specific MPS PoseNet op whose backward drifts (the FastViT attention / hydra-head numerics) and patch it in `tac.torch_mps_compat` (sibling of the BN-contiguous patch), then re-run THIS gate. Until one of those lands and PASSES the gate, MPS is not admitted for a pose-bearing basin run.

The harness (`experiments/measure_torch_vehicle_mps_descent_equivalence.py`) + the split-device wire-in are committed and reusable, so the re-test after any MPS-PoseNet fix is a single command.

## 5. Honest authority caveat + bottom line

Everything here is `[macOS advisory]`, NON-PROMOTABLE. The frontier is UNMOVED (177,169 B `[contest-CPU]`); this unit did NOT lower the exact score, and the planned MPS basin run is NOT launched because the gradient FAILED the validation gate. That is the headline: **the 90-104× MPS speedup is real, but the MPS gradient is only HALF trustworthy — SegNet-path bit-identical, PoseNet-path divergent — so it cannot drive the pose-bearing base_ch=20 basin.** The torch-CPU basin daemon (pid 42035) was NOT disturbed and remains the trustworthy path; Modal CUDA is the alternative. The pointer moves ONLY when a byte-closed `best/best_archive.bin` from a TRUSTWORTHY-gradient basin is run through `upstream/evaluate.py` on contest-CPU AND contest-CUDA (1:1 hardware).

This is a HONEST NEGATIVE that PREVENTED a fake: had we skipped the gate (trusting the 104× bench + ~1.0 cosine), we'd have burned a multi-hour MPS basin run and reported a pose-corrupted "basin" result. The BOTH-terms gate (the n600 lesson encoded) earned its keep here.

## 6. 6-hook wire-in (Catalog #125)

1. **Sensitivity-map** — N/A (this is a throughput/training-backend wire-in, not a byte-allocation change).
2. **Pareto constraint** — N/A (no new archive section).
3. **Bit-allocator** — N/A.
4. **Cathedral autopilot dispatch** — the split-device run is a local FREE actuator; not a paid-dispatch candidate (advisory only).
5. **Continual-learning posterior** — the descent-equivalence verdict + s/epoch is the empirical anchor (this memo + the verdict JSON); reseeds the prior to "MPS gradient is SEG-correct but POSE-divergent — not admissible for a pose-bearing basin until the MPS-PoseNet drift is patched + re-gated."
6. **Probe-disambiguator** — the A/B harness IS the disambiguator; it RESOLVED to **REJECT** (MPS pose gradient diverges) → the verdict is "fall back to torch-CPU pid 42035 / Modal CUDA," NOT "launch the MPS basin." The disambiguator did its job (prevented the fake).
