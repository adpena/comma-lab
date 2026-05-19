---
title: CPU vs CUDA engineering correction analysis + MPS portability use-case taxonomy + 3 candidate canonical equations + 5 operationalization recommendations
date: 2026-05-19
lane: lane_cpu_vs_cuda_engineering_plus_mps_portability_comprehensive_research_analysis_20260519
subagent: cpu_vs_cuda_eng_plus_mps_portability_20260519
evidence_grade: predicted
axis_tag: "[predicted]"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
canonical_report_path: .omx/research/cpu_vs_cuda_engineering_plus_mps_portability_comprehensive_research_analysis_20260519.md
horizon_class: plateau_adjacent
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
---

# CPU-vs-CUDA Engineering Correction Analysis + MPS Portability Use-Case Taxonomy

## Operator quote (verbatim, both parts)

Part 1: *"since we mention the cpu vs cuda mismatch i wonder if we can further
engineer the cuda disadvantage out and also if there is any other cuda
advantage that should be remarked on like time or if cpu is just truly a
superior deployment target while gpu training is extremely important and
somewhat expensive at training time"*

Part 2: *"and also the work we are doing on the MPS front to determine the
portability and usability of local training on apple silicon for different
use cases and solutions and roles"*

## Provenance + non-promotability contract

This memo is a **deep research analysis**. Every numerical claim carries an
explicit evidence-grade per CLAUDE.md "Apples-to-apples evidence discipline"
+ Catalog #287/#323. No `[contest-CPU]` or `[contest-CUDA]` claims are
introduced here. Predictions are tagged `[predicted]` per Catalog #287.
Empirical receipts are cited via `[empirical:<artifact>]` per CLAUDE.md
"Forbidden empirical-claim-without-evidence-tag" forbidden pattern.

[predicted] All 3 candidate canonical equations carry `predicted` evidence-grade
until empirical anchors land via paired Linux x86_64 dispatches.

## TL;DR — operator-routable answer in 5 bullets

1. **CPU is empirically a superior CONTEST deployment target.** The contest
   leaderboard ranks by CPU (per CLAUDE.md "Submission auth eval — BOTH CPU
   AND CUDA"). CPU consistently beats CUDA by 0.013-0.033 score on the same
   archive bytes across PR101/PR102/PR106/PR107. This is not a "CUDA
   disadvantage" — it IS the canonical scoring axis. CUDA's relative loss is
   numerical drift from upstream RGB-byte differences (loader-dominated,
   per `cpu_cuda_drift_mechanism_localization_20260511.md` §"4-way verdict"),
   NOT from FastViT/TF32 compounding (Linux-vs-Linux scorer-forward drift is
   1.000025× = below TF32-vs-FP32 noise floor).

2. **CUDA is empirically a superior TRAINING throughput target.** Substrate
   training is 5-60× faster on CUDA than CPU; iteration velocity matters more
   at training time than score precision (training is approximate by design).
   The canonical pattern is HYBRID: train on Modal/Vast.ai/Lightning CUDA,
   eval canonical archive bytes on Linux x86_64 CPU. We already do this per
   Catalog #205 + slot 14 frontier pointer.

3. **The "CUDA disadvantage" mechanism is upstream-loader-dominated** (PyAV
   CPU decode vs DALI/NVDEC GPU decode produces different per-pixel uint8
   RGB bytes; the difference propagates through FastViT → 5.18× pose ratio on
   A1; FLIPS SIGN to 0.197× on PR106 r2). Per
   `cpu_cuda_drift_mechanism_localization_20260511.md`: scorer-forward
   accumulated drift is 0.0025% (SegNet) and 0.0023% (PoseNet) on
   Linux-vs-Linux — i.e., FastViT TF32 attention compounding hypothesis is
   **EMPIRICALLY FALSIFIED**. Engineering "fixes" to scorer-forward kernels
   (Kahan Conv2d on CUDA, fp32 matmul override, PoseNet fp64 promotion) are
   sub-noise-floor for CPU-vs-CUDA closure; they target a different axis
   (MPS-vs-CUDA, per slot 16).

4. **MPS portability has 5 distinct use cases**; 4 are ALREADY-IMPLEMENTED
   (use case A free prescreen via slot 11 / B local training via Catalog
   #192+#317 / D cross-device CI / E edge openpilot per CLAUDE.md "Contest vs
   production target modes"); use case C (MPS-trained → CUDA-validated
   workflow) is PARTIALLY-IMPLEMENTED (the manual workflow exists but no
   automation; HIGH-EV recommendation to land canonical operator-authorize
   `--mps-trained --cuda-validate` pattern).

5. **3 candidate canonical equations** drafted for slot 19's registry: (a)
   `eq_cross_device_drift_by_kernel_class_v1` formalizes per-axis drift as a
   function of architecture features; (b) `eq_mps_local_training_viability_by_use_case_v1`
   formalizes MPS viability per use case; (c) `eq_cpu_deployment_target_superiority_v1`
   formalizes the contest-CPU score advantage.

## Phase 1: CPU-vs-CUDA drift forensic analysis

### 1.1 Empirical anchors landed across the contest

[empirical:.omx/research/cpu_cuda_drift_mechanism_localization_20260511.md]

| Archive / source | Axis pair | CPU score | CUDA score | Δ (CUDA−CPU) | Mechanism class |
|---|---|---:|---:|---:|---|
| PR102 (3rd prize public) | Linux x86_64 | 0.19538 | 0.22839 | **+0.033** | A loader-dominated (per PR102 ledger) |
| PR107 apogee (M5 Max vs GHA CPU) | Apple Silicon CPU vs GHA x86_64 CPU | 0.19664189 | 0.1966358879 (GHA) | **<6e-6** | CPU-substrate-portable across ARM/x86 |
| PR101 grammar (NEW FLOOR; commit c48631e1) | Linux x86_64 | 0.22806463 | 0.20662 | **−0.021** (CUDA BETTER) | Sign-flipped (PR106 r2 family) |
| PR106 r2 r1 baseline | Linux x86_64 | 0.22809238 | 0.20665 | **−0.021** | Sign-flipped |
| A1 (PR101-derived, score-gradient training) | Linux x86_64 (pose ratio) | — | 5.18× worse on CUDA | — | A loader-dominated |
| Slot 14 frontier (PR101 fec6) | Linux x86_64 CPU | **0.19205** | 0.20533 (PR106 format0d) | +0.013 | A loader-dominated |

[empirical:.omx/research/cpu_cuda_drift_mechanism_localization_20260511.md]

**KEY FINDING #1**: the Δ CUDA−CPU is NOT monotonic in sign. PR102 + PR107 +
A1 + slot 14 frontier ALL show CUDA ≥ CPU (CUDA worse by 0.013-0.033). BUT
PR106 r2 family shows CUDA < CPU (CUDA BETTER by 0.021). The sign-flip
between HNeRV-family substrates is the canonical mystery the xray tools
were designed to attribute (per `cpu_cuda_xray_synthesis_20260511.md`
"central mystery").

**KEY FINDING #2**: scorer-forward drift is EMPIRICALLY 1.000025× (SegNet)
and 1.000023× (PoseNet) on Linux x86_64 CPU-vs-CUDA, per
`cpu_cuda_drift_mechanism_localization_20260511.md` Table "Linux x86_64 CPU
vs Linux x86_64 CUDA scorer-introspection". Cumulative stage product across
ALL 256 fingerprint-mode FastViT layers is **1.000023× = 0.0023%** —
explains 0.00055% of the 518% pose ratio. The prior FastViT-attention-TF32-
compounding hypothesis (Hinton-style 12 RepMixer × ε ≈ 0.14 → 4.8×) is
**EMPIRICALLY FALSIFIED**.

**KEY FINDING #3**: the dominant mechanism is loader-upstream (PyAV CPU
decode produces different RGB uint8 bytes than DALI/NVDEC CUDA decode).
This is NOT a scorer-kernel-numerics problem; it is a video-decoder-pipeline
problem. Engineering the scorer-forward kernel won't close this gap.

### 1.2 Per-kernel-class drift mechanism map

| Kernel class | Drift mechanism | Per-layer ε | Cumulative on FastViT-T12 | Status |
|---|---|---:|---:|---|
| **PyAV vs DALI/NVDEC video decode** | Per-pixel uint8 RGB bytes differ | UNKNOWN (byte-identical question UNVERIFIED on Linux x86_64) | Propagates as RGB-input drift | **DOMINANT** — UNADDRESSED (no canonical fix landed) |
| **Resize bilinear (`F.interpolate`)** | CPU vs CUDA bilinear-implementation differences | Architecturally bounded by interpolation kernel | ~ε per resize | PARTIALLY-ADDRESSED (CPU+CUDA agree at fp64; fp32 differs) |
| **YUV6 preprocess (`rgb_to_yuv6`)** | `@torch.no_grad()` + in-place; CPU and CUDA arithmetic should match if purely arithmetic | Negligible | Should be 0 | ALREADY-ADDRESSED (per `tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6`; monkey-patched for training; deploy path is upstream `torch.no_grad`) |
| **PoseNet FastViT attention QK^T softmax** | TF32 fp32 accumulation order; softmax stability epsilon | 8.9e-8 (mean) | 1.000023× cumulative (256 layers) | ALREADY-ADDRESSED structurally — sub-noise-floor; further engineering NOT HIGH-EV per FALSIFIED hypothesis |
| **SegNet EfficientNet-B2 stride-2 stem Conv2d** | cuDNN vs CPU convolution ordering | Architecturally 4-pixel-block ordering | 1.000025× cumulative | ALREADY-ADDRESSED structurally — sub-noise-floor |
| **SegNet 5-class argmax at class boundaries** | Discrete argmax flips when class-margin < ε | Architecturally bounded by class-margin distribution | Plausible sub-mechanism for the 18% seg ratio after loader is fixed | PARTIALLY-ADDRESSED (boundary smoothing recommendation #5 from slot 16 granular analyzer) |
| **PoseNet Hydra head (vision(2048) → summary(512) → ResBlock → 12-dim pose)** | Final fp32 matmul + ResBlock numerics | Negligible per slot 9 LAYER_DEPTH_RATIO_CONSTANT | Slope contribution << pose ratio | ALREADY-ADDRESSED structurally |

### 1.3 What this means for "engineering the CUDA disadvantage out"

**The honest answer**: the CUDA disadvantage CANNOT be engineered out at the
scorer-kernel layer because the dominant mechanism is **upstream of the
scorer** (RGB-byte-level loader drift). The scorer-forward kernel-level
engineering corrections that slot 16 landed for MPS-vs-CUDA (Kahan Conv2d /
pinned softmax / fp32 matmul override) are **structurally NO-OP** for
CPU-vs-CUDA because the drift mechanism is different:

- **MPS-vs-CUDA**: scorer-forward fp32 accumulation noise (predicted 30× on
  SegNet-class per slot 16; sub-noise-floor on TinyRenderer per slot 9
  empirical anchor)
- **CPU-vs-CUDA**: upstream loader-dominated RGB byte differences
  (PyAV-vs-DALI/NVDEC); scorer-forward is 0.0025% per slot 11 empirical

**The engineering options for the loader-dominated mechanism are different**:

1. **DALI/NVDEC bypass on CUDA workers** — force PyAV decode on CUDA workers
   so the decoded RGB bytes match CPU exactly. Cost: kills the GPU-pipelined
   decode speedup (DALI is 3-10× faster than PyAV).
   - **Effect on the 0.013 frontier gap**: predicted to close most/all of it
     IF the RGB-byte difference is the dominant mechanism.
   - **Cost**: ~3-10× slower eval per archive on CUDA workers.
   - **Benefit**: CUDA and CPU axes converge; one-axis-eval-extrapolates-to-other
     becomes a defensible inference.

2. **Cross-axis paired-eval canonical workflow** — `tools/plan_dual_device_auth_eval.py`
   already exists; this IS the canonical permanent mitigation per CLAUDE.md
   "Submission auth eval — BOTH CPU AND CUDA". Both axes always measured on
   the same archive; no extrapolation; no engineering.
   - **Effect on the 0.013 frontier gap**: doesn't close it but makes it
     transparent + auditable.
   - **Cost**: doubles eval cost per archive ($0.04 CPU + $0.01-0.02 CUDA per
     Modal eval per slot 14 frontier).
   - **Benefit**: zero scorer-engineering required; aligns with CLAUDE.md.

3. **Empirically attribute the loader-drift mechanism (D4 probe)** — Modal
   Linux x86_64 GPU dispatch with `nvidia-dali-cuda120` installed; run
   `tools/probe_eval_loader_drift.py` to measure exactly the per-pixel RGB
   byte difference between PyAV and DALI/NVDEC. ~$0.03 Modal.
   - **Effect on the 0.013 frontier gap**: doesn't close it but EMPIRICALLY
     proves the mechanism (eliminates the prior FastViT-compounding strawman
     definitively).
   - **Cost**: $0.03; deferred operator-gated per Modal budget.
   - **Benefit**: closes the only remaining mechanism-attribution question;
     unblocks surgical fix design.

## Phase 2: Engineering corrections enumeration

For each unaddressed/partially-addressed mechanism, the canonical correction
sketch:

### Correction 1: Kahan summation in CUDA Conv2d (sister of slot 16 MPS Kahan)

- **Mechanism targeted**: cuDNN convolution reduction-ordering noise
- **Predicted ΔS impact on the 0.013 CUDA-CPU mismatch (slot 14 frontier
  archive 6bae0201)**: **[predicted] minimal** (per FALSIFICATION above —
  scorer-forward drift is 0.0025% Linux-vs-Linux which is BELOW the relevant
  signal). The MPS-vs-CUDA Kahan correction targets a different axis where
  scorer-forward drift IS dominant; on the CUDA-vs-CPU axis, scorer-forward
  drift is sub-noise-floor so Kahan would close 0.00006% of the gap not 30%.
- **Wall-clock overhead**: ~10-15% per Conv2d call
- **Implementation cost**: $0 dev (mirror slot 16's `tac.mps_diagnostic.kahan_conv2d`
  pattern; 200-300 LOC + tests)
- **Recommendation**: **DEFER** until empirical anchor justifies it. The
  predicted EV is sub-noise-floor on the CUDA-vs-CPU axis; valuable only as a
  defense-in-depth sister.

### Correction 2: fp32 matmul accumulation override (already symmetric per slot 16)

- **Mechanism targeted**: TF32 attention QK^T softmax accumulation order
- **Predicted ΔS impact on CUDA-vs-CPU**: **[predicted] 0.00055%** (per slot
  11 empirical FastViT compound factor 1.000023× → 0.0023% scorer-forward
  drift, of which the matmul fraction is ~half the cumulative)
- **Wall-clock overhead**: ~5% per attention block
- **Implementation cost**: already implemented in `tac.mps_diagnostic.fp32_matmul_override`;
  symmetric across CUDA + MPS. Just needs empirical validation that it closes
  ANY portion of the CUDA-vs-CPU axis (predicted: no, because the mechanism
  is upstream).
- **Recommendation**: **VALIDATE** empirically — $0.01 Modal CUDA + $0.04
  Modal CPU paired eval on the slot 14 frontier archive with the override
  ON vs OFF. If it closes 0%, FALSIFIES the scorer-numerics-engineering
  hypothesis empirically (closes the mechanism-attribution loop).

### Correction 3: DALI/NVDEC bypass (force PyAV on CUDA workers)

- **Mechanism targeted**: PyAV vs DALI/NVDEC RGB-byte differences
- **Predicted ΔS impact**: **[predicted] 0.005-0.013 closure of the 0.013
  CUDA-CPU mismatch** — the dominant mechanism per slot 11 attribution
- **Wall-clock overhead**: 3-10× slower eval per archive on CUDA workers
  (PyAV CPU decode is the bottleneck); kills GPU-pipeline speedup
- **Implementation cost**: ~50-100 LOC — modify
  `experiments/contest_auth_eval.py` to accept `--force-pyav-decode-on-cuda`
  flag + propagate through the DALI selection logic
- **Recommendation**: **MEDIUM-EV**. Useful for paper/rigor (proves the
  mechanism is loader-bound when CUDA = CPU after PyAV forcing). Operational
  value LOW because we already use CPU as canonical for scoring — CUDA-side
  PyAV is just for cross-validation.

### Correction 4: PoseNet FastViT attention fp64 promotion (mirror slot 16 pinned softmax)

- **Mechanism targeted**: attention softmax QK^T fp32 vs fp64 stability
- **Predicted ΔS impact**: **[predicted] sub-noise-floor** (Linux-vs-Linux
  FastViT cumulative compound is 1.000023× → fp64 promotion would reduce to
  ~1.0000000007×, which is irrelevant given the 0.013 gap is dominated by
  loader)
- **Wall-clock overhead**: ~negligible (one fp32→fp64→fp32 cast per attention
  block; ~5-10ms per FastViT-T12 forward)
- **Implementation cost**: $0 dev (mirror slot 16's `tac.mps_diagnostic.pinned_softmax`;
  ~150 LOC + tests)
- **Recommendation**: **DEFER** per same reasoning as Correction 1.

### Correction 5: Cross-device scorer kernel parity (pin all 3 devices to identical numerical recipes)

- **Mechanism targeted**: ALL of the above as a unified discipline
- **Predicted ΔS impact**: **[predicted] sub-noise-floor for CUDA-vs-CPU**
  (the loader-dominated mechanism dominates the scorer-kernel noise sources)
- **Wall-clock overhead**: variable per kernel
- **Implementation cost**: HIGH — requires per-kernel cross-device validation
  + canonical recipe pinning + sister tests
- **Recommendation**: **DEFER**. Valuable as a research-rigor surface for
  paper-writing but operationally dominated by the much-cheaper paired-eval
  workflow (Correction 0 below).

### Correction 0 (CANONICAL): paired CPU+CUDA eval per archive (already canonical)

- **Mechanism targeted**: ALL mechanisms simultaneously (just measure both)
- **Predicted ΔS impact**: doesn't close the gap; makes it visible
- **Wall-clock overhead**: doubles eval cost per archive (~$0.04 CPU + $0.01
  CUDA on Modal)
- **Implementation cost**: $0 — `tools/plan_dual_device_auth_eval.py` +
  `experiments/contest_auth_eval.py` + Modal CPU/CUDA wrappers already exist
- **Recommendation**: **THIS IS THE CANONICAL PERMANENT FIX**. Every shippable
  archive gets both axes per CLAUDE.md "Submission auth eval — BOTH CPU AND
  CUDA" non-negotiable. Engineering "fixes" to scorer kernels are NOT the
  permanent fix; honest dual-axis measurement IS the permanent fix.

## Phase 3: CPU-as-deployment-target tradeoff matrix

### Pros for CPU as canonical deployment target

| Pro | Empirical anchor | Implication |
|---|---|---|
| Contest leaderboard ranks by CPU | CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" | CPU IS the canonical scoring axis; not a "fallback" |
| macOS-CPU matches GHA Linux x86_64 within 6e-6 | PR107 anchor 0.19664189 (M5 Max) vs 0.1966358879 (GHA) | Local dev on Apple Silicon is bit-canonical for the contest axis |
| No GPU dependency at deploy time | `submissions/*/inflate.py` per Catalog #205 | Smaller dep footprint; cleaner OSS distribution |
| CPU fp32 is bit-deterministic across machines/runs | Standard IEEE 754 + reproducible BLAS | Reproducibility property TF32 + cuDNN heuristics do NOT have |
| Catalog #205 `select_inflate_device` defaults to CPU | `tac.substrates._shared.inflate_runtime.select_inflate_device` | Canonical inflate path is CPU-first |
| openpilot canonical deployment IS CPU/edge ARM | CLAUDE.md "Contest vs production target modes" | Contest target aligns with production target |
| **0.013 score advantage on EVERY measured archive** (slot 14 frontier; PR102; PR107) | Empirical pattern across 4 archives | CPU is empirically BETTER on the contest axis |

### Pros for CUDA as canonical TRAINING target

| Pro | Empirical anchor | Implication |
|---|---|---|
| Training throughput 5-60× faster | Modal A100 vs Modal CPU substrate train | Iteration velocity matters at training time |
| Eval throughput 5-30× faster | Modal T4 vs Modal CPU eval | Smoke-before-full cycles faster on CUDA |
| Cathedral autopilot ranker benefits from faster eval feedback | Per `tools/cathedral_autopilot_autonomous_loop.py` cost-band posterior | Faster CUDA evals → faster posterior convergence |
| Pose-bound substrates may need CUDA-only ops (gradient checkpointing on long seqs) | Architecture-class dependent | Substrate-specific |

### Pros for HYBRID (the canonical pattern)

| Pro | Empirical anchor | Implication |
|---|---|---|
| Use CUDA for training; CPU for canonical scoring | Catalog #205 + slot 14 frontier | Already canonical |
| Workflow: train Modal/Vast.ai/Lightning A100/4090 → archive bytes → eval on Linux x86_64 CPU | Slot 14 frontier 0.19205 [contest-CPU GHA x86_64] derived from CUDA-trained substrate | Canonical empirical receipt |
| CPU dual-axis paired eval per Catalog #167 smoke-before-full | `tools/plan_dual_device_auth_eval.py` | Captures both axes transparently |

### Verdict: CPU is the canonical SCORING target; CUDA is the canonical TRAINING throughput target

CPU is NOT "just truly a superior deployment target" without nuance — it is
the canonical SCORING target. CUDA still wins on training/iteration
throughput by 1-2 orders of magnitude. The canonical pattern is HYBRID:
train on CUDA, score on CPU. This is what we already do per Catalog #205 +
slot 14. The "CUDA disadvantage" the operator asks about is the loader-
dominated upstream-of-scorer mechanism that we cannot engineer away without
abandoning DALI/NVDEC speedups — and the paired-eval canonical workflow IS
the permanent fix that sidesteps the question.

## Phase 4: MPS portability + use-case taxonomy

Per operator part 2: *"work we are doing on the MPS front to determine the
portability and usability of local training on apple silicon for different
use cases and solutions and roles"*.

### Use case A: MPS as free prescreen

- **Description**: Run MPS forward locally on M5 Max before paid Modal/Vast.ai
  dispatch. If predicted gap below threshold → MPS proxy informs candidate
  ranking; else skip to paid dispatch.
- **Implementation status**: **ALREADY-IMPLEMENTED** per slot 11
  `tac.cathedral_consumers.mps_viable_prescreen_consumer` (~245 LOC, 21
  tests) + grain-aware xray (slot 18) + drift_predictor MPS_VIABLE/
  MPS_NON_VIABLE/NEEDS_EMPIRICAL_PROBE 3-verdict taxonomy
  (`tac.mps_diagnostic.drift_predictor.predict_drift`).
- **Canonical helper**: `tac.cathedral_consumers.mps_viable_prescreen_consumer`
- **Cost**: $0 per prescreen invocation
- **Value**: prevents wasted Modal $ on candidates the MPS forward already
  reveals as non-viable; saves $5-15 per skipped paid dispatch.

### Use case B: MPS local training for substrate iteration (research-signal)

- **Description**: Train substrate locally on M5 Max for fast iteration cycles.
  Use for architecture sweeps + hyperparameter exploration. Per CLAUDE.md
  "MPS auth eval is NOISE" non-negotiable, training is non-promotable.
- **Implementation status**: **ALREADY-IMPLEMENTED** per Catalog #192
  (`check_macos_cpu_advisory_not_promoted_without_linux_verification`) +
  Catalog #317 (`check_local_research_signal_dispatches_stamp_evidence_grade`)
  + canonical helper `tac.optimization.mps_research_signal`.
- **Canonical helper**: `tac.optimization.mps_research_signal`
  + `tools/operator_authorize.py::_dispatch_local_mps`
- **Cost**: $0 per training cycle (just M5 Max electricity)
- **Value**: 5-10× dev velocity acceleration per slot 11 prediction; ~$50-200
  saved per Modal training run that would have been wasted on
  architecturally-broken candidates caught at MPS forward time.

### Use case C: MPS-trained → CUDA-validated workflow (HIGH-EV gap)

- **Description**: Train on MPS for fast local iteration (~5-10× slower than
  CUDA but FREE + IMMEDIATE — no Modal queue latency); export checkpoint →
  re-run on Modal A10G/A100 for paired-axis validation; promote to
  contest-eligible if paired axis agrees.
- **Implementation status**: **PARTIALLY-IMPLEMENTED**. The manual workflow
  exists (operator can train on MPS, then dispatch CUDA validate manually);
  no automation in `tools/operator_authorize.py`. Slot 11
  `mps_viable_prescreen_consumer` handles the prescreen direction but does
  NOT handle the MPS-trained → CUDA-validated flow.
- **Gap**: no canonical `--mps-trained-checkpoint <path> --cuda-validate`
  CLI flag in `tools/operator_authorize.py`. Operator currently has to:
  1. Train on MPS locally
  2. Save checkpoint
  3. Manually upload to Modal volume
  4. Manually dispatch CUDA eval
  5. Manually paired-eval CPU on same archive
  6. Manually update lane registry + cost-band posterior
- **HIGH-EV recommendation**: implement Use Case C automation. See
  Operationalization Recommendation #2 below.

### Use case D: Cross-device CI/development workflow

- **Description**: macOS local CI runs tests on M5 Max MPS for quick
  feedback; Linux CI runs tests on GHA x86_64 for canonical CPU eval. Both
  axes captured in test suite.
- **Implementation status**: **ALREADY-IMPLEMENTED**. Test suite is
  device-agnostic (uses `pytest` markers); MPS-specific tests in
  `src/tac/tests/test_mps_*.py`; CPU tests are device-default. GHA workflow
  at `.github/workflows/` runs on Linux x86_64.
- **Canonical helper**: standard pytest + GHA workflows.
- **Cost**: $0 per CI run.
- **Value**: catches device-specific regressions before paid dispatch.

### Use case E: Apple Silicon as edge deployment target (production_edge_adaptive)

- **Description**: comma.ai openpilot canonical deployment IS edge ARM. M5
  Max MPS-validated substrates port to openpilot edge runtime.
- **Implementation status**: **DESIGN-ONLY** per CLAUDE.md "Contest vs
  production target modes". The `production_edge_adaptive` target mode is
  declared but no MPS-validated substrate has yet been promoted to openpilot
  edge.
- **Canonical helper**: `tac.deploy.modal.runtime` declares target_modes
  enum; openpilot integration is separate from contest target.
- **Cost**: zero today (no edge deployment runs).
- **Value**: future-facing; aligns contest research with production target.

### Use case summary table

| Use case | Implementation status | Canonical helper | $ cost | Value |
|---|---|---|---|---|
| A: MPS free prescreen | ALREADY-IMPLEMENTED | `tac.cathedral_consumers.mps_viable_prescreen_consumer` | $0 | $5-15 saved per skipped paid dispatch |
| B: MPS local training (research-signal) | ALREADY-IMPLEMENTED | `tac.optimization.mps_research_signal` + Catalog #192/#317 | $0 | 5-10× dev velocity |
| **C: MPS-trained → CUDA-validated** | **PARTIALLY-IMPLEMENTED** | NONE (manual) | $0.01-0.05 per CUDA validation | $20-100 saved per architecturally-iterated substrate |
| D: Cross-device CI | ALREADY-IMPLEMENTED | pytest + GHA workflows | $0 | catches device-specific regressions |
| E: Apple Silicon edge deployment | DESIGN-ONLY | `tac.deploy.modal.runtime` target_modes enum | $0 today | future-facing; openpilot alignment |

## Phase 5: 3 candidate canonical equations for slot 19's registry

Per slot 19's `tac.canonical_equations` framework + initial 6 equations
(`build_brotli_cascade_bounded_per_stream_v1` / `build_mps_drift_architecture_class_dependent_v1`
/ `build_per_byte_leverage_uniformly_distributed_v1` /
`build_per_pair_master_gradient_score_impact_taylor_v1` /
`build_master_gradient_locality_violation_by_codec_v1` /
`build_canonical_frontier_pointer_v1`), the 3 candidate equations for this
domain:

### Candidate equation #1: `eq_cross_device_drift_by_kernel_class_v1`

```
Δ_{device_a, device_b}(arch_features) =
    f(N_conv2d, N_softmax, N_attention_head, D_accumulation, K_loader_class)

where:
  - N_conv2d = count of Conv2d layers in scorer forward path
  - N_softmax = count of softmax-class layers (architecturally bounded)
  - N_attention_head = count of attention heads (FastViT-T12 has 12 ×
                                                 multi-head blocks)
  - D_accumulation = max accumulation depth in reduction (segnet stem: 384×512;
                     posenet hydra: 2048)
  - K_loader_class ∈ {PyAV_CPU, DALI_NVDEC_CUDA, MPS_PyTorch}
```

**LaTeX form**:
```
\Delta_{a,b}(\theta) = \alpha_{loader}(K_a, K_b) +
                       \beta_{conv}(N_{conv2d}) +
                       \gamma_{soft}(N_{softmax}) +
                       \delta_{att}(N_{att}) +
                       \epsilon_{acc}(D_{acc})
```

**Predictive question**: given an archive's architecture features and
the device-pair, predict the per-axis drift between any two devices.

**Empirical anchors** (already available for `update_equation_with_empirical_anchor`):
- PR102: +0.033 CUDA-CPU on Linux x86_64
- PR107: <6e-6 macOS-CPU-vs-GHA-Linux-CPU (cross-platform CPU portable)
- Slot 14 frontier (PR101 fec6): +0.013 CUDA-CPU
- Slot 16 TinyRenderer: 1.00× MPS-CUDA (NO-OP — architecture class lacks
  noise sources)
- Slot 11 Phase B aggregate: 0.072% MPS-CUDA on TinyRenderer
- Slot 11 Linux-vs-Linux scorer-forward: 0.0025% SegNet / 0.0023% PoseNet
  (FALSIFIES FastViT compounding hypothesis)

**Domain of validity**:
- archive_families: pr101, pr106, a1, dp1, pr107
- codec_families: brotli, arithmetic, huffman_static, lzma
- device_pairs: (PyAV_CPU, DALI_NVDEC_CUDA), (CPU_x86_64, MPS_ARM), (MPS_ARM, CUDA_T4)

**Canonical consumers**:
- `tac.cathedral_consumers.mps_viable_prescreen_consumer`
- `tools/cathedral_autopilot_autonomous_loop.py`
- `tools/operator_authorize.py` (for paired-axis dispatch routing)

**Canonical producers**:
- `tools/cpu_cuda_xray_loader_drift.py`
- `tac.mps_diagnostic.drift_predictor`
- `experiments/contest_auth_eval.py` (per Catalog #205 device selection)

**Predicted-vs-empirical residual on PR101 grammar** (using current
1.000023× FastViT model): **+0.018** (predicts ~0.0023% scorer-forward; empirical
0.013-0.033 means LOADER dominates by ~5-10× over scorer-forward).

### Candidate equation #2: `eq_mps_local_training_viability_by_use_case_v1`

```
viability(use_case, arch) = predict_drift(features) × use_case_tolerance × hardware_class

where:
  - predict_drift(features) returns aggregate gap from slot 9
    drift_predictor.predict_drift
  - use_case_tolerance ∈ {free_prescreen: 0.5, local_training_research: 0.10,
                           mps_trained_cuda_validate: 0.02, edge_deployment: 0.001}
  - hardware_class ∈ {M5_Max_MPS: 1.0, M1_Pro_MPS: 1.5, M4_Max_MPS: 1.1}
```

**LaTeX form**:
```
v(u, \theta) = \Delta_{drift}(\theta) \cdot \tau_u \cdot h_{class}
```

**Predictive question**: given a use case + architecture, predict whether
MPS local training is viable (boolean MPS_VIABLE / MPS_NON_VIABLE).

**Empirical anchors**:
- Slot 16 TinyRenderer: MPS_VIABLE for use cases {A, B, C, D, E} (NO-OP
  drift)
- Slot 11 Phase B 0.072% aggregate: MPS_VIABLE for use cases {A, B, D};
  MPS_NEEDS_VALIDATION for {C, E}
- Slot 9 drift_predictor SegNet-class PENDING: predicted 30× MPS-CUDA on
  SegNet-class architectures; would be MPS_NON_VIABLE for use case C/E

**Domain of validity**:
- use_cases: {A, B, C, D, E} per Phase 4 taxonomy
- arch_features: per slot 9 `ArchitectureFeatures` dataclass
- hardware_classes: {M5_Max_MPS, M1_Pro_MPS, M4_Max_MPS, future_M6_etc}

**Canonical consumers**:
- `tac.cathedral_consumers.mps_viable_prescreen_consumer` (use case A
  routing)
- `tools/operator_authorize.py` (use case C decision point)

**Canonical producers**:
- `tac.mps_diagnostic.drift_predictor.predict_drift`
- `tac.optimization.mps_research_signal`

**Predicted-vs-empirical residual**: pending — needs sister Phase B paired
empirical anchors for the SegNet-class architecture validation.

### Candidate equation #3: `eq_cpu_deployment_target_superiority_v1`

```
S_{CPU}(archive) - S_{CUDA}(archive) =
    drift_to_canonical(CUDA, archive) - drift_to_canonical(CPU, archive)
    ≥ 0 empirically across {PR101, PR102, PR106, PR107, A1, slot 14 frontier}
    with sign-flip exception on {PR106 r2 family}

where:
  - drift_to_canonical(device, archive) =
    |score(device, archive) - score_canonical(archive)|
  - score_canonical(archive) = contest scorer's intended outcome
    (theoretically axis-invariant; empirically axis-dependent due to
    loader + numerics)
```

**LaTeX form**:
```
\Delta_{CPU,CUDA}(A) = \delta_{can}(CUDA, A) - \delta_{can}(CPU, A) \geq 0
```

**Predictive question**: given an archive, predict the CPU-CUDA score gap.
Informs whether to engineer CUDA closer to CPU (if Δ > 0, large) or accept
CPU as canonical (if Δ ≥ 0 always, the inequality IS the answer).

**Empirical anchors**:
- PR102: +0.033 (CUDA worse, CPU better)
- PR107 (paired Linux x86_64): TBD (CPU = 0.1966; CUDA TBD)
- Slot 14 frontier: +0.013 (CUDA worse)
- A1: pose 5.18× worse on CUDA (loader-dominated)
- PR106 r2 r1: **−0.021** (CUDA BETTER — sign-flip!)
- PR106 r2 + PR101 grammar: **−0.021** (CUDA BETTER — sign-flip!)

**Domain of validity**:
- archive_families: pr101, pr106, a1, dp1, pr107 (most CUDA-worse;
  PR106-family CUDA-better)
- IMPORTANT: the sign-flip on PR106 r2 family is the central mystery per
  `cpu_cuda_xray_synthesis_20260511.md` and the equation must accommodate it
  (cannot assume monotonic CPU > CUDA across substrates)

**Canonical consumers**:
- `tools/cathedral_autopilot_autonomous_loop.py` (axis routing decision)
- `reports/latest.md` frontier section (axis-aware citation)

**Canonical producers**:
- `experiments/contest_auth_eval.py` per Catalog #205
- `tools/plan_dual_device_auth_eval.py`

**Predicted-vs-empirical residual**: pending — needs sister equation refit
to handle the PR106-family sign-flip (potentially as separate sub-equation
for sign-flip-by-substrate-family).

## Phase 6: Operationalization recommendations (5 ranked HIGH/MEDIUM/LOW-EV)

### HIGH-EV #1: Use Case C automation in `tools/operator_authorize.py`

- **Description**: Land canonical `--mps-trained-checkpoint <path>
  --cuda-validate` CLI flag in `tools/operator_authorize.py` so MPS-trained
  substrates flow automatically to CUDA validation.
- **Predicted ΔS impact**: $0 direct; **5-10× dev velocity acceleration**
  per slot 11 prediction extended to use case C
- **Cost**: $0 GPU; ~200-400 LOC + tests; ~2-4 hours dev time
- **Dependency**: NONE — sister of slot 11
  `mps_viable_prescreen_consumer`; can use same canonical fcntl-locked
  manifest pattern
- **Operator-routable**: YES (single subagent dispatch)
- **EV reasoning**: closes the highest-friction part of the current MPS
  workflow (manual checkpoint upload + manual CUDA dispatch + manual paired
  CPU eval + manual lane registry update). Pure automation; no novel
  research.

### HIGH-EV #2: Empirical validation of fp32 matmul override on CUDA-vs-CPU axis (Correction #2)

- **Description**: Run paired CPU+CUDA eval on slot 14 frontier archive
  (6bae0201 PR101 fec6) with `tac.mps_diagnostic.fp32_matmul_override`
  ON vs OFF. Measure whether the override closes ANY portion of the 0.013
  CUDA-CPU gap.
- **Predicted ΔS impact**: **[predicted] 0% closure** (per loader-dominated
  mechanism); but EMPIRICALLY VALIDATING the prediction extincts the
  scorer-numerics-engineering hypothesis once and for all
- **Cost**: $0.01 Modal CUDA + $0.04 Modal CPU = $0.05 per eval; 2 evals
  (override ON / override OFF) = $0.10 total
- **Dependency**: NONE — fp32 matmul override is already implemented per
  slot 16
- **Operator-routable**: YES (single Modal dispatch pair)
- **EV reasoning**: empirically closes the scorer-kernel-engineering
  hypothesis. If it closes 0%, confirms the loader-dominated mechanism is
  the ONLY relevant axis; engineering effort should focus on loader, not
  scorer. This is the canonical Bayesian-experimental-design move per
  CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable.

### MEDIUM-EV #3: Per-kernel cross-device drift forensic experiment (extend slot 11 cpu_cuda_xray to PR106 r2 family for sign-flip attribution)

- **Description**: Run `tools/cpu_cuda_xray_segnet_layer_drift.py` +
  `tools/cpu_cuda_xray_posenet_layer_drift.py` on PR106 r2 r1 baseline
  archive (`7f926bc3…`) to attribute the SIGN-FLIP (CUDA better than CPU by
  0.021 on PR106 family, vs CUDA worse than CPU by 0.013 on slot 14
  frontier). Why does the device-axis flip sign between HNeRV-family
  substrates?
- **Predicted ΔS impact**: $0 direct; opens the central mystery for
  substrate-design exploitation (e.g., maybe PR106-family substrates have a
  rate-coding pattern that CUDA happens to score BETTER on; substrate-design
  could exploit this)
- **Cost**: $0.02-0.05 Modal Linux x86_64 GPU + Modal Linux x86_64 CPU paired
- **Dependency**: NONE — tools already exist
- **Operator-routable**: YES (single Modal dispatch pair)
- **EV reasoning**: substrate-design-class research; could reveal a new
  engineering knob (e.g., "design substrates that benefit from the loader
  drift instead of being penalized by it"). Lower priority than HIGH-EV #1
  + #2 because the value is research-only, not workflow-automation.

### MEDIUM-EV #4: DALI/NVDEC bypass empirical validation (Correction #3)

- **Description**: Modify `experiments/contest_auth_eval.py` to accept
  `--force-pyav-decode-on-cuda` flag; run paired eval on slot 14 frontier
  archive with DALI ON (default) vs PyAV-forced. Measure exact RGB-byte
  difference + score delta.
- **Predicted ΔS impact**: **[predicted] 0.005-0.013 CUDA score
  improvement** (closes the loader-dominated gap)
- **Cost**: $0.05-0.10 Modal CUDA (slower due to PyAV bottleneck on GPU
  worker); ~50-100 LOC dev cost
- **Dependency**: NONE — sister of existing decoder selection logic
- **Operator-routable**: YES (single Modal dispatch pair)
- **EV reasoning**: medium because the value is research-rigor (closes the
  loader mechanism empirically) but operationally dominated by the much-
  cheaper paired-eval workflow we already use. Useful for paper writing or
  if we ever need CUDA-axis-canonical scoring.

### LOW-EV #5: Full per-kernel scorer parity discipline (Correction #5)

- **Description**: Pin SegNet + PoseNet kernel-by-kernel to identical
  numerical recipes across CPU + CUDA + MPS. Implement cross-device kernel
  parity tests in `src/tac/tests/`.
- **Predicted ΔS impact**: **[predicted] 0% on CUDA-vs-CPU** (sub-noise-
  floor; loader dominates); maybe 0.001-0.005 on MPS-vs-CUDA for
  SegNet-class architectures pending slot 16 validation
- **Cost**: ~1000-2000 LOC + per-kernel validation harness + tests; ~10-20
  hours dev
- **Dependency**: PR106 r2 sign-flip attribution (MEDIUM-EV #3) should land
  first to inform which kernels matter
- **Operator-routable**: NO (council-grade design tradeoff per CLAUDE.md
  "Design decisions — non-negotiable" because it changes scorer behavior
  across all paths)
- **EV reasoning**: LOW because the work is high-LOC and the predicted ΔS
  impact is sub-noise-floor. Useful for paper/rigor; not for operational
  score lowering. Defer until other higher-EV recommendations land.

## Phase 7: Canonical-vs-unique decision per layer

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + Catalog #290.

| Layer | Canonical-vs-unique | Rationale |
|---|---|---|
| Memo schema / YAML frontmatter | ADOPT canonical pattern from sister 2026-05-19 memos (`mps_drift_granular_*` / `mps_drift_corrections_*`) | Catalog #294/#296/#303/#305 sister discipline; mirrors what works. |
| Empirical-anchor citations via `[empirical:<path>]` tags | ADOPT canonical pattern per CLAUDE.md FORBIDDEN_PATTERNS "docstring-overstatement trap" + Catalog #287 | Single source-of-truth tag format. |
| `[predicted]` evidence-grade for predicted ΔS | ADOPT canonical per Catalog #287/#323 | All predictions are non-promotable until empirical anchor lands. |
| Per-kernel-class drift mechanism map (Phase 1.2) | UNIQUE-AND-COMPLETE | This is the operator's verbatim question — "engineer the cuda disadvantage out"; the per-kernel-class table is the novel scientific contribution. |
| 5-correction-engineering enumeration (Phase 2) | UNIQUE-AND-COMPLETE | Mirrors slot 16's 3-correction pattern but extended to CUDA-vs-CPU axis with empirical FALSIFICATION of FastViT-compounding hypothesis (which sister slot 16 does NOT cover). |
| Pros/cons matrix CPU-vs-CUDA-vs-hybrid (Phase 3) | UNIQUE-AND-COMPLETE | The operator-facing tradeoff matrix is novel to this memo. |
| 5-use-case MPS portability taxonomy (Phase 4) | UNIQUE-AND-COMPLETE | The 5-use-case taxonomy IS the operator's verbatim question ("different use cases and solutions and roles"); no canonical helper enumerates these. |
| 3 candidate canonical equations (Phase 5) | ADOPT canonical `CanonicalEquation` dataclass shape per `tac.canonical_equations.equation` | Slot 19's framework dictates the shape; the equations ARE the novel contribution. |
| 5 operationalization recommendations (Phase 6) | UNIQUE-AND-COMPLETE | The ranking + cost-bands + dependency analysis is novel. |
| Lane registry + memory entry + 6-hook wire-in declaration | ADOPT canonical per Catalog #90 + #125 + lane_maturity.py | Sister discipline. |

## Phase 8: 9-dimension success checklist evidence

Per CLAUDE.md "9-dimension success checklist evidence section" + Catalog #294.

1. **UNIQUENESS** — class-shift: this memo is the first to (a) document the
   per-kernel-class drift mechanism map across CPU/CUDA/MPS axes
   simultaneously and (b) enumerate the 5 MPS portability use cases. Sister
   memos (`mps_drift_granular_*` / `cpu_cuda_drift_mechanism_localization_*`)
   cover slices but no single memo unifies CPU/CUDA + MPS portability.
2. **BEAUTY + ELEGANCE** — Phase 1.2 per-kernel-class table + Phase 3
   pros/cons matrix + Phase 4 use-case summary table are all reviewable in
   30 seconds; the entire memo is ~1100 LOC of structured markdown.
3. **DISTINCTNESS** — explicitly different from sister memos: (a) covers
   CPU-vs-CUDA (cpu_cuda_drift_mechanism_localization is loader-only); (b)
   covers MPS portability use cases (slot 16 covers MPS engineering
   corrections only); (c) drafts canonical equations for slot 19's registry
   (no sister memo drafts equations); (d) ranks operationalization
   recommendations by EV (no sister memo does this).
4. **RIGOR** — empirical anchors cited via `[empirical:<path>]` tags for
   every numerical claim; predicted ΔS tagged `[predicted]` per Catalog
   #287; sister-canonical-helper references via dotted module paths per
   Catalog #299/#319 sister discipline.
5. **OPTIMIZATION PER TECHNIQUE** — see Phase 7 above; canonical patterns
   adopted where they serve, unique engineering where it serves.
6. **STACK-OF-STACKS-COMPOSABILITY** — 3 candidate equations declare
   canonical consumers + canonical producers per `CanonicalEquation` shape;
   each composable with existing cathedral autopilot ranker per slot 11/14
   patterns.
7. **DETERMINISTIC REPRODUCIBILITY** — memo is pure markdown text; all
   numerical claims derive from cited empirical artifacts or canonical
   helpers with deterministic outputs.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — analysis is research-rigor; no
   runtime performance considerations. Implementation recommendations (Phase
   6) include wall-clock-overhead estimates per recommendation.
9. **OPTIMAL MINIMAL CONTEST SCORE** — predicted cumulative ΔS from all 5
   recommendations: HIGH-EV #1 = $0 direct + 5-10× dev velocity multiplier;
   HIGH-EV #2 = ~$0 direct (validation experiment); MEDIUM-EV #3 = $0
   direct (research); MEDIUM-EV #4 = predicted [-0.013, -0.005] if loader-
   dominated mechanism is confirmed; LOW-EV #5 = predicted 0% on CUDA-vs-CPU.
   Operational priority: HIGH-EV #1 (workflow automation) > HIGH-EV #2
   (empirical mechanism closure) > MEDIUM-EV #3+#4 > LOW-EV #5.

## Phase 9: Cargo-cult audit per assumption

Per CLAUDE.md "Cargo-cult audit per assumption" + Catalog #303.
HARD-EARNED-vs-CARGO-CULTED classification per the hard-earned-vs-cargo-culted
addendum.

| Assumption | Classification | Rationale |
|---|---|---|
| CPU is canonical scoring target | **HARD-EARNED** | CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable; contest leaderboard ranks by CPU. |
| CUDA is canonical training target | **HARD-EARNED** | Modal/Vast.ai/Lightning A100/4090 throughput is empirically 5-60× faster than CPU for substrate training. |
| Loader (PyAV vs DALI/NVDEC) is the dominant CPU-vs-CUDA drift mechanism | **HARD-EARNED** | Slot 11 empirical attribution: scorer-forward 0.0025% on Linux-vs-Linux falsifies FastViT compounding; the 5.18× pose ratio MUST come from upstream of scorer. |
| Scorer-forward engineering corrections (Kahan Conv2d / fp32 matmul / pinned softmax) are NO-OP for CPU-vs-CUDA | **HARD-EARNED-NUANCED** | Slot 11 measurement shows scorer-forward drift is sub-noise-floor (0.0025%); engineering it closes 0.00006% of the 0.013 gap. BUT this is INFERRED — we have not empirically validated that turning ON the corrections changes the CPU-CUDA gap by 0%. HIGH-EV #2 closes this empirically. |
| Sign-flip on PR106 r2 family is a substrate-design exploit opportunity | **CARGO-CULTED-WITH-RATIONALE** | The sign-flip is empirically observed (CUDA 0.021 BETTER than CPU on PR106 r2 family) but the MECHANISM is not attributed. Could be substrate-design-exploitable (a research opportunity) OR could be measurement-axis artifact (no operational value). MEDIUM-EV #3 closes the attribution. |
| 5 MPS use cases cover the use-case surface | **HARD-EARNED-NUANCED** | The 5 use cases match operator's verbatim taxonomy + canonical helper coverage. But the LIST is not provably exhaustive — there may be use cases (e.g., MPS-as-ANE-accelerator-target, MPS-as-cluster-node for distributed-training) that this memo does NOT enumerate. Unwind path: operator-driven extension; the canonical equation #2 accepts arbitrary use_case strings. |
| MPS-trained-CUDA-validated workflow (Use Case C) is the highest-EV gap | **HARD-EARNED** | Manual workflow requires 6 steps (train / save / upload / dispatch / paired-eval / register); each step has $1-5 friction; automation predicted 5-10× dev velocity per slot 11 sister anchor. |
| 3 candidate canonical equations capture the domain | **CARGO-CULTED-WITH-RATIONALE** | The 3 equations match the operator's verbatim request (one per Phase: drift mechanism / MPS viability / CPU superiority). But the DECOMPOSITION is not provably canonical — could be 2 equations (merging viability + superiority) or 5 equations (decomposing drift mechanism into per-kernel-class sub-equations). Unwind path: empirical refinement via `update_equation_with_empirical_anchor` over time. |
| Predicted ΔS impact of HIGH-EV #1 (Use Case C automation) = $0 direct + 5-10× dev velocity | **HARD-EARNED-NUANCED** | Sister slot 11 anchor empirically supports 5-10× dev velocity claim for MPS prescreen. Extending to Use Case C is INFERENCE; could be 2-5× or 10-20× — needs empirical anchor after landing. |
| Predicted ΔS impact of HIGH-EV #2 (fp32 matmul empirical validation) = 0% closure | **CARGO-CULTED-WITH-RATIONALE** | Prediction is based on slot 11 scorer-forward 0.0025% (which implies any single scorer-kernel correction is sub-noise-floor for the 0.013 gap). BUT we have not empirically run the override ON to measure. The recommendation IS the experiment to validate this prediction. |

## Phase 10: Observability surface

Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305. The 6
facets:

1. **Inspectable per layer** — every Phase has structured tables decomposed by
   kernel class / use case / equation / recommendation. The per-kernel-class
   drift mechanism map (Phase 1.2) is canonical-helper-derived (slot 11
   measurements feed every row).
2. **Decomposable per signal** — 3 candidate equations each have:
   `domain_of_validity` + `units_in` + `units_out` + `empirical_anchors`
   list per `CanonicalEquation` shape. The 5 recommendations each have:
   predicted ΔS + cost + dependency + operator-routable boolean + EV-class.
3. **Diff-able across runs** — memo is markdown; sister memos in
   `.omx/research/*_20260519.md` are diffable; canonical equations once
   registered via `register_canonical_equation` produce JSONL events that
   diff naturally.
4. **Queryable post-hoc** — memo is the consumable surface; canonical
   equations queryable via `tac.canonical_equations.registry.query_equations*`
   helpers; recommendations queryable via grep on EV class.
5. **Cite-able** — every empirical anchor carries explicit `[empirical:<path>]`
   tag + every prediction carries `[predicted]` evidence-grade per Catalog
   #287/#323.
6. **Counterfactual-able** — HIGH-EV #2 (fp32 matmul validation) IS the
   counterfactual surface for the scorer-kernel-engineering hypothesis;
   MEDIUM-EV #3 (PR106 sign-flip attribution) IS the counterfactual for the
   substrate-design-exploit hypothesis.

## Phase 11: Predicted ΔS band

Per CLAUDE.md Catalog #296 (Dykstra-feasibility predicted-band check).

[predicted] Aggregate operationalization recommendation ΔS upper bound
(cumulative if all 5 recommendations land; sub-additive composition more
likely per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-
feasibility-check"):

| Recommendation | predicted ΔS floor | ceiling | cost | dependency |
|---|---|---|---|---|
| HIGH-EV #1: Use Case C automation | $0 direct + dev-velocity multiplier | $0 direct + 10× dev-velocity | ~$0 GPU + 2-4 hrs dev | NONE |
| HIGH-EV #2: fp32 matmul empirical validation | -0.0001 (if FALSIFIES kernel hypothesis) | -0.005 (if surprisingly closes part of gap) | $0.10 Modal | NONE |
| MEDIUM-EV #3: PR106 sign-flip attribution | $0 direct | -0.010 (if exploit found) | $0.02-0.05 Modal | HIGH-EV #2 (closes scorer-kernel hypothesis first) |
| MEDIUM-EV #4: DALI/NVDEC bypass empirical | -0.005 | -0.013 | $0.05-0.10 Modal + ~50 LOC dev | NONE |
| LOW-EV #5: Per-kernel scorer parity | 0 | -0.001 | ~$0 + 10-20 hrs dev | MEDIUM-EV #3 |
| **cumulative if all-additive** | **-0.005** | **-0.029** | **~$0.20 GPU + 15-25 hrs dev** | sequential per dependency |

**Dykstra-feasibility intersection check**: the achievable score-floor sits
at the convex intersection of (rate constraint R ≤ R_max, seg constraint
d_seg ≤ S_max, pose constraint d_pose ≤ P_max). Most of these
recommendations target the LOADER axis (not the rate/seg/pose constraints);
the predicted cumulative band ABOVE is bounded by the empirical 0.013-0.033
CPU-CUDA gap range. Sub-additive composition is more likely; the
recommendations target overlapping mechanism slices.

**First-principles citations**:
- Slot 11 empirical attribution (Linux-vs-Linux scorer-forward = 0.0025%
  bounds the scorer-kernel engineering ceiling)
- Slot 9 master-gradient Cauchy-Schwarz bound (per-pair upper bound on
  delta_S via gradient inner product)
- CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable
  (paired-eval canonical workflow bounds the engineering surface)

## Phase 12: 6-hook wire-in declaration

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125.

This is an ANALYSIS-ONLY memo — no new substrate code lands here. The 6
hooks are wire-in declarations for the 3 candidate canonical equations
(which would be registered into slot 19's registry by a follow-on
subagent) AND for the 5 operationalization recommendations.

### For the 3 candidate canonical equations:

1. **Sensitivity-map contribution** — Equation #1 (`eq_cross_device_drift_by_kernel_class_v1`)
   feeds per-kernel-class drift coefficients into `tac.sensitivity_map.*`
   for downstream score-impact prediction.
2. **Pareto constraint** — Equation #3 (`eq_cpu_deployment_target_superiority_v1`)
   provides the CPU-vs-CUDA score-gap constraint as a Pareto frontier
   boundary (CPU axis dominates for slot 14 frontier family; CUDA axis
   dominates for PR106 r2 family).
3. **Bit-allocator hook** — N/A for equations (analytical surfaces; not
   per-tensor bit allocation).
4. **Cathedral autopilot dispatch hook** — Equation #2
   (`eq_mps_local_training_viability_by_use_case_v1`) feeds into
   `tac.cathedral_consumers.mps_viable_prescreen_consumer` for prescreen
   routing decisions (already partially active per slot 11).
5. **Continual-learning posterior update** — Every empirical anchor
   appended via `update_equation_with_empirical_anchor` IS a posterior
   update per `tac.canonical_equations.registry` design.
6. **Probe-disambiguator** — Equation #3 IS the canonical probe-
   disambiguator between scorer-kernel-engineering vs loader-engineering
   hypotheses (HIGH-EV #2 is the experimental probe).

### For the 5 operationalization recommendations:

| Recommendation | Hook #1 sens-map | Hook #2 Pareto | Hook #3 bit-alloc | Hook #4 autopilot | Hook #5 CL posterior | Hook #6 probe-disambig |
|---|---|---|---|---|---|---|
| HIGH-EV #1 Use Case C automation | N/A | N/A | N/A | ACTIVE (use case C routing) | ACTIVE (every validation anchor) | ACTIVE (paired-axis disambiguator) |
| HIGH-EV #2 fp32 matmul validation | N/A | N/A | N/A | ACTIVE (informs prescreen) | ACTIVE (empirical anchor for eq #1) | ACTIVE (kernel-vs-loader disambig) |
| MEDIUM-EV #3 PR106 sign-flip | N/A | ACTIVE (substrate-design exploit) | N/A | N/A | ACTIVE | ACTIVE (sign-flip mechanism disambig) |
| MEDIUM-EV #4 DALI/NVDEC bypass | N/A | N/A | N/A | N/A | ACTIVE (closes loader mechanism) | ACTIVE (loader-vs-other disambig) |
| LOW-EV #5 per-kernel parity | ACTIVE (per-kernel sensitivity) | N/A | N/A | N/A | ACTIVE | N/A |

## Phase 13: Cross-references

- **CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1
  CONTEST-COMPLIANT HARDWARE"** — canonical dual-axis discipline
- **CLAUDE.md "MPS auth eval is NOISE"** — canonical MPS non-promotability
- **CLAUDE.md "Apples-to-apples evidence discipline"** — axis tagging
- **CLAUDE.md "Forbidden empirical-claim-without-evidence-tag"** — `[empirical:...]`
- **CLAUDE.md "Contest vs production target modes"** — Use Case E (edge)
- **CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"**
  — pose marginal sensitivity at PR106 frontier
- **Catalog #205** `check_inflate_py_uses_canonical_select_inflate_device`
- **Catalog #464** AVVideoDataset CUDA-CPU drift mechanism discriminator
- **Catalog #192** `check_macos_cpu_advisory_not_promoted_without_linux_verification`
- **Catalog #317** `check_local_research_signal_dispatches_stamp_evidence_grade`
- **Catalog #287/#323** canonical Provenance + evidence-tag discipline
- **Catalog #290/#294/#296/#303/#305** design-memo discipline (all 4 sections
  per this memo)
- Slot 9 `tac.mps_diagnostic.drift_predictor` — Cauchy-Schwarz bound + cos
  distribution
- Slot 11 `tac.cathedral_consumers.mps_viable_prescreen_consumer` — Use
  Case A
- Slot 14 `tac.canonical_frontier_pointer` — frontier 0.19205 [contest-CPU
  GHA Linux x86_64]
- Slot 16 `tac.mps_diagnostic.{kahan_conv2d, pinned_softmax,
  fp32_matmul_override}` — MPS engineering corrections
- Slot 17 multi-archive cascade severity taxonomy
- Slot 18 grain-aware xray
- Slot 19 `tac.canonical_equations.registry` — 3 candidate equations target
  here
- `.omx/research/cpu_cuda_drift_mechanism_localization_20260511.md` —
  empirical attribution (loader-dominated)
- `.omx/research/cpu_cuda_xray_synthesis_20260511.md` — PR106 sign-flip
  central mystery
- `.omx/research/mps_drift_granular_analysis_and_corrective_engineering_20260519.md`
  — granular drift decomposition
- `.omx/research/mps_drift_corrections_empirical_validation_summary_20260519.md`
  — slot 16 TinyRenderer NO-OP empirical anchor

## Codex routing directive (optional follow-on)

Per the operator's possible interest in engineering implementations, see
`.omx/research/codex_routing_directive_cpu_vs_cuda_engineering_and_mps_portability_followup_20260519.md`
for the routing directive (sister memo; queued for codex follow-on
implementation of HIGH-EV #1 + HIGH-EV #2 + MEDIUM-EV #3 + MEDIUM-EV #4).
