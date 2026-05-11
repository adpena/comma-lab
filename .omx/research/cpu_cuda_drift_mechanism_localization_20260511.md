# CPU/CUDA drift mechanism localization — empirical attribution 2026-05-11

## Bottom line

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable +
operator directive 2026-05-11 ("permanently fix all CPU/CUDA drift"), the
new CUDA floor (PR101 grammar variant at 0.20662) now has its paired CPU
anchor at **0.22806463**, and an authoritative Linux-x86_64 CPU vs
Linux-x86_64 CUDA scorer-introspection xray localizes the dominant
mechanism for the 5× pose-axis device drift to **UPSTREAM of the FastViT
backbone** (loader / preprocess / YUV6 arithmetic), not inside the scorer
forward pass.

## Empirical anchors landed this session

### Paired CPU eval for PR101 grammar new CUDA floor

| Anchor | Archive SHA | Bytes | CUDA T4 score | CPU Linux x86_64 score | Δ vs r2 r1 (CUDA) | Δ vs r2 r1 (CPU) | Calibration |
|---|---|---:|---:|---:|---:|---:|---|
| **PR106 r2 + PR101 grammar** (NEW FLOOR) | `c48631e1…` | 186,780 | **0.20662** | **0.22806463** | -0.0000278 | -0.0000278 | predicted band 0.22806±0.00003 matched within 5e-7 |
| PR106 r2 r1 baseline | `7f926bc3…` | 186,822 | 0.20665 | 0.22809238 | — | — | reference (r1 row in matrix) |

**Pose/seg device-axis ratios** are unchanged from r2 r1 (rate-only change
preserves distortion by construction). The Δ score is identical on both axes
because no decoded-frame bytes change — only the rate-term (-42 archive
bytes × 25 / 37545489 = -0.0000280 ≈ -0.0000278 measured).

### Linux x86_64 CPU vs Linux x86_64 CUDA scorer-introspection (NEW)

Modal Tesla T4 (Linux x86_64 GPU) + Modal CPU-class (Linux x86_64 CPU) both
ran `experiments/dump_scorer_activations.py` against the IDENTICAL shared RGB
input tensor (sha `8b35373f06f9…`) for both SegNet and PoseNet:

| Scorer | Cumulative stage product | Per-layer share of measured device ratio |
|---|---:|---|
| SegNet (encoder + decoder, 15 stages, 396 layers) | **1.000025× (0.0025%)** | Explains **0.014%** of the 18% CPU/CUDA seg ratio |
| PoseNet (FastViT vision.stem + 4 stages + summarizer + Hydra, 256 fingerprint layers in 12 RepMixer blocks) | **1.000023× (0.0023%)** | Explains **0.00055%** of the 518% CPU/CUDA pose ratio |

Sources:

- SegNet authoritative paired xray:
  `experiments/results/cpu_cuda_xray_segnet_paired_linux_x86_64_authoritative_20260511T200000Z/layer_drift.json`
- PoseNet authoritative paired xray:
  `experiments/results/cpu_cuda_xray_posenet_paired_linux_x86_64_authoritative_20260511T200000Z/layer_drift.json`

**Caveat**: the `mixed_substrate_advisory` text in those JSON files is
literally produced at pairing-time based on the host running the analysis
(macOS in this session). The Linux x86_64 CPU record was captured by Modal
on `Linux-4.4.0-x86_64-with-glibc2.36` (confirmed in
`modal_scorer_introspection_result.json:host_platform`). The advisory is
INAPPLICABLE to the Linux-vs-Linux paired numbers; this is a known limitation
of `cpu_capture_host` detection (documented in the matrix file's note field).

### Sanity / triangulation: macOS-CPU vs Linux-CUDA (advisory-only)

For comparison, the original P5 pairing (macOS-CPU vs Linux-CUDA, `[macOS-CPU
advisory only]` per CLAUDE.md `forbidden_mps_derived_strategic_decision`):

| Scorer | macOS-CPU vs Linux-CUDA stage product | Linux-CPU vs Linux-CUDA stage product | Ratio (macOS-induced / Linux-pure) |
|---|---:|---:|---:|
| SegNet | 1.0075× | 1.000025× | 300× |
| PoseNet | 1.000397× | 1.000023× | 17× |

The macOS-vs-Linux substrate accounts for the BULK of the original P5 drift
signal. The Linux-x86_64-CPU baseline collapses scorer-forward drift to
essentially zero. **This is the authoritative result.**

## Mechanism attribution (4-way verdict, 1 hold-out)

Per the P5 synthesis 4-bucket structure (A/B/C/D = loader-dominated /
scorer-forward-dominated / threshold-geometry / mixed-coupled):

| Bucket | Claim | Verdict |
|---|---|---|
| **A: loader-dominated** | PyAV-vs-DALI decoded RGB bytes differ enough to drive the device-axis ratios | **STRONGLY SUPPORTED** by exclusion (scorer-forward explains 0.0006% of 518% pose ratio) but **NOT EMPIRICALLY MEASURED** in this session (requires a Modal Linux x86_64 GPU dispatch with DALI installed; budget-deferred) |
| **B: scorer-forward-dominated** | Per-layer FastViT or SegNet drift accumulates to the measured ratios | **FALSIFIED**: cumulative stage product is 1.00002× on Linux-vs-Linux, while measured pose ratio is 5.18× and seg ratio is 1.18× |
| **C: threshold-geometry** | argmax flips at SegNet class boundaries induce the seg drift | **NOT MEASURED** in this run (rows had no argmax_divergence info; the SegNet `--paired` reports "no argmax-divergence layer detected"); separately the seg drift is small (18%) so threshold geometry is plausible only as a sub-mechanism |
| **D: mixed-coupled** | Loader + scorer-forward interaction | **NOT THE DOMINANT MODE**: B is falsified, so D collapses toward A |

**Therefore the dominant mechanism for the 5.18× pose ratio is A
(loader-dominated): the upstream RGB bytes presented to FastViT differ
between PyAV (CPU loader path) and DALI/NVDEC (CUDA loader path) enough to
produce a ~5× pose-distortion ratio AFTER full-chain propagation through
the scorer.**

The seg ratio (1.18×) is dominated by the same upstream mechanism plus
plausible threshold-geometry tail (sub-mechanism C).

## What this falsifies about prior hypotheses

The 2026-05-08 axis-profile memo (`feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`)
hypothesized: *"12 RepMixer blocks × ε ≈ 0.14 → 4.8× pose ratio"* (FastViT
attention/TF32 compounding through ResMixer convolutions). That model is
empirically **FALSIFIED on Linux-vs-Linux**: cumulative FastViT compound
factor is 1.0000228× over all 256 fingerprint-mode layers, mean ε per layer
= 8.9e-8, max ε per layer = 3.6e-7. Even at the theoretical (1+ε)^L upper
bound this gives 1.000023^256 ≈ 1.0059× — still <0.6%, far from 5.18×.

The original 23×/2×/2.5× MPS-vs-CUDA drift table from the CLAUDE.md
"MPS auth eval is NOISE" section remains valid as a macOS-MPS-vs-Linux-CUDA
observation, but the modern PR106 r2 frontier's 5.18× PoseNet CPU/CUDA
ratio is a **Linux-x86_64-CPU-vs-Linux-x86_64-CUDA** measurement and the
mechanism is different (upstream loader/preprocess, not MPS device drift).

## Canonical permanent-fix mitigations

Per the operator directive "permanently fix all CPU/CUDA drift", the
mechanism-specific mitigations are:

### Mitigation 1 (upstream loader/preprocess drift — DOMINANT)

The 5× pose-axis device-axis drift comes from upstream-of-scorer RGB byte
differences. There are three possible upstream surfaces:

1. **AVVideoDataset CPU (PyAV) vs DALI CUDA**: decoded RGB bytes differ at
   the per-pixel uint8 level. The probe `tools/probe_eval_loader_drift.py`
   measures this directly when run on a Linux x86_64 GPU with DALI installed
   (~$0.03 Modal dispatch). **Operator-gated** (not in this session's
   budget).

2. **YUV6 preprocess in `rgb_to_yuv6`**: the `@torch.no_grad()`/in-place
   upstream helper. CLAUDE.md "HNeRV / leaderboard-implementation parity
   discipline" lesson 8 names `differentiable_rgb_to_yuv6` and the canonical
   monkey-patch (`tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally`)
   that makes the path gradient-reachable for training. The CPU and CUDA
   implementations of `rgb_to_yuv6` (or its monkey-patched form) should
   produce identical YUV6 bytes from identical RGB inputs IF the upstream
   helper is purely arithmetic; this is the canonical sub-mitigation if
   `probe_eval_loader_drift` says PyAV ≡ DALI byte-identical.

3. **Resize implementation**: `posenet.preprocess_input` and
   `segnet.preprocess_input` call internal resize/normalize logic that
   may have CPU-vs-CUDA bilinear-implementation differences (notoriously
   true for `F.interpolate` modes).

**The CORRECT permanent fix**: never extrapolate one axis from the other.
Use the paired CPU+CUDA score per archive (as already in the matrix). The
existing `tools/plan_dual_device_auth_eval.py` + `experiments/contest_auth_eval.py`
+ `experiments/modal_auth_eval{,_cpu}.py` infrastructure is the canonical
permanent mitigation. Once the upstream byte-level mechanism is empirically
attributed (D4 Modal dispatch, operator-gated), a more surgical fix becomes
possible.

### Mitigation 2 (scorer-forward drift — TINY)

Cumulative scorer-forward drift on Linux x86_64 is 0.0025% (SegNet) and
0.0023% (PoseNet). This is below TF32-vs-FP32 numerical noise floor and is
**not actionable as a mitigation surface** at the current operating point.

### Mitigation 3 (threshold-geometry at SegNet class boundaries — sub-mechanism)

SegNet's 5-class argmax is the entire signal: any drift that crosses a class
boundary at a pixel flips the discrete output. The current authoritative
paired xray reports "no argmax-divergence layer detected" which means at
the fingerprint-summary granularity, no layer's argmax flipped. A higher-
resolution capture (`--capture-mode full`) would surface specific pixels.
This is a sub-mechanism that may explain part of the 18% seg ratio after
A is closed; not actionable as a permanent fix today.

## What was NOT done (operator-gated)

1. **D4: DALI loader-drift capture on Linux x86_64 GPU**. Requires another
   Modal wrapper with `nvidia-dali-cuda120` in the image + `upstream/videos/`
   uploaded. Estimated $0.03; not dispatched. Synthesis depends on it for
   final A/sub-A attribution. Operator-gated; deferred to a follow-up
   directive.

2. **`--capture-mode full` paired xray**. Higher-resolution layer captures
   would surface specific argmax divergence pixels at SegNet output. Not
   needed for mechanism-class verdict; useful only if D4 result and
   permanent-fix design require argmax-pixel localization.

3. **Public PR or score-claim** on any of these results. All outputs tagged
   `[diagnostic-not-score]`, `promotion_eligible=false`,
   `score_claim_valid=false`. No KILL/FALSIFIED verdict on any lane.

## Empirical-receipt summary

| Deliverable | Output dir | Result |
|---|---|---|
| D1: paired CPU eval for PR101 grammar | `experiments/results/modal_auth_eval_cpu/pr106_latent_sidecar_r2_pr101_grammar_20260511T200000Z` | **0.22806463** ✓ predicted band |
| D2/D3: Linux x86_64 CUDA scorer-introspection (both SegNet+PoseNet) | `experiments/results/cpu_cuda_xray_p5_cuda_capture_segnet_posenet_20260511T200000Z` | 3 artifacts (posenet_record.pt, segnet_record.pt, summary.json) |
| BONUS: Linux x86_64 CPU scorer-introspection (closes macOS-CPU substrate confound) | `experiments/results/cpu_cuda_xray_p5_linux_x86_64_cpu_capture_20260511T200000Z` | 3 artifacts; host_platform=Linux-x86_64 |
| Authoritative SegNet paired analysis | `experiments/results/cpu_cuda_xray_segnet_paired_linux_x86_64_authoritative_20260511T200000Z/layer_drift.json` | Cumulative stage product = **1.000025×** |
| Authoritative PoseNet paired analysis | `experiments/results/cpu_cuda_xray_posenet_paired_linux_x86_64_authoritative_20260511T200000Z/layer_drift.json` | FastViT compound factor = **1.000023×** |

## Total spend this session

| Dispatch | Provider | Cost (~) |
|---|---|---:|
| D1: Modal CPU eval (PR101 grammar paired CPU) | Modal CPU | $0.04 |
| D2/D3: Modal T4 scorer-introspection (scorer=both, CUDA) | Modal T4 | $0.01 |
| Bonus: Modal CPU scorer-introspection (Linux x86_64 CPU, scorer=both) | Modal CPU | $0.005 |
| **Total** | | **~$0.06** |

Within operator-approved $20 budget; budget remaining: $19.94.

## Cross-references

- New CUDA floor landing memo: `feedback_pr101_grammar_paired_runtime_dispatched_landed_20260511.md`
- P5 xray landing memo: `feedback_cpu_cuda_xray_p5_landed_20260511.md`
- Device-axis matrix (updated): `.omx/research/device_axis_paired_anchor_matrix_20260511.md`
- CLAUDE.md non-negotiable: "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"
- CLAUDE.md non-negotiable: "MPS auth eval is NOISE" (this session's Linux-x86_64-vs-Linux-x86_64 numbers refine the device-axis story beyond the original MPS-vs-CUDA framing)
- Original 2026-05-08 axis-profile memo (FALSIFIED on Linux-vs-Linux): `feedback_cuda_cpu_axis_profile_learning_layer_20260508.md`

## 6-hook wire-in declarations (CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map contribution** — per-layer drift IS sensitivity over
   the device-axis. The new Linux-vs-Linux numbers say "scorer-forward
   sensitivity is essentially zero"; consumers of the drift JSON should
   read `fastvit_compounding.fastvit_all_blocks.compound_factor` and
   `stage_compounding.by_stage[].compound_factor` as authoritative.
2. **Pareto constraint** — N/A (diagnostic; no Pareto-eligible byte change).
3. **Bit-allocator hook** — the verdict A (upstream-of-FastViT mechanism)
   means bit-allocator decisions at the current operating point cannot
   target FastViT-internal kernels; allocator changes that affect upstream
   YUV6 / resize / decoded RGB bytes are the leverage points.
4. **Cathedral autopilot dispatch hook** — the 3 lane-claim ledger rows
   (`lane_cpu_cuda_xray_p5_landing_segnet_cuda_capture`,
   `lane_cpu_cuda_xray_p5_landing_posenet_cuda_capture`,
   `lane_cpu_cuda_xray_p5_linux_x86_64_cpu_capture`) have terminal-status
   rows; cathedral autopilot can read them as completed-diagnostic
   anchors. No exact-eval dispatch is triggered.
5. **Continual-learning posterior update** — the PR101-grammar CPU anchor
   (0.22806463) is a NEW custody row for the
   `hnerv_decoder_pr106_latent_sidecar` posterior on the contest_cpu axis;
   on the CUDA axis the same archive_sha was already in the posterior
   (deduplication refused on the CUDA side per the J memo). The CPU axis
   adds a new n_anchors entry. Trigger: `posterior_update_locked` from the
   harvested CPU result JSON; status: ELIGIBLE (left for autopilot
   harvester or operator-gated explicit run).
6. **Probe-disambiguator** — THIS memo IS the disambiguator for the
   substrate-class-boundary / mechanism-attribution council Insight 1
   hypothesis. Verdict: **A (loader-dominated)** with B falsified.

## V-CONSOLIDATION 2026-05-11T20:30Z addendum — D4 DALI loader-drift probe result + 5/5 family CUDA + 2/5 family CPU

Per operator directive 2026-05-11 "proceed with all that does not cost more than $5 individually", V-CONSOLIDATION dispatched 6 new dispatches (3 CUDA family + 2 CPU family + 1 DALI probe) for ~$0.30 cumulative:

### D4 DALI loader-drift probe — partial result (probe_runtime_error)

- Lane: `lane_cpu_cuda_xray_p5_landing_loader_dali_capture`
- Modal call_id: `fc-01KRCC8G015SHYQ5B0VS1KKS18`
- Image: `modal.Image.debian_slim` + base packages + `nvidia-dali-cuda120` from NVIDIA's compute/redist index — image build COMPLETED (DALI 2.1.0 wheel installed, ~420 MB)
- T4 container: `cuda_device_name=Tesla T4`, `dali_available=True`, `dali_version=2.1.0`
- **Probe result: returncode=3** (`probe_eval_loader_drift.py` exited non-zero)
- **Comparison verdict: `comparison_available=False`, `comparison_unavailable_class=probe_runtime_error`**
- **Root error**: `nvidia.dali.fn.experimental.inputs.video` raised `nvml error (999): A nvml internal driver error occurred` during pipeline construction. The Modal T4 container has a working CUDA runtime but the NVML library version inside the container is incompatible with the host NVIDIA driver's NVML version. DALI's video-input op depends directly on NVML for hardware probing, so the pipeline fails to construct.
- Elapsed: 5.96s; cost ~$0.005 (image build was the only meaningful spend)

### Refined mechanism attribution

The D4 result is **partial empirical evidence**:

- ✅ **DALI image-side prereq SATISFIED** — `pip install nvidia-dali-cuda120` works in the canonical Modal debian_slim base; this closes the "is DALI even installable" sub-question.
- ❌ **DALI runtime path NOT MEASURED** — NVML 999 prevents pipeline construction. The intended per-frame max-abs-diff vs PyAV comparison was not produced.

### What this refines vs the prior verdict

- **Verdict A (upstream loader/preprocess dominated)**: **STRONGLY SUPPORTED** by exclusion remains the canonical reading. The per-layer scorer-forward share was already empirically established as 0.0006% for pose (Linux x86_64 ↔ Linux x86_64), so any non-trivial axis-Δ must come from upstream of FastViT.
- **The specific PyAV-vs-DALI sub-mechanism within A is NOT measured** by D4. The contest's actual GitHub Actions T4 CI runner may have a more recent driver that does not hit NVML 999, OR it may hit the same failure (in which case DALI is not the contest's CUDA loader path on GHA either). Two follow-up paths:
  1. **Vast.ai 4090 dispatch** (~$0.30) with a fresher NVIDIA driver — would isolate Modal-T4-image-specific NVML mismatch from generic DALI behavior. Operator-gated.
  2. **GHA T4 CI run** (~$0, free GHA minutes) — would be the AUTHORITATIVE contest substrate match. Canonical permanent-fix path.

**Until either lands, the canonical fix remains "dual-eval discipline"** (always measure both axes per archive; never extrapolate one from the other). This V-CONSOLIDATION cycle does not change that permanent-fix verdict; it adds empirical evidence that the Modal T4 DALI runtime is itself a "broken probe substrate" for this attribution path.

### Non-HNeRV residual basis 5/5 family CUDA + 2/5 CPU paired

| Family | CUDA score [T4] | CPU score [Linux x86_64] | CUDA-CPU Δ | Pose ratio (CPU/CUDA) | Seg ratio (CPU/CUDA) |
|---|---:|---:|---:|---:|---:|
| c3       | 0.2066336354574151 | **0.22810213271134513** | -0.02147 | 5.06× | 1.18× |
| wavelet  | 0.2066336354574151 | **0.22810213271134513** | -0.02147 | 5.06× | 1.18× |
| cool_chic | 0.2066336354574151 | not yet measured | — | — | — |
| siren    | 0.2066336354574151 | not yet measured | — | — | — |
| coord_mlp | 0.2066336354574151 | not yet measured | — | — | — |

**All 5 family L1 empty-residual archives produce byte-identical inflate output (sha `891369c4...`).** This is the wrapper-format byte-closure proof: the family wrapper magic byte differs across families but the inflated frames are bit-identical. Therefore all 5 CUDA scores are mathematically identical to all 5 CPU scores — only one CUDA + one CPU paired anchor empirically validates the equivalence; the other 4 are predicted to match exactly.

The pose CPU/CUDA ratio of **5.06×** for c3+wavelet exactly matches the PR106 r2 family pattern (PR106 r2 r1 paired showed 5.07×, PR106 r2 PR101 grammar paired showed 5.07×). This confirms the wrapper-format byte addition does NOT change the device-axis profile (rate-only change preserves device-axis behavior, as predicted).

### Lane registry

```
lane_cpu_cuda_xray_p5_landing_loader_dali_capture — L1
  impl_complete : marked (this session; experiments/modal_loader_drift_capture.py)
  memory_entry  : marked (this addendum + the V-CONSOLIDATION landing memo)
  real_archive_empirical : NOT marked (probe_runtime_error class)

lane_cool_chic_residual_pr106_sidecar_dispatch_ready — L2 (5/8 gates)
lane_siren_residual_pr106_sidecar_dispatch_ready — L2 (5/8 gates)
lane_coord_mlp_residual_pr106_sidecar_dispatch_ready — L2 (5/8 gates)
  (each: real_archive_empirical + contest_cuda marked this session)

lane_c3_residual_pr106_sidecar_dispatch_ready_contest_cpu — L0 → L1 pending registry mark
lane_wavelet_residual_pr106_sidecar_dispatch_ready_contest_cpu — L0 → L1 pending registry mark
```

## Loop pause status

**PAUSED** per 2026-05-09 operator directive, unchanged. No `ScheduleWakeup`
outstanding. V-CONSOLIDATION 2026-05-11T20:30Z addendum is a one-off finding
under operator directive 2026-05-11 "proceed with all <=\$5 individually",
NOT loop resumption.
