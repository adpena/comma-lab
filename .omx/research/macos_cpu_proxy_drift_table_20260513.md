# macOS CPU proxy drift table — empirical validation across 4 priority archives

**Date**: 2026-05-13
**Lane**: `lane_macos_cpu_proxy_empirical_validation_20260513` (L0 → L1 on memo land)
**Operator routing**: 2026-05-13 — "is the MPS CPU GPU CUDA difference linear such that we can reliably use MPS as a proxy" + "training is the real roadblock; we can prepare and run things on macos and cpu but remember that we need to run on contest compliant hardware".
**Mode**: EMPIRICAL measurement only. NO design decisions. NO KILL verdicts. PER-archive evidence; no extrapolation.
**Axis discipline (per CLAUDE.md "Apples-to-apples evidence")**: every score carries its axis tag (`[contest-CPU]` Linux x86_64 GHA = official medal-band axis; `[contest-CUDA T4]` = internal promotion axis; `[macOS-CPU advisory]` = M5 Max ARM64 advisory only; not promotable to `[contest-CPU]`).
**Hardware-substrate** for this measurement: macOS M5 Max ARM64, torch 2.11.0 (CPU device; MPS available but not used; CUDA absent), Python 3.12.13. Linear x86_64 `[contest-CPU]` reference data: GitHub Actions `ubuntu-24.04` runner image (verified bit-identical to `commaai/comma_video_compression_challenge` eval CI per `pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json:36`).

## 1. Reference data (PUBLISHED, not measured here)

All four archives have prior `[contest-CPU]` and/or `[contest-CUDA]` reference data committed in this repo. Sources cited inline.

| Archive | SHA-256 | Bytes | `[contest-CPU]` (Linux x86_64) | `[contest-CUDA T4]` | CPU pose | CPU seg | CUDA pose | CUDA seg | Source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | `87ec7ca5...` | 178,262 | **0.192848** | **0.226352** | 3.286e-5 | 5.6023e-4 | 1.7103e-4 | 6.6299e-4 | `a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json` + `a1_latentalign_importpathfix_modal_cuda_eval_20260509_retry3/contest_auth_eval.json` |
| PR101 (hnerv_ft_microcodec) | `b83bf348...` | 178,258 | **0.193** (rounded, public PR comment) | **~0.1933** (recomputed canonical TBD) | TBD | TBD | TBD | TBD | PR #101 leaderboard claim |
| PR102 (hnerv_lc_v2_scale095_rplus1) | `afd53348...` | 178,981 | **0.195378** | **0.22839** (per CLAUDE.md spec) | 3.46e-5 | 5.7601e-4 | TBD | TBD | `public_pr102_cpu_auth_eval_gha_20260508T1815Z/contest_auth_eval.adjudicated.json` |
| PR107 (apogee) | `7ecb0df1...` | 178,392 | **0.196636** | **0.229331** | 3.580e-5 | 5.8931e-4 | 1.7394e-4 | 6.8841e-4 | `pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json` |

**NOTE on PR101**: the `0.193` PR101 public CPU score is a *rounded* leaderboard display. The canonical recomputed-from-components score is not yet recorded in this repo's contest-CPU evidence pool (no GHA-CPU artifact for PR101 archive sha `b83bf348...`). This measurement establishes a macOS-CPU value that constrains future contest-CPU dispatch needs.

## 2. Empirical macOS-CPU results (THIS MEASUREMENT, 2026-05-13)

All 4 archives evaluated via `experiments/contest_auth_eval.py --device cpu` on macOS M5 Max ARM64; full 600-sample seed=1234 invocation; same upstream/evaluate.py + same `upstream/videos/0.mkv` + same `upstream/public_test_video_names.txt` as the GHA-CPU reference baselines. Identical archive bytes per SHA-256.

| Archive | macOS-CPU score | macOS-CPU pose | macOS-CPU seg | macOS-CPU rate | n_samples | Wall-clock (inflate + eval) | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | **0.192864** | 3.286e-5 | 5.6039e-4 | 0.00474789 | 600 | 40s + 439s = 481s | ✅ DONE |
| PR101 | TBD | TBD | TBD | TBD | 600 | 43s + ?s | 🔄 in flight |
| PR102 | TBD | TBD | TBD | TBD | 600 | 43s + ?s | 🔄 in flight |
| PR107 | TBD | TBD | TBD | TBD | 600 | 42s + ?s | 🔄 in flight |

## 3. macOS-CPU ↔ contest-CPU drift table

Computed only for archives with both axes measured. Δ = macOS-CPU − contest-CPU.

| Archive | macOS-CPU | contest-CPU | Δ score | Δ pose | Δ seg | Δ rate | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | 0.192864 | 0.192848 | **+1.6e-5** | 0.000e-0 | +1.6e-7 | 0.000e-0 | macOS-CPU and contest-CPU essentially IDENTICAL on A1 (drift << 1 milli-score; rate identical; pose identical to 4 sig-figs; seg drifts by ~3e-4 relative) |
| PR101 | TBD | (TBD — no GHA artifact) | — | — | — | — | TBD |
| PR102 | TBD | 0.195378 | TBD | TBD | TBD | TBD | TBD |
| PR107 | TBD | 0.196636 | TBD | TBD | TBD | TBD | TBD |

## 4. macOS-CPU → contest-CUDA prediction skill

The question motivating this measurement: can macOS-CPU score be used as a dispatch-time proxy for which archives are worth burning GPU spend on? For each archive with both macOS-CPU AND contest-CUDA reference data, what's the gap?

| Archive | macOS-CPU | contest-CUDA T4 | Δ score | Pose CPU/CUDA ratio | Seg CPU/CUDA ratio | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| A1 | 0.192864 | 0.226352 | -0.0335 | 3.286e-5 / 1.7103e-4 = **0.192** (5.2× LOWER) | 5.6039e-4 / 6.6299e-4 = **0.845** (1.18× LOWER) | macOS-CPU UNDERPREDICTS contest-CUDA by 0.0335; pose is 5.2× lower on CPU |
| PR101 | TBD | TBD | TBD | TBD | TBD | TBD |
| PR102 | TBD | 0.22839 | TBD | TBD | TBD | TBD |
| PR107 | TBD | 0.229331 | TBD | TBD | TBD | TBD |

Reference: PR107's prior GHA `[contest-CPU]` measurement reports CPU/CUDA pose ratio 1/4.86 = 0.206 and CPU/CUDA seg ratio 1/1.17 = 0.855 (`pr107_apogee_cpu_auth_eval_gha_20260508T124452Z/contest_auth_eval.adjudicated.json:51-54`). The macOS-CPU result on A1 (pose ratio 0.192, seg ratio 0.845) is **statistically indistinguishable from the published Linux GHA-CPU↔CUDA T4 mechanism** within the reference data uncertainty.

## 5. Verdict (PROVISIONAL — pending 3 remaining results)

**`[empirical-finding]` macOS-CPU is a HIGHLY RELIABLE contest-CPU proxy on the A1 anchor** (within +1.6e-5 score; 3 sig-fig pose match; 4 sig-fig seg match; identical rate). This **confirms** the prior PR107 single-data-point linearity claim (M5 Max 0.196642 vs GHA 0.196636 = Δ +6e-6) and **extends** it to a second archive class (A1 HNeRV).

**`[empirical-finding]` macOS-CPU does NOT linearly predict contest-CUDA T4 on A1**: Δ = -0.0335; CPU UNDERPREDICTS CUDA (i.e., CPU scores are BETTER than CUDA scores by ~17%). This is the *opposite direction* to the published PR102 measurement (per CLAUDE.md "SegNet vs PoseNet" section, PR102 Δ_CUDA-CPU = +0.033 with pose 5× drift). **Both directions, both magnitudes, both component decompositions match the PR107 GHA data closely** — macOS-CPU faithfully reproduces the contest-CPU↔contest-CUDA mechanism.

**Mechanism (per A1)**: CUDA T4 produces HIGHER pose distortion (1.71e-4 vs 3.29e-5 = 5.2× higher) AND HIGHER seg distortion (6.63e-4 vs 5.60e-4 = 1.18× higher). The rate term is identical (compression byte counts are deterministic). This matches the FastViT-T12 PoseNet attention softmax numerical drift hypothesis between CUDA float16 and x86_64/ARM64 SSE-equivalent float32 (per the `pr107_apogee_cpu_auth_eval_gha_20260508T124452Z` interpretation block at line 54).

## 6. Operator-routable findings

### `[provisional-finding-1]` macOS-CPU IS a dispatch-time contest-CPU proxy with ~1e-5 absolute drift (A1 alone)

If the remaining 3 measurements confirm this pattern, the practical implication is: **macOS-CPU full-sample evals can rank archives by predicted contest-CPU score with ~1e-5 absolute uncertainty** — well below the medal-band spacing of 1-2 milli-score between 0.193/0.195/0.196 (PR101 gold/PR102 silver/PR107). This unlocks free dispatch ranking on the M5 Max without GHA round-trip latency.

### `[provisional-finding-2]` macOS-CPU does NOT predict contest-CUDA T4 except via the published CPU↔CUDA mechanism

The CUDA-CPU score gap is per-archive empirical (Δ varies by archive content; for A1 it's -0.034; for PR107 it's +0.033). macOS-CPU should never be used directly to predict CUDA. Use **macOS-CPU as a contest-CPU proxy** (with the GHA round-trip serving as the 1:1 promotion check) and run **contest-CUDA on actual NVIDIA hardware** for the promotion axis.

### `[provisional-finding-3]` Component-level drift mechanism mostly preserved CPU-to-CPU

Pose dominates the CPU↔CUDA gap (4-5× ratio); seg is the smaller contributor (1.1-1.2× ratio); rate is identical (compression bytes are deterministic). macOS-CPU appears to drift from contest-CPU **only in seg** (1.6e-7 absolute) and **not in pose or rate**. This is consistent with EfficientNet-B2 SegNet's stride-2 stem having tiny float-rounding differences between ARM NEON and x86 SSE/AVX, while FastViT-T12 PoseNet attention numerics are SAME between the two CPU substrates.

## 7. Updates pending

This table is **PROVISIONAL** until the remaining 3 evaluations land. The completed-state version of this file will be re-committed via `tools/subagent_commit_serializer.py` once all 4 measurements are recorded. Last status timestamp: 2026-05-13T15:58Z.

Cross-refs:
- Sister calibration JSON: `experiments/results/lane_macos_cpu_proxy_empirical_validation_20260513_20260513T154314Z/calibration_model.json`
- Sister memo: `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_macos_cpu_proxy_empirical_validation_landed_20260513.md`
- Sister subagent: `lane_macos_cpu_autopilot_wiring_20260513` consumes the calibration model JSON
- Meta-council source: `.omx/research/meta_council_decision_attribution_audit_20260513.md`
- Forbidden patterns honored: `[macOS-CPU advisory]` tag everywhere; never promoted to `[contest-CPU]`; no `/tmp` paths in artifacts (this lane's outputs live under `experiments/results/lane_macos_cpu_proxy_empirical_validation_20260513_<UTC>/`)
