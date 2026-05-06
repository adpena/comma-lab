# src/index.md — Ara physical-layer index

This is a **kernel-mode** Ara index: rather than vendoring the full
implementation, we map each scientific claim to the production source code
that produced its evidence. The repository proper at `../../../src/`
remains the single source of truth.

For Ara reviewers running the optional Level 3 (execution reproducibility)
seal: every command listed in `../logic/experiments.md` is intended to run
from the repository root with the standard `.venv/bin/python` interpreter.

---

## Module map (claim -> code)

| Claim | Module                                          | Brief                                                |
|-------|--------------------------------------------------|------------------------------------------------------|
| C1    | `experiments/jacobian_svd_analysis.py`           | per-pair SVD of `dPose/dPixel`                       |
| C1    | `src/tac/jacobian.py`                            | finite-difference Jacobian helper                    |
| C2    | `experiments/trust_region_sweep.py`              | linearization-error sweep                            |
| C2    | `experiments/jacobian_optimal.py`                | Moore-Penrose single-step (failed)                   |
| C3    | `experiments/karpathy_cnn_residual_analysis.py`  | residual pixel-density + DCT signature               |
| C4    | `experiments/pipeline.py`                        | width-sweep entry point (--profile width_sweep)      |
| C4    | `src/comma_lab/task_codec/training.py`           | post-filter trainer with QAT + EMA                   |
| C5    | `src/comma_lab/task_codec/quantization.py`       | best-checkpoint int8 selection                       |
| C6    | `src/comma_lab/preflight/strict_checks.py`       | check_no_mps_fallback_default (#1)                   |
| C7    | `src/tac/optimize_poses.py`                      | per-pair pose TTO with rank-1 warm-start             |
| C8    | `src/tac/training.py`                            | KL distill loss term (T=2.0, weight=0.002)           |
| C8    | `src/tac/profiles.py`                            | Lane G v3 profile                                    |
| C11   | `src/tac/pfp16_codec.py`                         | PFP16 pose payload codec                             |
| C11   | `experiments/build_lane_g_v3_pfp16_stack.py`     | PFP16 archive builder                                |
| C9    | `src/comma_lab/preflight/strict_checks.py`       | check_42_train_inference_parity                      |
| C10   | `src/comma_lab/preflight/strict_checks.py`       | full STRICT check catalog                            |

## Entry points by use case

- **Reproduce Era 1 best (1.73 advisory)**:
  ```bash
  python experiments/pipeline.py --profile proven_baseline --device cuda \
    --output-dir results/era1_repro
  ```
- **Reproduce current PFP16 A++ floor (1.04 [contest-CUDA A++])**:
  ```bash
  .venv/bin/python experiments/contest_auth_eval.py \
    --archive experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/archive/archive.zip \
    --inflate-sh submissions/robust_current/inflate.sh \
    --upstream-dir upstream \
    --device cuda \
    --keep-work-dir \
    --work-dir experiments/results/lane_g_v3_pfp16/repro_eval_work
  ```
  The authoritative completed artifact is
  `experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`.
- **Reproduce historical Era 2 predecessor (1.05 [contest-CUDA])**:
  ```bash
  bash submissions/robust_current/inflate.sh \
    && python upstream/evaluate.py
  ```
  using the archive at `experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip`.
- **Run historical Modal T4 reproduction of the Lane G v3 predecessor (1.04)**:
  ```bash
  .venv/bin/python experiments/modal_auth_eval.py \
    --archive experiments/results/lane_g_v3_landed/archive_lane_g_v3.zip
  ```

## configs/

Profile definitions live in `../../../src/tac/profiles.py`. The relevant
profiles for the publishable claims:

| Profile             | Era | Brief                                    |
|---------------------|-----|------------------------------------------|
| `proven_baseline`   | 1   | h=64 QAT+EMA recipe (1.73 advisory)      |
| `psd_standard_adaptive` | 1 | PSD architecture (alternate, advisory) |
| `lane_g_v3`         | 2   | KL distill weight=0.002 + pose TTO retry |
| PFP16 archive       | 2   | deterministic fp16 pose payload on Lane G v3 renderer |

Profile keys NOT published here (Lane W, Lane Omega, Lane DARTS-S) are
gated by the disclosure policy in `../PAPER.md`.

## environment.md (compact)

- Python: 3.13 (uv-managed `.venv/`)
- PyTorch: 2.5+ (CUDA 12.x for contest-CUDA evals; MPS only for Era 1
  proxy training; CPU for canonical local smoke)
- NVDEC: pre-DALI probe required (Vast.ai NVDEC roulette mitigation)
- Hardware envelopes:
  - contest-CUDA A++: Tesla T4 or contest-equivalent hardware with exact
    `contest_auth_eval.py --device cuda` artifact custody
  - contest-CUDA A: CUDA exact eval on other audited GPUs such as A100/4090
  - Era 1 long-horizon training: single consumer GPU, ~12h for h=64 at
    1000 epochs
  - Era 2 lane training: see `../../../experiments/results/<lane>/provenance.json`

## Forbidden mutation frontier

Per repository protocol, the following paths are immutable for the agent:

- `../../../upstream/` (pinned scorer snapshot)
- `../../../submissions/exact_current/inflate.py`
- `../../../submissions/exact_current/inflate.sh`
- `../../../start.sh`
- `../../../LICENSE`
- `../../../THIRD_PARTY_NOTICES.md`

This Ara physical-layer index respects that frontier: nothing in `src/`
above lives inside the immutable set.
