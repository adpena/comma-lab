# logic/claims.md

Each claim below carries explicit forensic bindings: pointers to the
experiment that tested it, the source code that generated the evidence, and
the raw evidence file. Failure to find a binding means the claim is not
substantiated and should not appear on a public surface.

Format: each claim is one block. `id` is stable across compilations.

---

## C1 — PoseNet Jacobian has effective rank ~1

- **statement**: Per-pair `J = dPose/dPixel` has entropy-based effective rank
  ~1.008 out of 6, with top singular value 45x larger than the second.
- **falsifier**: an effective rank > 1.5 across >= 30 sampled frame pairs.
- **status**: empirically supported.
- **evidence**: `evidence/jacobian/svd_summary.json`
- **code**: `experiments/jacobian_svd_analysis.py`
- **experiment**: `experiments.md#E1`
- **public_safe**: yes.

## C2 — PoseNet trust radius is below 1e-4 pixels RMS

- **statement**: The Moore-Penrose linear approximation of PoseNet has median
  relative linearization error >= 1.0 at the 1e-4 pixel RMS knee. Concentrated
  pseudoinverse corrections degrade pose distortion (0.0742 -> 0.2349, 3x
  worse).
- **falsifier**: median relative error < 0.5 at any tested perturbation
  magnitude in (1e-5, 1e-3).
- **status**: empirically supported.
- **evidence**: `evidence/jacobian/trust_region_sweep.json`,
  `evidence/jacobian/pseudoinverse_failure.json`
- **code**: `experiments/jacobian_optimal.py`,
  `experiments/trust_region_sweep.py`
- **experiment**: `experiments.md#E2`
- **public_safe**: yes.

## C3 — Trained CNN post-filter residual is dense and mid-frequency

- **statement**: For the shipped Era 1 h=32 post-filter, 56.6% of pixels move
  by > 0.5 LSB and 90.3% of luma residual energy lies in the 4-32 cycles/frame
  mid-band.
- **falsifier**: mid-band fraction < 60% across 20+ pairs.
- **status**: empirically supported.
- **evidence**: `evidence/cnn_residual/karpathy_signature.json`
- **code**: `experiments/karpathy_cnn_residual_analysis.py`
- **experiment**: `experiments.md#E3`
- **public_safe**: yes.

## C4 — Width scaling is log-linear within the tested range

- **statement**: For the Era 1 post-filter, score follows
  `score = -0.159 ln(h) + 2.382` validated at h in {8, 16, 32, 48, 64} with
  R^2 > 0.97.
- **falsifier**: any tested width with score residual > 0.1 from the fit.
- **status**: empirically supported within the tested range; extrapolation
  beyond h=64 is not claimed.
- **evidence**: `evidence/era1/width_scaling/summary.csv`
- **code**: `src/comma_lab/task_codec/training.py`
- **public_safe**: yes.

## C5 — Best-checkpoint int8 selection closes the train-to-deploy gap

- **statement**: Without best-checkpoint selection, deployed int8 PoseNet
  distortion is up to 2.25x larger than the fp32 training proxy. With
  best-checkpoint selection, the gap collapses to the noise floor.
- **falsifier**: deployed-int8 / fp32 ratio remains > 1.5 across
  best-checkpoint runs.
- **status**: empirically supported.
- **evidence**: `evidence/era1/qat_ema_best_checkpoint/gap_summary.json`
- **code**: `src/comma_lab/task_codec/quantization.py`
- **public_safe**: yes (the technique is publishable).

## C6 — MPS scorer measurements are NOT contest-equivalent

- **statement**: PoseNet on Apple MPS produces pose distortion 23x larger
  than CUDA on identical model artifacts; final score drift is ~2.5x
  (2.26 MPS vs 0.90 CUDA on the dilated-h64 + CRF=50 pinned baseline).
- **falsifier**: PoseNet drift < 5x across any matched MPS/CUDA pair.
- **status**: empirically supported, single decisive measurement; preflight
  Check 1 (`check_no_mps_fallback_default`) prevents future regressions.
- **evidence**: `evidence/era2/mps_cuda_drift/summary_2026-04-25.json`
- **code**: `src/comma_lab/preflight/strict_checks.py:check_no_mps_fallback_default`
- **public_safe**: yes (this is differentiating engineering rigor).

## C7 — Pose TTO from baseline poses lands inside the rank-1 basin

- **statement**: 6-DOF per-pair pose TTO warm-started from the baseline poses
  (rather than zero or random init) reduces PoseNet from ~0.247 to ~0.0034
  (73x improvement) while incurring +401KB archive growth. Cold-start does
  NOT work.
- **falsifier**: cold-started TTO matching warm-start within 2x PoseNet at
  equal compute.
- **status**: empirically supported.
- **evidence**: `evidence/era2/lane_a_landed/contest_auth_eval.json`
- **code**: `src/tac/optimize_poses.py`
- **public_safe**: yes (this is the headline Era 2 mechanism).

## C8 — KL distill weight=0.002 dominates weight=0.01 on the same renderer

- **statement**: KL distillation on SegNet logits at T=2.0 with weight=0.002
  improves both PoseNet (-31%) and SegNet (-13%) at the same archive size,
  whereas weights >= 0.01 collapse PoseNet.
- **falsifier**: any tested weight in [0.001, 0.005] that fails to reproduce
  the joint improvement.
- **status**: empirically supported on Lane G v3 (Vast.ai 4090) and reproduced
  on Modal T4 within 0.01 noise.
- **evidence**: `evidence/era2/lane_g_v3_landed/contest_auth_eval.json`,
  `evidence/era2/modal_repro/9b20bdfca246.json`
- **code**: `src/tac/training.py`, `src/tac/profiles.py`
- **public_safe**: yes (the mechanism); the precise weight schedule for
  derivative lanes is private.

## C9 — Train/inference pose-pad asymmetry is a recurring measurement bug

- **statement**: The Lane M-V2 radial-zoom regression (1.84 vs Lane A 1.15)
  was not a refutation of the rank-1 hypothesis; it was a train/inference
  asymmetry where the optimizer fed `[zoom,0,0,0,0,0]` while inflate fed
  `[zoom, baseline_1..5]`. The hypothesis remains untested at the renderer
  input subspace.
- **falsifier**: a clean Lane M-V3 reproduction passing pose dim 1-5 through
  the projection helper at both train and inference time, scoring > 1.20.
- **status**: bug confirmed; clean retest pending.
- **evidence**: `evidence/era2/lane_m_v2_audit/council_findings.md`
- **code**: `src/tac/optimize_poses.py`,
  `src/comma_lab/preflight/strict_checks.py:check_42_train_inference_parity`
- **public_safe**: yes (the engineering catch is publishable; the precise
  Lane M-V3 protocol is private).

## C10 — Engineering rigor is publishable independent of any further score

- **statement**: The lab's preflight catalog grew from 36 to 78 STRICT checks
  in one week; every catastrophic measurement-bug class has a static check.
- **falsifier**: a documented measurement-bug class without a corresponding
  STRICT preflight check.
- **status**: empirically supported as of 2026-04-29.
- **evidence**: `evidence/preflight/strict_check_catalog.json`
- **code**: `src/comma_lab/preflight/strict_checks.py`
- **public_safe**: yes (this is the rigor moat under the score moat).
