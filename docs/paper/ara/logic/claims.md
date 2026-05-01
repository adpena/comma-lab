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

## C8 — KL/distillation-family status is forensic-gated

- **statement**: Primary scorer KL is promotion-ineligible. SegNet-auxiliary
  KL, JBL, and related distillation-family variants are hypotheses, not a
  public-safe mechanism claim, until exact CUDA archive eval proves archive
  SHA/bytes, full component gates, and explicit non-collapse evidence. Lane
  G v3/PFP16 remains the current score anchor, but it must not be attributed
  to a proven KL recipe without matched exact-CUDA ablations.
- **falsifier**: any public/paper claim that a KL-family recipe is promoted,
  causally proven, or safe without exact CUDA archive custody plus matched
  component evidence.
- **status**: forensic-gated policy. Historical primary-KL failures remain
  negative evidence; scoped auxiliary-KL variants require fresh exact CUDA
  proof before promotion or mechanism claims.
- **evidence**: `evidence/era2/lane_g_v3_landed/contest_auth_eval.json`,
  `evidence/era2/modal_repro/9b20bdfca246.json`,
  `.omx/research/kl_distill_hardening_grand_council_review_20260430_agent.md`
- **code**: `src/tac/training.py`, `src/tac/profiles.py`,
  `src/tac/losses.py`, `src/tac/losses_jbl.py`
- **public_safe**: yes only as a rigor lesson and gating policy; no KL-family
  mechanism is public-safe as a promoted result without matched ablation proof.

## C11 — PFP16 A++ is the current contest-grade frontier

- **statement**: The PFP16 A++ archive is the current exact contest-grade
  frontier: recomputed score `1.043987524793892`, archive `686635` bytes,
  SHA-256 `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  `n=600`, Tesla T4 CUDA, `gpu_t4_match=true`.
- **falsifier**: exact `contest_auth_eval.py --device cuda` on the same
  archive SHA recomputes a different score or fails payload closure.
- **status**: A++ exact archive evidence. Legacy remote-provenance parser
  fields (`contest_cuda_score=100.0`, `hard_kill_triggered=true`,
  `lane_status=HARD_KILL_REGRESSION`) are quarantined and invalid for claims.
- **evidence**:
  `../../../experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`,
  `../../../experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/custody/custody_manifest.json`
- **code**: `src/tac/pfp16_codec.py`,
  `experiments/build_lane_g_v3_pfp16_stack.py`,
  `submissions/robust_current/inflate_renderer.py`
- **experiment**: `experiments.md#E11`
- **public_safe**: yes, only with the custody manifest and
  `contest_auth_eval.json` authority note.

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
