# logic/experiments.md

Experiment declarations. Each entry binds a claim id from `claims.md` to
the executable code that tested it and the evidence file that records the
outcome. An Ara-native reviewer can verify any claim by walking
`claim -> experiment -> code -> evidence` without touching prose.

---

## E1 — PoseNet Jacobian SVD across 30 pairs

- **tests**: C1
- **prereq**: dilated-h64 renderer, 600 non-overlapping pairs
- **command**: `python experiments/jacobian_svd_analysis.py`
- **artifact**: `evidence/jacobian/svd_summary.json`
- **success criterion**: effective rank in [1.0, 1.2] over 30+ pairs
- **failure mode if regressed**: condition number > 1e4 or rank > 2 — would
  invalidate the rank-1 narrative and require a different post-filter
  architecture story.

## E2 — Trust-region sweep on `J*delta`

- **tests**: C2
- **prereq**: same as E1
- **command**: `python experiments/trust_region_sweep.py --range 1e-5..1e-3`
- **artifact**: `evidence/jacobian/trust_region_sweep.json`
- **success criterion**: median relative linearization error >= 0.5 at
  perturbation 1e-4
- **failure mode if regressed**: would suggest a Newton/Jacobian one-shot
  method might work; would require revisiting the closed-form alternative.

## E3 — CNN residual pixel-density and DCT signature

- **tests**: C3
- **prereq**: trained Era 1 h=32 EMA checkpoint
- **command**: `python experiments/karpathy_cnn_residual_analysis.py`
- **artifact**: `evidence/cnn_residual/karpathy_signature.json`
- **success criterion**: dense (>50% of pixels move > 0.5 LSB) AND
  mid-frequency-concentrated (>80% energy in band 4-32 cycles/frame)
- **failure mode if regressed**: would weaken the case for the CNN's
  inverse-rendering interpretation.

## E4 — Width scaling sweep h in {8, 16, 32, 48, 64}

- **tests**: C4
- **prereq**: same Era 1 recipe (QAT + EMA + best-checkpoint, alpha=20,
  1000 epochs)
- **command**: `python experiments/pipeline.py --profile width_sweep`
- **artifact**: `evidence/era1/width_scaling/summary.csv`
- **success criterion**: log-linear fit R^2 > 0.97
- **failure mode if regressed**: would need to retract the scaling-law claim;
  the paper's Section 5.1 figure depends on it.

## E5 — Best-checkpoint int8 gap measurement

- **tests**: C5
- **prereq**: 1000-epoch h=64 training trace with periodic int8 evaluation
- **command**: `python experiments/measure_qat_gap.py --trace ...`
- **artifact**: `evidence/era1/qat_ema_best_checkpoint/gap_summary.json`
- **success criterion**: mean(deployed_int8 / fp32_proxy) <= 1.1 with
  best-checkpoint selection enabled
- **failure mode if regressed**: best-checkpoint mechanism is the central
  Era 1 trick; failure here would mean we cannot generalize the recipe.

## E6 — MPS-vs-CUDA paired measurement

- **tests**: C6
- **prereq**: identical archive bytes evaluated on local MPS and CUDA A100
- **command**: `bash submissions/exact_current/inflate.sh && python upstream/evaluate.py`
  on each device
- **artifact**: `evidence/era2/mps_cuda_drift/summary_2026-04-25.json`
- **success criterion**: PoseNet ratio MPS/CUDA reproducible >= 5x
- **failure mode if regressed**: would suggest the MPS bug was transient and
  the `[MPS-PROXY]` tag is over-conservative. The current strict check would
  flag and gate any code path that defaults to MPS.

## E7 — Lane A pose TTO contest-CUDA eval

- **tests**: C7
- **prereq**: dilated-h64 anchor + baseline poses
- **command**: see `evidence/era2/lane_a_landed/run_command.txt`
- **artifact**: `evidence/era2/lane_a_landed/contest_auth_eval.json`
- **success criterion**: score in [1.10, 1.20]
- **failure mode if regressed**: would mean the rank-1 basin claim is wrong
  or the warm-start advantage is reproducible only on Vast.ai 4090.

## E8 — Lane G v3 KL-distill landing + Modal repro

- **tests**: C8
- **prereq**: Lane A anchor
- **command**: see `evidence/era2/lane_g_v3_landed/run_command.txt`
- **artifact**: `evidence/era2/lane_g_v3_landed/contest_auth_eval.json`,
  `evidence/era2/modal_repro/9b20bdfca246.json`
- **success criterion**: contest-CUDA in [1.00, 1.10]; Modal T4 within
  0.02 of Vast.ai 4090
- **failure mode if regressed**: would invalidate the Era 2 frontier claim;
  would require either a different KL weight or a new Era 2 path.

## E11 — PFP16 A++ exact T4 auth eval

- **tests**: C11
- **prereq**: Lane G v3 renderer archive, PFP16 pose payload, canonical
  `archive.zip -> inflate.sh -> upstream/evaluate.py` path
- **command**: see
  `../../../experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/run_command.sh`
- **artifact**:
  `../../../experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/eval/contest_auth_eval.json`,
  `../../../experiments/results/lane_g_v3_pfp16/final_deploy_bundle_20260430/custody/custody_manifest.json`
- **success criterion**: exact CUDA eval on Tesla T4, `gpu_t4_match=true`,
  `n_samples=600`, archive SHA
  `0af839abb30e0dfdcfbcbf75247b136db8731196ef26e58374c76a1b562ded7f`,
  recomputed score `1.043987524793892`
- **failure mode if regressed**: do not parse logs or build provenance; use
  `contest_auth_eval.json` only and quarantine stale parser fields.

## E9 — Lane M-V3 clean retest of rank-1 input subspace (PENDING)

- **tests**: C9
- **prereq**: Lane M-V2 BUG-1 fix landed (Check 42 STRICT live)
- **command**: TBD; lane spec in `trace/dead_ends_to_revisit.md`
- **artifact**: `evidence/era2/lane_m_v3_clean/contest_auth_eval.json`
  (will exist post-run)
- **success criterion**: score in [1.05, 1.20]
- **status**: PENDING. Not run because compute budget is currently flowing
  to Era 3 Selfcomp lanes.

## E10 — STRICT preflight catalog enumeration

- **tests**: C10
- **prereq**: `src/comma_lab/preflight/strict_checks.py` HEAD
- **command**: `python tools/preflight_catalog_dump.py`
- **artifact**: `evidence/preflight/strict_check_catalog.json`
- **success criterion**: every CLAUDE.md-listed bug class maps to a
  `check_*` function with `strict=True`
- **failure mode if regressed**: a documented bug class without a static
  check would invalidate the rigor narrative.
