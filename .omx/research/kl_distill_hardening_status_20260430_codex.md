# KL Distill Hardening Status - 2026-04-30

Purpose: preserve contest-grade interpretation of all KL/distillation results
and prevent a repeat of the primary-KL PoseNet-collapse class. This is a
control-plane document, not a score ledger.

## Controlling Verdict

Primary scorer KL distillation is forensic-only. It previously produced
authoritative PoseNet collapse, so it must not be used for promotion candidates.

SegNet-only auxiliary KL is not broadly killed. It is permitted only when all of
the following are true:

- `TrainConfig.loss_mode == "kl_distill"` has explicit
  `kl_distill_scope == "segnet_aux"`.
- The loss is standard scorer loss plus `kl_distill_weight *
  kl_distill_segnet_only(...)`, not replacement primary scorer KL.
- The KL input is the same eval-roundtripped renderer output used by the scorer
  loss, not raw pre-roundtrip renderer output.
- The reduction is per-pixel-per-class: `reduction="none"`, sum over class,
  mean over batch/spatial, then `T^2`.
- The operator-supplied `kl_distill_weight` and `kl_distill_temperature` are
  plumbed into the trainer and recorded in artifacts.
- Promotion requires exact CUDA auth eval on the exact archive, full sample
  count, recomputed components, and no PoseNet collapse versus the PFP16 A++
  baseline.

## Current Code Controls

- `src/tac/training.py`
  - `TrainConfig.kl_distill_scope` disambiguates `none`, `segnet_aux`, and
    `primary_scorer`.
  - `loss_mode="kl_distill"` with `kl_distill_scope="none"` is rejected.
  - `primary_scorer` requires `allow_banned_primary_kl_distill=True` and
    `promotion_eligible=False`.
  - KL temperature bounds reject historically unstable configs.
- `src/tac/losses.py`
  - `kl_distill_segnet_only` uses the corrected per-pixel reduction and
    preserves `T^2` scaling.
- `src/tac/segmap_renderer.py`
  - SegMapTrainer routes KL only as a SegNet auxiliary on the eval-roundtripped
    output.
  - Codex 2026-04-30 guard: SegMapTrainer now fails closed unless
    `kl_distill_scope == "segnet_aux"`.
- `experiments/train_segmap.py`
  - Non-plain SegMap variants map to explicit `kl_distill_scope="segnet_aux"`.
  - `--kl-distill-weight` and `--kl-distill-temperature` are threaded into
    `TrainConfig`.
- `src/tac/preflight.py`
  - Static guard for raw-pair KL call sites.
  - Reduction guard for `kl_distill_segnet_only`.
- Tests:
  - `src/tac/tests/test_config_validation.py`
  - `src/tac/tests/test_kl_distill_weight_plumbed.py`
  - `src/tac/tests/test_losses.py`
  - `src/tac/tests/test_preflight_meta_bugs.py`

## Remaining Hardening Work

Completed after xhigh KL council return:

- `src/tac/segmap_renderer.py` now rejects forensic/primary KL scope in
  SegMapTrainer.
- `experiments/optimize_poses.py` now treats `--kl-distill-snr-target` as an
  active KL path even when static `--kl-distill-weight` is omitted:
  GT frame pairs are materialized, per-step KL logging uses the effective
  controller weight, and Lane PS warnings no longer report false no-op status
  when the controller is active.
- `src/tac/training.py` checkpoint metadata now records KL scope, weight,
  temperature, forensic opt-in, and promotion eligibility.
- `src/tac/preflight.py` now includes `src/tac/segmap_renderer.py` in the
  roundtripped-KL raw-pair scanner.

Remaining:

1. Add a strict remote-script preflight that any script using
   `--variant kl_distill`, `--loss-mode kl_distill`, or
   `--kl-distill-weight > 0` must emit provenance fields:
   `kl_distill_scope`, `kl_distill_weight`, `kl_distill_temperature`,
   `eval_roundtrip`, and `promotion_eligible`.
2. Add adjudicator component gates for KL-active lanes:
   - CUDA device required.
   - `n_samples == 600`.
   - Archive SHA and byte count must match the submitted ZIP.
   - PoseNet and SegNet components must be compared to the PFP16 A++ frontier,
     not only total score.
3. Add lane-attribution metadata: if a lane stacks KL with logit margin,
   homography, sensitivity weighting, or architecture changes, the result must
   not be attributed to KL alone without ablations or an exact stacked-design
   claim.
4. Regenerate stale public docs that still say broad "KL dead" without scope.
   Correct wording: "primary scorer KL distill is forensic-only; scoped
   SegNet-auxiliary KL remains an experimental auxiliary under exact gates."
5. For active KL lanes, require a post-harvest Grand Council review before
   promotion or retirement:
   - verify no silent standard-loss fallback,
   - verify exact scope/provenance,
   - compare PoseNet component,
   - evaluate mitigation/stacking,
   - retire only the measured config if negative.

## Lane Interpretation Rules

- Historical primary KL failures are valid negative evidence for that primary
  scorer-loss design, not for all distillation.
- Historical SegMap KL failures before the reduction/plumbing/roundtrip fixes
  are confounded unless exact artifacts prove otherwise.
- Active or future SegNet-aux KL lanes are high-risk but valid experiments when
  scoped and provenance-complete.
- A disappointing KL-active result does not kill KL. It triggers exact artifact
  custody, component diagnosis, ablation planning, and scoped retirement.
- Any KL-active improvement must be treated as a stack claim unless there is an
  exact ablation isolating the KL contribution.

## Verification

2026-04-30T17:50Z focused verification passed:

- `py_compile` for touched KL/OWV3/Lightning Python files.
- `bash -n scripts/remote_lane_g_v3_owv3_fisher_stack.sh`.
- `pytest` focused suite:
  `test_owv3_sensitivity_conversion.py`,
  `test_remote_lane_g_v3_owv3_fisher_stack_script.py`,
  `test_optimize_poses_kl_distill_wiring.py`,
  `test_kl_distill_weight_plumbed.py`,
  `test_config_validation.py`,
  `test_losses.py`,
  `test_preflight_meta_bugs.py`,
  `test_lightning_batch_jobs.py`.
- Result: `291 passed`.

## 2026-04-30T21:49Z Renderer KL Scope Closure

Additional KL council gap closed:

- `src/tac/experiments/train_renderer.py` now exposes
  `--kl-distill-scope` and refuses positive `kl_distill_weight` unless the
  effective scope is exactly `segnet_aux`.
- `train_renderer.py` refuses `kl_distill_scope="primary_scorer"` outright.
  That path remains forensic-only in lower-level training code and cannot
  enter renderer-training lanes.
- Current positive-KL profiles in `src/tac/profiles.py` now declare
  `kl_distill_scope="segnet_aux"`.
- `src/tac/preflight.py` adds strict
  `check_train_renderer_kl_aux_explicit_scope`, preventing future profiles
  from activating KL/JBL auxiliaries from weight alone.
- Regression coverage landed in
  `src/tac/tests/test_train_renderer_auth_eval_wiring.py`.

Updated remaining work:

- Remote-script provenance enforcement for KL-active lanes is still needed,
  but renderer profile activation is now fail-closed locally.
- Any KL-active lane still requires exact CUDA archive eval and component gates
  before promotion, and disappointing results remain scoped implementation
  evidence only.
