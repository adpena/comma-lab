# KL Distill Hardening Grand Council Review - 2026-04-30

Author: Codex agent review
Status: adversarial design memo, not a score ledger
Scope: KL, distillation, SegNet-logit, Jaccard/BLS soft-label, pose-embedding
distillation, scorer-distillation, profiles, tests, lane scripts, and control
documents.

## Executive Verdict

Primary scorer KL remains forensic-only. It may explain historical failures or
run under an explicit banned-primary waiver, but it must not be routed through a
promotion path, ranked against other methods, or used as a kill signal.

The only future-safe KL-like path is a strictly auxiliary, explicitly scoped
SegNet objective stacked on top of the standard contest scorer objective:

```text
loss = standard_score_surrogate(roundtripped_render, ground_truth)
     + kl_distill_weight * mean_pixel_KL_T(
           SegNet(roundtripped_render), SegNet(teacher_frames)
       )
```

That form is not promotable by construction. It becomes evidence only after its
own exact CUDA archive eval on exact bytes through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
```

The hard separation is:

- Dead historical KL configs: primary scorer KL, old Lane G KL-TTO, batchmean
  spatial KL artifacts, any CPU/MPS/proxy/byte-only KL claim, and any
  unretuned 1.0-weight post-fix KL profile or script.
- Future-safe KL-like objectives: explicitly scoped SegNet-only auxiliary KL,
  explicitly declared non-KL soft-label distillation such as JBL/JML-style
  losses, and compress-time teacher-label distillation, provided they preserve
  exact-eval custody, payload closure, component gates, ablations, and
  provenance.

No production code was changed by this review. I found one important hardening
gap but not a safe one-line guard patch: `loss_mode="segnet_kl"` and the
`SEGNET_KL_*` profiles still expose a legacy KL-like training path without the
same explicit `kl_distill_scope` and promotion-fencing contract that now guards
`loss_mode="kl_distill"`. It should be fenced or retired before any full lane
dispatch.

## Evidence Rules

All score, ranking, promotion, retirement, and kill statements obey the contest
formula:

```text
score = 100 * seg_dist
      + sqrt(10 * pose_dist)
      + 25 * archive_bytes / 37,545,489
```

Evidence grades used in this memo:

- `A++`: exact 1:1 contest-grade archive custody and clean exact CUDA eval.
- `A`: exact local CUDA score-grade archive evidence with full custody and
  component recomputation.
- `A-negative`: exact archive CUDA evidence showing the measured implementation
  regresses.
- `B`: diagnostic CUDA with incomplete custody.
- `empirical`: byte, smoke, partial, loss, round-trip, or component-only signal.
- `derivation`: formula or mathematical reasoning only.
- `prediction`: hypothesis or forecast.
- `external`: paper, documentation, or OSS motivation.
- `invalid`: CPU, MPS, proxy, stale, no-op, sidecar, missing archive, or
  unreproducible score evidence.

Only exact CUDA auth eval can promote, rank, kill, or retire a method. Local
MPS/CPU, proxy scorer checks, byte-only wins, smoke tests, and training losses
are diagnostic only.

## Sources Inspected

Control plane:

- `AGENTS.md`
- `CLAUDE.md`
- `.omx/research/kl_distill_hardening_status_20260430_codex.md`
- `docs/BATTLE_PLAN.md`

Implementation:

- `src/tac/losses.py`
- `src/tac/losses_jbl.py`
- `src/tac/training.py`
- `src/tac/segmap_renderer.py`
- `src/tac/lagrangian_kl_weight.py`
- `src/tac/archive/scorer_distill.py`
- `src/tac/experiments/train_renderer.py`
- `experiments/train_segmap.py`
- `experiments/optimize_poses.py`
- `experiments/train_distill.py`
- `experiments/quantize_distilled.py`

Profiles and scripts:

- `src/tac/profiles.py`
- `src/tac/preflight.py`
- KL/distill lane scripts under `scripts/remote_lane_*.sh`
- focused remote-script tests under `src/tac/tests/`

Tests:

- `src/tac/tests/test_losses.py`
- `src/tac/tests/test_kl_distill_weight_plumbed.py`
- `src/tac/tests/test_optimize_poses_kl_distill_wiring.py`
- `src/tac/tests/test_lagrangian_kl_weight.py`
- `src/tac/tests/test_preflight_meta_bugs.py`
- profile and remote lane contract tests touching KL/JBL/distillation lanes

External primary sources:

- Hinton, Vinyals, and Dean, "Distilling the Knowledge in a Neural Network",
  arXiv:1503.02531:
  https://arxiv.org/abs/1503.02531
- Google Research publication page for "Distilling the Knowledge in a Neural
  Network":
  https://research.google/pubs/distilling-the-knowledge-in-a-neural-network/
- PyTorch `KLDivLoss` documentation:
  https://docs.pytorch.org/docs/2.11/generated/torch.nn.KLDivLoss.html
- PyTorch `torch.nn.functional.kl_div` documentation:
  https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.kl_div.html
- Wang, Ning, and Blaschko, "Jaccard Metric Losses: Optimizing the Jaccard
  Index with Soft Labels", arXiv:2302.05666:
  https://arxiv.org/abs/2302.05666

External sources motivate objective design only. They do not provide contest
evidence.

## Mathematical Contract

For a spatial SegNet tensor with logits shaped `(B, C, H, W)`, the safe helper
should compute:

```text
p_s(c | b,h,w) = softmax(student_logits[b,:,h,w] / T)
p_t(c | b,h,w) = softmax(teacher_logits[b,:,h,w] / T)

KL_T = mean_{b,h,w} sum_c p_t(c | b,h,w)
       * (log p_t(c | b,h,w) - log p_s(c | b,h,w))

aux = T^2 * KL_T
```

The reduction unit is a pixel distribution, not a whole image tensor. PyTorch
documents `batchmean` as sum divided by batch size, while `mean` divides by all
elements. For this contest's spatial SegNet logits, a naive `batchmean` over a
`(B,C,H,W)` output overweights the per-pixel objective by approximately `H*W`
relative to the intended pixel average. The corrected implementation therefore
uses:

```text
F.kl_div(..., reduction="none").sum(dim=class_dim).mean()
```

The `T^2` multiplier follows the standard distillation gradient-scale convention
from Hinton-style knowledge distillation. It is not evidence that a specific
temperature or weight is safe in this contest. Weight selection must be justified
by measured loss-scale ratios and exact CUDA archive evidence.

The auxiliary objective must not replace the standard contest surrogate. A safe
future loss has this shape:

```text
L_total = L_standard_scorer_surrogate + w_aux * L_segnet_aux
```

It must not have this shape:

```text
L_total = L_primary_pose_kl_or_mse + L_segnet_kl_only
```

The second form redefines the training target and is historically associated
with PoseNet collapse and component miscalibration. It is forensic-only unless a
new Grand Council review reopens it with exact evidence.

## Historical Failure Taxonomy

Primary scorer KL:

- Implementation: `kl_distill_scorer_loss` in `src/tac/losses.py`.
- Current guard: generic trainer only allows it with
  `kl_distill_scope="primary_scorer"`,
  `allow_banned_primary_kl_distill=True`, and `promotion_eligible=False`.
- Verdict: forensic-only. It can be used to reproduce or diagnose old collapse
  behavior, not to promote or rank.

Spatial `batchmean` KL:

- Historical bug: applying `F.kl_div(reduction="batchmean")` to spatial SegNet
  logits divided by `B`, not by `B*H*W`, over-weighting the KL term by the
  spatial area relative to a per-pixel objective.
- Current guard: `kl_distill_segnet_only` uses per-pixel reduction and tests
  assert resolution invariance.
- Verdict: all old artifacts whose safety depended on the bugged scale are
  historical and must be retuned.

Old Lane G KL TTO:

- Script: `scripts/remote_lane_g_kldistill_pose_tto.sh`.
- Status: deprecated/superseded by corrected V3 variants.
- Verdict: forensic-only. Its `5e-6` scale compensated for the old reduction
  bug and cannot be compared with corrected 0.002-scale runs without exact
  archive evidence.

Unretuned 1.0-weight profiles and scripts:

- Examples in `src/tac/profiles.py`: WILDE, SHIRAZ, DEN,
  DILATED_H64_HALF_FRAME, SELF_COMPRESS_RENDERER_FULL.
- Examples in lane scripts: per-class SegNet pose scripts using
  `--kl-distill-weight 1.0`.
- Verdict: non-promotable until retuned under corrected per-pixel KL. A 1.0
  weight may only re-enter through an explicit sweep, loss-ratio record,
  component gates, and exact CUDA archive eval.

CPU/MPS/proxy KL claims:

- Verdict: invalid for promotion, ranking, retirement, or kill. They may remain
  in debugging ledgers only if clearly labeled diagnostic.

## Current Guard Strengths

The repo now has several contest-grade protections:

- `AGENTS.md` and `.omx/research/kl_distill_hardening_status_20260430_codex.md`
  explicitly forbid primary KL promotion.
- `TrainConfig` exposes `kl_distill_scope`, `kl_distill_weight`,
  `kl_distill_temperature`, `allow_banned_primary_kl_distill`, and
  `promotion_eligible`.
- `TrainConfig` rejects `loss_mode="kl_distill"` without a scope, and rejects
  `primary_scorer` unless the banned-primary waiver is explicit and promotion
  eligibility is false.
- `SegMapTrainer` requires `eval_roundtrip=True`, rejects MPS, and only accepts
  `loss_mode="kl_distill"` with `kl_distill_scope="segnet_aux"`.
- `experiments/train_segmap.py` uses the corrected default
  `--kl-distill-weight 0.002` and temperature `2.0`.
- `kl_distill_segnet_only` implements per-pixel reduction, temperature scaling,
  finite-temperature checks, optional class weights, and teacher no-grad.
- Tests cover per-pixel KL scale, temperature scaling, gradients, metadata
  plumbed through training, SegMap guard behavior, `optimize_poses.py` wiring,
  and controller bounds.
- `preflight.py` scans for raw-frame KL call sites and `batchmean` KL in source,
  experiments, scripts, and submissions.

These are real controls, but they are not yet enough to treat every KL-like lane
as promotion-safe.

## Open Hardening Gaps

1. `loss_mode="segnet_kl"` is not fenced like `loss_mode="kl_distill"`.

   `SEGNET_KL_SMOKE` and `SEGNET_KL_FULL` still exist in `src/tac/profiles.py`.
   The training path adds `segnet_kl_divergence_loss` to the standard loss, but
   it does not require explicit `kl_distill_scope`, a banned-primary waiver, or
   `promotion_eligible=False`. This should be frozen as forensic until a new
   guard and scale review lands.

2. Some comments and profiles still encode stale post-batchmean assumptions.

   `train_renderer.py` and older profiles reference weight `1.0` as if it were
   a proven KL recipe. That was not proven after the per-pixel reduction fix.
   It must be treated as historical configuration text, not dispatch guidance.

3. KL teacher round-trip semantics are not globally identical.

   `train_renderer.py` can roundtrip both rendered and ground-truth pairs before
   auxiliary KL. SegMap and pose-optimization call sites roundtrip the rendered
   student frames but use the ground-truth/teacher frames as the teacher input.
   This may be defensible, but it must be explicit in provenance. A promotion
   path should state whether the teacher distribution is raw-GT teacher,
   roundtripped-GT teacher, or scorer-input teacher, then pin that contract in
   tests.

4. Adaptive "Lagrangian" language is too strong.

   `src/tac/lagrangian_kl_weight.py` is a bounded proportional ratio controller,
   not a proof-bearing Lagrange multiplier method. Lane records must call it an
   SNR/ratio controller unless a formal dual derivation and exact archive
   evidence are added.

5. Distillation variants are not uniformly tagged as KL-like.

   JBL/JML, pose embedding distillation, and scorer-distillation heads are not
   KL losses, but they still import a teacher-student objective into a contest
   training or runtime path. They need the same provenance, payload-closure, and
   component-gate discipline.

6. Remote scripts need stricter machine-readable KL provenance.

   The status addendum already calls this out. Every lane with a KL-like knob
   must write structured provenance for weight, temperature, scope, eval
   roundtrip, teacher semantics, controller trajectory if adaptive, exact eval
   command, device, archive bytes, archive SHA-256, and component gates.

## Fail-Closed Config Contract

All KL-like promotion-capable config must satisfy the following checks before
training is allowed to start:

- `eval_roundtrip=True` unless the run is explicitly marked smoke or forensic
  and `promotion_eligible=False`.
- CUDA device for promotable training signals where scorer gradients or exact
  CUDA behavior matter. CPU/MPS is smoke only.
- `loss_mode="kl_distill"` must set `kl_distill_scope="segnet_aux"` for any
  future-safe path.
- `loss_mode="kl_distill"` with `kl_distill_scope="primary_scorer"` must require
  `allow_banned_primary_kl_distill=True` and `promotion_eligible=False`, and
  must never be routed through `SegMapTrainer`.
- `kl_distill_weight` must be finite, nonnegative, and recorded.
- `kl_distill_temperature` must be finite and positive, with start/end schedules
  recorded if annealed.
- Any KL class weighting must record class order, finite nonnegative weights,
  derivation source, calibration split, and holdout stability.
- Any adaptive weight controller must record initial weight, target ratio,
  lower/upper bounds, per-step or summarized trajectory, final weight, and
  nonfinite-input behavior.
- `loss_mode="segnet_kl"` must be rejected from promotion-capable profiles until
  it is migrated to the same explicit scope and metadata contract as
  `kl_distill_segnet_only`.
- Any profile with `kl_distill_weight >= 0.1` must require an explicit
  `kl_scale_review_tag`, loss-ratio artifact, and `promotion_eligible=False`
  until exact CUDA archive evidence proves component safety.

Recommended immediate guard:

```text
if loss_mode == "segnet_kl":
    require promotion_eligible is False
    require explicit forensic flag or new segnet_aux migration flag
```

That should be paired with tests before any dispatch changes.

## Invariant Test Plan

Existing tests to keep:

- Per-pixel KL reduction is resolution-invariant.
- `batchmean` spatial KL is detected by preflight.
- `kl_distill_segnet_only` is zero on identical teacher/student distributions.
- Teacher logits receive no gradients.
- Student logits receive gradients.
- Temperature scaling uses the configured value.
- TrainConfig records KL scope, weight, temperature, and promotion status in
  checkpoint metadata.
- SegMap KL requires `segnet_aux`, refuses primary scope, and enforces
  roundtrip.
- `optimize_poses.py` gates KL on effective positive weight and available
  ground-truth frames.
- SNR controller rejects nonfinite or negative residuals and obeys bounds.

New tests to add:

1. `test_segnet_kl_profile_is_forensic_only`

   Assert `SEGNET_KL_FULL` and `SEGNET_KL_SMOKE` cannot be promotion-eligible
   unless they are migrated to an explicit `segnet_aux` contract.

2. `test_train_config_rejects_promotion_segnet_kl`

   Instantiate `TrainConfig(loss_mode="segnet_kl", promotion_eligible=True)` and
   require a validation failure or explicit forensic override.

3. `test_remote_kl_scripts_emit_provenance_schema`

   Static-scan `scripts/remote_lane_*` for every `--kl-distill-weight`,
   `--loss-mode jbl`, `--loss-mode segnet_kl`, or `--variant kl_distill`, and
   require JSON fields:
   `loss_family`, `kl_distill_scope`, `kl_distill_weight`,
   `kl_distill_temperature`, `eval_roundtrip`, `teacher_roundtrip_contract`,
   `promotion_eligible`, `auth_eval_device`, `archive_sha256`, and
   `archive_bytes`.

4. `test_kl_teacher_roundtrip_contract`

   Pin whether each call site uses raw teacher frames or roundtripped teacher
   frames. This avoids silent semantic drift in future refactors.

5. `test_no_high_weight_kl_without_review_tag`

   Static-scan profiles and scripts for `kl_distill_weight >= 0.1`; require a
   non-promotable flag or an explicit `kl_scale_review_tag`.

6. `test_jbl_is_distill_family_not_pose_safe_by_assertion`

   Prevent docs or metadata from claiming that JBL/JML "cannot" harm PoseNet.
   It may be hypothesized to be safer, but only exact component eval can prove
   it for a specific archive.

7. `test_no_kl_attribution_without_ablation`

   Harvest/adjudication should refuse "KL improved score" language unless a
   matched non-KL archive exists with exact CUDA custody or the claim is marked
   prediction/diagnostic.

## Preflight Rules

`preflight_all` should continue to fail on:

- `F.kl_div(..., reduction="batchmean")` in source, experiments, scripts, or
  submissions outside an explicit negative test.
- `kl_distill_segnet_only` fed raw rendered frames where roundtripped frames are
  required.
- promotion metadata missing `kl_distill_scope`, weight, temperature, and
  roundtrip contract.
- `AUTH_EVAL_DEVICE` not resolving to `cuda` for promotable runs.
- `loss_mode="kl_distill"` with no explicit scope.
- `primary_scorer` KL with promotion eligibility true.
- hidden sidecars, zip-slip paths, or score-affecting payloads outside archive.

New preflight gates should be added for:

- `loss_mode="segnet_kl"` in any promotion-capable profile.
- `kl_distill_weight >= 0.1` without scale-review metadata.
- SNR/controller KL lanes missing controller trajectory.
- remote scripts that pass KL/JBL/distill flags but do not write structured
  provenance.
- docs or reports that use broad `killed`, `proven`, or `PoseNet-safe` language
  for KL-like objectives without exact evidence.

## Exact CUDA Evidence Requirements

No KL-like run is promotable until it produces:

- exact `archive.zip` bytes and SHA-256;
- deterministic archive manifest with member ordering, timestamps,
  permissions, compression settings, sizes, and hashes;
- payload closure proof showing all score-affecting files are inside the
  archive or fixed contest code;
- clean canonical eval:

```bash
.venv/bin/python experiments/contest_auth_eval.py \
  --archive <candidate archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir upstream \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence dir>
```

- `contest_auth_eval.json` with full sample count, archive bytes, SegNet
  distance, PoseNet distance, recomputed score, device, hardware, command, and
  logs;
- source/staged-tree manifest;
- exact upstream hash;
- comparison against a matched non-KL or lower-risk ablation before attributing
  causality to KL;
- component gates, not just total score. PoseNet collapse is a first-class
  failure even if total score is near target;
- adversarial review status.

For A++ evidence, the run must also satisfy contest-equivalent hardware and
inflate budget proof. A total-score improvement without component custody is not
promotion evidence.

## Lane Migration Matrix

### Core Helpers

`src/tac/losses.py::kl_distill_scorer_loss`

- Status: dead historical primary KL.
- Required action: keep only for forensic reproduction behind explicit banned
  waiver and `promotion_eligible=False`.
- Evidence requirement: exact CUDA archive only if a forensic archive is being
  preserved. It cannot support promotion.

`src/tac/losses.py::kl_distill_segnet_only`

- Status: canonical future-safe SegNet auxiliary helper.
- Required action: keep per-pixel reduction, T-scaling tests, no teacher
  gradients, explicit weight/temp provenance.
- Evidence requirement: exact CUDA archive plus matched ablation before any
  positive claim.

`src/tac/losses.py::segnet_kl_divergence_loss`

- Status: legacy KL-like helper, not yet aligned with the newer safety contract.
- Required action: freeze for promotion until a guard, scale review, and tests
  are added. Prefer migrating callers to `kl_distill_segnet_only` with explicit
  `segnet_aux` metadata.

`src/tac/losses_jbl.py`

- Status: non-KL soft-label distillation family.
- Required action: remove or supersede absolute safety language. Treat JBL/JML
  as KL-like for provenance and component-gate purposes.
- Evidence requirement: exact CUDA archive and component gates.

`src/tac/archive/scorer_distill.py`

- Status: scorer-head distillation research utility.
- Required action: never treat distilled heads as score truth. If heads affect
  inflate/runtime, they must be archive-contained and evaluated by canonical
  upstream scorers.
- Evidence requirement: exact archive custody and payload closure.

### Training Entry Points

`src/tac/training.py`

- Current status: primary `kl_distill` is guarded; `segnet_kl` remains an open
  hardening gap.
- Migration: add fail-closed validation for `segnet_kl`, record KL provenance
  for all distill-like modes, and reject promotion if the helper is legacy.

`src/tac/segmap_renderer.py`

- Current status: strongest KL guard. It enforces `eval_roundtrip=True`,
  rejects MPS, and allows only `segnet_aux`.
- Migration: record teacher roundtrip contract in artifacts; require matched
  plain SegMap ablation for attribution.

`experiments/train_segmap.py`

- Current status: uses corrected 0.002/T=2.0 defaults and explicit
  `kl_distill_scope="segnet_aux"` for `--variant kl_distill`.
- Migration: require remote lane provenance schema and exact CUDA component
  gates before any result claim.

`src/tac/experiments/train_renderer.py`

- Current status: supports auxiliary KL during Fridrich/JBL-style phases and
  roundtrips scorer inputs when configured.
- Migration: purge stale 1.0-as-proven comments, require scale-review tags for
  high weights, and treat JBL as distill-family evidence.

`experiments/optimize_poses.py`

- Current status: KL is default-off, gated by effective positive weight and
  available GT frames. Static 0.002 and SNR-controller paths exist.
- Migration: record raw-vs-roundtripped teacher semantics, controller
  trajectory, final effective weight, and component gates. Old Lane G remains
  forensic-only.

`experiments/train_distill.py` and `experiments/quantize_distilled.py`

- Current status: renderer distillation utilities with standard/focal/PCGrad
  modes, not the banned primary KL path.
- Migration: keep eval-roundtrip and archive-eval requirements. Any teacher
  labels or distilled artifacts that affect runtime must be in archive custody.

### Profiles

`SEGNET_KL_SMOKE` and `SEGNET_KL_FULL`

- Status: legacy, insufficiently fenced.
- Migration: mark forensic or migrate to explicit `segnet_aux` KL helper with
  corrected scale and metadata. Do not dispatch as full promotion lanes.

WILDE, SHIRAZ, DEN, DILATED_H64_HALF_FRAME, SELF_COMPRESS_RENDERER_FULL

- Status: contain stale 1.0-weight KL settings.
- Migration: freeze old claims; retune under corrected per-pixel KL using
  0.002-centered sweeps or SNR ratio control. Exact CUDA archive evidence is
  required before any renewed claim.

DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL,
QUANTIZR_REPLICA_88K_HALFFRAME, LANE_Q_FAITHFUL, DSCONV_QUANTIZR_KILLER,
DILATED_H64_GHOST

- Status: use corrected 0.002-style KL settings.
- Migration: still non-promotable until exact archive eval, component gates, and
  matched ablations exist. Correct weight is not correctness evidence.

LANE_19_LOGIT_MARGIN

- Status: KL disabled by design.
- Migration: if KL is re-enabled, treat it as a new stacked method requiring
  full provenance and exact archive eval.

J_JBL and related JBL profiles

- Status: non-KL distillation variants.
- Migration: use distill-family provenance, do not claim PoseNet safety without
  exact component evidence.

### Remote Lane Scripts

SegMap KL lanes:

- Scripts include SC++, DARTS-S SegMap, homography SegMap, Film Canvas,
  Fridrich block FP, Hessian block FP, weighted-curator, and pose-as-affine
  variants.
- Current status: generally use `experiments/train_segmap.py --variant
  kl_distill --kl-distill-weight 0.002 --kl-distill-temperature 2.0 --device
  cuda` and canonical auth eval.
- Migration: require structured provenance, exact CUDA JSON, archive custody,
  and plain/ablation comparisons. Do not attribute stacked wins solely to KL.

Lane G KL TTO:

- `remote_lane_g_kldistill_pose_tto.sh`: deprecated/forensic.
- `remote_lane_g_v3_corrected_kl_weight.sh`: corrected static-weight candidate.
- `remote_lane_g_v3_v2_lagrangian_snr.sh`: adaptive SNR-controller candidate.
- Migration: old script remains historical. Corrected and SNR variants must be
  separate methods with exact archive evidence, component gates, controller logs
  if adaptive, and no proof language from the word "Lagrangian".

Per-class SegNet pose lanes:

- Scripts include `remote_lane_ps_per_class_segnet.sh` and
  `remote_lane_ps_v2_learnable_class_weights.sh`.
- Current status: use high 1.0 KL weight plus class weights.
- Migration: forensic/non-promotable until retuned. Class weights require
  calibration split, holdout stability, finite nonnegative values, exact CUDA
  archive eval, and ablations against unweighted KL.

Renderer KL lanes:

- Scripts include D-V3, V/Quantizr replica, V-V2 annealed, Q faithful, H-V3,
  mixed precision FP4/FP8, Omega Hessian QAT, and related renderer profiles.
- Migration: read profile-level KL weight before dispatch. Any inherited 1.0
  weight blocks promotion unless a scale-review tag and exact evidence exist.
  0.002-weight lanes still need exact archive eval and component gates.

JBL/JML lane:

- `remote_lane_j_jbl_jaccard_bls.sh`.
- Migration: classify as `distill_family="jbl_jml"`, not KL. It still requires
  exact CUDA archive eval, component gates, and no categorical PoseNet-safety
  claims.

Pose embedding distillation lane:

- `remote_lane_m_v3_pose_from_embedding.sh`.
- Migration: keep separate from KL. The distilled pose model must be
  archive-contained if used at inflate, and success is only exact CUDA upstream
  PoseNet/SegNet evidence.

## Adjudication Rules

Harvest/adjudication must reject or downgrade any KL-like result when:

- `contest_auth_eval.json` is missing;
- device is not CUDA for a promotable claim;
- archive SHA-256 or bytes are missing;
- sample count is incomplete;
- component distances are missing or parsed from human logs while structured
  JSON exists;
- recomputed score disagrees with recorded score;
- PoseNet component collapses relative to anchor, even if total score looks
  acceptable;
- payload closure fails;
- sidecars outside archive affect runtime;
- exact eval was not `archive.zip -> inflate.sh -> upstream/evaluate.py`;
- provenance omits KL weight, temperature, scope, teacher semantics, or
  controller logs;
- a broad kill/ranking/promotion conclusion rests on one bad diagnostic run.

Negative KL results should be scoped as measured implementation/config
retirements unless independent exact reproductions or a mathematical
impossibility proof support broader language.

## Paper And Reporting Language

Allowed:

- "The measured implementation of primary scorer KL is retired for promotion
  use under the preserved evidence."
- "SegNet-only auxiliary KL is a future-safe hypothesis, not a proven win."
- "Corrected per-pixel KL removes the historical batchmean scale bug."
- "This exact archive improved/regressed under CUDA auth eval with the following
  components..."

Forbidden without exact evidence:

- "KL is killed" as a family.
- "KL is safe for PoseNet."
- "JBL cannot induce PoseNet collapse."
- "The Lagrangian controller proves optimal allocation."
- "The 1.0 weight recipe is proven."
- "MPS/CPU/proxy result ranks this lane."

## Immediate Action List

1. Add a fail-closed guard and tests for `loss_mode="segnet_kl"` promotion use.
2. Add remote-script provenance schema checks for every KL/JBL/distill-family
   flag.
3. Add high-weight KL profile/script preflight requiring scale-review metadata.
4. Record teacher roundtrip semantics for every `kl_distill_segnet_only` call
   site.
5. Supersede stale 1.0-weight profile comments and docs with corrected
   post-batchmean guidance.
6. Require matched non-KL ablations before attributing any exact-score movement
   to KL.
7. Keep old Lane G and primary KL artifacts as forensic records only.

## Bottom Line

The current repo has correctly moved away from primary scorer KL and fixed the
spatial `batchmean` failure mode in the canonical SegNet auxiliary helper.
However, contest-grade hardening is not complete until every remaining KL-like
entry point, profile, lane script, and report path is either fenced as forensic
or migrated to an explicit `segnet_aux`/distill-family contract with structured
provenance, invariant tests, exact CUDA archive eval, and component-gated
adjudication.
