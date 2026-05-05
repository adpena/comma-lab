---
name: Lane 19 (SegNet logit-margin boundary loss) — Level 1 → Level 2 hardened, contest-CUDA pending
description: 2026-04-30. Lane 19 graduates from scaffold (Level 1, 8 tests) to fully integrated, 6-round-adversarial-reviewed, STRICT-preflight-guarded production loss with the CRITICAL detach() fix that flipped the gradient direction on boundary-wrong pixels. 18 module tests + 11 preflight tests = 29 total. Profile LANE_19_LOGIT_MARGIN registered (loss_mode='logit_margin', logit_margin_weight=10.0, logit_margin_threshold=1.0, kl_distill_weight=0.0, seed=89). Remote dispatch script ready (~$1.50, ~5h). Contest-CUDA dispatch pending operator approval.
type: project
originSessionId: lane-19-impl-20260430
---

## TL;DR

Lane 19 = SegNet logit-margin boundary loss = Fridrich UNIWARD applied to segmentation. Per-pixel CE weight `(threshold - margin) / threshold` so confident pixels (margin >> threshold) contribute zero loss; boundary pixels (margin < threshold) contribute proportional to their ambiguity. The detach() fix from Round 3 ensures the gradient direction is correct on boundary-wrong pixels.

## Status

- **Maturity:** Level 2 (INTEGRATION) per `feedback_production_hardened_standard_definition_20260430.md` schema.
- **Gates achieved (5 of 7):**
  - ✓ impl_complete: `src/tac/losses_logit_margin.py` (+ teacher helper + segnet aux)
  - ✗ real_archive_empirical: pending dispatch
  - ✗ contest_cuda: pending dispatch
  - ✓ strict_preflight: Check 93 (`check_logit_margin_loss_uses_boundary_mask`) STRICT @ 0 violations on real codebase
  - ✓ three_clean_review: Round 4/5/6 all clean (3/3 after Round 3 CRITICAL caught + fixed)
  - ✓ memory_entry: this file
  - ✓ deploy_runbook: `scripts/remote_lane_19_logit_margin.sh` with NVDEC probe + heartbeat + provenance + canonical 4 stages

## What landed

### Module (`src/tac/losses_logit_margin.py`, 340 LOC)
- `fragility_weights(logits, threshold)` — per-pixel `(threshold - margin) / threshold` clamped to [0, 1]
- `logit_margin_loss(logits, gt_argmax, threshold, reduction)` — CE × weights.detach() (CRITICAL: detach added Round 3)
- `logit_margin_loss_with_teacher(student_logits, teacher_logits, threshold, reduction)` — derives gt from teacher.argmax under no_grad
- `compute_segnet_logit_margin_aux(rendered_pair, gt_pair, segnet, threshold, reduction)` — train_renderer.py call site signature

### Tests (`src/tac/tests/test_losses_logit_margin.py`, 18 tests)
- 8 original scaffold tests (extremes, spatial, zero-on-confident, positive-on-ambiguous, gradient direction/magnitude, no-silent-default, shape mismatch)
- 10 new tests including `test_fragility_weights_detached_in_loss_synthetic` (the Round 3 CRITICAL regression pin), teacher helper tests (3), segnet aux tests (2), determinism, A/B, scorer-loss-handles-confident-wrong, profile-resolver-pipe.

### Train wiring (`src/tac/experiments/train_renderer.py`)
- `--logit-margin-weight` and `--logit-margin-threshold` argparse args (FORBIDDEN PATTERNS verified — added new flags, not invented existing ones).
- `_VALID_LOSS_MODES` widened to include `"logit_margin"`.
- Resolver block adds `args.logit_margin_weight = _resolve(..., "logit_margin_weight", 0.0)` + threshold default 1.0.
- Auxiliary loss block fires alongside KL distill (when weight > 0): `loss = loss + weight × compute_segnet_logit_margin_aux(...)`. Tagged `KL_RAW_PAIRS_OK` (rendered_pair was reassigned to roundtripped output at L1642).

### Profile (`src/tac/profiles.py`)
- `LANE_19_LOGIT_MARGIN` inherits `DILATED_H64_HALF_FRAME_V3_ANNEALED_KLDISTILL` (Lane G v3 1.05 anchor).
- Overrides: `loss_mode="logit_margin"`, `logit_margin_weight=10.0` (Round 1 Contrarian fix), `logit_margin_threshold=1.0`, `kl_distill_weight=0.0` (Round 1 Quantizr fix), `seed=89`.
- Registered in `PROFILES` dict as `"lane_19_logit_margin"`.

### Preflight (`src/tac/preflight.py`)
- New Check 93: `check_logit_margin_loss_uses_boundary_mask`. AST-scans all `src/tac/`, `experiments/`, `scripts/` for calls to `logit_margin_loss` / `logit_margin_loss_with_teacher` / `compute_segnet_logit_margin_aux` / `fragility_weights` and asserts each carries explicit `threshold=` kwarg. STRICT @ 0 violations on real codebase. Wired into `preflight_all()` after Check 92.
- Tests (`src/tac/tests/test_preflight_logit_margin_threshold.py`, 11 tests): real-codebase clean, synthetic-violation flagged, attribute-form recognized, syntax-error tolerated, etc.

### Remote dispatch (`scripts/remote_lane_19_logit_margin.sh`, 13.2K, exec)
- Stage 0: NVDEC probe + Check 93 STRICT preflight + dead-flag scan.
- Stage 1: training (~5h on Vast.ai 4090).
- Stage 1b: FP4A renderer.bin export.
- Stage 2: half-frame archive build.
- Stage 3: pose TTO 500 steps.
- Stage 4: contest_auth_eval on the EXACT submitted archive bytes.
- Provenance JSON + heartbeat + RESULT_JSON guard.

## Council audit log (3-clean-pass gate complete)

`.omx/research/council_lane_19_logit_margin_design_20260430.md` — Phase A design memo.
`.omx/research/council_lane_19_logit_margin_round{1,2,3,4,5,6}_20260430.md` — 6 rounds.

| Round | New issues | Fix |
|---|---|---|
| 1 | Quantizr KL-drowned, Contrarian weight-too-small, Fridrich threshold guidance | profile rewrite |
| 2 | Karpathy resolver-pipe regression test missing | added test |
| 3 | **CRITICAL** Filler/Fridrich/Hinton: weights must be detached (gradient direction reversed on boundary-wrong) | one-line fix + regression test |
| 4 | 0 | counter 1/3 |
| 5 | 0 | counter 2/3 |
| 6 | 0 | counter 3/3 ✓ |

## Round 3 CRITICAL math (the most important finding)

Without `weights.detach()`:
- `∂L_lane19/∂z[top1] = 10 × (∂w/∂z[top1] × CE + w × ∂CE/∂z[top1])`
- `∂w/∂z[top1] = -1/threshold` for boundary pixels.
- For boundary-WRONG pixel (margin < threshold, top1 ≠ GT, CE > 0):
  - First term: 10 × (-1) × CE = -10 × CE → optimizer INCREASES z[top1] (BAD)
  - Second term: 10 × w × p[top1] > 0 → optimizer DECREASES z[top1] (good)
  - Net direction depends on magnitudes; with w=0.5, CE=1, p[top1]=0.4: **net = -10 + 2 = -8 → wrong direction**.

With `weights.detach()`:
- `∂L_lane19/∂z[top1] = 10 × w_detached × ∂CE/∂z[top1]` (only second term).
- Direction: same as standard CE.
- Magnitude: scaled by w (boundary pixels amplified, confident zeroed).

The Round 3 fix is the same convention as Hinton focal loss (`(1 - p_t)^γ` is detached) and Hinton KL distillation (teacher logits detached). Lane 19 now follows the standard importance-weighted-loss recipe.

## Predicted band [prediction]

`[0.75, 1.05]` `[contest-CUDA]` standalone (vs Lane G v3 1.05 anchor):
- Floor 0.75: -3e-3 SegNet distortion → -0.30 score
- Mid 0.95: -1e-3 SegNet distortion → -0.10 score
- Ceiling 1.05: margin loss buys nothing over standard CE → KILL → demote to Phase 3

## Next step

Phase F: Vast.ai 4090 dispatch via Pattern A nohup detach. Cost: ~$1.50 ($1.25 train + $0.25 auth eval). After RESULT_JSON lands, mark gates 2 + 3 (real_archive_empirical + contest_cuda) → Lane 19 promotes Level 2 → Level 3 if score < 1.05; otherwise demote to Phase 3 with the empirical kill.

## Cross-refs

- `.omx/research/council_lane_19_logit_margin_design_20260430.md` (Phase A)
- `.omx/research/council_lane_19_logit_margin_round{1..6}_20260430.md`
- `feedback_production_hardened_standard_definition_20260430.md` (Level definitions)
- `project_phases_2_3_4_design_implementation_math_provenance_20260429.md` §"Lane 19"
- `feedback_silent_default_bug_class_findings_20260429.md` (Check 93 motivates)
- CLAUDE.md "FORBIDDEN PATTERNS" (CLI flag inventions, default-to-convenience)
- CLAUDE.md "Exact scorer architectures" (SegNet EfficientNet-B2 stride-2 stem; argmax disagreement)
- CLAUDE.md "SegNet paradigm shift" (SegNet 77x more important than PoseNet at our operating point)
- `reports/lane_19_logit_margin_local_smoke.json` (Phase E [smoke-only])
