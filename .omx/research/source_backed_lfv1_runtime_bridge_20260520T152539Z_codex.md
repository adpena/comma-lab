# Source-Backed LFV1 Runtime Bridge Control

**Author:** Codex  
**UTC:** 2026-05-20T15:25:39Z  
**Scope:** LA-Pose/Telescope-style LFV1 pose/foveation substrate, local no-spend bridge hardening  
**Score claim:** false  
**Dispatch attempted:** false  
**Ready for exact eval dispatch:** false

## Result

The LFV1 pose/foveation candidate has moved from archive custody plus structural digest mutation to a local scorer-visible RGB warp control:

- `lapose_foveation_tuples.lfv1` is still lowered deterministically into archive-contained `foveation_params.bin` (`HFV1`).
- `runtime_consumer.py` now includes a standalone HFV1 loader and a deterministic `apply_lfv1_to_rgb_frames` / `functional_hyperbolic_foveation` bridge.
- The local control proves identity HFV1 params are allclose-identity on a deterministic RGB tensor and LFV1-derived nonidentity params change output pixels.
- Exact dispatch remains fail-closed because this is not yet wired into a full contest inflate path that reconstructs and writes scored output frames/masks.

## Empirical Anchor

Fresh artifact:

- Archive: `experiments/results/theoretical_floor_lfv1_pose_foveation_bridge_20260520_codex/archive_candidate/archive.zip`
- Archive bytes: `84971`
- Archive SHA-256: `5e4827864b17ab98112b3b1ec52278342b51ad8e313b58bb6f0100bdebddc520`
- Readiness: `experiments/results/theoretical_floor_lfv1_pose_foveation_bridge_20260520_codex/archive_candidate/readiness.json`
- Readiness SHA-256: `bb3538fcfc1a7514726e7b239dca014b7f8bb023625e4fb2867405b796b8f879`
- LFV1 payload SHA-256: `be4d51599b715196c78cf5e2824290990e9fb1f56c835dfa4c7ea12365167e41`
- HFV1 foveation params SHA-256: `e4c10b98f8e686d9534451c21b5126bce2115073c0b68d8d2df0d1309e89a275`

Frame-warp control from readiness:

- Contract: `lapose_foveation_scorer_visible_frame_warp_control_v1`
- Passed: `true`
- Probe frame indices: `[238, 239, 248, 249]`
- HFV1 target frame count: `1184`
- Image size: `384x512`
- Identity max abs delta: `5.960464477539063e-08`
- Nonidentity max abs delta: `0.0037032365798950195`
- Nonidentity output changed: `true`

Remaining dispatch blockers:

- `runtime_loader_parity_not_passed`
- `lapose_foveation_scorer_visible_output_parity_not_proven`
- `lapose_foveation_runtime_output_parity_not_proven`
- `exact_cuda_auth_eval_missing`

## Code Changes

- `src/tac/lapose_foveation_runtime_skeleton.py`
  - Added HFV1 decoding, standalone frame warp, scorer-visible frame write helper, and local RGB warp control report.
  - SHA-256: `0d3b951703a996cad3b88f1799af98103ee80e790fb0eaf5bc821cca93d19322`
- `src/tac/lapose_foveation_payload_candidate.py`
  - Promoted the new frame-warp control into required local no-op/runtime controls.
  - SHA-256: `263a8b32dc150081074131379fe180e94237f6c8c98fec05e34d8d160cd3ed7b`
- `src/tac/tests/test_build_lapose_foveation_payload_archive.py`
  - Added direct RGB warp control coverage and updated candidate readiness assertions.
  - SHA-256: `ac36cd55f8b0e86cff821d43c1eed94dd5288ce450ae106476429fa6dc53191f`

## Verification

Focused test:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_lapose_foveation_payload_archive.py -q
```

Result: `5 passed in 2.18s`.

Static diff hygiene:

```bash
git diff --check -- src/tac/lapose_foveation_runtime_skeleton.py src/tac/lapose_foveation_payload_candidate.py src/tac/tests/test_build_lapose_foveation_payload_archive.py
```

Result: clean.

## Interpretation

This does not lower the public score and must not be promoted. It does close a real bridge gap: the foveation bytes are no longer only structurally decoded; they are consumed by a deterministic RGB transform whose output changes scorer-visible pixels under local controls.

The next frontier-moving artifact is a tiny full-runtime adapter that runs:

1. A decoded RGB fixture or source-frame tensor through `apply_lfv1_to_rgb_frames`.
2. A byte-output writer path that is compared byte-for-byte under identity params.
3. A nonidentity control that proves output bytes change only when HFV1 params change.
4. A fail-closed candidate archive that records this as runtime output parity evidence without claiming exact CUDA score.

Only after that adapter exists should this lane attempt a real contest-runtime integration and claimed exact auth eval.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:source-backed-LFV1-runtime-bridge-control-codex-memo-trigger-tokens-in-bridge-control-content-not-new-empirical-equation -->
