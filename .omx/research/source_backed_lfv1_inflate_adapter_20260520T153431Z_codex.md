# Source-Backed LFV1 Local Inflate Adapter

**Author:** Codex  
**UTC:** 2026-05-20T15:34:31Z  
**Scope:** LFV1/HFV1 pose-foveation substrate, local RGB24 adapter bridge  
**Score claim:** false  
**Dispatch attempted:** false  
**Ready for exact eval dispatch:** false

## Result

Built the next bridge component after local RGB warp: a packaged local RGB24 inflate adapter plus a guarded official-signature facade.

`runtime_consumer.py` can now run in three modes:

1. Default archive verification mode: verifies charged LFV1/HFV1/proof members and exits fail-closed with return code `2`.
2. Local adapter mode: accepts `--input-rgb24`, `--output-rgb24`, `--frame-count`, `--height`, `--width`, and optional `--frame-indices`; applies archive-contained HFV1 geometry; writes deterministic NHWC RGB24 output bytes; exits `0`.
3. Official-signature facade mode: accepts the challenge shape through `inflate.sh <archive_dir> <output_dir> <file_list>` only when `LFV1_BASE_RAW_DIR` is explicitly supplied; streams base `.raw` RGB24 files in fixed frame chunks; applies archive-contained HFV1 geometry; writes `<output_dir>/<base>.raw`.

The contest `inflate.sh` still exits fail-closed under judge-shaped calls unless `LFV1_BASE_RAW_DIR` is provided, so this is not a self-sufficient submission decoder and cannot be promoted as an exact-eval result.

## Empirical Anchor

Fresh artifact:

- Archive: `experiments/results/theoretical_floor_lfv1_pose_foveation_adapter_20260520_codex/archive_candidate/archive.zip`
- Archive bytes: `131418`
- Archive SHA-256: `f3877dc9f324292aab82b4614106332348e614d801781281ff4db10acaf1c0f2`
- Readiness: `experiments/results/theoretical_floor_lfv1_pose_foveation_adapter_20260520_codex/archive_candidate/readiness.json`
- Readiness SHA-256: `685440106c666cc6d49c63904a516b4a02b204d850ad543976e525c2163ab64c`
- Runtime member SHA-256: `9890a0a866802b9127e2840dc0c13a2605603e7e2de7b1a012f30d3ec5cf455c`
- LFV1 payload SHA-256: `be4d51599b715196c78cf5e2824290990e9fb1f56c835dfa4c7ea12365167e41`
- HFV1 params SHA-256: `e4c10b98f8e686d9534451c21b5126bce2115073c0b68d8d2df0d1309e89a275`

Readiness controls now pass:

- `scorer_visible_frame_warp_control`
- `scorer_visible_byte_output_control`
- `inflate_adapter_byte_output_control`

Adapter control details:

- Format: `nchw_float_rgb_to_nhwc_uint8_rgb24`
- Probe frame indices: `[238, 239, 248, 249]`
- Byte count: `2359296`
- Identity adapter bytes exact: `true`
- Nonidentity adapter output changed: `true`
- Nonidentity adapter output SHA-256: `3c32ddbf24611fba0de4253af31aec2a0128e5a35bf09a8d8b14d02e71b288b5`

Packaged runtime adapter smoke:

- Summary: `experiments/results/theoretical_floor_lfv1_pose_foveation_adapter_20260520_codex/adapter_smoke/adapter_smoke_summary.json`
- Summary SHA-256: `53507069c8c1a81d4d6693015968ab52344a70e25fefd91e2bd40b96f75d2e69`
- Input bytes: `589824`
- Input SHA-256: `5c05497591bd3ce58ea3a558f24d0efe01d1916cec66421074c623e494d9e270`
- Output bytes: `589824`
- Output SHA-256: `362daecb595b34211a44aaa11f4d81bd7ef778fc036d8f088c3432c4347cf1c9`
- Output changed: `true`
- Runtime return code: `0`

Official-signature facade and hardware/domain controls:

- `inflate.sh` recognizes the official 3-argument facade only with explicit `LFV1_BASE_RAW_DIR`.
- Facade streams base `.raw` files in `LFV1_CHUNK_FRAMES` chunks; default `16` frames.
- Test fixture verifies `LFV1_CHUNK_FRAMES=1` to force chunked behavior.
- Raw format remains flat uint8 RGB NHWC with no header, matching the upstream evaluator contract.
- Runtime records `external_teacher_models_at_inflate=false`, `posenet_segnet_sensitivity_collapsed=false`, and keeps LA-Pose/RAFT/etc. as compress-time priors only.
- Pair/frame routing records frame parity and pair index so SegNet per-frame sensitivity and PoseNet per-pair sensitivity are not collapsed.
- Optimization metadata explicitly allows compress-time PCGrad/gradient projection, stagewise freeze/unfreeze, pose-sensitive pair waterfill, segmentation-sensitive per-frame weighting, and master-gradient atom selection, while forbidding inflate-time scorer/training dependencies.

Remaining dispatch blockers:

- `runtime_loader_parity_not_passed`
- `lapose_foveation_scorer_visible_output_parity_not_proven`
- `lapose_foveation_runtime_output_parity_not_proven`
- `exact_cuda_auth_eval_missing`

## Code Changes

- `src/tac/lapose_foveation_runtime_skeleton.py`
  - Added RGB24 byte conversion, adapter core, local CLI adapter mode, official-signature facade mode, chunked streaming, byte-output control, adapter-output control, sensitivity routing metadata, and hardware/research-fidelity contracts.
  - SHA-256: `9890a0a866802b9127e2840dc0c13a2605603e7e2de7b1a012f30d3ec5cf455c`
- `src/tac/lapose_foveation_payload_candidate.py`
  - Promoted byte-output and adapter-output controls into required local readiness controls.
  - Hardened runtime-loader parity so `foveation_params.bin` must be loaded as a charged member.
  - Added guarded official-signature facade activation through `LFV1_BASE_RAW_DIR`/`LFV1_CHUNK_FRAMES`.
  - SHA-256: `9bd384fa04a74ae49a2762fa824cdc736e0c3351fb3b118e25a6ce413165a9f2`
- `src/tac/tests/test_build_lapose_foveation_payload_archive.py`
  - Added packaged runtime subprocess coverage for local RGB24 adapter mode and official-signature facade mode.
  - SHA-256: `ea6939300c25cd9e9be05f3477ce763bbb968e380c715aa39c46c66dab37f355`

## Verification

Focused test:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_lapose_foveation_payload_archive.py -q
```

Result: `7 passed in 3.84s`.

Parallel read-only review confirmed the local adapter surface and identified the `foveation_params.bin` charged-member requirement above. The patch is included in this landing.

The follow-on aggregate verification is recorded in the terminal session. This artifact still intentionally fails exact-eval readiness until a full contest inflate path reconstructs and writes scorer-visible frames/masks from contest inputs.

## Next Necessary Component

Build the PR101-integrated contest-facing adapter:

1. Fork the PR101/FEC6 runtime locally, not the live PR submission.
2. Insert `apply_lfv1_adapter_to_full_rgb24_bytes` after PR101 selector application and before raw write, preserving 0..255 uint8 semantics deliberately.
3. Run identity HFV1 against PR101 output and prove byte-for-byte equality.
4. Run nonidentity HFV1 and prove scorer-visible raw bytes change with pair/frame sensitivity routing recorded.
5. Measure local CPU runtime overhead with chunk size `16`, then tune for Modal/Linux CPU and T4 memory behavior.
6. Keep `ready_for_exact_eval_dispatch=false` until exact CUDA auth eval is available for the full candidate archive.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:source-backed-LFV1-inflate-adapter-codex-memo-trigger-tokens-in-adapter-content-not-new-empirical-equation -->
