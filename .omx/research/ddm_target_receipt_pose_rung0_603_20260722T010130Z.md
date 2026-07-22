---
title: Task 603 SHA-bound target receipt and real-target Pose rung zero
utc: 2026-07-22T01:34:04Z
task: 603
master_task: 578
lane_id: lane_ddm_target_receipt_pose_rung0_603_20260722
status: TWO_SCOPED_BLOCKERS_FLIPPED_PRIMARY_STILL_BLOCKED
research_only: true
execution_allowed: false
evidence_axis: "[macOS-CPU real-target subset n64 apparatus]"
verdict_scope: exact existing C1 target custody and bounded n64 real-target apparatus only
main_landing_review_required: true
---

# DDM target receipt and Pose-active rung zero

## Outcome first

The next two fixed Task #603 blockers are now `GREEN_CUSTODY_SCOPE`. The DDM target is bound to the
existing complete n600 C1 solved-pair scorer-plane bytes, GT Pose6 source, archive, official
contest-CPU receipt/provenance, producer source, and upstream snapshot. A fresh deterministic n64
subset rung then used the real custodied planes and real Pose6-derived target codes in a typed joint
integer apparatus objective, saved every stage, and reproduced the terminal result byte-for-byte
after disk resume.

Here, “full precision target planes” means the exact full scorer-resolution uint8 solved-plane bytes
already emitted by C1, not a float surrogate and not a new solve. The rung's 8x8 projection and
ordinal Pose6 codes are deterministic apparatus transforms whose recipes and hashes are explicit.

No scorer was loaded. No remote, provider, GPU, paid dispatch, candidate archive, d_seg, d_pose, or
score measurement occurred. PRIMARY `execution_allowed=false` remains sealed. Pointer
`0.1910828242 [contest-CPU]` is unchanged.

## Custody receipt

Target receipt: `.omx/research/ddm_full_precision_target_planes_603_20260722T010130Z.json`, SHA-256
`a8d94f0f8338036fb3224a92078eff1f1fb5fd2eb598ed994a1f965b6561efb2`.

- Complete source: 600 pairs in 50 canonical 12-pair chunks at 384x512 RGB uint8 per plane.
- Aggregate Y0 SHA-256: `5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566`.
- Aggregate Y1 SHA-256: `6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc`.
- Source cache: 5,078,017,610 bytes, SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`; `gt_poses` is finite
  float64 `[600,6]`, content SHA-256
  `bee5821eeb892ad430878946dfcdf365e1a14202efb7cd1af2b019d75da0f481`.
- Exact source archive: 409,526,925 bytes, SHA-256
  `e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42`.
- Original C1 producer: `tools/measure_v10_two_plane_receiver_timing.py` at git
  `c78340236328d2d4f5f2649695cda45a91639799`, source SHA-256
  `2950ba48bbba0c6be5db57158a651dcdc2403bf8498605c39a8bdd2d0b595d01`.
- Upstream snapshot SHA-256:
  `d46d89155dbf0848e357858c8f62e12ef450a2914ef65814a4359ef6768d2d41`; upstream evaluator
  SHA-256 `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b`.
- n64 apparatus projection recipe: exact integer 48x64 block mean to 8x8 with half-up rounding;
  projection SHA-256 `84c743ebd35a8459adf41ac5e56e124c8fed078eebb782f9e16bbb2fa23cdba5`.
- Pose6 apparatus recipe: per-coordinate n600 ordinal uint8 with pair-ID tie break; n64 code SHA-256
  `d441e093384617a96c9bec0e6dda7b6afd1a5f433d308f8e59788455329d1406`.

Bulky source bytes remained read-only on `/Volumes/VertigoDataTier/pact`; no duplicate plane tree or
cache was created locally.

## Measured apparatus trajectory

Rung receipt:
`.omx/research/ddm_target_receipt_pose_rung0_603_20260722T010130Z_artifacts/ddm_real_target_pose_rung0_receipt.json`,
SHA-256 `c14fe757bacb727392d1bb5a591c786bc08ecc06c7fe3e344e6d036033a245e0`.

The fixed seed-1234 n64 tuple `(plane integer L1 debt, Pose6 integer L1 debt, joint debt, exact archive
bytes)` moved as follows:

| Candidate-search stage | Before | After |
|---|---:|---:|
| `real_cells_rung0` | `(670993, 32382, 703375, 1585)` | `(670981, 32382, 703363, 1585)` |
| `real_pose6_rung0` | `(670981, 32382, 703363, 1585)` | `(670977, 32370, 703347, 1585)` |
| `real_xi_joint_rung0` | `(670977, 32370, 703347, 1585)` | `(670953, 32370, 703323, 1585)` |

All three stages strictly descended their registered lexicographic objective. The Pose stage consumed
the counted `pose6_dxi_residuals` stream, reduced Pose6 apparatus debt by 12, and did not increase
plane debt. These integers are neither official distortions nor proxies authorized to rank or
promote a candidate.

Three immutable primary checkpoints and three disk-resume checkpoints are preserved. Corresponding
primary/resume bytes are identical, with stage SHA-256s `c40be765...`, `b901b1ec...`, and
`bf5a0c89...`. The resumed terminal archive, receiver output, and objective are bit-identical. The
1,585-byte terminal archive is explicitly named `.not_a_candidate.zip.receipt-bytes`, SHA-256
`3f9af1b19c61809599a1ea7239ebe4139c2900d7c28c0e6dcd76d98edc5b2dfd`.

## Blocker delta

The append-only register
`.omx/research/ddm_target_receipt_pose_rung0_603_blocker_register_20260722T010130Z.json` carries the
fixed 19-row order. Green count moved `4 -> 6`; red count moved `15 -> 13`:

1. `FULL_PRECISION_SHA_BOUND_TARGET_RECEIPT`: `RED -> GREEN_CUSTODY_SCOPE`.
2. `FRESH_V3_FAMILY_POSE_IN_OBJECTIVE_RUNG_ZERO`: `RED -> GREEN_CUSTODY_SCOPE`.

The four predecessor greens remain scoped greens. All other 13 blockers remain red, including
PRIMARY execution, canonical resume-registry/compiler/launcher integration, the four-rung ladder,
n600 same-artifact closure, new contest CPU/CUDA replay, completion/attestation, and operator GO.

## Adversarial review

Round 1 found and fixed two strict-JSON boundary defects: canonical arrays were rejected as tuple
fields after object-mode parsing in both target-receipt and checkpoint reload. Regression coverage now
round-trips the typed receipt and a real stage checkpoint. A final provenance pass added exact
committed producer module/CLI source hashes and git SHAs to the rung receipt. Re-materialization
confirmed unchanged archive and checkpoint bytes.

Disposition: `CLEAN_AFTER_FIX_FOR_TWO_SCOPED_BLOCKERS`, not a PRIMARY seal. See
`.omx/research/codex_findings_ddm_target_receipt_pose_rung0_603_20260722T010130Z_codex.md`.

## Exact re-derivation argv

```text
/usr/bin/env python3 tools/run_direct_description_real_target_rung0.py --config .omx/research/ddm_target_receipt_pose_rung0_603_20260722T010130Z.config.json --output-dir <new-empty-output-directory> --execution-allowed false
```

The checked-in receipt uses the same argv with output directory
`.omx/research/ddm_target_receipt_pose_rung0_603_20260722T010130Z_artifacts`. Immutable outputs refuse
overwrite by design, so MAIN should use a new empty directory for re-derivation and compare hashes.

Focused verification:

```text
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/optimization/tests/test_direct_description_real_target_rung0.py src/tac/optimization/tests/test_direct_description_minimizer.py tools/tests/test_run_direct_description_minimizer.py
35 passed
```

Ruff, Python compilation, canonical receipt validation, source-hash revalidation, checkpoint pair
comparison, archive parse/re-encode, and `git diff --check` passed.

## Triality and pointer delta

- DSL: `DirectDescriptionRealTargetRung0ConfigV1` and its only compiled local consumer argv.
- DAG: `.omx/research/ddm_target_receipt_pose_rung0_603_DAG_FEED_20260722T010130Z.md`.
- Equations: no new canonical law; apparatus-only transforms/objective are scoped in
  `.omx/research/ddm_target_receipt_pose_rung0_603_canonical_equations_20260722T010130Z.md`.
- Pointer delta: none; `0.1910828242 [contest-CPU]` unmoved.

## STORES CONSULTED

- Delegated authority; `CLAUDE.md`; `AGENTS.md`; project memory top; latest sister findings/session,
  council/design, canonical pointer/lane/task/subagent state, and both watched inboxes.
- `PROGRAM.md`, v7.5 §8, v8 spec, DDM PRIMARY spec, predecessor builder/register/receipt/DAG/equations,
  C1 prepare receipt and all 50 source chunks, official contest-CPU receipt/provenance, upstream
  snapshot, GT cache, exact archive, and [the operating manual](../../docs/operating_manual_craft_handoff.md).
- Fresh target/config/rung receipts, all six checkpoints, not-a-candidate archive, tests, and the
  base-to-branch diff.

## MAIN landing review

MAIN must review `aa6f7eae252b2fee262913c65d1ea6c6bd0c9ee9..codexwt/ddm_target_receipt_pose_rung0_20260722T010130Z`
before merge. Re-derive the target source hashes/geometry, integer projection, Pose6 ordinal recipe,
strict JSON reload fixes, producer source custody, objective admission, checkpoint continuation, and
primary/resume byte equality. Do not promote n64 integer debts or the `.not_a_candidate.zip.receipt-bytes` into
n600/scorer/score evidence.
