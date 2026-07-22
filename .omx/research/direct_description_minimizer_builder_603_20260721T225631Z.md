---
title: Task 603 direct-description minimizer builder landing
utc: 2026-07-21T22:56:31Z
task: 603
master_task: 578
route: R3_DIRECT_DESCRIPTION_REDIRECT
lane_id: lane_direct_description_minimizer_builder_603_20260721
status: STATIC_SCAFFOLD_LANDED_LAUNCH_READINESS_FALSIFIED
research_only: true
execution_allowed: false
verdict_scope: Task 603 static builder and settled legacy S4 control container only
main_landing_review_required: true
---

# Direct-description minimizer builder

## Outcome

The landing is a fail-closed static scaffold, not a minimizer and not launch readiness. Three
independent reviews rejected the initial launch-ready interpretation because it had no optimizer,
no operational continuation loop, no Pose6-consuming receiver, and no valid final-ZIP unique-home
rate attribution. The implementation and receipts now state those blockers directly.

No training/evaluation/optimizer worker, scorer, provider, GPU, paid dispatch, optimization, or
candidate archive was started. The verification suite executed only local fail-closed CLI processes.
Pointer `0.1910828242 [contest-CPU]` is unchanged.

## What is implemented

- A dedicated `DirectDescriptionTypedConfigV1` and `DirectDescriptionWitnessProgramV1` compile the
  actual fail-closed DDM preflight command. `TypedLever`/`LawRef` provide seed custody; the code no
  longer routes through an unrelated level-set trainer config.
- The checked-in owner manifest is exact reproducible compiler output. Its command is preflight-only,
  seals `--execution-allowed false`, and exits 2 while any readiness predicate is red.
- Exact RFC 8785 canonicalization, full-precision SHA-bound cap-receipt validation, immutable atomic
  stage checkpoint save/load, strict same-artifact ladder receipt validation, structural recursive
  allocation checks, and externally attested failure-receipt verification are callable and tested.
- Completion checking computes the maximum absolute residual per governed coordinate, so different
  coordinates cannot cancel. It binds terminal evidence and a separate independent-audit receipt to
  caller-supplied hashes and rejects an empty/all-gate-excluded search.
- Path and raw-byte archive inputs are checked against quarantine by content digest. A rename or a
  raw-bytes call cannot bypass an archive-SHA quarantine entry.
- The per-stratum tool now uses the full-precision 154,524-byte planning value and accepts a typed
  receiver-rate receipt input. Its v1 consumer permanently refuses monolithic-Deflate attribution.

## Exact measured control fact

The settled S4 archive was opened read-only and canonically parse/re-encoded:

- archive: 451,191 bytes, SHA-256
  `d84f2fe053239d1542ba381420e9569d431ed2015e22e60e49ef48f1321696ed`;
- member: 1,285,943 bytes, SHA-256
  `595e69d41f96cc1a33ca7b58c0ed386549bfda6389a8176b24d7044d1f55955b`;
- exact archive equality after parse/re-encode: true.

This proves only a legacy control-container re-expression. The six S4 sections are recorded as
`LEGACY_OPAQUE_SECTION_REEXPRESSION`; they are not evidence that the PRIMARY entropy, xi, Pose6,
event, ground, and exception semantics are live. In particular, `causal.pcr3` is empty and the S4
receiver rejects a nonempty causal/Pose stream. Receiver consumption is therefore false.

The 216,207-byte `base.pbase3 + components.pcomp3` value is a legacy payload subtotal, not
`len(A(z))`, not PRIMARY description semantics, and not receiver-closed rate. Null-space dimension
does not imply byte savings, so null savings and projected candidate archive length remain null.

## Planning arithmetic only

The displayed ceil-minus-one results are 216,223 bytes at the pointer ceiling and 154,524 bytes at
the 0.15 ceiling. They are excluded from launch configuration because the required full-precision,
SHA-bound solved-target receipt is absent. The stale displayed approximation 154,600 is removed from
the per-stratum tool.

## Exact blockers

1. The PRIMARY spec seals `execution_allowed:false`.
2. No Pose6-consuming integer/uint8 receiver exists.
3. No DDM optimizer or stage-continuation loop exists; checkpoint schema/restore audit is not a run.
4. No canonical resume-registry/cadence wiring, governed-launcher/memory-preflight adapter, or
   operational cleanup/cold-store hook exists.
5. The dedicated local typed config/program is not yet integrated into canonical
   `WitnessProgram`/`TypedWitnessConfig` compiler surfaces.
6. One Deflate-compressed member cannot provide stable per-semantic-stream final-ZIP byte homes; the
   v1 receiver-rate receipt is non-authorizing until a versioned independently framed carrier exists.
7. No fresh v3-family pose-in-objective rung zero or four-rung cells-then-pose measurement ladder
   exists.
8. No deterministic n64 custody smoke, fresh n600 same-artifact exact evaluator closure, separate
   contest-CPU/CUDA replay receipts, or separate operator-GO receipt exists.
9. No valid completion certificate or externally signed failure token was produced.

Fallback #366 is ineligible. All negatives are formulation- and landing-scoped; representation
families remain open.

## Verification

- Focused DDM/per-stratum/CLI suite: 37 passed.
- Adjacent quarantine, DSL-gate, storage, resume, and checkpoint apparatus: 131 passed, 1 skipped.
- Ruff and Python compilation: pass.
- Checked-in owner bundle equals fresh compiler output and its preflight refuses with exit 2.
- No bulk artifact was created; compiler scratch is context-managed and success-only.
- Reviewed code commit: `0e352c122a` on
  `codexwt/direct_description_minimizer_builder_20260721T221054Z`; MAIN merge is still required.

Exact pytest argv:

```text
/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/optimization/tests/test_direct_description_minimizer.py tools/tests/test_run_direct_description_minimizer.py tools/tests/test_measure_per_stratum_recursive_fractal_optimal.py
/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/tests/test_artifact_quarantine.py src/tac/tests/test_artifact_quarantine_runtime_gate.py src/tac/tests/test_dsl_compile_hash_enforcement.py src/tac/tests/test_dsl_config_gate_apparatus.py src/tac/tests/test_launch_dsl_config_gate.py src/tac/tests/test_operator_storage_waterfall.py src/tac/tests/test_resume_registry.py src/tac/tests/test_subagent_checkpoint_resume_lookup.py
```

## STORES CONSULTED

- Delegated authority, PRIMARY spec, CLAUDE.md, AGENTS.md, v7.5 operating contract, SPEC v8, and
  current lane/subagent state.
- Settled S4 archive/build surfaces, quarantine manifest, per-stratum receipt consumer, canonical
  task and pointer surfaces, and latest applicable Codex/Claude memos.
- Three independent delegated review reports covering core semantics, integration artifacts, and
  PRIMARY-spec compliance.

## MAIN landing review

MAIN must review this landing before merge. The review must confirm that the owner artifact is exact
compiler output, the CLI remains no-spawn and exit-2 fail-closed, the z0 aliases are never promoted
to PRIMARY semantics, receiver-rate v1 cannot authorize Deflate byte attribution, checkpoints are
not described as an operational runner, and no score/pointer/failure-token claim is inferred.
