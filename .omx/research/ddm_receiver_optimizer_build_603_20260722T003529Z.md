---
title: Task 603 DDM receiver and optimizer build
utc: 2026-07-22T00:35:29Z
task: 603
master_task: 578
lane_id: lane_ddm_receiver_optimizer_build_603_20260722
status: REAL_CUSTODY_RECEIVER_AND_OPTIMIZER_LANDED_PRIMARY_STILL_BLOCKED
research_only: true
execution_allowed: false
evidence_axis: "[custody-smoke]"
verdict_scope: local deterministic n64 receiver, exact byte custody, and search continuation only
main_landing_review_required: true
---

# DDM receiver and optimizer build

## Outcome first

The two requested implementation cruxes are real at the bounded custody surface: a nonempty
Pose6-consuming integer/uint8 receiver with six independently framed semantic ZIP members, and an
actual typed three-stage description-space coordinate-search runner with immutable per-stage
checkpoints and disk continuation. The required n64 end-to-end custody smoke passed.

This is apparatus, not evidence. No scorer, evaluator, training, governed launch, provider, paid
dispatch, candidate archive, d_seg, d_pose, or score measurement occurred. PRIMARY
`execution_allowed` remains false. Pointer `0.1910828242 [contest-CPU]` is unchanged.

## Measured `[custody-smoke]` facts

- Exact final custody ZIP: 1,585 bytes, SHA-256
  `a504d876da469e1afec27f8aafd93fa2a06f0c157ddf7da04fcff75b1d74aa58`; filename is explicitly
  `ddm_n64_custody_final.not_a_candidate.zip`.
- Receiver output: 24,576 uint8 bytes over 64 pair records, SHA-256
  `911e96dc398add283b5af450e685f0d1186c7e8cf4f83e45eed1380b475d3352`.
- Pose consumption: 64 nonempty Pose6 records / 384 scalar residual bytes.
- Unique-home custody: six independent stored members plus EOCD cover exactly 1,585/1,585 final-ZIP
  bytes without gaps or overlap.
- No-op detector: all 773 semantic payload bytes were independently changed and every mutation
  changed receiver output; all 1,585 final-ZIP byte mutations were read and rejected by canonical
  parse/re-encode custody.
- Determinism: two same-seed runs emitted identical final archive, output, and all three checkpoint
  hashes. A separate run stopped after stage 0, reloaded from disk, and reproduced the same terminal
  archive/output.
- Search diagnostics only: integer tuple `(cell, pose-pair-delta, joint, bytes)` moved from
  `(389,143,532,1585)` to `(365,142,507,1585)`. All three stages are named and receipted as
  `candidate_search`; these integers are not task distortions or a score.
- The 64 fixture pairs explicitly cover all 11 applicable class/stratum combinations.

Custody receipt SHA-256:
`dfb4a3b57eaa3cb0311b6815e2475d92e60a9da3b68cbd2134de4020cf7cac8b`.

## Blocker-register delta

Four of the fixed 19 blockers are `GREEN_CUSTODY_SCOPE`:

1. `POSE_CONSUMING_INTEGER_UINT8_RECEIVER`
2. `DDM_OPTIMIZER_AND_STAGE_CONTINUATION_RUNNER`
3. `N64_DETERMINISTIC_CUSTODY_SMOKE`
4. `INDEPENDENTLY_FRAMED_UNIQUE_HOME_RATE_CUSTODY`

Fifteen remain red. In particular, canonical resume-registry integration remains red even though
the local runner's checkpoint/resume control is green; the round-1 review corrected that distinction.
PRIMARY target receipt, live v8/v9 owners, canonical compiler integration, governed launcher/memory,
heavy-run cleanup, fresh pose rung, four-rung ladder, n600 same-artifact closure, contest CPU/CUDA,
healthy completion/attestation, operator GO, and `execution_allowed` are still absent.

Authoritative register:
`.omx/research/ddm_receiver_optimizer_build_603_blocker_register_20260722T003529Z.json`.

## Implementation

- `DirectDescriptionZV2` owns six fixed, nonempty semantic bodies. Each is framed with stream magic,
  version, pair count, and exact body length, then stored in its own deterministic ZIP member.
- Strict parse validates order, storage method, metadata, lengths, CRCs, stream frames, and exact
  canonical re-encode. Final-ZIP local records, central records, and EOCD are partitioned into unique
  byte homes.
- The NumPy receiver uses integer accumulators and emits uint8 only. Pose6 affects frame-0 pair state;
  all other streams have explicit output effects verified by mutation.
- `DirectDescriptionOptimizerConfigV1` is the only tuning surface. The CLI adds only parser-backed
  custody paths/mode; no hidden trainer flags exist.
- `DirectDescriptionOptimizerCheckpointV1` binds config/DSL/semantic argv, current and target archive
  bytes, receiver hashes, recomputed objective, optimizer/RNG state, stage history, and continuation
  cursor. Writes are immutable and atomic; every stage file is preserved.
- The inherited PRIMARY preflight/optimize behavior remains fail-closed. The checked-in owner bundle is
  exact compiler output and still emits only `--execution-allowed false`.

## Adversarial review

Round 1 found and fixed: truncated multi-stream search coverage, a false canonical-resume green claim,
and name-only class/stratum coverage. Fix review added round-robin per-stream coordinates, restored the
canonical registry blocker to red, bound 64 pair assignments, and added config-mismatch/checkpoint-
tamper refusals. See
`.omx/research/codex_findings_ddm_receiver_optimizer_build_603_20260722T003529Z_codex.md`.

Verdict: `CLEAN_AFTER_FIX_FOR_CUSTODY_SCOPE`, not a seal.

## Verification

Exact typed smoke argv:

```text
/usr/bin/env python3 tools/run_direct_description_minimizer.py --owner-manifest .omx/research/direct_description_minimizer_owner_bundle_603_20260721T225631Z.json --mode custody-smoke --execution-allowed false --custody-config .omx/research/ddm_n64_custody_config_603_20260722T003529Z.json --output-dir .omx/research/ddm_receiver_optimizer_build_603_20260722T003529Z_artifacts
```

Exact focused test argv and result:

```text
PYTHONPATH=src /Users/adpena/Projects/pact/.venv/bin/python -m pytest -q src/tac/optimization/tests/test_direct_description_minimizer.py tools/tests/test_run_direct_description_minimizer.py
30 passed
```

Expanded DDM/S4 regression result:

```text
50 passed in 2.34s
```

Ruff, Python compilation, and `git diff --check` passed. The lane was preregistered at L0; global
`lane_maturity.py validate` remains refused on 110 pre-existing missing-evidence rows unrelated to this
lane.

## Triality and pointer delta

- DSL: typed local optimizer config and typed custody program compiler; PRIMARY owner still false.
- DAG: `.omx/research/ddm_receiver_optimizer_build_603_DAG_FEED_20260722T003529Z.md`.
- Equations: no new canonical law; custody-only objective disposition in
  `.omx/research/ddm_receiver_optimizer_build_603_canonical_equations_20260722T003529Z.md`.
- Pointer delta: none; `0.1910828242 [contest-CPU]` unmoved.

## STORES CONSULTED

- Delegated authority; `CLAUDE.md`; `AGENTS.md`; project memory top; current Codex findings/session
  summary; latest council/design memo; canonical pointer, lane, task, subagent, and inbox surfaces.
- `PROGRAM.md`, `HANDOFF.md`, `SYSTEM_MAP.md`, v7.5 §8, v8 spec, DDM PRIMARY spec, predecessor builder,
  predecessor DAG/equation notes, S4 archive composer/receiver, and
  [the operating manual](../../docs/operating_manual_craft_handoff.md).
- Fresh code, tests, owner/config manifests, custody receipt/archive, nine preserved checkpoints,
  blocker register, and the base-to-working-tree diff.

## MAIN landing review

MAIN must review the complete base-to-branch diff before merge. Re-derive: fixed ZIP metadata/ranges,
all-byte mutation coverage, Pose6 output causality, stage objective monotonicity, checkpoint identity and
continuation, the V1-legacy/V2-runner schema split, and all four scoped blocker flips. Do not promote the
8x8 custody fixture, 1,585-byte non-candidate archive, or integer L1 diagnostics into n600/scorer/score
evidence.
