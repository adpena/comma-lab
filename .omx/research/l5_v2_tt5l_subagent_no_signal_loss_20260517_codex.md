# L5 v2 TT5L Subagent No-Signal-Loss Ledger - 2026-05-17

## Context

- Repo branch: `main`
- Preserved at local head before this ledger: `5a9f06a31a982dce87ece644573a3e06a208277f`
- Scope: preserve completed read-only subagent findings before the next commit/push cycle.
- Authority: research/control-plane ledger only. No score claim, no provider dispatch claim, no promotion claim.

## Subagent Findings Preserved

### 019e354e-3c89-7f82-a7fe-f17ff0323569 - Post-harvest converter audit

Finding: the subagent did not see a complete end-to-end converter from the TT5L Lightning paired-axis plan plus harvested result directories into `--cell-json` inputs for `tools/build_l5_v2_sideinfo_effect_curve.py`.

Important caveat: this finding must be re-verified against current `main` before action because TT5L harvest/cell-generation work has been moving quickly in parallel.

Reusable evidence surfaces named by the subagent:

- `tools/build_l5_v2_sideinfo_effect_curve.py`
- `src/tac/optimization/l5_v2_sideinfo_effect_curve.py`
- `src/tac/optimization/l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan.py`
- `src/tac/optimization/l5_v2_probe_intake.py`
- `src/tac/tests/test_l5_v2_sideinfo_effect_curve.py`
- `src/tac/tests/test_l5_v2_tt5l_sideinfo_effect_curve_lightning_paired_axis_plan.py`

Recommended invariant if gap remains: adapter reads Lightning plan cells, locates harvested `local_artifact_dir`, normalizes `contest_auth_eval(.adjudicated).json` through `exact_eval_evidence_from_auth_eval_artifact`, attaches axis/variant/pair/run/source-manifest/sideinfo-liveness fields, and emits builder-compatible cell JSON.

### 019e356e-9011-7bd3-bc36-3318c8a4ab86 - TT5L Lightning bundle field and test audit

Finding: current TT5L Lightning preflight reported all 10 cells ready for operator claiming while still `ready_for_provider_dispatch=false`; the bundle should preserve that false-authority boundary explicitly.

Recommended top-level bundle fields:

- schema/tool/generated-at/current-head metadata
- source plan path/SHA/commit
- source variant manifest path/SHA
- dispatch plan path/SHA
- runtime submission dir
- runtime tree SHA, runtime content SHA, runtime file count
- required variants, axes, expected cell count, ready cell count, execution order
- global blockers

Required false-authority fields:

- `planning_only=true`
- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`
- `operator_execute_required=true`

Recommended per-cell fields:

- variant, axis, axis label, role, required device
- lane id, platform, job name, pair group id, run id
- archive path/bytes/SHA
- local artifact dir and expected remote output dir
- source spec command SHA
- queue metadata and adjudication spec
- expected result/adjudicated/provenance/report/log paths
- runner preflight files, supply-chain scan files, CUDA-only DALI files
- claim command and terminal success/failure claim templates
- harvest command and active claim conflicts

Required Dykstra gate: if `.omx/state/dykstra_feasibility_time_traveler_l5.json` is missing or invalid, promotional/provider-ready sideinfo claims must remain blocked. Dry-run parse readiness may pass, but sideinfo proof, paired-anchor, or promotion-adjacent readiness must not pass.

Recommended tests:

- bundle artifact field coverage
- Dykstra missing/bad/valid gating
- source plan SHA/current-head staleness
- source-staging and identity blockers
- active-claim closure semantics
- CPU/CUDA axis invariant checks
- stale harvest archive SHA refusal
- CPU vs CUDA expected-artifact requirements
- source command hash integrity
- effect-curve handoff only when all 10 exact-eval cells and pair identities are present

### 019e357f-da29-75c0-b15d-6fd3b2ea82d9 - Adversarial verifier audit

Finding: Lightning dry-run success is parse-only authority. It does not prove dispatch claims, identity, remote preflight, source staging, archive/runtime availability, or promotion readiness.

Required invariant: top-level and per-cell records must keep:

- `planning_only=true`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `ready_for_non_dry_run_submit=false`

Custody checks the verifier should own directly:

- compute local archive SHA/bytes for all five variants
- compare archive custody against bundle, paired-axis plan, preflight, and variant manifest
- check `inflate.sh` and runtime manifest files exist
- compare recorded runtime tree/content hashes

Command custody distinction:

- `source_spec_command_sha256` must match the paired-axis plan cell command SHA
- launcher dry-run state must have `queue.command_sha256 == sha256(spec.command)`
- `spec.queue_metadata.source_spec_command_sha256` must match bundle cell `source_spec_command_sha256`
- launcher command SHA equality to the source spec is not required if the submit layer intentionally adds metadata

All-cell coverage:

- exact set must be `{zero, random_lsb, shuffled, trained, ablated} x {contest_cpu, contest_cuda}`
- no duplicates and no extras
- each variant's CPU/CUDA pair must share archive SHA, archive bytes, run id, and pair group

Edge cases to harden:

- stale stdout/state split where ignored `dry_run.stdout` and `launcher_dry_run_state.json` disagree
- T4 non-dry-run templates need explicit inflate torch env pins
- CPU and CUDA axes must remain separate with no implied conversion

## Next Non-Retread Actions

1. Wire route-unblock and doctor-plan artifact status into the L5 v2 architecture-lock/readiness packet so those artifacts are discoverable from the normal control plane.
2. Add missing verifier/bundle invariants above if not already implemented on current `main`.
3. Rebuild the TT5L sideinfo chain after any source-control-plane edits because generated source-relevant hashes include `src/tac/optimization/l5_staircase_v2.py`.
4. Keep all resulting state false-authority clean until a claimed non-dry-run provider lifecycle, exact-eval harvest, and adjudicated CPU/CUDA pair exist.

## Classification

- Signal class: subagent research and adversarial engineering review.
- Current evidence grade: read-only codebase review.
- Promotion eligibility: false.
- Provider dispatch eligibility: false.
- Score claim: false.
