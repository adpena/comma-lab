# L5 v2 TT5L Dry-Run Verifier Paired-Axis Follow-Up - 2026-05-17

## Scope

Follow-up to the preserved TT5L subagent adversarial review signal. This is a
test-hardening artifact, not a provider dispatch, score claim, promotion claim,
or architecture-lock claim.

## Finding

The TT5L Lightning dry-run verifier already enforces the major false-authority
and custody invariants surfaced by the subagent review:

- dry-run success remains parser/custody evidence only;
- `ready_for_non_dry_run_submit=false`;
- `ready_for_provider_dispatch=false`;
- `score_claim=false`;
- stdout/state JSON core records must match;
- local archive SHA/byte custody must match the bundle and variant manifest;
- CPU and CUDA command/spec markers remain axis-separated;
- T4 runtime env pins are required in dry-run and non-dry-run templates;
- launcher queue command SHA is not required to equal source-spec command SHA.

The concrete uncovered surface was regression coverage for the paired-axis
contract itself. The verifier implementation already checks that each variant's
`contest_cpu` and `contest_cuda` cells share archive SHA, archive byte count,
`run_id`, and `pair_group_id`, but the test suite did not have an explicit
unpaired-axis regression.

## Landed Guard

Added regression coverage in:

- `src/tac/tests/test_l5_v2_tt5l_sideinfo_lightning_execution_bundle_dry_run.py`

The new test mutates a CUDA cell's `pair_group_id` for one TT5L side-info
variant and asserts the verifier refuses the bundle with:

- `paired_axis_pair_group_id_mismatch:<variant>`
- `all_dry_runs_passed=false`
- `ready_for_dry_run_submit=false`

## Authority

- score_claim: false
- promotion_eligible: false
- ready_for_exact_eval_dispatch: false
- ready_for_provider_dispatch: false
- dispatch_attempted: false
- provider_spend_attempted: false
