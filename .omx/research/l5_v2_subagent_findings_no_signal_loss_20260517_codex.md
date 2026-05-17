# L5 v2 Subagent Findings No-Signal-Loss Ledger

Date: 2026-05-17
Parent commit at capture: `29ca652e7e7f2458bf1779364c8f92b561613df3`
Scope: preserve returned subagent findings after the L5-v2 artifact-hash
hardening push.

## Plato the 3rd: artifact-hash binding review

Status: fixed in `29ca652e7`.

Findings captured:

- `l5_v2_probe_disambiguator.py` did not require `artifact_sha256` for axis
  exact-eval evidence and did not map missing/mismatched artifact-hash
  blockers.
- `l5_v2_measurement_schedule.py` did not require `artifact_sha256` when
  revalidating side-info effect-curve cells.
- The tracked side-info effect curve carried `artifact_path` without
  `artifact_sha256`.
- Side-info builder auto-computation of missing artifact hashes needed explicit
  regression coverage.
- `exact_eval_custody.py` accepts strict artifact hashes without
  `artifact_base_dir` as format validation only; named L5 callers pass a base
  directory.

Actions landed in `29ca652e7`:

- Shared exact-eval validator now supports `require_artifact_sha256`.
- L5-v2 probe intake, probe gate, side-info builder, and measurement schedule
  require or emit artifact-hash custody.
- Probe gate and side-info artifacts were regenerated.
- Regression tests cover validator alias/missing/mismatch, probe axis
  hash-binding, probe-intake exact source SHA, side-info mismatch, and
  measurement-schedule missing/mismatch.

## Euler the 3rd: readiness/adversarial review

Status: preserved as open follow-up. These were returned after the
artifact-hash hardening commit was already pushed and should not be lost.

Findings captured:

1. Stale or invalid Modal provider-blocker artifacts can fall through into an
   executable next action when only `provider_blocker_status["active"] is True`
   is treated as blocking. Recommendation: if a provider-blocker artifact
   exists but is invalid, surface its blockers, set
   `ready_for_operator_dispatch=false`, and route to
   `refresh_or_retire_provider_blocker`.
2. Lightning paired-axis source-custody mismatch can still report dry-run
   readiness. Recommendation: split structural dry-run validity from current
   source-custody readiness, or set dry-run readiness false when source commit
   mismatches head.
3. Per-axis TT5L evidence rows can preserve stale paired-axis state such as
   `contest_cuda_pending_for_internal_promotion` even after the pair ledger has
   CPU and CUDA evidence. Recommendation: create or refresh a pair-level
   machine-readable anchor artifact and make `exact_anchor_or_diagnostic_pair`
   consume it instead of stale per-axis flags.
4. Materialized work-unit validation checks that runtime-content SHA fields
   exist, but not that CPU and CUDA runtime-content hashes match at that layer.
   Recommendation: block materialized work units with divergent per-axis
   runtime-content hashes before surfacing executable work.

Verification reported by Euler:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q -p no:cacheprovider src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_readiness_surfaces_current_lightning_paired_axis_plan
pass
```

## Next Action

Treat Euler's four findings as the next L5-v2 hardening queue before any
operator-facing execution action is surfaced from those readiness paths.
