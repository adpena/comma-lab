# L5 v2 TT5L materialized paired work-unit plan

- schema: `modal_paired_auth_eval_dispatch_plan_v2`
- materialized artifact: `.omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json`
- artifact sha256: `49898f162c5d312dc3de4f1d2571294cb7e4ffb9ba628c76ad9acf6868489b03`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- dispatch_attempted: `false`

## Purpose

This landing moves the L5 v2 TT5L paired probe from a generic blocked
dispatch template to a materialized, byte-closed work unit with archive,
runtime, pair-group, lane-id, and per-axis Modal command custody.

It does not execute the job and does not claim score movement. Its job is to
make the next frontier action concrete: review the materialized TT5L
archive/runtime packet, then run the canonical paired dispatcher only if the
operator accepts the current-runtime rerun policy.

## Materialized Custody

- archive path:
  `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip`
- archive bytes: `34603`
- archive sha256:
  `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`
- submission runtime:
  `experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime`
- pair group:
  `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- CPU lane:
  `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu`
- CUDA lane:
  `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda`
- CPU expected Modal uploaded runtime tree:
  `2b2b9dfdb0f3e59af3511e4502a3a4c0cbe9c1f52405b98eb4dec331db248584`
- CUDA expected Modal uploaded runtime tree:
  `2b0dcb5a148ddef7bf56c833bd46fa5830bdde88929b9fd417b4985bea678a28`
- shared runtime content tree:
  `630970e9dc78c6e2f8dc2ed8d1e22503ea7d0cab17b4da5615a8a1c5b83ac718`

## Runtime Policy

The older recovered TT5L `[contest-CUDA]` anchor remains a real single-axis
artifact, but it is not reused as the CUDA half of this paired work unit. The
prior runtime-mismatch packet classified that case as
`existing_cuda_anchor_runtime_mismatch_for_paired_measurement`.

This materialized work unit therefore represents the clean current-runtime
policy: rerun both `[contest-CPU]` and `[contest-CUDA]` under the same
materialized archive/runtime contract, then harvest through the canonical
Modal recovery path.

## Hardening

`src/tac/optimization/l5_staircase_v2.py` now validates the materialized work
unit before it can become the next TT5L action. The validator refuses:

- missing or malformed archive path, archive bytes, or archive SHA;
- archive path that does not exist on disk;
- archive byte or SHA mismatch;
- missing runtime directory or `inflate.sh`;
- missing per-axis runtime tree or runtime content hashes;
- pair-group mismatch;
- lane-id mismatch;
- per-axis command lists that are not Modal `run` commands;
- wrong CPU/CUDA wrapper scripts;
- missing provider detach flags;
- command flag mismatches for archive, archive SHA, runtime, pair group, lane,
  or expected runtime tree.

This closes the weak-command gap where any non-empty command list could have
made a materialized plan appear valid.

## Operator Next Action

The TT5L campaign readiness surface now advances to:

`review_and_execute_l5_v2_tt5l_materialized_paired_measurement`

The emitted operator command remains non-executing until `--execute` is added
after review. The generated execute template routes through
`tools/dispatch_modal_paired_auth_eval.py`, not direct single-axis wrappers,
and the per-axis wrappers remain owned by the paired dispatcher.

## Verification

Focused tests:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_advances_after_work_unit_materialized \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_materialized_work_unit_rejects_weak_axis_commands \
  src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_probe_action_uses_existing_fail_closed_intake -q
```

Result: `3 passed`.

Lint:

```text
.venv/bin/ruff check src/tac/optimization/l5_staircase_v2.py src/tac/tests/test_l5_staircase_v2.py
```

Result: `All checks passed!`.

Live readiness check:

```text
artifact_valid=True
blockers=[]
next_action=review_and_execute_l5_v2_tt5l_materialized_paired_measurement
ready_for_operator_dispatch=True
ready_for_exact_eval_dispatch=False
```

## Remaining Blockers

- No dispatch has been attempted from this packet.
- The returned CPU/CUDA artifacts do not exist yet.
- Side-info effect-curve cells still need paired evidence.
- Probe observations must be regenerated from harvested paired results before
  L5 v2 architecture lock, rank, kill, promotion, or stack-of-stacks authority.
