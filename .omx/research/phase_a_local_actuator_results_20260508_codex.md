# Phase A local actuator results - 2026-05-08

## Scope

This ledger preserves the local CPU actuator artifacts produced after the
`phase_a_dispatch_wrapper` landed on `main`. These are not score claims and do
not authorize GPU dispatch by themselves.

## A0 MDL lower bound

- Command:
  `.venv/bin/python tools/dispatch_phase_a_track_1_ablations.py --decision A0 --substrate pr101 --output experiments/results`
- Rollup:
  `experiments/results/phase_a_dispatch_rollup_20260508T154126Z.json`
- Artifact:
  `experiments/results/track1_phase_a0_mdl_20260508T154125Z/A0_result.json`
- Status:
  `exit_code=0`, `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`
- Evidence grade:
  `byte_proxy_only_deterministic_closed_form`

Observed closed-form values:

| Quantity | Value |
|---|---:|
| PR101 proxy elements | 228,958 |
| Per-tensor iid floor | 175,916 B |
| Joint floor estimate | 148,000-162,000 B |
| ChARM hyperprior overhead | 3,500 B |
| Total lower/aggressive bound | 151,700 B |
| Total realistic bound | 158,700 B |
| Rate term at lower bound | 0.1010108032 |
| Rate term at realistic bound | 0.1056718159 |

Interpretation:
The A0 result keeps the Track 1 lower-bound math live and reproducible. It is
still a closed-form planning primitive using the synthetic PR101 proxy when no
explicit weights path is supplied, so it must remain below score-authority and
dispatch-authority thresholds.

## A2 sensitivity-weighted coarsening

- Command:
  `.venv/bin/python tools/dispatch_phase_a_track_1_ablations.py --decision A2 --substrate pr101 --output experiments/results`
- Rollup:
  `experiments/results/phase_a_dispatch_rollup_20260508T154209Z.json`
- Artifact:
  `experiments/results/track1_phase_a2_sensitivity_quant_20260508T154125Z/A2_result.json`
- Status:
  `completed_local_diagnostic`, `score_claim=false`,
  `ready_for_exact_eval_dispatch=false`, `remote_dispatch_allowed=false`
- Evidence grade:
  `CPU-local allocator proxy`

Selected diagnostic allocation at `rms_target=0.0386`:

```text
selected_Ks = [7, 18, 3, 7, 1, 9, 3, 1, 1, 1, 2, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
total_bytes = 159,544
rel_err = 0.03360543944803241
payload_brotli_bytes = 143,422
archive_overhead_bytes = 16,094
side_info_bytes = 28
```

Required blockers intentionally remain:

- `diagnostic_or_stub_sensitivity_map_not_score_authority`
- `no_byte_closed_runtime_packet_built`
- `no_contest_cpu_auth_eval`
- `no_exact_cuda_auth_eval`
- `score_sensitivity_artifact_must_be_certified_before_promotion`

Interpretation:
The A2 actuator is useful because it changes the allocation pattern rather than
replaying the retired uniform lossy-coarsening config, but this run used the
PR106 stub sensitivity map at
`experiments/results/sensitivity_map_pr106_20260504_claude/sensitivity_map_stub.pt`.
It is therefore a planning artifact only. The next valid step is to replace the
stub with a certified PR101 component-sensitivity map and then feed the selected
K schedule into a byte-closed packet builder before any paired CPU/CUDA auth
eval.

## Promotion discipline

Neither artifact may rank, promote, kill, or dispatch a lane. The only valid
uses are:

1. Keep the Track 1 lower-bound and allocator math reproducible.
2. Guide the next local build order toward certified sensitivity and
   byte-closed packet closure.
3. Supply falsifiable expectations for later paired `[contest-CPU]` and
   `[contest-CUDA]` archive evals.
