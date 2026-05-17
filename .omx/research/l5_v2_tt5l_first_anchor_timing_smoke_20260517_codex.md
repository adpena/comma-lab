# L5 v2 TT5L first-anchor timing smoke

Date: 2026-05-17

## Scope

This promotes the TT5L first-anchor timing-smoke custody surface from raw
`.omx/state` into durable `.omx/research` so `main` preserves the timing
evidence needed by the L5-v2 architecture lock packet.

## Artifact

- JSON: `.omx/research/l5_v2_tt5l_first_anchor_timing_smoke_20260517_codex.json`
- schema: `tt5l_first_anchor_timing_smoke_v1`
- predicate: `tt5l_first_anchor_timing_smoke_rate_v1`
- provider: `modal`
- provider_call_id: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Measured Timing

The artifact is derived from the recovered paired exact CPU/CUDA TT5L anchor.
It is timing/custody evidence only; it does not change the measured-config
classification.

| axis | contest_auth_eval_elapsed_seconds | inflate_seconds | evaluate_seconds | source artifact |
| --- | ---: | ---: | ---: | --- |
| `contest_cpu` | `293.79739159300004` | `95.58252872700001` | `194.805340313` | `experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval_cpu/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cpu/contest_auth_eval.json` |
| `contest_cuda` | `53.279486186` | `14.870239315000001` | `28.000436932999996` | `experiments/results/l5_v2_probe/measure_tt5l_autonomy_paired_exact/modal_auth_eval/l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_cuda/contest_auth_eval.json` |

The timing rate used by the packet is the slower paired wall-clock axis:
`seconds_per_candidate = 296.600486271`, taken from the CPU Modal result sidecar
for the paired run. The CUDA sidecar recorded `56.412989613`.

## Effect On Lock Packet

`first_anchor_timing_smoke_artifact_valid` is now `true` in
`.omx/research/l5_v2_architecture_lock_packet_20260516_codex.{json,md}`.

Architecture lock remains blocked by:

- `requires_all_l5_v2_gate_evidence_valid`
- `requires_c1_z5_tt5l_probe_gate_evidence`
- `requires_paired_cpu_cuda_sideinfo_effect_curve`

This is intentional. The paired anchor was a measured-config failure with zero
TT5L side-info utility; timing evidence removes a custody blocker but does not
make the candidate score-rankable, promotable, or dispatch-ready.

## Verification

- `.venv/bin/pytest -q src/tac/tests/test_l5_staircase_v2.py::test_l5_v2_tt5l_first_anchor_timing_accepts_paired_axis_timings`
- `.venv/bin/python tools/build_l5_v2_architecture_lock_packet.py`

