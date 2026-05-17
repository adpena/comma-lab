# L5 v2 TT5L paired CPU/CUDA axis plan from anchor

- date: 2026-05-17
- gate_id: `paired_cpu_cuda_axis_plan`
- predicate_id: `l5_v2_tt5l_paired_cpu_cuda_axis_plan_from_anchor_v1`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Basis

The canonical TT5L paired exact anchor already carries both `contest_cpu` and `contest_cuda` rows for the same archive, runtime-content tree, inflate/eval devices, and component deltas. This artifact exposes those same rows under the `paired_axis_plan` gate shape so the L5 v2 readiness surface no longer reports paired-axis evidence as missing while the exact anchor pair is present.

- source anchor: `.omx/research/l5_v2_tt5l_paired_exact_anchor_pair_20260516_codex.json`
- source anchor sha256: `a390882dadc6160e40f3745faf49c4002de4aa4e64e864ecb54dab4fa9d0dc85`
- derived artifact sha256: `245c818d64860f0c48b0c075c3a397ccdc1a6fa047a3fd20788ecfb588133c28`
- pair group: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- source classification: `paired_exact_measured_config_failure_non_promotional_anchor`

## Authority

Non-promotional gate evidence only. This does not create a score claim, promotion eligibility, or dispatch readiness. It removes only the stale missing paired-axis-plan blocker by reusing already-reviewed paired exact anchor custody.
