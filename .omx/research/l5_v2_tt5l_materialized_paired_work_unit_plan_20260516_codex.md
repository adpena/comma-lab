# L5 v2 TT5L materialized paired work-unit plan

- schema: `modal_paired_auth_eval_dispatch_plan_v2`
- tool: `tools/build_l5_v2_tt5l_materialized_paired_work_unit.py`
- materialized artifact: `.omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json`
- run_id: `l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_b6a5b63c0ea8`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- dispatch_attempted: `false`
- materialized_variant: `random_lsb`

## Materialized Custody

- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_20260517_codex/random_lsb/archive.zip`
- archive bytes: `38911`
- archive sha256: `b6a5b63c0ea8acd582d8f273a1ee9e00f74becc9d1993a2f3085f2f89d64b1c7`
- submission runtime: `experiments/results/time_traveler_recovered_exact_eval_20260514_codex/runtime`
- pair group: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- CPU lane: `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu`
- CUDA lane: `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda`
- CPU expected Modal uploaded runtime tree: `f419ca980f533652a90d227fe675b433f0784e40547b68327fa091bcf95fb453`
- CUDA expected Modal uploaded runtime tree: `5518199a6f95f43366aa37bfb7452ae76892abdda73ff489f1fffaaf1dd27583`
- CPU runtime content tree: `b9bd9ffecaac4d926b4c9174cdd7e23cb872f3139e1abf617cd4423e3a65e9c0`
- CUDA runtime content tree: `b9bd9ffecaac4d926b4c9174cdd7e23cb872f3139e1abf617cd4423e3a65e9c0`

## Materialization Source

- variant manifest: `.omx/research/l5_v2_tt5l_sideinfo_variant_packets_20260517_codex.json`
- variant manifest sha256: `db9ddcc7f43acbeddf61209fbead038cebcfabf86af95e291faef04aa8b68839`
- variant report: `.omx/research/l5_v2_tt5l_sideinfo_variant_packets_20260517_codex.md`
- source archive: `experiments/results/lane_substrate_time_traveler_l5_autonomy_modal_a100_dispatch_20260514T100758Z__smoke__25ep_modal/lane_substrate_time_traveler_l5_autonomy_results/output/archive.zip`
- source archive sha256: `2b05b7351b690b0b2251ddc620d80dd9a1833051cfa07e679106d00fbc70024a`

## Operator Next Action

Review this materialized CPU/CUDA work unit, then execute through the canonical paired dispatcher only if the archive/runtime custody is still accepted. This packet intentionally stays `ready_for_provider_dispatch=false` until that explicit operator step.
