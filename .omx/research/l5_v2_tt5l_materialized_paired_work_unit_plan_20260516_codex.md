# L5 v2 TT5L materialized paired work-unit plan

- schema: `modal_paired_auth_eval_dispatch_plan_v2`
- tool: `tools/build_l5_v2_tt5l_materialized_paired_work_unit.py`
- materialized artifact: `.omx/research/l5_v2_tt5l_materialized_paired_work_unit_plan_20260516_codex.json`
- run_id: `l5_v2_measure_tt5l_autonomy_paired_exact_paired_measurement_random_lsb_ccce77aaf190`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- dispatch_attempted: `false`
- materialized_variant: `random_lsb`

## Materialized Custody

- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/random_lsb/archive.zip`
- archive bytes: `38681`
- archive sha256: `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1`
- submission runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`
- pair group: `pair_l5_v2_measure_tt5l_autonomy_paired_exact_cpu_cuda`
- CPU lane: `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cpu`
- CUDA lane: `lane_l5_v2_measure_tt5l_autonomy_paired_exact_contest_cuda`
- CPU expected Modal uploaded runtime tree: `025123f140784b92b0d6145a05df6e97aedace7cf3e620a10551454cfa8057a2`
- CUDA expected Modal uploaded runtime tree: `83f5f760000802b69abb187e8239403a0d8fd9ed21ec8d127fa0542ca91001aa`
- CPU runtime content tree: `bf857cb79055e914b64e5cd5788a5a15a37824e01b32c871b84715c153c0ab32`
- CUDA runtime content tree: `bf857cb79055e914b64e5cd5788a5a15a37824e01b32c871b84715c153c0ab32`

## Materialization Source

- variant manifest: `.omx/research/l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.json`
- variant manifest sha256: `80962a29c5abc8c8de2dbae742e228c396e5f3ce3423397cfe7d79bae1f06459`
- variant report: `.omx/research/l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.md`
- source archive: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/archive.zip`
- source archive sha256: `2a8cf3389aaeb217a96058bf7b47b43e4fc364193795d66fe7ab75479102d11f`

## Operator Next Action

Review this materialized CPU/CUDA work unit, then execute through the canonical paired dispatcher only if the archive/runtime custody is still accepted. This packet intentionally stays `ready_for_provider_dispatch=false` until that explicit operator step.
