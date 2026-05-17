# L5 v2 TT5L side-info effect-curve dispatch plan

- schema: `l5_v2_tt5l_sideinfo_effect_curve_dispatch_plan_v1`
- plan_id: `l5_v2_tt5l_sideinfo_effect_curve_dispatch_7f9dc3f41851ef3a`
- source_manifest_path: `.omx/research/l5_v2_tt5l_current_code_fullshape_sideinfo_variant_packets_20260517_codex.json`
- source_manifest_sha256: `53520df3292bcf9a1f4dce23f4da0ea65f7e54de5d07b42ee80aee4b2a9966ec`
- measurement_id: `measure_tt5l_sideinfo_effect_curve`
- required_axes: `['contest_cpu', 'contest_cuda']`
- required_variants: `['zero', 'random_lsb', 'shuffled', 'trained', 'ablated']`
- work_unit_count: `5`
- ready_work_unit_count: `5`
- planning_only: `true`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`
- dispatch_attempted: `false`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- operator_execute_required: `true`
- blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'requires_paired_cpu_cuda_exact_eval_before_score_claim']`

## Work Units

### zero

- work_unit_id: `measure_tt5l_sideinfo_effect_curve__zero`
- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/zero/archive.zip`
- archive bytes: `34373`
- archive sha256: `b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3`
- submission runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`
- lane_id: `lane_l5_v2_tt5l_sideinfo_effect_curve_zero`
- lanes: `{'contest_cuda': 'lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cuda', 'contest_cpu': 'lane_l5_v2_tt5l_sideinfo_effect_curve_zero_contest_cpu'}`
- pair_group_id: `pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102`
- required_cells: `[{'axis': 'contest_cpu', 'variant': 'zero'}, {'axis': 'contest_cuda', 'variant': 'zero'}]`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- dispatch_blockers: `[]`
- score_claim_blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'requires_paired_cpu_cuda_exact_eval_before_score_claim']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/zero/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_zero --expected-archive-sha256 b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3 --run-id l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_zero --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/zero --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=zero;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102;archive_sha=b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`
- operator_execute_command_after_review: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/zero/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_zero --expected-archive-sha256 b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3 --run-id l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_zero --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/zero --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=zero;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_zero_b444cc91f102;archive_sha=b444cc91f102c9807a865ed59f182ca5c83f3239a49ec2aa400b497d7dea37a3' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists --execute`

### random_lsb

- work_unit_id: `measure_tt5l_sideinfo_effect_curve__random_lsb`
- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/random_lsb/archive.zip`
- archive bytes: `38681`
- archive sha256: `ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1`
- submission runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`
- lane_id: `lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb`
- lanes: `{'contest_cuda': 'lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_contest_cuda', 'contest_cpu': 'lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_contest_cpu'}`
- pair_group_id: `pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190`
- required_cells: `[{'axis': 'contest_cpu', 'variant': 'random_lsb'}, {'axis': 'contest_cuda', 'variant': 'random_lsb'}]`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- dispatch_blockers: `[]`
- score_claim_blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'requires_paired_cpu_cuda_exact_eval_before_score_claim']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/random_lsb/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_random_lsb --expected-archive-sha256 ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1 --run-id l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/random_lsb --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=random_lsb;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190;archive_sha=ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`
- operator_execute_command_after_review: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/random_lsb/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_random_lsb --expected-archive-sha256 ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1 --run-id l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_random_lsb --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/random_lsb --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=random_lsb;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_random_lsb_ccce77aaf190;archive_sha=ccce77aaf1907d6e70d8cba498261708b241ac7d78a9bf22978aa459cb6b7fd1' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists --execute`

### shuffled

- work_unit_id: `measure_tt5l_sideinfo_effect_curve__shuffled`
- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/shuffled/archive.zip`
- archive bytes: `43284`
- archive sha256: `c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3`
- submission runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`
- lane_id: `lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled`
- lanes: `{'contest_cuda': 'lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled_contest_cuda', 'contest_cpu': 'lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled_contest_cpu'}`
- pair_group_id: `pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4`
- required_cells: `[{'axis': 'contest_cpu', 'variant': 'shuffled'}, {'axis': 'contest_cuda', 'variant': 'shuffled'}]`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- dispatch_blockers: `[]`
- score_claim_blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'requires_paired_cpu_cuda_exact_eval_before_score_claim']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/shuffled/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_shuffled --expected-archive-sha256 c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3 --run-id l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/shuffled --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=shuffled;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4;archive_sha=c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`
- operator_execute_command_after_review: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/shuffled/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_shuffled --expected-archive-sha256 c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3 --run-id l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_shuffled --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/shuffled --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=shuffled;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_shuffled_c235e5cb91f4;archive_sha=c235e5cb91f4122c3bb642354b424dd21f8b37cc0d2ac7b3e03c0b84dcc49bc3' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists --execute`

### trained

- work_unit_id: `measure_tt5l_sideinfo_effect_curve__trained`
- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/trained/archive.zip`
- archive bytes: `43323`
- archive sha256: `f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a`
- submission runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`
- lane_id: `lane_l5_v2_tt5l_sideinfo_effect_curve_trained`
- lanes: `{'contest_cuda': 'lane_l5_v2_tt5l_sideinfo_effect_curve_trained_contest_cuda', 'contest_cpu': 'lane_l5_v2_tt5l_sideinfo_effect_curve_trained_contest_cpu'}`
- pair_group_id: `pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779`
- required_cells: `[{'axis': 'contest_cpu', 'variant': 'trained'}, {'axis': 'contest_cuda', 'variant': 'trained'}]`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- dispatch_blockers: `[]`
- score_claim_blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'requires_paired_cpu_cuda_exact_eval_before_score_claim']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/trained/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_trained --expected-archive-sha256 f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a --run-id l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_trained --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/trained --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=trained;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779;archive_sha=f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`
- operator_execute_command_after_review: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/trained/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_trained --expected-archive-sha256 f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a --run-id l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_trained --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/trained --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=trained;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_trained_f08299c5e779;archive_sha=f08299c5e77908eeeb82cc9948e530fd5a790894902a726ebd8a258596b4bf1a' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists --execute`

### ablated

- work_unit_id: `measure_tt5l_sideinfo_effect_curve__ablated`
- archive path: `experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/ablated/archive.zip`
- archive bytes: `42419`
- archive sha256: `ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39`
- submission runtime: `experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir`
- lane_id: `lane_l5_v2_tt5l_sideinfo_effect_curve_ablated`
- lanes: `{'contest_cuda': 'lane_l5_v2_tt5l_sideinfo_effect_curve_ablated_contest_cuda', 'contest_cpu': 'lane_l5_v2_tt5l_sideinfo_effect_curve_ablated_contest_cpu'}`
- pair_group_id: `pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998`
- required_cells: `[{'axis': 'contest_cpu', 'variant': 'ablated'}, {'axis': 'contest_cuda', 'variant': 'ablated'}]`
- ready_for_operator_dispatch: `True`
- ready_for_provider_dispatch: `false`
- dispatch_blockers: `[]`
- score_claim_blockers: `['requires_paired_cpu_cuda_exact_eval_for_sideinfo_effect_curve', 'requires_dispatch_lane_claim_before_auth_eval', 'score_claim_forbidden_until_effect_curve_artifact_passes', 'requires_paired_cpu_cuda_exact_eval_before_score_claim']`
- dispatch_command: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/ablated/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_ablated --expected-archive-sha256 ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39 --run-id l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_ablated --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/ablated --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=ablated;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998;archive_sha=ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists`
- operator_execute_command_after_review: `.venv/bin/python tools/dispatch_modal_paired_auth_eval.py --archive experiments/results/time_traveler_l5_v2/tt5l_sideinfo_variant_packets_current_code_fullshape_advisory_20260517T052719Z/ablated/archive.zip --submission-dir experiments/results/time_traveler_l5_v2/tt5l_current_code_fullshape_sideinfo_cpu_advisory_20260517T052719Z/submission_dir --inflate-sh inflate.sh --label l5_v2_tt5l_sideinfo_effect_curve_ablated --expected-archive-sha256 ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39 --run-id l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998 --pair-group-id pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998 --lane-id-base lane_l5_v2_tt5l_sideinfo_effect_curve_ablated --output-root experiments/results/l5_v2_probe/tt5l_sideinfo_effect_curve/ablated --modal-bin .venv/bin/modal --gpu T4 --claim-agent codex:l5_v2_tt5l_sideinfo_effect_curve --claim-notes 'l5_v2_tt5l_sideinfo_effect_curve;variant=ablated;pair_group_id=pair_l5_v2_tt5l_sideinfo_effect_curve_ablated_ec3432658998;archive_sha=ec343265899859495fca1d2874c1b85d211210ea13a2de5bb7e52c9816ba6b39' --expected-runtime-tree-sha256 auto --skip-axis-if-promotable-anchor-exists --execute`

## Classification

This is a byte-closed operator dispatch plan for the TT5L side-info effect curve. It does not launch provider work, does not create lane claims, and does not claim score movement. Each variant still needs paired `[contest-CPU]` and `[contest-CUDA]` exact-eval cells harvested through the canonical Modal recovery path before the side-info usefulness predicate can be evaluated.
