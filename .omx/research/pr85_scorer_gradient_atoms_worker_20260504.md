# PR85 Scorer-Gradient Atom Opportunity Profiler

- planning_only: true
- score_claim: false
- compression_time_only: true
- dispatch_performed: false
- remote_jobs_dispatched: false
- inflate_time_scorer_load_allowed: false

## Formula Checks

- score = 100*seg_dist + sqrt(10*pose_dist) + 25*archive_bytes/37545489
- dscore/dseg = 100.0
- dscore/dpose = 88.45748207237921
- dscore/dbyte = 6.658589531221714e-07
- recomputed_score = 0.271005857399254
- reported_score = 0.27100583104425036
- abs_error_vs_reported = 2.635500362391241e-08

## Inputs

- exact_eval_json: experiments/results/lightning_batch/exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z/contest_auth_eval.json
- component_traces: 1
- profiles: 0

## Top Atoms

- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0139: score_opportunity=0.0014743526278965591, break_even_bytes=2214.211614912454, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0526: score_opportunity=0.0011523730552869397, break_even_bytes=1730.6563948468872, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0396: score_opportunity=0.0007831681797888973, break_even_bytes=1176.1772911765624, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0160: score_opportunity=0.0006926329761866469, break_even_bytes=1040.2097515381204, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0307: score_opportunity=0.0006432076320361907, break_even_bytes=965.9818029332337, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0067: score_opportunity=0.0005841195161058267, break_even_bytes=877.2421146654656, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0502: score_opportunity=0.0005652650335331117, break_even_bytes=848.9260839440831, gate=blocked_planning_only
- exact_eval_pr85_preserve_post_all_shift_frac2_frac3_t4hedge_g4dn2_20260504T0246Z:pair_0122: score_opportunity=0.0005648416663994297, break_even_bytes=848.2902629016583, gate=blocked_planning_only

## Dispatch Gates

- status: blocked_planning_only
- required next proof: build a closed archive candidate, prove non-noop payload/raw-output change, then run exact CUDA auth eval.
