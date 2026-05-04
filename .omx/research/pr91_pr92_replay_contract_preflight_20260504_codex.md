# PR91/PR92 Replay Contract Preflight - 2026-05-04

- tool: `experiments/preflight_pr91_pr92_replay_contracts.py`
- status: `passed_pr92_a_plus_plus_pr91_fail_closed`
- dispatch_performed: `false`
- remote_jobs_dispatched: `false`

## PR91 HPM1

- status: `blocked_hpm1_probability_range_contract_mismatch`
- dispatch_allowed: `False`
- bug_class: `hpm1_probability_range_contract_mismatch`
- failure_reason: `no_probability_variant_decodes_pr91_hpm1_prefix`
- failed_variants: `source_float64_perfect_false, source_float32_perfect_false, source_float64_perfect_true, source_float32_perfect_true`

PR91 remains fail-closed. The local source-contract variant still fails at `frame=0 group=10 symbol=191` after `5951` decoded symbols, and no tested probability/range variant decodes frame 0.

## PR92 RMB1 Stack

- status: `passed_t4_exact_pr92_rmb1_stack_validated`
- evidence_grade: `A++`
- score: `0.2535063602939779`
- archive bytes: `229480`
- archive sha256: `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`
- avg_segnet_dist: `0.00057185`
- avg_posenet_dist: `0.0001894`
- runtime_tree_sha256: `9a9a71afefe7c154ecc188068bea26f01212369883c4b32c9706b32951e267ba`

PR92/RMB1 is not blocked: the validated opportunity is already realized as a pure-rate randmulti recode stacked onto the PR85 STBM1BR frontier.

## Next Safe Commands

```bash
.venv/bin/python experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py --stdout
```

claim lane first, then exact-eval the rebuilt bytes with `experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/archive.zip` and custom inflate `experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/replay_submission_stbm_rmb1/inflate.sh`; skip if archive SHA still equals the already validated exact JSON.
