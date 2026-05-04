# PR79 Action Subset Worker - 2026-05-03

Scope: local-only PR79 action-mining worker. No GPU dispatch was performed.

Inputs:

- C102 A++ frontier archive:
  `experiments/results/lightning_batch/exact_eval_c091_next_cem_pose_waterfill_top192_s0125_m06_t4_20260503T1238Z/archive.zip`
- C102 bytes: `276485`
- C102 SHA-256:
  `79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8`
- C102 score: `0.31514430182167497`
- PR79 public archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip`
- PR79 bytes: `277388`
- PR79 SHA-256:
  `01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446`

Outputs:

- Worker:
  `experiments/build_pr79_action_subset_candidates.py`
- Candidate matrix:
  `experiments/results/pr79_action_subset_worker_20260503/candidate_matrix.json`
- Tests:
  `src/tac/tests/test_build_pr79_action_subset_candidates.py`

Local closure:

- PR79 action slice charged Brotli bytes: `1162`
- Worker SG2 decode: `true`
- SG2 record count: `672`
- Worker runtime SHA-256:
  `a48bd4e49f8928158756610fd8094e8fb1611a2040121611055266f840faf13f`
- Runtime matches robust unpacker: `true`
- Exact prior matches in PR79 stream: `97`
- Pose-safe exact prior matches: `2`

Candidate screen:

| Candidate | Bytes | Delta vs C102 | Records | Expected proxy | Required to <=0.31 | Plausible |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `replace_pr79_exact_positive_top16_on_c102_p6` | `276254` | `-231` | `5` | `0.00002440268208356543` | `0.0049904884035037456` | `false` |
| `replace_pr79_pose_safe_top16_on_c102_p6` | `276244` | `-241` | `2` | `0.000012779667651705912` | `0.004983829813972551` | `false` |
| `replace_pr79_pair_opportunity_top32_on_c102_p6` | `276332` | `-153` | `32` | `0.000005880111112056621` | `0.005042425401847295` | `false` |
| `augment_c102_pr79_exact_positive_top16_on_c102_p6` | `276506` | `21` | `113` | `0.00002440268208356543` | `0.005158284859690554` | `false` |
| `replace_pr79_all_on_c102_p6` | `277539` | `1054` | `672` | `0.00010236918594599464` | `0.005846117158265718` | `false` |

Decision:

- No local candidate clears the `<=0.31` break-even screen.
- `candidate_matrix.json` sets `exact_eval_justified=false`.
- Dispatch requires a fresh lane claim, exact archive byte/SHA match, runtime
  closure fields passing, and exact CUDA auth eval on the identical archive.

Verification:

```text
.venv/bin/python -m py_compile experiments/build_pr79_action_subset_candidates.py src/tac/tests/test_build_pr79_action_subset_candidates.py
.venv/bin/python -m pytest src/tac/tests/test_build_pr79_action_subset_candidates.py -q
....                                                                     [100%]
4 passed in 0.09s
.venv/bin/python experiments/build_pr79_action_subset_candidates.py --force
candidate_count=8 exact_eval_justified=false skipped_count=0
```
