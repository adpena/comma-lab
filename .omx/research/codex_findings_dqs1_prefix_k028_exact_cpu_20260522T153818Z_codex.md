# Codex Findings: DQS1 Prefix K=28 Exact CPU Calibration

## Verdict

`prefix_k028` is exact-eval falsified as a new `[contest-CPU]` frontier. It is
byte-closed and locality-clean, but exact CPU scoring is `+0.00000033656418753`
worse than the current DQS1 top32 gap-ULEB CPU frontier.

## Artifacts

- Selector packet plan: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_gap_uleb_selector_packet_plans_20260522/prefix_k028.json`
- Candidate archive: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_prefix_k028/submission_dir/archive.zip`
- Candidate archive SHA-256: `9da13be0a0eac60d0aa22219c325f38e56d2534fa0045050f365615ded3f9c5a`
- Candidate archive bytes: `178556`
- Candidate member bytes: `178456`
- DQS1 payload bytes: `39`
- DQS1 payload SHA-256: `20f4d11ffb9d563ef4b65ba4c7c8b335f667096ff4383bbdc795faae287cdb95`
- Locality controls: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_prefix_k028/locality_controls_prefix_k028.json`
- Exact CPU result: `experiments/results/modal_auth_eval_cpu/dqs1_prefix_k028_gap_uleb_selective_decoderq_20260522T153210Z_cpu/contest_auth_eval.json`

## Exact CPU Result

- Axis: `[contest-CPU]`
- Samples: `600`
- Score: `0.1920292853802774`
- Seg contribution: `0.055981`
- Pose contribution: `0.017155174146594957`
- Rate contribution: `0.11889311123368243`
- Avg SegNet dist: `0.00055981`
- Avg PoseNet dist: `0.00002943`

Comparison:

- FEC6 base CPU anchor: `0.1920513168811056`
- Current DQS1 top32 gap-ULEB CPU frontier: `0.19202894881608987`
- `prefix_k028` delta vs FEC6 base: `-0.00002203150082820`
- `prefix_k028` delta vs DQS1 top32 frontier: `+0.00000033656418753`
- `prefix_k028` delta vs `drop_rank032_pair0520`: `+0.00000000242314067`

## Interpretation

This confirms the selector curve around the current DQS1 optimum is rate-limited
but not byte-limited enough for smaller pair sets to beat top32. Removing four
selected pairs saves three archive bytes relative to top32, but loses enough
SegNet/PoseNet component gain to land essentially tied with the 31-pair
drop-one calibration and still behind top32.

The correct next DQS1 selector work is not more prefix/drop-one replay. Higher
EV options are:

- generate new pair-set families from scorer-response or per-frame
  decomposition rather than rank prefixes;
- train/update the MLX scorer-response surrogate on exact CPU calibration
  points;
- use the new Pareto-frontier metadata to select exact-replay batches only when
  a candidate changes the selector family, not merely the same prefix curve.
