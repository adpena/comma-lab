# PacketIR Exact-Eval Closure

- lane_id: `pr106_r2_packetir_pr101_grammar_lowlevel_closure`
- classification: `exact_measured_not_current_frontier`
- score_claim: `false`
- promotion_eligible: `false`
- ready_for_exact_eval_dispatch: `false`

## Archive

- source_sha256: `c48631e11a9bb18d051da9100ca4d5773558a8a81ac38dc8f6f4e8b6119d0383`
- candidate_sha256: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- candidate_bytes: `186629`
- byte_delta_vs_source: `-151`
- rate_delta_if_components_equal: `-0.00010054470192144788`

## Axes

- [contest-CUDA] score: `0.2065174760196528` bytes: `186629` sha: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- [contest-CPU] score: `0.22796397327358284` bytes: `186629` sha: `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`

## Comparisons

- delta_vs_packetir_source_cuda: `-0.00010065943776227382`
- delta_vs_current_best_cuda: `0.0001371669443431811`
- not_current_frontier: `true`

## Blockers

- none

## Duplicate Dispatch Blockers

- `same_candidate_archive_already_exact_evaluated`
- `candidate_not_current_frontier_on_contest_cuda`

## Warnings

- `candidate_exact_cuda_closed_but_not_current_frontier`: candidate improved its PacketIR source line but is worse than the current exact-CUDA reference
