# Codex Findings: DQS1 Selector Pareto Exact CPU Replay

## Verdict

`drop_rank032_pair0520` is exact-eval falsified as a new `[contest-CPU]`
frontier. It is byte-closed and locality-clean, but exact CPU scoring regressed
by `+0.00000033414104686` versus the current DQS1 top32 gap-ULEB CPU frontier.

## Artifacts

- Selector plan JSON: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_gap_uleb_selector_pareto_20260522.json`
- Selector plan Markdown: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/dqs1_gap_uleb_selector_pareto_20260522.md`
- Candidate archive: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_drop_rank032_pair0520/submission_dir/archive.zip`
- Candidate archive SHA-256: `924d21681a9b9f7b8022dfc036af3b63db30b954faf0ae26664943a093d8bde6`
- Candidate archive bytes: `178559`
- Candidate member bytes: `178459`
- DQS1 payload bytes: `42`
- DQS1 payload SHA-256: `feb4544f22c4ec1f10b3ee0b34998509808f390cfbc93dbf462637616aedd1d0`
- Locality controls: `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_drop_rank032_pair0520/locality_controls_drop_rank032_pair0520.json`
- Exact CPU result: `experiments/results/modal_auth_eval_cpu/dqs1_drop_rank032_pair0520_gap_uleb_selective_decoderq_20260522T151723Z_cpu/contest_auth_eval.json`

## Exact CPU Result

- Axis: `[contest-CPU]`
- Samples: `600`
- Score: `0.19202928295713673`
- Seg contribution: `0.055979`
- Pose contribution: `0.017155174146594957`
- Rate contribution: `0.11889510881054179`
- Avg SegNet dist: `0.00055979`
- Avg PoseNet dist: `0.00002943`

Comparison:

- FEC6 base CPU anchor: `0.1920513168811056`
- Current DQS1 top32 gap-ULEB CPU frontier: `0.19202894881608987`
- `drop_rank032_pair0520` delta vs FEC6 base: `-0.00002203392396887`
- `drop_rank032_pair0520` delta vs DQS1 top32 frontier: `+0.00000033414104686`

## Implementation Corrections Landed Before Commit

The adversarial review found no critical or high severity issues. Medium issues
were fixed before commit:

- Explicit `selected_pair_indices` now fail closed outside the canonical FEC6
  `0..599` pair range.
- Selector bridge `pair_window` parsing now rejects non-integral values instead
  of silently coercing floats.
- The selector plan now records true dominance metadata:
  `selector_rank`, `pareto_frontier`, `pareto_rank`, and
  `dominated_by_selector_id`. Emitted packet-plan summaries carry the full
  false-authority field set.

## Selector Plan State

The regenerated selector plan has:

- Candidate count: `73`
- Pareto-frontier candidate count: `12`
- Prefix candidate count: `10`
- Singleton candidate count: `31`
- Drop-one candidate count: `32`
- Recommended selector by calibrated non-authoritative planner:
  `prefix_k032`

`drop_rank032_pair0520` is the best 42-byte frontier point, but exact CPU
settled it as slightly worse than `prefix_k032`. The correct next selector
work is not more drop-one exact replay unless seeking robustness; higher-EV
work is to use this frontier/dominance surface for new pair-set families or for
the MLX scorer-response training queue.

## Verification

- `ruff check src/tac/optimization/decoder_q_selective_selector_pareto.py tools/plan_decoder_q_selective_selector_pareto.py src/tac/tests/test_decoder_q_selective_selector_pareto.py src/tac/optimization/decoder_q_selective_runtime_packet.py tools/plan_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_packet.py`
- `.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_selector_pareto.py src/tac/tests/test_decoder_q_selective_runtime_packet.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_decoder_q_selective_selector_pareto.py src/tac/tests/test_decoder_q_selective_runtime_packet.py src/tac/tests/test_decoder_q_selective_runtime_materializer.py src/tac/tests/test_decoder_q_selective_runtime_controls.py src/tac/tests/test_decoder_q_selective_runtime_feedback.py -q`
