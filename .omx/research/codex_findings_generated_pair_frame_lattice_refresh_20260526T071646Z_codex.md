# Generated Pair-Frame Lattice Refresh

UTC: 2026-05-26T07:16:46Z

## Verdict

The feedback refresh can now compile an existing `decoder_q_pairset_acquisition.v1`
directly into a `pair_frame_scorer_geometry_lattice.v1` and feed the generated
queue-executable DQS1 starts into the local follow-up queue. This closes the
manual handoff between eureka/drop-many pairset planning and queue-owned
pair/frame exploration.

Authority remains false. The generated lattice, refresh report, DQS1 queue,
operation-chain queue, receiver-repair queue, and materialized proof artifacts
all remain planning/local-only and carry no score, promotion, rank/kill, or
exact-dispatch authority.

## Implementation

- `tools/build_frontier_rate_attack_feedback_refresh.py` accepts
  `--pair-frame-pairset-acquisition`, optional `--pair-frame-curriculum`,
  repeated `--pair-component-xray`, `--pair-frame-drop-counts`, and
  `--pair-frame-max-requests`.
- The tool writes `pair_frame_scorer_geometry_lattice.json` into the refresh
  output directory, wires it through existing pair-frame geometry discovery, and
  exposes a false-authority generated-lattice summary in CLI/report output.
- `src/comma_lab/scheduler/dqs1_local_first_queue.py` now returns validated
  partial selections when fewer than `candidate_limit` safe generated requests
  exist, instead of dropping valid queue starts and raising a false fatal.
- `src/tac/tests/test_frontier_rate_attack_refresh_pair_frame_cli.py` covers
  inline lattice generation, queue wiring, artifact discoverability, and
  false-authority preservation.

## Real Artifact

Refresh output:
`.omx/research/frontier_rate_attack_feedback_refresh_20260526T071254Z_generated_pair_frame_lattice/`

Inputs:

- Pairset acquisition:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/dqs1_pairset_acquisition_eureka_drop_many.json`
- Action summary:
  `.omx/research/codex_eureka_beyond_drop_two_acquisition_20260525T143351Z/action_summary.json`
- Pair component xray:
  `experiments/results/d1_pair_component_xray_a1_baseline_cpu_20260515_codex/pair_component_xray.json`

Generated queue starts:

- `pairset_geometry_lowimpact_k003_h26140419e5` drops `[376, 378, 555]`
- `pairset_geometry_lowimpact_k004_h8b2ce76737` drops `[376, 378, 440, 555]`
- `pairset_geometry_lowimpact_k006_h2866d55ee6` drops `[371, 376, 378, 440, 479, 555]`
- `pairset_geometry_lowimpact_k008_h5122a378ca` drops `[242, 371, 376, 378, 440, 479, 555, 588]`
- `pairset_geometry_lowimpact_k012_h4bdbac6259` drops `[242, 259, 296, 371, 376, 378, 430, 440, 479, 544, 555, 588]`
- `pairset_geometry_lowimpact_k016_h33130b677c` drops `[229, 242, 259, 296, 371, 376, 378, 412, 430, 440, 459, 467, 479, 544, 555, 588]`

The generated lattice had 32 rows, 1.0 geometry coverage over the selected
source-pair universe, and six queue-executable requests.

## Bounded Execution

Initialized and ran the generated DQS1 queue for one candidate with:

```bash
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_rate_attack_feedback_refresh_20260526T071254Z_generated_pair_frame_lattice/dqs1_followup_queue.json init
.venv/bin/python tools/experiment_queue.py --queue .omx/research/frontier_rate_attack_feedback_refresh_20260526T071254Z_generated_pair_frame_lattice/dqs1_followup_queue.json run-worker --execute --max-steps 4 --max-experiments 1 --max-parallel 1
```

Result: 4 succeeded, 0 failed, stopped at `max_steps_reached` before local CPU
advisory. The executed candidate was `pairset_geometry_lowimpact_k003_h26140419e5`.

Materialization facts:

- Candidate archive:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/geometry_lowimpact_k003_h26140419e5/submission_dir/archive.zip`
- Base archive bytes: 178517
- Candidate archive bytes: 178557
- Delta versus this PR101 base archive: +40 bytes
- DQS1 payload bytes: 40
- Selected pair count: 29
- Locality controls passed: true
- Locality mismatches: selected frames 0, unselected frames 0, raw-size 0,
  missing raw files 0

This is therefore a runtime-closed probe and correction-budget signal, not a
byte-save claim. The next score-relevant step is local CPU/MLX response harvest
or a more compressed receiver/materializer variant that removes the DQS1 payload
overhead.

## Verification

- `ruff check src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_frontier_rate_attack_feedback_refresh.py src/tac/tests/test_frontier_rate_attack_refresh_pair_frame_cli.py`
- `pytest src/tac/tests/test_frontier_rate_attack_refresh_pair_frame_cli.py -q`
- `pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice.py src/tac/tests/test_frontier_rate_attack_refresh_pair_frame_cli.py -q`
- `pytest src/tac/tests/test_frontier_rate_attack_feedback.py::test_frontier_feedback_compiler_promotes_pair_frame_geometry_requests_to_queue src/tac/tests/test_dqs1_local_first_queue_builder.py -q`
- `experiment_queue.py validate` passed for generated DQS1 follow-up queue,
  operation-chain compiler queue, and receiver-repair queue.

## Remaining Work

- Execute local CPU/MLX response harvest on the generated k-band candidates and
  canonicalize deltas into the targeted correction acquisition surface.
- Replace payload-positive DQS1 starts with receiver/materializer variants that
  make the rate budget real at archive level.
- Add executable mask/feather/within-set correction materializers so the freed
  rate budget can be spent on SegNet/PoseNet repair rather than only queued as
  advisory geometry.
