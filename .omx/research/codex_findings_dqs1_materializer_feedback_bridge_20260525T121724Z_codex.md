# Codex Findings: DQS1 Materializer Feedback Bridge

## Summary

Generic family-agnostic ZIP/member materializers are now converted into an
explicit DQS1 follow-up bridge before re-entering the DQS1 local-first queue.
This prevents raw `family_agnostic_materializer_empirical_observation.v1` rows
from becoming executable DQS1 work directly, while preserving their signal as
false-authority replanning context.

## Landed Surfaces

- `src/tac/optimization/dqs1_materializer_feedback_bridge.py`
  - groups materializer feedback by `target_kind`;
  - records receiver-positive/no-delta, receiver-negative, and rate-positive
    buckets;
  - emits `dqs1_materializer_feedback_bridge.v1` with false-authority fields;
  - recommends DQS1 pairset-composition follow-up when generic materializers
    are receiver-safe but saturated.
- `src/comma_lab/scheduler/dqs1_local_first_queue.py`
  - accepts materializer feedback payloads;
  - stamps the derived bridge into every emitted DQS1 experiment metadata row.
- `tools/build_dqs1_local_first_queue.py`
  - accepts repeated `--materializer-feedback` inputs;
  - can write `--materializer-feedback-bridge-out`.
- `tools/run_byte_shaving_materializer_campaign.py`
  - now counts canonical materializer sweep artifacts in the dynamic-sparse
    receiver gate, so positive receiver-safe rows can feed compiler hints while
    zero-byte rows stay demotion/replan signal.

## Live Artifact

- Bridge:
  `.omx/research/codex_dqs1_materializer_feedback_bridge_20260525T121724Z/dqs1_materializer_feedback_bridge.json`
- Queue:
  `.omx/research/codex_dqs1_materializer_feedback_bridge_20260525T121724Z/dqs1_pairset_local_first.materializer_feedback.json`

The live bridge consumed the current CPU-frontier receiver smoke:

- `experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/header_elide/sweep.json`
- `experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/recompress/sweep.json`

It found two materializer observations, both receiver-safe but zero-byte/no-delta:

- `packet_member_recompress_v1`
- `packet_member_zip_header_elide_v1`

The derived queue is valid and selects the next four DQS1 pairset-composition
candidates:

- `pairset_drop_two_r010_001_p0376_0501`
- `pairset_drop_two_r009_005_p0459_0467`
- `pairset_drop_two_r009_003_p0459_0479`
- `pairset_drop_two_r005_003_p0467_0479`

## Verification

- `.venv/bin/ruff check tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/optimization/dqs1_materializer_feedback_bridge.py src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py src/tac/tests/test_dqs1_local_first_queue_builder.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_counts_receiver_positive_sweep_artifact_rows src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_emits_receiver_gated_dynamic_sparse_hint src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_blocks_dynamic_sparse_hint_without_receiver src/tac/tests/test_dqs1_local_first_queue_builder.py -q --durations=30 --durations-min=0.01`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_materializer_feedback_bridge_20260525T121724Z/dqs1_pairset_local_first.materializer_feedback.json validate`

## Next

Run the generated four-candidate local-first queue under the storage-preflight
policy, then canonicalize the resulting DQS1 local CPU/MLX harvests back into
the materializer/DQS1 feedback bridge. If the next DQS1 candidates stay flat,
the bridge should widen pairset composition rather than re-testing saturated
archive/member transforms.
