# Codex Findings - Materializer Receiver-Negative Feedback

Timestamp: 2026-05-25T09:24:15Z
Agent: Codex
Lane: `codex_archive_section_receiver_safe_signal_20260525`

## Finding

Receiver-negative, byte-closed family-agnostic materializer artifacts were being
blocked correctly for score/promotion authority, but the empirical signal was
too easy to lose inside failed queue artifacts. This is the wrong failure shape
for inverse-steganalysis automation: a receiver-negative archive is not a
candidate, but it is a useful observation about the materializer/scorer/runtime
surface.

## Landing

- Preserved receiver/materializer metadata in queue observation artifact records,
  including readiness blockers, receiver verification blockers, runtime
  consumption proof paths, and candidate archive byte metadata.
- Promoted receiver-negative materializer outputs into explicit queue-health and
  inverse-steganalysis blockers so water-bucket acquisition suppresses matching
  cells until the receiver contract is repaired.
- Changed queue feedback recovery planning so byte-closed receiver-negative
  materializer artifacts produce `record_materializer_receiver_feedback`
  maintenance actions instead of blind rewind work.
- Added campaign-runner synthesis of
  `family_agnostic_materializer_empirical_sweep.v1` observations from
  receiver-negative manifests under `materializer_outputs/`, then wired the
  generated sweep into `feedback_observation_paths`.
- Preflight caught and fixed an undefined helper in the acquisition merge-key
  fallback before commit; the module now uses a local deterministic JSON SHA-256
  helper for receiver-negative observation merge identity.

## Authority Contract

Generated receiver-negative observation rows are explicitly false authority:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

The allowed use is local materializer receiver feedback for replanning only. The
forbidden use remains score claim, promotion, rank/kill, or paid dispatch
authority.

## Verification

Local queue-owned materializer proof artifact:

- Compact tracked summary:
  `.omx/research/codex_receiver_negative_queue_feedback_20260525T092415Z_summary.json`
- `.omx/research/codex_receiver_negative_queue_feedback_20260525T094050Z/materializer_receiver_feedback_observations.json`
- Schema: `family_agnostic_materializer_empirical_sweep.v1`
- Observation count: 1 merged receiver-negative row
- Preserved rate signal: 66 saved bytes, rate-positive, receiver-negative
- Allowed feedback path:
  `.omx/research/codex_receiver_negative_queue_feedback_20260525T094050Z/queue_feedback_replan_request.json`
- The feedback request includes the generated sweep as `--observation` and
  remains false authority.
- Executing that feedback request wrote
  `.omx/research/codex_receiver_negative_queue_feedback_20260525T094050Z/inverse_steganalysis_action_functional.feedback.json`
  with 1 cell, 1 blocked cell, 0 selected cells, 66 realized saved bytes, and
  false authority. The archive-delta blocker is
  `receiver_negative_materializer_success`, not rate-negative feedback.

```bash
.venv/bin/python -m ruff check \
  src/comma_lab/scheduler/experiment_queue_observer.py \
  src/comma_lab/scheduler/queue_feedback_replan_policy.py \
  src/tac/optimization/inverse_steganalysis_acquisition.py \
  tools/run_byte_shaving_materializer_campaign.py \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_queue_feedback_replan_policy.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py
```

Result: passed.

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_emits_receiver_negative_observation_sweep \
  src/tac/tests/test_experiment_queue_observer.py::test_observer_preserves_materializer_metadata_for_recovery_grouping \
  src/tac/tests/test_inverse_steganalysis_acquisition.py::test_queue_observation_health_blocker_suppresses_water_bucket_cell \
  -q
```

Result: 3 passed.

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  -q
```

Result: 108 passed.

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_experiment_queue_observer.py \
  src/tac/tests/test_queue_feedback_replan_policy.py \
  src/tac/tests/test_inverse_steganalysis_acquisition.py \
  src/tac/tests/test_byte_shaving_materializer_campaign_runner.py \
  -q
```

Result: 147 passed after the empirical-delta merge fix.

## Next Integration

The next highest-value step is to use the now-proven receiver-negative feedback
path as the repair/demotion input for the next queue-owned materializer campaign,
then confirm the follow-up queue selects a repaired target instead of refilling
the blocked bucket. The same observation path should also be generalized to
`packet_member_recompress_v1` and `tensor_factorize_v1` so failed receiver
contracts become reusable optimization signal across HNeRV, HNeRV bolt-ons,
NeRV-family, and non-NeRV archives.
