# Codex Findings: Inverse Scorer Cell Receiver Chain

UTC: 2026-05-23T18:39:07Z
Agent: Codex
Lane: lane_inverse_scorer_cell_receiver_chain_20260523

## Scope

Follow-up implementation after the IAS1 candidate materializer landing. The
goal was to consume the emitted IAS1 descriptor bytes through a deterministic
receiver path, build the existing receiver-proof schema from that observation,
and wire the local proof chain into the materializer queue/DAG surface without
granting score authority.

## Findings

1. The receiver proof builder had a missing producer. Fix landed:
   `build_inverse_scorer_cell_runtime_adapter_manifest(...)` opens the candidate
   archive, consumes the IAS1 packet at the manifest-recorded offset, verifies
   packet SHA, descriptor JSON SHA, raw-video digest, atom ids, selected cell
   count, packet-at-member-tail, archive SHA, and member SHA, then emits
   `inverse_scorer_cell_runtime_adapter_v1`.

2. Receiver-proof authority was too permissive for hand-authored adapter
   manifests. Fix landed: proof construction now rejects truthy authority
   fields, adapter blockers, missing runtime-tree identity, promotion/rank
   authority, dispatch attempts, archive SHA mismatch, and member SHA mismatch.

3. Candidate verification kept stale receiver blockers even after a valid proof.
   Fix landed: `verify_inverse_scorer_cell_candidate_manifest(...)` removes
   receiver-proof blockers only when receiver verification and archive custody
   both pass; inflate parity and exact-auth blockers remain.

4. The queue could execute a descriptor-only materializer row but not the full
   receiver chain. Fix landed: inverse-scorer cell rows can now run
   `tools/run_inverse_scorer_cell_candidate_chain.py` when context supplies
   `output_dir` or `chain_output_dir`; the postcondition checks
   `inverse_scorer_cell_candidate_chain_v1`.

5. Adversarial review caught a false-readiness naming bug. Fix landed:
   descriptor receiver verification now produces
   `ready_for_receiver_verification=true` while keeping
   `ready_for_exact_eval_runtime=false`; exact runtime/full-frame parity remains
   explicitly blocked until an IAS1-aware inflate/runtime path is proven.

6. DAG/autopilot observability was dropping materializer metadata. Fix landed:
   staircase DAG and Dask task specs now carry materializer id, target kind,
   receiver contract, experiment metadata, timeout, and recursive artifact
   telemetry from materializer execution queues.

## Real Smoke

- Command:
  `.venv/bin/python tools/run_inverse_scorer_cell_candidate_chain.py --candidate-archive-template experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/submission_dir/archive.zip --inverse-action-functional experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_steganalysis_action_functional_20260523_codex.json --raw-contest-video-digest 7ea7b4565d623155b4d022787840839affa9c1b83253bbb8524e49d5dcc64bdb --output-dir experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_cell_chain_ias1_receiver_split_20260523_codex --selected-limit 4 --min-free-bytes 1048576 --fail-if-receiver-blocked`
- Chain manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_cell_chain_ias1_receiver_split_20260523_codex/inverse_scorer_cell_candidate_chain_manifest.json`
- Candidate archive SHA-256:
  `d230a079918d633ad74b2e2edc873ac9f6b1de8cbc871b1ce47ff4d5527cae40`
- Runtime adapter descriptor receiver ready: true
- Receiver proof ready for receiver verification: true
- Ready for exact-eval runtime: false
- Receiver contract satisfied: true
- Runtime adapter blocker cleared: true
- Remaining blockers:
  `candidate_inflate_output_parity_missing`,
  `exact_auth_eval_required_before_score_claim`

## Queue Smoke And Acquisition Feedback

- Queue artifact:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_cell_chain_queue_20260523_codex/`
- Worker result:
  `steps_started=1`, `success_count=1`, `failure_count=0`,
  `claim_refused_count=0`, `status_counts={"succeeded": 1}`
- Performance summary:
  `event_count=1`, `elapsed_seconds_mean=0.2615928339655511`,
  `artifact_record_bytes_mean=211520`, `dominant_resource_kind=local_mlx`
- Calibrated action functional:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_steganalysis_action_functional_queue_calibrated_20260523_codex.json`
- Feedback integration:
  queue-performance candidate-map expansion now lets one bundled proof-chain
  step attach timing/artifact denominator observations to every selected
  inverse-scorer candidate id, while preserving false authority.
- Calibrated functional to queue/DAG smoke:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_action_functional_queue_calibrated_materialization_20260523_codex/execution_queue.json`
  ran through the shared worker with `steps_started=1`, `success_count=1`,
  `failure_count=0`, `elapsed_seconds_mean=0.2610312919714488`, and a passing
  `json_equals` postcondition on
  `inverse_scorer_cell_candidate_chain_v1`.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_staircase_dag.py` (`68 passed`)
- `.venv/bin/python -m pytest -q src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_staircase_dag.py` (`156 passed`)
- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_staircase_dag.py` (`162 passed`)
- `.venv/bin/python -m ruff check ...`
- `.venv/bin/python -m py_compile ...`
- `git diff --check`

## Remaining Gaps

1. Add full-frame inflate parity for IAS1-aware runtime behavior.
2. Route only parity-proven candidates toward exact CPU/CUDA auth-eval dispatch.
3. Teach the queue compiler to emit per-cell proof-chain queue experiments when
   bundle amortization stops being the right cost model.
