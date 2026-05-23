# Codex Findings: Inverse Scorer Cell Materializer

UTC: 2026-05-23T18:22:35Z
Agent: Codex
Lane: lane_inverse_scorer_cell_candidate_materializer_20260523

## Scope

Adversarial review and implementation pass for turning inverse-scorer action
cells into deterministic byte-closed candidate packets without granting score
authority.

## Findings

1. The queue could plan inverse-scorer cells but had no deterministic candidate
   materializer. Fix landed: `inverse_scorer_cell_candidate_adapter` now emits
   `inverse_scorer_cell_candidate_v1` by appending an IAS1 descriptor packet to
   a strict single-member archive template.

2. A descriptor-only packet is not a valid runtime result unless inflate code
   proves it consumes the packet. Fix landed: every plan, manifest, and receiver
   verification stays fail-closed with `score_claim=false`,
   `promotion_eligible=false`, `rank_or_kill_eligible=false`, and
   `ready_for_exact_eval_dispatch=false` until receiver proof, inflate parity,
   and exact auth eval exist.

3. The byte-shaving materializer registry advertised a cell materializer but
   pointed at the action-functional acquisition module. Fix landed: registry
   ownership now points at `tac.optimization.inverse_scorer_cell_materializer`
   and declares the concrete output archive and manifest context fields.

4. The materializer work queue could not execute inverse cell rows. Fix landed:
   `scorer_inverse_surface_cell` + `materialize_inverse_scorer_cell_candidate`
   rows compile to `tools/materialize_inverse_scorer_cell_candidate.py` with
   JSON-schema postconditions and explicit dispatch blockers.

5. The scheduler claim path could mutate stale ready-step snapshots when queue
   control, definition hashes, or resource capacity changed after `ready_steps`
   was computed. Fix landed: ready-step execution now rechecks queue mode,
   definition/command/postcondition hashes, resource kind, and per-resource
   running counts inside the claim transaction before starting a process.

6. Staircase DAG task specs did not carry experiment-metadata-aware step hashes
   through the writeback contract. Fix landed: DAG construction now hashes
   steps with experiment metadata and dispatch specs expose the same hash packet
   the executor must claim against.

## Empirical Smoke

- Real archive template:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/selective_runtime_candidate_append_tail_top32/submission_dir/archive.zip`
- Real inverse action functional:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_steganalysis_action_functional_20260523_codex.json`
- Emitted candidate manifest:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/inverse_scorer_cell_candidate_ias1_20260523_codex/inverse_scorer_cell_candidate_manifest.json`
- Candidate archive SHA-256:
  `d230a079918d633ad74b2e2edc873ac9f6b1de8cbc871b1ce47ff4d5527cae40`
- Selected cells: 4
- IAS1 packet bytes: 2615
- Receiver contract: false
- Readiness blockers:
  `runtime_consumption_proof_missing`,
  `inverse_scorer_cell_receiver_contract_not_satisfied`,
  `candidate_inflate_output_parity_missing`,
  `exact_auth_eval_required_before_score_claim`

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_byte_shaving_campaign_queue.py`
- `.venv/bin/python -m pytest -q src/tac/tests/test_experiment_queue.py src/tac/tests/test_experiment_queue_observer.py src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_scorer_cell_materializer.py src/tac/tests/test_staircase_dag.py` (`150 passed`)
- `.venv/bin/python -m ruff check ...`
- `.venv/bin/python -m py_compile ...`

## Remaining Gaps

1. Build the actual inflate/runtime adapter that consumes IAS1 descriptors and
   emits `inverse_scorer_cell_runtime_adapter_v1`.
2. Compare full-frame inflate output before and after IAS1 consumption.
3. Run exact CPU/CUDA auth eval only after receiver proof and parity are present.
4. Feed measured materializer execution timing back into acquisition as queue
   performance signal.
