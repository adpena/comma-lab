# Codex Findings: Pact-NeRV IA3 Queue Observer Closure

- UTC: 2026-05-28T04:10Z
- Scope: Pact-NeRV IA3 MLX -> PyTorch bridge, byte-closed candidate materialization, queue-owned replay bundle, observer revalidation.
- Axis: `[macOS-MLX research-signal]`; no score, rank, promotion, or exact-eval authority claimed.

## Findings

1. The IA3 paired-dispatch recipe had drifted into `dispatch_enabled: true` despite the local work being advisory MLX/CPU smoke evidence. It is now fail-closed again with explicit dispatch blockers for exact auth eval, Catalog #325 symposium revalidation, and Catalog #167 smoke-before-full routing.
2. Queue postconditions for advisory JSON artifacts were too strong: `json_completion_contract` caused the observer to demand archive/runtime receiver custody from parity and replay-bundle reports. Those non-archive reports now use `json_equals` plus `json_false_authority`; only the byte-closed materializer keeps the full archive/runtime completion contract.
3. The byte-closed IA3 materializer and runtime custody path are proven in a queue-owned v4 run:
   - queue: `.omx/research/pact_nerv_diffusion_blocks_byte_closed_queue_20260528Tlocal_v4/queue.json`
   - worker: `.omx/research/pact_nerv_diffusion_blocks_byte_closed_queue_20260528Tlocal_v4/worker_result.json`
   - observer: `.omx/research/pact_nerv_diffusion_blocks_byte_closed_queue_20260528Tlocal_v4/observation.json`
   - result: 7/7 steps succeeded; observer `healthy=true`; blockers `[]`.

## Verification

- `.venv/bin/ruff check src/comma_lab/scheduler/pact_nerv_diffusion_blocks_queue.py src/tac/tests/test_pact_nerv_diffusion_blocks_queue.py tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py src/tac/substrates/pact_nerv_ia3/tests/test_pact_nerv_ia3_bridge_and_gate.py`
- `.venv/bin/python -m py_compile src/comma_lab/scheduler/pact_nerv_diffusion_blocks_queue.py src/tac/tests/test_pact_nerv_diffusion_blocks_queue.py tools/gate_mlx_candidate_contest_equivalence_pact_nerv_ia3.py src/tac/substrates/pact_nerv_ia3/tests/test_pact_nerv_ia3_bridge_and_gate.py`
- YAML parse check for `.omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_paired_dispatch.yaml`.
- `PYTHONPATH=. .venv/bin/pytest src/tac/substrates/pact_nerv_ia3/tests src/tac/tests/test_pact_nerv_diffusion_blocks_queue.py src/tac/tests/test_mlx_replay_bundle.py src/tac/tests/test_pact_nerv_ia3_archive_candidate.py -q` -> 33 passed.
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/pact_nerv_diffusion_blocks_byte_closed_queue_20260528Tlocal_v4/queue.json observe --output .omx/research/pact_nerv_diffusion_blocks_byte_closed_queue_20260528Tlocal_v4/observation.json` -> `healthy=true`, blockers `[]`.

## Next Hook

This closes the local queue-owned control-arm smoke. The next frontier-bearing step is to feed the same strict materializer/replay pattern into longer IA3/PR95-class MLX runs and then into exact CPU/CUDA only after an explicit dispatch turn and paired-axis claim.
