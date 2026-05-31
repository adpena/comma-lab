# Codex Findings - MLX Bridge Helper Consolidation

## Verdict

The current MLX bridge surface had a real orphaned-symbol drift: `tac.local_acceleration.mlx_to_pytorch_export` exported shared helper names in `__all__` before those helpers existed. That made the intended TAC bridge API non-durable and left six export tools carrying duplicate false-authority blockers and MLX dotted-parameter assignment code.

## Fix Landed

- Implemented canonical MLX bridge helper primitives in `tac.local_acceleration.mlx_to_pytorch_export`:
  - shared false-authority blockers;
  - file SHA helper;
  - dotted MLX parameter/buffer assignment with VQ private-buffer overrides;
  - MLX-HWIO to PyTorch-OIHW wrapper;
  - fail-closed forward-parity proof builder;
  - fail-closed substrate bridge manifest builder.
- Migrated the IA3, selector-v2, selector-v3, selector-v4, VQ, and Z6-v2 MLX export tools onto the shared blocker/hash/assignment helpers.
- Added regression tests that require every public bridge symbol to exist and require the export manifest, parity proof, and bridge manifest to remain score-authority false.

## Authority And Score Discipline

This landing does not claim a score, promote a candidate, or grant exact-dispatch authority. It only strengthens the local MLX-to-PyTorch bridge and keeps MLX outputs advisory until paired contest CPU/CUDA evidence and receiver/archive custody gates sign them.

## Verification

- `.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_to_pytorch_export.py src/tac/tests/test_mlx_to_pytorch_export.py tools/export_pact_nerv_ia3_mlx_to_pytorch_state_dict.py tools/export_pact_nerv_selector_v2_mlx_to_pytorch_state_dict.py tools/export_pact_nerv_selector_v3_mlx_to_pytorch_state_dict.py tools/export_pact_nerv_selector_v4_mlx_to_pytorch_state_dict.py tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py tools/export_z6_v2_cargo_cult_unwind_mlx_to_pytorch_state_dict.py`
- `.venv/bin/python -m pytest src/tac/tests/test_mlx_to_pytorch_export.py src/tac/substrates/pact_nerv_ia3/tests/test_pact_nerv_ia3_bridge_and_gate.py src/tac/framework_agnostic/tests/test_convert_mlx_state_dict_to_pytorch_oihw.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_archive_bound_runtime_bridge_remaining_mlx_emitters.py src/tac/tests/test_pr95_mlx_pytorch_archive_package.py src/tac/tests/test_archive_bound_candidate_adapter_spine.py -q`
- `.venv/bin/python tools/review_gate_hook.py`

## Remaining Work

The next cleanup should migrate manifest construction itself onto `build_substrate_bridge_manifest` where the substrate-specific config and operator route can be preserved without duplicating the false-authority envelope. Do that after a quick line-level review of each tool's unique schema fields so the consolidation does not erase substrate-distinguishing signal.
