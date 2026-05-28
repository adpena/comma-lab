# Codex Findings: Pact-NeRV Runtime-Custody Byte-Closed Queue

UTC: 2026-05-28T04:02Z
Reviewer: codex

## Landing

Hardened the Pact-NeRV IA3 byte-closed materializer and replay bundle so
archive custody, receiver proof, and runtime-tree custody are all visible and
validated as separate facts.

Changes:

- packaged receiver runs with `PYTHONDONTWRITEBYTECODE=1` so runtime proof does
  not mutate the vendored runtime tree with `__pycache__`
- materializer emits runtime adapter directory, observed tree SHA, expected tree
  SHA, and runtime-consumption proof SHA
- queue postconditions require runtime adapter custody, proof path, proof SHA,
  archive path, and receiver proof path
- MLX replay bundle preserves both `byte_closed_receiver_proof_present` and
  `runtime_custody_present`

## Executed Queue Evidence

Artifact root:
`.omx/research/pact_nerv_diffusion_blocks_runtime_custody_smoke3_20260528Tlocal/`

Queue status:

- `step_count`: 7
- `status_counts`: `{"succeeded": 7}`
- `ready_steps`: `[]`
- `orphaned_step_count`: 0

Byte-closed candidate:

- archive: `.omx/research/pact_nerv_diffusion_blocks_runtime_custody_smoke3_20260528Tlocal/outputs/pact_nerv_ia3_byte_closed_candidate/archive.zip`
- archive bytes: 26259
- archive sha256: `ea08242904cadfcec8682a29a037c06cc6f2bf806ca9b68dab4b79d39f114b49`
- runtime tree sha256: `475432eb7b6afa3da91e86961c569353ead6a24a8ed8f9169ea69678ee82ff78`
- receiver proof sha256: `dbeb61e0891f7f5fb25933fd3a1b391817cb38ce3d006d6bbdaa965d522ae447`
- `byte_closed_candidate_emitted`: true
- `receiver_contract_satisfied`: true
- `full_frame_inflate_parity_satisfied`: true
- `runtime_adapter_ready`: true
- `candidate_runtime_adapter_blocker_cleared`: true

Replay bundle:

- `local_replay_ready`: true
- `byte_closed_receiver_proof_present`: true
- `runtime_custody_present`: true
- remaining blockers: MLX advisory authority, contest CPU/CUDA exact eval, redacted secret env

## Verification

- `ruff check` on touched files: passed
- `pytest src/tac/tests/test_mlx_replay_bundle.py src/tac/tests/test_pact_nerv_ia3_archive_candidate.py src/tac/tests/test_pact_nerv_diffusion_blocks_queue.py src/tac/tests/test_mlx_to_pytorch_export.py -q`: 13 passed
- queue validation: 7 steps valid
- queue execution: 7 succeeded

## Residual Authority

This is still `[macOS-MLX research-signal]` structure. It is local replay and
exact-ready custody evidence, not a contest score or promotion.
