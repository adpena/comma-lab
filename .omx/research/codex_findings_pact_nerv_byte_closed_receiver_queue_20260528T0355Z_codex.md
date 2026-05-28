# Codex Findings: Pact-NeRV IA3 Byte-Closed Receiver Queue

UTC: 2026-05-28T03:55Z
Reviewer: codex

## Landing

Converted the Pact-NeRV IA3 MLX/PyTorch parity row into a byte-closed PIA3
candidate artifact:

- new reusable materializer: `tac.substrates.pact_nerv_ia3.archive_candidate`
- new operator CLI: `tools/materialize_pact_nerv_ia3_byte_closed_candidate.py`
- queue-owned child step: `materialize_pact_nerv_ia3_byte_closed_candidate`
- replay-bundle signal preservation for `byte_closed_receiver_proof_present`

The receiver side remains deterministic inflate-only. MLX remains advisory and
has no score, promotion, rank/kill, or exact-dispatch authority.

## Executed Queue Evidence

Artifact root:
`.omx/research/pact_nerv_diffusion_blocks_byte_closed_smoke2_20260528Tlocal/`

Queue status:

- `step_count`: 7
- `status_counts`: `{"succeeded": 7}`
- `ready_steps`: `[]`
- `orphaned_step_count`: 0

Byte-closed candidate:

- archive: `.omx/research/pact_nerv_diffusion_blocks_byte_closed_smoke2_20260528Tlocal/outputs/pact_nerv_ia3_byte_closed_candidate/archive.zip`
- archive bytes: 26235
- archive sha256: `0b5d9fe33c8e082387f921471d805db971c8cf242ca1b0d6754d63704160a71f`
- receiver proof sha256: `5f9157bdc64727b7c2345a8e5cf2a4f850a1a6db0061a154ab2ee12726a7f0a0`
- `byte_closed_candidate_emitted`: true
- `receiver_contract_satisfied`: true
- `full_frame_inflate_parity_satisfied`: true

Replay bundle now preserves the proof signal:

- `local_replay_ready`: true
- `byte_closed_receiver_proof_present`: true
- stale byte-closed/proof blocker removed
- remaining blockers: MLX advisory authority, contest CPU/CUDA exact eval, redacted secret env

## Verification

- `ruff check` on touched files: passed
- `pytest src/tac/tests/test_mlx_replay_bundle.py src/tac/tests/test_pact_nerv_ia3_archive_candidate.py src/tac/tests/test_pact_nerv_diffusion_blocks_queue.py src/tac/tests/test_mlx_to_pytorch_export.py -q`: 12 passed
- queue validation: 7 steps valid
- queue execution: 7 succeeded
- review-tracker policy checks on touched code/tests: 0 violations after three adversarial passes

## Residual Authority

This is exact-ready structure, not a score claim. Promotion still requires
contest CPU/CUDA exact eval on a full contest payload with canonical custody.
