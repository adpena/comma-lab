# Packet Header Elide All-Member Feedback Landing - Codex - 2026-05-24T23:18:55Z

## Scope

This landing extends the packet-member ZIP header elision materializer from a
single selected member to an explicit multi-member/all-member mode, and wires
family-agnostic materializer observations into the inverse-steganalysis feedback
surface without requiring manual JSON path threading when observations already
exist under the campaign run directory.

## Changes

- `materialize_packet_member_zip_header_elide_candidate(...)` now accepts
  `member_names` and `all_members`.
- The raw ZIP writer strips selected member extras/comments across multiple
  members while preserving each selected member payload and compressed stream
  length.
- Runtime-consumption proof now records per-member payload identity,
  compressed-stream SHA-256 identity, and `candidate_member_sha256s`; verification
  can require a map of member hashes instead of only one primary member hash.
- `tools/run_family_agnostic_materializer.py` and
  `tools/run_family_agnostic_materializer_sweep.py` expose `--all-members`.
- Final-byte context and materializer queue command generation propagate
  `member_selection=all` / `all_members=true` to the materializer command.
- `tools/build_inverse_steganalysis_action_functional.py` reads observation
  JSONL files in addition to JSON containers.
- `tools/run_byte_shaving_materializer_campaign.py` auto-discovers
  `family_agnostic_materializer_empirical_observation.v1` JSON/JSONL files
  under the run directory and passes them into the feedback action-functional
  command.

## Real-Archive Anchor

Command:

```bash
.venv/bin/python tools/run_family_agnostic_materializer_sweep.py \
  --archive robust_current_archive_correct=submissions/robust_current/archive_correct.zip \
  --all-members \
  --output-dir experiments/results/packet_header_elide_all_members_20260524T232046Z \
  --output-json experiments/results/packet_header_elide_all_members_20260524T232046Z/sweep.json \
  --observation-jsonl experiments/results/packet_header_elide_all_members_20260524T232046Z/observations.jsonl
```

Result:

- Source archive SHA-256:
  `4dd46fed78ed064bc97c9b3205088e82838c03667394f7936c8ae8d20f9837ab`
- Candidate archive SHA-256:
  `544c5f580ec261f6c8f80643ed7dee2e122f533c06c96a54c28624f9db75ce50`
- Source bytes: `345802`
- Candidate bytes: `345646`
- Saved bytes: `156`
- Selected members: `renderer.bin`, `masks.mkv`, `optimized_poses.pt`
- Elided header bytes reported by the selected-member summary: `72`
- Receiver contract satisfied: `true`
- Score/promotion/rank/dispatch authority: all `false`

This supersedes the prior one-member `renderer.bin` local rate proof for this
archive class as the stronger local materializer-proof signal. It remains a
local byte-closed/runtime-consumption proof only; exact inflate parity and
contest auth eval are still required before any score or dispatch authority.

## Verification

- `.venv/bin/ruff check ...` on the touched implementation and test files:
  passed.
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q`:
  `191 passed`.

## Remaining Follow-Up

- Promote packet-member reorder onto the same raw-stream-preserving ZIP writer
  so it cannot regress by recompressing payloads.
- Let queue feedback policy consume the newly auto-discovered materializer
  observations in a bounded one-step local autopolicy path where all authority
  fields remain false.
- Add exact-readiness handoff policy edges that dry-run the shared dispatch
  audit before any exact queue child is emitted.
