# Codex Findings: Packet Header Elide All-Member Sweep

UTC: 2026-05-24T23:18:51Z
Lane: `codex_packet_header_elide_all_members_20260524`

## Finding

The singleton `packet_member_zip_header_elide_v1` materializer was correct but
left repeatable ZIP local-header overhead on multi-member archives. The real
`submissions/robust_current/archive_correct.zip` archive has three members with
the same removable extra-field pattern, so applying the operation to all members
tripled the deterministic rate savings.

This is still a local materializer proof-chain result. It is not a contest
auth-eval score claim and it is not promotion authority.

## Custody Note

This memo is a duplicate sidecar for the same implementation tranche as
`codex_packet_header_elide_all_members_feedback_20260524`. It is preserved to
avoid losing the parallel sweep signal, but it is not a separate active
execution lane.

## Landing

- Generalized `packet_member_zip_header_elide_v1` to selected-member and
  all-member modes while preserving raw compressed member streams.
- Extended runtime-consumption proof validation from a singleton
  `candidate_member_sha256` to a per-member `candidate_member_sha256s` map.
- Added `--all-members` to the family-agnostic materializer CLI and empirical
  sweep surface. Explicit member subsets remain available through the member
  manifest contract.
- Wired `all_members`, `member_selection`, and explicit member lists through
  final-byte contexts and queue command generation.
- Let the materializer campaign feedback replan auto-discover local
  materializer observation JSON/JSONL under the run directory, so queue feedback
  can consume sweep rows without manual path plumbing.
- Let `tools/build_inverse_steganalysis_action_functional.py` consume a single
  observation JSON object or JSONL file via `--observation`.

## Empirical Anchor

Command:

```bash
.venv/bin/python tools/run_family_agnostic_materializer_sweep.py \
  --archive robust_current_archive_correct=submissions/robust_current/archive_correct.zip \
  --all-members \
  --output-dir experiments/results/packet_header_elide_all_members_sweep_20260524T232249Z \
  --output-json experiments/results/packet_header_elide_all_members_sweep_20260524T232249Z/sweep.json \
  --observation-jsonl experiments/results/packet_header_elide_all_members_sweep_20260524T232249Z/observations.jsonl
```

Result:

- source archive: `345802` bytes, SHA-256 `4dd46fed78ed064bc97c9b3205088e82838c03667394f7936c8ae8d20f9837ab`
- candidate archive: `345646` bytes, SHA-256 `544c5f580ec261f6c8f80643ed7dee2e122f533c06c96a54c28624f9db75ce50`
- saved bytes: `156`
- selected members: `renderer.bin`, `masks.mkv`, `optimized_poses.pt`
- selected member payload SHA-256s preserved
- selected member compressed-byte counts preserved
- receiver proof passed
- observation axis: `[local-materializer-proof]`
- observed deterministic rate gain: `0.00010387399668705873`
- score authority remains false

## Verification

```bash
.venv/bin/ruff check src/tac/optimization/family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializers.py tools/run_family_agnostic_materializer.py tools/run_family_agnostic_materializer_sweep.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/tests/test_byte_shaving_campaign_queue.py src/comma_lab/scheduler/final_byte_operation_contexts.py src/tac/tests/test_final_byte_operation_contexts.py tools/build_inverse_steganalysis_action_functional.py tools/run_byte_shaving_materializer_campaign.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py
.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py -q
.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_materializer_campaign_runner.py::test_materializer_campaign_runner_auto_discovers_materializer_observations src/tac/tests/test_inverse_steganalysis_action_functional_cli.py::test_cli_reads_materializer_observation_jsonl -q
PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_final_byte_operation_contexts.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py src/tac/tests/test_inverse_steganalysis_action_functional_cli.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q
```

Result: ruff clean; `98 passed`; `2 passed`; broader regression slice
`191 passed`.

## Remaining Work

`packet_member_merge_v1` is the next executable family-agnostic materializer.
It should stay false-authority until a cooperative receiver reconstructs the
original member map byte-identically from the merged member and proves unchanged
members are preserved.
