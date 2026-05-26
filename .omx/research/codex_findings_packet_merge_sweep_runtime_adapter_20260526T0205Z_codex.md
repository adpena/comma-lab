# Codex Findings: Packet-Merge Sweep Runtime Adapter

Date: 2026-05-26T02:05Z
Agent: Codex
Scope: final-rate materializer automation, packet_member_merge_v1 receiver proof,
PR95 MLX sidecar audit intake

## Findings

- `packet_member_merge_v1` had a real direct runtime-adapter path in
  `tools/run_family_agnostic_materializer.py`, but
  `tools/run_family_agnostic_materializer_sweep.py` only emitted the weak
  original-member reconstruction proof. Broad sweeps could therefore find rate
  wins that still required manual receiver repair.
- The packet-merge receiver smoke helper passed a filename-looking string as
  the third contest argument. The real contest contract is
  `inflate.sh archive_dir output_dir file_list`, where `file_list` is a path to
  a newline-delimited file. This created a false smoke failure against
  `submissions/robust_current/inflate.sh`.
- Manual robust-current packet-merge runtime-adapter materialization produced a
  byte-closed candidate with 258 realized saved bytes and a clean runtime
  adapter proof. A full renderer smoke entered the true robust runtime path, but
  was stopped after confirming runtime consumption because it generated multi-GB
  raw output. A signature-only empty-file-list smoke passed and stayed compact.
- The PR95 MLX sidecar audit found a strong MLX timing/parity/export spine, but
  not yet source-faithful 1:1 recovered PR95 training. The next PR95 blockers are
  real SegNet/PoseNet loss wiring, recovered-stage semantics, and full-frame
  source-vs-candidate inflate parity.

## Landed Integration

- Added packet-member-merge source-runtime support to the materializer sweep
  CLI, including runtime directory/proof generation and runtime-adapter
  verification per sweep row.
- Wired `packet_member_merge_source_runtime_dir` through the byte-shaving queue
  sweep command so automated sweeps can produce receiver-ready packet-merge
  observations instead of leaf repair hints.
- Hardened `run_packet_member_merge_receiver_smoke` to generate or accept a real
  file-list path, and record file-list fixture provenance.

## Verification

- `.venv/bin/python -m ruff check tools/run_family_agnostic_materializer_sweep.py src/comma_lab/scheduler/byte_shaving_campaign_queue.py src/tac/optimization/packet_member_merge_receiver.py src/tac/tests/test_family_agnostic_materializer_sweep.py src/tac/tests/test_family_agnostic_materializers.py src/tac/tests/test_byte_shaving_campaign_queue.py src/tac/tests/test_frontier_rate_attack_feedback.py`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializer_sweep.py -k 'packet_member_merge or cli_writes_packet_merge'`
- `.venv/bin/python -m pytest src/tac/tests/test_family_agnostic_materializers.py -k 'packet_member_merge_receiver_runtime'`
- `.venv/bin/python -m pytest src/tac/tests/test_byte_shaving_campaign_queue.py -k 'packet_member_merge_empirical_sweep'`
- `.venv/bin/python -m pytest src/tac/tests/test_frontier_rate_attack_feedback.py -k 'receiver_closed or compiler_discovers_materializers or cli_writes_valid_followup_queue'`

## Next

- Add queue-owned submission-closure insertion after harvest for exact-readiness
  follow-up, so receiver-ready materializers automatically become contest-shaped
  submission packets before bridge evaluation.
- Add the same runtime-adapter sweep support to `tensor_factorize_v1` once its
  receiver readiness is strong enough for family-wide grouped sweeps.
- Resume PR95 MLX on the three hard blockers above, keeping it separate from
  score authority until full-frame parity and exact auth-axis calibration exist.
