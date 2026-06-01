# Codex Findings - SNeRV Receiver Archive Proof

UTC: 2026-06-01T21:55:00Z
Author: Codex
Axis: `[receiver-proof:false-authority]`

## Verdict

SNeRV now has a scorer-free receiver archive grammar proof artifact for the toy
packet path. This is not a score, not promotion authority, and not exact-eval
authority. It does prove that the current receiver packet grammar can serialize,
hash, parse, decode, and reconstruct a deterministic tiny SNeRV frame from
receiver-visible archive bytes.

## Artifact

- Report:
  `.omx/research/snerv_receiver_archive_proof_bins4_20260601T215000Z.json`
- Report SHA-256:
  `e66133d5ff5c689aed71f638e1f53df0691fd87e6a55a802d6c0824e0cc4867a`
- Receiver packet:
  `.omx/research/snerv_receiver_archive_proof_bins4_20260601T215000Z.snar`
- Receiver packet SHA-256:
  `ac9b21eb9a2d4e01f1b9f245faad1eb4be87c58b9fea015e0c3a4c04c25f88b2`

Key machine-readable fields:

```json
{
  "schema": "snerv_receiver_archive_proof.v1",
  "axis_tag": "[receiver-proof:false-authority]",
  "archive_packet_bytes": 2100,
  "archive_packet_sha256": "ac9b21eb9a2d4e01f1b9f245faad1eb4be87c58b9fea015e0c3a4c04c25f88b2",
  "packet_artifact_bytes": 2100,
  "packet_artifact_sha256": "ac9b21eb9a2d4e01f1b9f245faad1eb4be87c58b9fea015e0c3a4c04c25f88b2",
  "packet_artifact_matches_proof": true,
  "receiver_contract_satisfied": true,
  "runtime_consumption_proof_ready": true,
  "receiver_matches_direct": true,
  "max_abs_diff": 0.0,
  "score_claim": false,
  "frontier_score_claim": false,
  "promotion_eligible": false,
  "rank_or_kill_eligible": false,
  "ready_for_exact_eval_dispatch": false
}
```

## Hardening Landed

- Added `tac.substrates.snerv_inverse_steg_carrier.receiver_proof`.
- Added `tools/prove_snerv_receiver_archive.py`.
- Hardened the proof CLI so the packet artifact match requires both byte count
  and SHA-256 equality against the proof record.
- Added a CLI regression test that writes a report and `.snar` packet, then
  asserts the packet artifact hash matches `archive_packet_sha256`.
- Exported archive/proof helpers through
  `tac.substrates.snerv_inverse_steg_carrier`.

## Verification

- `.venv/bin/ruff check tools/prove_snerv_receiver_archive.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_receiver_proof.py`
  - PASS
- `.venv/bin/ruff check src/tac/substrates/snerv_inverse_steg_carrier/archive.py src/tac/substrates/snerv_inverse_steg_carrier/receiver_proof.py src/tac/substrates/snerv_inverse_steg_carrier/__init__.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_receiver_proof.py tools/prove_snerv_receiver_archive.py`
  - PASS
- `.venv/bin/python -m pytest src/tac/substrates/snerv_inverse_steg_carrier/tests/test_archive.py src/tac/substrates/snerv_inverse_steg_carrier/tests/test_receiver_proof.py -q`
  - PASS, 10 tests

## Blockers Remaining

- `toy_receiver_proof_not_full_600_pair_replay`
- `not_packaged_as_contest_archive_zip`
- `paired_contest_cpu_cuda_auth_eval_missing`

## Next Implementation Hook

The next SNeRV receiver step is no longer "invent archive grammar"; it is
"scale this grammar from toy packet to full contest receiver packet and archive
zip." The full packet must carry the compact step-map payload, LF quant payload,
decoder payload, metadata, and deterministic receiver decode path, then pass
same-runtime CPU/CUDA auth gates before any promotion language is valid.
