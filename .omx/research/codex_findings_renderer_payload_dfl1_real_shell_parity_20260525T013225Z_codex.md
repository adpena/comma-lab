# Codex Findings: Renderer Payload DFL1 Real Shell Parity

- UTC: 2026-05-25T01:32:25Z
- Lane: `codex_dfl1_parity_queue_execution_20260525`
- Scope: source-runtime same-shell inflate parity for `renderer_payload_dfl1_v1`
- Authority: no score claim; no promotion claim; no exact CPU/CUDA auth claim

## Result

Built and verified a real DFL1 candidate from
`submissions/robust_current/archive_correct.zip` using the same
`submissions/robust_current/inflate.sh` runtime on both source and candidate.

- Source archive: `submissions/robust_current/archive_correct.zip`
- Source SHA-256: `4dd46fed78ed064bc97c9b3205088e82838c03667394f7936c8ae8d20f9837ab`
- Candidate archive:
  `experiments/results/renderer_payload_dfl1_shell_parity_20260525T012946Z/archive.dfl1.zip`
- Candidate SHA-256:
  `e20295f0a662101567f36afe8ce17142635f976cb2f2b8937b4cbc76455ee3c4`
- Source bytes: `345802`
- Candidate bytes: `345422`
- Realized rate saving: `380` bytes

The shell parity proof inflated `0.mkv` through the robust renderer runtime on
CPU and deleted raw scratch output after hashing.

- Source raw bytes: `3662409600`
- Candidate raw bytes: `3662409600`
- Source raw SHA-256:
  `e5d05ee72b4cf9c9a8699009bb97a508ef3826da21f5767acfe15c334a2afad1`
- Candidate raw SHA-256:
  `e5d05ee72b4cf9c9a8699009bb97a508ef3826da21f5767acfe15c334a2afad1`
- Full-frame parity claim: `true`
- Proof SHA-256:
  `690f17ec1905609c3b46090bbe9253a146141eccb0ead2a2a5c8f534df4d2f07`

## Artifacts

- `experiments/results/renderer_payload_dfl1_shell_parity_20260525T012946Z/shell_parity/shell_inflate_parity.json`
- `experiments/results/renderer_payload_dfl1_shell_parity_20260525T012946Z/materializer_manifest.json`
- `experiments/results/renderer_payload_dfl1_shell_parity_20260525T012946Z/runtime_consumption_proof.json`
- `experiments/results/renderer_payload_dfl1_shell_parity_20260525T012946Z/source_queue.with_bridge.json`
- `experiments/results/renderer_payload_dfl1_shell_parity_20260525T012946Z/exact_readiness_bridge.json`

## Wiring Verified

- The materializer consumed the parity proof and cleared
  `renderer_payload_dfl1_full_frame_inflate_parity_missing`.
- Harvest preserved proof path, proof SHA-256, and parity verification fields.
- Exact-readiness revalidated the proof and no longer reports
  `renderer_payload_dfl1_strict_full_frame_inflate_parity_missing`.

Remaining exact-readiness blockers are expected for this standalone artifact:
it is not packaged as a full submission runtime directory, and the candidate is
above the active byte floor.

