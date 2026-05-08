# HNeRV Packet-Section Manifest Gate - 2026-05-08

Scope: bounded monolithic public HNeRV packet-section manifest lane for the
representation integration-gap audit. This is parser-section custody only. It
does not claim scores, dispatch work, write `.omx/state`, or alter existing
dirty files.

## Artifact

- Reusable helper: `src/tac/analysis/hnerv_packet_sections.py`
- CLI: `tools/build_hnerv_packet_section_manifest.py`
- Tests: `src/tac/tests/test_hnerv_packet_sections.py`
- Durable manifest:
  `experiments/results/hnerv_packet_section_manifest_20260508_codex/hnerv_packet_section_manifest.json`
- CLI summaries:
  `experiments/results/hnerv_packet_section_manifest_20260508_codex/hnerv_packet_section_manifest.txt`
  and
  `experiments/results/hnerv_packet_section_manifest_20260508_codex/hnerv_packet_section_manifest_validation.txt`

## Gate Contract

The manifest records:

- archive path, bytes, and SHA-256;
- single ZIP member name, bytes, SHA-256, compression method, compressed bytes,
  and CRC;
- parser identity for PR101 fixed microcodec, PR103 lc_ac, or PR106 len24
  packet;
- parser-proven section names, offsets, starts, ends, lengths, byte counts,
  SHA-256s, roles, and `score_claim: false`;
- coverage facts proving the sections are contiguous and cover the member
  payload;
- `parser_section_gate.ready` and fail-closed blockers from recomputing the
  manifest against the archive bytes.

## Public Frontier Coverage

The default CLI target set is the local PR101, PR103, and PR106 public intake
archives:

- `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
- `experiments/results/public_pr103_intake_20260504_codex/archive.zip`
- `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`

PR103 uses the existing `tac.hnerv_pr103_lc_ac_schema.parse_pr103_lc_ac_payload`
parser rather than a duplicate grammar.

Durable manifest validation returned `parser-section gate: ready`.

## Non-Claims And Blockers

- This gate is CPU byte custody and section validation only.
- It does not prove component quality, runtime parity, or exact CUDA behavior.
- It is not dispatch authorization.
- Candidate promotion still requires old/new archive SHA-256 boundaries,
  runtime consumption proof, strict pre-submission compliance, lane dispatch
  claim, and exact CUDA auth eval.
