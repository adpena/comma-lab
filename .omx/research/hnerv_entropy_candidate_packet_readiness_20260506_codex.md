# HNeRV Entropy Candidate Packet Readiness - 2026-05-06

## Scope

This local helper converts an HNeRV entropy codec gap audit or stream profile
into a deterministic candidate-packet readiness manifest. It is bounded to the
rate-candidate packet handoff layer and makes no score claim.

## Guardrails

- `ready_for_exact_eval_dispatch` remains `false`; the packet is not a dispatch
  authorization.
- Missing source custody, byte-equivalence, runtime parity, archive manifest,
  strict compliance, and lane-claim/CUDA eval requirements stay explicit in the
  manifest.
- Available requirement artifacts are recorded by path, byte size, and
  SHA-256. Missing or invalid JSON artifacts remain blockers.
- The tool can consume a full entropy audit with
  `entropy_overhead_target_ranking` or build that audit locally from a stream
  profile containing `streams`.

## Remaining Blockers

- A selected HNeRV entropy row still needs byte-equivalent recode artifacts:
  source stream ranges, candidate stream ranges, decoded-output equality,
  roundtrip validation, runtime-tree parity, candidate archive manifest, strict
  pre-submission compliance JSON, and meta-lagrangian atom export.
- After local packet review, any GPU exact-eval attempt still requires the
  dispatch-claim protocol and exact CUDA auth eval on archive bytes.

## 2026-05-06 Materialization Discovery

- [empirical:experiments/results/hnerv_entropy_packet_discovery_20260506_codex/discovery_report.json]
  `tools/build_hnerv_entropy_candidate_packet.py` now discovers candidate
  HNeRV entropy audit/profile inputs when `--entropy-audit` is omitted.
- Discovery scanned the deterministic HNeRV/profile/audit JSON surface under
  `experiments/results` and `.omx/research/artifacts`, found 9 plausible source
  JSON files, and accepted 0 as valid entropy packet inputs.
- The report records exact byte size and SHA-256 for every discovered source
  JSON. Current blockers are explicit:
  `hnerv_entropy_codec_gap_audit_json_with_entropy_overhead_target_ranking` or
  `or_hnerv_stream_profile_json_with_streams_actual_bytes_and_symbol_counts`.
- Existing HNeRV packing/section profiles remain fail-closed because they do
  not carry the stream `symbol_counts` required to build an entropy audit. No
  packet manifest, dispatch readiness, score claim, or lane claim was produced.
