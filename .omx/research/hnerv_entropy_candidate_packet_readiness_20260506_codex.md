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

## 2026-05-06 Structural Profile Adapter

- [empirical:experiments/results/hnerv_entropy_packet_discovery_20260506_codex/discovery_report.json]
  Discovery still scans 9 plausible HNeRV JSONs, but now accepts 1 valid local
  audit source: `experiments/results/hnerv_decoder_recode_pr106_20260506_codex/profile.json`.
- [empirical:experiments/results/hnerv_entropy_packet_discovery_20260506_codex/entropy_overhead_audit_from_pr106_decoder_recode_profile.json]
  The adapter materializes a planning-only `entropy_overhead_target_ranking`
  audit from the existing PR106 decoder structural-recode profile. It uses only
  recorded HDC2 fixture accounting where `raw_equal`, `q_roundtrip_equal`, and
  `scale_roundtrip_equal` are true:
  `bytes=221381`, `header_bytes=40840`, `range_payload_bytes=180429`,
  `raw_scale_bytes=112`.
- The selected rank-1 target is
  `public_pr106_belt_and_suspenders:hdc2_global_prev_symbol_contexts` with
  `target_kind=known_model_overhead` and `target_bytes=40840`. This is a
  local accounting target, not a byte-closed candidate and not score evidence.
- [empirical:experiments/results/hnerv_entropy_packet_discovery_20260506_codex/candidate_packet_from_pr106_entropy_audit.json]
  Packet materialization remains fail-closed. Required next artifacts are still
  missing: byte-accounted model-overhead reduction manifest, model-context
  table diff, source/candidate stream manifests, decoded-output equivalence,
  roundtrip validation, candidate archive manifest, strict compliance JSON,
  meta-lagrangian export, and runtime-tree parity.
- Section-only HNeRV profiles remain rejected with a precise missing-data
  report. Their `entropy_bits_per_byte` summaries are not treated as
  reconstructable symbol counts; valid stream-profile input still requires
  full `streams[*].label`, `streams[*].actual_bytes_or_bytes_charged`, and
  `streams[*].symbol_counts_full_histogram`.
