# Monolithic packet candidate bridge - 2026-05-08

Evidence grade: `empirical_archive_construction_no_score`

Purpose: turn parser-section ideas (AAC, low-level Brotli, ADMM, stacked
CodecOps) into deterministic HNeRV candidate archives without reintroducing the
incorrect separate ZIP-member mask/pose/component-budget assumption.

## What landed

- `src/tac/monolithic_packet_candidate.py`
  - replaces parser-proven sections inside a single-member monolithic archive
  - verifies source archive bytes/SHA
  - verifies old/new section bytes/SHA
  - rejects no-op replacement payloads by default
  - forbids caller replacement of PR106 `ff_header`; the header is derived from
    the decoder section length
  - validates source and rebuilt section coverage: unique names, contiguous
    offsets, in-bounds lengths, final cursor equals member length, and SHA match
  - rewrites PR106 `0xff + uint24 decoder_len` header when the decoder section
    changes size
  - writes deterministic single-member `ZIP_STORED` archives with fixed
    timestamp, Unix create-system metadata, regular-file external attributes,
    and safe member-name checks
  - emits a manifest that is explicitly non-promotable:
    `score_claim=false`, `promotion_eligible=false`,
    `rank_or_kill_eligible=false`,
    `ready_for_exact_eval_dispatch=false`
- `tools/build_monolithic_stack_candidate.py`
  - thin operator CLI for one section replacement or an atomic
    `--replacement-manifest` containing multiple parser-proven section
    replacements
  - accepts optional `--runtime-parity-json`, `--claims-path`, and
    `--lane-claim-json` gates; dispatch readiness opens only when a strict
    `tac_runtime_consumption_proof_v1` binds the exact candidate archive SHA,
    changed-section SHAs, command SHA, and log SHA, and an exported canonical
    Level-2 lane claim matches explicit dispatch identity
- `tools/export_active_lane_claim_json.py`
  - read-only helper that exports the newest active nonterminal
    `.omx/state/active_lane_dispatch_claims.md` row as
    `tac_active_lane_claim_json_v1`
  - includes the claim row SHA and claims-file SHA so bridge manifests cannot
    open dispatch from caller-attested JSON alone
- `tools/build_monolithic_runtime_consumption_proof.py`
  - read-only proof builder for `tac_runtime_consumption_proof_v1`
  - consumes a monolithic candidate manifest plus a concrete runtime command and
    runtime log
  - opens `ready_for_exact_eval_runtime=true` only when the runtime log contains
    the exact candidate archive SHA, rebuilt member SHA, and every changed
    section SHA from the candidate manifest
  - records command SHA and log SHA; it does not execute inflate and does not
    claim score
- `src/tac/monolithic_codec_op_replacement.py`
  - strict bridge from materialized CodecOp/Joint-ADMM payload bytes into the
    existing `--replacement-manifest` shape consumed by
    `tools/build_monolithic_stack_candidate.py`
  - requires an actual replacement payload path; planner rows with only
    `bytes_out`/hashes cannot build archive sections by themselves
  - binds source archive bytes/SHA, parser-proven section bytes/SHA, replacement
    bytes/SHA, and optional evidence JSON into
    `tac_monolithic_codec_op_replacement_manifest_v1`
  - rejects PR106 `ff_header`, unknown sections, PR101 fixed-offset decoder or
    latent length changes, no-op payloads, non-Brotli PR106 section payloads
    under Brotli contracts, and known CodecOp/JCS envelope bytes (`COBM1`,
    `CPL1`, `JCSP`, `JCSK`) when the caller claims raw runtime section bytes
- `tools/build_monolithic_codec_op_replacement_manifest.py`
  - operator CLI for the materialized-payload bridge
  - intentionally does not build archives or open dispatch; its output is the
    replacement-manifest input to the monolithic archive builder
  - can now infer the replacement payload from `--evidence-json` when that
    evidence contains `materialized_payload_path`, `replacement_payload_path`,
    `payload_path`, or `blob_path`
- `tools/codec_op_param_sweep_manifest.py`
  - added opt-in `--materialized-payload-output-dir` and
    `--materialized-payload-contract`
  - when requested, each CodecOp `encode()` result writes `result.blob` to a
    deterministic `.section` file and records `materialized_payload_path`,
    `materialized_payload_bytes`, `materialized_payload_sha256`, and
    `materialized_payload_contract` in the sweep candidate row
  - this closes the planner-to-bridge custody gap for grid sweeps without
    changing the default predicted-band behavior or marking any candidate
    dispatchable
- `tools/codec_op_cma_search.py`
  - added opt-in `--materialized-payload-output-dir` and
    `--materialized-payload-contract`
  - successful evaluations can now write `eval_<idx>.section` files and carry
    `materialized_payload_path`, bytes, SHA-256, and contract fields in the
    search report, atom-ledger rows, and best-evidence row
  - payload files are written only after decode coverage is accepted; failed
    full-decode rows leave no materialized section file
  - rows remain CPU/planning evidence only and keep exact CUDA, archive
    substitution, and dispatch blockers
- `src/tac/codec_op_bitstream_materializer.py`
  - now accepts standardized producer aliases:
    `materialized_payload_path`, `materialized_payload_bytes`, and
    `materialized_payload_sha256`
  - also accepts `replacement_payload_*` aliases so monolithic bridge evidence
    rows and golden-vector materialization use the same byte-custody vocabulary
- `tools/run_monolithic_candidate_preflight.py`
  - read-only wrapper for an existing `tac_monolithic_packet_candidate_v1`
    manifest
  - optionally validates runtime proof and lane-claim JSON, emits
    `tac_monolithic_candidate_preflight_v1`, and does not mutate archives,
    touch `.omx/state`, dispatch, or claim score
- `src/tac/frontier_archive_layout.py`
  - parser hardening: PR106-style `0xff + len24` no longer silently wins over
    PR101 fixed-offset grammar for long member `x` packets. Ambiguous packets
    fail closed unless both PR106 logical Brotli streams decode; the layout
    records `parser_ambiguous=false`, `parser_alternatives`, and
    `validated_streams`.

## First materialized candidate

Artifact directory:
`experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/`

Source archive:
`experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/pr106x_hnerv_brotli_repack_candidate.zip`

- source bytes: 186,080
- source SHA-256:
  `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- target section: `decoder_packed_brotli`
- old section bytes: 170,127
- old section SHA-256:
  `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`

Candidate archive:
`experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/pr106x_lgblock16_monolithic_candidate.zip`

- candidate bytes: 186,079
- candidate SHA-256:
  `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2`
- replacement section bytes: 170,126
- replacement section SHA-256:
  `a812f1e837afd0e463a7f133b680ea6c027339ff8816db7012dd41253435afbf`
- archive byte delta: -1
- manifest:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/manifest.json`
- parser layout proof:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/layout.json`
- Brotli raw-equivalence proof:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/brotli_equivalence_proof.json`

This reproduces the prior lgblock16 candidate archive byte-for-byte while now
using the canonical monolithic section bridge. Its refreshed manifest separates
dispatch and promotion:

- `ready_for_exact_eval_dispatch=false`
- `dispatch_blockers=["runtime_consumption_proof_missing",
  "active_lane_claim_missing"]`
- `promotion_blockers=["contest_cuda_auth_eval_missing"]`
- layout proof: `parser_proof_strength="magic_len_and_brotli_streams"`,
  `parser_alternatives=["pr106_ff_packed_hnerv",
  "pr101_fixed_offset_hnerv_microcodec"]`, and both logical Brotli streams
  validate
- raw-equivalence proof: decoder packed Brotli payload changes by `-1` byte
  but decompresses to the same raw decoder bytes; latent/sidecar payload is
  byte-identical
- materialized CodecOp replacement manifest:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/codec_op_replacement_manifest.json`
- manifest-roundtrip candidate:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/pr106x_lgblock16_monolithic_candidate_from_manifest.zip`
  - SHA-256:
    `866dc135e9168d61fab02b6b1c218c4b1d6eed779154a6dc3095fd05e48024f2`
  - this matches the original candidate archive byte-for-byte, proving the
    new adapter does not introduce archive drift
- preflight summary:
  `experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/preflight_from_replacement_manifest.json`
  - `ready_for_exact_eval_dispatch=false`
  - blockers:
    `candidate_manifest_ready_for_exact_eval_dispatch_false`,
    `runtime_consumption_proof_missing`, `active_lane_claim_missing`

It remains not dispatch-ready until runtime consumption proof and an active
Level-2 lane claim are present. Even after those dispatch gates open, it remains
non-promotable until exact CUDA auth eval lands on the exact archive bytes.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_hnerv_generated_schema_codec.py src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_preflight_implementation_model_match.py`
  - result: 66 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_codec_op_replacement.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_build_hnerv_generated_schema_candidate.py src/tac/tests/test_hnerv_generated_schema_codec.py src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_preflight_implementation_model_match.py`
  - result: 79 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_codec_op_param_sweep_manifest.py`
  - result: 5 passed
- `.venv/bin/python -m ruff check tools/codec_op_param_sweep_manifest.py src/tac/tests/test_codec_op_param_sweep_manifest.py`
  - result: all checks passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_codec_op_cma_search.py`
  - result: 9 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_codec_op_bitstream_materializer.py`
  - result: 8 passed
- `.venv/bin/python -m ruff check src/tac/codec_op_bitstream_materializer.py src/tac/tests/test_codec_op_bitstream_materializer.py`
  - result: all checks passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_prove_hnerv_generated_schema_runtime_packet.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/tests/test_monolithic_codec_op_replacement.py src/tac/tests/test_codec_op_cma_search.py src/tac/tests/test_codec_op_param_sweep_manifest.py src/tac/tests/test_codec_op_bitstream_materializer.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_build_hnerv_generated_schema_candidate.py src/tac/tests/test_hnerv_generated_schema_codec.py`
  - result: 70 passed
- `.venv/bin/python -m ruff check tools/prove_hnerv_generated_schema_runtime_packet.py src/tac/tests/test_prove_hnerv_generated_schema_runtime_packet.py tools/run_monolithic_candidate_preflight.py src/tac/tests/test_run_monolithic_candidate_preflight.py src/tac/monolithic_codec_op_replacement.py tools/build_monolithic_codec_op_replacement_manifest.py src/tac/tests/test_monolithic_codec_op_replacement.py tools/codec_op_cma_search.py src/tac/tests/test_codec_op_cma_search.py tools/codec_op_param_sweep_manifest.py src/tac/tests/test_codec_op_param_sweep_manifest.py`
  - result: all checks passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_preflight_implementation_model_match.py src/tac/tests/test_monolithic_packet_candidate.py`
  - result: 47 passed
- `.venv/bin/python -m ruff check src/tac/monolithic_codec_op_replacement.py tools/build_monolithic_codec_op_replacement_manifest.py src/tac/tests/test_monolithic_codec_op_replacement.py src/tac/frontier_archive_layout.py src/tac/monolithic_packet_candidate.py src/tac/tests/test_monolithic_packet_candidate.py tools/build_monolithic_stack_candidate.py tools/export_active_lane_claim_json.py tools/build_monolithic_runtime_consumption_proof.py tools/build_hnerv_generated_schema_candidate.py src/tac/tests/test_build_hnerv_generated_schema_candidate.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/hnerv_generated_schema_packet.py src/tac/tests/test_hnerv_generated_schema_packet.py`
  - result: all checks passed
- `.venv/bin/python -m ruff check src/tac/frontier_archive_layout.py src/tac/monolithic_packet_candidate.py src/tac/tests/test_monolithic_packet_candidate.py tools/build_monolithic_stack_candidate.py tools/export_active_lane_claim_json.py tools/build_monolithic_runtime_consumption_proof.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/hnerv_generated_schema_packet.py src/tac/tests/test_hnerv_generated_schema_packet.py`
  - result: all checks passed
- `git diff --check -- <scoped bridge and HNGP files>`
  - result: clean
- `.venv/bin/python tools/pr106_archive_decomposition.py --archive experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/pr106x_lgblock16_monolithic_candidate.zip --output-json experiments/results/monolithic_stack_candidate_pr106x_lgblock16_20260508_codex/layout.json`
  - result: PR106 grammar, member `x`, header len 170,126, decoder section SHA
    `a812f1e837afd0e463e...`, latent section preserved

## Next tranche

1. Extend Optuna CodecOp search reports with the same optional
   materialized-payload path fields now available in grid sweeps and CMA-ES.
2. Feed sweep/CMA-emitted materialized payloads through
   `tools/build_monolithic_codec_op_replacement_manifest.py` for PR106/PR101
   parser-proven sections.
3. Use the bridge for the generated-schema arch-shrink export once the runtime
   loader/export path lands.
4. Add runtime-log emission to the actual submission runtimes so
   `tools/build_monolithic_runtime_consumption_proof.py` can consume real
   inflate logs rather than synthetic tests.
5. Add a preflight wrapper that runs lane-claim export, runtime-parity proof,
   bridge build, and static compliance in one deterministic operator command.
