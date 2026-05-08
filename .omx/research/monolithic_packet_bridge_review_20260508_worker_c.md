# Monolithic packet bridge review - Worker C - 2026-05-08

Evidence grade: `empirical_archive_construction_guard_no_score`.
Score claim: false. Dispatch performed: false.

## Scope

Worker C reviewed the current monolithic bridge surfaces under the PR101/PR106
single-member packet constraint. The narrow implementation gap was in the
direct archive builder: `src/tac/monolithic_codec_op_replacement.py` already
rejects non-Brotli PR106 replacement payloads when the caller declares a PR106
Brotli section contract, but `src/tac/monolithic_packet_candidate.py` could
still accept arbitrary bytes through the direct `--target-section` /
`--replacement-section` path.

That path could build a parser-section candidate whose logical PR106
`decoder_packed_brotli` or `latents_and_sidecar_brotli` section was no longer
runtime-decodable. Runtime proof and lane-claim gates still blocked normal
dispatch, but the archive builder should fail closed before a malformed
monolithic PR106 packet can enter a dispatch-readiness workflow.

## Guard Added

`src/tac/monolithic_packet_candidate.py` now validates PR106 logical Brotli
sections at both boundaries:

- the source PR106 packet must have `decoder_packed_brotli` and
  `latents_and_sidecar_brotli`, and both must Brotli-decompress;
- the rebuilt candidate packet must keep both logical PR106 Brotli sections
  present and Brotli-decodable after replacements and after the derived
  `0xff + uint24 decoder_len` header is refreshed.

If either section is missing or not Brotli-decodable, candidate construction
raises `MonolithicPacketCandidateError` and no dispatch-ready manifest can be
emitted from that malformed packet.

## Regression

`src/tac/tests/test_monolithic_packet_candidate.py` now has an explicit
regression for direct PR106 replacement with non-Brotli section bytes:
`test_pr106_replacement_must_remain_brotli_runtime_section`.

The existing direct-builder PR106 fixtures were updated to use Brotli-compressed
logical sections so the tests keep modeling the deployed PR106 wire contract,
not generic placeholder bytes.

## Verification

- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_packet_candidate.py`
  - result: `13 passed`
- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_monolithic_codec_op_replacement.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_run_monolithic_candidate_preflight.py`
  - result: `29 passed`
- `.venv/bin/python -m ruff check src/tac/monolithic_packet_candidate.py src/tac/tests/test_monolithic_packet_candidate.py src/tac/monolithic_codec_op_replacement.py tools/build_monolithic_stack_candidate.py tools/build_monolithic_runtime_consumption_proof.py tools/run_monolithic_candidate_preflight.py src/tac/tests/test_monolithic_codec_op_replacement.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_run_monolithic_candidate_preflight.py`
  - result: `All checks passed`

## Next Exact-Dispatch Blocker

The monolithic candidate remains dispatch-blocked until real submission-runtime
logs emit the exact candidate archive SHA-256, rebuilt member SHA-256, and each
changed logical section SHA-256, so
`tools/build_monolithic_runtime_consumption_proof.py` can produce a
non-synthetic `tac_runtime_consumption_proof_v1`. An active Level-2 lane claim
is also required before any exact-eval dispatch.
