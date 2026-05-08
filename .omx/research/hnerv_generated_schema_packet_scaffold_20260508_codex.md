# HNeRV generated-schema packet scaffold - 2026-05-08

Evidence grade: `empirical_packet_scaffold_no_score`

Purpose: give the generated-schema `.hngs` lane a deterministic monolithic
packet grammar before any runtime, exact eval, or score claim is attempted.

## What landed

- `src/tac/hnerv_generated_schema_packet.py`
  - defines HNGP v1 as `header + hngs_decoder + latent_blob + sidecar_blob`
  - requires the decoder section to start with `HNGS`
  - writes a canonical JSON header with section lengths and SHA-256 digests
  - parses fail-closed on bad magic/version, invalid JSON, duplicate/unknown
    sections, truncation, trailing bytes, and section SHA mismatch
  - emits only non-score manifest fields:
    `score_claim=false`, `promotion_eligible=false`,
    `rank_or_kill_eligible=false`, `ready_for_exact_eval_dispatch=false`
- `src/tac/tests/test_hnerv_generated_schema_packet.py`
  - covers deterministic build/parse, section accounting, truncation, SHA
    mismatch, duplicate-section rejection, HNGS magic guard, and reserved
    score/readiness metadata rejection.
- `src/tac/frontier_archive_layout.py`
  - recognizes HNGP packets as `grammar="hngp_v1"` with
    `parser_proof_strength="canonical_hngp_parse"`
  - exposes logical sections `header`, `hngs_decoder`, `latent_blob`, and
    `sidecar_blob` for downstream packet/bridge tooling.
- `tools/build_hnerv_generated_schema_candidate.py`
  - wraps `.hngs` decoder bytes plus latent and sidecar blobs into a
    deterministic single-member `ZIP_STORED` HNGP archive
  - derives a safe `<candidate_id>.hngp` member name and rejects traversal,
    hidden, absolute, or otherwise unsafe member names
  - writes `tac_hnerv_generated_schema_candidate_archive.v1` with explicit
    non-score, non-promotable, non-dispatchable fields and dispatch blockers
    for missing runtime/inflate/output-parity/CUDA evidence
- `src/tac/tests/test_build_hnerv_generated_schema_candidate.py`
  - covers deterministic archive construction, safe member-name rejection,
    non-promotable manifest fields, no `.omx/state` mutation claim, and
    malformed HNGS rejection before outputs are written
- `tools/prove_hnerv_generated_schema_runtime_packet.py`
  - proves standalone runtime-style consumption of a deterministic HNGP archive
    member without importing the scorer, dispatching, or touching `.omx/state`
  - parses the packet with the `tac.hnerv_generated_schema_packet` oracle and
    with an independent minimal runtime-style parser, then compares section
    names, offsets, lengths, and SHA-256 digests
  - emits `tac_hnerv_generated_schema_runtime_packet_proof_v1` with
    `proof_family="tac_runtime_consumption_proof_v1"` and fail-closed readiness
- `src/tac/tests/test_prove_hnerv_generated_schema_runtime_packet.py`
  - covers good proof, tampered ZIP metadata, malformed HNGP member, and CLI
    JSON emission

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_generated_schema_packet.py`
  - result: 5 passed
- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_generated_schema_codec.py src/tac/tests/test_hnerv_generated_schema_packet.py`
  - result: 9 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_monolithic_packet_candidate.py`
  - result: 17 passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_hnerv_generated_schema_codec.py`
  - result: 22 passed
- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_build_hnerv_generated_schema_candidate.py`
  - result: 10 passed
- `.venv/bin/python -m pytest src/tac/tests/test_prove_hnerv_generated_schema_runtime_packet.py`
  - result: 4 passed
- `.venv/bin/python -m ruff check tools/prove_hnerv_generated_schema_runtime_packet.py src/tac/tests/test_prove_hnerv_generated_schema_runtime_packet.py`
  - result: all checks passed
- `.venv/bin/python -m pytest -q src/tac/tests/test_monolithic_codec_op_replacement.py src/tac/tests/test_build_monolithic_runtime_consumption_proof.py src/tac/tests/test_monolithic_packet_candidate.py src/tac/tests/test_export_active_lane_claim_json.py src/tac/tests/test_hnerv_generated_schema_packet.py src/tac/tests/test_build_hnerv_generated_schema_candidate.py src/tac/tests/test_hnerv_generated_schema_codec.py src/tac/tests/test_lossy_coarsening_lightning_tools.py src/tac/tests/test_preflight_implementation_model_match.py`
  - result: 79 passed
- `git diff --check -- src/tac/hnerv_generated_schema_packet.py src/tac/tests/test_hnerv_generated_schema_packet.py`
  - result: clean

## Current limits

- This is not a contest runtime.
- No `inflate.sh` consumes HNGP yet.
- No runtime parity proof exists.
- No CUDA auth eval exists.
- No score, rank, promotion, or kill claim is permitted from this scaffold.

## Next tranche

1. Add a self-contained `submissions/hnerv_generated_schema_runtime/` loader
   that consumes HNGP without importing `tac` or scorer code.
2. Wire a self-contained runtime to consume the HNGP candidate archive produced
   by `tools/build_hnerv_generated_schema_candidate.py`.
3. Add an adapter that converts the HNGP runtime proof into the strict
   monolithic runtime-consumption gate if HNGP becomes a monolithic dispatch
   substrate.
4. Only after those proofs pass, claim a lane and run exact CUDA auth eval on
   the exact archive bytes.
