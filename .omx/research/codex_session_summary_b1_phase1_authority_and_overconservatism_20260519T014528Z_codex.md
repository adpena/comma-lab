# Codex Session Summary - B1 Phase 1 Authority And Over-Conservatism

`research_only=true`  
`score_claim=false`

## Completed This Session

1. Claimed canonical task `codex_routing_directive_rate_attack_vector_3_b1_contest_video_codebook_20260518::PHASE_1_PROBES`.
2. Implemented B1 local Phase 1 probe helpers in `src/tac/contest_exploits/contest_video_codebook.py`.
3. Added operator CLIs:
   - `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py`
   - `tools/probe_b1_patch_distribution_density.py`
4. Added focused tests in `src/tac/tests/test_rate_attack_b1_contest_video_codebook.py`.
5. Ran local evidence probes and recorded durable JSON hashes in `.omx/research/codex_findings_b1_phase1_hevc_evaluator_decode_authority_20260519T014528Z_codex.md`.
6. Closed three completed background agents and preserved the over-conservatism findings in `.omx/research/codex_findings_overconservative_authority_bottlenecks_20260519T014528Z_codex.md`.

## Key Outcome

B1 is not retired and not promoted. It is a live high-EV rate-attack path with fail-closed blockers.

The directive's AV1 premise is false for the canonical video; the local source is HEVC/H.265 YUV420p. The implementation therefore establishes an HEVC/evaluator-decode authority path rather than pretending the AV1 path was proven.

Current B1 blockers:

- `directive_av1_premise_false_actual_codec_is_not_av1`
- `cuda_decode_not_attempted`
- `query_source_not_rendered_frontier_output`
- `no_runtime_consumption_proof`
- `no_full_frame_inflate_parity`
- `no_exact_cuda_auth_eval`
- `no_paired_linux_cpu_replay`

## Empirical Artifacts

Decode identity:

- path: `experiments/results/b1_phase1_local_probe_20260519T013300Z/decode_identity.json`
- SHA-256: `7cc08e32b05492e209181c7ef3b1a5b22660757fe9307d63e4b70cdf4a91f3b0`
- verdict: `DEFER`
- blocker status: `blocking`
- evidence grade: `[local-decode-custody-probe]`

Patch density:

- path: `experiments/results/b1_phase1_local_probe_20260519T013300Z/patch_density.json`
- SHA-256: `16492f1dc1488e4d667a10e719236b6fb1844b764d98572577d7ae6452cddd7f`
- verdict: `DEFER`
- blocker status: `blocking`
- evidence grade: `[local-patch-density-probe]`

## Recommended Next Codex Task

Prioritize the deterministic-packet authority bottleneck next, unless canonical pending queue priority says otherwise:

- shared runtime-consumption proof validator for PR101 plus deterministic packet compiler proofs;
- AST/import scorer-load classifier replacing raw token bans;
- exact-readiness acceptance for self-contained `contest_one_video_replay` deterministic packets.

This is the structural correction behind the operator's A1 clarification: specialized deterministic receivers can be valid without shipping the full scorer.

## State Notes

- No probe outcomes were registered because these local probes are blocking and non-promotional; the JSON reports plus canonical task status preserve the evidence without touching a shared partner-dirty ledger.
- Exact B1 lane ID `lane_rate_attack_b1_contest_video_codebook_substrate_20260518` was not found by exact `rg` in the current lane registry. Do not dispatch paid B1 work until this is registered or the directive is superseded.
- An unrelated research memo `.omx/research/magic_codec_plus_water_filling_plus_lagrangian_redirection_unified_synthesis_cross_pollination_20260518.md` was dirty during this session and was intentionally left out of the B1 commit.
