# PR75 Minp P6 Stream-Mix Worker - 2026-05-03

Evidence grade: empirical/byte-screen only. These artifacts are deterministic
local archive candidates, not score evidence. Promotion requires exact CUDA auth
eval on the exact archive bytes via `archive.zip -> inflate.sh ->
upstream/evaluate.py`.

## Inputs

- C089 A++ frontier archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
  - bytes: 276342
  - sha256: `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
  - score: 0.3154707273953505
- Public PR75 minp archive:
  `experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip`
  - bytes: 276481
  - sha256: `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`

## Local Builder

- Tool: `experiments/build_pr75_minp_p6_stream_mix_candidates.py`
- Output matrix:
  `experiments/results/pr75_minp_p6_stream_mix_worker_20260503/candidate_matrix.json`
- Guardrails:
  - deterministic stored ZIP member `p`
  - robust unpacker parse validation
  - decoded stream closure recorded per candidate
  - selected encoded stream SHA-256s recorded
  - zip-slip, duplicate, and hidden/system source members rejected
  - `score_claim=false`, `promotion_eligible=false`

## Top Candidates

| rank | candidate | bytes | delta vs C089 | sha256 | stream mix |
| --- | --- | ---: | ---: | --- | --- |
| 1 | `p6_public_renderer_only` | 276132 | -210 | `189301b6b7b8bc7f166ee7ec835bb71ed61ac56570133ac2bfea975b4377dec7` | C089 mask/action/pose, public renderer |
| 2 | `p6_c089_action_resweep` | 276341 | -1 | `428757d21a6237d284e80f2a0305e0aff99ef0523c3cbcbd476254c84948787b` | C089 decoded streams, 1-byte-smaller P6 action Brotli |
| 3 | `p6_public_renderer_pose` | 276353 | +11 | `c5d2d83cdfc128cafe4ae59278ba872cc9ef423d5dd32a6ab31b95648343439a` | C089 mask/action, public renderer/pose |
| 4 | `p6_public_pose_only` | 276562 | +220 | `5e339c212a2cfdee239145a8bebf978177a0770d413fad87c9d62190d0988239` | C089 mask/action/renderer, public pose |

Best candidate manifest:
`experiments/results/pr75_minp_p6_stream_mix_worker_20260503/p6_public_renderer_only/manifest.json`.

The best candidate is byte-positive versus C089 and keeps the current C089 P6
action stream semantics and pose stream while transplanting only the public
minp renderer stream. Its formula-only rate delta is
`-0.000139830380155656` if components were unchanged. That is not a score
claim.

## Skips And Negative Guardrail

Public-action P6 mixes were skipped. The public minp SG2 decoded action records
are not nondecreasing in pair index, so they cannot be encoded into the current
P6 pair-delta varint stream without changing decoded action bytes. The builder
records this as `skipped_not_p6_delta_varint_encodable` instead of silently
reordering or changing semantics.

## Exact Dispatch Recommendation

Recommended next exact-eval lane/job, when dispatch capacity is available and
after claiming the lane per AGENTS.md:

- lane id: `pr75_minp_p6_public_renderer_only_exact_eval`
- archive:
  `experiments/results/pr75_minp_p6_stream_mix_worker_20260503/p6_public_renderer_only/archive.zip`
- archive bytes: 276132
- archive sha256: `189301b6b7b8bc7f166ee7ec835bb71ed61ac56570133ac2bfea975b4377dec7`
- rationale: highest byte EV from the local P6 stream mix screen; preserves
  C089 mask/actions/pose and changes only the public minp renderer stream.

Do not promote, rank, or claim sub-0.314/sub-0.30 progress from this ledger
without exact CUDA auth eval JSON for the exact archive SHA above.
