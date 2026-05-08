# A5 Frame-Conditional Runtime Readiness Closure — 2026-05-08

## Scope

Local A5 packet-readiness closure for the PR101 frame-conditional bit-budget
candidate. No remote dispatch, no scorer run, no score claim, and no rank or
kill promotion.

## Candidate Packet

- Packet directory:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet/`
- Archive:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet/archive.zip`
- Archive bytes: `172615`
- Archive SHA-256:
  `cde5a1e0ad49ec56856a8b1f0e4d7c329193955211d0f9fe314366d992c4c737`
- Member `x` SHA-256:
  `cea4e6a10613e6920873cf1bd48a987f05a9bbeb3bb7b6de17f324b81af30569`
- A5 side-info SHA-256:
  `2685786664c499e41e768259823c91e8b4164cd95845e5765aa658620fee05d4`
- A5 latent-wire payload SHA-256:
  `7b222b242523d7f86f4e4e420da6467f1a7916d91577bf6fa0912a469a3c0f13`

## New Artifacts

Static compliance artifact:

- Path:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/pre_submission_compliance.no_auth.json`
- Status: `passed=true`
- Semantics: strict static packet compliance only. Auth eval is intentionally
  absent, so this is not score evidence.

Per-pair score-marginal artifact:

- Path:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json`
- Schema: `pr101_a5_per_pair_score_marginals.v1`
- Evidence grade: `[macOS-MPS advisory pair-difficulty proxy]`
- Semantics: advisory score-domain routing evidence only. It binds the A5
  q-bit schedule consumed by the candidate archive to the existing
  `pair_difficulty_v2` per-pair score proxy. Exact CUDA and contest-CPU eval
  remain required before any score or promotion claim.
- Alignment:
  - q-bits vs score Pearson: `0.1803565159334396`
  - q-bits vs pose Pearson: `0.07102893855091898`
  - q-bits vs seg Pearson: `0.33426668978500884`
  - low-q pair count (`q<=2`): `283`
  - high-q pair count (`q>=8`): `198`

Readiness artifact:

- Path:
  `experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/readiness.with_runtime_packet.json`
- Local blockers: `[]`
- `ready_for_local_packet_review=true`
- `ready_for_archive_preflight=true`
- `ready_for_exact_eval_after_lane_claim=true`
- `ready_for_exact_eval_dispatch=false`
- Remaining blockers:
  - `requires_level2_dispatch_claim_before_exact_eval`
  - `requires_exact_cuda_auth_eval_before_score_promotion`

## Guard Fix

`scripts/pre_submission_compliance_check.py` now accepts candidate manifests
whose archive identity is nested under `candidate_archive`, matching the A5
manifest shape that the same checker already accepted for member records. It
also accepts hex-string CRCs in archive-member manifests. The stricter
identity checks remain intact; stale SHA, stale size, and stale member metadata
still fail.

## Focused Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_a5_per_pair_score_marginal_manifest.py \
  src/tac/tests/test_pre_submission_compliance_check.py -q
```

Result: `24 passed`.

```bash
.venv/bin/python scripts/pre_submission_compliance_check.py \
  --submission-dir experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet \
  --archive experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/packet/archive.zip \
  --archive-manifest-json experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/candidate_archive_manifest.json \
  --expect-single-member x \
  --expected-archive-sha256 cde5a1e0ad49ec56856a8b1f0e4d7c329193955211d0f9fe314366d992c4c737 \
  --expected-archive-size-bytes 172615 \
  --json-out experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/pre_submission_compliance.no_auth.json \
  --strict
```

Result: exit `0`.

## Next Gate

Before exact eval: claim a Level-2 lane for this exact A5 candidate archive,
then run paired exact CUDA and contest-CPU auth eval. The advisory per-pair
artifact is a dispatch-prior signal only and must not be promoted as achieved
score.
