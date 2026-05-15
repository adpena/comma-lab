# PR106 PacketIR Exact Review Join Blocker Fix - 2026-05-15

score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Fix

`tools/profile_pr106_latent_sidecar_recode.py` now treats a valid exact-CUDA
result review as both exact-eval evidence and adjudication evidence. When a
candidate archive SHA matches a `tac_result_review_packet_v1` row with
`exact_cuda_evidence=true` and `score_claim_valid=true`, the profile removes
both stale blockers:

- `exact_cuda_auth_eval_missing`
- `contest_auth_eval_adjudication_missing`

It then adds the intentional duplicate-dispatch blocker:

- `exact_cuda_result_review_already_exists`

## Evidence

Focused regression:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Observed: `12 passed`.

Lint:

```bash
.venv/bin/ruff check \
  tools/profile_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_pr106_latent_sidecar_recode.py
```

Observed: `All checks passed`.

Regenerated proof profile:

```bash
.venv/bin/python tools/profile_pr106_latent_sidecar_recode.py \
  --sidecar-archive experiments/results/pr106_hdm9_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm2_inner_headerless_fixed_meta_rank_elided_sidecar_format_0x09.archive.zip \
  --json-out experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.json \
  --md-out experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/profile.with_proofs.md \
  --emit-runtime-candidates-dir experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates \
  --runtime-consumption-proof experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/runtime_consumption_hdm10_hlm3.json \
  --same-runtime-full-frame-parity experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/same_runtime_full_frame_parity_hdm9_vs_hdm10.json
```

Format `0x0A` now reports:

- archive SHA-256:
  `186a3d59f2038be61bfda7aa97cdc7abcf970ce4f2d20cd84d42386e894d2ce7`
- exact `[contest-CUDA]` score review:
  `.omx/research/pr106_hdm10_hlm3_format0a_exact_cuda_result_review_20260515_codex.json`
- `candidate_exact_eval_blockers=["exact_cuda_result_review_already_exists"]`
- `exact_cuda_auth_eval_claim=true`
- `ready_for_exact_eval_dispatch=false`

## Impact

This does not lower score directly and does not create a score claim. It
prevents stale PacketIR profiles from telling operators that an already
result-reviewed archive still needs adjudication, while still blocking duplicate
dispatch of the same exact-evaluated archive.
