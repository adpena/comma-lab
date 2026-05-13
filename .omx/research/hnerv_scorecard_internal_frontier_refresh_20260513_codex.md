# HNeRV scorecard internal-frontier refresh (Codex, 2026-05-13)

## Summary

The HNeRV frontier scorecard now preserves two distinct exact-CUDA routing
surfaces:

- `current_frontier`: canonical/public-promotion frontier, still restricted to
  `canonical_frontier_eligible=true`.
- `score_lowering_frontier`: internal exact-CUDA score-lowering frontier. This
  can route byte-closed optimizer work through a lower exact score even when
  public/adjudication blockers such as `promotion_ineligible` remain.

This does not weaken promotion policy. It prevents a lower exact CUDA artifact
from being hidden from optimizer routing solely because the artifact still needs
adjudication or public-promotion cleanup.

## Refreshed scorecard

Artifact:

`experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.json`

Canonical/public frontier:

- label: `PR103-ac-repack`
- score: `0.20898105277982337` `[contest-CUDA T4]`
- archive bytes: `185578`
- archive SHA-256:
  `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce`
- next canonical exact-evaluable target:
  `merged_range_coded_weights_and_hi_latents` (`153856` bytes)

Internal exact-CUDA score-lowering frontier:

- label: `PR106-R2-lowlevel`
- score: `0.2065174760196528` `[contest-CUDA T4]`
- archive bytes: `186629`
- archive SHA-256:
  `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- canonicality blockers: `promotion_ineligible`
- promotion authority: `false`
- next internal exact-evaluable target:
  `inner_decoder_packed_brotli` (`170127` bytes)

Parser correction, same day:

- Initial refresh incorrectly let the path substring `pr101` classify the whole
  `0xfe` PR106 sidecar wrapper as a PR101 fixed-offset payload.
- Correct parser truth: `pr106_sidecar_wrapper` with inner PR106 `0xff`
  payload.
- Correct PR106-R2 internal byte target: `inner_decoder_packed_brotli`, not
  `decoder_compact_brotli_streams`.
- Correct sidecar-wrapper sections:
  `pr106_sidecar_header_fe_fmt_len_u32` (6 bytes),
  `inner_packed_header_ff_len24` (4 bytes),
  `inner_decoder_packed_brotli` (170127 bytes),
  `inner_latents_and_sidecar_brotli` (15849 bytes),
  `sidecar_len_u16` (2 bytes),
  `sidecar_payload_pr101_ranked_no_op` (527 bytes),
  `sidecar_framing_meta_pr101` (6 bytes).

## Rigor check

The new internal surface requires exact CUDA `A++` evidence and a numeric score.
It excludes severe blockers:

- `score_claim_invalid`
- `regression_triggered`
- `lane_status_*`
- `paper_claim_grade_*`
- `evidence_grade_*`

It deliberately does not exclude `promotion_ineligible` by itself. That blocker
prevents public promotion, but should not prevent internal byte-closed
score-lowering work from targeting the best exact-CUDA artifact.

## Verification

Commands:

```bash
PYTHONPATH=. .venv/bin/pytest \
  src/tac/tests/test_profile_hnerv_frontier_payloads.py \
  src/tac/tests/test_hnerv_brotli_saturation.py \
  src/tac/tests/test_hnerv_lowlevel_packer.py \
  src/tac/tests/test_hnerv_frontier_entropy_ranking.py \
  src/tac/tests/test_audit_hnerv_frontier_scorecard.py \
  src/tac/tests/test_build_hnerv_frontier_scorecard.py
.venv/bin/python experiments/profile_hnerv_frontier_payloads.py \
  --json-out experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/new_section_profiles.json \
  --md-out experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/new_section_profiles.md \
  experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip
.venv/bin/python experiments/build_hnerv_frontier_scorecard.py \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260510_codex/section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/new_section_profiles.json \
  --candidate-manifest experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/archive_manifest.candidate.json \
  --json-out experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.json \
  --md-out experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/scorecard.md \
  PR103-ac-repack=experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json \
  PR106-R2-lowlevel=experiments/results/modal_auth_eval/pr106_r2_pr101_grammar_lowlevel_repack_cuda_20260513_codex/contest_auth_eval.json \
  PR106x-lowlevel=experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json \
  PR106x=experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json \
  PR106=experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json \
  PR102=experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json \
  PR105x=experiments/results/lightning_batch/exact_eval_public_pr105_kitchen_sink_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json \
  PR105=experiments/results/lightning_batch/exact_eval_public_pr105_kitchen_sink_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json \
  PR104=experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/contest_auth_eval.adjudicated.json
.venv/bin/python tools/all_lanes_preflight.py
```

Results:

- focused parser/scorecard/ranking/saturation tests: `32 passed`
- all-lanes preflight: `ALL 30 PREFLIGHT CHECKS PASSED`

## Next routing implication

Canonical/public hidden-gem routing still starts from `PR103-ac-repack`. Internal
score-lowering routing starts from `PR106-R2-lowlevel` and should prioritize
byte-different transforms against `inner_decoder_packed_brotli`, with old/new
section SHA-256, charged-byte proof, lane claim, and exact CUDA eval before any
score claim. A bounded local Brotli parameter-grid audit over that inner decoder
section found no rate-positive replacement: best grid bytes `170127`, delta `0`,
verdict `no_rate_positive_brotli_grid_candidate_same_size_variants_only`. The
next score-lowering move is therefore not another generic Brotli parameter
sweep; it is a semantic decoder transform, PR101 split-codec primitive,
sidecar/latent grammar move, or scorer-aware retraining path.
