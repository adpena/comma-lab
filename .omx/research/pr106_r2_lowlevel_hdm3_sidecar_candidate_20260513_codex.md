# PR106-R2 Lowlevel HDM3 Sidecar Candidate - 2026-05-13

## Classification

This is a byte-closed exact-eval packet candidate for the PR106-R2 lowlevel
frontier archive. It is not a score claim and it is not promotion-eligible
until a claimed exact CUDA auth eval returns.

Evidence axis: packet/static custody plus runtime-adapter payload identity
proof. Exact score axis remains pending.

## Source Frontier

- Source label: `PR106-R2-lowlevel`
- Source exact score: `0.2065174760196528 [contest-CUDA T4]`
- Source archive:
  `experiments/results/pr106_r2_pr101_grammar_lowlevel_repack_20260513_codex/pr106_r2_pr101_grammar_hnerv_brotli_repack_candidate.zip`
- Source archive bytes: `186629`
- Source archive SHA-256:
  `287e6edc612803a9a9d5de3ce50b421c039704f38bae442a6dcc97a3e8d6ed4d`
- Source payload kind: `pr106_sidecar_wrapper`
- Source decoder section: `inner_decoder_packed_brotli`
- Source decoder section bytes: `170127`

## Candidate

- Candidate archive:
  `experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/pr106_r2_lowlevel_hdm3_archive_candidate.zip`
- Candidate static release surface:
  `experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/exact_eval_static_release_surface/`
- Candidate archive bytes: `186615`
- Candidate archive SHA-256:
  `8cc7e3b21a5f77604331abb727c105e21351e8c199456db741eecb1fc7714093`
- Candidate payload kind: `pr106_sidecar_wrapper`
- Candidate decoder section bytes: `170113`
- Decoder-section byte delta: `-14`
- Rate-only score delta if components are equal:
  `-0.0000093220253437104`

## Proofs

- Structural recode profile:
  `experiments/results/pr106_r2_lowlevel_decoder_structural_recode_profile_20260513_codex/profile.json`
- Build/readiness manifest:
  `experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/manifest.with_tool_run.json`
- Runtime adapter proof:
  `experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/runtime_adapter_proof.with_tool_run.json`
- Static compliance proof:
  `experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/pre_submission_compliance.static.json`

Observed proof facts:

- `ready_for_archive_preflight=true`
- `ready_for_exact_eval_packet=true`
- `static_packet_ready=true`
- `strict_static_compliance.passed=true`
- `inflate_output_parity_proven_by_payload_identity=true`
- `restored_payload_matches_source=true`
- PR106 sidecar payload and framing are preserved while the inner decoder
  section is normalized through the HDM3 runtime adapter.

## Current Blockers

Dispatch blockers:

- `lane_dispatch_claim_missing`
- `exact_cuda_auth_eval_missing`

Score blockers:

- `exact_cuda_auth_eval_missing`
- `contest_auth_eval_adjudication_missing`
- `operator_score_claim_review_missing`

Do not promote, rank, or submit this candidate from the rate-only delta. The
only valid next score-lowering step is a claimed exact CUDA auth eval against
the candidate archive and its exact static release runtime surface.

## Adversarial Review

This candidate is deliberately tiny: the measured gain is only `14` bytes, so a
runtime or harness mismatch would dominate the effect. The payload-identity
proof is therefore necessary but not sufficient; exact CUDA eval is still the
promotion boundary.

The result does confirm a reusable implementation surface: PR106 sidecar
wrappers can now flow through the structural recode profiler, HDM3 archive
builder, and HDM3 runtime proof path without losing wrapper bytes. That matters
for the next larger score-lowering target: semantic transforms of
`inner_decoder_packed_brotli`, especially the 40,840 byte
`hdc2_self_describing_context_header` overhead called out by the recode
profile.

## Verification

- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_hnerv_hdm3_runtime_adapter.py src/tac/tests/test_hnerv_hdm3_archive_candidate.py src/tac/tests/test_hnerv_decoder_recode.py`
  - `22 passed in 14.06s`
- `.venv/bin/ruff check tools/profile_hnerv_decoder_structural_recode.py tools/prove_hnerv_hdm3_runtime_adapter.py src/tac/hnerv_hdm3_runtime_adapter.py src/tac/hnerv_hdm3_archive_candidate.py src/tac/tests/test_hnerv_decoder_recode.py src/tac/tests/test_hnerv_hdm3_runtime_adapter.py src/tac/tests/test_hnerv_hdm3_archive_candidate.py`
  - passed
- `.venv/bin/python scripts/pre_submission_compliance_check.py --submission-dir experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/exact_eval_static_release_surface --archive experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/exact_eval_static_release_surface/archive.zip --archive-manifest-json experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/exact_eval_static_release_surface/archive_manifest.json --expect-single-member 0.bin --expected-archive-sha256 8cc7e3b21a5f77604331abb727c105e21351e8c199456db741eecb1fc7714093 --expected-archive-size-bytes 186615 --public-scan-path experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/exact_eval_static_release_surface --json-out experiments/results/pr106_r2_lowlevel_hdm3_archive_candidate_20260513_codex/pre_submission_compliance.static.json --strict`
  - passed

## 2026-05-13 Runtime Mismatch And Fix

An initial Modal T4 dispatch used the wrong runtime:

- Lane id: `pr106_r2_lowlevel_hdm3_sidecar_exact_cuda`
- Instance/job id: `pr106_r2_lowlevel_hdm3_candidate_cuda_20260513_codex`
- Modal call id: `fc-01KRG076Y249V7W7MWT0HHNCH2`
- Archive SHA-256:
  `8cc7e3b21a5f77604331abb727c105e21351e8c199456db741eecb1fc7714093`
- Runtime used:
  `experiments/public_runtime_adapters/pr106_belt_and_suspenders_adapter/inflate.sh`
- Result: failed closed, no score claim, no promotion.
- Failure class: `archive/runtime grammar mismatch`.

The failure is not a candidate score result. The PR106-R2 lowlevel source
frontier was evaluated with
`submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh`, not the public
PR106 belt-and-suspenders parser. The wrong parser reached
`brotli.error: decoder failed` while reading the incompatible metadata grammar.

Follow-up fix:

- `submissions/pr106_latent_sidecar_r2_pr101_grammar/src/codec.py` now decodes
  HDM3 fixed-schema q-Brotli/raw-scale decoder sections directly.
- The candidate release-surface wrapper now delegates to
  `submissions/pr106_latent_sidecar_r2_pr101_grammar/inflate.sh`.
- Regression test:
  `src/tac/tests/test_submission_pr101_decoder_adapter.py::test_pr106_r2_pr101_runtime_accepts_hdm3_decoder_section`
  proves legacy Brotli and HDM3 sections reconstruct identical decoder state
  dictionaries under the exact PR101-grammar runtime.
- The corrected exact-eval command must use the PR101-grammar runtime via
  `--submission-dir submissions/pr106_latent_sidecar_r2_pr101_grammar --inflate-sh inflate.sh`.

## 2026-05-13 Corrected Exact CUDA Result

The corrected Modal T4 dispatch used the matching PR106-R2 PR101-grammar
runtime and passed exact CUDA path validation:

- Lane id: `pr106_r2_lowlevel_hdm3_sidecar_pr101_runtime_exact_cuda`
- Instance/job id:
  `pr106_r2_lowlevel_hdm3_candidate_pr101_runtime_cuda_20260513_codex`
- Modal call id: `fc-01KRG0WVJQ0W6ASF1RZY29W26K`
- Result artifact:
  `experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm3_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json`
- Evidence grade: `contest-CUDA`
- GPU: `Tesla T4`
- Archive bytes: `186615`
- Archive SHA-256:
  `8cc7e3b21a5f77604331abb727c105e21351e8c199456db741eecb1fc7714093`
- Runtime tree SHA-256:
  `43c286c56247fa7d82eb3142c6e9565a69a1ab50bef616f773b35fbd1a0ef54d`
- Inflated raw aggregate SHA-256:
  `5f65c70f59c78e5a4394dc062fe750cf721619f6d67790c4844d52f14d248993`
- `avg_segnet_dist`: `0.0006426`
- `avg_posenet_dist`: `0.00003236`
- Recomputed canonical score: `0.2065081539943091 [contest-CUDA T4]`

This is a byte-only score lowering from `0.2065174760196528` to
`0.2065081539943091`. The raw aggregate SHA matches the source lowlevel replay,
so the observed score movement is exactly the 14 charged archive bytes removed
by HDM3. It remains promotion-ineligible until the normal adjudication/policy
gates run, but it is valid internal exact-CUDA score-lowering evidence.

The HNeRV frontier scorecard was refreshed so the internal score-lowering
frontier is now `PR106-R2-lowlevel-HDM3`.
