# HNeRV HDM12 format0C active scorecard refresh - 2026-05-15

score_claim: `false`
promotion_eligible: `false`
ready_for_exact_eval_dispatch: `false`
dispatch_attempted: `false`

## Result

The internal HNeRV exact-CUDA score-lowering frontier now routes to the already
evaluated PR106 R2 format0C exact-radix packet:

- label: `PR106-R2-HDM12-HLM3-MAGICLESS-FMT0C`
- `[contest-CUDA]` score: `0.2063163866158099`
- `[contest-CPU]` sibling score: `0.22776488386973992`
- archive bytes: `186327`
- archive SHA-256:
  `56cdd10bdc43708f2021458d0877b6c5e5a065a482a61280e727078462aed8e7`
- runtime: `submissions/pr106_latent_sidecar_r2_pr101_grammar`

This is a pure-rate improvement over the prior `FMT0B` routing default
(`0.20632570864115363`, `186341` bytes). CPU and CUDA remain separate evidence
spaces; the CPU result is diagnostic and is not converted into CUDA authority.

## Artifacts

- scorecard:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/scorecard.json`
- scorecard markdown:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/scorecard.md`
- format0C section profile:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/hdm12_format0c_section_profiles.json`
- entropy ranking:
  `experiments/results/hnerv_frontier_entropy_gap_ranking_20260515_hdm12_format0c_codex/frontier_entropy_gap_ranking.json`

## Canonical surfaces updated

- `src/tac/hnerv_frontier_defaults.py`
  - `HNERV_ACTIVE_SCORECARD`
  - `HNERV_ACTIVE_ENTROPY_RANKING`
  - `ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_SCORE`
  - `ACTIVE_NONPROMOTIONAL_EXACT_CUDA_REFERENCE_LABEL`
  - `ACTIVE_SCORE_FRONTIER_SCORE`
  - `ACTIVE_SCORE_FRONTIER_LABEL`
- `tools/all_lanes_preflight.py`
  - `HNERV_SCORECARD_REQUIRED_EVALS` now requires the format0C paired CUDA
    artifact in addition to the predecessor lineage rows.

## Rebuild commands

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/profile_hnerv_frontier_payloads.py \
  --json-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/hdm12_format0c_section_profiles.json \
  --md-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/hdm12_format0c_section_profiles.md \
  experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip
```

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/build_hnerv_frontier_scorecard.py \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/hdm12_format0c_section_profiles.json \
  --candidate-manifest experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.manifest.json \
  --json-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/scorecard.json \
  --md-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm12_format0c_codex/scorecard.md \
  PR106-R2-HDM12-HLM3-MAGICLESS-FMT0C=experiments/results/modal_auth_eval/pr106_format0c_exact_radix_paired_20260515T0918Z_cuda/contest_auth_eval.json \
  ...predecessor exact-CUDA rows...
```

## Verification

Planned verification bundle after code edits:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_pr106_latent_sidecar_recode.py \
  src/tac/tests/test_optimizer_exact_readiness.py \
  src/tac/tests/test_all_lanes_pr106_sidecar_runtime_gate.py
```
