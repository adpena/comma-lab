# HNeRV HDM11 Active Scorecard Refresh - 2026-05-15

## Result

PR106 PacketIR format `0x0B` is now the active internal `[contest-CUDA]`
score-lowering reference.

- label: `PR106-R2-HDM11-HLM3-MAGICLESS-FMT0B`
- score: `0.20632570864115363`
- archive bytes: `186341`
- archive SHA-256:
  `5c9ef623a089893d6f2dd13c0417149b86a6bdfe52b1f472988e73bd2cfddc4d`
- auth eval JSON:
  `experiments/results/modal_auth_eval/pr106_hdm11_hlm3_fmt0b_t4_20260515T073414Z/contest_auth_eval.json`
- auth eval JSON SHA-256:
  `236a3a539fe1befc9568695dcf963a5ffaab2157d137586064a98c1e60fb46df`
- result review packet:
  `.omx/research/pr106_hdm11_hlm3_magicless_format0b_exact_cuda_result_review_20260515_codex.json`
- exact closure packet:
  `.omx/research/pr106_hdm11_hlm3_magicless_format0b_packetir_exact_closure_20260515_codex.json`

This is a rate-only exact-CUDA improvement over the prior HDM8 fixed-meta row.
Component distances remain `avg_segnet_dist=0.0006426` and
`avg_posenet_dist=0.00003236`.

## Routing

- active scorecard:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/scorecard.json`
- scorecard SHA-256:
  `056ab9023b483dc13a4770195a9cc11d025d42ef6a50c98ef9118027f4d7ab68`
- HDM11 section profile:
  `experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/hdm11_magicless_section_profiles.json`
- HDM11 section profile SHA-256:
  `231ad2e55841dbe1b6301ed750987478fe101dd53f6fe7e3e300ed4cad23db7d`
- entropy ranking:
  `experiments/results/hnerv_frontier_entropy_gap_ranking_20260515_hdm11_codex/frontier_entropy_gap_ranking.json`
- entropy ranking SHA-256:
  `14a481cce24b4c97408ec119ce01a9168a5920290de51cf0cc59df24900a26f9`

`src/tac/hnerv_frontier_defaults.py` now routes active internal exact-CUDA
comparisons to HDM11 format `0x0B`. `tools/all_lanes_preflight.py` now requires
the HDM11 exact eval row while retaining HDM8 as an explicit predecessor.

The same preflight refresh updates the PR106 PR101-grammar runtime-source
custody hash to the current four-file runtime manifest:

- runtime-source tree SHA-256:
  `8ddce462a0d300f29ff9dddca8683cbe08bf97fc7959c296e06647ef12d4249b`

## Profiler Backfill

`experiments/profile_hnerv_frontier_payloads.py` now understands the
format-`0x0B` magicless PacketIR payload. It splits the charged payload into:

- `inner_decoder_packed_brotli_hdm9_magicless_tail`: `169946` bytes
- `inner_latents_and_sidecar_brotli_hlm3_magicless_tail`: `15770` bytes
- `sidecar_payload_pr101_fixed_meta_noop_rank_elided`: `525` bytes

The first byte-mass target remains decoder-stream recoding; the 525-byte
sidecar is tracked but is low priority for CUDA score movement unless a new
charged objective changes component distances.

## Rebuild

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/profile_hnerv_frontier_payloads.py \
  --json-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/hdm11_magicless_section_profiles.json \
  --md-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/hdm11_magicless_section_profiles.md \
  experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.archive.zip
```

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python experiments/build_hnerv_frontier_scorecard.py \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260510_codex/section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/new_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/hlm1_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260513_codex/xmember_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260514_hlm2_codex/hlm2_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm6_codex/hdm6_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm7_codex/hdm7_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm8_codex/hdm8_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm8_codex/hdm8_fixed_meta_section_profiles.json \
  --profile-json experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/hdm11_magicless_section_profiles.json \
  --candidate-manifest experiments/results/pr106_hdm8_fixed_meta_rank_elided_20260514_codex/packetir_manifest.json \
  --candidate-manifest experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.manifest.json \
  --json-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/scorecard.json \
  --md-out experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/scorecard.md \
  PR106-R2-HDM11-HLM3-MAGICLESS-FMT0B=experiments/results/modal_auth_eval/pr106_hdm11_hlm3_fmt0b_t4_20260515T073414Z/contest_auth_eval.json \
  PR106-R2-HDM8-HLM2-XMEMBER=experiments/results/modal_auth_eval/pr106_hdm8_fixed_meta_rank_elided_exact_cuda_20260515T002100Z/contest_auth_eval.json \
  PR106-R2-HDM7-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z/contest_auth_eval.json \
  PR106-R2-HDM6-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm6_hlm2_modal_t4_20260514T081819Z/contest_auth_eval.json \
  PR106-R2-HDM4-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z/contest_auth_eval.json \
  PR106-R2-HDM4-HLM1-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm1_xmember_modal_t4_20260514/contest_auth_eval.json \
  PR106-R2-HDM4-HLM1=experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json \
  PR106-R2-lowlevel-HDM4=experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm4_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json \
  PR106-R2-lowlevel-HDM3=experiments/results/modal_auth_eval/pr106_r2_lowlevel_hdm3_candidate_pr101_runtime_cuda_20260513_codex/contest_auth_eval.json \
  PR106-R2-lowlevel=experiments/results/modal_auth_eval/pr106_r2_pr101_grammar_lowlevel_repack_cuda_20260513_codex/contest_auth_eval.json \
  PR103-ac-repack=experiments/results/lightning_batch/pr103_pr106_ac_repack_exact_eval_t4_20260507T181300Z/contest_auth_eval.adjudicated.json \
  PR106x-lowlevel=experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json \
  PR106x=experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json \
  PR106=experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json \
  PR102=experiments/results/lightning_batch/pr102-public-exact-replay-hardened-g4dn2-20260508T103725Z/contest_auth_eval.adjudicated.json \
  PR105x=experiments/results/lightning_batch/exact_eval_public_pr105_kitchen_sink_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json \
  PR105=experiments/results/lightning_batch/exact_eval_public_pr105_kitchen_sink_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json \
  PR104=experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/contest_auth_eval.adjudicated.json
```

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/rank_hnerv_frontier_entropy_gaps.py \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/scorecard.json \
  --frontier-mode score_lowering \
  --candidate-manifest experiments/results/pr106_hdm10_hlm3_packetir_recode_20260515_codex/candidates/pr101_hdm9_hlm3_magicless_fixed_meta_noop_rank_elided_sidecar_format_0x0b.manifest.json \
  --json-out experiments/results/hnerv_frontier_entropy_gap_ranking_20260515_hdm11_codex/frontier_entropy_gap_ranking.json \
  --md-out experiments/results/hnerv_frontier_entropy_gap_ranking_20260515_hdm11_codex/frontier_entropy_gap_ranking.md
```

## Verification

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest -p no:cacheprovider -q \
  src/tac/tests/test_profile_hnerv_frontier_payloads.py
```

Result: `5 passed`.

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/python tools/audit_hnerv_frontier_scorecard.py \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260515_hdm11_codex/scorecard.json \
  --required-eval PR106-R2-HDM4-HLM1=experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json \
  --required-eval PR106-R2-HDM4-HLM1-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm1_xmember_modal_t4_20260514/contest_auth_eval.json \
  --required-eval PR106-R2-HDM4-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM7-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM8-HLM2-XMEMBER=experiments/results/modal_auth_eval/pr106_hdm8_fixed_meta_rank_elided_exact_cuda_20260515T002100Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM11-HLM3-MAGICLESS-FMT0B=experiments/results/modal_auth_eval/pr106_hdm11_hlm3_fmt0b_t4_20260515T073414Z/contest_auth_eval.json
```

Result:
`PASS (18 rows, 3 payload groups, 54 follow-up targets, internal score-lowering=PR106-R2-HDM11-HLM3-MAGICLESS-FMT0B (0.20632570864115363))`.
