# HNeRV HDM8 Active Scorecard Refresh - 2026-05-14

## Result

HDM8 fixed-length decoder recode is now the active internal `[contest-CUDA]`
score-lowering reference.

- label: `PR106-R2-HDM8-HLM2-XMEMBER`
- score: `0.20636166502462222`
- archive bytes: `186395`
- archive SHA-256: `8a30730e863a2f846d7ca3a707b3191ad64312f5270976dc5f9322ba4228e8c2`
- auth eval JSON: `experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json`
- result review packet: `.omx/research/hdm8_fixed_lengths_exact_cuda_result_review_20260514_codex.json`

HDM8 preserves the HDM7 SegNet/PoseNet components and moves score only through
the 10-byte rate reduction.

## Routing

- active scorecard: `experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm8_codex/scorecard.json`
- entropy ranking: `experiments/results/hnerv_frontier_entropy_gap_ranking_20260514_hdm8_codex/frontier_entropy_gap_ranking.json`
- post-exact candidate manifest: `.omx/research/hdm8_fixed_lengths_candidate_post_exact_eval_manifest_20260514_codex.json`

`src/tac/hnerv_frontier_defaults.py` now routes the active internal reference
to HDM8. `tools/all_lanes_preflight.py` requires the HDM8 exact eval row in
the HNeRV scorecard audit while retaining HDM7 as an explicit predecessor.

## Runtime Hash Guard

HDM8 exposed a provider-custody footgun: local runtime-tree hashes and Modal
uploaded `--submission-dir` runtime-tree hashes differ because the tree hash
includes the runtime root path/name. The content tree hash is equal.

- local runtime tree: `e8c149b7ea6cd84682ac2d0c792cfdcfc79b709616cd54bcfe1cdad9328d5d26`
- Modal uploaded runtime tree: `77a8859782093d96c2fa7b88aea4bd646bada39837ea4e8e776b89065c38d11b`
- content tree: `08353bf112fc60ee4ec7fc683d8fddb7e5330e8ade04e5910233a5f45e62b9e3`

`experiments/modal_auth_eval.py` and `experiments/modal_auth_eval_cpu.py`
now fail locally before claim/provider spend when `--submission-dir` is paired
with the wrong expected runtime-tree hash. HDM packet readiness artifacts also
surface the Modal uploaded-runtime hash explicitly.

## Verification

```bash
.venv/bin/python tools/audit_hnerv_frontier_scorecard.py \
  --scorecard experiments/results/hnerv_frontier_scorecard_refresh_20260514_hdm8_codex/scorecard.json \
  --required-eval PR106-R2-HDM4-HLM1=experiments/results/modal_auth_eval/hnerv_hlm1_fixed_latent_recode_modal_t4_enforced_20260513/contest_auth_eval.adjudicated.json \
  --required-eval PR106-R2-HDM4-HLM1-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm1_xmember_modal_t4_20260514/contest_auth_eval.json \
  --required-eval PR106-R2-HDM4-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hlm2_xmember_modal_t4_20260514T065903Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM7-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm7_hdm6_hlm2_modal_t4_retry1_20260514T090222Z/contest_auth_eval.json \
  --required-eval PR106-R2-HDM8-HLM2-XMEMBER=experiments/results/modal_auth_eval/hnerv_hdm8_fixed_lengths_modal_t4_retry2_20260514T095000Z/contest_auth_eval.json
```

Result: `PASS (17 rows, 3 payload groups, 51 follow-up targets, internal score-lowering=PR106-R2-HDM8-HLM2-XMEMBER (0.20636166502462222))`.

