# PR101 Kaggle/CMA-ES proxy runtime exact CUDA result — negative classification (2026-05-10)

## Summary

The `proxy_cmaes_0037_pr101_proxy_runtime_packet` candidate was dispatched
through the canonical Modal T4 auth-eval wrapper after passing the live
exact-ready audit. It completed the exact CUDA path:

`archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda`

The result is a CUDA negative relative to the current exact CUDA frontier. It
must not be promoted, submitted, or redispatched as a score-lowering candidate.

## Custody

- Lane id: `pr101_kaggle_proxy_runtime_packet_exact_eval`
- Candidate id: `proxy_cmaes_0037_pr101_proxy_runtime_packet`
- Modal job id:
  `pr101_kaggle_proxy_runtime_packet_exact_cuda_modal_20260510T194142Z`
- Local artifact dir:
  `experiments/results/modal_auth_eval/pr101_kaggle_proxy_runtime_packet_exact_cuda_modal_20260510T194142Z/`
- Candidate archive:
  `experiments/results/kaggle_pr101_proxy_sweep_20260510_codex/pr101_proxy_sweep/proxy_runtime_packet/archive.zip`
- Archive SHA-256:
  `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
- Runtime tree SHA-256:
  `84afb14b741a7250046e6956b00710be02615b8cc500551f77576546245dfaf2`
- Archive bytes: `178258`
- GPU: Modal Tesla T4 (`gpu_t4_match=true`)
- Dispatch claim terminal status: `completed_contest_cuda_auth_eval_negative`

## Exact CUDA result

- `score_recomputed_from_components`: `0.22688160652506983`
- `final_score`: `0.23`
- `avg_posenet_dist`: `0.00017051`
- `avg_segnet_dist`: `0.00066894`
- `archive_size_bytes`: `178258`
- `n_samples`: `600`

## Classification

This is legitimate exact CUDA score movement, but it does not beat the active
exact CUDA anchor `0.20898105278`.

The measured configuration is falsified for score lowering. This does not kill
Kaggle/M5/CMA-ES proxy infrastructure; it invalidates this specific PR101
proxy runtime packet and reinforces that proxy-positive PR101 runtime tweaks
need CUDA calibration before promotion.

## Implications

- Do not treat Kaggle/MPS/macOS CPU proxy scores as auth-eval evidence.
- Do not redispatch this archive/runtime identity.
- New PR101 proxy work must carry either a CUDA-calibrated drift posterior or a
  substantially new charged-byte/runtime transform.
- The exact-ready queue audit should now suppress this same lane/archive/runtime
  identity through the terminal claim row.

## Reactivation criteria

Reopen only with at least one of:

- a different archive SHA-256 and explicit charged-byte diff;
- a different runtime-tree SHA-256 with `score_affecting_runtime_changed=true`
  and runtime-consumption proof;
- a CUDA-calibrated search objective that predicts T4 behavior directly;
- exact CUDA evidence from a new PR101 proxy candidate that beats
  `0.20898105278`.
