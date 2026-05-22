# Codex Findings: MLX Auth Tensor Cache Identity Landed

timestamp_utc: 2026-05-22T06:21:34Z
lane_id: mlx_auth_tensor_materialization_fec6_pr101_cpu
author: codex
verdict: PROCEED_FOR_LOCAL_MLX_ACCELERATION_NON_AUTHORITATIVE

## Summary

The detached Modal CPU tensor export completed and produced a full scorer-input
tensor cache for the FEC6/PR101 archive. The downloaded auth-side tensor cache
passes cache/auth identity against the recovered `contest_auth_eval.json`, and a
fresh singleton-window MLX response packet now passes the production contract.

This does not create score, rank/kill, promotion, or submission authority. It
does unlock local MLX scorer-response acceleration for transfer calibration and
candidate-generation loops on the matching auth cache.

## Recovered Auth Eval

- call_id: `fc-01KS74NQBMB5S0XCXKY9T177V1`
- output_dir: `experiments/results/modal_auth_eval_cpu/fec6_pr101_mlx_tensor_export_cpu_20260522T060605Z`
- score_axis: `[contest-CPU]`
- score: `0.1920513168811056`
- archive_sha256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- inflated_outputs_aggregate_sha256: `10c68e4266e79fc3e878fd20136e8aaa56262b3a2ff45eed7b8d5a4b1e1ee66d`

## Downloaded Tensor Cache

Downloaded with:

```bash
.venv/bin/modal volume get --force comma-auth-eval-cache-artifacts \
  fec6_pr101_cpu_auth_tensors_20260522T060605Z/ \
  ./modal_fec6_pr101_cpu_auth_tensors_20260522T060605Z/
```

Cache manifest:

`modal_fec6_pr101_cpu_auth_tensors_20260522T060605Z/fec6_pr101_cpu_auth_tensors_20260522T060605Z/scorer_input_cache_tensors/manifest.json`

The cache contains `segnet_last_rgb.npy`, `posenet_yuv6_pair.npy`,
`pair_indices.npy`, and `manifest.json`.

## Audit Results

The old local cache correctly fails identity against this auth export:

`experiments/results/mlx_production_contract_fec6_pr101_singleton_window_20260522T051339Z/cache_vs_modal_contest_cpu_tensor_export_audit.json`

Blockers:

- `inflated_outputs_aggregate_sha256_mismatch_or_missing`
- `raw_sha256_mismatch_or_missing`
- `scorer_input_array_sha256_mismatch:segnet_last_rgb`
- `scorer_input_array_sha256_mismatch:posenet_yuv6_pair`

The downloaded auth tensor cache passes identity:

`experiments/results/mlx_production_contract_fec6_pr101_singleton_window_20260522T051339Z/cache_vs_modal_downloaded_tensor_cache_audit.json`

Result:

- `PASS_CACHE_AUTH_EVAL_IDENTITY`
- `eligible_for_local_mlx_transfer_calibration=true`
- `identity_residual=0`
- false-authority fields all false

## MLX Production Contract

Fresh auth-cache packet:

- response: `response_auth_tensor_cpu_b1_pairs16_20.json`
- parity: `torch_parity_auth_tensor_cpu_singleton_pairs16_20.json`
- profile: `profile_auth_tensor_cpu_b1_pairs16_20.json`
- stability: `stability_auth_tensor_cpu_b1_pairs16_20.json`
- production contract: `production_contract_auth_tensor_cpu_b1_pairs16_20.json`
- materialization plan: `auth_cache_materialization_plan_downloaded_tensor.json`

Production contract result:

- `PASS_MLX_SCORER_PRODUCTION_CONTRACT`
- `score_claim=false`
- `score_claim_valid=false`
- `promotion_eligible=false`
- `promotable=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`

Strict score-calibration remains intentionally unsatisfied:

`production_contract_auth_tensor_cpu_b1_pairs16_20_require_calibration.json`

Blocker:

- `score_calibration_manifest_not_supplied`

## Code Hardening

- Added top-level `eligible_for_local_mlx_transfer_calibration` and
  `identity_residual` to cache/auth audits so operators do not need to dig into
  the nested canonical-equation payload for the pass/fail transfer signal.
- Added `complete_false_authority_fields(...)` for Modal auth-eval metadata and
  routed non-authoritative spawn, failure, local-request, and diagnostic result
  payloads through it.
- Updated generated Modal volume download commands to include `--force`; the
  first real download exposed that an existing empty local destination directory
  makes `modal volume get` fail without it.
- Ignored local `modal_*_auth_tensors_*/` download roots; the durable signal is
  this memo plus compact manifests, not multi-GB tensor artifacts.

## Next Action

Use the downloaded auth tensor cache as the reference cache for local MLX
candidate-generation and training loops. Before local MLX output can guide
paid exact-eval spend triage, build a strict score-calibration manifest over
multiple byte-closed candidate rows with matching `[contest-CPU]` or
`[contest-CUDA]` auth-axis payloads.
