# Codex Findings: FEC6 PR101 MLX CPU Auth Tensor Export Dispatch

## Verdict

DISPATCHED_DETACHED_PENDING_RECOVERY.

The first durable auth-axis tensor materialization run for the FEC6 PR101
archive has been dispatched on Modal CPU with tensor payloads routed to the
`comma-auth-eval-cache-artifacts` Modal Volume. This is a materialization and
local-training custody run only; it is not a score/frontier/promotion claim.

## Dispatch

- Lane: `mlx_auth_tensor_materialization_fec6_pr101_cpu`
- Instance/job id: `fec6_pr101_cpu_auth_tensors_20260522T060605Z`
- Modal call id: `fc-01KS74NQBMB5S0XCXKY9T177V1`
- Archive:
  `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`
- Archive SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Uploaded runtime:
  `experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir`
- Runtime transport SHA-256:
  `14c1dffef0107dbf3424bf04736f995687d77acec91f5432e2d34aa5da2fbb2b`
- Modal output dir:
  `experiments/results/modal_auth_eval_cpu/fec6_pr101_mlx_tensor_export_cpu_20260522T060605Z`
- Tensor volume run id:
  `fec6_pr101_cpu_auth_tensors_20260522T060605Z`
- Tensor volume path:
  `/modal_auth_cache/fec6_pr101_cpu_auth_tensors_20260522T060605Z/scorer_input_cache_tensors`

## Recovery Commands

```bash
.venv/bin/python tools/recover_modal_auth_eval.py \
  --output-dir experiments/results/modal_auth_eval_cpu/fec6_pr101_mlx_tensor_export_cpu_20260522T060605Z
```

```bash
.venv/bin/modal volume get comma-auth-eval-cache-artifacts \
  fec6_pr101_cpu_auth_tensors_20260522T060605Z/ \
  ./modal_fec6_pr101_cpu_auth_tensors_20260522T060605Z/
```

## Current Status

Immediate recovery probe returned `pending` at `2026-05-22T06:06:55Z`.

Next action after recovery: download the Modal Volume subtree, then run the
MLX cache/auth identity audit and require `PASS_CACHE_AUTH_EVAL_IDENTITY`
before any local MLX training consumes the tensor cache.
