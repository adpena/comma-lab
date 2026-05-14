# PR103 brotli-retuned packet exact-CUDA result - 2026-05-14

## Packet

- Lane: `pr103_global_combo_mid32_latent_hi_brotli_retune_exact_cuda`
- Modal call: `fc-01KRM78XACSYXSPBP8RQP6B6H0`
- Artifact root: `experiments/results/modal_auth_eval/pr103_global_combo_mid32_latent_hi_brotli_retune_cuda_20260514T214611Z`
- Archive: `experiments/results/pr103_global_combo_mid32_latent_hi_brotli_retune_20260514_codex/packet/archive.zip`
- Archive bytes: `178205`
- Archive SHA-256: `7d1e46331a048abeeb40a59e95eb87970bc93f070a2f51f3bf9af8e107ec2c10`
- Runtime: uploaded packet `submission_dir`, `inflate.sh`, and `inflate.py`

## Exact result

- Axis: `[contest-CUDA]`
- Passed path validation: yes
- Score: `0.22776607182321268`
- PoseNet distance: `0.00017199`
- SegNet distance: `0.00067635`
- Evidence JSON: `experiments/results/modal_auth_eval/pr103_global_combo_mid32_latent_hi_brotli_retune_cuda_20260514T214611Z/contest_auth_eval.json`
- Recovery summary: `experiments/results/modal_auth_eval/pr103_global_combo_mid32_latent_hi_brotli_retune_cuda_20260514T214611Z/modal_auth_eval_recover_summary.json`

## Classification

This is a valid byte-positive PR103 packet, but it is not a frontier-lowering
CUDA result. The packet is useful as a trust-region datapoint:

- The candidate saved `18` charged archive bytes relative to the PR103 source
  archive byte count `178223`.
- Local full-shell parity against the PR103 source runtime had already passed
  before dispatch, so this result primarily measures the PR103 CUDA component
  basin plus the byte retune.
- The [contest-CUDA] score is materially above the current HNeRV control basin
  target and must not be promoted or submitted as a frontier candidate.

## Follow-up

- Preserve the byte-retune machinery; it is composable and safe for packets
  whose scorer component basin is already competitive.
- Do not spend more CUDA on PR103 brotli-only retunes until a component-bearing
  PR103 edit exists.
- If a CPU-axis comparison is needed, run the same packet through the canonical
  CPU auth-eval wrapper and label it `[contest-CPU]`; do not infer CPU ranking
  from this CUDA result.
