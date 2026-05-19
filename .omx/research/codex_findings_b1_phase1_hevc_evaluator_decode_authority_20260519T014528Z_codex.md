# Codex Findings - B1 Phase 1 HEVC Evaluator Decode Authority

`research_only=true`  
`score_claim=false`  
`promotion_eligible=false`

## Scope

Source directive: `.omx/research/codex_routing_directive_rate_attack_vector_3_b1_contest_video_codebook_20260518.md`.

Codex implemented the Phase 1 local proof surface as reusable helper code plus two operator tools:

- `src/tac/contest_exploits/contest_video_codebook.py`
- `tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py`
- `tools/probe_b1_patch_distribution_density.py`
- `src/tac/tests/test_rate_attack_b1_contest_video_codebook.py`

The legacy decode-probe filename keeps the directive slug for routing compatibility, but the implementation is now evaluator-path-aware.

## Findings

### P0 - AV1 premise is false for the canonical contest video

`upstream/videos/0.mkv` is HEVC/H.265, not AV1.

Local probe evidence:

- video SHA-256: `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- codec: `hevc`
- pixel format: `yuv420p`
- shape: `1164x874`
- evidence JSON: `experiments/results/b1_phase1_local_probe_20260519T013300Z/decode_identity.json`
- evidence JSON SHA-256: `7cc08e32b05492e209181c7ef3b1a5b22660757fe9307d63e4b70cdf4a91f3b0`

The report records blocker `directive_av1_premise_false_actual_codec_is_not_av1`.

### P0 - Generic PyAV RGB decode is not contest decode authority

The evaluator uses `AVVideoDataset` on CPU and `DaliVideoDataset` on CUDA. The helper now routes CPU decode through upstream `frame_utils.yuv420_to_rgb`, matching `AVVideoDataset`, and treats CUDA/DALI as the required CUDA authority leg.

Local CPU-only output:

- evaluator dataset: `AVVideoDataset`
- frame count: `8`
- frame shape: `874x1164x3`
- frame SHA-256: `ecb0193405644e76171392da7f5b2e9abe1d92d2f1e5cd4daf6a96bb4d7155a5`

CUDA/DALI was not attempted locally, so the report is `DEFER` with `blocker_status=blocking`.

### P1 - Patch density is machinery evidence until rendered-frontier custody exists

The patch-density probe is intentionally non-authoritative when the query source is held-out upstream frames rather than inflated frontier output with archive/runtime custody.

Local held-out self-density evidence:

- evidence JSON: `experiments/results/b1_phase1_local_probe_20260519T013300Z/patch_density.json`
- evidence JSON SHA-256: `16492f1dc1488e4d667a10e719236b6fb1844b764d98572577d7ae6452cddd7f`
- NN backend: `numpy_bruteforce_l2`
- p50 RMSE: `5.821036338806152`
- threshold RMSE: `10.0`
- dense at threshold: `true`
- authority: `density_authority=false`
- blocker: `query_source_not_rendered_frontier_output`

This validates the local machinery and suggests B1 is not nonsense, but it is not Phase 2 dispatch authority.

### P1 - Faiss/OpenMP is an optional backend, not a default authority path

A local patch-density attempt importing Faiss after evaluator CPU decode triggered an OpenMP duplicate-runtime abort. The helper now defaults to deterministic NumPy brute force and exposes Faiss as explicit opt-in via `--nn-backend faiss`.

This keeps the Phase 1 probe stable and avoids converting a local library-load collision into a false method negative.

### P1 - Probe-outcome registration must use canonical fields and blocking status

Both tools construct `probe_outcome_kwargs` using the actual `tac.probe_outcomes_ledger.register_probe_outcome` API:

- `substrate`
- `recipe_path`
- `probe_kind`
- `verdict`
- `metric_name`
- `metric_value`
- `threshold`
- `threshold_token`
- `next_action`
- `blocker_status`

Missing CUDA/DALI decode authority and missing rendered-frontier query custody are `blocking`, not advisory.

## Authority Boundary

B1 Phase 1 currently proves:

- local HEVC stream custody;
- evaluator CPU decode repeatability through `AVVideoDataset` conversion semantics;
- local patch-density machinery;
- fail-closed non-promotable JSON reports.

B1 Phase 1 does not prove:

- CUDA/DALI/NVDEC bit identity;
- rendered-frontier query density;
- runtime consumption of B1 archive bytes;
- archive byte savings;
- exact CUDA auth eval;
- paired Linux CPU replay.

## Required Next Evidence

Before B1 substrate dispatch:

1. Run the decode identity probe on a contest T4 with `--enable-cuda-decode` and require `HEVC_EVALUATOR_CPU_DALI_CUDA_BIT_IDENTICAL`.
2. Run the patch-density probe against rendered frontier frames with archive SHA, runtime tree SHA, inflated raw-output aggregate SHA, and `query_is_rendered_frontier=true`.
3. Rename or supersede AV1-specific directive language to HEVC/evaluator-decode authority.
4. Register the exact B1 lane ID in the canonical lane registry before paid dispatch. The directive claimed `lane_rate_attack_b1_contest_video_codebook_substrate_20260518` was pre-registered, but current exact `rg` did not find that lane ID.

## Verification

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_b1_pyav_av1_cpu_cuda_bit_identity.py \
  --frame-count 8 \
  --output-json experiments/results/b1_phase1_local_probe_20260519T013300Z/decode_identity.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/probe_b1_patch_distribution_density.py \
  --codebook-frame-count 4 \
  --query-frame-count 2 \
  --codebook-start-frame 0 \
  --query-start-frame 16 \
  --stride 128 \
  --max-codebook-patches 1024 \
  --max-query-patches 512 \
  --nn-backend numpy \
  --output-json experiments/results/b1_phase1_local_probe_20260519T013300Z/patch_density.json
```

The evidence JSON lives under ignored `experiments/results/*` per `.gitignore`, so this memo records the durable hashes.
