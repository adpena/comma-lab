# Codex Findings: MLX Auth-Scorer Port Inventory + Hash-Identity Blocker

timestamp_utc: 2026-05-21T21:48:28Z
agent: codex
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_grade: macOS-MLX-research-signal
score_claim: false
promotion_eligible: false

## Summary

Codex landed a fail-closed MLX scorer-port inventory surface for the upstream contest scorer. This is a port-planning and coverage artifact only; it does not claim local MLX scorer parity.

New reusable surfaces:

- `src/tac/local_acceleration/mlx_scorer_port_inventory.py`
- `tools/plan_mlx_scorer_port.py`
- `src/tac/tests/test_mlx_scorer_port_inventory.py`

Generated inventories:

- `experiments/results/mlx_scorer_port_inventory_20260521.json`
- `experiments/results/mlx_scorer_port_inventory_with_weights_20260521.json`

## Empirical Inventory Result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/plan_mlx_scorer_port.py \
  --repo-root . \
  --load-weights \
  --output experiments/results/mlx_scorer_port_inventory_with_weights_20260521.json
```

Result:

```json
{
  "total_modules": 1156,
  "total_leaf_modules": 768,
  "total_parameter_count": 23445971,
  "total_parameter_bytes": 93783884,
  "total_buffer_count": 108414,
  "total_buffer_bytes": 434320,
  "total_blocking_modules": 682,
  "total_unknown_modules": 0,
  "total_state_dict_key_count": 1160,
  "total_state_dict_tensor_bytes": 94218204,
  "full_mlx_port_claim_allowed": false
}
```

Interpretation: the inventory has no unclassified module classes after adding direct rules for `SiLU`, `Sigmoid`, and `Flatten`, but it still has 682 blocking adapter/composite module instances and 1,160 upstream state-dict keys requiring explicit MLX tensor-name, dtype, shape, layout, and transform mapping. A full MLX auth-scorer port is not claimable until those blockers have tested implementations and forward-parity evidence.

Primary blocker families:

- `torch.nn.Conv2d`: requires explicit NCHW/BHWC layout contract, grouped/depthwise parity, and weight mapping.
- `torch.nn.BatchNorm1d` / `BatchNorm2d`: requires eval-mode running-stat parity and channel-axis handling.
- `timm` FastViT PoseNet composites: requires faithful RepMixer/MobileOne/GELUTanh/LayerScale rewrite or validated equivalent.
- `segmentation_models_pytorch` EfficientNet-B2/Unet composites: requires faithful encoder/decoder rewrite or validated equivalent.
- `torch.interpolate` scorer preprocess: already covered by scorer-input hash identity at the cache surface, but a full MLX scorer must preserve this exact preprocessing behavior.

## Modal CPU Hash Anchor

Recovered detached Modal CPU auth eval with scorer-input hashes:

- Output dir: `experiments/results/modal_auth_eval_cpu/mlx_fec6_scorer_input_hash_cpu_20260521T214030Z`
- Call id: `fc-01KS67R1HR4T63VQ95201KBVHB`
- Archive SHA-256: `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Score axis: `contest_cpu`
- Score recomputed from components: `0.1920513168811056`
- Scorer-input hash artifact: `scorer_input_cache_hashes.json`

## Transfer-Calibration Audit

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_mlx_scorer_input_cache.py \
  --cache-manifest experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/manifest.json \
  --auth-eval experiments/results/modal_auth_eval_cpu/mlx_fec6_scorer_input_hash_cpu_20260521T214030Z/contest_auth_eval.json \
  --reference-cache-manifest experiments/results/modal_auth_eval_cpu/mlx_fec6_scorer_input_hash_cpu_20260521T214030Z/scorer_input_cache_hashes.json \
  --output experiments/results/modal_auth_eval_cpu/mlx_fec6_scorer_input_hash_cpu_20260521T214030Z/cache_vs_modal_cpu_hash_audit.json
```

Result: `FAIL_CACHE_AUTH_EVAL_IDENTITY`.

Blockers:

- `inflated_outputs_aggregate_sha256_mismatch_or_missing`
- `raw_sha256_mismatch_or_missing`
- `scorer_input_array_sha256_mismatch:segnet_last_rgb`
- `scorer_input_array_sha256_mismatch:posenet_yuv6_pair`

The pair-index tensor hash matches, so topology is aligned. The raw surface does not match: the local cache was built from the macOS advisory inflate output (`raw_sha256=d1afc583...`), while Modal Linux contest-CPU produced `raw_sha256=fef02ccd...`. Therefore this local cache is debug-only and must not be used for auth-axis transfer calibration.

## Verdict

`NEEDS_MATCHING_RAW_SURFACE_BEFORE_LOCAL_MLX_TRANSFER_CALIBRATION`.

The MLX path remains valuable for local training throughput, but the current calibrated input cache must be rebuilt from a raw/scorer-input surface that matches the recovered Modal Linux hash artifact, or the runtime must be made deterministic across macOS and Linux and re-audited.

## Conformance Bugs Fixed

The adversarial review agent identified three MLX-bridge conformance bugs during this turn. Codex fixed them before landing:

1. `--allow-missing-inflated-output-identity` no longer permits a present inflated-output hash mismatch. It only relaxes missing custody; present mismatch remains a blocker.
2. MLX training-signal fidelity now requires `evidence_grade="macOS-MLX-research-signal"` on the MLX payload. Missing grade is a blocker.
3. `tools/build_mlx_scorer_input_cache.py` now rejects `--max-pairs < 1` and `--batch-pairs < 1`; the underlying cache writer also rejects nonpositive `max_pairs`.

## Next Actions

1. Decide how to obtain the matching scorer-input tensors for local MLX training: download/materialize the Modal Linux raw/scorer-input tensor surface through a large-artifact path, or make local inflate reproduce `raw_sha256=fef02ccd...`.
2. Implement tested MLX adapters in dependency order: scorer preprocess parity, Conv2d layout/weight mapping, BatchNorm eval parity, then PoseNet FastViT blocks and SegNet EfficientNet/Unet blocks.
3. For every adapter, add fixed-input PyTorch-vs-MLX tensor parity tests before using it in a training loop.
4. Keep all MLX outputs marked `macOS-MLX-research-signal`, `score_claim=false`, and `promotion_eligible=false` until paired contest CPU/CUDA auth eval validates byte-closed outputs.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_scorer_fidelity.py \
  src/tac/tests/test_mlx_scorer_port_inventory.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_modal_auth_eval.py \
  -q
```

Observed: `68 passed`.
