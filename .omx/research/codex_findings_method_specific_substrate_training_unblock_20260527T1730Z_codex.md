# Codex Findings - Method-Specific Substrate Training Unblock

UTC: 2026-05-27T17:30Z

## Scope

Closed two more local-training gaps without forcing every substrate through the
same abstraction:

- `mdl_ibps_j_discrete_categorical_mine_hybrid` now has a trainable MLX
  `nn.Module` renderer (`MDLIBPSJTrainableRendererMLX`) and routes `--mode full`
  through the canonical `mlx_score_aware` harness against real contest-video
  targets.
- `faiss_ivf_pq_residual` now has an honest `--full` codebook-fitting trainer:
  deterministic K-means over real-video residual tiles, FAISSPQ1 archive
  emission, archive hash, and residual reconstruction MSE. This deliberately
  does not use the gradient score-aware harness because PQ assignment is an
  argmin-over-centroids codebook method, not a differentiable renderer.
- Added MLX harness-unlock tests for ATW, COIN++, and MDL trainable renderers.

All outputs remain false-authority local research signal:
`score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`.

## Local Execution Evidence

MDL/IBPS one-pair MLX full-path smoke:

```bash
.venv/bin/python experiments/train_substrate_mdl_ibps_j_discrete_categorical_mine_hybrid.py \
  --mode full \
  --output-dir .omx/research/mlx_score_aware_cli_smoke_mdl_20260527Tlocal \
  --video-path upstream/videos/0.mkv \
  --num-pairs 1 \
  --epochs 1 \
  --full-lr 1e-3 \
  --distillation-weight 0.0
```

Result: `epochs=1 promotable=False wall=0.1s`.

FAISS IVF-PQ one-pair codebook-fit smoke:

```bash
.venv/bin/python experiments/train_substrate_faiss_ivf_pq_residual.py \
  --full \
  --output-dir .omx/research/faiss_ivf_pq_codebook_fit_smoke_20260527Tlocal \
  --video-path upstream/videos/0.mkv \
  --num-pairs 1 \
  --m-sub-quantizers 2 \
  --ksub-codebook-size 2 \
  --tile-h 192 \
  --tile-w 256 \
  --kmeans-iters 2 \
  --seed 42
```

Result: `residual_recon_mse=0.0002106040`,
`archive_bytes=391813`, `promotable=false`.

Artifact hashes:

- MDL training artifact:
  `cc5255c177e05068f6a784dc9b6a39639feabac5c3a86826f5a6b6ffef493535`
- MDL EMA checkpoint:
  `df77cf7873f82b3756b4c7f879525d8a216ee18f61e95c466b4e58b5d3500b35`
- MDL live checkpoint:
  `ee8f2ff0631a120c894eb5bfa40ea69121e98e207655a86fc5e74316ae83a097`
- FAISS `0.bin`:
  `7f67a0423c3f99c2cc88def5f4e8cdbc726bc3e8a3b34282eede3ea5fccf6f12`
- FAISS `codebook_fit_stats.json`:
  `483f1da3318dc4a541b19c88b5c4070219fbf2503c7d8915fa6720acb41bebd7`

## Verification

- `ruff` on the touched FAISS/MDL trainers, MDL renderer, and MLX unlock tests.
- `py_compile` on the touched FAISS/MDL trainers, MDL renderer, and MLX unlock
  tests.
- `pytest src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/tests/test_mlx_harness_unlock.py
  src/tac/substrates/atw_v2_cooperative_receiver_v2/tests/test_mlx_harness_unlock.py
  src/tac/substrates/coin_pp_implicit_neural_representation/tests/test_mlx_harness_unlock.py
  -q`: 12 passed.
- `pytest src/tac/substrates/faiss_ivf_pq_residual/tests/test_basic.py
  src/tac/substrates/mdl_ibps_j_discrete_categorical_mine_hybrid/tests -q`:
  70 passed.
- `tools/lane_maturity.py validate`: 1440 lanes validated cleanly.

## Remaining Promotion Gates

- FAISS residual MSE is a proxy training diagnostic, not a contest score.
- MDL/IBPS still defers MINE critic integration into the loss, PyTorch export,
  byte-closed archive builder, component decomposition, paired CPU/CUDA exact
  anchors, and Catalog #319 deliverability proof.
- All four substrate-local paths now need queue-owned acquisition policy so
  MLX local results automatically route to exact-eval spend candidates only
  after the false-authority gates pass.
