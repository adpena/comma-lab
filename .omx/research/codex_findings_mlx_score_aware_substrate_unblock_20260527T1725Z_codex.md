# Codex Findings - MLX Score-Aware Substrate Unblock

UTC: 2026-05-27T17:25Z

## Scope

Closed the next MLX-local substrate execution gap after the shared
`tac.substrates._shared.mlx_score_aware` harness refactor:

- `atw_v2_cooperative_receiver_v2` now has a trainable MLX `nn.Module` wrapper
  that exposes the canonical `reconstruct_pair(idx) -> (rgb_0, rgb_1)` NCHW
  `[0, 1]` harness contract.
- `experiments/train_substrate_atw_v2_cooperative_receiver_v2.py --full` routes
  through the canonical MLX score-aware harness against real contest-video
  targets.
- `coin_pp_implicit_neural_representation` now has a trainable MLX COIN++
  renderer with per-pair modulation codes plus a shared FiLM-conditioned
  coordinate MLP.
- `experiments/train_substrate_coin_pp_implicit_neural_representation_mlx.py
  --full` routes through the same harness, with local width/depth flags for
  deterministic one-pair MLX smoke without changing the full default topology.

All outputs remain non-promotable `[macOS-MLX research-signal]`:
`score_claim=false`, `promotion_eligible=false`,
`ready_for_exact_eval_dispatch=false`.

## Local MLX Execution Evidence

ATW one-pair full-path smoke:

```bash
.venv/bin/python experiments/train_substrate_atw_v2_cooperative_receiver_v2.py \
  --full \
  --output-dir .omx/research/mlx_score_aware_cli_smoke_atw_20260527Tlocal \
  --video-path upstream/videos/0.mkv \
  --num-pairs 1 \
  --latent-dim 8 \
  --epochs 1 \
  --full-lr 1e-3 \
  --distillation-weight 0.0
```

Result: `epochs=1 promotable=False wall=0.0s`.

COIN++ one-pair full-path smoke:

```bash
.venv/bin/python experiments/train_substrate_coin_pp_implicit_neural_representation_mlx.py \
  --full \
  --output-dir .omx/research/mlx_score_aware_cli_smoke_coin_20260527Tlocal \
  --video-path upstream/videos/0.mkv \
  --num-pairs 1 \
  --mod-dim 2 \
  --pos-dim 1 \
  --hidden-dim 2 \
  --num-hidden-layers 1 \
  --epochs 1 \
  --full-lr 1e-3 \
  --distillation-weight 0.0
```

Result: `epochs=1 promotable=False wall=0.1s`.

Artifact hashes:

- ATW training artifact:
  `62adac7266b234fcd1ee3256877833bf1ae3dbadb45ac674fb00639a8d1c169c`
- COIN++ training artifact:
  `76d75108aa89b75346e1c9ab694c9253e956816f039d93aff5d7257576a62947`
- ATW EMA checkpoint:
  `b092951a87ee820fa0854ed8d3d4da540411d347f08f494d1d074d766e97d4bd`
- ATW live checkpoint:
  `6c5dc77716f9af4316c1665c1657fa79461deb36f7084f1417efbdf124b4efa1`
- COIN++ EMA checkpoint:
  `6e43389dd485ebb36eb78668cbfb66159673a965b67e714af7df8ab3bbc92586`
- COIN++ live checkpoint:
  `32452ae23ead0b2993c977eaf7ad170851f4c74cb62d4987061b87c64b0967c3`

## Verification

- `ruff` on the touched ATW/COIN++ trainer, renderer, and test files.
- `py_compile` on both trainers and both MLX renderers.
- `pytest src/tac/substrates/atw_v2_cooperative_receiver_v2/tests/test_basic.py
  src/tac/substrates/coin_pp_implicit_neural_representation/tests/test_basic.py
  -q`: 78 passed.

## Remaining Promotion Gates

- MLX rows remain local research signal only.
- PyTorch export bridges, byte-closed archive builders, paired CPU/CUDA exact
  anchors, component decomposition, and contest-axis calibration remain required
  before any promotion, rank, kill, or dispatch authority.
