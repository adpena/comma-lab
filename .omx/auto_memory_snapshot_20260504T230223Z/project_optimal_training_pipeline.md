---
name: Optimal Training Pipeline
description: The correct pipeline for maximum compute efficiency across local and cloud resources. MLX Phase 1 locally, PyTorch Phase 2 on CUDA.
type: project
originSessionId: ecc348c1-2829-48d3-9d4c-10ed2aa188a8
---
## The Pipeline

### GPU Renderer Lane (to catch Quantizr at 0.60)

**Phase 1: MLX Pre-training (LOCAL, FREE, FAST)**
- Run on M5 Max Metal GPU via MLX (4.7x faster than PyTorch MPS)
- L1 reconstruction loss only — no scorer needed
- `python -m tac.experiments.train_renderer_mlx --precomputed precomputed_local --epochs 500`
- Output: MLX checkpoint → convert to PyTorch via `mlx_to_pytorch()`

**Phase 2: Scorer Fine-tuning (MODAL A10G, ~$5)**
- Transfer Phase 1 weights from MLX to PyTorch
- Fine-tune with scorer loss on CUDA (10x faster than MPS)
- `--resume-from <phase1_checkpoint>` flag
- Output: PyTorch checkpoint with scorer-optimized weights

**Phase 3: Quantization + Evaluation**
- FP4 QAT on best checkpoint
- Auth eval via `tools/auth_eval.sh`

### CPU Postfilter Lane (target sub-1.0)

**Training (MODAL A10G or local MPS)**
- `python -m tac lossy --profile proven_baseline --precomputed precomputed_local`
- MPS is slow (~4 min/epoch) — use Modal for serious runs
- Precomputed data eliminates 5-min video decode

**Techniques to stack:**
- TTO at inflate time (zero training cost)
- Multi-pass inflate (zero training cost)
- LSQ fine-tuning (on best checkpoint)
- Ensemble top-K checkpoints
- CRF sweep (33, 34, 35)

### Compute Fleet Assignment

| Platform | Best Use | Cost |
|----------|----------|------|
| M5 Max MLX | Phase 1 renderer pre-training | FREE |
| M5 Max MPS | Quick smoke tests, TTO, inflate | FREE |
| Modal A10G | Phase 2 scorer fine-tune, serious training | ~$1/hr |
| Kaggle P100 | Long overnight runs (30h/week) | FREE |
| Lightning T4 | Secondary experiments | FREE |
| AWS free tier | CRF sweeps, ensemble, packaging | FREE |

### Key Requirements for Modal
- Upstream scorer repo must be cloned in image (PoseNet/SegNet modules)
- Precomputed data on Modal volume (skip 5-min decode)
- `PYTHONPATH=/root/src:/root/upstream`
- `add_local_dir` must be LAST in image build
- Guard REPO_ROOT with try/except for container paths

**Why:** MPS training is 10x slower than CUDA for scorer forward+backward. MLX pre-training is free and fast. Only spend Modal credits on Phase 2 (scorer-dependent).
