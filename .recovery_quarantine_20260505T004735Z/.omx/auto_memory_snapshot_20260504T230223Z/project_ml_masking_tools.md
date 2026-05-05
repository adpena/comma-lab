---
name: ML Masking Pipeline Tools
description: Available ML tools for generating importance masks — SAM 3, Falcon Perception, masking strategies
type: reference
---

Installed tools:
- **SAM 3 MLX** (`workspace/tools/mlx_sam3/`): Native Apple Silicon via mlx_sam3. HuggingFace weights: mlx-community/sam3.1-bf16.
- **SAM 3 CUDA** (`workspace/tools/sam3/`): Requires Triton — use on bat00 (RTX 2070 Super).
- **SAM 2.1**: Installed as fallback in sam3 venv. Works on MPS.
- **Falcon Perception** (`workspace/tools/falcon-perception/`): MLX backend, Apple Silicon. 0.6B params. Text-prompted segmentation. Outperforms SAM 3 on mask quality (68.0 vs 62.3 Macro-F1).
- **Grounded-SAM-2**: GroundingDINO + SAM 2 for text-prompted detection+segmentation+tracking.
- **Mask2Former/OneFormer**: Via HuggingFace Transformers, Cityscapes-pretrained.
- **comma10k**: 10k labeled driving images (MIT license) for validation.

Masking script: `submissions/robust_current/ml_mask_generator.py`
Strategies: specific (5 driving classes), general ("everything dynamic and salient"), hybrid (union).
Toolchain priority: SAM 3 MLX > SAM 3 CUDA > SAM 2.1 > gradient+temporal fallback.

User wants to experiment with various prompts including general prompts like "segment everything that is dynamic and salient to a self-driving car from the dashcam footage."
