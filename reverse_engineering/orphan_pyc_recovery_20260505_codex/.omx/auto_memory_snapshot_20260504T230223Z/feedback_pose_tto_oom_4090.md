---
name: Pose TTO OOM on 4090 with EfficientNet-B2 SegNet
description: pose_batch_pairs default 16 OOMs on RTX 4090 24GB. Cap at 8.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
**Rule:** When running `pipeline.py compress` on a 4090 (24GB), pass `--pose-batch-pairs 8`. Default of 16 OOMs because the SegNet (EfficientNet-B2) gradient graph + the 5-stage forward + the 600-pair pose tensor consumes 22.5GB+.

**Why:** 2026-04-26 SHIRAZ post-training crashed at step_pose_tto with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 216.00 MiB.
GPU 0 has a total capacity of 23.52 GiB of which 41.75 MiB is free.
Including non-PyTorch memory, this process has 23.47 GiB memory in use.
```
The crash was inside `timm/models/_efficientnet_blocks.py` BatchNorm forward — i.e., the SegNet backbone running through the differentiable graph during pose TTO.

**How to apply:**
- `scripts/remote_pose_tto_bootstrap.sh` now passes `--pose-batch-pairs 8` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default.
- For ad-hoc invocations of `experiments/pipeline.py compress`, ALWAYS add `--pose-batch-pairs 8` on a 4090. On A100 (40GB) you can use 16 again.
- A100 80GB can probably handle 32. NEVER trust the comment in PipelineConfig that says "50 OOMs" — even 16 OOMs on the most common GPU we use.
