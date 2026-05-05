---
name: Vast.ai instance with old CUDA driver silently falls back to CPU — score becomes [advisory only]
description: 2026-05-01 ~11:32 UTC. Instance 35956905 had nvidia driver version 12070 (~12.0.7) but PyTorch 2.5.1+cu124 needs ≥12.4. torch.cuda.is_available() returned False, inflate ran on CPU, scorer would have run on CPU too. Per CLAUDE.md, CPU scores are [advisory only], NOT contest-CUDA. Cost: 5 min wasted GPU + 2 min destroy/relaunch. Fix: ALWAYS filter `cuda_vers>=12.4` in Vast.ai search; verify `nvidia-smi --query-gpu=driver_version` ≥ 535 at Stage 0; FAIL LOUD if torch.cuda.is_available() is False.
type: feedback
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---

## Bug class: silent CPU fallback after CUDA init failure

The chain log showed:
```
torch=2.5.1+cu124 cuda_available=True   ← LIES (Stage 0 banner from before torch fully loaded)
cuda_device=NVIDIA GeForce RTX 4090     ← LIES
...
[Later inside inflate_renderer.py:]
UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 12070).
Device: CPU (32 cores)
```

The `remote_archive_only_eval.sh` Stage 0 check uses a separate Python process that imports torch lightly and prints `torch.cuda.is_available()`. The real CUDA init only happens when the inflate pipeline tries to use CUDA — at which point it silently falls back to CPU with only a `UserWarning`.

Per CLAUDE.md MPS-falsification trap rule: **scores from CPU device may NOT be tagged `[contest-CUDA]`.** They are `[advisory only]` at best.

## Why: torch lazy CUDA init

`torch.cuda.is_available()` returns True if the NVIDIA libraries can be loaded, even if device init would fail. The real check is `torch.cuda.init()` or actually running a kernel. Stage 0 banner check was insufficient.

## How to apply

1. **Vast.ai search filter** must include `cuda_vers>=12.4` for any instance running PyTorch 2.5+:
   ```bash
   .venv/bin/vastai search offers 'gpu_name=RTX_4090 cuda_vers>=12.4 reliability>0.95 ...'
   ```

2. **Vast.ai create command** should pin a known-good PyTorch image:
   ```bash
   .venv/bin/vastai create instance $OFFER --image 'pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel' ...
   ```

3. **`scripts/remote_archive_only_eval.sh` Stage 0** should ACTUALLY exercise CUDA, not just probe availability:
   ```python
   import torch
   assert torch.cuda.is_available(), 'no CUDA'
   x = torch.zeros(10, device='cuda') + 1   # actual CUDA op
   assert x.device.type == 'cuda'
   y = x.matmul(x.t())  # forces kernel launch
   torch.cuda.synchronize()
   ```
   If this fails on driver-too-old, exit FATAL not warn.

4. **Preflight check** (proposed): `check_remote_archive_eval_exercises_cuda_at_stage_0` — scan all `scripts/remote_*.sh` for "Stage 0" and verify they include a real CUDA kernel launch, not just `is_available()`.

## Today's incident

- Old instance: 35956905 (offer 28758519, cuda_vers 12.0.7) — DESTROYED
- New instance: 35957332 (offer 25277946, cuda_vers 12.4) — CHAIN LIVE
- Cost wasted: ~$0.10 + 5 min wall-clock

## Sister bug class: uv-not-on-PATH (`feedback_uv_not_on_path_vast_instance_20260501.md`)

Same dispatch path produced 2 distinct silent-failure bugs in succession. Both fall into the "Stage 0 doesn't validate runtime preconditions hard enough" class. The fix for both is:

```bash
# Stage 0 should be a hard precondition gate, not a friendly banner.
# FAIL LOUD on:
#   - missing uv (raise, not warn)
#   - CUDA init failure (raise, not warn)
#   - missing upstream/{videos,models} (raise, not warn)
#   - wrong PyTorch CUDA wheel for system driver (raise, not warn)
```
