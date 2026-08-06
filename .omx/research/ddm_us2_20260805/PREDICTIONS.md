# ddm_us2 predict-then-diff predictions

Created before reading the upstream source surfaces for this arm.

## Scope

Arm charter: `.omx/tmp/codex_runs/us2_prompt.md`.
Common contract: `.omx/tmp/codex_runs/_common_contract.md`.
Axis: `$0` source/dependency analysis only; no scorer runs, no dispatches, no edits under
`upstream/`.

## Predictions Before Source Read

| surface | prediction before read |
|---|---|
| `upstream/evaluate.py` | It loads the submission directory, runs `inflate.sh`, computes `compressed_size` from `archive.zip` only, decodes the generated video and GT with the same frame utility, evaluates 600 non-overlapping pairs, computes `d_seg`, `d_pose`, and returns/prints a rounded final score. It likely has no runtime term except the contest external timeout and may use direct `Path.rglob`/file-name assumptions for output discovery. |
| `upstream/modules.py` | It defines frozen `SegNet` and `PoseNet` wrappers plus preprocessing. Prediction from corpus: both scorers resize with `torch.nn.functional.interpolate(..., mode="bilinear")` to the literal SegNet input size before their task-specific normalization; no explicit `align_corners` or `antialias` kwargs unless recently drifted. Pose preprocessing uses `rgb_to_yuv6`, and wrapper forward methods probably have `torch.no_grad()` or non-differentiable helper boundaries inherited from challenge code. |
| `upstream/frame_utils.py` | It owns the authoritative GT/video decode path. Prediction: PyAV decodes YUV420 into RGB via a custom integer-ish YUV conversion path, not naive `rgb24`, and frame pairing/order is container-order deterministic enough for the official scorer. It likely converts generated RGB video frames into torch tensors with uint8 clamping already fixed by video encoding rather than an explicit differentiable round path. |
| torch interpolate dependency | Prediction: exact semantics are PyTorch defaults for bilinear 4D tensors: `align_corners=False` when unspecified and `antialias=False` unless explicitly set. If upstream omits kwargs, our in-loop R must match those defaults exactly rather than cargo-culting a named kwargs set. |
| scorer checkpoint objects | Prediction: SegNet is `segmentation_models_pytorch` Unet with `timm-efficientnet-b2` encoder, 5 classes, and checkpoint-loaded weights including BN running stats; PoseNet is FastViT-T12-like regression with a 6-output target after preprocessing. The checkpoints may encode exploitable BN/statistical or stride/null-space facts not captured by file-level code. |
| runtime closure | Prediction: README sets 30-minute decode budget and rule-118 counted-artifact boundary; CI likely installs dependencies through `uv`/lock, invokes evaluation on a submission directory, and may not enforce the same exact host versions as local. `uv.lock` drift may include extra packages that matter only if they change scorer/import resolution. |
| `upstream/videos/0.mkv` metadata | Prediction: the official GT is a single MKV with 1200 frames / 600 non-overlapping pairs; stream metadata may encode pixel format, time base, frame rate, color range/space, and decode determinism details that current tooling consumes implicitly. |
