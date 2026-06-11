# De-risk survey: existing Metal/MLX/PyTorch-GPU OSS — don't hand-port the scorer from scratch (2026-06-11)

**Operator ask:** "search online for Metal/MLX PyTorch backends/projects/OSS/DSL so we don't do a lot of
work for nothing." **Verdict: MLX is confirmed the only fidelity-safe GPU path, and substantial reuse
exists — we should NOT hand-port EfficientNet/Unet/FastViT from scratch.**

## The fidelity/GPU matrix (confirmed)
| Path | GPU? | Fidelity vs torch-CPU authority | Verdict |
|---|---|---|---|
| torch-CPU | NO (no Metal backend for PyTorch) | AUTHORITY | the oracle; but no GPU |
| torch **MPS** | yes | TRASH (CLAUDE.md: 23× PoseNet drift, corrupts 95.5% orderings) | FORBIDDEN |
| torch.compile **Metal/Inductor** (MetalKernel) | yes | same MPS numerics underneath (speed layer on MPS) | NOT a fidelity escape |
| **CoreML** (coremltools) | yes (GPU/ANE) | **defaults to FP16** (fidelity loss); FP32 forceable but ANE/GPU numerics uncertain | risky for argmax-flip d_seg |
| **MLX (CPU + GPU)** | yes | HIGH (operator-trusted; numpy-like IEEE) | **THE path** |

So the operator's instinct is right: MLX is the only GPU-fast path that preserves the exact-scorer
fidelity d_seg (argmax-flip-sensitive) requires.

## Reusable OSS — what covers OUR scorer (SegNet = smp.Unet(efficientnet_b2, 5-class); PoseNet = FastViT-T12 + Hydra head)
- **mlx-image** (riccardomusmeci) — has **EfficientNet-B0..B7 incl. B2** (SegNet's BACKBONE) with **automatic
  PyTorch .pth → MLX safetensors conversion + ImageNet-1K-validated parity**. → REUSE the SegNet encoder +
  its weight-conversion machinery. (Encoders only — NO segmentation/Unet decoder.)
- **apple/ml-fastvit** — the OFFICIAL FastViT (PoseNet backbone) is Apple's own (torch + CoreML export).
  No ready MLX port found, but it's the authoritative reference to port FastViT-T12 from (Apple authored
  both FastViT and MLX).
- **YOLO-MLX** (thewebAI) — proves CNN + a **segmentation/mask DECODER head** runs in MLX at parity and
  **2.07× faster than MPS** → a working reference that Unet-style decoders are feasible+fast in MLX.
- **torch2mlx** (SynapticSage) / **Xforge** (SattamAltwaim) — general PyTorch→MLX converters; **Xforge has
  built-in parity testing** + auto-detects ViT/ResNet. → conversion + parity-harness reuse.
- **ExecuTorch MLX delegate** (PyTorch-OFFICIAL, 2025) — runs torch models on the Apple GPU via MLX,
  supports **FP32** + "vision / encoder-decoder" models, BUT limited to **~90 ATen ops** (transformer-
  focused) and "cannot run arbitrary torch.nn modules" — unsupported ops fail export.

## The strategy (avoids hand-porting from scratch)
**Tier 1 — try the ZERO-PORT path first (a few-hour spike):** export the actual `upstream/modules.py`
SegNet + PoseNet through the **ExecuTorch MLX delegate at FP32**, run on the Apple GPU, and check d_seg
(argmax-flip rate) + d_pose vs torch-CPU. If both nets fit the ~90 ATen ops → **near-zero porting**,
GPU-fast, fidelity-gated. Highest payoff if it works.

**Tier 2 — if the spike fails on custom ops (likely: smp-Unet decoder, FastViT RepMixer, the Hydra head):
reuse backbones + port ONLY the heads:**
- SegNet = **mlx-image EfficientNet-B2** (parity-validated, auto weight-convert) + **port the Unet decoder**
  (YOLO-MLX as the reference that MLX decoders work) + the 5-class seg head.
- PoseNet = **port FastViT-T12 from apple/ml-fastvit** (official reference) + the Hydra pose head.
- Use **Xforge/torch2mlx** for the weight conversion + an automated parity harness.
This reuses the two heavy backbones + all weight-conversion + parity tooling; we hand-port only the small
custom heads (not EfficientNet-B2 or FastViT layer-by-layer from scratch).

**Invariant either way:** torch-CPU stays the ONLY authority. The MLX scorer is the fast research-signal;
a periodic torch-CPU parity check (argmax-flip count for d_seg, relative MSE for d_pose) certifies it. Any
MLX scorer piece ships with a real-input torch-parity GATE (not zero-init — the grid-PE fake-parity lesson).

## What this changes (routes the in-flight MLX subagents)
The two running MLX subagents (drift audit a642…, comprehensive takeover a458…) map the CURRENT state +
torch authority + drift sources — still needed. This memo redirects the BUILD phase of the takeover from
"hand-port the scorer" to "Tier-1 ExecuTorch spike → else reuse mlx-image EfficientNet + apple/ml-fastvit +
converters, port only the heads." Fold this into the takeover roadmap when synthesizing their outputs.

## Sources
mlx-image (github.com/riccardomusmeci/mlx-image) · apple/ml-fastvit (github.com/apple/ml-fastvit) ·
ExecuTorch MLX delegate (pytorch.org/blog/running-pytorch-models-on-apple-silicon-gpus-with-the-executorch-mlx-delegate) ·
torch2mlx (github.com/SynapticSage/torch2mlx) · Xforge (github.com/SattamAltwaim/Xforge) ·
YOLO-MLX (github.com/thewebAI/yolo-mlx) · coremltools convert-pytorch (apple.github.io/coremltools).
