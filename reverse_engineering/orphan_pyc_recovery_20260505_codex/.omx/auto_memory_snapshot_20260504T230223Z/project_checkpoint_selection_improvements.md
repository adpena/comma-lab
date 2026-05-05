---
name: Best-checkpoint int8 selection improvements
description: Six concrete ways to improve the key breakthrough mechanism — from evaluating more pairs to LSQ learned quantization scales
type: project
---

The best-checkpoint int8 selection mechanism (evaluating EMA weights after
simulated int8 quantization every N epochs, saving the best) was the key
breakthrough that unlocked h=48 (1.762) and h=64 (1.727). It closes the
2.25× train-to-deploy gap by finding the rare epoch where quantized weights
happen to perform well.

Six improvements ranked by impact:

1. **LSQ (Learned Step Size Quantization)** — make the int8 scale a learnable
   parameter in FakeQuantSTE. The model learns weight distributions that
   quantize cleanly BY CONSTRUCTION. Eliminates the scanning lottery entirely.
   Expected: -0.05 to -0.12. 5-line code change.

2. **Full 600-pair evaluation** for checkpoint selection (vs current 150).
   More accurate selection catches the pairs that matter for final score.
   Cost: 4× eval time per checkpoint. Expected: -0.01 to -0.03.

3. **Actual save/load round-trip** in checkpoint eval instead of simulated
   quantize_state_dict. Catches serialization drift. Expected: -0.005.

4. **Top-K checkpoint averaging** — SWA but only over the K best int8
   checkpoints. Combines the noise-cancellation of averaging with the
   quality gate of int8 selection.

5. **Quantization-loss-weighted EMA** — Kalman idea applied to quant noise.
   Epochs that quantize badly get lower EMA weight.

6. **Per-layer independent scale scanning** — find the best quantization
   scale per layer independently instead of globally.
