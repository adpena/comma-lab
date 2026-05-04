---
name: Neural Compression Research 2025-2026
description: Latest papers and techniques for single-video neural compression. Cool-Chic, C3, self-compression, LRConv-NeRV. Prioritized for our contest.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Most Relevant Techniques (2025-2026)

### Tier 1: Directly Applicable
1. **Self-Compressing Neural Networks** (arXiv:2301.13142) — learns per-layer bit-depth during training. Replaces QAT pipeline. Could reduce archive 30%+.
2. **Cool-Chic Video Codec** (arXiv:2402.03179, Orange/CNES) — 800 params/frame overfitted codec. Beats VVC. CPU-decodable 100ms. Paradigm alternative to our renderer approach.
3. **C3: High-Performance Low-Complexity** (c3-neural-compression.github.io) — single-video overfitting with minimal decoder. Could be used as residual codec on top of our renderer.

### Tier 2: Architectural Improvements
4. **LRConv-NeRV** (arXiv:2603.18261, Mar 2026) — low-rank separable convolutions. 68% decoder complexity reduction. 9.2% bitrate savings. 1-day integration.
5. **Bias Modulation for INR** (CVPR 2025) — action-focused representation. Better PoseNet preservation.
6. **PNVC** (arXiv:2409.00953) — combines autoencoder + per-video overfitting. Matches VVC VTM-20.0.
7. **Improved Encoding for Overfitted Codecs** (arXiv:2501.16976) — motion from optical flow, joint rate allocation.

### Tier 3: Background Knowledge
8. **How to Design INR for Video Compression** (arXiv:2506.24127, Jun 2025) — comprehensive design guide
9. **COIN++** (arXiv:2201.12904) — quantized INR weights + entropy coding
10. **NeuralLVC** (arXiv:2604.03353, Apr 2026) — masked diffusion lossless codec

### Action Items
- Next training cycle: integrate self-compression (`--self-compress` flag in train_distill.py)
- If time before May 3: prototype C3 residual codec
- Post-contest / paper: Cool-Chic paradigm comparison
- Quick win: LRConv low-rank convolutions in MotionPredictor

**Why:** Current approach (renderer + QAT + TTO) is strong but leaves rate optimization on the table. Self-compression and Cool-Chic represent the next level of compression efficiency.
