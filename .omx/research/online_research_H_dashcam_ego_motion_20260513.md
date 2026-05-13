# Online research ledger — Domain H: Dashcam / ego-motion / video

Per-paper notes; 9 entries.

---

## H.1 — RAFT (Teed-Deng, ECCV 2020 Best Paper)
- **arXiv**: https://arxiv.org/abs/2003.12039
- **Empirical claim**: GRU-based recurrent refinement of all-pairs correlation; SOTA optical flow on KITTI / Sintel.
- **Relevance**: Foundation for any optical-flow side-info pipeline. Our contest scorer is downstream of motion.

## H.2 — GMFlow (Xu et al., CVPR 2022)
- **arXiv**: https://arxiv.org/abs/2111.13680
- **Empirical claim**: Global matching via transformer attention; no iterative refinement.
- **Relevance**: Architectural reference. Less directly applicable than RAFT.

## H.3 — SEA-RAFT (Eslami et al., 2024)
- **arXiv**: https://arxiv.org/html/2405.14793v1
- **Empirical claim**: Simple+Efficient+Accurate RAFT; best zero-shot on KITTI.
- **Relevance**: Plug-in successor to RAFT. Already wired in our scaffold.

## H.4 — FlowSeek (Poggi et al., 2025)
- **arXiv**: https://arxiv.org/html/2509.05297v1
- **Empirical claim**: Combines depth-foundation-models with motion bases for flow estimation. Easier inference.
- **Relevance**: Latest from optical-flow community; tracks the depth-foundation-model trend.

## H.5 — NeuFlow (Zhang et al., 2024)
- **arXiv**: https://arxiv.org/html/2403.10425v1
- **Empirical claim**: Real-time + high-accuracy flow on EDGE devices.
- **Relevance**: For our contest the encoder side is unlimited compute, but the inflate side has 30-min T4 budget. NeuFlow shows real-time-on-edge flow is feasible.

## H.6 — comma.ai openpilot research
- **GitHub**: https://github.com/commaai/openpilot
- **Relevance**: The CONTEXT of our contest. Pose + segmentation are downstream of openpilot's actual driving stack.

## H.7 — DROID-SLAM (Teed-Deng, NeurIPS 2021)
- **arXiv**: https://arxiv.org/abs/2108.10869
- **Empirical claim**: Deep visual SLAM; pose + structure estimation.
- **Relevance**: Strong pose-estimation reference. The TYPE of ego-motion estimate is what our PoseNet target encodes.

## H.8 — DeepVO / classical visual odometry
- **Reference**: Wang et al. 2017+; foundational deep VO.
- **Relevance**: Reference for understanding what PoseNet actually computes.

## H.9 — Lee Tau / Time-to-collision (classical Gibson optic flow)
- **Reference**: D. N. Lee 1976; visually-guided braking; tau = optical-flow-divergence-based TTC.
- **Relevance**: Foundation for understanding pose-relevance. The scorer ultimately measures pose-fidelity because pose drives the braking-decision relevant signal.

## H.10 — Cityscapes / BDD100K / comma10k datasets
- **References**: Public driving-dataset benchmarks.
- **Relevance**: Our contest video is implicitly drawn from a similar distribution. Cross-dataset generalization claims should be tested before assuming external improvements transfer.

---

## Domain-H insights for our contest

1. **Optical flow as side-info**: A high-quality flow field between consecutive frames is reconstructible from neighbors. Per Wyner-Ziv (Domain G), if our inflate side has frame N-1 reconstruction, we can encode frame N as a flow-residual rather than full-frame. **[literature-prediction: 2-5× rate reduction for non-keyframes via flow-residual coding]**.

2. **Pose head fidelity**: PoseNet's pose-12-dim output is over-parameterized; only first 6 dims matter. Compression schemes can exploit this (already in PR93 pose codec primitives).

3. **Mask is a derivative of frame**: SegNet operates on the LAST FRAME only of each pair. Storing mask explicitly is partially redundant with frame fidelity — the Wyner-Ziv view says we can amortize mask cost against frame reconstruction.

## Follow-up reads:
- Hierarchical Motion Field Alignment (PMC 2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12074433/
- Optical-flow-for-edge-devices (NeuFlow-v2): https://arxiv.org/abs/2403.10425
- comma.ai blog posts on openpilot perception stack
