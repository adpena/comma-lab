# External Research Intake — Shannon-Floor Push

**Date:** 2026-04-30  
**Purpose:** map current papers and OSS into the α/β/γ implementation plan without letting external benchmarks substitute for contest evidence.

---

## Intake Rule

External research may guide architecture and implementation. It cannot promote or kill a lane in this contest unless our exact archive is evaluated through `experiments/contest_auth_eval.py` or an equivalent `archive.zip -> inflate.sh -> upstream/evaluate.py` CUDA path.

Every external idea gets one of these labels:

- **Copy:** implementation pattern can be adapted directly.
- **Translate:** concept applies, but the contest objective or payload shape differs.
- **Watch:** useful reference, not on the critical path.

---

## α — Mask Payload Overhaul

| Source | Type | Intake | Contest translation |
|---|---|---|---|
| NeRV official implementation, `haochen-rye/NeRV` | OSS + NeurIPS paper | Copy architecture/compression recipe | Current Lane 12 is a coordinate/payload MLP; compare our `nerv_mask_codec.py` against official frame-wise INR compression and model-quantization loop. |
| HiNeRV project / NeurIPS 2023 | paper + project | Translate | Hierarchical encoding improves INR video compression. For masks, use hierarchy only if flat Lane 12 underfits boundaries. |
| Boosting NeRV official implementation | OSS + CVPR 2024 | Translate | Conditional decoder and high-frequency reconstruction loss are directly relevant if Lane 12 boundary disagreement remains high. |
| FFNeRV / flow-guided NeRV family | paper/project | Watch | Flow guidance could pair with half-frame or RAFT/radial pose preimage, but it may add side-info and inflate-time complexity. |

Immediate α changes to consider after Lane 12 CUDA truth:

1. Add boundary-weighted CE or high-frequency loss if disagreement localizes at mask boundaries.
2. Add hierarchical/coarse-to-fine NeRV variant if flat hidden=64 underfits.
3. Add half-frame NeRV variant only after full-frame archive lands.
4. Keep VQ-VAE/wavelet as parallel backup, but do not preempt NeRV until its exact archive score is known.
5. Add a boundary-residual sidecar over any neural base only if the residual is sparse under actual archive bytes. Candidate encodings: chain-code/RLE varints, 3-bit class bit planes, and mixed pure/boundary blocks.

---

## β — Sensitivity-Aware Renderer Compression

| Source | Type | Intake | Contest translation |
|---|---|---|---|
| HAWQ / HAWQ-V3 | papers | Copy objective form | Use Hessian/Fisher layer or channel sensitivity to solve mixed-precision allocation under a byte budget. |
| AWQ official implementation | OSS + MLSys 2024 | Translate | Activation-aware protection maps naturally to scorer-activation-aware protection. Protect channels that amplify PoseNet/SegNet loss. |
| GPTQ-style second-order quantization | paper/OSS family | Translate | The local renderer is small enough for block/channel second-order error compensation, but inflate format must stay simple. |
| SmoothQuant-style scale migration | paper/OSS family | Watch | Might help block-FP scaling, but any activation rescale must be encoded or reproducible at inflate. |

Immediate β implementation targets:

1. Define sensitivity as score-gradient energy per channel, not generic reconstruction loss.
2. Use train/holdout split for sensitivity stability before trusting the map.
3. Implement mixed precision as a simple per-channel bitmask: protected fp16, medium fp8/fp6, safe fp4.
4. Store the bitmask and per-channel scales inside `renderer.bin`; do not require sensitivity computation at inflate.
5. Validate against Ω-W-V2's exact failure: recover rate savings without PoseNet regression.
6. Add allocation beyond thresholds next: Fisher first, Hutchinson trace optional, then byte-measured greedy/DP knapsack. AWQ/SmoothQuant/GPTQ-style compensation stays behind equivalence and byte-gate tests.

---

## γ — Joint Stack / Entropy / MDL

| Source | Type | Intake | Contest translation |
|---|---|---|---|
| CompressAI entropy models | OSS | Copy for experiments, not necessarily runtime | Use EntropyBottleneck/GaussianConditional as reference for learned priors and side-info accounting. |
| CompressAI ScaleHyperprior model docs | OSS | Translate | Hyperprior side-info must pay for itself on renderer/mask streams; do not use for tiny pose stream. |
| `bamler-lab/constriction` | OSS | Copy if dependency policy allows | ANS/range coding is implementation-ready for qint streams; otherwise port minimal static range coder. |
| Ballé 2018 scale hyperprior | paper | Copy math | Side-info only helps streams >= ~30KB; current Lane 20 confirms static wins on Lane G v3 qints. |

Immediate γ implementation targets:

1. Keep Ballé/hyperprior deferred until α or Ω-W-V3 creates a stream with nontrivial heteroscedasticity.
2. Use static arithmetic/ANS first; learned priors must beat static net of model/header bytes.
3. Joint-ADMM gets real value only after at least two component codecs have exact archive evidence.
4. MDL ranker should count payload bytes, side-info bytes, code bytes required by archive, and scorer distortion.
5. For mask/qint entropy experiments, prefer small static/context coders with byte-exact round trips before learned priors. Every side table is payload.

---

## Source Links

- NeRV official repo: https://github.com/haochen-rye/NeRV
- HiNeRV project: https://hmkx.github.io/hinerv/
- HiNeRV paper: https://arxiv.org/abs/2306.09818
- Boosting NeRV official repo: https://github.com/Xinjie-Q/Boosting-NeRV
- AWQ official repo: https://github.com/mit-han-lab/llm-awq
- HAWQ-V3 paper: https://arxiv.org/abs/2011.10680
- CompressAI entropy models: https://interdigitalinc.github.io/CompressAI/entropy_models.html
- CompressAI models: https://interdigitalinc.github.io/CompressAI/models.html
- constriction entropy coders: https://github.com/bamler-lab/constriction
- VQ-NeRV paper: https://arxiv.org/abs/2403.12401
- FSQ paper: https://arxiv.org/abs/2309.15505
- MAGVIT-v2 paper: https://arxiv.org/abs/2310.05737
- GPTQ paper: https://arxiv.org/abs/2210.17323
- SmoothQuant paper: https://arxiv.org/abs/2211.10438
- SparseGPT paper: https://arxiv.org/abs/2301.00774
