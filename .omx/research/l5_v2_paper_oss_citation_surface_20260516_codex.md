# L5-v2 paper and OSS citation surface - 2026-05-16

## Scope

Research-only citation and follow-up surface for the L5-v2 staircase, PR106
PacketIR, HNeRV-family successors, learned entropy models, and production
custody. This ledger does not claim score movement, dispatch readiness, or
promotion eligibility.

## Primary References

| theme | reference | L5-v2 use |
|---|---|---|
| HNeRV control substrate | HNeRV: A Hybrid Neural Representation for Videos, arXiv:2304.02633, https://arxiv.org/abs/2304.02633 | Direct public-frontier family ancestor; use as control for complete archive/training/runtime packets. |
| NeRV baseline | NeRV: Neural Representations for Videos, arXiv:2110.13903, https://arxiv.org/abs/2110.13903 | Baseline frame-index neural video representation for ablations. |
| E-NeRV | E-NeRV: Expedite Neural Video Representation with Disentangled Spatial-Temporal Context, arXiv:2207.08132, https://arxiv.org/abs/2207.08132 | Spatial/temporal disentanglement path for latent and frame-split work. |
| FFNeRV | FFNeRV: Flow-Guided Frame-Wise Neural Representations for Videos, arXiv:2212.12294, https://arxiv.org/abs/2212.12294 | Frame propagation and neighboring-frame aggregation ideas for nullspace/sidecar design. |
| HiNeRV | HiNeRV: Video Compression with Hierarchical Encoding-based Neural Representation, arXiv:2306.09818, https://arxiv.org/abs/2306.09818 | Hierarchical encoding, pruning, and quantization reference for a unique L5-v2 substrate. |
| COIN / INR compression | COIN: COmpression with Implicit Neural representations, arXiv:2103.03123, https://arxiv.org/abs/2103.03123 | Tiny residual or scorer-inverse correction sidecars. |
| SIREN | Implicit Neural Representations with Periodic Activation Functions, arXiv:2006.09661, https://arxiv.org/abs/2006.09661 | High-frequency residual atoms when HNeRV blurs scorer-sensitive details. |
| Ballé hyperprior | Variational image compression with a scale hyperprior, arXiv:1802.01436, https://arxiv.org/abs/1802.01436 | Side-information and PMF modeling for PR106 latent/weight streams. |
| Context + hyperprior | Joint Autoregressive and Hierarchical Priors for Learned Image Compression, arXiv:1809.02736, https://arxiv.org/abs/1809.02736 | Local context plus hyperprior for per-section entropy models. |
| CompressAI | CompressAI, arXiv:2011.03029, https://arxiv.org/abs/2011.03029 and OSS https://github.com/InterDigitalInc/CompressAI | Research scaffold for entropy bottlenecks and rate-distortion training, not an inflate dependency. |
| DCVC | Deep Contextual Video Compression, arXiv:2109.15047, https://arxiv.org/abs/2109.15047 | Conditional video-coding ideas for frame/context propagation; keep final runtime small. |
| DeepCABAC | DeepCABAC: A Universal Compression Algorithm for Deep Neural Networks, arXiv:1905.08318, https://arxiv.org/abs/1905.08318 | Neural-weight arithmetic coding reference for decoder and latent byte mass. |
| Entropy-penalized model compression | Entropy-Constrained Model Compression, arXiv:1906.06624, https://arxiv.org/abs/1906.06624 | Train compressibility into weights instead of hoping generic compressors find it after export. |
| ANS | Asymmetric numeral systems, arXiv:0902.0271, https://arxiv.org/abs/0902.0271 | Foundation for rANS/tANS payload coding. |
| Practical ANS | Understanding ANS, arXiv:2201.01741, https://arxiv.org/abs/2201.01741 | Practical bridge from latent-variable compression to ANS implementations. |
| constriction | https://github.com/bamler-lab/constriction | Python/Rust range/ANS reference for research prototypes and small-decoder design. |
| EOT robustness | Synthesizing Robust Adversarial Examples, arXiv:1707.07397, https://arxiv.org/abs/1707.07397 | Use eval-roundtrip and transform robustness to avoid overfitting a single differentiable scorer path. |
| reproducibility | ReproZip, DOI:10.1145/2882903.2899401, https://doi.org/10.1145/2882903.2899401; safetensors, https://github.com/huggingface/safetensors | Dependency capture and safe tensor serialization patterns for public custody. |

## Codebase-Facing Recommendations

1. Make PR106 PacketIR section-first: run DeepCABAC, entropy-penalized
   quantization, range, and ANS candidates against `decoder_packed_brotli` and
   `latents_and_sidecar_brotli` before any more ZIP/header cosmetics.
2. Treat Ballé/CompressAI as a PMF and rate-distortion research scaffold only.
   Final inflate must stay byte-closed, small, dependency-light, and scorer-free.
3. For HiNeRV/FFNeRV successors, require archive grammar, parser-section
   manifest, export contract, and reviewable inflate runtime before training.
4. Make nullspace/frame-split search paired by construction: optimize with
   eval-roundtrip/EOT transforms, then require runtime consumption and paired
   exact CPU/CUDA custody before rank or frontier language.
5. Every candidate artifact should emit archive SHA, payload SHA, runtime tree
   SHA by axis, runtime content tree SHA, section offsets, PMF table hashes,
   codec version, dependency closure, full-frame parity status, and exact axis
   labels.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
