# L5 Staircase / Cathedral Source Map - 2026-05-16

## Purpose

Bind paper/OSS evidence to concrete code and campaign changes for the L5
staircase, cathedral autopilot, and cross-family score-lowering stack. This is
not a score ledger. Every item below must still pass Pact evidence rules:
byte-closed archive/runtime, no-op/consumption proof, scorer-aware objective,
and paired contest-axis custody before promotion or family retirement.

## Source-Anchored Actions

| Source | Local target | Required action |
|---|---|---|
| Ballé et al., *Variational image compression with a scale hyperprior* (`https://arxiv.org/abs/1802.01436`) | `src/tac/substrates/balle_renderer/`, Z3/BRV2 | Hyperprior side-info must be decoded and consumed, not merely stored as validation metadata. BRV2 promotion requires side-info mutation changes reconstructed latents or inflated pixels. |
| Minnen, Ballé, Toderici, *Joint Autoregressive and Hierarchical Priors* (`https://papers.nips.cc/paper/8275-joint-autoregressive-and-hierarchical-priors-for-learned-image-compression`) | BRV2/Z3 entropy path | Combine hierarchical side information with context/autoregressive modeling only if runtime LOC, T4 time, and deterministic decoder closure remain inside contest bounds. |
| HNeRV CVPR 2023 (`https://openaccess.thecvf.com/content/CVPR2023/html/Chen_HNeRV_A_Hybrid_Neural_Representation_for_Videos_CVPR_2023_paper.html`) | PR95/PR101 control arm, NeRV-family wave | Keep HNeRV as the control substrate. Further score lowering should come from contest-specific objective/curriculum/export/sidecar engineering, not paper-faithful HNeRV regression alone. |
| Cool-Chic official docs/repo (`https://orange-opensource.github.io/Cool-Chic/`, `https://github.com/Orange-OpenSource/Cool-Chic`) | Cool-Chic/C3 residual lanes | Clean-room the useful ideas: low-complexity overfitted decoder, inter-feature entropy, and fast decode. Do not import GPL-sensitive code. Paper BD-rate does not transfer without scorer-aware training. |
| constriction entropy coders (`https://github.com/bamler-lab/constriction`) | PacketIR, PR106 recode, BRV2/Z3 side streams | Use Python API for research prototypes and Rust-compatible design constraints for runtime paths. Any ANS/range candidate must include bitstream roundtrip, section-byte delta, and runtime-consumption proof. |
| DreamerV3 / Hafner et al. (`https://arxiv.org/abs/2301.04104`) | C1/DP1/world-model probes | Use posterior/prior/residual/KL discipline as the model-based-control reference, but do not claim score movement from world-model structure until archive bytes and scorer deltas are measured. |
| Rao-Ballard predictive coding (`https://www.cnbc.cmu.edu/~tai/readings/nature/Rao-Ballard-99-NN-nv.pdf`) | Z5/C2/L5 predictive receiver | Mandatory identity-predictor ablation: prove gains come from prediction/residual signaling, not extra parameter capacity. |

## Paper-Fidelity Risks To Audit Next

1. **BRV2 side-info consumption.** BRV1 is correctly fail-closed because
   hyper-latents are render-silent. BRV2 must demonstrate that hyper-latents
   alter decoded main latents or output pixels under byte mutation.
2. **Metric mismatch.** HNeRV, Cool-Chic, CompressAI/Ballé, and Cheng-style
   learned codecs optimize PSNR/MS-SSIM or video regression. Pact optimizes
   fixed SegNet/PoseNet plus archive bytes. Every imported method must be
   re-derived through score-aware loss, eval-roundtrip, archive grammar, and
   paired CPU/CUDA custody.
3. **Runtime mismatch.** Rich autoregressive/context models are attractive for
   rate but can blow the T4 30-minute decode budget. Any BRV2/PacketIR coder
   proposal must report decode complexity before exact-eval dispatch.

## Immediate Integration Targets

- Finish BRV2 consumed-sideinfo contract with mutation-consumption tests.
- Extend PacketIR context recode only when section bytes decrease and the
  runtime decoder consumes the recoded stream.
- Keep TT5L/L5 macOS smoke advisory-only; use it as an escalation signal, not
  promotion or family-retirement evidence.
