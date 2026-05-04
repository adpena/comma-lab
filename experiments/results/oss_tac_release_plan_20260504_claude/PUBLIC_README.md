# tac — Task-Aware Codec

**A research codebase for the [comma.ai video compression challenge](https://github.com/commaai/comma_video_compression_challenge).**

`tac` (Task-Aware Codec) is a research-grade Python package built around the principle that lossy video compression should optimize against downstream task scorers, not against pixel-distortion proxies. The repo packages the rate-distortion compiler, the score-aware codec primitives, the deterministic-archive submission contract, the production-hardened preflight gate (90+ STRICT bug-class extinctions), and a complete inflate runtime that ships under the contest's strict 30-minute on-device evaluation cap.

> Companion to a public submission to the comma.ai video compression challenge. Released to support reproducibility of the methodology, atomic decomposition framework, Yousfi-Fridrich theoretical floor, and the deterministic codec primitives.

## What is the comma.ai video compression challenge?

A 60-second driving video at 1164×874×3×60×30 = 5.5 GB raw must be compressed and decompressed losslessly enough that two frozen neural networks (`SegNet` for semantic segmentation, `PoseNet` for ego-motion) produce nearly identical outputs on the reconstruction. The contest score is:

```
S = 100 * average_segnet_distortion + sqrt(10 * average_posenet_distortion) + 25 * compression_rate
```

where `compression_rate = submission_archive_bytes / 37,545,489`. The contest is therefore a **rate-distortion problem against frozen perceptual-network scorers**, not against MSE or LPIPS.

## What does `tac` provide?

### Codec primitives (`src/tac/`)

- **`qp1_pose_codec.py`** — quantized pose codec; ships in C-067 lineage (0.31561703 contest-CUDA T4 A++).
- **`qzs3_renderer_codec.py`** — Quantizr-style grouped variable-bit-depth FP4 packer for renderer state-dicts.
- **`pfp16_codec.py`** — FP16 renderer codec (PFP16 baseline 1.044 contest-CUDA T4).
- **`block_fp_codec.py`** — block-FP weight codec (Selfcomp-paradigm sibling).
- **`water_filling_codec_v2.py`** — score-Jacobian sensitivity-aware bit allocation; achieved 40.98% byte savings on a Lane G v3 renderer (empirical).
- **`arithmetic_qint_codec.py`** — arithmetic coder for quantized-integer streams.
- **`balle_hyperprior_codec.py`** — Ballé 2018 scale-hyperprior entropy bottleneck.
- **`mask_codec.py`** + **`mask_entropy_coder.py`** + **`mask_grayscale_lut.py`** + **`wavelet_mask_codec.py`** + **`stc_boundary_codec.py`** — multiple paradigms for the mask channel (AV1 monochrome, lossless entropy coder, Selfcomp Gaussian-LUT, Mallat wavelet, Filler syndrome-trellis).
- **`nerv_mask_codec.py`** — NeRV implicit neural representation for masks.
- **`vqvae_mask_codec.py`** — VQ-VAE codebook for masks.
- **`neural_weight_codec.py`** — VQ-VAE weight codec (Lane J-NWC).
- **`sjkl_basis.py`** — score-Jacobian Karhunen-Loève basis (Wave-Ω paradigm primitive).
- **`sensitivity_map.py`** — β-Fisher information sensitivity map producer (foundational artifact for water-fill / IMP / hyperprior).

### Score-aware optimization (`src/tac/`)

- **`fridrich_losses.py`** + **`fridrich.py`** + **`uniward_delta.py`** + **`uniward_texture.py`** — Fridrich UNIWARD steganalysis losses; embedding error in textured regions where SegNet is blind.
- **`scorer_exploits.py`** — documented scorer blind-spot exploits (compress-time only).
- **`saliency.py`** + **`saliency_inversion.py`** + **`hyperbolic_foveation.py`** — saliency-aware encoding.
- **`logit_margin_sensitivity_weighted.py`** — Lane 19 logit-margin loss.
- **`joint_admm_coordinator.py`** + **`joint_admm_proximal_water_filling_v2.py`** + **`joint_admm_proximal_pose_delta.py`** — Boyd ADMM coordinator for joint rate / seg / pose / archive-size feasibility projection.
- **`stack_compositions.py`** + **`optimal_stack_orchestrator.py`** + **`joint_codec_stack_orchestrator.py`** + **`trick_stack.py`** — meta-tools for composing codec primitives onto a Pareto-optimal stack.

### Pose & motion (`src/tac/`)

- **`raft_pose.py`** + **`raft_radial_pose.py`** — RAFT optical-flow-derived pose (openpilot-style ego-motion).
- **`pose_gaussian_process.py`** — Gaussian-process Bayesian pose fit.
- **`pose_delta_codec.py`** + **`pose_delta_codec_v2.py`** — quantized pose-delta codec.
- **`pose_from_embedding.py`** — pose-from-embedding decoder.
- **`geodesic_pose.py`** — geodesic SO(3) pose interpolation.
- **`riemannian_pose_optimizer.py`** — Riemannian optimizer for SE(3).
- **`se3.py`** — SE(3) Lie-algebra primitives.
- **`lora_pose.py`** + **`lora_pose_v2.py`** — LoRA pose adapters.
- **`radial_zoom.py`** — empirically rank-1.008 radial zoom (PoseNet Jacobian effective rank).

### Production infrastructure (`src/tac/`)

- **`submission_archive.py`** — deterministic single-payload archive packer; enforces strict-scorer rule (no scorer load at inflate time, all charged bytes inside `archive.zip`).
- **`eval_roundtrip_gate.py`** — closes the proxy-auth gap (2-11x on PoseNet) by simulating the full contest eval roundtrip.
- **`preflight.py`** — 90+ STRICT preflight checks; permanently extincts known bug classes (MPS-fallback, `set -uo` no-`-e`, dead-flag wiring, scorer-at-inflate, etc.).
- **`lane_c_compliance.py`** — cryptographic Ed25519 attestation gate for compliance-pending lanes.
- **`codec_magic_registry.py`** — magic-byte registry for runtime codec dispatch.
- **`checkpoint_names.py`** — canonical checkpoint naming.
- **`forensics.py`** — boundary artifact detection; PoseNet per-pixel Jacobian sensitivity maps; eval-roundtrip distortion maps.

## Methodology

### Rate-distortion compiler

Every pipeline element is treated as a typed atom with measurable byte/Δseg/Δpose interactions. The compiler searches the convex intersection (rate ≤ R, seg ≤ S, pose ≤ P, archive ≤ A) for the Pareto frontier via Dykstra alternating projections.

### Yousfi-Fridrich theoretical floor

The challenge IS inverse steganalysis. SegNet (EfficientNet-B2) and PoseNet (FastViT-T12) are forensic detectors. Reconstruction errors should be:

1. **Concentrated in textured regions** (Fridrich UNIWARD principle — undetectable embedding).
2. **Spread to small magnitudes** (square-root law — concentrated errors are detectable).
3. **Aligned with score-Jacobian null space** (errors in directions the scorer is insensitive to).

The Yousfi-Fridrich floor `R_YF(D)` is bounded above by the Shannon floor `R_Sh(D)`: scorer-aware embedding can be more efficient than information-theoretic distortion-rate.

### Deterministic single-payload archive

All score-affecting bytes ship as a single ZIP member. No hidden sidecars, no scorer patches at inflate time, no runtime dependencies that aren't auditable file-by-file. The archive SHA-256 is the integrity contract.

### Atomic decomposition framework

Every contribution to the archive (pose codec, mask codec, decoder weights, latent stream, sidecar corrections) is a typed atom. Each atom has measurable byte cost and known Δseg/Δpose impact. Stacking is the convex sum of atoms over the feasible region.

### Game-theoretic premature-convergence analysis

Contest deadline incentives create local-optimum traps: small wins compound, but paradigm-level jumps are deferred until a competitor lands one first. The grand council (10-voice quintet pact + 12-member grand bench) deliberates against this bias by mandating non-conservative voting and adversarial review.

## What this repo does NOT include

Per the strategic-secrecy rule of the comma.ai contest:

- The contest video (`upstream/videos/0.mkv`) — we are not the rights holder; users obtain it from the comma.ai repo
- The pinned upstream scorer snapshot — see comma.ai repo
- Per-experiment private custody ledgers
- Cloud-platform dispatch credentials (Modal, Lightning, Vast.ai)
- Internal multi-agent coordination tooling

## Quick start

```bash
# Install
git clone https://github.com/adpena/tac.git
cd tac
uv venv
uv pip install -e ".[dev,runtime]"

# Run the test suite (no GPU required)
uv run pytest src/tac/tests -m "not cuda and not slow"

# Build a submission archive (requires the contest video)
uv run python experiments/pipeline.py compress \
    --video /path/to/0.mkv \
    --checkpoint /path/to/checkpoint.pt \
    --device cuda \
    --output-dir results/run_01

# Run the deterministic auth eval
uv run python experiments/contest_auth_eval.py \
    --archive results/run_01/archive.zip \
    --device cuda
```

## Architecture (the deployed PR107 architecture)

The `apogee` submission (PR107 in the comma.ai repo) ships an HNeRV-style learned decoder with per-pair latent stream, brotli-packed weights, single-payload archive at ~178 KB and an exact contest-CUDA T4 score of `0.22933`.

```
archive.zip                                # 178,392 bytes, single ZIP member `0.bin`
├── packed_header_ff_len24                 # 4 bytes
├── decoder_compact_brotli_streams         # ~162 KB — HNeRV decoder weights
├── latents_raw_lzma_delta_u8              # ~15 KB — per-pair latent
└── sidecar_dim_delta_huffman_enum         # ~600 bytes — small corrections
```

The deployed `inflate.sh` resolves the contest contract: 30-minute budget, fixed-video `0.mkv`, deterministic single-payload archive.

## Score lineage (this codebase's milestones)

| Date (UTC) | Lane / submission | Score | Hardware | Notes |
|---|---|---:|---|---|
| 2026-04-25 | MPS-vs-CUDA correction | 0.90 baseline | A100 | First contest-compliant baseline (corrected from MPS artifact 2.26) |
| 2026-04-28 | Lane G v3 OWv3 | 1.05 | RTX 4090 | First Wave-3 sub-frontier |
| 2026-05-01 | Lane G v3 OWv3 R7 | 1.013 | RTX 4090 | Component-balanced selector |
| 2026-05-01 | Lane G v3 OWv3 0120 | 1.0024 | RTX 4090 | 5 sub-frontiers in one chain |
| 2026-05-01 | OWv3 0120 PD-V2 | 0.9974 | RTX 4090 | First sub-1.0 |
| 2026-05-02 | C-067 (composite) | 0.31561703 | T4 | Long-form pose + arithmetic stack |
| 2026-05-04 | apogee (HNeRV pivot) | 0.22933 | T4 | Submitted as PR107 |

## Acknowledgments

This codebase is the result of intensive iteration alongside many public submissions to the comma.ai video compression challenge. We acknowledge the work of (in chronological order of contributions that shaped our paradigm):

- **comma.ai contest organizers** — for designing the challenge as inverse steganalysis. Yassine Yousfi (challenge creator, formerly Fridrich's PhD student at Binghamton DDE Lab) and the canonical scorer architectures.
- **Quantizr (Jimmy, UCLA CSE)** — first public sub-0.40 lane (PR #55, score 0.33). Opened the unknown-unknown that ultra-compact ~88K-parameter FiLM-DSConv renderers are reachable.
- **EthanYangTW** — multiple frontier-pushing PRs (PR #67 R(D)-joint coordinate descent on pose, PR #75/#79/#98/#102).
- **henosis-us** — PR #65 / PR #82 30-byte-header / 10-section randmulti packer paradigm.
- **szabolcs-cs** (Selfcomp) — PR #56 SegMap + Gaussian-softmax-LUT + AV1 grayscale + xz-int8 stack (score 0.36).
- **rem2** — PR #96 / PR #103 HNeRV submissions opening the 0.21-tier.
- **valtterivalo** — PR #105 `kitchen_sink` and PR #106 `belt_and_suspenders` (current public exact contest-CUDA T4 frontier at 0.20946).
- **BradyMeighan** — PR #97 / PR #99 / PR #100 HNeRV-LC variants.
- **AaronLeslie138** — PR #95 `hnerv_muon` opening the HNeRV pivot.
- **SajayR** — PR #101 `hnerv_ft_microcodec`.
- **patattzel, ottokunkel, nick-neely, jas0xf, hypery11, manthedan, erichasinternet, Janos95, dllu, Josemedinan, narenn99, celikemir, 1kuna, avocardio** — all other contest entrants whose PRs we replayed and learned from.

The grand council (an internal review framework) channels the work of:

- **Claude Shannon (council lead)** — information theory grounding; every score-improvement claim must trace back to a rate-distortion or entropy argument
- **Richard Dykstra (co-lead)** — alternating projections; convex feasibility intersection
- **Yassine Yousfi** — challenge creator + steganalysis foundations
- **Jessica Fridrich** — UNIWARD framework + adversarial-detector embedding
- **George Hotz** — engineering shortcuts; raw analytical instinct
- **Jimmy / Quantizr** — adversarial member; reverse-engineers competitor approaches
- **szabolcs-cs / Selfcomp** — Gaussian-LUT + block-FP paradigm
- **David MacKay** (memorial seat) — *Information Theory, Inference, and Learning Algorithms*
- **Johannes Ballé** — modern neural-compression SOTA (entropy bottleneck + scale hyperprior)

## License

MIT — see `LICENSE`.

## AI assistance disclosure

This codebase was developed with assistance from Anthropic Claude (Opus 4.7) and OpenAI Codex (GPT-5.5) acting as research-engineering collaborators under careful human review and direction.
