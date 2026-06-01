# Fresh-Eyes Compact Learned Video Stack Audit A

UTC: 2026-06-01T12:13:41Z
Author: Codex
Authority: research/audit memo only; no score claim; no promotion authority; no dispatch

## Bottom Line

The strongest near-term score-lowering route is not a new generic video codec.
It is a contest-sized PR95/HNeRV-family renderer whose archive bytes are treated
as the optimization target: decoder weight stream first, learned pair-latent
stream second, tiny scorer-aware residual or token stream third. PR95 is tiny
because it pays roughly one 229K-parameter overfit decoder plus 600 28-d pair
latents, then encodes both with INT8/UINT8 transforms and Brotli into a single
`0.bin`. Local profiles put the public PR95 archive at 178,417 bytes: 162,349
decoder bytes, 15,868 latent bytes, and 80 metadata bytes.

Modern NeRV variants change the best next move as follows:

- RNeRV/VINRB says the architecture/training recipe still matters under equal
  training time, so use it to redesign the base renderer, not as an authority
  to bypass archive/runtime proof.
- HiNeRV/SRNeRV say capacity should be hierarchical and shared across scales,
  but the local HIV1 sketch currently spends too many raw latent/FP16-pickle
  bytes for a 100K target.
- RT/VQ-NeRV says shallow residual and inter-frame support should be tokenized;
  in this repo that maps cleanly to a compact VQ/token sidecar or PR95-style
  latent stream, not MB-scale Z8 wavelet detail.

## Top 3 Architectures

1. PR95/HNeRV-core, byte-optimized
   - Use the existing PR95 topology as the control arm: 28-d pair latent,
     36-base decoder, 6 PixelShuffle stages, two RGB heads.
   - First changes: sweep `base_channels` and `latent_dim`; add INT6/FP4 or
     selective per-layer quantization experiments; keep PR95 single-member
     grammar and runtime proof.
   - Why first: the archive/runtime/custody path already exists, public family
     packets already live at 178K-186K, and the dominant stream is known.
   - Budget:
     - 100K target: decoder 80K-92K, latents 6K-12K, metadata/sidecar <=3K.
     - 180K target: decoder 145K-165K, latents 12K-17K, sidecar <=5K.
     - 285K target: current PR95 plus 40K-90K residual/token repair sidecar.

2. RNeRV/recursive shared-renderer base plus scorer-aware residual
   - Use RNeRV/VINRB-style component search and SRNeRV-style scale sharing to
     reduce redundant decoder bytes, then spend a small sidecar only where
     SegNet/PoseNet surfaces say bytes matter.
   - Treat Z8 as residual sidecar after a compact base renderer, not as the
     primary carrier: current Z8 profiles are MB-scale because wavelet detail
     dominates the packet.
   - Budget:
     - 100K target: 65K-90K base decoder, <=8K latents/indices, no residual.
     - 180K target: 90K-125K base, 10K-25K latent/token stream, 10K-35K sidecar.
     - 285K target: base plus 60K-120K scorer-aware residual/token repair.

3. PVQ/RT-VQ-NeRV hybrid with depthwise decoder
   - Merge the local `pact_nerv_vq` depthwise renderer with RT/VQ-NeRV residual
     tokenization ideas: codebook indices for pair/feature residuals, entropy
     coded indices, quantized codebook, PR95-style quantized decoder.
   - The local PVQ sketch is promising because 600 uint16 indices are only 1.2K
     raw, but it currently ships FP16-pickled decoder and raw int16 codebook.
   - Budget:
     - 100K target: decoder 70K-80K, codebook 4K-8K, indices <1K, residual <=10K.
     - 180K target: decoder 110K-130K, codebook 12K-24K, sidecar 5K-20K.
     - 285K target: richer codebook/multiscale tokens plus repair residual.

## Exact Code Paths To Modify

- PR95 control arm:
  - `src/tac/local_acceleration/pr95_hnerv_mlx.py`
  - `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
  - `src/tac/hnerv_arch_schema.py`
  - `tools/run_pr95_mlx_timing_smoke.py`
  - `tools/export_pr95_mlx_to_pytorch_state_dict.py`
  - `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
  - `src/tac/analysis/hnerv_packet_sections.py`

- PVQ/Hi/RT token branch:
  - `src/tac/substrates/pact_nerv_vq/architecture.py`
  - `src/tac/substrates/pact_nerv_vq/archive.py`
  - `experiments/train_substrate_pact_nerv_vq.py`
  - `src/tac/substrates/hi_nerv/architecture.py`
  - `src/tac/substrates/hi_nerv/archive.py`

- Residual sidecar branch:
  - `src/tac/substrates/z8_hierarchical_predictive_coding/archive.py`
  - `tools/z8_detail_coeff_entropy_headroom_report.py`
  - `tools/run_z8_joint_p18_p19_relinearized_deadzone_search.py`
  - `.omx/research/rnerv_pact_z8_residual_lane_design_20260531T222520Z_codex.md`

## Risks And Blockers

- No MLX, RGB-MSE, local CPU, or decoded-latent result is score authority.
  Full-frame inflate parity and exact auth eval remain mandatory.
- The current PR95 MLX history shows the packaging gap is often latents, not
  decoder weights: trained decoder plus separate `.latents.npy` must enter the
  byte-closed archive path explicitly.
- HiNeRV/PVQ local sketches are not byte-optimal yet: FP16 pickles, raw int16
  latents/codebooks, and unevaluated runtime decode will blow the 100K target.
- Z8 residuals are currently far above contest byte ceilings unless a compact
  base renderer collapses the residual first.
- CUDA/CPU hardware drift is real for PR95-family submissions; report every
  result with axis labels.

## First Executable Commands

Plan the PR95 stage-8 control experiment without spending or generating bulk:

```bash
.venv/bin/python tools/run_pr95_mlx_timing_smoke.py \
  --stage 8 \
  --output-dir /Volumes/VertigoDataTier/pact/fresh_eyes_A/pr95_stage8_plan_20260601T121341Z \
  --base-channels 36 \
  --latent-dim 28 \
  --train-on-source-video-pairs \
  --source-video-loss-surface rgb_yuv6_mse \
  --plan-only \
  --write-execution-queue
```

Run the smallest PVQ/token smoke under SSD output:

```bash
.venv/bin/python experiments/train_substrate_pact_nerv_vq.py \
  --video-path upstream/videos/0.mkv \
  --output-dir /Volumes/VertigoDataTier/pact/fresh_eyes_A/pvq_smoke_20260601T121341Z \
  --epochs 2 \
  --batch-size 2 \
  --latent-dim 8 \
  --codebook-size 16 \
  --max-pairs 16 \
  --val-pair-count 8 \
  --smoke \
  --skip-auth-eval \
  --device cpu
```

Measure whether Z8 residual bytes have any headroom before using it as a sidecar:

```bash
.venv/bin/python tools/z8_detail_coeff_entropy_headroom_report.py \
  --num-pairs 8 \
  --quant-steps 0.5,1.0,2.0,4.0 \
  --out-json /Volumes/VertigoDataTier/pact/fresh_eyes_A/z8_entropy_headroom_20260601T121341Z.json
```

## Citations And Anchors

- PR95 pull request and public report:
  https://github.com/commaai/comma_video_compression_challenge/pull/95
- HNeRV paper and code:
  https://arxiv.org/abs/2304.02633
  https://github.com/haochen-rye/HNeRV
- NeRV paper and code:
  https://arxiv.org/abs/2110.13903
  https://github.com/haochen-rye/NeRV
- HiNeRV paper and code:
  https://arxiv.org/abs/2306.09818
  https://github.com/hmkx/HiNeRV
- RNeRV/VINRB paper and code:
  https://arxiv.org/abs/2506.24127
  https://github.com/mgwillia/vinrb
- RT/VQ residual-token NeRV:
  https://arxiv.org/abs/2403.12401
- SRNeRV:
  https://arxiv.org/abs/2603.08227
