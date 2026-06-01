# Subagent B latent-codebook and implicit/procedural full-stack design

UTC: 2026-06-01T12:02:16Z
Author: Codex Subagent B
Status: research_only=true, score_claim=false, promotion_eligible=false
Write boundary: this memo only; no code, queue, claim, artifact, or dispatch state was modified.

## Verdict

Top candidate: LCCI-C3VQ, a latent-codebook conditional implicit stack.

The stack should transmit a compact C3/Cool-Chic-style learned multi-scale
latent/index stream as the primary carrier, amortized through VQ/shared
codebooks, with implicit-network atoms only as sparse residual dictionaries.
RAFT/ego-motion and CLAdE/SPADE-style conditioning are allowed only as compact
procedural predictors or tiny affine/modulation tables that survive section
pricing. Dense flow fields, external RAFT/semantic models, hidden scorer state,
or sidecars outside archive.zip are forbidden.

This is distinct from the current HPRC compact receiver: HPRC's current receiver
uses a compact learned base plus latent/residual sections, while LCCI-C3VQ makes
multi-scale codebook indices and entropy-coded codebook streams the transmitted
commodity. It is also distinct from PR106 residual sidecars: PR106 sidecars are
only staging scaffolds; promotion should go through an HPRC packet with charged
DECODER_QW, LATENTS_RC, CODEBOOKS_Q, SELECTORS_RC, RESIDUAL_RC, RDO_PLAN,
RECEIVER_STATE, and MANIFEST_JSON sections.

## Online provenance

- Cool-Chic: ICCV 2023 "COOL-CHIC: Coordinate-based Low Complexity
  Hierarchical Image Codec" and Orange Open Source implementation. The repo
  positions Cool-Chic as a low-complexity neural image/video codec based on
  overfitting with range coding in its toolchain.
  https://openaccess.thecvf.com/content/ICCV2023/html/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.html
  https://github.com/Orange-OpenSource/Cool-Chic
- C3: CVPR 2024 "C3: High-performance and low-complexity neural compression
  from a single image or video". C3 explicitly builds on Cool-Chic, overfits a
  small model per image/video, and targets low decoding MACs.
  https://c3-neural-compression.github.io/
  https://openaccess.thecvf.com/content/CVPR2024/papers/Kim_C3_High-Performance_and_Low-Complexity_Neural_Compression_from_a_Single_Image_CVPR_2024_paper.pdf
- VQ-VAE: "Neural Discrete Representation Learning" supplies the discrete
  latent/codebook premise: learned code vectors plus transmitted indices.
  https://papers.neurips.cc/paper/7210-neural-discrete-representation-learning.pdf
- SIREN: official implementation of periodic-activation implicit neural
  representations.
  https://github.com/vsitzmann/siren
- FINER: variable-periodic activation functions for tunable spectral bias in
  implicit neural representations.
  https://arxiv.org/abs/2312.02434
- WIRE: wavelet implicit neural representations using complex Gabor wavelet
  activations with image-friendly space-frequency bias.
  https://arxiv.org/abs/2301.05187
- BACON: band-limited coordinate networks with analytical Fourier spectrum and
  multiscale representation behavior.
  https://www.computationalimaging.org/publications/bacon/
- RAFT: recurrent all-pairs field transforms for optical flow. It is useful as
  an offline analysis prior, but the model and dense flow fields are not archive
  positive unless compressed into tiny deterministic state.
  https://arxiv.org/abs/2003.12039

## Local provenance and code surfaces

- HPRC already has the right outer grammar. `src/tac/substrates/hprc/archive.py`
  defines the representation-level sections: DECODER_QW, LATENTS_RC,
  CODEBOOKS_Q, SELECTORS_RC, RESIDUAL_RC, RDO_PLAN, RECEIVER_STATE, and
  MANIFEST_JSON. Its manifest also correctly says hprc.bin payload accounting is
  not contest authority until archive.zip/runtime bytes and receiver replay are
  proven.
- `src/tac/substrates/hprc/lineage.py` already classifies C3/Cool-Chic, SIREN,
  RAFT, CLAdE/SPADE, VQ, entropy coding, and exact replay into HPRC roles and
  gates. The crucial local gates are: codebook plus indices must beat scalar
  residual coding, shared dictionaries must amortize across pairs/GOPs, residual
  tokens need P18/P19 marginal value above measured byte cost, motion state must
  be procedural or compact, and promotion needs local CPU replay then exact auth
  only for true local winners.
- `src/tac/substrates/hprc/spine_bounded_runner.py` already has the binding
  policy: every PR95/HNeRV/RNeRV/PACT-NeRV/VQ/selector candidate emits a spine
  projection, then receiver proof, then full-video MLX scorer replay, then exact
  gate. Section and residual bytes are admitted only when measured
  `delta_nonrate + charged_rate_cost < 0`; MLX is advisory.
- `src/tac/substrates/pact_nerv_vq/archive.py` proves a simple VQ archive
  grammar already exists: monolithic 0.bin with decoder blob, int16 codebook,
  uint16 indices, and JSON quantization metadata. Use this as codebook grammar
  precedent, not as promotion authority.
- `src/tac/residual_basis/c3_encoder_l2.py` is the closest staging spine for C3:
  first-difference frame deltas, sparse-aware byte cost, Hinton-distilled scorer
  option, saliency masking, eval-roundtrip, and sparse PacketIR repack.
- `src/tac/residual_basis/cool_chic_encoder_l2.py` is the closest staging spine
  for Cool-Chic: Laplacian/multilevel pyramid residuals, per-level top-k budget,
  sparse-aware byte cost, Hinton scorer option, and eval-roundtrip.
- `src/tac/residual_basis/pr106_sidecar_packing.py` and the materializers under
  `tools/materialize_{c3,cool_chic}_residual_pr106_sidecar.py` can smoke-test
  sparse residual atom value, but their PR106 wrapper is not the final stack.
- `tools/profile_pact_nerv_selector_v3_mlx_section_value.py` is the concrete
  pattern for a future LCCI section-value profiler: make valid neutralized
  variants, replay full-video MLX scorer, and emit false-authority-safe rows.

## Archive grammar proposal

archive.zip:

1. `inflate.sh`
2. `inflate.py` or a compact HPRC runtime closure
3. `hprc.bin`

`hprc.bin` section plan:

1. DECODER_QW: tiny codebook-conditioned decoder weights. Quantize int8/FP4
   plus scales, entropy-code with brotli/range/ANS. Target 35-65 KB.
2. LATENTS_RC: pair/GOP multi-scale VQ index stream. Temporal first-difference
   and C3 context prediction before range/ANS coding. Target 8-24 KB.
3. CODEBOOKS_Q: learned codebooks for latent grids plus optional implicit atom
   dictionaries. Target 4-16 KB.
4. SELECTORS_RC: block/pair/GOP mode IDs only if the selector section prices in.
   Target 0.5-4 KB; default to absent or tiny because prior selector pricing
   showed a near-zero-impact selector stream.
5. RESIDUAL_RC: sparse C3/Cool-Chic/SIREN/FINER/WIRE/BACON atoms. Default 0
   bytes; admit only under measured section-value win.
6. RDO_PLAN: P18/P19 thresholds and section price thresholds. Target <= 1 KB.
7. RECEIVER_STATE: compact procedural state only: camera/ego-motion scalar
   model, GOP seeds, or deterministic predictor constants. Target <= 2 KB.
8. MANIFEST_JSON: hashes, false-authority flags, section manifests, runtime
   closure. Target <= 1 KB.

No constants, state, codebook, model weights, procedural seeds, or runtime
assumptions may live outside archive.zip. Offline teacher/scorer/RAFT/semantic
models can generate training targets but cannot be required by the receiver.

## Byte budgets

Contest-rate-relevant working ceilings using 37,545,489 original bytes and the
25 * archive_bytes / original_bytes convention:

- 75,090 bytes corresponds to a 0.05 rate term.
- 150,181 bytes corresponds to a 0.10 rate term.
- 225,272 bytes corresponds to a 0.15 rate term.
- 285,345 bytes corresponds to a 0.19 rate term.

LCCI-C3VQ should target three envelopes:

- Aggressive smoke: 65-110 KB total archive.zip. Use tiny decoder, VQ indices,
  and no residual atoms unless the residual section already prices in.
- Practical candidate: 110-160 KB total archive.zip. This is the first realistic
  receiver-proven local replay target if distortion drops sharply.
- Max exploration: 160-225 KB total archive.zip. Allowed only if the nonrate
  score is plausibly below the current exact target after local replay.

Initial per-section budget:

- DECODER_QW: 35-65 KB.
- LATENTS_RC: 8-24 KB.
- CODEBOOKS_Q: 4-16 KB.
- SELECTORS_RC: 0.5-4 KB, or zero if priced dead.
- RESIDUAL_RC: 0-32 KB, admitted token-by-token.
- RDO_PLAN + RECEIVER_STATE + MANIFEST_JSON: <= 4 KB combined.
- inflate runtime and zip container: must be charged in archive.zip and kept
  small enough that the packet budget is not illusory.

## Section value-pricing experiment

Future tool surface: `tools/profile_lcci_hprc_section_value.py`, modeled on
`tools/profile_pact_nerv_selector_v3_mlx_section_value.py`.

Procedure:

1. Build a full LCCI HPRC candidate with all sections.
2. For each section, create a valid semantic neutralization, not a random byte
   flip:
   - DECODER_QW: replace decoder/atom weights with mean or zero decoder.
   - LATENTS_RC: replace indices with predictor/default indices.
   - CODEBOOKS_Q: replace codebook entries with zero/mean entries.
   - SELECTORS_RC: use default mode map.
   - RESIDUAL_RC: empty residual stream.
   - RECEIVER_STATE: identity ego-motion/procedural predictor.
   - RDO_PLAN: uniform thresholds.
3. Inflate every variant to SSD-backed output and run full-video MLX scorer
   replay.
4. Emit per-section rows with `presence_delta_nonrate`, `charged_rate_cost`,
   admission delta, archive bytes, hashes, mutated-section name, false-authority
   flags, and exact blocker state.
5. Protect a section only if it is necessary and cannot be smaller-recoded.
   Residual atoms are admitted only when the repo policy delta is negative.
   Missing coverage routes to MLX replay, not promotion.

## Training and sweep commands

All future bulky outputs should go to `/Volumes/VertigoDataTier/pact` first.
These are design commands, not executed in this memo turn.

Staging smoke with current C3 residual sidecar:

```bash
python tools/materialize_c3_residual_pr106_sidecar.py \
  --output-dir /Volumes/VertigoDataTier/pact/lcci_c3_sparse_smoke_20260601T120216Z \
  --residual-mode l2_encoded \
  --encoding sparse \
  --sparse-aware \
  --decoded-raw /Volumes/VertigoDataTier/pact/inputs/pr106_decoded_full600.rgb \
  --gt-raw /Volumes/VertigoDataTier/pact/inputs/gt_full600.rgb \
  --byte-budget 2048 \
  --l2-iterations 2 \
  --use-hinton-distilled-scorer \
  --use-saliency-masking \
  --skip-no-op-smoke
```

Staging smoke with current Cool-Chic residual sidecar:

```bash
python tools/materialize_cool_chic_residual_pr106_sidecar.py \
  --output-dir /Volumes/VertigoDataTier/pact/lcci_cool_chic_sparse_smoke_20260601T120216Z \
  --residual-mode l2_encoded \
  --encoding sparse \
  --sparse-aware \
  --decoded-raw /Volumes/VertigoDataTier/pact/inputs/pr106_decoded_full600.rgb \
  --gt-raw /Volumes/VertigoDataTier/pact/inputs/gt_full600.rgb \
  --byte-budget 4096 \
  --l2-candidate-n-levels 1 2 3 \
  --per-level-top-k-budget 0:0,1:2048,2:1024,3:512 \
  --use-hinton-distilled-scorer \
  --use-saliency-masking \
  --skip-no-op-smoke
```

First true LCCI implementation queue after code landing:

```bash
python tools/build_hprc_compact_receiver_training_queue.py \
  --output /Volumes/VertigoDataTier/pact/lcci_c3vq_hprc_queue_20260601T120216Z/queue.json \
  --run-id lcci_c3vq_hprc_32pair_smoke_20260601T120216Z \
  --campaign-pairs 32 \
  --decode-height 96 \
  --decode-width 128 \
  --epochs 20 \
  --basis-count 4 \
  --residual-grid-h 24 \
  --residual-grid-w 32 \
  --training-backend mlx \
  --enable-native-rate-aware-hprc \
  --enable-hprc-mlx-prefilter-before-local-replay \
  --storage-tier /Volumes/VertigoDataTier/pact
```

Then expand only after receiver proof and local advisory improvement:

```bash
python tools/build_hprc_compact_receiver_training_queue.py \
  --output /Volumes/VertigoDataTier/pact/lcci_c3vq_hprc_queue_20260601T120216Z/queue_128_600.json \
  --run-id lcci_c3vq_hprc_128_600_sweep_20260601T120216Z \
  --campaign-pairs 128 600 \
  --decode-height 96 \
  --decode-width 128 \
  --epochs 50 \
  --basis-count 8 \
  --residual-grid-h 24 \
  --residual-grid-w 32 \
  --training-backend mlx \
  --enable-native-rate-aware-hprc \
  --enable-hprc-mlx-prefilter-before-local-replay \
  --storage-tier /Volumes/VertigoDataTier/pact
```

Section pricing after a true LCCI archive exists:

```bash
python tools/profile_lcci_hprc_section_value.py \
  --archive /Volumes/VertigoDataTier/pact/lcci_c3vq_hprc_candidate/archive.zip \
  --projection-manifest /Volumes/VertigoDataTier/pact/lcci_c3vq_hprc_candidate/hprc_projection_manifest.json \
  --output-dir /Volumes/VertigoDataTier/pact/lcci_c3vq_section_value_20260601T120216Z \
  --sections decoder_qw latents_rc codebooks_q selectors_rc residual_rc receiver_state \
  --max-pairs 600 \
  --window-pairs 25 \
  --scorer-batch-pairs 1
```

Exact CPU/CUDA dispatch remains blocked until the candidate is receiver-proven,
local full-video replay is plausibly below the exact target, and dispatch is
claimed through the canonical claim helper.

## Implementation plan

1. Add a family-specific LCCI codec package under
   `src/tac/substrates/lcci_codebook_receiver/` while keeping the outer packet
   format in `src/tac/substrates/hprc/archive.py`.
2. Implement `archive.py` helpers that map LCCI payloads into HPRC sections and
   preserve section hashes, bytes, and false-authority flags.
3. Implement `learned_receiver.py` with deterministic numpy/torch-free decode if
   possible: decode codebook entries, reconstruct multi-scale latent grids,
   apply tiny decoder, apply optional procedural motion predictor, then apply
   priced residual atoms.
4. Implement `mlx_trainer.py` for score-aware local training with hard archive
   byte estimates: VQ commitment/codebook loss, temporal index prediction,
   sparse residual admission, and eval-roundtrip.
5. Implement `section_value.py` with semantic neutralizers for every LCCI
   section. Do not allow random byte flips as evidence.
6. Implement `tools/profile_lcci_hprc_section_value.py` by adapting the PSV3
   profiler pattern to HPRC sections.
7. Use PR106 C3/Cool-Chic sidecar materializers only to price residual atom
   value before the true LCCI receiver exists.
8. Add tests after code landing:
   - deterministic pack/unpack roundtrip;
   - no sidecar or hidden state;
   - section neutralization stays syntactically valid;
   - residual tokens fail closed without negative admission delta;
   - archive.zip includes all constants/model bytes;
   - local receiver proof blocks score/promotion flags until exact replay.

## Precise blockers

- No current code emits a C3/Cool-Chic VQ codebook stack as HPRC CODEBOOKS_Q plus
  LATENTS_RC. Existing C3/Cool-Chic code is PR106 residual staging, not an
  independent full stack.
- Current HPRC compact receiver evidence says distortion, not byte count, is the
  live blocker. LCCI must lower nonrate score sharply before exact spend.
- Prior VQ full-renderer evidence had drift/gate failure. VQ should be used as
  codebook atoms and indices first, not as standalone promotion authority.
- SIREN/implicit-family prior has phantom/false-authority risk unless byte
  mutation and section-value proof show the atoms are actually consumed.
- RAFT/procedural priors are only admissible as compact deterministic state; a
  RAFT model or dense flow field inside archive.zip is almost certainly
  byte-negative.
- CLAdE/SPADE-style semantic conditioning is blocked unless it compiles to tiny,
  charged, deterministic modulation tables. Hidden masks or scorer outputs are
  forbidden.
- Full-video MLX replay is advisory only. Exact CPU/CUDA promotion needs a
  receiver-proven archive/runtime packet and canonical dispatch claim.
- Before any training/replay landing, storage preflight and auto-clean hooks must
  certify large artifacts or block, with SSD as the first target.

