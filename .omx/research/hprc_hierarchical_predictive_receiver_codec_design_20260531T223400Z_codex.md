# HPRC — Hierarchical Predictive Receiver Codec

Axis: design/control artifact
Score claim: `false`
Promotion eligible: `false`

## Why This Exists

Z8/HPC byte profiling shows the explicit wavelet-store representation is not
contest-rate competitive:

- baseline Z8 `archive.zip`: `28,504,909` bytes;
- best profiled explicit-detail collapse artifact: `10,195,155` bytes;
- current CPU frontier archive bytes: `178,493`;
- PR95/HNeRV public archive: about `178 KB`, with compressed decoder around
  `162 KB` and full `(600,28)` latents around `16 KB`;
- Z8 top-LL raw payload alone: `11,059,200` bytes.

Therefore the winning representation must collapse the object being transmitted.
Wavelets remain useful as teacher, residual basis, and action surface; they must
not remain the primary archive payload.

## Design Principle

Transmit only the information the receiver cannot infer under the contest
scorer.

The scalar objective is:

```text
S = 100*d_seg(full_video)
  + sqrt(10*d_pose(full_video))
  + 25*archive_bytes/rate_denominator
```

HPRC treats this as a scorer-conditioned MDL problem:

```text
min_theta,z,c,r  S(receiver(theta, z, c, r))

theta = tiny decoder / receiver weights
z     = pair or GOP latents
c     = selectors / codebook ids / mode stream
r     = conditional residual tokens
```

The receiver predicts as much as possible deterministically, then the archive
pays only for residual tokens whose full-video scorer marginal justifies their
bytes.

## Archive Grammar V0

```text
archive.zip
  inflate.sh
  inflate.py
  hprc.bin

hprc.bin:
  magic: "HPRC"
  version: u8
  header:
    frames=1200
    pairs=600
    H=384
    W=512
    decoder_family_id
    color_transform_id
    gop_size
    section_count
  section_table:
    id, offset, length, crc32, sha256_prefix
  sections:
    DECODER_QW
      quantized tiny decoder weights
    LATENTS_RC
      pair/GOP latent stream, entropy-coded
    CODEBOOKS_Q
      residual-token codebooks
    SELECTORS_RC
      block/band/pair mode selectors
    RESIDUAL_RC
      conditional residual token stream
    RDO_PLAN
      truncation thresholds and band budgets
```

Every charged section must have a receiver-consumption proof: mutating it must
change inflated pixels or be refused as archive-bound metadata only.

## Receiver

For each pair/GOP:

1. Decode latent `z_i`.
2. Predict coarse pair frames and multiscale features:
   `base_i = decoder(theta, pair_index, z_i, receiver_state)`.
3. Generate side information:
   `side_i = f(prev_decoded, ego_motion_proxy, pair_index, codebooks)`.
4. Decode selectors:
   `mode[pair, block, band] in {copy, receiver_predict, latent_decode, residual_token, zero}`.
5. Decode residual tokens only where selected.
6. Reconstruct final frames.

Mamba/RSSM/Dreamer state is allowed only when regenerated at inflate time. Do
not serialize per-pair hidden state; that reintroduces the explicit-video bug.

## Training Loop

1. Train PR95/HNeRV-like tiny decoder on full contest video with real uint8 and
   scorer preprocessing in loop.
2. Use current Z8 wavelet pyramids as teacher surfaces, not archive payload.
3. Train receiver/side-information model to predict coarse top-LL and detail
   residuals.
4. Train VQ/residual tokenizer over `target - receiver_prediction`.
5. Optimize selector streams with hard packed-byte estimates and scorer
   gradients:
   `DeltaS = DeltaDistortion + lambda*DeltaBytes`.
6. Export byte-closed `hprc.bin`.
7. Run local MLX scorer prefilter, full local CPU replay, then exact CPU/CUDA
   only if local evidence clears the eureka gate.

## Immediate Implementation Path

1. Keep PR95 MLX as the control arm:
   full `(600,28)` latents, decoder parity, full-frame inflate parity, archive
   package proof, then bounded longer training with scorer-oriented loss.

2. Convert Z8 explicit wavelet artifacts into HPRC teacher data:
   top-LL, details, joint P18/P19 gradients, PoseNet-null masks, SegNet-region
   weights, and byte-profile curves become training targets and RD priors.

3. Build a minimal HPRC V0 materializer:
   PR95-style decoder+latents plus optional residual token sidecar. The first
   sidecar can target the exact Z8 top-LL residual finding:
   frame1 top-LL residual `q=0.25` is about `28.7 KB` advisory payload under the
   fast byteplane probe.

4. Add section-level pair-blob materializer for the current Z8 artifact:
   q=11 solid raw-pair brotli is `402,311` bytes smaller than independent pair
   blobs. This is a real but secondary bridge, useful while HPRC V0 trains.

5. Replace scalar MSE-only acceptance with full-video scorer-conditioned RD:
   SegNet/PoseNet gradients, component calibration, exact archive bytes, and
   hard replay decide acceptance.

## What To Avoid

- Do not store top-LL/detail float fields as the final representation.
- Do not serialize Mamba/RSSM state per pair.
- Do not rely on ZIP/repack to fix pre-entropy payload entropy.
- Do not promote MLX/local advisory results as exact score authority.
- Do not make residual tokens independent of receiver side information; the
  whole point is conditional coding.

## Open Engineering Tasks

- `hprc.bin` parser/packer with section table and mutation proofs.
- MLX-local HPRC V0 trainer based on PR95 decoder/latents.
- Residual-token/codebook trainer over Z8 teacher residuals.
- Selector stream coder with PacketIR/range/ANS backend.
- Full-video queue-owned replay gate.
- Exact CPU/CUDA promotion gate with axis-separated custody.

## External Reference Families

- PR95/HNeRV: compact decoder plus learned latents.
- NeRV/HNeRV/HiNeRV/SRNeRV: implicit neural video representations.
- RT-NeRV: residual tokenization.
- Cool-Chic/C3: overfitted neural codec with entropy-coded latents.
- DCVC-FM/DCVC-CM: conditional learned video coding and side information.
- JPEG2000/EBCOT: wavelet bitplane truncation and RD layer selection.
- ZFP/SZ/MGARD/fpzip: float-array compression, useful as baselines but not
  sufficient to collapse top-LL sample count alone.
