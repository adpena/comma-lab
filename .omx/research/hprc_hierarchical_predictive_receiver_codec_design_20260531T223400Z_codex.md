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
change inflated pixels through a valid semantic mutation and full receiver
replay, or be refused as archive-bound metadata only. Raw byte flips only prove
parser/hash integrity.

Authority manifests use full SHA-256 digests. Digest prefixes are allowed only
for human display, never for custody or promotion.

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
   fast byteplane probe. This number is not archive authority until the payload
   is packed into `archive.zip`, consumed by `inflate.sh`, replayed locally, and
   counted with runtime/config/table/container bytes.

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
- Do not count latent/residual payload bytes alone as the byte ceiling. HPRC
  candidates must account for `archive.zip`, `inflate.py`, config, decoder code,
  entropy tables, and every receiver-side dependency before promotion.

## Current Implementation Landing

UTC: 2026-05-31T22:43:51Z

Code landed:

- `src/tac/substrates/hprc/archive.py`
  - deterministic `hprc.bin` fixed header + section table + raw charged
    section payload grammar;
  - full SHA-256 + CRC32 per section;
  - fail-closed non-authority manifest with byte accounting that explicitly
    excludes runtime/ZIP until a full submission packet is built;
  - receiver proof status that refuses to confuse raw hash flips with semantic
    receiver replay.
- `src/tac/substrates/hprc/lineage.py`
  - family-binding registry that maps PR95/HNeRV, RNeRV/PACT, Z8, C3/Cool-Chic,
    SIREN/COIN, RAFT, CLade/SPADE, and EBCOT/SPIHT into HPRC roles and allowed
    packet sections;
  - hard rule that Z8 is teacher/residual/allocator, not a primary explicit
    wavelet payload again.
- `src/tac/substrates/hprc/archive_candidate.py`
  - byte-closed HPRC `archive.zip` exporter;
  - `hprc_archive_byte_ledger.json` with runtime/container bytes counted;
  - archive-bound candidate package emission with exact-readiness blockers.
- `src/tac/substrates/hprc/inflate.py`
  - deterministic decode-only scaffold receiver for mutation/replay custody;
  - pixel-driving digest excludes metadata-only sections.
- `src/tac/substrates/hprc/pr95_adapter.py`
  - imports PR95's compressed decoder+latent envelope as the HPRC byte-scale
    control arm;
  - observed PR95 control wrap: `hprc_bin=179,187`, source `archive.zip=178,417`,
    delta `+770` bytes from section table + HPRC manifest overhead.
- Tests:
  - deterministic pack/parse/repack;
  - corruption refusal;
  - manifest non-authority;
  - family-role separation and rate-first rules.
  - PR95 control wrapping;
  - archive-bound export and mutation proof.

Receiver-proven scaffold artifact:

- `.omx/research/hprc_minimal_archive_candidate_20260531T225027Z/archive.zip`
  - `8,291` bytes, rate term `0.005520636580335922`;
  - `score_claim=false`, `promotion_eligible=false`,
    `ready_for_exact_eval_dispatch=false`;
  - `inflate.sh` receiver proof streamed `3,662,409,600` raw output bytes in
    `2.102343` seconds and deleted the raw output afterward;
  - disk preflight recorded `4,736,151,424` scratch bytes required and
    `1,019,568,099,328` free bytes available;
  - proves export/ledger/mutation-proof plumbing only. It is not a trained
    renderer and not a score candidate.

This is still a scaffold, not a candidate. It is the archive spine that the
MLX train/export/archive lane should target next.

Codex review fix:

- The first HPRC package made the section mutation proof visible only as an
  HPRC-side JSON. The shared archive-bound contract therefore reported
  `runtime_payload_consumption_metadata_absent`, which would orphan the next
  training/materializer tasks from the bounded runner.
- Fixed by making HPRC's transform kind explicitly predictive-coding and
  projecting section-level pixel-consumption proof into the shared runtime
  payload surface.
- Current contract status is
  `section_pixel_consumption_proven_full_stack_claim_blocked`, with three
  runner-visible backlog tasks:
  `replace_hprc_v0_receiver_scaffold_with_trained_renderer_export`,
  `attach_z8_scorer_weighted_residual_sidecar`, and
  `prove_mamba_dreamer_wyner_ziv_sections_drive_receiver_pixels`.

## Synergy Stack Policy

HPRC is a stack-of-stacks only when each member contributes at the right entropy
position:

- PR95/HNeRV/RNeRV/PACT: compact receiver and latent stream.
- C3/Cool-Chic: overfitted neural codec discipline, entropy-coded latents, and
  low decode complexity.
- Z8/HPC: teacher residuals, wavelet atoms, P18/P19 allocator surfaces, and
  residual sidecar coding only.
- RAFT/motion: side-information teacher or compact procedural receiver state;
  never dense flow serialization.
- CLade/SPADE/semantic conditioning: scorer-region conditioning and allocation
  if class/semantic state is derived or compactly charged; never hidden SegNet
  outputs.
- SIREN/COIN: patch/residual implicit atoms when byte-positive under P18/P19,
  not default full-video carrier.
- EBCOT/SPIHT/bitplane coding: residual-token sidecar after base receiver
  collapse, not a rescue plan for full explicit wavelet video.

The new paradigm is therefore: compact learned receiver first, then conditional
residual tokens selected by exact full-video scorer marginal value.

## V0 Simplicity Contract

HPRC V0 should stay faithful, simple, and byte-auditable:

1. PR95/HNeRV-scale compact receiver control.
2. Deterministic `hprc.bin` section grammar.
3. Archive-bound export through `archive.zip` and `inflate.sh`.
4. Full SHA-256 section custody.
5. Semantic mutation proof plus full receiver replay before promotion.
6. Byte ledger that includes runtime/config/table/container overhead.

Do not pull every NeRV-family idea into V0. HNeRV variants, RNeRV, PACT-NeRV,
C3/Cool-Chic, SIREN/COIN, RAFT/motion priors, CLade/SPADE semantic
conditioning, EBCOT/SPIHT residual coders, Muon/AdamW/LSQ-QAT, and PacketIR/
range/ANS compiler work are HPRC V1/V2 ingredients. They should be admitted
only when they solve a measured HPRC bottleneck: receiver fidelity, latent
entropy, residual entropy, scorer allocation, decode speed, or archive/runtime
byte overhead.

Sooner-than-later research sweep:

- audit every `src/tac/substrates/*nerv*`, `siren`, `coin*`, `cool_chic`,
  `balle`, `dreamer`, `z7`, `z8`, `d4`, and optimizer/export surface;
- classify each as `base_receiver`, `latent_stream`, `residual_tokenizer`,
  `motion_side_info`, `semantic_conditioner`, `entropy_model`,
  `scorer_allocator`, `teacher_only`, or `baseline_oracle`;
- copy no code into HPRC until the role, byte budget, receiver proof, and
  exact replay gate are defined.

## Optimizer Lever Taxonomy

The complete HPRC optimizer menu is broader than the core V0 knobs. Each lever
must still enter through measured byte accounting and replay:

- receiver weight quantization, pruning, low-rank adapters;
- latent prediction, latent entropy coding, vector quantization, shared
  codebooks;
- residual token waterfill, bitplane truncation, significance-tree coding;
- motion-compensated side information without dense flow serialization;
- semantic conditioning without hidden scorer/class-map sidecars;
- scorer-weighted ablation, SegNet boundary repair, PoseNet-null allocation;
- QAT/LSQ noise shaping, Muon/AdamW curriculum, per-layer schedules;
- range/ANS/arithmetic coding, brotli/repack ordering, PacketIR section
  compilation;
- native Rust/Zig decode kernels when Python decode is the bottleneck;
- full-video bundle/KKT allocation and exact replay acceptance.
- invented receiver paradigms, generated operator search, and cross-family stack
  synthesis, provided they first register role, section mapping, byte budget,
  receiver proof, and replay gate.

The ordering matters: first collapse the receiver representation, then spend
residual bits only where full-video P18/P19 scorer marginals justify them.

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
