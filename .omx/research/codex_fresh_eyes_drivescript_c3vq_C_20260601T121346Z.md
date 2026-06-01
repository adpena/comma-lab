# Codex fresh-eyes task C: DriveScript-C3VQ substrate

UTC: 2026-06-01T12:13:46Z
Agent: Codex
Scope: fresh-eyes research/audit task C, no staging, no commit, no push

## Executive decision

Pick **DriveScript-C3VQ** as the best next tiny generative-program /
latent-codebook / procedural-driving-prior substrate for this contest.

The receiver is not a full neural video codec and not a dense flow codec. It is
a byte-charged program:

1. a tiny C3/Cool-Chic-style coordinate/feature synthesis core,
2. multi-scale VQ residual tokens,
3. a compact procedural road/ego-motion state,
4. sparse scorer-valued residual atoms,
5. HPRC section accounting and semantic receiver proof.

This deliberately demotes HPRC/Z8 to **teacher, allocator, and residual
proposal source**, not primary payload. The local evidence says Z8/HPRC can make
rate collapse look cheap while PoseNet becomes catastrophically bad; it is not
currently a promotion substrate without a candidate-bound pose-preserving base.
This also demotes SIREN/FINER/WIRE/BACON from full-frame receiver to optional
micro-atoms, because their full continuous coordinate networks spend too many
charged constants for this archive regime.

Current frontier surfaces inspected by `tac.frontier_scan`:

- `[contest-CPU]` frontier: score 0.19198533626623068, archive 178493 bytes,
  archive SHA-256 b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c.
- `[contest-CUDA]` frontier: score 0.20533002902019143, archive 186876 bytes,
  archive SHA-256 9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4.

The objective is not RGB quality. The objective is hard archive-score RD:

```text
score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489
```

Every byte below must include runtime, ZIP/container overhead, all constants,
all entropy tables, all codebooks, all seeds, all masks, all transforms, and all
manifest fields needed by the receiver.

## Local audit observations

`.omx/code` is absent in this checkout, so I inspected repository code,
experiment artifacts, and `.omx/research` instead.

Relevant existing surfaces:

- `src/tac/substrates/hprc/archive.py` already provides a deterministic HPRC
  packet with `DECODER_QW`, `LATENTS_RC`, `CODEBOOKS_Q`, `SELECTORS_RC`,
  `RESIDUAL_RC`, `RDO_PLAN`, `RECEIVER_STATE`, and `MANIFEST_JSON`.
- `src/tac/substrates/hprc/lineage.py` already classifies C3/Cool-Chic, VQ,
  SIREN-family bases, RAFT motion state, and Z8 teacher residuals into HPRC
  roles and levers.
- `src/tac/raft_pose_stream.py` has the right doctrine for this design:
  RAFT may run at compress time, but inflate must read a charged compact stream;
  no RAFT model or dense flow field belongs in the submission runtime.
- Existing C3/Cool-Chic and residual materializers are useful, but the receiver
  must be promoted only through semantic section-value proof, not raw byte-flip
  parser proof.

Recent sibling memos were also consistent: the highest-EV family is compact
HNeRV/C3/VQ with explicit archive accounting, while pure HPRC/Z8 is still an
allocator/rate lesson rather than a byte-closed scorer winner.

## Literature synthesis

The outside literature supports the hybrid but not a naive transplant.

- COOL-CHIC demonstrates an extremely small coordinate-based hierarchical codec
  with hundreds of parameters and low per-pixel compute, which matches the
  contest need for a tiny receiver rather than a general pretrained model:
  https://openaccess.thecvf.com/content/ICCV2023/html/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.html
- C3 improves the COOL-CHIC line by overfitting a small model per image/video
  and targeting low decoding complexity with strong RD, which is almost exactly
  the contest setting except that this contest scores SegNet/PoseNet rather than
  PSNR/MS-SSIM:
  https://c3-neural-compression.github.io/
- VQ-VAE gives the discrete latent/codebook contract: indices plus a learned
  codebook are easier to charge, entropy-code, mutate, and value-profile than
  continuous residual blobs:
  https://papers.neurips.cc/paper/7210-neural-discrete-representation-learning
- RAFT is strong optical-flow machinery, but its actual correlation-volume model
  is not a submission receiver. Use RAFT offline to fit compact ego/procedural
  state and token priorities:
  https://arxiv.org/abs/2003.12039
- SIREN, FINER, WIRE, and BACON are valuable frequency/coordinate bases, but in
  this byte regime their full MLP weights are too expensive. The right use is
  sparse local atoms chosen only when section-value positive:
  https://arxiv.org/abs/2006.09661
  https://arxiv.org/abs/2312.02434
  https://arxiv.org/abs/2301.05187
  https://www.computationalimaging.org/publications/bacon/
- HNeRV and RT-NeRV support the idea that hybrid video representations need
  content-adaptive embeddings and residual tokenization. The contest variant
  must replace generic reconstruction RD with scorer-aware hard byte accounting:
  https://arxiv.org/abs/2304.02633
  https://arxiv.org/abs/2403.12401
- ANS/range coding is relevant only after the symbol model is shaped. Entropy
  coding does not rescue raw random mantissas; it should be used for final
  section streams and only with charged tables:
  https://dblp.uni-trier.de/rec/journals/corr/Duda13.html

## Byte grammar

Submission archive target:

```text
archive.zip
  inflate.sh
  inflate.py
  0.hprc
  optional tiny generic receiver module files
```

No video-specific constants are allowed in runtime code. The runtime may contain
generic decoding logic; every learned or fitted constant lives in `0.hprc` and
is charged.

`0.hprc` uses the existing HPRC outer structure:

```text
magic              : "HPRC\0\0\0\0"
schema_version     : u8 = 1
frames             : u16 = 1200
pairs              : u16 = 600
height             : u16 = 384
width              : u16 = 512
decoder_family_id  : u16 = DriveScript-C3VQ
color_transform_id : u16
gop_size           : u16 = 2
section_count      : u16
section_table[]    : kind:u16, offset:u64, length:u64, crc32:u32, sha256:32B
section_payloads   : raw charged bytes
```

Section budget envelopes:

| Section | Target bytes | Content |
| --- | ---: | --- |
| `DECODER_QW` | 40000-90000 | int4/int8 tiny synthesis core, scales, per-layer layout |
| `LATENTS_RC` | 8000-28000 | range/rANS coded multi-scale VQ indices and coarse fields |
| `CODEBOOKS_Q` | 3000-18000 | shared int8/int16 codebooks, residual atom dictionaries |
| `SELECTORS_RC` | 500-6000 | pair/block/region selectors, token priorities, significance tree |
| `RESIDUAL_RC` | 0-24000 | admitted sparse residual atoms only |
| `RECEIVER_STATE` | 300-2500 | procedural ego/road/horizon splines, affine warp coefficients |
| `RDO_PLAN` | 300-1500 | quant tiers, section gates, entropy model IDs |
| `MANIFEST_JSON` | 400-1500 | hashes, provenance, false-authority flags, proof pointers |

Global archive envelopes:

- 96 KB: stress target, must preserve PoseNet first.
- 150 KB: primary target; meaningful win region against current CPU byte term.
- 180 KB: control parity; must beat current `[contest-CPU]` score, not just bytes.
- 225 KB: escape envelope only if nonrate distance drops enough to pay for it.

Entropy coding:

- Use static range/rANS or the existing repository range-coder primitives for
  `LATENTS_RC`, `SELECTORS_RC`, and `RESIDUAL_RC`.
- Charge all CDF/table bytes.
- Use Brotli/ZIP only as final container shaping, not as the main explanation.
- Refuse raw f32 detail streams and raw random mantissas as primary payload.

## Receiver adapter shape

Inflate-time adapter:

1. Parse `0.hprc`; verify CRC/SHA and deterministic section order.
2. Decode section streams. Missing or extra section payloads fail closed.
3. Materialize `RECEIVER_STATE`: low-order camera/horizon/road splines, per-GOP
   affine/ego coefficients, and optional per-pair residual predictors. No RAFT,
   scorer, source video, network fetch, or sidecar path at inflate.
4. Decode multi-scale VQ grids:
   - coarse grid: road/sky/car/lane layout support;
   - mid grid: object and boundary support;
   - fine grid: only scorer-valued token sites.
5. Run the tiny synthesis core:
   - coordinate inputs: normalized x/y, pair index, GOP phase, procedural
     ego-state features;
   - latent inputs: decoded VQ embeddings and selector bits;
   - output: RGB frame pair in the exact evaluator roundtrip dtype/range.
6. Apply sparse residual atoms from `RESIDUAL_RC`. Atom families may include
   DCT/wavelet/Gabor and micro-SIREN/FINER/WIRE/BACON atoms, but each atom must
   carry its type, position, scale, coefficients, and quantization in bytes.
7. Write full-frame outputs for `inflate.sh archive_dir output_dir file_list`.

The adapter should be ordinary and auditable: one deterministic interpreter,
not a hidden model zoo.

## Training objective

Train with hard-pack outer loops, not RGB-only surrogate success.

```text
L_total =
  100 * L_seg_teacher
  + sqrt(10 * L_pose_teacher + eps)
  + lambda_rgb * L_eval_roundtrip_rgb
  + lambda_rate * R_hat
  + beta_vq * L_commitment
  + beta_entropy * CE(symbols | context)
  + beta_motion * L_warp_residual
  + beta_sparse * L_sparse_atoms
  + beta_parity * L_train_vs_receiver_drift
```

Where:

- `L_seg_teacher` is class/boundary-weighted SegNet distillation from the
  current exact-frontier decoded outputs and/or ground truth where permissible.
- `L_pose_teacher` is full 6D PoseNet-aware distance, not dim0-only drift.
- `R_hat` begins as entropy estimate but is replaced by actual packed
  `archive.zip` bytes at every promotion gate.
- `L_eval_roundtrip_rgb` forces the training path and receiver path to agree
  after uint8/file-list roundtrip.
- The allocator uses actual contest-score deltas, never PSNR-only deltas.

Curriculum:

1. Fit base C3/Cool-Chic/VQ receiver at 32 pairs, then 128 pairs, then full 600.
2. Freeze and quantize `DECODER_QW`; retrain codebooks and latents under QAT.
3. Introduce procedural state using RAFT/ego signals as compress-time teachers.
4. Add residual atoms only through section-value admission.
5. Repack and replay every accepted checkpoint.

## Value-per-byte measurement

For every section, token class, atom family, and motion-state field, measure:

```text
score_nonrate = 100 * seg_dist + sqrt(10 * pose_dist)
rate_score    = 25 * archive_bytes / 37545489
score_total   = score_nonrate + rate_score
```

For candidate item `i`:

```text
delta_nonrate_i =
  score_nonrate(with_i) - score_nonrate(semantic_neutralized_i)

delta_rate_i =
  25 * (archive_bytes(with_i) - archive_bytes(without_i)) / 37545489

net_delta_i = delta_nonrate_i + delta_rate_i
value_per_kib_i = -delta_nonrate_i / max(1, byte_delta_i / 1024)
```

Admission rule:

- Admit if `net_delta_i < -max(3 * replay_noise, 0.00002)` on local advisory
  replay and still negative under a second semantic neutralization run.
- Promote only if the full candidate, including ZIP/runtime/container bytes,
  improves the matching auth axis.
- Neutralization must be semantic: zeroing token coefficients, replacing a
  codebook entry with its decoded mean, disabling a residual atom through the
  parser, or dropping a receiver-state field. Raw bit flips prove parser
  integrity only.

Section-value output must record:

- original path, archive bytes, archive SHA-256;
- runtime-tree hash and inflated output aggregate hash;
- exact command, argv, environment summary, and file list;
- per-section bytes, neutralized score, component deltas, and proof status;
- whether the result is `[macOS-MLX research-signal]`, `[macOS-CPU advisory]`,
  `[contest-CPU]`, or `[contest-CUDA]`.

## Promotion gates

G0: Ownership and storage

- Work on `main`; do not stage partner files.
- Before dispatch/training, claim the lane with `tools/claim_lane_dispatch.py`
  when the run is training/eval/remote-GPU.
- Use `/Volumes/VertigoDataTier/pact` first, then `/Volumes/APDataStore/pact`.
- Run storage preflight and record cleanup/cold-store manifests for large
  artifacts. If reproducibility proof is missing, block cleanup and keep bytes.

G1: Archive grammar authority

- All constants are in archive bytes or generic runtime bytes.
- No hidden source-video, scorer-output, RAFT-output, local path, pretrained
  checkpoint, or uncharged table.
- Manifest says `score_claim=false` until exact replay.

G2: Receiver proof

- `inflate.sh archive_dir output_dir file_list` consumes the archive.
- Full-frame output aggregate hashes are recorded.
- Valid semantic section mutations change decoded outputs as expected.

G3: Section-value proof

- `DECODER_QW`, `LATENTS_RC`, `CODEBOOKS_Q`, `SELECTORS_RC`, `RESIDUAL_RC`, and
  `RECEIVER_STATE` each have neutralization deltas.
- No section survives on provenance or hope alone.

G4: Local advisory gate

- Full-600 local CPU replay must beat the matching current local/control
  target by at least 0.00002 total score or 3x measured replay noise.
- MLX outputs may rank candidates but may not claim authority.

G5: Auth-axis gate

- `[contest-CPU]` and `[contest-CUDA]` are separate.
- No CPU-to-CUDA inference, CUDA-to-CPU inference, or MLX-to-contest promotion.
- Exact eval dispatch requires active claim row and complete harvest plan.

G6: Submission hygiene

- Archive SHA, bytes, member hashes, runtime hash, commands, seed/config, and
  exact score schema are recorded.
- Public report labels every score/rank phrase with its axis.

## First three executable experiments

These are deliberately local/plan-first. They write under SSD paths and do not
stage, commit, push, or spend remote GPU by themselves.

### Experiment 1: DriveScript HPRC baseline queue and timing smoke

Purpose: turn the design into a queue-backed 32/128-pair compact receiver smoke
with native-rate HPRC and MLX prefilter, preserving byte/proof metadata.

```bash
OUT=/Volumes/VertigoDataTier/pact/drivescript_c3vq_C_20260601T121346Z
mkdir -p "$OUT"

.venv/bin/python tools/build_hprc_compact_receiver_training_queue.py \
  --output "$OUT/hprc_queue.json" \
  --plan-output "$OUT/hprc_plan.json" \
  --run-id drivescript_c3vq_32_128_C_20260601T121346Z \
  --campaign-pairs 32 \
  --campaign-pairs 128 \
  --decode-height 96 \
  --decode-width 128 \
  --epochs 80 \
  --batch-pair-indices-per-step 16 \
  --learning-rate 0.001 \
  --curriculum-preset hprc_native_rate_ramp_v1 \
  --basis-count 8 \
  --residual-grid-h 24 \
  --residual-grid-w 32 \
  --training-backend mlx \
  --enable-native-rate-aware-hprc \
  --enable-hprc-mlx-prefilter-before-local-replay \
  --hprc-mlx-prefilter-scorer-batch-pairs 1 \
  --storage-tier /Volumes/VertigoDataTier/pact \
  --storage-workload-subdir drivescript_c3vq_C_20260601T121346Z \
  --auth-frontier-score 0.19198533626623068 \
  --local-baseline-score 0.19198533626623068 \
  --min-local-improvement 0.00002
```

Success: queue plus plan are created with no local-disk bulk, timing smoke
reports seconds/epoch, and every output directory has cleanup/provenance.

### Experiment 2: C3/Cool-Chic sparse atom sidecar smoke

Purpose: price whether existing C3/Cool-Chic residual materializers can supply
positive-value residual atoms for DriveScript rather than raw residual blobs.

Precondition: `DECODED_RAW` and `GT_RAW` must be durable SSD files with byte and
SHA-256 manifests. If they do not exist, create them through existing replay
tooling first; do not use `/tmp`.

```bash
OUT=/Volumes/VertigoDataTier/pact/drivescript_c3vq_C_20260601T121346Z
DECODED_RAW=$OUT/cache/frontier_decoded_full600.rgb
GT_RAW=$OUT/cache/gt_full600.rgb

.venv/bin/python tools/materialize_c3_residual_pr106_sidecar.py \
  --output-dir "$OUT/c3_sparse_atom_smoke" \
  --residual-mode l2_encoded \
  --encoding sparse \
  --sparse-aware \
  --decoded-raw "$DECODED_RAW" \
  --gt-raw "$GT_RAW" \
  --byte-budget 2048 \
  --l2-iterations 2 \
  --use-hinton-distilled-scorer \
  --use-saliency-masking \
  --skip-no-op-smoke

.venv/bin/python tools/materialize_cool_chic_residual_pr106_sidecar.py \
  --output-dir "$OUT/cool_chic_sparse_atom_smoke" \
  --residual-mode l2_encoded \
  --encoding sparse \
  --sparse-aware \
  --decoded-raw "$DECODED_RAW" \
  --gt-raw "$GT_RAW" \
  --byte-budget 4096 \
  --l2-candidate-n-levels 1 2 3 \
  --per-level-top-k-budget 0:0,1:2048,2:1024,3:512 \
  --use-hinton-distilled-scorer \
  --use-saliency-masking \
  --skip-no-op-smoke
```

Success: at least one sparse atom family shows negative `net_delta_i` after
byte cost on full receiver replay. Failure still becomes a canonical negative:
"C3/Cool-Chic atom sidecar is not value-positive at this budget."

### Experiment 3: section-value control against current PSV3 archive

Purpose: establish the neutralization/value protocol on a known current archive
before trusting DriveScript section profiles.

```bash
OUT=/Volumes/VertigoDataTier/pact/drivescript_c3vq_C_20260601T121346Z

.venv/bin/python tools/profile_pact_nerv_selector_v3_mlx_section_value.py \
  --archive /Volumes/VertigoDataTier/pact/pact_nerv_selector_v3_int8_decoder_raw_adapter_20260601Tlocal/local_cpu_replay/submission/archive.zip \
  --output-dir "$OUT/psv3_section_value_control" \
  --sections decoder_qw latents_rc selectors_rc residual_rc \
  --max-pairs 600 \
  --window-pairs 25 \
  --scorer-batch-pairs 1 \
  --device gpu \
  --allow-large-tensor-cache \
  --force
```

Success: section neutralization produces component deltas and catches dead or
byte-negative sections. This becomes the acceptance template for a future
DriveScript-specific section-value profiler.

## Kill criteria

Retire or redesign DriveScript-C3VQ if any of these hold:

- Base receiver cannot get full-600 local advisory score within 0.005 of the
  current `[contest-CPU]` frontier at <=180 KB after hard pack.
- PoseNet distance explodes under native-rate pressure as in prior HPRC/Z8
  corrected runs.
- `RECEIVER_STATE` is not value-positive after its byte cost.
- Residual atoms are only RGB-positive but scorer-negative.
- Section neutralization shows hidden constants or dead sections.
- Exact replay fails to consume the same archive/runtime path used in local
  proof.

## Final recommendation

Run Experiment 3 first if the goal is to harden the measurement protocol with
the least new surface area. Run Experiment 1 first if the goal is frontier
escape. Do not run pure Z8/HPRC or pure SIREN-family full-frame training as the
next primary campaign; both are valuable as teachers or atom sources, but the
archive-byte ruling constraint favors a tiny C3/Cool-Chic/VQ/procedural receiver
with section-value admission.
