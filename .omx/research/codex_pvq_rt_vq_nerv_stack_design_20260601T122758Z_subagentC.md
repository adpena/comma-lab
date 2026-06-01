# PVQ / RT-VQ-NeRV Latent-Codebook Stack Design - Subagent C

utc: 20260601T122758Z
author: Codex subagent C
status: design_memo_only
scope: PR95-style byte grammar compatible latent-codebook / PVQ / RT-VQ-NeRV / C3-Cool-Chic-inspired design
authority: proposal only; no score claim; no rank, kill, promotion, or exact-dispatch authority

## Write boundary and preflight note

This pass read AGENTS.md and CLAUDE.md first, inspected the relevant local PR95, VQ, selector, C3, Cool-Chic, HPRC, Z8/residual surfaces, and used online research for the codec design references below. The operator requested exactly one memo write. I therefore did not modify code, dispatch claims, lane registry state, canonical task state, or experiment queues. Any real training/eval/dispatch follow-up must first register/claim the lane through the existing canonical helpers.

## Online sources used

- PVQ: Valin and Terriberry, "Perceptual Vector Quantization For Video Coding", arXiv:1602.05209. The paper applies gain-shape vector quantization, transmits gain plus direction/shape, and derives coding for non-uniform PVQ codewords. https://arxiv.org/abs/1602.05209
- RT-NeRV: Xu et al., "RT-NeRV: Rethinking Hybrid Neural Representations for Video via Residual Tokenization", arXiv:2403.12401, last revised 2026-05-22. It frames the key low-rate gap as costly continuous shallow/inter-frame residual support and proposes residual tokens plus residual-aware codebook learning. https://arxiv.org/abs/2403.12401
- COOL-CHIC: Ladune et al., "COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec", ICCV 2023. It overfits a tiny coordinate-based hierarchical decoder and sends quantized latent/parameter bitstreams with entropy coding. https://openaccess.thecvf.com/content/ICCV2023/html/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.html
- C3: Kim et al., "C3: High-Performance and Low-Complexity Neural Compression from a Single Image or Video", CVPR 2024. It builds on COOL-CHIC, overfits a small model per image/video, and extends the approach to video with low decoding complexity. https://openaccess.thecvf.com/content/CVPR2024/html/Kim_C3_High-Performance_and_Low-Complexity_Neural_Compression_from_a_Single_Image_CVPR_2024_paper.html

## Local surfaces inspected

- PR95 grammar and helpers: `src/tac/pr95_hnerv.py`, `src/tac/local_acceleration/pr95_hnerv_mlx.py`, `src/tac/substrates/hprc/pr95_adapter.py`.
- PR95 residual planning: `src/tac/pr95_residual_atoms.py`.
- VQ and latent-codebook surfaces: `src/tac/substrates/pact_nerv_vq/archive.py`, `src/tac/quantization_wave/vq_codebook_quantization.py`, `src/tac/vqvae_as_full_renderer.py`.
- Selector/entropy grammar surfaces: `src/tac/codec/pr101_polymorphic.py`, `src/tac/hnerv_pr101_schema_packer.py`, `src/tac/hnerv_pr103_lc_ac_schema.py`.
- C3/Cool-Chic residual scaffolds: `src/tac/residual_basis/c3_residual.py`, `src/tac/residual_basis/c3_encoder_l2.py`, `src/tac/residual_basis/cool_chic_residual.py`, `src/tac/residual_basis/cool_chic_encoder_l2.py`.
- HPRC/Z8 spine surfaces: `src/tac/substrates/hprc/archive.py`, `src/tac/substrates/hprc/learned_receiver.py`, `src/tac/substrates/hprc/lineage.py`, `src/tac/substrates/hprc/native_rate_surface.py`, `src/tac/substrates/hprc/spine_bounded_runner.py`.
- Recent sister findings/design memos, especially the HPRC Z8 distortion/rate failures, PR95 raw-section value finding, and Subagent B latent-codebook proposal.

## Current local prior

The strongest immediate route is not a full new image/video codec. PR95 already has the right contest-shaped receiver: one stored `archive.zip` member named `0.bin`, three length-prefixed brotli blobs, compact HNeRV decoder, and 600 rows of 28-dimensional latents. The most promising stack is a PR95-compatible latent-token graft:

1. Keep PR95/HNeRV as the base receiver and byte grammar.
2. Replace the scalar latent blob with a typed latent packet that can reconstruct exactly the 600x28 latent tensor consumed by the decoder.
3. Encode residual latent support using product VQ, PVQ gain-shape escapes, and RT-NeRV-style residual tokens.
4. Admit C3/Cool-Chic residual atoms only when exact section-value replay proves they beat their byte charge.

This avoids the failure pattern seen in the current HPRC compact receiver: tiny archives with excellent rate but catastrophic PoseNet. It also avoids the current C3/Cool-Chic scaffold problem: dense residual grids are too expensive unless made sparse and score-priced.

## Top-level PR95-style packet

Keep the public-style PR95 envelope:

```text
archive.zip
  0.bin  # stored ZIP member

0.bin =
  meta_len:u32
  meta_brotli:bytes
  decoder_len:u32
  decoder_brotli:bytes
  latents_len:u32
  latents_brotli:bytes
```

`decoder_brotli` remains PR95 decoder weights at first. A decoder-adapter section is forbidden in phase 1 unless section value later proves it. `latents_brotli` becomes `PVQRT1`, a typed packet decoded by the modified runtime into the same float latent matrix that PR95 already consumes.

`meta_brotli` carries only charged deterministic state:

- schema id, archive/runtime contract id, and parser version;
- number of pairs, latent dim, quantization profile, and PR95 decoder family id;
- entropy table hashes and section manifest;
- predictor parameters and seeds, if any;
- false-authority flags and exact-gate blockers;
- codebook and residual-stream SHA-256 entries.

No external flow, no scorer-derived sidecar, no uncharged procedural source, and no hidden runtime constants are allowed. If a prior, predictor, table, or seed changes decoded pixels, it lives in `0.bin`.

## Latent packet grammar

Recommended inner grammar:

```text
PVQRT1 packet =
  magic:"PVQRT1\0"
  schema:u8
  n_pairs:u16          # 600
  latent_dim:u8        # 28
  group_count:u8       # default 4
  stream_count:u8
  stream_table[stream_count]
  stream_payloads...

stream_table entry =
  stream_id:u8
  codec_id:u8          # raw, brotli, canonical_huff, range, enum_rank
  raw_len:u32
  coded_len:u32
  sha256_16:bytes[16]
```

Payload streams:

- `predictor_state`: tiny temporal predictor state. Phase 1 is previous-row delta only; phase 2 may add archive-charged affine ego/procedural coefficients.
- `mode_map`: entropy-coded mode per pair and latent group. Modes: no-op, scalar escape, PVQ, VQ, VQ+PVQ residual, residual-atom pointer.
- `product_codebooks_q`: product codebooks for residual latent vectors in quantized latent units.
- `vq_indices`: entropy-coded codebook indices, conditioned on previous pair/group.
- `pvq_gain_k`: gain and pulse-count stream for PVQ escapes.
- `pvq_shape_rank`: enumerative PVQ shape ranks and sign bits.
- `scalar_escape`: PR101/PR103-style sidecar for rows/groups where codebook/PVQ would damage scorer output.
- `residual_atoms`: optional score-priced sparse image/feature residual atoms. Default absent.
- `section_value_manifest`: charged machine-readable map from each stream to bytes, neutralization hash, and replay delta. Metadata only; not authority by itself.

The runtime must decode these streams into a deterministic `latents_f32[600,28]` tensor before calling the unchanged PR95 decoder. A no-op packet must round-trip to the source PR95 latents byte-for-byte after the same dequantization path.

## Codebook and index design

The first codebook should be product-VQ over PR95 latent residuals, not a full-frame VQ-VAE:

- Base latent predictor: `pred[i] = latent[i-1]` for pair `i>0`, plus optional charged affine correction after evidence.
- Residual: `r[i] = latent[i] - pred[i]` in PR95 quantized latent units.
- Grouping: split 28 dims into 4 groups of 7 dims. A sensitivity-covariance grouping can replace fixed groups only after VJP/Jacobian evidence.
- Codebooks: per-group K=32 first, K=64 only if section value pays. Store entries as int8 residual vectors with one scale and zero point per group.
- Indices: one symbol per pair/group. K=32 costs 600*4*5 = 12,000 raw bits = 1,500 bytes before entropy. K=64 costs 1,800 bytes raw.
- Overhead: 4 groups * 32 entries * 7 dims = 896 raw codebook bytes plus scales/table overhead; K=64 is 1,792 raw bytes.
- Fallback: scalar escape must be available per group. Do not force high-sensitivity PoseNet rows through a lossy shared codebook.

The product VQ should be trained in a scorer-weighted latent metric, not plain L2. Approximate the group metric with local VJP/Jacobian anchors:

```text
distance(group_residual) =
  alpha_rgb * ||decoder_delta_rgb||_yuv6^2
  + alpha_seg * predicted_segnet_delta
  + alpha_pose * predicted_posenet_delta
  + alpha_rate * coded_bits
```

When VJP evidence is stale or missing, default to PR95 scalar escape rather than trusting Euclidean latent distance.

## PVQ escape design

PVQ is valuable where a group has a coherent direction and varying gain. For each group residual `x` in a whitened 7D space:

1. Quantize gain `g = ||x||` with a small Rice/Huffman-coded scalar.
2. Choose pulse count `K_p` from a learned table conditioned on group, gain bucket, and previous pair.
3. Approximate shape by integer pulse vector `y` with `sum(abs(y_j)) = K_p`.
4. Encode shape by combinatorial rank plus sign bits; entropy-code rank residuals or bucketed ranks when distribution is non-uniform.
5. Reconstruct `x_hat = dewhiten(scale(g) * y / ||y||)`.

PVQ should not replace VQ universally. It is the escape mode for:

- high-energy residuals where VQ would require large K;
- smooth directional changes where gain-shape beats scalar deltas;
- rows with high PoseNet sensitivity, where preserving direction matters more than scalar quantizer density.

The PR101 `HUFF_ENUM` and PR103 range-coded streams are the local pattern to reuse: keep enumerative ranks for structured symbols, split streams by family, and avoid mixing distributions just because brotli can absorb the bytes.

## RT-NeRV residual tokenization

RT-NeRV's relevant idea for Pact is not the full architecture; it is residual support tokenization. The PR95 graft should tokenize residual latent support and only later consider shallow feature tokens:

- Phase 1: tokenized latent residuals only. No decoder topology change.
- Phase 2: add a tiny `support_token_decoder` only if section value proves the extra decoder bytes beat the latent-only stack. Hard cap: 1-8 KB charged decoder overhead.
- Phase 3: shallow feature residual tokens for selected low-resolution feature planes, but only with receiver proof that the runtime consumes the tokens and exact section replay shows positive value.

This sequencing keeps the first prototype byte-closed and limits blast radius.

## C3/Cool-Chic-inspired residual atoms

The current local C3/Cool-Chic residual implementations are useful as priors, not as ready archive sections. The dense quarter-resolution or pyramid streams are too expensive and historically fail small budgets. Use them only as sparse atom generators:

```text
residual_atom =
  pair_or_gop_id
  level_id
  yx_rank_or_delta
  channel_or_basis_id
  amplitude_code
  duration_or_first_difference_mode
```

Allowed atom families:

- C3 first-difference atoms: small temporal residual corrections that propagate across a short GOP.
- Cool-Chic pyramid atoms: coordinate/pyramid residuals at coarse levels first.
- Learned basis atoms: a tiny archive-charged basis table, admitted only if amortized across enough pairs.

Forbidden in phase 1:

- dense quarter-resolution deltas;
- uncharged learned entropy model;
- external optical flow, RAFT, ego trajectory, or scorer-derived maps;
- residual packets that have no receiver-consumption mutation proof.

## Procedural and ego priors

Procedural/ego priors are allowed only if archive-charged and deterministic. The safe phase-2 candidate is a tiny temporal affine predictor:

```text
pred[i,g] = A_g * latent[i-1,g] + b_g + c_g * t_norm
```

Archive charge: `A_g`, `b_g`, `c_g`, quantized as int8 or fp16 and hashed in meta. It may save index entropy if latent motion is smooth. Anything resembling dense flow, RAFT features, vehicle-state side information, source-video-derived maps, or uncharged procedural tables is disallowed until encoded in the archive and proven by runtime mutation tests.

## Model-byte overhead and rate math

Contest rate cost is:

```text
rate_score_per_byte = 25 / 37,545,489 = 0.000000665865 score/byte
```

Approximate byte charges:

| bytes | rate score |
| ---: | ---: |
| 1,024 | 0.000682 |
| 4,096 | 0.002727 |
| 8,192 | 0.005454 |
| 16,384 | 0.010909 |
| 32,768 | 0.021818 |

Local VQ math already says K=256,D=28,N=600 is about 14,936 bytes before better entropy, which is too close to the existing PR95 latent blob to be worth much unless distortion improves. K=64,D=28,N=600 is about 4,034 bytes, but a single global 28D codebook is too coarse for scorer-sensitive rows. The stronger design is 4 product codebooks:

| latent representation | rough raw bytes | first verdict |
| --- | ---: | --- |
| PR95 scalar latent packet | about 15.4 KB from local PR101/PR103-family priors | protected baseline |
| Product VQ K=32, 4x7D | 0.9 KB codebooks + 1.5 KB indices + 0.2-1.5 KB entropy tables/escapes | best first target |
| Product VQ K=64, 4x7D | 1.8 KB codebooks + 1.8 KB indices + 0.3-2.5 KB tables/escapes | use only if nonrate holds |
| Full VQ-VAE 256x64 | about 32 KB codebook before decoder | too expensive unless replacing decoder |
| C3/Cool-Chic dense residual | unbounded/dense | reject by default |
| Sparse residual atoms | 0.5-8 KB | admit only after section value |

The first realistic win is latent rate reduction: compress 15.4 KB of scalar latent support into roughly 3-6 KB with scalar escapes, saving 9-12 KB. Pure rate score gain is 0.0060-0.0080 if nonrate is preserved. That is large enough to matter against PR95-style frontier packets but not large enough to tolerate PoseNet damage.

## Entropy coding plan

The entropy stack should be boring and runtime-safe:

1. Brotli q11 remains the outer compressor for each top-level PR95 blob.
2. Inside the latent packet, split streams by symbol family before brotli.
3. Use canonical Huffman and enumerative ranks first, matching PR101 `HUFF_ENUM` discipline.
4. Reuse or port the PR103 range/AC pattern only if the runtime dependency/LOC budget is proven and packaged in the archive/runtime tree.
5. Do not add an external entropy dependency unless the dependency closure is part of the byte/runtime proof.

Recommended stream codecs:

| stream | codec |
| --- | --- |
| `mode_map` | canonical Huffman or 2-bit raw then brotli |
| `vq_indices` | Markov-1 delta to previous pair/group, canonical Huffman/range |
| `pvq_gain_k` | Rice bucket + canonical Huffman |
| `pvq_shape_rank` | enumerative rank, optional range over rank buckets |
| `scalar_escape` | PR101 `HUFF_ENUM` or PR103 split low/high streams |
| `product_codebooks_q` | delta-coded int8 rows, brotli q11 |
| `residual_atoms` | sorted sparse tokens, delta-coded coordinates, Huffman amplitudes |

## Scorer-priced residual admission rule

A residual section or token is admitted only when measured score value beats rate cost:

```text
delta_total = delta_nonrate + delta_bytes * (25 / 37,545,489)
admit iff delta_total < 0
```

For local planning, use the PR95 residual atom derivative already encoded in `src/tac/pr95_residual_atoms.py`:

```text
pose_score = sqrt(10 * avg_posenet_dist)
d(pose_score)/d(avg_posenet_dist) = 5 / sqrt(10 * avg_posenet_dist)
seg_score = 100 * avg_segnet_dist
```

But local derivatives are only acquisition hints. Final admission needs section replay:

1. Build candidate with token/residual included.
2. Build semantic neutralization packet that removes just that section and leaves parser/runtime valid.
3. Replay full-video MLX/advisory first, then local CPU if promising.
4. Record `delta_bytes`, `delta_segnet`, `delta_posenet`, `delta_total`, archive SHA, runtime tree SHA, and neutralization SHA.
5. Promote to exact queue only after local gates pass and axis labels remain separate.

No atom gets admitted from RGB MSE alone.

## Section value-per-byte analysis

Local evidence from the recent PR95/Pact-NeRV section-value memo says:

| section | local prior | design action |
| --- | --- | --- |
| decoder weights | protected; neutralization was strongly harmful despite rate savings | do not replace in phase 1 |
| base latents | protected; neutralization was harmful | replace only with exact latent reconstruction or measured token distortion |
| selectors | near-zero value in the inspected PACT-NeRV artifact | encode/cut aggressively; never treat as authority |
| residual section | absent or demoted in current packets | add only with measured positive value |
| HPRC residual grids | rate-efficient but PoseNet-catastrophic in recent runs | do not import dense HPRC residuals |
| VQ codebooks | promising only when codebook+indices beat scalar latent bytes | product VQ plus scalar escape |
| PVQ ranks/gains | promising for high-energy directional residuals | use as escape, not universal mode |
| C3/Cool-Chic atoms | useful atom generators but not ready dense codecs | sparse scorer-priced atoms only |

The first prototype's section-value target:

```text
decoder_delta_total ~= 0
latent_packet_delta_total <= -0.003 after exact reconstruction escapes
residual_atoms_delta_total <= -0.001 if present
total local advisory delta <= -0.004 before exact dispatch consideration
```

These are design thresholds, not score claims.

## MLX training objective

Start from a trained PR95 checkpoint and train the tokenization layer around it. Do not train a new receiver first.

Recommended objective:

```text
J =
  score_surrogate(decoded_frames, target_frames)
  + lambda_rgb * yuv6_or_rgb_reconstruction
  + lambda_rate * R_hat(mode_map, indices, pvq, escapes, atoms)
  + lambda_vq * (||stopgrad(z) - e||^2 + beta * ||z - stopgrad(e)||^2)
  + lambda_pvq * H_hat(gain, pulse_count, rank)
  + lambda_escape * escape_byte_penalty
  + lambda_parity * receiver_roundtrip_penalty
```

`score_surrogate` must prioritize SegNet/PoseNet semantics. If full scorer loss is not wired in MLX, use Hinton-style distillation or cached scorer-response targets only as local `[macOS-MLX research-signal]` triage and keep exact blockers live. RGB/YUV6 MSE can stabilize training but cannot authorize promotion.

Training stages:

1. Freeze PR95 decoder and source latents. Fit product VQ/PVQ packet to reconstruct latents exactly enough to keep decoded frames stable.
2. Enable scorer-weighted latent metric from VJP/Jacobian anchors; raise scalar escape penalty gradually.
3. Fine-tune codebooks and optional latent predictor with straight-through indices.
4. Admit sparse residual atoms through greedy/beam section-value waterfill.
5. Export `latents.npy` or a `.pt` bundle that contains both decoder and trained latent/token state.
6. Package into PR95-style `0.bin`; prove runtime consumption and full-frame inflate.

## Receiver proof design

Required receiver proofs before exact dispatch:

- Parser proof: `0.bin` has exactly the PR95-style three top-level blobs and no trailing bytes.
- No-op proof: a `PVQRT1` no-op packet reconstructs the original PR95 latents and produces identical decoded output under the modified runtime.
- Mutation proof: flipping one VQ index, PVQ gain, PVQ rank, scalar escape, and residual atom each changes the inflated output and is reflected in a proof hash.
- Section-neutralization proof: every optional section can be neutralized without breaking parser/runtime; replay reports value per byte.
- Full-frame proof: `inflate.sh archive_dir output_dir file_list` writes the expected full raw output, with aggregate SHA and byte count recorded. Prior PR95 packaged runtime proof used a 3,662,409,600-byte full-output boundary; match that style.
- Runtime custody: runtime tree SHA-256, dependency list, archive SHA-256, member SHA-256, command, env, and cleanup/cold-store manifest.
- Axis separation: local MLX, macOS CPU, contest CPU, and contest CUDA remain separate evidence spaces.

## Exact-gate blockers

These blockers must remain fail-closed until real artifacts clear them:

- `pvqrt1_runtime_not_implemented`: no parser/decoder exists yet.
- `pvqrt1_noop_roundtrip_missing`: no no-op parity proof.
- `pvqrt1_mutation_consumption_missing`: no token mutation proof.
- `full_frame_inflate_output_parity_missing`: no full-frame runtime proof.
- `mlx_scorer_loss_unwired`: current PR95 MLX surfaces are RGB/YUV6 timing/advisory, not full scorer authority.
- `mlx_to_pytorch_export_token_bridge_missing`: token packet state is not exported into runtime/package format.
- `section_value_full_video_missing`: no semantic neutralization replay for each section.
- `p18_p19_surfaces_not_recomputed`: Z8/HPRC protection surfaces cannot be reused blindly.
- `dependency_closure_unproven`: range/ANS coder or token runtime dependency not packaged/proven.
- `storage_cleanup_manifest_missing`: large training/replay artifacts require certified SSD/cold-store hygiene.
- `dispatch_claim_missing`: any training/eval/GPU job must claim the lane first.
- `exact_cpu_cuda_pair_missing`: no contest-axis promotion until exact CPU/CUDA evidence exists.

## Prototype commands

Storage preflight first:

```bash
uv run python tools/plan_experiment_storage.py \
  --output .omx/research/pr95_pvqrt1_storage_plan_20260601T122758Z.json \
  --storage-tier vertigo=/Volumes/VertigoDataTier/pact \
  --storage-tier apstore=/Volumes/APDataStore/pact \
  --workload-subdir pr95_pvqrt1_20260601T122758Z \
  --requested-bytes 120000000000 \
  --min-free-bytes 160000000000 \
  --create
```

Baseline PR95 local timing/export smoke using existing tooling:

```bash
uv run python tools/run_pr95_mlx_long_training.py \
  --output-report /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/pr95_smoke_report.json \
  --checkpoint-root /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/checkpoints \
  --telemetry-path /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/telemetry.jsonl \
  --max-frames 64 \
  --training-loss-surface rgb_yuv6_mse \
  --smoke-mode \
  --smoke-epochs-per-stage 2 \
  --execute-smoke
```

Proposed new packet builder, not yet implemented:

```bash
uv run python tools/build_pr95_pvqrt1_latent_packet.py \
  --source-archive-zip experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip \
  --trained-latents-npy /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/latents.npy \
  --grouping fixed4x7 \
  --vq-k 32 \
  --enable-pvq-escape \
  --escape-policy scorer_weighted \
  --output-packet /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/pvqrt1_latents.bin \
  --report-out /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/pvqrt1_packet_report.json
```

Package through existing PR95 export path once the bridge supports token packets:

```bash
uv run python tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py \
  --input-pt /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/export/pr95_decoder.pt \
  --source-archive-zip experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip \
  --latents-npy /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/latents.npy \
  --output-submission-dir /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission \
  --report-out /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/package_report.json
```

Proposed receiver-consumption proof:

```bash
uv run python tools/prove_pr95_pvqrt1_receiver_consumption.py \
  --submission-dir /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission \
  --archive-zip /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission/archive.zip \
  --output-json /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/receiver_consumption_proof.json \
  --mutation-suite all \
  --allow-large-output
```

Existing full PR95 runtime consumption proof style:

```bash
uv run python tools/prove_pr95_public_archive_runtime_consumption.py \
  --archive-zip /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission/archive.zip \
  --inflate-sh /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission/inflate.sh \
  --output-json /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/runtime_consumption_proof.json \
  --work-dir /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/runtime_proof \
  --allow-large-output
```

Proposed section-value profiler:

```bash
uv run python tools/profile_pr95_pvqrt1_section_value.py \
  --submission-dir /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission \
  --archive-zip /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission/archive.zip \
  --sections vq_indices,pvq_shape_rank,scalar_escape,residual_atoms \
  --neutralized-output-root /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/section_value \
  --summary-json /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/section_value_summary.json
```

Local replay and exact gate after receiver and section-value proofs:

```bash
uv run python tools/run_local_submission_replay.py \
  --runtime-submission-dir /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission \
  --archive-zip /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/submission/archive.zip \
  --output-dir /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/local_replay_cpu \
  --device cpu \
  --summary-json /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/local_replay_cpu/summary.json

uv run python tools/gate_local_candidate_for_exact_auth.py \
  --local-replay-summary-json /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/local_replay_cpu/summary.json \
  --exact-auth-axis "[contest-CPU]" \
  --expected-local-axis-tag "[macOS-CPU advisory]" \
  --out-json /Volumes/VertigoDataTier/pact/pr95_pvqrt1_20260601T122758Z/exact_gate.json
```

Any remote exact dispatch must then use `tools/claim_lane_dispatch.py` first and keep CPU/CUDA axes separate.

## Full-run campaign command shape

After the prototype proves no-op parity and token consumption, the full run should be a managed campaign, not a one-off:

```bash
uv run python tools/build_pr95_pvqrt1_training_queue.py \
  --output /Volumes/VertigoDataTier/pact/pr95_pvqrt1_full_20260601T122758Z/queue.json \
  --plan-output /Volumes/VertigoDataTier/pact/pr95_pvqrt1_full_20260601T122758Z/plan.json \
  --run-id pr95_pvqrt1_full_20260601T122758Z \
  --source-archive-zip experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip \
  --checkpoint-root /Volumes/VertigoDataTier/pact/pr95_pvqrt1_full_20260601T122758Z/checkpoints \
  --curriculum-total-epochs 3000 \
  --loss scorer_distilled_yuv6_rate \
  --vq-k 32 \
  --pvq-escape-grid 0,2,4,6,8 \
  --section-value-after-each-stage \
  --storage-tier /Volumes/VertigoDataTier/pact \
  --allow-overwrite
```

The command is intentionally proposed, because the current repo has PR95 MLX training and HPRC queue builders but not this exact PVQRT1 packet trainer.

## Likely theoretical floor

This is the honest range, assuming the current PR95-style CPU frontier remains in the same neighborhood and must be re-derived before execution:

- Pure latent-packet rate win: save 9-12 KB from the roughly 15.4 KB scalar latent support while preserving nonrate. Score gain: about 0.006-0.008.
- Residual token/atom win: 0.002-0.010 possible if sparse atoms are admitted by section value; 0 if PoseNet sensitivity rejects them.
- Decoder replacement win: deferred. Replacing PR95 decoder bytes is high upside but high risk; current section-value priors say decoder removal is expensive in nonrate.

Therefore the realistic first floor is around current PR95 exact-axis score minus 0.004 to 0.008, after exact gates. A stretch floor of minus 0.010 to 0.015 requires sparse residual atoms that improve PoseNet/SegNet more than their 2-12 KB byte charge. A sub-0.175-style result would probably require a true C3/Cool-Chic/RT-NeRV receiver replacement or shallow feature token decoder, and that should not be claimed until byte-closed runtime and scorer evidence exists.

## Strongest next implementation slice

Build only the no-op and lossless-ish latent-token path first:

1. Implement `PVQRT1` packet parser/writer inside the PR95-style top-level envelope.
2. Add no-op packet mode that reconstructs source PR95 latents exactly.
3. Add product VQ K=32 with scalar escape and a no-op escape fallback.
4. Add receiver mutation proof and section neutralization proof.
5. Run local replay. Only then train scorer-weighted codebooks.

The decisive point: this stack should compete by shrinking or reorganizing PR95 latent support while preserving the proven PR95 receiver, not by immediately importing a dense learned residual codec. The codebook is a byte-saving layer first; residual atoms are admitted only after the scorer prices them.
