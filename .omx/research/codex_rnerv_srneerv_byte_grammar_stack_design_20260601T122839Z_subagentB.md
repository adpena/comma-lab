# RNeRV/SRNeRV PR95 Byte-Grammar Stack Design - Subagent B

UTC: 2026-06-01T12:28:39Z

Author: Codex implementation-design subagent B

Status: design memo only; `research_only=true` until the code landing exists.

Write boundary: this memo is the only write from this subagent. No code, state,
lane registry, queue, checkpoint, dispatch claim, archive, or result file was
modified.

Authority boundary: all MLX/local outputs proposed here are
`[macOS-MLX research-signal]` or local timing evidence only. Any score, rank,
promotion, or kill decision remains blocked until a byte-closed archive/runtime
packet passes the matching `contest-CPU` or `contest-CUDA` auth-eval path.

## Preflight And Local Inputs

Read first per operator contract:

- `AGENTS.md`
- `CLAUDE.md`

Current local surfaces inspected:

- PR95/HNeRV package grammar and runtime:
  - `submissions/a1/src/codec.py`
  - `submissions/a1/inflate.py`
  - `src/tac/pr101_split_brotli_codec.py`
  - `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
  - `tools/run_pr95_mlx_long_training.py`
  - `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
- PACT-NeRV/RNeRV-like renderer surfaces:
  - `src/tac/substrates/pact_nerv_selector_v3/architecture.py`
  - `src/tac/substrates/pact_nerv_selector_v3/archive.py`
  - `src/tac/substrates/pact_nerv_selector_v3/mlx_renderer.py`
  - `src/tac/substrates/pact_nerv_selector_v3/section_value.py`
  - `src/tac/substrates/pact_nerv_selector_v4/architecture.py`
  - `src/tac/substrates/pact_nerv_mamba/architecture.py`
  - `src/tac/substrates/boost_nerv/architecture.py`
  - `src/tac/substrates/boost_nerv/archive.py`
  - `src/tac/substrates/boost_nerv_pr110_residual/architecture.py`
- Compact-base runner/spine:
  - `src/tac/substrates/hprc/archive.py`
  - `src/tac/substrates/hprc/pr95_adapter.py`
  - `src/tac/substrates/hprc/representation_spine.py`
  - `src/tac/substrates/hprc/spine_bounded_runner.py`
  - `src/tac/substrates/_shared/mlx_score_aware/loss.py`
- Latest compact renderer memos:
  - `.omx/research/compact_hnerv_family_base_renderer_stack_design_20260601T115829Z_codex.md`
  - `.omx/research/subagent_b_latent_codebook_implicit_stack_design_20260601T120216Z_codex.md`
  - `.omx/research/codex_fresh_eyes_compact_learned_video_stack_A_20260601T121341Z.md`
  - `.omx/research/codex_findings_pr95_compact_carrier_controls_and_latent_guard_20260601T122243Z_codex.md`
  - `.omx/research/codex_findings_pact_nerv_int8_raw_section_value_20260601T114712Z_codex.md`
  - `.omx/research/codex_findings_pact_nerv_spine_emitters_20260601T045127Z_codex.md`
  - `.omx/research/codex_findings_hprc_rate_collapse_compact_substrate_spine_20260601T042500Z_codex.md`
  - `.omx/research/rnerv_pact_z8_residual_lane_design_20260531T222520Z_codex.md`
  - `.omx/research/codex_findings_rnerv_pact_z8_residual_lane_research_20260531T223036Z_subagent.md`

Online sources used:

- NeRV: [arXiv:2110.13903](https://arxiv.org/abs/2110.13903)
- HNeRV: [arXiv:2304.02633](https://arxiv.org/abs/2304.02633)
- HiNeRV: [arXiv:2306.09818](https://arxiv.org/abs/2306.09818)
- BoostNeRV: [arXiv:2402.18152](https://arxiv.org/abs/2402.18152)
- RT-NeRV residual tokenization: [arXiv:2403.12401](https://arxiv.org/abs/2403.12401)
- RNeRV/VINRB design study: [arXiv:2506.24127](https://arxiv.org/abs/2506.24127), [code](https://github.com/mgwillia/vinrb)
- SRNeRV scale-wise recursion: [arXiv:2603.08227](https://arxiv.org/abs/2603.08227)

## Bottom Line

Build a new `rnerv_srneerv_pr95_grammar` lane as a PR95/HNeRV-fair renderer,
not as another standalone PACT-NeRV variant. The lane should use the same
contest-facing shape as PR95: one monolithic ZIP member, one deterministic
inflate runtime, full RGB frame production, and section bytes that can be
projected into the HPRC spine.

The highest-EV stack is:

1. PR95-faithful HNeRV decoder and byte codec as the control arm.
2. RNeRV-style training/design choices as the architecture-search prior.
3. SRNeRV-style scale-recursive parameter sharing to reduce decoder bytes.
4. RT/BoostNeRV residual/token machinery only after section-value pricing says
   the extra bytes reduce `100*d_seg + sqrt(10*d_pose) + lambda_B*bytes`.

Do not optimize for MSE alone. RGB/YUV MSE is useful for warm start and sanity,
but the lane lives or dies by contest score per byte and by byte-closed
receiver proof.

## Why This Lane Is Different From Existing Local PACT-NeRV

Existing local substrates establish pieces but not the final comparison:

- `pact_nerv_selector_v3` and `selector_v4` are HNeRV-like depth-separable
  PixelShuffle decoders with pair latents and optional selector streams. Recent
  section pricing found decoder and latents protected, while selectors were
  effectively dead unless they drive pixels.
- `boost_nerv` has a useful residual-refinement pattern, but its current BSV1
  archive ships base and boosting heads together, uses raw int16 latents, and
  is not PR95 grammar-compatible enough for an apples-to-apples PR95/HNeRV
  comparison.
- `pact_nerv_mamba` is a recurrence scaffold over latent sequences. It is useful
  as a recurrence sanity check, but the L0 proxy is not yet a byte-winning
  contest runtime.
- HPRC already gives the right comparison spine: `DECODER_QW`, `LATENTS_RC`,
  `CODEBOOKS_Q`, `SELECTORS_RC`, `RESIDUAL_RC`, `RDO_PLAN`, `RECEIVER_STATE`,
  and `MANIFEST_JSON`.
- PR95/A1 gives the byte standard to match: compact decoder stream, compact
  latent stream, deterministic sidecar, tiny metadata, full-frame inflate.

Therefore the new lane should be a thin, PR95-shaped full-stack packet with
swappable RNeRV/SRNeRV architecture internals, not another isolated archive
grammar.

## Fair-Comparison Contract

A candidate can compare against PR95/HNeRV only when all of these are true:

- Same contest source video and 1200-frame output.
- Same paired-frame convention: 600 pair inputs, two RGB heads or an equivalent
  deterministic pair renderer.
- Same archive authority surface: `archive.zip` bytes plus runtime tree bytes
  counted under the same contest packaging rules.
- Same inflate contract:
  `inflate.sh archive_dir output_dir file_list`.
- Same output shape and order:
  1200 RGB frames, `1164 x 874`, aggregate raw byte count
  `3,662,409,600`.
- Same axis label: never compare `[macOS-MLX research-signal]` with
  `[contest-CUDA]`, `[contest-CPU]`, or public CPU numbers.
- Same HPRC section projection, so section prices and byte deltas are visible.
- Same exact-eval gate before promotion, retirement, or rank language.

## Architecture Variants

### Variant A: RNeRV-Lite PR95 Control

Purpose: lowest-risk bridge from PR95/HNeRV to RNeRV design ideas.

Structure:

- Keep PR95/HNeRV paired latent table initially: `600 x 28`.
- Keep the PR95-ish decoder geometry initially: latent embed to `3 x 4`,
  repeated upsample blocks, two RGB heads.
- Search RNeRV/VINRB-inspired training choices and architecture knobs under a
  fixed byte budget: activation, channel schedule, stem size, block order,
  quantization schedule, pruning schedule, QAT, and optimizer/curriculum.
- Materialize latents in the archive for the first pass. Do not put a recurrent
  latent generator into inflate until it beats materialized latents after bytes.

Byte goal:

- <= PR95/A1 bytes for first fair win attempt.
- Decoder must shrink enough to pay for any extra receiver state or policy
  fields.

When to continue:

- Full-video MLX advisory nonrate improves relative to PR95-local control at
  equal or lower packaged bytes.
- Receiver proof passes with the exact packaged bytes.

When to stop:

- It only improves RGB/YUV MSE, or only improves partial frames, or requires an
  uncharged runtime helper.

### Variant B: SRNeRV Shared-Scale Decoder

Purpose: spend SRNeRV's key insight on the PACT byte frontier: avoid paying
separate weights at every upsample scale when most parameters can be shared.

Structure:

- Split each upsample block into:
  - scale-specific spatial mixer: tiny depthwise `3x3` or `5x5`, per scale;
  - scale-invariant channel mixer: shared pointwise/MLP core reused across
    scales;
  - scale-specific adapters: small affine/gain/bias or low-rank correction.
- Keep PR95's paired output contract. The renderer still emits frame `2i` and
  `2i+1`.
- Start with one shared channel core for the seven scale steps. If this
  underfits high resolution, use two shared cores: early low-res and late
  high-res.
- Store per-scale adapters in `DECODER_QW`; store the policy in
  `RECEIVER_STATE`.

SRNeRV's online paper argues that stacked multi-scale INR blocks contain
parameter redundancy, and that decoupling scale-specific spatial mixing from
shared channel mixing reduces model size while preserving scale-specific
patterns. That maps cleanly to PR95 because decoder bytes dominate the current
packet.

Byte goal:

- Save at least 12-25 KB from decoder bytes before admitting any new latent,
  selector, or residual section.
- If decoder shrink is <8 KB after real packing, this variant should fall back
  to Variant A.

### Variant C: RNeRV Latent Recurrence With Epsilon Stream

Purpose: replace some or all of the materialized latent table with a tiny
recurrent generator when it wins after all receiver and epsilon bytes.

Structure:

```
h_i = f_theta(h_{i-1}, t_i)
z_hat_i = g_theta(h_i)
z_i = z_hat_i + dequant(epsilon_i)
frames_i = decoder(z_i)
```

Start as a training-only prior:

- Train recurrence to predict the learned latent table.
- Export both materialized latents and recurrent prediction diagnostics.
- Only switch archive mode when:
  `bytes(recurrence_weights + epsilon_stream + receiver_state) <
   bytes(materialized_latents)` and score is not worse.

The recurrence should be small enough to live inside `DECODER_QW` or
`RECEIVER_STATE`, not a hidden source-side object. It must decode determinis-
tically in the shipped runtime.

Recommended first knobs:

- `latent_dim=28` for PR95 parity.
- hidden width 16 or 24, one gated recurrent/update block.
- epsilon quantization: per-dim signed int4/int6/int8 deltas, temporally coded.
- fallback keyframes: every 32 or 64 pairs if drift accumulates.

Do not use this path as score authority until recurrence output and materialized
latent output are compared through full-frame inflate.

### Variant D: Boost/RT Residual Token Branch

Purpose: import BoostNeRV and RT-NeRV only where their residual support is
byte-positive.

Structure:

- Base renderer is Variant A or B.
- Add a small residual branch that consumes shallow features or rendered RGB
  and emits a bounded correction.
- Instead of continuous residual features, tokenize residual support into:
  - `CODEBOOKS_Q`: small codebook(s) for feature or pixel residual atoms;
  - `SELECTORS_RC`: entropy-coded token indices;
  - `RESIDUAL_RC`: optional sparse signs/magnitudes or pair-local deltas.
- Apply residuals only at scorer-sensitive regions or pairs if the full-frame
  branch is too expensive.

Admission rule:

`delta_nonrate + lambda_B * section_bytes < 0`

where `delta_nonrate` is measured on the same local replay surface and
eventually exact auth-eval, not inferred from RGB MSE. If the branch does not
beat this inequality, its bytes stay out of the candidate.

BoostNeRV's conditional decoder and temporal-aware affine idea is useful as a
low-byte feature-alignment layer. RT-NeRV's residual tokenization is more
directly aligned with PACT because it can make residual support discrete and
accountable. Neither branch should be allowed to become an unpriced dense
sidecar.

## Learned Upsample Policy

Current PACT-NeRV variants hardcode depth-separable convolution, sinusoidal
activation, and `PixelShuffle(2)` at each scale. The first RNeRV/SRNeRV lane
should make the upsample operator a byte-priced policy:

Per-scale candidate operators:

- `ps`: depthwise+pointwise -> sin -> PixelShuffle(2), PR95-compatible control.
- `bilinear_dw`: bilinear resize -> depthwise spatial mixer -> shared channel
  mixer.
- `nearest_lite`: nearest resize -> low-rank/channel mixer, only if artifacts
  are acceptable.
- `subpixel_residual`: PixelShuffle base plus a small scale adapter.

Policy encoding:

- L1: one static u2/u3 operator id per scale, <= 7 bytes before compression.
- L2: per-scale adapter rank and gain, stored in `RECEIVER_STATE`.
- L3 only if L1/L2 prove value: pair-bucket policy selectors in
  `SELECTORS_RC`.

Forbidden first-pass policy:

- per-pixel or dense per-feature policy masks. These are almost certainly
  byte-negative unless tokenized and proven through section-value pricing.

Training policy:

- During search, use Gumbel/soft selection or straight-through gates.
- At export, harden to deterministic operator ids and delete inactive branches.
- Archive must contain only active branch weights and the hardened policy.

## Base/Residual Split

The base renderer owns global structure and most segmentation/pose geometry.
The residual branch owns only measured scorer-sensitive gaps.

Base section:

- `DECODER_QW`: decoder, optional recurrence generator, SR shared core,
  scale adapters, RGB heads.
- `LATENTS_RC`: materialized pair latents or epsilon/keyframe stream.
- `RECEIVER_STATE`: minimal config needed for deterministic decode.

Residual sections:

- `CODEBOOKS_Q`: atom/codebook payloads.
- `SELECTORS_RC`: pair, scale, region, or token selectors.
- `RESIDUAL_RC`: sparse signs, magnitudes, or encoded residual coefficients.

Base-first rule:

- Ship no residual section in the first fair RNeRV/SRNeRV candidate unless the
  base candidate has a receiver proof and section-value report.
- A residual byte is admitted only when measured score gain beats its rate
  cost under the contest formula.

Z8 lesson:

- Full wavelet residual blobs are too large for this contest budget. If Z8 or
  wavelet material reappears here, it must be a tiny scorer-selected token
  branch, not a dense sidecar.

## Latents, Selectors, And Codebooks

Latent modes:

- `table_pr95`: PR95-style `600 x 28` learned pair latents. Control mode.
- `table_shrunk`: `600 x D`, D in `{16, 20, 24, 28, 32}`, priced after packing.
- `recurrence_epsilon`: recurrent prediction plus coded epsilon stream.
- `keyframe_delta`: periodic absolute latent keyframes plus temporal deltas.

Latent packing:

- Preserve PR95/A1 latent codec as control: temporal-delta uint8 with per-dim
  fp16 min/scale and dimension ordering where it remains byte-winning.
- Also test raw LZMA, Brotli q11, and range/ANS coding on the actual post-QAT
  latent stream.
- Do not infer a coder win from entropy estimates alone; pack actual bytes.

Selectors:

- Existing selector streams are not protected unless they drive pixels.
- Use selectors only for:
  - hardened upsample policy buckets;
  - residual token ids;
  - pair/frame region flags;
  - codebook bank selection.
- For sequential selectors, test RLE, canonical Huffman, range/ANS, and simple
  finite-state delta coding. Keep the smallest real payload, not the prettiest
  coder.

Codebooks:

- Start with 8, 16, or 32 atoms.
- Quantize codebook values independently from selectors.
- Admit only if `codebook_bytes + selector_bytes + residual_bytes` is cheaper
  than the equivalent continuous correction and score-positive.

## Exact Byte Accounting

Contest rate term:

```
N = 37,545,489
lambda_B = 25 / N = 6.6585909503e-7 score per byte
score = 100*d_seg + sqrt(10*d_pose) + lambda_B*archive_bytes
```

Useful byte ceilings at zero distortion:

| bytes | rate term |
| ---: | ---: |
| 75,090 | 0.0500 |
| 100,000 | 0.0666 |
| 120,145 | 0.0800 |
| 150,181 | 0.1000 |
| 180,218 | 0.1200 |
| 225,272 | 0.1500 |
| 285,345 | 0.1900 |

PR95/A1 comparison anchor:

- Public compact PR95-family packet is roughly 178 KB.
- A1 codec constants expose a decoder blob around 162 KB and latent blob around
  15 KB, with sidecar/metadata tiny relative to decoder.

First RNeRV/SRNeRV target budget:

| section | target |
| --- | ---: |
| `DECODER_QW` | 110-145 KB |
| `LATENTS_RC` | 8-22 KB |
| `CODEBOOKS_Q` | 0-8 KB |
| `SELECTORS_RC` | 0-1 KB until pixel-driving |
| `RESIDUAL_RC` | 0-16 KB, only if score-positive |
| `RDO_PLAN` | 0 bytes in contest packet unless needed for proof only |
| `RECEIVER_STATE` + metadata | <=1.5 KB |
| ZIP/container overhead | measured, not estimated |

Byte ledger required for every export:

- archive path, bytes, SHA-256;
- runtime tree hash and file manifest;
- ZIP member order, methods, CRCs, timestamps, flags;
- section bytes, SHA-256, compression method, codec params;
- decoder parameter count and packed-byte count by tensor group;
- latent dimensions, quantization metadata, coder bytes;
- codebook/selector/residual bytes and section-value verdict;
- false-authority flags and exact-axis blockers.

## PR95-Style Byte Grammar

The contest-facing packet should be a single ZIP member, preferably `0.bin`.
The archive grammar should remain PR95-shaped but extended by mode, not a
totally separate BSV/PSV/HPRC packet.

Recommended RNRS1 `0.bin` payload:

```
MAGIC              4 bytes   b"RNRS"
VERSION            u8        1
MODE               u8        0=table, 1=srshared, 2=recurrence, 3=residual-token
FLAGS              u16       feature bits, fail closed on unknown
META_LEN           u32       brotli-json receiver state
DECODER_LEN        u32       packed decoder / shared-core / heads
LATENT_LEN         u32       table latents or recurrence epsilon stream
CODEBOOK_LEN       u32       optional residual/codebook atoms
SELECTOR_LEN       u32       optional policy/token selectors
RESIDUAL_LEN       u32       optional signs/magnitudes/corrections
META_BROTLI        ...
DECODER_QW         ...
LATENTS_RC         ...
CODEBOOKS_Q        ...
SELECTORS_RC       ...
RESIDUAL_RC        ...
```

Why not use HPRC directly as the contest packet?

- HPRC is the correct internal comparison spine, but PR95 fairness is better
  served by a PR95-like monolithic contest packet with tiny receiver metadata.
  Emit an HPRC projection as an artifact and proof surface, not as the only
  contest runtime.

Compatibility rules:

- First three logical sections map to the PR95 public archive roles:
  receiver/meta, decoder, latents.
- Extra sections are length-prefixed and omitted when zero length.
- Unknown modes or flags fail closed in parser and inflate.
- `RDO_PLAN` is not shipped unless the runtime actually consumes it; it belongs
  in proof artifacts by default.

Packing choices:

- Decoder:
  - PR95 split Brotli q11 control.
  - per-tensor byte maps and storage permutations where measured positive.
  - fp16/int8/heterogeneous per-tensor QAT export.
  - range/ANS only after actual bytes beat split Brotli, including decoder
    complexity and runtime closure.
- Latents:
  - PR95 temporal-delta uint8/raw LZMA control.
  - recurrence epsilon streams tested under int4/int6/int8 plus entropy coding.
  - per-dim scales stored as fp16 unless exact proof requires fp32.
- Codebooks/selectors/residuals:
  - canonical Huffman, RLE, range/ANS, and raw LZMA tested on actual streams.
  - no Python package dependency in inflate unless already in runtime closure.

## MLX Long-Training Plan

Storage and cleanup:

- Use the SSD waterfall before any full run:
  `/Volumes/VertigoDataTier/pact`, then `/Volumes/APDataStore/pact`, then local
  only by explicit opt-in.
- Results root:
  `/Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run_id>`.
- Every long run must write a manifest with original paths, bytes, SHA-256,
  argv/env, source archive/runtime hashes, and cleanup/cold-store policy.
- Large scratch outputs should be context-managed or certified/moved before
  local deletion.

Training stages:

1. `timing_smoke`: 8 pairs, 32 pairs, and full-shape microbatch timing. Output
   seconds/epoch and projected GPU-hours.
2. `pr95_control`: reproduce PR95/HNeRV local advisory under the current
   MLX/PyTorch bridge and archive packer.
3. `rnerv_search`: RNeRV/VINRB-inspired channel schedules, activation choices,
   QAT schedules, latent dims, optimizer/curriculum.
4. `srshared_search`: shared channel mixer across scales; scale-specific
   spatial mixers/adapters; learned upsample policy hardened at export.
5. `rate_loop`: archive-in-loop byte surrogate and periodic real pack/export.
6. `score_loop`: real or distilled SegNet/PoseNet scorer-aware objective.
7. `residual_token_admission`: only after base receiver proof.
8. `export_proof`: MLX to PyTorch/numpy bridge, packed archive, runtime
   consumption, full-frame output manifest, HPRC projection, exact-gate report.

Optimizer/curriculum:

- Warm start with RGB/YUV reconstruction only to get stable frames.
- Switch to score-aware weighting early; do not spend the long run on MSE-only
  if bytes are not improving.
- Use QAT before final export, not as a post-hoc surprise.
- Keep Muon/AdamW choice as a measured RNeRV search knob under equal wall-clock,
  matching the RNeRV paper's emphasis on comparing design choices under equal
  training time.

## Full-Video Scorer-Aware Objective

Training objective should approximate:

```
J = 100 * d_seg_proxy
  + sqrt(10 * d_pose_proxy)
  + lambda_B * estimated_archive_bytes
  + alpha * rgb_yuv_aux
  + beta * quantization_commitment
  + gamma * temporal_latent_smoothness
```

Rules:

- `rgb_yuv_aux` is an auxiliary, not the target.
- Segmentation/pose proxies must be calibrated against real full-sample
  auth-axis payloads before they route paid spend.
- Pair sampling must cover all 600 pairs over an epoch window. No cherry-picked
  pair score language.
- Section-value training batches should log estimated and actual packed bytes.
- Residual/codebook tokens should be selected by contest-score delta, not by
  PSNR or MSE salience alone.

Concrete loss implementation target:

- Reuse `src/tac/substrates/_shared/mlx_score_aware/loss.py` rather than
  inventing a second scorer-aware path.
- Add a `rate_estimator` callable that reads the current quantized state and
  predicts section bytes.
- Add `full_video_epoch_evaluator` that periodically renders every pair,
  packs a candidate, and records advisory score plus bytes.

## Receiver Adapter Design

Contest runtime:

- `submissions/rnerv_srneerv_pr95_grammar/inflate.sh`
- `submissions/rnerv_srneerv_pr95_grammar/inflate.py`
- `submissions/rnerv_srneerv_pr95_grammar/src/codec.py`
- `submissions/rnerv_srneerv_pr95_grammar/src/model.py`

Runtime constraints:

- No MLX dependency.
- No scorer import.
- Prefer numpy/PIL/torch only if already acceptable for the PR95-like runtime.
- Deterministic CPU path; CUDA optional only if runtime contract supports it.
- Fail closed on malformed section lengths, unknown flags, mismatched latent
  count/dim, output shape mismatch, or raw byte-count mismatch.

Receiver flow:

1. Locate single archive member.
2. Parse RNRS1 sections.
3. Decode receiver metadata.
4. Decode decoder/shared-scale weights.
5. Decode latents or recurrence epsilon stream.
6. Render 384x512 paired frames.
7. Apply optional residual tokens if present.
8. Resize/format exactly as contest runtime expects.
9. Write files requested by `file_list`.
10. Emit a compact machine-readable receiver proof outside the contest packet.

Adapter proof:

- `tools/prove_rnerv_srneerv_archive_runtime_consumption.py` must verify:
  - archive SHA and bytes;
  - section table and hashes;
  - runtime tree SHA;
  - all requested files written;
  - aggregate raw output bytes;
  - optional raw-output aggregate SHA;
  - HPRC projection artifact path;
  - exact-axis blockers.

## Commands

These are the intended operator commands after the code landing exists. They
are commands to implement, not commands already run by this memo.

Timing smoke:

```bash
python tools/claim_lane_dispatch.py claim \
  --lane-id rnerv-srneerv-pr95-grammar \
  --instance rnerv_srneerv_timing_smoke_20260601 \
  --status active_timing_smoke \
  --notes "local MLX timing smoke; no score authority"

python experiments/train_substrate_rnerv_srneerv_mlx_local.py \
  --run-id rnerv_srneerv_timing_smoke_20260601T122839Z \
  --artifact-root /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar \
  --variant srshared_v0 \
  --pairs 8 \
  --epochs 16 \
  --objective rgb_yuv_aux \
  --execute-smoke
```

Full local MLX run:

```bash
python experiments/train_substrate_rnerv_srneerv_mlx_local.py \
  --run-id rnerv_srneerv_srshared_scoreaware_full_20260601T122839Z \
  --artifact-root /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar \
  --variant srshared_v0 \
  --pairs 600 \
  --epochs 3000 \
  --objective contest_score_aware \
  --rate-estimator rnrs1_actual_pack_periodic \
  --quant-aware \
  --score-teacher real_or_distilled_full_video \
  --checkpoint-every 400 \
  --execute
```

Export bridge:

```bash
python tools/export_rnerv_srneerv_mlx_to_runtime_state.py \
  --input /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/checkpoint_best.npz \
  --variant srshared_v0 \
  --output /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/runtime_state.npz \
  --parity-report /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/export_parity.json \
  --fail-closed-on-drift
```

Pack archive:

```bash
python tools/package_rnerv_srneerv_pr95_grammar_archive.py \
  --runtime-state /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/runtime_state.npz \
  --mode srshared \
  --decoder-packing split_brotli_q11 \
  --latent-packing pr95_temporal_lzma \
  --output /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/archive.zip \
  --byte-ledger /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/byte_ledger.json
```

Receiver proof:

```bash
python tools/prove_rnerv_srneerv_archive_runtime_consumption.py \
  --archive /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/archive.zip \
  --runtime submissions/rnerv_srneerv_pr95_grammar \
  --file-list data/file_list.txt \
  --output-root /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/receiver_proof \
  --expect-raw-bytes 3662409600
```

HPRC projection and section value:

```bash
python tools/project_rnerv_srneerv_archive_to_hprc_spine.py \
  --archive /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/archive.zip \
  --output /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/hprc_projection.json

python tools/profile_rnerv_srneerv_section_value.py \
  --archive /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/archive.zip \
  --runtime submissions/rnerv_srneerv_pr95_grammar \
  --sections DECODER_QW,LATENTS_RC,CODEBOOKS_Q,SELECTORS_RC,RESIDUAL_RC \
  --output /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/section_value.json
```

Exact gate after local proof:

```bash
python tools/claim_lane_dispatch.py claim \
  --lane-id rnerv-srneerv-pr95-grammar \
  --instance rnerv_srneerv_exact_cuda_candidate_<sha> \
  --status active_exact_eval_candidate \
  --notes "byte-closed archive/runtime proved locally; exact CUDA requested"

python experiments/modal_auth_eval.py \
  --lane-id rnerv-srneerv-pr95-grammar \
  --archive /Volumes/VertigoDataTier/pact/experiments/rnerv_srneerv_pr95_grammar/<run>/archive.zip \
  --runtime submissions/rnerv_srneerv_pr95_grammar \
  --target-mode contest_exact_eval \
  --plan-only
```

## Concrete Code Changes

New substrate package:

- `src/tac/substrates/rnerv_srneerv_pr95_grammar/__init__.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/architecture.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/mlx_renderer.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/archive.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/packing.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/section_value.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/receiver_proof.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/tests/test_archive.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/tests/test_receiver_proof.py`
- `src/tac/substrates/rnerv_srneerv_pr95_grammar/tests/test_mlx_bridge.py`

Runtime:

- `submissions/rnerv_srneerv_pr95_grammar/inflate.sh`
- `submissions/rnerv_srneerv_pr95_grammar/inflate.py`
- `submissions/rnerv_srneerv_pr95_grammar/src/codec.py`
- `submissions/rnerv_srneerv_pr95_grammar/src/model.py`

Tools/experiments:

- `experiments/train_substrate_rnerv_srneerv_mlx_local.py`
- `tools/export_rnerv_srneerv_mlx_to_runtime_state.py`
- `tools/package_rnerv_srneerv_pr95_grammar_archive.py`
- `tools/prove_rnerv_srneerv_archive_runtime_consumption.py`
- `tools/project_rnerv_srneerv_archive_to_hprc_spine.py`
- `tools/profile_rnerv_srneerv_section_value.py`
- `tools/build_rnerv_srneerv_compact_base_sweep.py`

Modifications to shared surfaces:

- Add `RNERV_SRNEERV_PR95_GRAMMAR` to
  `src/tac/substrates/hprc/representation_spine.py`.
- Add archive projection in `src/tac/substrates/hprc/pr95_adapter.py` or a new
  sibling adapter.
- Add compact-base sweep support in
  `src/tac/substrates/hprc/spine_bounded_runner.py`.
- Wire operator discoverability through `tools/all_lanes_preflight.py` or an
  explicit runbook once the package exists.

Tests must prove:

- Parser rejects malformed lengths, unknown flags, impossible latent shapes,
  and unconsumed bytes.
- Pack/unpack roundtrip preserves section bytes and hashes.
- Runtime render consumes the archive, not source-side model files.
- HPRC projection section bytes sum to contest packet bytes excluding only
  explicitly accounted container overhead.
- MLX bridge cannot claim archive authority without exported runtime state and
  receiver proof.
- Section-value profiler refuses score/promotion authority on local advisory
  outputs.

## Canonical-Vs-Unique Decision Per Layer

| Layer | Decision | Rationale |
| --- | --- | --- |
| Contest packet shape | Canonical PR95-style | Fair PR95/HNeRV comparison requires same archive/runtime surface. |
| HPRC projection | Canonical | Section-value and compact-base runner already consume HPRC roles. |
| Full-frame receiver proof | Canonical | Prevents latent/parser proof from becoming false frame proof. |
| MLX training substrate | Canonical helper reuse | Reuse shared score-aware loss and SSD hygiene; keep local outputs advisory. |
| Decoder topology | Principled fork | SRNeRV shared-scale decoder must differ from PR95 to save bytes. |
| Upsample operator | Principled fork | Policy search is the point of RNeRV/SRNeRV adaptation. |
| Latent table | Start canonical, fork if bytes win | PR95 table is the fair control; recurrence only if byte-positive. |
| Residual branch | Unique but gated | RT/BoostNeRV bytes are admitted only by section-value inequality. |
| Entropy coding | Canonical controls plus measured forks | PR95 split Brotli/LZMA first; range/ANS only if actual bytes win. |
| Exact eval | Canonical | No MLX/proxy promotion. |

## 18-Shared-Assumption Profile

| # | Assumption | Verdict For This Lane |
| ---: | --- | --- |
| 1 | Full RGB frame output is mandatory. | Adopt. No parser-only authority. |
| 2 | PR95/HNeRV is the primary control arm. | Adopt. Every candidate compares to PR95-shaped packet. |
| 3 | `archive.zip` plus runtime is the byte authority. | Adopt. No hidden sidecars. |
| 4 | Single monolithic member is preferable. | Adopt. Use `0.bin` unless a measured ZIP/layout win exists. |
| 5 | Pair-latent convention remains useful. | Adopt initially; fork only if recurrence beats bytes. |
| 6 | PixelShuffle is not sacred. | Fork. Keep as control but search learned upsample policy. |
| 7 | Decoder bytes dominate early. | Adopt. SRNeRV targets decoder redundancy first. |
| 8 | Selectors are valuable by default. | Reject. They need section-value proof. |
| 9 | Residual bytes are valuable by default. | Reject. Admit only by contest-score inequality. |
| 10 | RGB/YUV MSE is enough to rank variants. | Reject. It is warm-start only. |
| 11 | Local MLX can route exact spend. | Adopt with blockers. It can triage only after calibration gates. |
| 12 | Local MLX can promote/kill. | Reject. Exact CPU/CUDA axis required. |
| 13 | PR95 latent codec is the first baseline. | Adopt. It is the comparison control. |
| 14 | Range/ANS should replace Brotli/LZMA automatically. | Reject. Actual packed bytes decide. |
| 15 | Runtime dependencies are free. | Reject. Runtime closure and bytes are part of proof. |
| 16 | SR scale sharing is likely byte-relevant. | Adopt as primary hypothesis. |
| 17 | RNeRV training-time design studies are directly useful. | Adopt as search prior, not as score proof. |
| 18 | Negative partial/local results retire the lane. | Reject. They classify next proof gaps unless exact auth-axis evidence exists. |

## Six-Hook Wire-In For Implementation

The code landing should not leave this as orphan research. Wire at least:

- Sensitivity map: section and pair deltas keyed by PR95/HPRC section.
- Pareto constraint: bytes vs nonrate score for each architecture variant.
- Bit allocator: section budgets and residual-token admission decisions.
- Cathedral/autopilot dispatch hook: exact-gate candidate only after receiver
  proof and byte ledger.
- Continual-learning posterior: empirical timing, section-value, and exact
  outcomes.
- Probe disambiguator: upsample-policy and recurrence-vs-table arbitration.

## Stop/Continue Gates

Timing smoke continue:

- full-shape seconds/epoch measured;
- projected full run within available local/remote budget;
- no storage preflight blocker.

Base run continue:

- packaged bytes <= PR95 comparison budget or measured nonrate gain pays the
  byte excess;
- full receiver proof passes;
- HPRC projection emits section bytes.

Residual/token continue:

- measured `delta_nonrate + lambda_B*bytes < 0`;
- section-value profiler agrees on held-out/full-video replay;
- selectors/codebooks are consumed by runtime.

Exact dispatch continue:

- archive/runtime custody complete;
- exact-axis payload blockers clear;
- active dispatch claim exists;
- plan-only exact command is reproducible.

Stop and record blocker:

- missing receiver/runtime adapter;
- uncharged dependency or sidecar;
- MLX-only score/rank language;
- partial-frame or parser-only parity;
- section bytes cannot be deterministically reconstructed;
- no SSD tier has enough free space for the planned run.

## Recommended First Landing

First implementation tranche should be narrow:

1. Add RNRS1 archive parser/packer and HPRC projection tests.
2. Add SR-shared decoder in PyTorch/numpy first, with no residual branch.
3. Add MLX renderer and export bridge after parser tests pass.
4. Package one tiny smoke archive and prove runtime consumption.
5. Only then run a timing smoke over 8/32/full-shape pairs.

This gives a fair PR95/HNeRV comparison surface before spending time on
residual tokens, recurrence, or paid exact eval. It also prevents the main
failure mode seen in nearby lanes: a local MLX artifact or a clever parser
stream being mistaken for byte-closed contest evidence.
