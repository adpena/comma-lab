# Codex Findings: RNeRV / PACT-NeRV Base Renderer + Z8 Residual Sidecar Research

UTC: 2026-05-31T22:30:36Z

Scope: read-only research follow-up for the approved RNeRV/PACT-NeRV base renderer
and Z8 scorer-aware residual sidecar lane. This memo is advisory only. It is not
a score claim, not a promotion claim, and not CPU/CUDA authority.

## Executive Verdict

The current Z8 rate problem is structural: as long as `wavelet_blob` carries
full-video top-LL / residual mass, sub-0.19 is unreachable by better detail
quantization alone. The only plausible route is to make a compact neural base
renderer carry most SegNet/PoseNet-preserving structure, then encode a tiny
scorer-targeted Z8 residual sidecar whose bytes are selected by full-video
P18/P19 authority and entropy-coded as one global stream.

Useful outside work:

- HNeRV and VINRB/RNeRV are the best architecture/training sources to adapt.
- CompressAI, DCVC, torchac, and constriction are useful for entropy-model and
  range/ANS coding experiments, but should not become runtime dependencies.
- Mamba and Dreamer-style models are best treated as training-time/context
  priors unless distilled into a tiny deterministic archive-contained state
  machine.
- VQ-NeRV/SNeRV/HiNeRV/FFNeRV are idea sources for residual/codebook/frequency
  structure, but none should bypass Pact custody or exact-eval gates.

## Constraint Translation

Contest score is:

```text
S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489
lambda_B = 25 / 37545489 = 6.65859e-7 score / byte
```

Rate-only byte ceilings:

| total score budget | max archive bytes if distortion is zero |
| --- | ---: |
| 0.19 | 285,345 |
| 0.15 | 225,272 |
| 0.12 | 180,218 |
| 0.10 | 150,181 |
| 0.08 | 120,145 |
| 0.05 | 75,090 |

The local Z8 profile makes the near-term implication sharp: a 10.2 MB
`quantized_detail_probe` and a 23-25 MB custody-valid Z8 archive are many
orders too large. Even perfect detail collapse leaves the current top-LL/raw
structure in the megabyte range. Therefore the lane should reject any prototype
whose byte-closed packet cannot plausibly land below about 180-285 KB after
including decoder weights, latent/state payload, residual payload, metadata,
runtime contract, and ZIP overhead.

## External Sources Worth Adapting

| source | URL | useful idea | Pact adaptation |
| --- | --- | --- | --- |
| NeRV | https://arxiv.org/abs/2110.13903 and https://github.com/haochen-rye/NeRV | frame-index-to-frame neural video representation; model compression as video compression | baseline only; PSNR/RGB authority is not enough for SegNet/PoseNet |
| HNeRV | https://arxiv.org/abs/2304.02633 and https://github.com/haochen-rye/HNeRV | content-adaptive embeddings, balanced decoder capacity, model/embed quantization knobs | adapt decoder/backbone/export patterns; require archive-contained weights/embeddings and Pact inflate parity |
| VINRB / RNeRV benchmark | https://github.com/mgwillia/vinrb and https://mgwillia.github.io/vinrb/ | unified NeRV-family benchmark with RNeRV config, short-run training command, FFNeRV/HNeRV/HiNeRV/DiffNeRV variants | adapt architecture search and short timing smokes; do not trust hybrid compression eval blindly because README flags bitstream/set-zero issues |
| VQ-NeRV | https://arxiv.org/abs/2403.12401 | codebook/discrete shallow residual features and inter-frame residuals | use for residual-codebook sidecar design; count codebook bytes and forbid hidden codebooks |
| SNeRV | https://arxiv.org/abs/2501.01681 | wavelet/frequency-aware NeRV to counter spectral bias | relevant because Z8 is already wavelet-shaped; use as base-renderer loss/architecture inspiration |
| DCVC family | https://arxiv.org/abs/2109.15047 and https://github.com/microsoft/DCVC | conditional coding, temporal context, hybrid spatial-temporal entropy models | borrow entropy/context objective ideas only; final inflate cannot depend on heavy learned codec stack |
| CompressAI | https://github.com/InterDigitalInc/CompressAI and https://interdigitalinc.github.io/CompressAI/entropy_models.html | entropy bottleneck, Gaussian conditional, learned CDF update workflow | training oracle for residual/latent code lengths; final packet needs small deterministic coder/tables |
| constriction | https://github.com/bamler-lab/constriction | production/research entropy coders including range coding and ANS | good oracle and golden-vector generator; final runtime can port only the tiny needed decoder |
| torchac | https://github.com/fab-jul/torchac | fast arithmetic coding for PyTorch tensors | useful offline; GPL/C++/PyTorch dependency makes it poor final inflate material |
| Mamba / MambaIC | https://github.com/state-spaces/mamba and https://openaccess.thecvf.com/content/CVPR2025/papers/Zeng_MambaIC_State_Space_Models_for_High-Performance_Learned_Image_Compression_CVPR_2025_paper.pdf | sequence/state-space priors and channel-spatial entropy modeling | train-time prior or distilled tiny context model; no CUDA custom-op dependency in inflate |
| DreamerV3 | https://arxiv.org/abs/2301.04104 and https://github.com/danijar/dreamerv3 | categorical latent RSSM/world-model discipline | useful for latent-transition priors and acquisition, not as raw runtime unless distilled |
| Balle learned compression | https://arxiv.org/abs/1611.01704 and https://arxiv.org/abs/1802.01436 | differentiable quantization proxies, rate-distortion objective, hyperprior | use `lambda_B` plus scorer-gradient distortion, not RGB-only RD |

## Ranked Roadmap

### P0: Byte-Ledger Floor Gate Before More Training

Prototype: create a read-only/advisory ledger row for every candidate packet
with `archive.zip` bytes, payload bucket bytes, runtime-tree hash, source
artifact hashes, score-axis tag, and authority blockers. Compare all candidates
against the 285,345 byte zero-distortion ceiling and the 180-240 KB realistic
ceiling.

Acceptance:

- every row distinguishes decoder weights, latent/state, residual, entropy
  tables, metadata, runtime closure, and ZIP overhead;
- rows with missing archive/runtime custody are `score_claim=false`;
- any model/residual plan with projected packet >285 KB is rejected for
  sub-0.19 promotion unless it is explicitly a diagnostic.

Likely byte floor: no direct savings, but prevents another 10-25 MB false
frontier loop.

Risk/anti-pattern: treating model PSNR, MLX loss, or residual MSE as byte-floor
evidence without a real archive packet.

### P1: RNeRV-Lite Latent Generator Over Existing PACT-NeRV Decoder

Prototype: replace independent per-pair latent tables with a recurrent or
state-space latent generator:

```text
h_i = f(t_i, h_{i-1})
z_i = g(h_i) + quantized_epsilon_i
frame_i = existing PACT-NeRV/HNeRV-style decoder(z_i)
```

First materialize the generated `z_i` table into the existing archive path to
separate training value from runtime byte value. Then, only if it wins, move the
recurrence into inflate and encode initial state plus tiny weights instead of a
600-pair table.

Acceptance:

- 600-pair MLX-first timing smoke with deterministic seed/config and no score
  claim;
- compressed latent/residual payload at least 2x smaller than independent
  latent table at comparable local proxy distortion;
- projected byte-closed packet <285 KB before exact-eval dispatch is considered;
- exported archive/runtime proof, not only `.latents.npy`.

Likely byte floor: 180-320 KB if materialized latents remain; 105-280 KB only
if recurrence replaces most per-pair bytes in inflate.

OSS worth adapting: VINRB RNeRV config/training harness and HNeRV/PACT-NeRV
decoder blocks. Avoid importing VINRB's hybrid compression-eval path as
authority because its README warns the current hybrid bitstream eval is
unreliable.

### P2: Replace Z8 Top-LL Payload With Neural Base + Tiny Wavelet Residual

Prototype: train the RNeRV/PACT-NeRV base to reconstruct the semantic structure
that keeps SegNet/PoseNet stable, then compute residuals in Z8 wavelet space.
Encode only the residual coefficients selected by full-video P18/P19 slopes,
not a full top-LL or dense detail image.

Acceptance:

- current multi-megabyte top-LL/raw section is gone from the packet;
- residual sidecar target <=75 KB for strong candidates, <=120 KB for early
  custody smokes;
- runtime consumes the residual through `inflate.sh archive_dir output_dir
  file_list`, with no hidden sidecar and no scorer import;
- full-frame inflate output and archive/runtime hashes are preserved before any
  exact CPU/CUDA claim.

Likely byte floor: 180-285 KB is plausible; 120-180 KB is a stretch only if the
base renderer carries almost all semantic structure and the sidecar is sparse.

Risk/anti-pattern: keeping full Z8 top-LL "for safety" makes the lane dead on
arrival; scoring a residual sidecar against RGB MSE will spend bytes in the
wrong locations.

### P3: True Full-Video P18/P19 Sidecar Allocation

Prototype: promote the residual allocator from scalar/local proxies to full
600-pair authority: P18 target/boundary surfaces plus P19 six-axis PoseNet
Mahalanobis gradients or finite-difference null subsets. Feed these slopes into
KKT/Dykstra/RD-waterfill over residual atom classes.

Acceptance:

- six PoseNet axes and inverse-variance weights are present for every authority
  row;
- all 600 pairs are covered, finite, and tied to the archive candidate hash;
- allocation emits per-atom `Delta S`, bytes, quantizer, and blocker metadata;
- no `budget_spend_authority=true` from scalar pose loss or sampled-only rows.

Likely byte floor: can reduce residual bytes by 2-10x versus MSE-sidecar
selection. Target residual payload 20-80 KB.

Risk/anti-pattern: scalar P19 or pair-local gradients can look good while
breaking contest PoseNet after inflate.

### P4: Global Entropy Coder For Latent/Residual Streams

Prototype: build an offline oracle with constriction or torchac, then implement
or reuse a tiny deterministic range/ANS decoder with golden vectors in the Pact
runtime. Encode global streams by class/subband/time/context, not per-pair ZIP
fragments.

Acceptance:

- exact coefficient round-trip and stable SHA across machines;
- compressed bytes within 5-10% of empirical entropy on full 600-pair streams;
- final inflate runtime has no torch, CUDA, scorer, network, or hidden cache;
- entropy tables are counted in archive bytes.

Likely byte floor: 5-30 KB savings on small streams; more if current residual
symbols remain poorly modeled. This is not enough alone but is mandatory for a
sub-285 KB packet.

Risk/anti-pattern: shipping a dependency-heavy coder or per-section overhead
that erases all entropy gains.

### P5: Learned Hyperprior / Context Model For Residual Symbols

Prototype: use CompressAI/DCVC-style entropy models to predict residual symbol
scales from base-renderer state, wavelet band, time, and class/boundary
features. Distill the result into small tables or a tiny deterministic model
only if overhead amortizes over all 600 pairs.

Acceptance:

- net archive bytes lower than static contexts after counting hyperprior/model
  bytes;
- context model is archive-contained and runtime-deterministic;
- exact same decoded residuals from oracle and Pact runtime.

Likely byte floor: 10-50 KB savings if residual stream is still >100 KB; often
negative when residual stream is already tiny.

Risk/anti-pattern: learned entropy model overhead exceeding payload savings.

### P6: Mamba/Dreamer Priors For Acquisition, Not First Runtime

Prototype: train a Mamba/RSSM prior to predict latent transitions, residual
contexts, or code lengths, then use it to rank residual atoms or initialize the
RNeRV-lite generator. Only distill into runtime if the state machine is small
and deterministic.

Acceptance:

- lower residual entropy or better component preservation at the same byte
  budget versus non-Mamba/non-RSSM baseline;
- no custom CUDA state-space op or large Python stack in inflate;
- model/state bytes counted and compared to saved residual bytes.

Likely byte floor: uncertain; high-upside if it removes most per-pair latent
payload, otherwise mostly training acceleration.

Risk/anti-pattern: hiding a large temporal model in runtime or treating a
training prior as archive compression.

### P7: Quantization-Aware Training With Real Archive Bytes In The Loop

Prototype: train PACT-NeRV weights, recurrence state, residual coefficients,
and codebooks with STE/additive-noise/soft-to-hard quantization, but the loss
must include Pact `lambda_B` and scorer-aware P18/P19 distortion proxies. Emit
an actual packet during training checkpoints.

Acceptance:

- checkpoint artifacts include quantized bytes, entropy tables, and packet
  hashes, not only floating weights;
- post-training archive bytes match training-estimated bytes within tolerance;
- candidate can be inflated deterministically without training code.

Likely byte floor: enabling technology for all lower floors; by itself it does
not solve the top-LL structural issue.

Risk/anti-pattern: post-hoc quantization of a float model after the search has
already learned to spend unavailable precision.

## Immediate Prototype Order

1. Build the P0 byte-ledger/floor gate from existing Z8 q0156,
   `quantized_detail_probe`, PACT-NeRV selector artifacts, and PR95/HNeRV
   packets.
2. Run P1 RNeRV-lite as a latent generator while still materializing latents,
   so the first result answers "does recurrence learn useful temporal structure?"
   before claiming byte savings.
3. In parallel, prototype P2 residual-sidecar projection from a fixed PACT-NeRV
   base output into Z8 wavelet residual atoms, with P18/P19 advisory flags.
4. Upgrade P3 authority before spending exact-eval budget: true six-axis P19,
   P18 target/boundary coverage, full 600-pair finite surfaces.
5. Only after a residual stream is demonstrably <=120 KB, spend time on P4/P5
   entropy-model polish.

## Hard Acceptance Gates For A Candidate Worth Exact Eval

- `archive.zip` <=285,345 bytes for any sub-0.19 attempt, with a preferred
  pre-dispatch target <=240 KB.
- All model weights, recurrence state, residuals, entropy tables, codebooks,
  metadata, and runtime files are inside the archive/runtime custody surface.
- `inflate.sh` produces frames without network access, scorer imports, hidden
  sidecars, or local cache dependence.
- MLX outputs are tagged `[macOS-MLX research-signal]` only until the same
  archive/runtime packet is replayed on contest CPU/CUDA authority.
- P19 authority uses six axes plus inverse-variance weights; scalar pose loss
  can rank diagnostics only.
- Byte ledger records archive SHA, runtime tree SHA, source candidate hashes,
  exact command, seed/config, and blocker state.

## Anti-Patterns To Block

- "RNeRV helped training" while exporting a full 600-pair latent table with no
  byte win.
- PSNR/RGB-MSE promotion of an INR that loses SegNet/PoseNet.
- Retaining Z8 full top-LL/raw payload as a safety sidecar.
- Using VINRB, CompressAI, torchac, Mamba, or Dreamer as an uncounted runtime
  dependency.
- Treating MLX-local component proxies as contest authority.
- Counting residual entropy without table/model overhead.
- Hiding codebooks, CDFs, latents, or base-frame caches outside `archive.zip`.
- Exact-eval dispatch before archive/runtime custody and full-frame inflate
  proof exist.

## Bottom Line

The best first-class lane is not "better Z8 quantization." It is:

```text
compact RNeRV/PACT-NeRV semantic base
+ scorer-selected sparse Z8 wavelet residual
+ global entropy-coded latent/residual packet
+ full-video P18/P19 allocation
+ exact archive/runtime custody
```

If the neural base cannot remove the megabyte-scale top-LL burden, this lane
cannot reach sub-0.19. If it can, the byte floor becomes realistic only when
the sidecar is tens of kilobytes, not megabytes.
