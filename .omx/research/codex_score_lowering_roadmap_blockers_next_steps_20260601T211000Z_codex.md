# Codex Roadmap: Score Lowering, Blockers, And Next Steps

UTC: 2026-06-01T21:10:00Z
Agent: Codex
Scope: end-to-end score lowering roadmap from current `main`

## Current Authority State

Verified with `tools/scan_best_anchor_per_axis.py` from current `main`:

- Best `[contest-CPU Linux x86_64]`: `0.1919853363`, archive SHA-256 prefix
  `b7106c9bdbb8`.
- Best `[contest-CUDA T4]`: `0.2053300290`, archive SHA-256 prefix
  `9cb989cef519`.
- MLX, macOS CPU, local replay, section-value profiling, and planning rows are
  candidate-generation signals only. They are not score authority.

The frontier is already close enough that ordinary byte grammar polish on the
current HNeRV packet is mostly saturated. The highest-EV remaining work is not
"better prose" or broad proxy sweeps; it is building a smaller, score-faithful
learned carrier and admitting only bytes whose measured score value exceeds the
contest byte price.

## Governing Objective

The contest score is:

`S = 100*d_seg + sqrt(10*d_pose) + 25*(archive_zip_bytes/uncompressed_total)`

The fixed byte price is `25/uncompressed_total`. Every component must answer
one question: does this charged byte reduce `100*d_seg + sqrt(10*d_pose)` by
more than its rate cost after receiver-proofed archive replay?

Important scorer facts:

- Required output is `1164x874`, but both PoseNet and SegNet resize to
  `512x384`.
- SegNet scores only the last frame of each pair.
- PoseNet scores both frames through YUV6 and the first six pose dimensions.
- Therefore the stack has frame asymmetry, pair coupling, and a real
  resolution-axis dead-zone above scorer input resolution.

## Primary Score-Lowering Roadmap

### 1. Compact Learned Carrier Race

Purpose: beat the rate axis by replacing explicit residual/field storage with a
tiny learned decoder plus high-value latents/tokens.

Current executable carriers:

- `pr95_hnerv`: control arm and public-frontier reference; useful because it
  proves tiny decoder plus tiny latents can be score-competitive.
- `pact_nerv_vq`: executable MLX full-video compact-base lane; long run active
  on SSD with 600 pairs and score-aware distillation.
- `pact_nerv_selector_v4`: executable MLX selector lane, currently secondary.

Newly routed carrier targets:

- `hi_nerv`: primary carrier hypothesis. Existing HIV1 archive bytes can now be
  projected into HPRC sections. Missing piece: MLX/portable trainer/exporter
  and receiver-proven archive path.
- `snerv`: primary carrier hypothesis. It must become a real archive/runtime
  carrier with charged LF/HF/wavelet/state bytes, not advisory side-channel
  signal.

Enhancers/design knobs:

- `sr_nerv`: highest-priority enhancer. Encode internally at or below scorer
  resolution, super-resolve to legal `1164x874`, then prove the scorer
  downsample round-trip preserves SegNet/PoseNet.
- `rnerv`: component-search/recurrent latent generator prior over the winning
  carrier. Useful only if it reduces charged decoder+latent entropy.
- `ffnerv_flow`: pose-channel enhancer if flow bytes are charged.
- `boostnerv`: temporal-affine/conditional decoder enhancer if value-per-byte
  is positive.

Immediate tasks:

1. Let the current 600-pair `pact_nerv_vq` MLX run finish and export.
2. Run receiver proof and archive byte profile on its exported candidate.
3. Run full-video MLX section-value pricing for decoder, codebook, index/token,
   selector, and residual sections.
4. Demote sections/families whose `delta_nonrate + rate_cost >= 0`.
5. Port HiNeRV/SNeRV to MLX/portable archive exporters under the same spine.
6. Add SR-NeRV low-res/SR mirror check before treating SR as an enhancer.

### 2. PR95 / Public HNeRV Exploitation

Purpose: understand and extend the proven small-rate grammar without assuming
HNeRV is theoretically optimal.

Current facts:

- PR95-style learned receiver is valuable because it pays for a small decoder
  and small latent stream instead of explicit full-video residual fields.
- PR101/PR103-style grammar refinements appear near saturated on the current
  packed HNeRV substrate; perfect grammar alone is unlikely to move score much.

Tasks:

1. Continue scorer-faithful PR95 Stage-8 from public archive.
2. Import only source-faithful trained weights/latents into the packet spine.
3. Compare PR95/HNeRV, PACT-NeRV-VQ, HiNeRV, SNeRV, and SR-enhanced variants
   under identical byte ceilings: `100k`, `178k`, `216k`, `285k`.
4. Exact-gate only candidates with plausible full-video local evidence.

### 3. Score-Coupled RD Allocation

Purpose: give every atom exactly the bits it earns, and no more.

Required allocator:

- SegNet saliency: last-frame boundary/argmax-flip sensitivity.
- PoseNet saliency: pairwise pose Jacobian / Fisher sensitivity over both
  frames.
- Rate price: fixed contest byte price.
- Resolution policy: do not pay for high spatial frequencies that vanish under
  scorer resize unless they affect downstream pose/seg after resize.

Tasks:

1. Replace static sensitivity maps with exact-reduced full-video MLX VJP where
   feasible.
2. Use chunks only as exact reductions: no minibatch promotion before full
   reduction.
3. Add bundle/trust-region handling for nonsmooth argmax boundaries.
4. Use hard archive projection and replay acceptance; STE/soft quantization may
   propose, but byte-closed replay decides.
5. Feed every positive and negative section-value result into posterior routing.

### 4. Codec And Packet Grammar

Purpose: minimize paid bytes for the chosen carrier and its tokens.

Already landed:

- Shared decoder-state codec portfolio with `int8`, `int4`, `int2`, and fp16
  envelope modes.
- Shared integer-stream codec portfolio: varints, delta-zigzag, zero-run
  varints, fixed-width bitpacking, and packed bitmasks.
- PACT VQ / selector-v4 exporters now preserve codec metadata and can use the
  shared portfolio.

Remaining:

1. Train with codec awareness, not just post-hoc quantize.
2. Add per-section entropy-gap reports to acquisition.
3. Use int2/int4/int8/fp16 only where section-value pricing justifies them.
4. Add native/Rust lowering only after profiling proves Python or decode time is
   the bottleneck.

### 5. Z8 / HPRC / Residual Knowledge

Purpose: use explicit wavelet/HPRC work as saliency and residual knowledge, not
as the primary carrier unless rate collapses enough.

Current facts:

- Z8 can be faithful but is rate-bound when explicit wavelet blobs dominate.
- Dead-zone and wavelet allocation findings remain valuable as scorer-aware
  priors for compact carriers.

Tasks:

1. Continue Z8 work only where it informs compact carrier allocation or tiny
   residual tokens.
2. Admit residual tokens only when full-video replay proves
   `delta_nonrate + rate_cost < 0`.
3. Keep every Z8/HPRC residual archive byte-closed and receiver-proven.

### 6. Exact Promotion Loop

Purpose: prevent proxy/local wins from consuming attention or paid exact spend.

Required loop:

1. Choose contract-backed candidate work.
2. Train/materialize archive bytes.
3. Prove `inflate.sh` consumes those bytes.
4. Run local/full-video MLX replay as advisory triage.
5. Build exact CPU/CUDA blocker or dispatch packet.
6. Harvest exact result.
7. Update posterior budget routing and demote failures.

Promotion blockers:

- No hidden sidecars.
- No scorer state inside receiver.
- No local/MLX/MPS row promoted as exact authority.
- No archive candidate without runtime/custody hashes.
- No exact dispatch without lane claim and reproducibility manifests.

## Outstanding Blockers

Hard blockers:

- HiNeRV/SNeRV are not yet MLX/portable executable compact runner targets.
- SR-NeRV needs the low-res -> SR -> contest-output -> scorer-downsample mirror
  check before it can be trusted.
- PR95 Stage-8 continuation must prove source-faithful training/export and full
  receiver parity before exact claims.
- PACT VQ full run must finish, export, receiver-proof, and get section-value
  replay before exact gating.

Soft blockers / risks:

- Long training may require many epochs; one-epoch smokes are not evidence of
  convergence.
- Pose may collapse or plateau if the carrier lacks temporal/geometry capacity.
- High visual fidelity is not the objective; scorer-faithful low-res and
  boundary/pose preservation are.
- Partner work is active in the dirty shared checkout; merge only reviewed,
  disjoint, ownership-clear slices.

## Immediate Next Executable Queue

1. Harvest the current 600-pair PACT-NeRV-VQ MLX run when it completes.
2. Receiver-proof its exported archive, profile archive bytes, and run section
   value replay.
3. If the candidate is far from frontier, use its section profile to cut or
   recode decoder/codebook/index sections and relaunch the smallest high-EV
   long run.
4. Implement SR-NeRV scorer mirror check as a $0 proof.
5. Port HiNeRV trainer/exporter to MLX/portable packet spine.
6. Build SNeRV archive/runtime grammar, then MLX exporter.
7. Keep PR95 Stage-8 as the control-arm reproduction and comparison anchor.
8. Exact-gate only receiver-proven candidates whose byte/value profile can
   plausibly improve `0.1919853363 [contest-CPU]` or the CUDA anchor.

## Definition Of Done For This Phase

This phase is complete only when at least one of these is true:

- A byte-closed, receiver-proven compact learned carrier beats the current
  exact frontier on a contest axis.
- A precise exact-axis blocker identifies the missing component preventing
  improvement.
- Durable negative evidence demotes a carrier/enhancer family and routes budget
  to the next best family without manual interpretation.

