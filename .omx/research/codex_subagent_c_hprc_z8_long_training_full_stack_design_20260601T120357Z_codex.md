# Subagent C HPRC/Z8 Long-Training Full-Stack Design

Captured: 2026-06-01T12:03:57Z
Author: Codex Subagent C
Repo: /Users/adpena/Projects/pact

Authority: design and dispatch plan only. No score claim, no promotion claim, no exact-eval
authority, no CUDA/CPU equivalence claim, and no sidecar authority. This memo creates no
candidate archive and launches no job.

## Scope And Constraint

The requested lane is the third full-stack design around HPRC/Z8 residual knowledge and
long-training dynamics. The artifact is intentionally a memo because the operator allowed
at most one new `.omx/research/` file and forbade touching other files. That also means the
normal `tools/lane_maturity.py add-lane ...` state write is blocked for this turn; any
future executable continuation must register a lane before execution.

This memo treats all current Z8, HPRC, MLX, and local CPU rows as evidence for routing,
not authority. The exact frontier surfaces remain distinct by axis:

- Current `[contest-CPU]` frontier from `tac.frontier_scan.build_frontier_scan_payload`:
  score `0.19198533626623068`, archive bytes `178493`, archive SHA-256
  `b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c`,
  hardware `linux_x86_64_cpu`, measured `2026-05-28T17:56:34Z`.
- Current `[contest-CUDA]` frontier from the same scan: score
  `0.20533002902019143`, archive bytes `186876`, archive SHA-256
  `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`,
  hardware `linux_x86_64_t4`, measured `2026-05-16T07:20:32Z`.

## Research Anchors Used

- Rao and Ballard predictive coding: hierarchy predicts lower-level input and sends
  residual errors upward. This supports HPRC as a base receiver plus priced residual-error
  stream, not as a bag of unbound pixels.
- NeRV/HNeRV: neural video representations fit a compact decoder/embedding system to
  video frames. This supports a compact base receiver, but in this repo the base is only
  useful when the exported runtime consumes the actual weights/latents/categories.
- Mamba and Mamba-2/SSD: selective state-space sequence models and Mamba-2's SSD
  refinement are useful temporal priors for long video residual prediction. They are not
  runtime authority unless their weights and deterministic recurrence are archive-bound.
- DreamerV3: world-model categorical latent dynamics are a useful allocator prior for
  long-horizon consistency. They become contest-relevant only as exported categories,
  selectors, or decoder weights consumed by the receiver.
- Deep contextual video compression and hyperprior learned compression: conditional
  coding and entropy models justify a context-conditioned residual stream with explicit
  entropy pricing, but the contest packet must still be deterministic and self-contained.

Primary sources checked online:

- https://arxiv.org/abs/2312.00752
- https://arxiv.org/abs/2405.21060
- https://arxiv.org/abs/2301.04104
- https://arxiv.org/abs/2110.13903
- https://arxiv.org/abs/2304.02633
- https://arxiv.org/abs/2109.15047
- https://arxiv.org/abs/1802.01436
- https://www.nature.com/articles/nn0199_9

## Local State Read

### Z8

Z8 is faithful but rate-bound. Prior local MLX long training solved pose strongly while
SegNet remained the hard part: full-600, 2000 epoch advisory training drove pose from
about `104.6` to `0.067` while SegNet moved from about `6.45` to `2.68`. The important
negative is not pose collapse; it is archive authority. Z8 detail streams can be enormous
without the entropy/materializer bridge, and local MLX outputs are advisory until an
archive-bound runtime consumes the exported bytes.

Recent Z8 rate work found the raw-f32 detail blob dominated archive size. The useful
knob is per-subband/per-cell delta with scorer protection, not a global detail drop.
The corrected rule is:

```
keep_priority = coefficient_magnitude * scorer_protection
```

Scorer protection must not replace coefficient magnitude, because zeroing a large
coefficient just because it sits in a low-priority cell can create unpriced visible or
PoseNet damage.

### HPRC

The HPRC spine is the right container for this lane: `decoder_qw`, `latents_rc`,
`codebooks_q`, `selectors_rc`, `residual_rc`, `rdo_plan`, and `receiver_state`.
The receiver is already numpy-portable and decode-only; scorer-aware information must
be bound into sections consumed by inflate/runtime.

The latest corrected full-video P18/P19 HPRC run is the best current diagnostic for
pose collapse and exact gating:

- Run root:
  `/Volumes/VertigoDataTier/pact/experiments/results/hprc_compact_receiver_training_queue/hprc_full600_corrected_p18p19_20260601T112753Z`
- Training artifact: 8 epochs, MLX training backend, numpy-portable runtime export.
- Training archive before rate collapse: `1094029` bytes,
  SHA-256 `5ceed509facf2f901973a59a6a96377af370e9a7951ba67e849e8a1a4c0e4e0d`.
- Native P18/P19 protection surface: full video, shape `[1200,24,32,3]`,
  protection min `0.15000000596046448`, mean `0.9579427242279053`, max `1.0`,
  rate pressure mean `0.04205729067325592`.
- P19 source: 30 selected/in-range pairs, artifact SHA-256
  `96e17d3c997dbbadfea0ff449fa1376b0cee0029bbe90dd6712346fe2746b090`.
- P18 source: 30 in-range rows, 240 protected cells, artifact SHA-256
  `0fa5d5872ff9deab18899df528437007fbff42f7d27bb5eee5146f7960245462`.
- Rate collapse best variant: `residual_tokens_dz0_qd10`; archive bytes saved
  `984627`; best archive bytes `109402`; best archive SHA-256
  `d9a5026f10e7a00df73279f5cad4c305583f30c0f56e23fb47e145855c31ce8c`.
- Rate gate: local replay recommended on archive rate alone, rate term
  `0.07284630118947179`.
- Local replay `[macOS-CPU advisory]`: archive bytes `109402`, rate `0.00291385`,
  SegNet distortion `0.06640102`, PoseNet distortion `35.64191818`, score
  `25.59201554157985`, evaluation passed.
- Exact gate: blocked with `local_score_not_below_auth_target`; exact dispatch not
  recommended.
- Binding caveat from rate collapse: P18/P19 artifacts are compatible proxy priors
  but include `p19_source_archive_is_proxy_not_hprc_source_archive`. This is useful
  for design and triage, not for final authority.

This corrected run is rate-good enough to replay but scorer-bad because PoseNet is
catastrophic. The earlier full600 native-rate campaign was even more pose-bad:
archive bytes `57493`, local score `43.226745304360634`, SegNet `0.06693314`,
PoseNet `133.18959045`. The trend is clear: current HPRC rate collapse can shrink
bytes dramatically while destroying pose if pose is not an active full-video training
and replay constraint.

## Answer: Could Pose Collapse If HPRC Trains Longer?

Yes. Longer HPRC training can collapse pose unless the training loop measures pose as a
first-class constraint and preserves pose-sensitive residual capacity. The current HPRC
training surface is mostly a compact receiver gain/residual projection with RGB/MSE and
rate/prox pressure. P18/P19 surfaces currently guide residual protection, but they are
not a full differentiable PoseNet objective and, in the corrected full600 run, they are
proxy priors from the source frontier archive rather than HPRC-bound sensitivity for the
actual candidate.

The evidence split is important:

- Z8 long MLX training did not show pose collapse; pose improved sharply under a more
  faithful frame teacher. That is evidence that long training itself is not toxic.
- HPRC compact/rate-collapse rows do show pose collapse when the receiver is tiny and
  residual pressure is not actively priced against full-video PoseNet.
- Therefore the failure mode is not "long training always collapses pose." It is
  "long training plus compact receiver plus residual byte pressure plus stale/proxy
  pose protection can converge to pose-bad but rate-good packets."

The likely mechanisms are:

1. PoseNet is sensitive to low-amplitude geometric and temporal cues that RGB MSE and
   SegNet boundary priors do not preserve.
2. P19-null pairs identify places where a particular source archive was safe to coarsen;
   they do not prove all HPRC residual cells are safe after decoder dynamics change.
3. The HPRC receiver can use low-res/global structure that keeps SegNet tolerable while
   erasing small pose-bearing cues.
4. Rate collapse selects residual token coarsening by archive win first. If pose
   damage is only checked after export, a catastrophic pose row can be durable but
   non-promotable.

## Pose Guardrails

All longer HPRC training must add these guardrails before spending serious wall clock:

1. Full-video singleton scorer checkpoints. Every accepted checkpoint family must run a
   full-600 HPRC-direct MLX scorer-response prefilter with `scorer_batch_pairs=1`.
   MLX is advisory only, but it is the cheap pose sentinel.

2. Pose stop rule. Stop or revert the candidate when:

   ```
   delta_nonrate + rate_cost >= 0
   ```

   or when PoseNet's contribution alone exceeds the reserved byte gain. In score units,
   `rate_cost = 25 * delta_archive_bytes / 37545489`.

3. Pose reserve, not just total score. A rate win cannot hide pose collapse behind a
   small SegNet improvement. Track:

   ```
   pose_score = sqrt(10 * d_pose)
   seg_score = 100 * d_seg
   total = seg_score + pose_score + rate_score
   ```

   If `pose_score` rises by more than the remaining rate headroom after SegNet movement,
   the checkpoint is demoted before local CPU replay.

4. P19 non-null protection floor. P19-null cells may be eligible for low-priority
   coarsening, but P19 non-null, high-gradient, or unmeasured pose cells must retain a
   high-protection floor. P19-null is "cheaper first," not "free to destroy."

5. Six-axis P19 accounting. The surface must preserve pose-axis covariance/weights,
   not just pair ids. Pair-level null flags are insufficient after the receiver changes.

6. Recompute P18/P19 after material changes. Any accepted decoder, selector, residual,
   or entropy-mode change must rebuild or revalidate P18/P19 against the candidate.
   Prefix-projected, stale, proxy, or cross-archive surfaces may guide search but block
   authority.

7. Boundary and pose scopes need cache extension. The current MLX component profile
   records `class_region_boundary_scopes_require_logits_or_boundary_cache_extension`.
   That blocks class/boundary byte-value authority. Add logits/boundary cache before
   claiming section/class pricing beyond advisory scope.

8. Archive-bound only. No hidden Mamba state, Dreamer latent, Z8 tensor, or calibration
   file may be required outside `archive.zip` and the submitted runtime tree.

## Full-Video MLX Loss Design

The training objective should be two-layered:

### Inner differentiable HPRC trainer

Use MLX for fast local training, but export a numpy-portable packet. Train only archive
consumed state: decoder weights, latents, selectors, residual tokens, codebooks, and
receiver constants.

Candidate loss:

```
L_train =
    w_rgb * mse_rgb255(candidate, reference)
  + w_boundary * D_boundary_proxy(candidate, P18)
  + w_pose_proxy * D_pose_proxy(candidate, P19_axes)
  + lambda_bytes * estimated_section_bits
  + lambda_residual * sum(abs(q_residual) * (1 - protection))
  + lambda_prox * residual_update_prox
  + lambda_temporal * temporal_consistency
```

The first implementation can keep `D_boundary_proxy` and `D_pose_proxy` as cached
surrogate terms, but the checkpoint acceptance gate cannot rely on those terms alone.

### Outer full-video scorer closure

Every checkpoint or rate-collapse survivor is decoded through the HPRC receiver and
scored through the full-video MLX scorer-response path:

```
score_mlx_advisory =
    100 * d_seg_mlx
  + sqrt(10 * d_pose_mlx)
  + 25 * archive_bytes / 37545489
```

This row is `[macOS-MLX research-signal]`, not score authority. It decides whether to
spend local CPU replay time. Local CPU replay remains `[macOS-CPU advisory]` and exact
CPU/CUDA authority requires the exact gate.

The long-training controller should accept a checkpoint only if all are true:

- Full-video MLX pose component is not worse than the previous accepted checkpoint by
  more than the byte savings can pay for.
- Full-video MLX total score satisfies `delta_nonrate + rate_cost < 0`.
- The accepted checkpoint's archive-bound sections have deterministic SHA-256 custody.
- P18/P19 surface provenance is candidate-bound or explicitly tagged proxy/advisory.

## Byte Pricing

Global contest byte price:

```
score_per_byte = 25 / 37545489 = 0.0000006658589531221714
score_per_1000_bytes = 0.0006658589531221714
score_per_1024_bytes = 0.0006818395679971035
score_per_10KiB = 0.006818395679971035
score_per_50KiB = 0.03409197839985518
score_per_100KiB = 0.06818395679971036
```

Every residual, selector, latent, or decoder change must pay:

```
value_score = -(delta_seg_score + delta_pose_score)
cost_score = score_per_byte * added_bytes
admit iff delta_nonrate + rate_cost < 0
```

where:

```
delta_nonrate = 100 * delta_d_seg + delta_sqrt_pose
delta_sqrt_pose = sqrt(10 * d_pose_after) - sqrt(10 * d_pose_before)
rate_cost = score_per_byte * delta_archive_bytes
```

Section pricing:

- `decoder_qw`: global section. Admit bytes only when a full-video replay proves the
  decoder's nonrate gain amortizes across many pairs. Expensive decoder bytes need
  stronger evidence than local RGB MSE.
- `latents_rc`: pair-local/global hybrid. Price per pair and per frame chunk; reject
  latent growth on P19-null or P18-low cells unless it buys nonrate score.
- `selectors_rc`: boundary and motion-mode chooser. Price per block/class/boundary.
  Needs logits/boundary cache extension before class-specific authority.
- `residual_rc`: main Z8/HPRC bridge. Price per residual cell:

  ```
  cell_cost = encoded_cell_bytes * score_per_byte
  cell_value = -(delta_seg_score_cell + delta_pose_score_cell)
  ```

  Coarsen lowest positive-cost/negative-value cells first, but never below pose and
  boundary protection floors.
- `codebooks_q`: shared residual bases. Admit only if multiple pairs/classes consume
  the codebook enough to amortize fixed overhead.
- `receiver_state` and `rdo_plan`: metadata overhead. Prefer deterministic generation
  from archive-bound constants; keep only when reproducibility or decode speed requires
  explicit bytes.

Pair/class/boundary pricing:

- Pair value should be measured from singleton-batch full-video arrays when possible,
  because scorer batching can move PoseNet.
- Boundary cells get high protection if P18 marks SegNet vulnerability or if local
  boundary/logit cache shows class transition risk.
- P19-null cells enter the low-priority candidate set; P19 non-null and unmeasured cells
  stay protected until candidate-bound pose sensitivity says otherwise.
- If two cells have equal byte cost, prefer preserving the one with larger
  `coefficient_magnitude * scorer_protection`.

## Third Full Stack Design

Name: `hprc_z8_mamba_dreamer_full_stack_v1`

All components must have clean provenance and must be archive-bound:

1. Base receiver: HPRC compact numpy receiver, using `decoder_qw`, `latents_rc`,
   `selectors_rc`, and `receiver_state`.
2. Neural base prior: PR95/HNeRV-like decoder or PACT-NeRV-VQ only if full600 archive
   export exists and runtime consumes the exact exported weights/latents. Partial
   32-pair rows are training signal, not a base candidate.
3. Z8 residual teacher: Z8 detail surfaces produce scorer-aware residual-token priors.
   They become `residual_rc` tokens after P18/P19 and byte pricing. Z8 raw bins or
   tensors outside archive are blockers.
4. Mamba/Mamba-2 temporal prior: used to predict residual token means/scales or selector
   contexts across frame order. Runtime may regenerate hidden state deterministically
   from archive bytes; no external state. Weights must live in `decoder_qw` or
   `receiver_state`, or be distilled into static codebooks/selectors.
5. Dreamer-style dynamics: used as an allocator for categorical latent modes and
   long-horizon consistency. Only committed categories/codebooks/selectors survive into
   the archive.
6. HPC stack member: local MLX/HPC acceleration is a training and profiling substrate,
   not a runtime dependency unless the contest runtime consumes an explicitly supported,
   portable implementation. The exported inflate path remains numpy/torch-contest-safe.
7. Entropy mode: keep section-local brotli for current receiver-proof path; add static
   range/rANS only for `residual_rc` after a deterministic decoder is implemented and
   receiver proof covers it. No sidecar probability table; tables live in the archive.

## Archive-Bound Adapter Plan

The adapter must produce a normal submission tree:

```
submission/
  archive.zip
  src/
    inflate.sh
    inflate.py
    tac_hprc_receiver.py
```

Archive member plan:

```
0.hprc
  header/version/runtime-contract
  decoder_qw
  latents_rc
  codebooks_q
  selectors_rc
  residual_rc
  receiver_state
  entropy_tables
  section_hashes
```

Adapter requirements:

- Runtime reads only `archive.zip` plus submitted `src/`.
- All sections have bytes, SHA-256, compression mode, and decode shape metadata.
- Receiver proof decodes at least a deterministic smoke and records output SHA-256.
- Lossy rate collapse emits a manifest with source archive SHA-256, candidate archive
  SHA-256, changed sections, residual collapse spec, P18/P19 artifact SHAs, and false
  authority flags.
- Existing tools that expect `archive.zip` continue to work; no operator-facing path is
  `/tmp`.
- Cleanup is certified: non-best variants can be reclaimed only with the existing
  cleanup manifest preserving bytes, SHA-256, command/config, and rebuild reason.

## Concrete Long-Training Plan

### Phase 0: lane registration and storage preflight

Required before any execution because this memo could not touch state:

```
.venv/bin/python tools/lane_maturity.py add-lane hprc_z8_mamba_dreamer_full_stack_v1 \
  --name "HPRC/Z8 Mamba Dreamer full stack" \
  --phase 0
```

Then rely on the queue builder's storage waterfall. Bulky outputs must land under
`/Volumes/VertigoDataTier/pact` if available.

### Phase 1: corrected HPRC full600 long-training queue

Build the queue, but do not execute until the lane is registered and the operator or
parent agent confirms this is the chosen frontier action:

```
.venv/bin/python tools/build_hprc_compact_receiver_training_queue.py \
  --output .omx/research/hprc_z8_mamba_dreamer_full_stack_v1_queue.json \
  --plan-output .omx/research/hprc_z8_mamba_dreamer_full_stack_v1_plan.json \
  --run-id hprc_z8_mamba_dreamer_full_stack_v1_20260601T120357Z \
  --campaign-pairs 32 \
  --campaign-pairs 128 \
  --campaign-pairs 600 \
  --decode-height 96 \
  --decode-width 128 \
  --epochs 2000 \
  --batch-pair-indices-per-step 16 \
  --learning-rate 0.001 \
  --curriculum-preset hprc_native_rate_ramp_v1 \
  --basis-count 8 \
  --residual-grid-h 24 \
  --residual-grid-w 32 \
  --training-backend mlx \
  --enable-native-rate-aware-hprc \
  --native-rate-residual-l1-weight 0.0001 \
  --native-rate-residual-prox-weight 0.25 \
  --native-rate-p19-posenet-null-pairs /Volumes/VertigoDataTier/pact/scorer_region_cascade_b7106_campaign_20260531T214623Z/nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch/p19_posenet_null_pairs.json \
  --native-rate-p18-segnet-region-waterfill /Volumes/VertigoDataTier/pact/scorer_region_cascade_b7106_campaign_20260531T214623Z/nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch/p18_segnet_region_waterfill.json \
  --hprc-rate-collapse-p19-posenet-null-pairs /Volumes/VertigoDataTier/pact/scorer_region_cascade_b7106_campaign_20260531T214623Z/nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch/p19_posenet_null_pairs.json \
  --hprc-rate-collapse-p18-segnet-region-waterfill /Volumes/VertigoDataTier/pact/scorer_region_cascade_b7106_campaign_20260531T214623Z/nf0_05_r2_p12_rp1_rgb__1__1__1_cffec10_adaptive_blend_p11_then_p15_then_receiver_patch/p18_segnet_region_waterfill.json \
  --hprc-rate-collapse-importance-selection-domain eligible_low \
  --hprc-rate-collapse-waterfill-low-spec dz0_qd10 \
  --hprc-rate-collapse-waterfill-high-spec dz0_qd1 \
  --hprc-rate-collapse-waterfill-coarsen-quantile 0.25 \
  --enable-hprc-mlx-prefilter-before-local-replay \
  --hprc-mlx-prefilter-scorer-batch-pairs 1 \
  --hprc-mlx-prefilter-max-score-for-local-replay 0.5 \
  --auth-frontier-score 0.19198533626623068 \
  --local-baseline-score 0.19198533626623068 \
  --min-local-improvement 0.00002 \
  --timeout-seconds 7200
```

Then validate and execute bounded workers:

```
.venv/bin/python tools/experiment_queue.py \
  --queue .omx/research/hprc_z8_mamba_dreamer_full_stack_v1_queue.json validate

.venv/bin/python tools/experiment_queue.py \
  --queue .omx/research/hprc_z8_mamba_dreamer_full_stack_v1_queue.json run-worker \
  --execute \
  --max-steps 1 \
  --max-parallel 1
```

### Phase 2: Z8 residual teacher binding

The follow-up in the corrected HPRC run is blocked by missing Z8 archive/surface inputs:

- `z8_source_archive_bin_not_provided_or_missing`
- `z8_full_video_p18_p19_surface_or_reference_pairs_missing`

The next runnable plan must provide:

```
--z8-archive-bin <candidate-bound-Z8-0.bin>
--z8-surface <full-video-candidate-bound-z8-p18-p19-surface.npz>
--z8-reference-pairs-npy <full600-reference-pairs.npy>
```

Do not use prefix-projected surfaces for authority. They can create a search candidate,
but exact gating must remain blocked until the surface is full-video and candidate-bound.

### Phase 3: Mamba/Dreamer distillation

Only train Mamba/Dreamer/HPC stack members as priors or allocators until their outputs
are distilled into archive-bound HPRC sections. A valid distillation run must emit:

- model/config SHA-256,
- training command and seed,
- exported section bytes/SHA-256,
- runtime proof showing the numpy receiver consumes the distilled sections,
- comparison against the no-Mamba/no-Dreamer HPRC baseline at the same archive bytes.

No hidden state, Python pickle, external checkpoint, or sidecar table may be required.

## Replay And Exact Gate Policy

Use three gates:

1. Archive-rate gate:
   - Run before local replay.
   - Candidate must leave distortion headroom after rate:

     ```
     archive_rate_term + distortion_reserve < auth_frontier_score
     ```

   - Default distortion reserve remains `0.04` unless the parent lane writes a stronger
     empirical reserve.

2. Full-video local gates:
   - MLX full-video prefilter first, advisory only.
   - Local CPU replay only if MLX is below hard demote threshold and archive rate leaves
     headroom.
   - Local CPU replay must be full600 and singleton-safe.
   - Any local score above target writes a blocker and does not dispatch exact auth.

3. Exact auth gate:
   - Dispatch exact CPU/CUDA only if local replay beats the matching target by
     `min_local_improvement`, receiver proof is present, runtime custody is complete,
     and blockers are empty.
   - Before any provider job:

     ```
     .venv/bin/python tools/claim_lane_dispatch.py claim \
       --lane-id hprc_z8_mamba_dreamer_full_stack_v1 \
       --instance <provider-or-local-job-id> \
       --status active \
       --notes "contest exact eval candidate after full600 local replay"
     ```

   - Terminal claim row required on success, fail, stop, or refused dispatch.
   - Exact CPU and exact CUDA are separate axes. Do not infer one from the other.

## Blockers

1. Current HPRC compact receiver is pose-bad after rate collapse: latest corrected
   full600 local CPU advisory score `25.59201554157985`, PoseNet `35.64191818`.
2. Existing P18/P19 artifacts are compatible proxy priors but not HPRC-candidate-bound;
   rate collapse records `p19_source_archive_is_proxy_not_hprc_source_archive`.
3. Z8 full-video residual binding is blocked by missing candidate-bound Z8 archive bin,
   surface, and reference pairs.
4. MLX component profile lacks class-region/boundary logits cache, so byte-value pricing
   by class/boundary is advisory until the cache is extended.
5. Mamba/Dreamer/HPC stack members are not yet archive-bound runtime sections in HPRC;
   they remain training priors until distilled/exported with receiver proof.
6. Normal lane registration was not performed because this turn was constrained to one
   `.omx/research/` memo. Execution must register the lane first.

## Recommended Next Action

Do not exact-dispatch current HPRC rows. The next frontier-moving action is a small
executable closure, not more prose:

1. Register `hprc_z8_mamba_dreamer_full_stack_v1`.
2. Extend the full-video MLX cache to emit pose-axis and boundary/class rows that the
   byte pricer can consume.
3. Rebuild candidate-bound P18/P19 for the HPRC receiver after each accepted checkpoint.
4. Launch the bounded 32/128/600 HPRC long-training queue with the pose stop rule.
5. Bind Z8 residual tokens into `residual_rc` only when the per-cell proof satisfies
   `delta_nonrate + rate_cost < 0`.

Until then, the durable verdict is:

```
HPRC/Z8 long training is promising only as a pose-guarded, archive-bound residual
pricing system. Current HPRC rows prove byte compression and receiver plumbing, but
they also prove that pose can collapse badly if residual rate pressure outruns
candidate-bound PoseNet measurement.
```
