# C067 Byte/Self-Compression Opportunity Review - 2026-05-03 Codex

Evidence grade: `empirical_byte_review` and `planning_only`.

Score claim: false. Promotion eligible: false. Remote jobs dispatched: none.

## Scope And Constraint

This memo reviews C067 archive byte decomposition for implementation
opportunities that could plausibly save at least `23,500` charged bytes without
changing decoded mask geometry. Under this constraint, the preferred path is to
keep the current PR67/C067 `masks.mkv` stream byte-identical. A mask-codec
replacement is in scope only if it proves the same decoded mask tensor, including
tensor SHA `0344fcfc39e683f21a71db1085a8697a94c4606f91f883362e9acc02fc7b5b45`,
and remains smaller after all decoder/runtime bytes are charged.

That exact-lossless mask replacement is not currently available. The measured
lossless probes are byte-regressive versus the C067/PR67 charged mask stream, so
the top no-mask-geometry opportunities are renderer self-compression
implementations.

## C067 Anchor

Current C067 exact archive:

- Archive:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Archive bytes: `276214`
- Archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- Exact CUDA/T4 adjudicated recomputed score: `0.31561703078448233`
- Components: PoseNet `0.00049637`, SegNet `0.00061244`, samples `600`
- Source attribution:
  `masks.mkv` from public PR67, `renderer.bin` and `optimized_poses.bin` from
  internal C-059/C067 bridge.

Byte decomposition from
`experiments/results/c067_archive_byte_accounting_20260502/archive_byte_accounting_refreshed.json`:

| Stream | Charged bytes | Codec | Notes |
| --- | ---: | --- | --- |
| `masks.mkv` | `219472` | `brotli_av1_obu` | Largest stream; generic nested recompression saves `0` bytes. |
| `renderer.bin` | `55965` | `brotli_qzs3` | Only stream besides masks large enough to close the unchanged-distortion gap. |
| `optimized_poses.bin` | `677` | `public_qp1_brotli` | Too small to matter for the `23,500` byte target. |
| ZIP/container overhead | `100` | stored single member `p` | Already near irreducible. |

At unchanged distortion, C067 needs `23454` fewer archive bytes to reach
sub-`0.300`; the buffered planning target is `23455` bytes, or archive size
`252759`.

Renderer-only implication: the charged renderer stream must drop from `55965`
bytes to roughly `32511` bytes or less if the mask and pose streams are held
fixed.

## Reverse-Engineering Signals

Public PR67 uses the same fixed-slice order that C067 inherits: `masks.mkv`,
`renderer.bin`, then `optimized_poses.bin`. The PR67 mask segment is already
`219472` charged bytes and QZS3 renderer decoding validates to `111` finite
tensors with `87836` parameters. C067 improves PR67 by replacing the renderer
and pose slices while preserving the PR67 mask stream.

Public PR65 shows a larger `x` member with a `219472` byte mask stream and
extra post/side streams. Existing qpost and pair-gated PR65 transfers were
exact-screened negative or byte-additive, so PR65 is useful as anatomy and
distortion-basin evidence, not as a direct byte-saving packer.

Selfcomp is the most relevant public renderer clue. The local reverse-
engineering refresh records PR56 Selfcomp reporting about `1.017` bits per
weight on an approximately `88K` parameter renderer. If that density can be
made contest-runtime legal for the C067 renderer basin, the nominal weight
payload is about `11168` bytes before headers/metadata, leaving enough room to
clear the `23455` byte target even after decoder and manifest overhead.

## Top 3 Opportunities

### 1. Learned Bit-Depth Renderer Self-Compression

Opportunity: train/export a C067-compatible self-compressed renderer using the
existing `tac.self_compress` machinery or a minimal successor codec. Keep
`masks.mkv` and `optimized_poses.bin` unchanged; replace only `renderer.bin`
with a charged, deterministic, pickle-free self-compressed payload and loader.

Why this can plausibly save `>=23500` bytes:

- Current encoded renderer stream: `55965` bytes.
- Target renderer stream if it alone closes the gap: `<=32511` bytes.
- Public/selfcomp signal: `87836` weights at `1.017` bits/weight is about
  `11168` raw weight bytes before metadata.
- Even a `16KB` to `24KB` renderer payload would save roughly `32KB` to `40KB`
  versus the current renderer stream.

Required implementation shape:

- Start from the C067 QZS3 renderer state and fixed C067 mask/pose streams.
- Learn per-channel or per-tensor bit depth with explicit rate loss, not a
  post-hoc generic compressor.
- Preserve `eval_roundtrip=True` and do not load scorers at inflate time.
- Export a small deterministic runtime contract with all score-affecting bytes
  charged inside `archive.zip`.
- Gate locally on archive bytes, loader parity, absence of sidecars, and fixed
  mask/pose stream SHA before any exact CUDA eval is considered.

Primary risks:

- Self-compression must be trained in the C067 scorer basin. A small byte
  payload alone is not evidence.
- Runtime metadata can erase the byte win if the export is not deliberately
  minimal.
- Any Torch-pickle or scorer-side dependency at inflate is a compliance bug.

### 2. Trained IMP/Sparse Renderer Recovery

Opportunity: use the existing IMP/sparsity bridge as a byte-feasible prior, but
replace the no-train pruning candidates with a real prune-recover training loop
and a charged sparse or sparse-aware QZS/MQZ export.

Why this can plausibly save `>=23500` bytes:

- The no-train IMP bridge already produced a local archive byte screen at
  `244623` bytes for `imp_c10_qzs3_b0128`, saving `31591` bytes versus C067.
- That candidate had actual sparsity `0.8927915398341556`.
- Cycle 5 is `20021` bytes saved and cycle 10 is `31591` bytes saved, so the
  byte target lies inside the observed sparsity/packing curve.

Why the existing candidate is not evidence:

- Exact diagnostic eval of `imp_c10_qzs3_b0128` scored
  `78.9903710766749` with PoseNet `79.82865143` and SegNet `0.50573522`.
- Exact diagnostic eval of lower-cycle IMP candidates also collapsed as
  sparsity increased.
- Therefore the opportunity is trained sparse recovery, not dispatching or
  promoting no-train IMP.

Required implementation shape:

- Use IMP masks only as initialization/ranking, then train with the C067
  mask/pose streams fixed.
- Record `training_applied=true`, sparsity masks, tensor groups, exported
  bytes, and exact loader provenance.
- Prefer sparse exports only if the runtime contract charges index metadata
  and beats a dense learned-bit-depth export.
- Local acceptance target before exact eval: archive `<=252759`, unchanged
  mask segment SHA, unchanged pose segment SHA, and renderer-only runtime
  parity smoke.

Primary risks:

- High sparsity destroys PoseNet quickly without recovery training.
- Sparse indices can dominate bytes at moderate sparsity.
- The bridge currently saves many bytes from packer/stream effects but only
  `2988` bytes from the raw renderer slice, so a real sparse format must beat
  existing QZS3 end to end, not only in an uncompressed proxy.

### 3. Slim/Frozen-Mask Teacher-Student JFG

Opportunity: train a smaller or factorized JointFrameGenerator against the
fixed C067/PR67 mask geometry and C067 pose stream, then export it through the
existing QZS3-compatible path or a small deterministic successor. This changes
renderer architecture/parameterization, not mask geometry.

Why this can plausibly save `>=23500` bytes:

- C067's renderer has `87836` parameters and an encoded stream of `55965`
  bytes.
- A student with about `55%` to `60%` of the current effective parameters, at a
  similar compressed bits-per-parameter, is in the `30KB` to `34KB` renderer
  range before small headers.
- Unlike scalar QZS3 reblocking, architecture shrink can be trained to preserve
  the C067 output basin rather than only repacking an already-fit weight set.

Required implementation shape:

- Keep `masks.mkv` and `optimized_poses.bin` unchanged.
- Distill from the C067 renderer on the exact contest window and record
  architecture, parameter count, seeds, losses, and export SHA.
- Prefer factorized layers, narrower hidden channels, grouped/depthwise
  replacements, or low-rank heads only when a deterministic export exists.
- Use local image/video parity and component-trace proxy only as screening;
  promotion still requires exact CUDA auth eval after a dispatch claim.

Primary risks:

- Prior Q-FAITHFUL and scalar QZS block changes show that small renderer
  perturbations can collapse PoseNet.
- A student that preserves visible reconstruction can still leave the scorer
  basin.
- If the architecture requires extra runtime code, those bytes must be charged
  and can consume the savings.

## Rejected Or Lower-Priority Paths

- Exact lossless decoded-mask recoding: rejected for this target today. AMRC
  exact-lossless mask bytes were `593417` versus source mask raw bytes `223385`.
  CMG2 lossless probes found best `raw_u8_bz2_9` at `340315`, `+120843` bytes
  versus the current `219472` charged mask stream.
- PMG/CMG row-span or downsampled mask geometry: out of scope for this memo
  because it changes decoded mask geometry, and exact T4 evidence showed
  PoseNet/SegNet collapse despite large byte wins.
- Generic nested compression: exhausted. Encoded stream recompression probes
  show `0` deployable savings on mask, renderer, and pose streams.
- ZIP/container changes: at most around the existing `100` byte overhead, far
  below target.
- Pose-only compression: current pose stream is `677` bytes, so it cannot close
  a `23455` byte gap.
- SJ-KL residual payloads: useful distortion-side work, but recent C067
  sibling candidates were byte-additive, not byte-saving.
- PR65 qpost/extra stream transfer: public anatomy signal only; direct
  transfers are byte-additive or exact-negative.

## Concrete Next Commands

Refresh the byte-accounting surface with the current archive and exact JSON:

```bash
.venv/bin/python experiments/profile_archive_byte_accounting.py \
  --archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --eval-json experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json \
  --output-json experiments/results/c067_archive_byte_accounting_20260503_review/archive_byte_accounting.json \
  --output-md experiments/results/c067_archive_byte_accounting_20260503_review/archive_byte_accounting.md \
  --target-score 0.300
```

Refresh the deterministic renderer self-compression planning report, including
the known exact-negative QZS3 evidence:

```bash
.venv/bin/python experiments/plan_c067_renderer_self_compression_v2.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --output-json experiments/results/c067_renderer_self_compression_v2_20260503_review/plan.json \
  --exact-negative-evidence-json experiments/results/lightning_batch/exact_eval_c067_qzs3_b512_l40sdiag_20260502T1710Z/contest_auth_eval.adjudicated.json
```

Rebuild the local IMP byte-screen only after a trained export exists or when
checking deterministic bridge plumbing; do not treat output as scorer evidence:

```bash
.venv/bin/python experiments/build_imp_c067_bridge_candidates.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --output-dir experiments/results/imp_c067_bridge_candidates_20260503_review \
  --cycle-counts 1,2,5,10 \
  --qzs3-block-sizes 16,24,32,48,64,96,128 \
  --force
```

Scan for trained renderer exports already on disk and write the promotion-
unlock plan:

```bash
.venv/bin/python experiments/plan_trained_renderer_export_unlock.py \
  --output experiments/results/trained_renderer_export_unlock_20260503_review/plan.json
```

When a real trained renderer export exists, run local preflight only:

```bash
.venv/bin/python experiments/preflight_trained_renderer_transplant.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip \
  --renderer-export PATH/TO/TRAINED_RENDERER_EXPORT \
  --output-dir experiments/results/trained_renderer_transplant_preflight_20260503_review \
  --block-sizes 32,64,96,128,256,512 \
  --force
```

Before any future exact CUDA dispatch, claim the lane with
`tools/claim_lane_dispatch.py claim ...` per AGENTS.md. This memo does not
authorize dispatch.

## Decision

The highest-EV byte-only path that preserves decoded mask geometry is renderer
self-compression, not another mask-grammar byte cut. The implementation order is
therefore:

1. Learned bit-depth/self-compressed C067 renderer export.
2. Trained sparse/IMP recovery with a charged sparse-aware export.
3. Slim/factorized teacher-student JFG with fixed C067 mask and pose streams.

All three must be treated as planning-only until they produce deterministic
archives, pass local closure/preflight, and then receive exact CUDA auth eval
through `archive.zip -> inflate.sh -> upstream/evaluate.py` after a dispatch
claim.
