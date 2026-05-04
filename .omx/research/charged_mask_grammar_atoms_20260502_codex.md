# Charged Mask Grammar Atom Planner - 2026-05-02

Evidence grade: `empirical_planning_only_non_score`.
Score claim: `false`.
GPU jobs launched: `false`.
Retraining launched: `false`.

This ledger records a deterministic CPU-only atom-planning tool for strict
charged mask grammar work. It does not build a scoreable archive, does not
integrate an inflate decoder, and does not promote or rank any result.

## Artifact

Tool:

```text
experiments/plan_charged_mask_grammar_atoms.py
```

Focused result:

```text
experiments/results/charged_mask_grammar_atoms_20260502_codex/atom_plan_manifest.json
experiments/results/charged_mask_grammar_atoms_20260502_codex/extracted_mask_stream.bin
```

Input archive:

```text
reports/raw/leaderboard_intel_20260501/pr67_archive.zip
bytes    276564
sha256   a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765
```

Extracted mask stream:

```text
member              masks.mkv
decoded bytes       223385
decoded sha256      a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb
charged slice bytes 219472
charged slice sha   d1ae0d39c848e5715b74bb0122269066a4a1ab60ba9e12f34e70fd62ac136d87
```

## Evidence Boundary

The planner reuses the existing runtime helper:

```text
submissions/robust_current/unpack_renderer_payload.py
```

It emits:

- bounded frame-group atoms;
- ego/foveal region atoms;
- exact mask-stream chunk custody atoms;
- connected-component and RLE span atoms only when a decoded `(T,H,W)` mask
  array is supplied;
- non-promotable candidate policies for later charged archive builders.

No CUDA auth eval was run, and no score claim is made. The next promotion
boundary remains a closed archive evaluated through:

```text
archive.zip -> inflate.sh -> upstream/evaluate.py
experiments/contest_auth_eval.py --device cuda
```

## Next Integration Step

Decode the extracted PR67 mask stream to a deterministic `(600,384,512)` class
tensor through an auditable CPU decode path, rerun the planner with
`--mask-array`, and feed the resulting connected-component/RLE atoms into a
strict CMG builder that charges every grammar table, selector, codebook, and
residual inside `archive.zip`.

## Decoded PR67 Mask Contract And Entropy Probe - 2026-05-02T06:50Z

The decoded-mask integration is now available as a bounded planning result:

```text
result_dir=experiments/results/charged_mask_grammar_atoms_pr67_decoded_20260502_codex
decoded_array=/tmp/pact_cmg/pr67_decoded_mask_array_20260502T0632Z.npy
shape=600x384x512
dtype=uint8
class_range=0..4
sha256=13f81851ae206f6142cd3348db5537eda4ea832665052824816f4d1bf432de0c
atom_tables=stream_chunks,frame_groups,connected_components,rle_spans,class_regions,class_boundaries
score_claim=false
```

Naive class-mask compression is not enough:

```text
public_pr67_mask_stream_bytes=219472
raw_uint8_bz2_9=340315
raw_uint8_brotli11=418867
packed_3bit_brotli11=422543
row_rle_lzma9=385896
row_rle_brotli11=422441
row_rle_class_varint=1406567
```

Interpretation: PR67's mask stream is already well beyond simple
class-label/RLE/standard-compressor baselines. The next CMG experiment must be
learned, predictive, residual, foveated, or decoder-aware; a raw class-RLE
grammar will spend more bytes than the public-floor stream before scorer
benefits are even evaluated.

Additional entropy-bound probe:

```text
artifact=experiments/results/charged_mask_grammar_atoms_pr67_decoded_20260502_codex/mask_entropy_bounds.json
global_class_entropy_lower_bound=23820963 bytes
temporal_cond_entropy_curr_given_prev_lower_bound=1230082 bytes
x_left_cond_entropy_lower_bound=356093 bytes
y_up_cond_entropy_lower_bound=1124768 bytes
temporal_same_fraction=0.9875411435372974
mean_temporal_change_pixels_per_frame=2449.5108514190315
```

Even the most favorable simple first-order spatial bound is still above the
public stream by roughly `136621` bytes. That sharply lowers the EV of a
hand-coded raw class arithmetic coder and raises the EV of video-codec-style
prediction, learned latent masks, full-frame Q-FAITHFUL closure, or a hybrid
grammar that codes only sparse residuals against an existing decoder prior.
