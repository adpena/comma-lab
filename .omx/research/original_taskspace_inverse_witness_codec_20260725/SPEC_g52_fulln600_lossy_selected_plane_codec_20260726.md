# G52 full-n600 lossy selected-plane codec diagnostic

Date: 2026-07-26  
Lane: `lane_g52_fulln600_lossy_selected_plane_codec_20260726`  
Authority: `[macOS-CPU research-signal]`, `research_only=true`

## Objective and non-authority boundary

Run the shortest production-grade full-n600 experiment that can falsify or
retain a selected-preimage temporal codec at the current competitive byte
scale. The source is the historical C1 two-plane packet **only as an
encoder-side mechanism oracle**. Its arrays, bitstreams, hashes, distortions,
and descendants are forbidden candidate lineage. This experiment cannot move
the frontier and cannot support a score claim.

The distortion-independent diagnostic ceiling is `258312` total stored bytes,
the byte scale at which the rate term alone reaches the current upstream
`0.172` pointer. Crossing it without a negative distortion term is a strict
rate falsifier for that endpoint. The stored artifact is deliberately named
`diagnostic_bundle.zip`, never `archive.zip`.

## Typed experiment

Exactly 600 ordered pairs are partitioned into five immutable, resumable
segments:

`[0,120), [120,240), [240,360), [360,480), [480,600)`.

Each arm is encoded at the same declared endpoint:

1. `DIRECT_INTERLEAVED_RGB`: stream order
   `Y0[0],Y1[0],Y0[1],Y1[1],...`. This exposes temporal coding both within and
   across pairs.
2. `TASK_LAYERED`: a temporal `Y1` base stream plus a true conditional
   enhancement
   `E = clip(round((Y0-Y1)/2)+128, 0, 255)`. Decode is
   `Y0_hat = clip(Y1_hat + 2*(E_hat-128), 0, 255)`.

The conditional quantizer is a typed, generic decoder mechanism. It is not
claimed optimal; it is a full-scale structural comparison against direct
interleaving.

The initial endpoint uses deterministic single-thread SVT-AV1 in an IVF
container with explicit raw RGB input, `yuv420p` encoded pixel format,
full-range BT.709 declarations, fixed frame rate, fixed segment GOP, and
declared bitrate per stream. A higher endpoint is allowed only after the low
endpoint closes and only if wall time permits.

## Custody and resume contract

- The source archive and member must match exact allowlisted SHA-256 values.
- The existing strict V10 packet parser and two-plane decoder reopen C1.
- Every source and reconstructed plane is shape/order checked and hashed.
- Every encoded bitstream is atomically published and SHA-256 bound.
- Every bitstream is decoded twice; frame count, shape, order, and byte hashes
  must match exactly.
- Each segment writes a canonical, immutable receipt only after all streams
  close. Resume adopts it only when config identity and every file hash agree.
- Aggregation refuses anything other than the five exact contiguous segments.
- The final diagnostic bundle uses `ZIP_STORED`; its exact size/hash and
  separate/interleaved byte accounting are authoritative only for rate.
- True scratch lives under the selected SSD run directory and is removed only
  after successful stage closure. Preserved bitstreams and receipts remain.
- Full launch is permitted only through `tools/safe_run.py`, after SSD storage
  preflight and with bounded process-group RSS and wall time.

## Success and falsifiers

Success is a real encode/rate/decode receipt at n600, not a score:

- `DIRECT_INTERLEAVED_RGB` or `TASK_LAYERED` produces a deterministic decoded
  plane state and an exact stored bundle at or below 258,312 bytes; or
- both honestly fail, locating the rate wall at this codec/formulation.

Strict falsifiers:

- any endpoint bundle exceeds 258,312 bytes before scorer distortion is known;
- non-identical double decode;
- missing/extra/reordered frames or shape drift;
- any source/hash/config mismatch;
- incomplete segment coverage;
- an undeclared encoder or color/GOP/rate-control setting.

## Triality and six operational hooks

**DSL:** typed JSON config, two named arm transforms, fixed segment lattice,
strict source/candidate-lineage declarations.

**DAG:** source SHA closure → V10 parse/decode → five independent encode/decode
stages → immutable receipts → exact aggregate bundle and rate verdict.

**Equations:** conditional enhancement and inverse above; endpoint rate
falsifier `B_bundle <= 258312`.

1. Exact-eval: intentionally not run; historical lineage forbids promotion.
2. Serialization: canonical JSON receipts and `ZIP_STORED` bundle.
3. Reload: source packet and every emitted stream are reopened from bytes.
4. `uint8`: both transforms and reconstructions are explicit `uint8`.
5. Activation: not applicable to a post-solve codec diagnostic.
6. Curriculum: not applicable; endpoint bitrate is typed and measured, not a
   training schedule.

Pointer effect: **none by construction**. A retained family must next consume
fresh current own-lineage selected-preimage planes and close exact
`upstream/evaluate.py` at n600.

## Fresh production successor amendment

The historical diagnostic is closed and must not be extended into a candidate.
Its production successor is
`src/tac/witness_dsl/taskspace_fresh_selected_plane_codec_v1.py`.

The successor has a strict injected `FreshOperandProviderV1` protocol and no
historical source field, archive loader, or fallback. Production validation
requires 600 chronological pairs in five 120-pair stages and rejects known C1
payload identities and historical input paths. Freshness is a property of the
input custody and derivation; output hash novelty is not required.

The truthful implemented representation is `DIRECT_TASK_LAYERED`: fresh `Y1`
plus the centered q2 `Y0|Y1` enhancement. It does **not** establish V15
semantic composition. `PROGRAM_RESIDUAL_LAYERED` remains a distinct missing
type until a fresh producer supplies the actual semantic predictor/base bytes;
every stage, final, bundle, and aggregate receipt carries that blocker and
sets `v15_composition_claim=false`.

Five independently encoded checkpoints remain immutable and resumable. Only
after all five close, the final recode reopens the provider and encodes two
chronological 600-frame, whole-population streams. The ten stage streams are
not included in the counted bundle. A closed final resume rehashes both long
streams and the deterministic bundle without deleting or rewriting stage
artifacts.

FFmpeg is encoder-side only. Public parse-back authority is PyAV:
`av.open -> container.decode(video=0) ->
VideoFrame.to_ndarray(format=rgb24)`, single-threaded and double-decoded. The
bundle binds the PyAV and linked FFmpeg-library versions, both reconstructed
scorer-plane hashes, and the exact chronological camera-raw hash/byte count
obtained by applying the generic factor-2 disjoint integer realization to
those PyAV bytes. This removes an otherwise brittle dependency on a public
`ffmpeg` executable.

The exact PyAV pin is derived from and SHA-bound to `upstream/uv.lock`
(`av==17.0.0`); production refuses any other version. For x264rgb the native
decoded format must be `gbrp`, and public RGB is assembled by direct G/B/R
plane extraction and reorder, bypassing conversion-sensitive libswscale.
Requested encoder/pixel-format flags and actual PyAV codec/native-format
observations are separate receipt fields. The raw contract names and hashes
the exact shared
`tac.optimization.uint8_lattice_feasibility.realize_factor2_uint8_scorer_plane`
helper and fixes public order to ascending pair id, frame0 from `Y0`, then
frame1 from `Y1`.

The production race is restricted to chroma-faithful explicit contracts:
`libx264rgb/rgb24/H.264` and `libx265/yuv444p/HEVC`. The live competitive target
is loaded and reverified through
`tac.witness_dsl.dynamic_frontier_target`; no literal score target is admitted
to the config or receipt.
