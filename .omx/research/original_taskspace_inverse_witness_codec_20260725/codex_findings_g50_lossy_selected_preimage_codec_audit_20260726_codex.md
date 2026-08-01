# G50 findings — lossy selected-preimage codec audit

Date: 2026-07-26  
Lane: `lane_g50_lossy_selected_preimage_codec_audit_20260726`  
Mode: `research_only=true`; read-only audit plus experiment specification; no
heavy run, archive build, exact evaluation, dispatch, or pointer mutation  
Repository HEAD observed: `0058123af31779d83d1fc10a728389b0ce7823ec`  
Pointer observed: official-leaderboard effective frontier `0.172`, pointer
SHA-256 `2a61b052be496d3a9a1be1a9c230c8d179a788e61fd03472e50fc85832da94c6`

## Verdict

The proposed family is **not closed**.

The repository has:

1. tested exact/lossless codecs over the historical C1 two-scorer-plane state;
2. tested one deliberately lossy family over a selected camera-preimage
   residual on 24 pairs; and
3. tested conventional lossy inter-frame codecs on source/keyframe streams.

It has **not** tested the missing composition:

> a full-n600, counted, receiver-closed conventional or hybrid lossy stream
> over a fresh selected-preimage solution, directly or as a residual/quotient
> against the current compact program, followed by the real V10 factor-2
> realization and upstream batch-16 exact scoring.

No repository or SSD evidence located by this audit contains AV1, HEVC, H.264
RGB, VP9, or lossy JXL applied to the two C1/MS1 scorer-plane streams. The V10
donor-codec race was lossless only. The one lossy selected-preimage experiment
used raw signed range-coordinate numerators with Brotli/zstd, not a transform,
motion-compensated, inter-frame codec, and was neither n600 nor an archive.

The structural eureka is not “try MPEG on another RGB video.” It is to use a
modern codec as a **learned-free projection and entropy engine inside the
task-space witness**:

```text
fresh batch-16 target custody
  -> fresh selected-preimage program
  -> Y1 semantic base layer
  -> Y0 | Y1 pose enhancement layer
  -> motion/transform/quantization/entropy coding
  -> deterministic generic decode
  -> V10 exact factor-2 realization R
  -> upstream/evaluate.py, batch 16, n600
```

This layout follows the evaluator rather than treating both frames and all
channels symmetrically: Seg scores only `Y1`; Pose scores `(Y0,Y1)`. A single
interleaved video is a useful control, but the scorer-native candidate is a
scalable two-layer stream:

- `Y1` is the semantic base stream, coded temporally across all pairs;
- `Y0 | Y1` is a pose-only enhancement, coded conditionally and admitted only
  when its exact nonlinear Pose contribution per byte pays.

That is the codec-shaped version of the selected-preimage program. It composes
motion prediction, transform sparsity, quantization, chroma allocation, entropy
contexts, and layered rate control instead of asking raw int32 residuals to
compress themselves.

## Authority correction: the old byte budgets are conditional

The precise historical MS1 diagnostic used batch geometry 32:

- `d_seg = 0.0001519690619574653`;
- `d_pose = 0.00010184327939026322`;
- distortion contribution `D = 0.04710980004607969`.

At exactly those components, strict integer ceilings are:

| target | archive ceiling if historical MS1 distortion survives |
|---|---:|
| `<0.172` | `187,562 B` |
| `<0.15` | `154,522 B` |

With a `133,941 B` V15-shaped base, that arithmetic leaves `53,621 B` and
`20,581 B`, respectively. These are useful scale indicators, **not current
candidate gates**.

The upstream evaluator defaults to batch 16. G46's fresh teacher audit found
three Seg target-cell differences between the fresh batch-16 bank and the
historical cache, and the historical precise MS1 diagnostic is explicitly
batch-32. The historical C1 archive did receive a contest-CPU batch-16
evaluation, but its report records only rounded components
`d_seg=0.00015196`, `d_pose=0.00010184`; it does not supply the precise
fresh-current selected-plane candidate components needed for a strict byte
ceiling.

Therefore:

- every selected-plane codec row must be replayed through exact batch-16
  `upstream/evaluate.py`;
- `187,562 B` is not a distortion-independent rejection threshold;
- the only distortion-independent strict rate impossibility thresholds are
  `258,312 B` for `<0.172` and `225,272 B` for `<0.15`, obtained from `D >= 0`;
- a row between `187,563 B` and `258,312 B` is not rejected without scoring;
- the V15 `53,621 B` residual allowance remains a counterfactual conditional
  on preserving the old precise MS1 distortion.

No fixed independent Seg/Pose/rate gates should be introduced. The rate
controller must minimize the coupled exact objective:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

## What was actually tested

### 1. Historical C1 exact/lossless plane coding — tested and rate-dead

The C1 state consists of two independent scorer-plane RGB arrays,
`Y0,Y1 : (600,384,512,3) uint8`. The V10 receiver realizes each through the
generic integer factor-2 construction to the required
`(600,2,874,1164,3) uint8` camera output.

The exact donor race in
`experiments/v10_ratecrush_rank_donor_coders.py` used the true temporal order
`Y0_0,Y1_0,Y0_1,Y1_1,...`, decoded every stream, and admitted it only on byte
equality. On n24:

| exact codec | bytes/pair | exact parse-back |
|---|---:|---|
| lossless JXL | `422,090.667` | yes |
| FFV1 | `442,848.542` | yes |
| lossless WebP | `443,857.000` | yes |
| x264rgb QP0 | `481,853.000` | yes |
| x265 RGB lossless | `529,761.417` | yes |

The accompanying phase-1 verdict extrapolated the best lossless JXL row to
`253,302,698 B` at n600. The exact C1 production archive itself is
`409,526,925 B`, SHA-256
`e4cd154f79a30e2b1d759af0d26e54444d22807f81700565e475392eae064f42`.

Its contest-CPU batch-16 closure measured:

- archive: `409,526,925 B`;
- `d_seg = 0.00015196`;
- `d_pose = 0.00010184`;
- exact canonical score `272.73427793588485`;
- inflate `215.805690679 s`;
- evaluation `175.190748548 s`;
- total `394.393399917 s`;
- output `3,662,409,600 B`, SHA-256
  `31d77be9ab9f00e9f814542368396a35ffa119a32571e701636d4747540e255b`.

This closes the **exact raster** formulation only. It does not test whether a
lossy video transform can move the selected state to a nearby representative
inside the same or a sufficiently cheap evaluator cell.

### 2. Historical selected-preimage lossy coarsening — tested only as n24 raw residual

The orphaned signal is the 2026-07-19 Seg-secant experiment:

- memo SHA-256
  `a8bf5f188c3342c89c9abb26b018ec3aa8ef1e9a71c34e103e0ed1427ac0f6e0`;
- composed receipt SHA-256
  `28940965904e9238668de6350785ef0e12348275b64fab83b22901726b0d1f85`;
- implementation SHA-256
  `ed4eef0a55b3d6a8bd507f657a02a7bb3c5f657c4302e997eb62a75f79c78bd6`;
- measurement tool SHA-256
  `1f1e45c92ee0eab8d4d92c9078e8b5cc6237328bfec0f60bdadf54ce9656ea4d`.

It started from a selected camera preimage and a generated-fill predictor,
then applied three real lossy families:

- native-margin ordered abandonment;
- low-bit truncation of signed camera residuals;
- fixed-grid residual subsampling plus bilinear reconstruction.

It counted a signed little-endian int32 range-numerator residual through
Brotli-Q11 and zstd-19 with exact parse-back and measured native CPU-Torch
Seg/Pose on 24 selected pairs:

| point | Brotli bytes/pair | `d_seg` | `d_pose` |
|---|---:|---:|---:|
| margin `0.3` | `2,214,597.46` | `3.9207e-5` | `8.4393e-7` |
| drop 1 low bit | `1,770,993.33` | `1.6276e-4` | `3.8683e-5` |
| drop 2 low bits | `1,313,066.92` | `1.8247e-4` | `7.5275e-5` |
| drop 3 low bits | `1,145,117.33` | `1.7399e-4` | `1.4155e-4` |
| stride 8 | `1,139,842.04` | `2.1362e-2` | `8.4976e-1` |
| stride 16 | `1,119,166.46` | `7.5493e-3` | `1.0206` |

This established that the selected-preimage face has a real lossy RD
direction. It also showed that this particular raw range-numerator
representation stays hundreds of megabytes to more than a gigabyte when
scaled, and spatial interpolation destroys Pose.

Its verdict scope is decisive:

- n24 selected historical-cache pairs, excluding unavailable VJP pairs;
- no archive ZIP;
- no public receiver/inflate closure;
- no full n600;
- no batch-16 current teacher;
- no conventional transform/motion codec.

Under the current n600-only rule this is historical advisory mechanism
evidence, not a current architecture verdict. It falsifies the tested raw
int32 residual forms, not lossy selected-preimage coding.

### 3. Conventional lossy coding elsewhere — reusable machinery, orthogonal evidence

`src/tac/boundary_math/keyframe_codec.py` already implements real
encode/decode for:

- x265 4:2:0;
- x265 4:4:4;
- SVT-AV1;
- VP9;
- arbitrary CRF, GOP, B-frame, and preset choices.

It is used for counted keyframe/pose-carrier research. It currently writes
temporary streams, returns byte counts and decoded frames, and deletes the
stream. It has no selected-preimage grammar, persistent resumable segments,
archive member, V10 realization closure, or public-inflate receipt. Its codec
argument construction is a useful generic mechanism; its keyframe numbers are
not C1 evidence.

Public upstream submissions and the source-video baselines prove that PyAV and
standard video bitstreams are accepted submission mechanisms. They supply no
payload bytes, parameters, or candidate state to this proposal.

## Negative search result

Targeted searches across `src/`, `tools/`, `experiments/`, and `.omx/research`
found no selected-plane use of a lossy `crf`, `qp`, `qscale`, AV1, HEVC,
H.264, VP9, or lossy JXL encoder. The selected-state hits reduce to:

- exact Brotli predictor/residual;
- exact lossless JXL per plane;
- exact lossless donor video coders;
- the n24 Seg-secant raw numerator coarsening above.

No full-n600 conventional/hybrid selected-preimage row or negative exists.
The correct family status is:

`FORMULATION_OPEN: FRESH_SELECTED_PREIMAGE_LOSSY_INTERFRAME_OR_HYBRID_CODEC`.

It must not be called proven, promising, dead, or frontier-ready until the
full experiment below is measured.

## The shortest honest full-n600 experiment

### Prerequisite and lawful input

Do not encode the historical C1/MS1 plane bytes into a new candidate. Those
bytes remain encoder-only existence/diagnostic evidence.

G49 is landing the fresh interface:

```python
decode_selected_preimage_pair(program, pair_index, decoder) -> (Y0, Y1)
realize_selected_preimage_pair_factor2(...)
```

It binds fresh batch-16 custody, rejects historical C1/V15 payload reuse, and
double-decodes deterministically. The codec runner should consume this
interface in pair order `0..599`, preferably through a bounded 120-pair
iterator. No dense n600 plane bank is required.

### Candidate arms

Run these as one preregistered full-population race, not disconnected toy
probes:

1. `DIRECT_INTERLEAVED_RGB`: one RGB-native stream in
   `Y0_0,Y1_0,Y0_1,Y1_1,...` order. This is the simplest whole-object control
   and exposes within-pair and cross-pair prediction.
2. `TASK_LAYERED`:
   - temporal `Y1_0,Y1_1,...` semantic base stream;
   - conditional `Y0|Y1` pose enhancement stream.
3. `PROGRAM_RESIDUAL_LAYERED`: the same two layers expressed as deterministic
   residuals against the fresh current counted program/P, with that base's
   exact archive bytes charged once. V15 historical bytes are not a base.

The direct and residual arms disambiguate a real unknown. A low-amplitude
residual can be lower entropy, but a modulo/signed residual can destroy spatial
structure that conventional transforms exploit. Do not assume either wins.

Start with RGB-native lossy H.264 (`libx264rgb`) because it avoids uncontrolled
4:2:0 chroma loss and the host has already exact-decoded the lossless form.
Race it against one modern 4:4:4 arm, x265 or AV1, with explicit pixel format,
range, and color-matrix metadata. Add 4:2:0 only as a separately named rate
arm; chroma is a real Seg and Pose actuator and must not disappear silently.
Lossy JXL is an intra control, not the primary route.

### Resumability, custody, and disk shape

Encode five immutable segments of 120 pairs each. Each segment consumes ten
existing/fresh 12-pair source checkpoints or the equivalent G49 callback
window and contains 240 interleaved frames. This gives:

- full n600 only;
- at most one segment of crash loss;
- five atomic per-stage checkpoints;
- bounded RAM;
- much lower header/IDR overhead than 50 twelve-pair streams.

Each segment receipt must preserve:

- program/archive/runtime SHA-256;
- fresh target receipt SHA-256 and batch geometry 16;
- pair range and source-plane double-decode hashes;
- exact encoder argv, executable/version, seed, thread count, pixel format,
  color range/matrix, GOP/B-frame policy;
- bitstream bytes/SHA-256;
- decoded scorer-plane bytes/SHA-256;
- codec/container overhead;
- stage timing and resume identity.

Use SSD preflight and success-only scratch cleanup. Keep encoded segments and
receipts durable. Assemble them into one strict archive grammar with a compact
index. The final archive size, not summed elementary payload estimates, is the
rate authority.

### Deterministic receiver

The public receiver should:

1. strict-parse the archive and segment index;
2. decode every counted standard bitstream;
3. require exactly 1,200 scorer-plane frames in the declared layout;
4. map them to exactly 600 typed `(Y0,Y1)` pairs;
5. apply the existing generic V10 factor-2 integer realization independently
   to each plane;
6. write exactly one `0.raw` of `3,662,409,600 B`;
7. preserve per-segment/pair checkpoints and output hashes;
8. fail closed on frame count, geometry, dtype, pixel format, hash, parse,
   resume, or output-length drift.

Decode twice before score admission. Receiver closure is not a private Python
call; it is the submitted `inflate.sh`/`inflate.py` path recursively invoked by
`upstream/evaluate.sh`.

### Nonarbitrary rate controller

Do not establish one fixed CRF, Seg threshold, Pose threshold, or bytes target.
For each arm:

1. materialize a high-rate and low-rate full-n600 endpoint;
2. count the exact archive;
3. reject only above the distortion-independent target ceiling if no negative
   distortion term exists (`258,312 B` for `<0.172`);
4. exact-score admissible endpoints at batch 16;
5. choose the next rate by measured score secant / whole-object Lagrangian,
   not equal CRF steps;
6. stop when adjacent measured points bracket the score minimum or all
   reachable points have a distortion-only score above `0.172`.

The controller should expose per-layer byte deltas and:

- `Delta d_seg`;
- `Delta sqrt(10*d_pose)`, not only `Delta d_pose`;
- `Delta archive bytes`;
- `Delta S`;
- value per byte for `Y1`, `Y0|Y1`, chroma, residual, and escape sections.

An enhancement layer is admitted only if its complete exact score improves.

### Exact end condition

The first row is scientifically complete only when it has:

- fresh current own-lineage selected-preimage input;
- all 600 pairs;
- strict archive parse-back;
- public receiver output closure;
- exact archive bytes/SHA-256;
- contest-CPU and/or contest-CUDA `upstream/evaluate.py` batch-16 result;
- component distortions, runtime, hardware, dependency closure, logs, and
  pointer comparison.

Only a score below the dynamic effective frontier moves the pointer.

## Decode legality and runtime closure

The contest explicitly permits external libraries and tools; video-derived
bitstreams remain counted. The generic decoder and factor-2 realization are
free code.

Evidence:

- `upstream/pyproject.toml` includes hard dependencies `av` and `pillow`;
- many upstream submissions import `av` in `inflate.py`;
- the upstream README setup installs `ffmpeg`;
- the harvested contest-CPU C1 provenance recorded Debian FFmpeg
  `5.1.8-0+deb12u1`, configured with `libaom`, `libdav1d`, `libsvtav1`,
  `libvpx`, `libx264`, `libx265`, `libjxl`, and `libwebp`.

Receiver choice:

- **PyAV is the best dependency-closed container/bitstream decoder** because it
  is in the authoritative upstream environment.
- **FFmpeg CLI is a valid fallback and encoder-side tool**, but the receiver
  should not depend on ambient PATH without a public closure receipt.
- `imagecodecs` is not in the root or upstream dependency lock; the local JXL
  wheel compliance test does not establish public runtime closure.
- no in-code HEVC/AV1 decoder is warranted. The in-code work should be the
  strict packet/index parser, deterministic plane extraction/color handling,
  layered residual reconstruction, and factor-2 realization.

For deterministic cross-host output, do not hide a version-sensitive implicit
YUV-to-RGB conversion behind an image iterator. Prefer RGB-native bitstreams
where competitive. For YUV arms, declare range/matrix/pixel format, extract
decoded planes, and use a fixed generic integer conversion/upsampling path in
the receiver; then require double-decode output identity on local and contest
hosts. `frame_utils.yuv420_to_rgb` is suitable only for the exact layout it
names and must not silently decode 4:4:4.

## Risks and decisive cures

| risk | why it matters | decisive cure |
|---|---|---|
| selected planes have weak temporal coherence | each C1 plane was solved independently | race interleaved, `Y1` temporal, and program-residual layouts |
| codec spends on frame0 Seg-irrelevant texture | only Y1 carries Seg debt | layered `Y1` base plus pose-only enhancement |
| 4:2:0 destroys task chroma | chroma affects both scorers | 4:4:4/RGB primary; 4:2:0 separately typed |
| historical plane bytes leak into candidate | violates current own-lineage contract | consume only G49 fresh batch-16 callback |
| old `53,621 B` treated as a hard budget | based on precise batch32 MS1 D | exact batch16 score every full row |
| segment resets waste the tiny budget | five IDRs/SPS/PPS are not free | raw elementary segments, compact index, measure exact archive |
| decoder conversion differs by host | can flip boundary cells | fixed pixel/color contract plus double-decode hashes |
| a tiny stream is mistaken for progress | distortion may already exceed target at zero bytes | score full n600 through R before any claim |

## Forest-level conclusion

The exact/lossless C1 work proved that a very low-distortion selected solution
exists and that naive raster custody is catastrophically expensive. The
Seg-secant work proved that lossy movement within the selected-preimage family
exists, but represented it in the wrong coordinate: dense signed range
numerators. The missing bridge is to let the codec carry the **solution
quotient**, not the original problem or an exact raster:

- analytic/current program as decoder side information;
- scorer-asymmetric layered selected-preimage corrections;
- motion/transform/quantization/entropy coding for the irreducible quotient;
- exact inverse realization and evaluator closure.

This is not proven to fit below roughly a quarter megabyte. It is, however, a
real untested structural family directly aligned with the user’s “contest
codec” framing, and G49's fresh pair decoder seam removes the main integration
excuse. The next unit should build and score this full-n600 receiver-closed
race, not perform another local subset characterization.

Pointer delta from G50: **UNMOVED**. This audit found and scoped a real open
family; it did not lower the exact score.

## Evidence index

- `.omx/research/seg_secant_rd_curve_20260719_codex.md`
- `.omx/research/seg_secant_rd_curve_n24_20260719_v2.json`
- `src/tac/optimization/seg_secant_rd_curve.py`
- `tools/measure_seg_secant_rd_curve.py`
- `.omx/research/v10_ratecrush_phase1_20260719.md`
- `experiments/v10_ratecrush_rank_donor_coders.py`
- `experiments/v10_ratecrush_build_jxl_archive.py`
- `src/tac/codec/v10_predictor_residual.py`
- `src/tac/codec/v10_jxl_plane_codec.py`
- `src/tac/witness_dsl/v10_production_receiver.py`
- `src/tac/witness_dsl/v10_two_plane_timing_receiver.py`
- `src/tac/boundary_math/keyframe_codec.py`
- `upstream/evaluate.py`
- `upstream/pyproject.toml`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/codex_findings_g44_fresh_teacher_source_custody_20260726_codex.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/v15_ms1_coordinate_compatibility_receipt_20260726.json`
- `/Volumes/VertigoDataTier/pact/evidence/v10_ratecrush_20260719/rank_donor_coders_n24.json`
- `/Volumes/VertigoDataTier/pact/evidence/v10_ratecrush_20260719/r0a_jxl_n24_verify_e9.json`
- `/Volumes/VertigoDataTier/pact/evidence/v10_ratecrush_20260719/r0b_pip_decoder_compliance.json`
- `/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719/modal_contest_cpu/harvest_fc01KXXRAR/contest_auth_eval.json`

