# Contest Docs And Video Reverse Engineering - 2026-05-01

Evidence type: `external` for public docs/PRs, `empirical_video_anatomy` for
local source-video analysis, `prediction` for dispatch implications.

This note reviews the active contest docs and re-reverse-engineers the exact
video we are overfitting. It is a control-plane artifact only. It does not
rank, promote, kill, or retire any lane without exact CUDA archive evidence.

## Contest Contract Refresh

Official/public sources checked:

- Contest repo:
  `https://github.com/commaai/comma_video_compression_challenge`
- Openpilot logging docs:
  `https://docs.comma.ai/concepts/logs/`
- comma calibration challenge:
  `https://github.com/commaai/calib_challenge`
- comma2k19 paper:
  `https://arxiv.org/abs/1812.05752`

Local upstream snapshot:

- `upstream` remote:
  `https://github.com/commaai/comma_video_compression_challenge.git`
- local upstream HEAD:
  `11ad728f563d8970929e8947a1cf6124ee6303e4`
  (`11ad728 ci: update leaderboard tables`)

Contest facts reaffirmed:

- Source clip: `upstream/videos/0.mkv`, one minute, HEVC, `1164x874`,
  `20 fps`, `1200` frames, `37,545,489` bytes.
- Public eval list: only `0.mkv`.
- Scored samples: `600` two-frame samples.
- Score:
  `100*segnet_dist + sqrt(10*posenet_dist) + 25*archive_bytes/37545489`.
- PoseNet consumes two consecutive frames; SegNet consumes the last frame.
- Submission surface: public PR with `archive.zip` plus `inflate.sh`.
- Official GPU inflate: T4-class if the inflate path needs GPU, 30 minute
  budget.
- External libraries are allowed, but large artifacts that affect output must
  be charged inside the archive.

Current public leaderboard/document signal:

- `qpose14` PR #63: score `0.32`, archive `287,573` bytes, PoseNet
  `0.00052154`, SegNet `0.00061261`; described as a rate-optimized
  Quantizr-style neural renderer with quantized pose side-channel and
  single-member archive packing.
- `unified_brotli` PR #64: score `0.33`, archive `287,165` bytes, PoseNet
  `0.00061622`, SegNet `0.00061261`; described as Quantizr/qpose-derived
  single-stream brotli over raw mask/model/pose payloads, delta-encoded
  velocity, and dropped rotation.
- `quantizr` PR #55: score `0.33`, archive `299,970` bytes, PoseNet
  `0.00051328`, SegNet `0.00061261`.
- `fp4_mask_gen` PR #62: score `0.37`, archive `249,624` bytes, PoseNet
  `0.00076576`, SegNet `0.00121106`; described as AV1 CRF55 full-res
  5-class mask video, uint16 poses, and an FP4 depthwise-separable
  FiLM-conditioned neural generator.

Implication: sub-0.4 is publicly demonstrated by compact full-frame semantic
controls plus tiny charged neural renderers and aggressively packed pose
signals. Our C-044 score is not limited by theoretical possibility; it is
limited by not yet reproducing that representation/packing geometry under our
exact custody.

## Local Video Anatomy Artifact

Implemented:

- `experiments/reverse_engineer_contest_video.py`
- `src/tac/tests/test_reverse_engineer_contest_video.py`

Focused verification:

- `.venv/bin/python -m py_compile experiments/reverse_engineer_contest_video.py src/tac/tests/test_reverse_engineer_contest_video.py`
- `.venv/bin/python -m pytest src/tac/tests/test_reverse_engineer_contest_video.py -q`
- Result: `3 passed in 0.42s`.

Generated artifacts:

- `experiments/results/contest_video_reverse_engineering_20260501/contest_video_reverse_engineering.json`
  - SHA-256:
    `98b3baf69b9d039fed77d01244ded853ee808adebb469b2c8e00ce84b1394600`
  - Uses low-res `submissions/robust_current/masks.mkv` proxy; decoded mask
    shape `1200x48x64`. This is forensic only and must not drive foveation
    selectors.
- `experiments/results/contest_video_reverse_engineering_20260501/contest_video_reverse_engineering_fullres_masks.json`
  - SHA-256:
    `b852daeb64f2a950f961a3cb3d09cea3717a20cfab398f87444ae6087e749bd1`
  - Uses full-res `submissions/robust_current/masks_fullres.mkv`; decoded
    mask shape `1200x384x512`.

Source video facts from the artifact:

- Video SHA-256:
  `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`
- Size: `37,545,489` bytes.
- Stream: HEVC, `1164x874`, `yuv420p`, `20/1` fps.
- Decoded frames/pairs: `1200` frames, `600` pairs.
- Luma mean over frames:
  - mean `34.00110161622365`
  - std `2.3648488539428842`
  - min `30.106128692626953`
  - max `44.911746978759766`
- Pair mean absolute luma delta:
  - mean `2.3112798257668814`
  - std `0.4103515130284133`
  - max `6.0819993019104`
- Highest luma-motion pairs:
  - pair `523`, mean delta `6.0819993019104`,
    horizon delta `6.698453426361084`,
    radial inner-minus-outer `9.288901805877686`
  - pair `514`, mean delta `5.972390651702881`,
    horizon delta `7.092188835144043`,
    radial inner-minus-outer `5.770525932312012`
  - pair `515`, mean delta `4.573449611663818`,
    horizon delta `7.798402786254883`,
    radial inner-minus-outer `3.4384708404541016`
  - pair `516`, mean delta `4.440291881561279`,
    horizon delta `9.921575546264648`,
    radial inner-minus-outer `10.79887080192566`
  - pair `521`, mean delta `4.2121148109436035`,
    horizon delta `9.379375457763672`,
    radial inner-minus-outer `17.540607690811157`

Full-res mask proxy anatomy:

- `submissions/robust_current/masks_fullres.mkv`
  - bytes `2,016,204`
  - SHA-256:
    `cd092b918bdf2b95104ce68b1191507d2009065fce94fde3d4a16beaaa61a7a9`
  - class fractions:
    `[0.23236990610758462, 0.005847850375705295, 0.49518194834391277, 0.012359754774305556, 0.25424054039849175]`
  - lane-mark fraction: `0.005847850375705295`
- Lane-mark log-zoom summary:
  - mean `0.004242304301199814`
  - std `0.14425052012530618`
  - min `-0.4533148407936096`
  - max `0.4154999554157257`
- Zero-cost pose dim0 summary from full-res masks:
  - mean `31.328938446044923`
  - std `1.1540041177918048`
  - min `27.668481826782227`
  - max `34.61899948120117`
- Top pairs by absolute lane-mark log-zoom:
  `208`, `218`, `94`, `528`, `544`, `212`, `91`, `598`, `157`, `513`.

## Hardware And Ego-Motion Implications

Known geometry from local code and external comma docs:

- `upstream/frame_utils.py` uses `camera_size=(1164,874)`,
  `camera_fl=910`, `segnet_model_input_size=(512,384)`.
- `src/tac/camera.py` records native intrinsics
  `fx=fy=910`, `cx=582`, `cy=437`, and scorer-scale FoE/VP
  `(256,174)`.
- Native FoE from scorer scaling: approximately `(582,396)`.
- The calibration challenge independently states one-minute, 20 fps dashcam
  clips and a 910 px focal length estimate, with pitch/yaw camera alignment
  as the core openpilot geometry problem.
- Openpilot logging docs confirm one-minute route segments and HEVC camera
  streams.

Implications for overfitting:

1. Telescope/hyperbolic foveation should be centered by calibrated FoE and
   allowed to move only inside a small trust region unless exact response
   shows otherwise.
2. The hard PoseNet pairs should not be selected only by prior Lane W lists.
   Combine at least four priors:
   - exact per-pair PoseNet/SegNet deltas when available;
   - luma-motion top pairs around the FoE/horizon;
   - lane-mark log-zoom top pairs from full-res masks;
   - public-leaderboard pose-side-channel geometry: velocity dominates
     rotation at this archive scale.
3. Pose side-channel design should prefer velocity/log-zoom/delta-coded
   scalars before full 6-DOF storage, then exact-eval whether dropped rotation
   stays inside PoseNet basin.
4. Full-res 5-class mask semantics are still the dominant control signal.
   Low-res `48x64` masks are forensic only for foveation/ego-motion work.
5. Openpilot/supercombo belongs at compress time as a prior/teacher only.
   Any score-affecting output must be charged in the archive, and no
   uncharged model sidecar can run at inflate.

## Next Dispatch Decisions

Highest-EV allowed work under the Lane 12 retraining gate:

1. Build a `hardware_ego_foveation_probe_v1` planning artifact that joins:
   C-044 component baseline, full-res mask log-zoom pairs, luma-motion pairs,
   and exact component-manifold probes.
2. Generate concrete build-only variants that do not require new training:
   - pose side-channel packing: quantized velocity/log-zoom + small correction;
   - deterministic brotli/unified stream packing over existing payloads;
   - hyperbolic foveation parameter side-channel with identity-safe defaults
     for exact cliff probes;
   - mask full-res control stream variants that mimic the public
     Quantizr/qpose/fp4_mask_gen byte allocation.
3. Use H100/A100/4090 for iteration, then Lightning T4 only for A++
   confirmation.
4. Do not use the low-res `masks.mkv` proxy for geometry claims.
5. Treat every public PR and external doc as `external` evidence only until
   our own archive bytes pass exact CUDA auth eval.

## Live Top-Submission Reverse Engineering - 2026-05-01T17:14Z

Source refresh:

- Official comma.ai leaderboard:
  `https://comma.ai/leaderboard`
- Contest repo README and rules:
  `https://github.com/commaai/comma_video_compression_challenge`
- PRs inspected:
  `#63 qpose14`, `#64 unified_brotli`, `#55 quantizr`,
  `#62 fp4_mask_gen`, `#56 selfcomp`
- Branch-code clones inspected outside the repo under:
  `/tmp/pact_topsubs_dpEPDD/`

Current public lowest lossy-video scores:

| rank | name | public score | bytes | PoseNet | SegNet | score anatomy |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | `qpose14` | `0.32` | `287573` | `0.00052154` | `0.00061261` | `0.0613 seg + 0.0722 pose + 0.1915 rate` |
| 2 | `unified_brotli` | `0.33` | `287165` | `0.00061622` | `0.00061261` | `0.0613 seg + 0.0785 pose + 0.1912 rate` |
| 3 | `quantizr` | `0.33` | `299970` | `0.00051328` | `0.00061261` | `0.0613 seg + 0.0716 pose + 0.1997 rate` |
| 4 | `fp4_mask_gen` | `0.37` | `249624` | `0.00076576` | `0.00121106` | `0.1211 seg + 0.0875 pose + 0.1662 rate` |
| 5 | `selfcomp` | `0.38` | `279036` | `0.00055221` | `0.00122167` | `0.1222 seg + 0.0743 pose + 0.1858 rate` |

Evidence grade for this table is `external`: public official leaderboard and
public PR CUDA logs, not our local archive custody.

Implementation anatomy from cloned branch code:

- `qpose14`:
  - Quantizr-style renderer with one 5-class mask frame per pair.
  - Mask decoded from gray AV1 OBU by `round(gray / 63)` into class ids.
  - Depthwise-separable shared mask trunk plus two RGB heads.
  - Frame 2 is static from mask; frame 1 is FiLM-conditioned on pose.
  - FP4 packed weights, brotli-compressed.
  - Quantized pose stream: velocity-like dim0 stored as uint16/512 + 20;
    remaining dims stored as int16/2048.
  - Single zip member `p` with fixed slices:
    mask `219472` bytes, model `66841` bytes, remainder pose.
- `unified_brotli`:
  - Same renderer family as qpose14.
  - One zip member `p`, but now a single brotli stream over concatenated raw
    mask + model + pose payloads with a 12-byte length header.
  - Pose reduced to delta-coded velocity only:
    first uint16 plus 599 int16 deltas; rotation is set to zero.
  - Net effect: fewer bytes than qpose14, slightly worse PoseNet.
- `quantizr`:
  - Full compression script is public.
  - SegNet mask is only odd-frame/second-frame semantic control, so it stores
    half as many mask frames as a two-mask-per-pair scheme.
  - Renderer is trained through the same distortions used by evaluation:
    upscale, clamp/round to uint8, downscale to SegNet/PoseNet input, then
    loss against SegNet and PoseNet outputs.
  - Training shuffles frame pairs, stages anchor -> frame1 finetune -> joint,
    and exports FP4/brotli weights.
  - PR author notes sub-0.30 looked possible by architecture/conv-dim sweeps.
- `fp4_mask_gen`:
  - Same broad renderer family, but documented around FP4 quantized generator,
    AV1 CRF=55 mask stream, uint16 poses, and joint QAT.
  - Its code comments mention more aggressive rate ideas: temporal mask stride,
    spatial downsample, and pair-index FiLM conditioning, though the submitted
    public inflate still uses one class mask per pair at 512x384.
- `selfcomp`:
  - Different but highly relevant grayscale route.
  - The grayscale video is not hard-decoded to argmax classes. It is mapped by
    a Gaussian softmax LUT with `CLASS_TARGETS=[0,255,64,192,128]` and
    `sigma=15`, then passed as a 5-channel probability map.
  - SegMap adds a learned 30x40 latent canvas warped by per-frame affine
    embeddings: zoom, aspect, shear, translation.
  - This explains why plain CRF grayscale replacement collapsed for us:
    successful grayscale needs train/inflate parity through the soft LUT plus a
    learned renderer/latent motion model.

Actionable conclusion:

The current public frontier is not an OWV3/PFP16-like renderer-byte polish
frontier. It is a scorer-aligned overfit renderer frontier with approximately:

```text
full-res semantic control stream
+ tiny charged neural renderer
+ velocity/log-zoom/pose side-channel
+ training through exact uint8 + resize scorer path
+ single-member / unified entropy packing
```

The next highest-EV local path is therefore:

1. **Quantizr/Qpose clone lane, build-only until legal to train:** adapt our
   SegMap/FilmCanvas/Q-FAITHFUL export to the exact contract:
   one mask per pair, pose-conditioned frame1, static frame2, FP4 weights,
   scorer-path differentiable rounding, and unified payload packing.
2. **Velocity/log-zoom pose channel:** replace full 6-DOF side-channel searches
   with a water-filled basis: velocity dim0, lane-mark log-zoom, and residual
   correction atoms. Public qpose/unified evidence says most rotation can be
   dropped at this scale, but our exact eval must verify the basin.
3. **Soft-LUT grayscale parity lane:** keep the patched Selfcomp-style soft LUT
   path alive, but only with a trained SegMap/latent affine decoder. Do not
   rerun hard-onehot CRF grayscale as a promotion path.
4. **Hardware/ego/foveation atom lane:** center foveation transforms on the
   native 910 px camera model/FoE and measure them as charged atoms. Treat
   openpilot/supercombo only as compress-time prior/teacher unless its outputs
   are charged inside the archive.
5. **Packing lane:** run no-training byte prototypes over existing payload
   contracts immediately, but recognize that C-044 only has about 330 bytes of
   zip overhead. Packing matters most after we switch to qpose-like payloads.

Hard stop:

Do not claim sub-0.3 from these public submissions or branch-code reverse
engineering. They are design evidence. The only promotable local claim remains
exact `archive.zip -> inflate.sh -> upstream/evaluate.py` CUDA eval on our own
archive bytes.
