# ddm_us2 Receipt - upstream recursive semantics pass

Arm: `ddm_us2`
Date: 2026-08-05
Mode: `$0`, read-only upstream/runtime/scorer-object inspection. No scorer run,
no dispatch, no archive mutation, no upstream edit.
Pointer: own-vehicle `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`;
contest pointer borrowed/unmoved.

## RECALL EVIDENCE

Loaded governing files before acting:

- `PROGRAM.md`
- `CLAUDE.md` and `AGENTS.md`; `cmp -s AGENTS.md CLAUDE.md` returned equal.
- `docs/operating_manual_craft_handoff.md`
- `.omx/state/main_hot_state.md`
- `.omx/tmp/codex_runs/us2_prompt.md`
- `.omx/tmp/codex_runs/_common_contract.md`

Prior receipts and equations consulted, with consumption:

- `ddm_us1_upstream_reread_20260731.md`: dynamic rate denominator, MPS branch,
  yuv6 polyphase, scorer size/topology, inflate contract, replica-vs-lab venv
  drift.
- `ddm_ua2_upstream_defenses_and_budget_surface_20260731.md`: job-level
  30-minute wall, free contest dependency surface, raw-short truncation,
  archive-integrity absence, CI runtime surface.
- `ddm_pz1_pose_axis_cx1_base_20260803.md`: PoseNet preprocess order correction:
  resize `D` first, `rgb_to_yuv6` second, same shared `D` as SegNet.
- `ddm_vs1_20260805/SCORER_INVISIBLE_NAMING.md`: #839 naming split:
  `RESIZE_KERNEL_NULLITY_DOF`, `CERTIFIED_ZERO_WEIGHT_BLIND_MASK`,
  `RANGE_A_COMPLEMENT_RENDER_ENERGY`, and `COUNTED_PAYLOAD_RATE_CREDIT`.
- `ddm_gt1_upstream_gt_unmined_inventory_20260803.md` and
  `ddm_gt2_gt_tongue_induction_20260803.md`: generic/free/operator vs fitted
  counted payload taxonomy.
- `src/tac/canonical_equations/gap_decomposition_against_floor_20260802.py`:
  rate denominator must be a measured input because upstream uses dynamic
  `rglob`.
- `src/tac/canonical_equations/ddm_m4_rate_floor_20260723.py`: `P_NULL_GAUGE`
  explicitly forbids converting nullity/blind-mask/gauge energy into byte
  credit unless counted parser payload bytes are actually removed.

Checkpoint: `CHECKPOINTS.md` recorded the first inspection phase before scorer
object/runtime extraction.

## Source Custody

Current git/base facts:

- Parent repo HEAD: `954d21db228bbc991b8406537ee29601dbc301a9`.
- Nested upstream HEAD: `11ad728f563d8970929e8947a1cf6124ee6303e4`.
- `git -C upstream status --porcelain=v1` returned empty.

Upstream source hashes captured this turn:

| path | sha256 |
|---|---|
| `upstream/evaluate.py` | `7da71a84ce24286bc6b583470f9bbd25c998971da301320d0d4e9d6fd40baa4b` |
| `upstream/evaluate.sh` | `9612284ce6e9585aefcf636f3027808a56160ffd572edffdf4b8622a65fac917` |
| `upstream/modules.py` | `065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa` |
| `upstream/frame_utils.py` | `d689aca7d263997cb2fb980d6098d503f955e56e8642cd0a04cc437f0ffdab90` |
| `upstream/.github/workflows/eval.yml` | `8a6cd6300b51a44f36b49774bc0c6100dbb37ef8290d42bf8e584f1dceddce56` |
| `upstream/pyproject.toml` | `8651cd684a38cbe5f477d6904ff10bf0c64a917c58dab8e14221e2cc5d879459` |
| `upstream/uv.lock` | `eca4542ad8d21354fd1f2bada74e8659329c0176b17f1ae808e04e023674231f` |
| `upstream/README.md` | `68ea239d7333696e79716e47a9c4288d2918efbcd8912f78932b0befe0af872b` |
| `upstream/models/posenet.safetensors` | `0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576` |
| `upstream/models/segnet.safetensors` | `68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6` |
| `upstream/videos/0.mkv` | `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9` |

As-installed upstream venv facts:

- `upstream/.venv/bin/python`
- `torch 2.10.0`, `torchvision 0.25.0`, `timm 1.0.22`,
  `segmentation-models-pytorch 0.5.0`, `safetensors 0.6.2`, `numpy 2.3.4`,
  `av 17.0.0`
- `torch.nn.functional.interpolate` signature has `align_corners=None` and
  `antialias=False` defaults.

Video lineage:

- `upstream/videos/0.mkv`: 37,545,489 bytes, HEVC Main, 1164x874, yuv420p,
  tv range, chroma location left, 20 fps, 60.0 s.
- MKV tag `SEGMENT = b0c9d2329ad1606b|2018-07-27--06-03-57/10/video.hevc`.
- `upstream/public_test_video_names.txt` is 6 bytes and contains `0.mkv`.
- Derived raw RGB size for 1200 frames: `1164 * 874 * 3 * 1200 =
  3,662,409,600` bytes. Pair count: 600.

## Typed Findings

| surface | prediction | actual | finding class | three-way classification | named consumer | fire-order |
|---|---|---|---|---|---|---|
| evaluator invocation split | Predicted `evaluate.py` might load a submission and run inflate. | `evaluate.py` is scoring-only; `evaluate.sh` performs `unzip`, calls `inflate.sh <archive_dir> <inflated_dir> <video_names_file>`, existence-checks raw outputs, then calls `evaluate.py`. | SEMANTICS-FACT | n/a | `experiments/contest_auth_eval.py`, submission-packet builder/linter | FIRED: recorded exact split here; no code change. |
| dynamic rate denominator | Predicted archive numerator only and possible file-name assumptions. | `evaluate.py` charges `archive.zip` by `stat().st_size`, but denominator is live `sum(file.stat().st_size for file in uncompressed_dir.rglob("*") if file.is_file())`. Current upstream/videos is clean and sums to 37,545,489. | HAZARD | counted-rate arithmetic; no free credit | `src/tac.contest_score.verify_upstream_videos_clean`, `experiments/contest_auth_eval._validate_uncompressed_dir` | FOLDED: existing guards cover it; every future rate claim must keep those guards in path. |
| score accumulation and report precision | Predicted rounded final score/report. | Upstream accumulates `posenet_dists`, `segnet_dists`, and `batch_sizes` as torch scalar tensors, divides, calls `.item()`, computes Python score, prints components at 8 decimals and final at 2 decimals. Local parsers cannot recover the unrounded internal score from `report.txt`. | HAZARD | n/a | `experiments/contest_auth_eval._parse_report`, score ledgers | QUEUED-WITH-FIRE-ORDER: rename/label JSON fields as report-8dp-derived, add a worst-case rounding bound, and test on synthetic reports. |
| partial raw / short iterator | Predicted generated raw path might rely on file-name assumptions. | `TensorVideoDataset` sets `N = file_size // frame_bytes`; `zip(dl_gt, dl_comp)` truncates to the shorter iterator. Upstream only reveals this through printed sample count. | HAZARD | n/a | `experiments/contest_auth_eval` raw byte-count check and `n_samples == 600` parser check | FOLDED: guard exists; do not bypass `contest_auth_eval` for row custody. |
| `contest_auth_eval.py` raw-size comment | Not predicted. | The code computes `1164*874*expected_num_frames*3`, but the local comment at lines 1491/1501 names `3,663,237,120`; the correct 1200-frame byte count is `3,662,409,600`. Runtime behavior is correct; human proof text is wrong. | HAZARD | n/a | `experiments/contest_auth_eval.py` maintainers | QUEUED-WITH-FIRE-ORDER: comment-only patch plus existing tests; no scorer run. |
| shared resize `D` and blind geometry | Predicted both scorers resize to SegNet input size with bilinear defaults. | `modules.py` calls `F.interpolate(..., mode="bilinear")` with no explicit `align_corners` or `antialias`; upstream venv signature gives `align_corners=None`, `antialias=False`. PoseNet does the resize before `rgb_to_yuv6`, so both scorers share the same `D`. | EXPLOIT | GENERIC-FREE operator property; not counted bytes by itself | `#401` blind-coordinate fill, `m86`, `bp2`, `sg2`, MLX R adapters | FOLDED: already exploited and named; do not convert `RESIZE_KERNEL_NULLITY_DOF` or `CERTIFIED_ZERO_WEIGHT_BLIND_MASK` into byte credit unless counted payload bytes are removed. |
| in-loop R parity surface | Predicted local R must match upstream defaults. | Live witness path routes through `apply_contest_faithful_roundtrip_nhwc`: render-grid bicubic to camera, uint8 STE at camera, then bilinear downsample to scorer with no trailing uint8. The scorer-resolution uint8 twin is deprecated. Source-level mismatch against upstream scorer downsample was not found. Existing comments cite bit-identity tests; I did not rerun them. | CONFIRMED-NO-DRIFT | GENERIC-FREE algorithm semantics | v7.5/v8 witness trainer and byte-close verifier | FOLDED: no blast radius from this pass. |
| PoseNet preprocessing order | Predicted Pose preprocessing uses yuv6 and resize. | The exact order is NHWC->BTCHW, bilinear resize to 384x512, then `rgb_to_yuv6`, then pack two frames into 12 channels. This confirms PZ1's correction and rejects the old reversed-order wording. | SEMANTICS-FACT | GENERIC-FREE operator property | pose-axis carriers, Q3/chroma/null-space consumers | FOLDED: PZ1 owns the correction; consumers should cite source lines, not CLAUDE wording. |
| scorer checkpoint objects | Predicted SegNet/PoseNet topology and BN/state facts might expose unregistered operator properties. | PoseNet checkpoint file is 55,835,560 B; tensor payload 13,943,652 fp32 elems. SegNet checkpoint file is 38,502,892 B; tensor payload 9,610,645 elems. PoseNet is FastViT with 4 stages of 2/2/6/2 RepMixerBlock, 8 scalar `AllNorm` BatchNorm1d(1) sites, pose head 32->12. SegNet is SMP Unet `efficientnet_b2`, encoder channels `[3,16,24,48,120,352]`, decoder widths 256/128/64/32/16, final `(5,16,3,3)` head. | HAZARD | topology is GENERIC-FREE; checkpoint weights are COUNTED if used at decode; compress-time use is legal | `check_no_scorer_load_at_inflate`, submission linter/runtime scanner | QUEUED-WITH-FIRE-ORDER: extend actual-runtime-root scan to reject decode-time opens/imports of `upstream/models/*.safetensors`, not just `submissions/*/inflate*` patterns. |
| contest venv free imports | Predicted CI installs deps through uv and runtime may differ from local. | `eval.yml` runs `uv run --group "$UV_GROUP" bash evaluate.sh`, and `evaluate.sh` calls `bash inflate.sh` inside that environment. Free import surface includes torch/torchvision/numpy/einops/timm/safetensors/SMP/tqdm/pillow/av/requests stack, plus DALI on CUDA groups; `ffmpeg` is on PATH. | EXPLOIT | ECONOMIC: code/tools/libs free; extra declared deps are budget trade; large artifacts remain COUNTED | #214 decode ladder, e4 declared-dep decision, runtime compiler | FOLDED: consume UA2's priced surface; do not relitigate as a ban or as zero-time. |
| local auth-eval environment purity | Predicted local/contest versions may drift. | `experiments/contest_auth_eval.py` invokes `upstream/evaluate.py` with `sys.executable`; if called as `.venv/bin/python experiments/contest_auth_eval.py`, scorer source is upstream but dependency versions are root-lab (`torch 2.12.1`, `torchvision 0.27.1`, `timm 1.0.27`, `numpy 1.26.4`, etc.), not upstream contest (`torch 2.10.0`, `torchvision 0.25.0`, `timm 1.0.22`, `numpy 2.3.4`). | HAZARD | n/a | `experiments/contest_auth_eval.py`, exact/advisory ledgers | QUEUED-WITH-FIRE-ORDER: add/require an upstream-python path for local authority replays, record package versions in JSON, and downgrade root-venv runs to advisory unless parity is proven for that exact lock. |
| upstream nested repo drift | Predicted `uv.lock` drift may matter. | Current `git -C upstream status --porcelain=v1` is empty at `11ad728f`; no live nested upstream worktree drift found in this checkout. Root `uv.lock` still differs from upstream lock, which is the env-purity hazard above. | CONFIRMED-NO-DRIFT | n/a | upstream-custody checks | FIRED: current clean fact recorded; no upstream action. |
| GT video and segment metadata | Predicted MKV metadata could encode implicit decode facts. | `ffprobe` reports HEVC Main, yuv420p, tv range, chroma left, 20 fps, 60 s, segment tag matching `public_test_segments`. `frame_utils.yuv420_to_rgb` then applies limited-range BT.601 to decoded planes and rounds to uint8. | SEMANTICS-FACT | GT content is COUNTED/FORBIDDEN at decode unless in archive; metadata/provenance facts are free to know | compression pipeline, provenance receipts, runtime linter | QUEUED-WITH-FIRE-ORDER: scan actual runtime roots for decode-time reads of `upstream/videos/0.mkv` or `videos/0.mkv`; compression-time use remains legal. |
| source-video denominator equals GT file size today | Predicted official single MKV has 1200 frames / 600 pairs. | `stat(upstream/videos/0.mkv) == 37,545,489`, equal to current denominator. This is a current-filesystem fact, not a hard law because `evaluate.py` uses dynamic `rglob`. | SEMANTICS-FACT | n/a | `gap_decomposition_against_floor_20260802`, rate ledgers | FIRED: record current equality; keep denominator measured at consumption. |

Class count, primary class per row: EXPLOIT 2, HAZARD 6, SEMANTICS-FACT 4,
CONFIRMED-NO-DRIFT 2.

## In-loop-R Mismatch Verdict

I did not find a source-level mismatch between the live local witness R path and
the upstream scorer boundary. The live path places uint8 at camera resolution,
then performs scorer-resolution bilinear downsample with no trailing uint8; that
matches the contest surface where raw reconstructed frames are camera-resolution
uint8 and `modules.py` downsamples to 384x512 float before SegNet/PoseNet.

Blast radius: none from this pass. Existing bit-identity claims were read from
source comments/tests but not rerun, because this charter was `$0` and scorer-free.

## Score And Pointer Honesty

No exact row was produced. No score moved. This memo is a semantics/runtime
receipt and follow-on queue, not frontier progress.

Own-vehicle frontier line remains:
`S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`.

Contest pointer remains borrowed/unmoved.
