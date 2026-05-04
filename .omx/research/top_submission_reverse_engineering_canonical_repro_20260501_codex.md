# Top Submission Reverse Engineering Canonical Repro - 2026-05-01

Evidence grade: `external_plus_empirical_byte_anatomy`.

Score claim: `false`. Public submission anatomy is design signal only until a
concrete archive from this repo passes exact CUDA auth eval.

## Canonical Command

From repo root:

```bash
.venv/bin/python experiments/reverse_engineer_top_submissions.py \
  --output-dir experiments/results/top_submission_reverse_roundtrip_20260501 \
  --work-dir /tmp/pact_topsubs_repro_20260501
```

For an already-fetched local checkout:

```bash
TMP=$(cat .omx/state/latest_pr65_pr67_reverse_tmpdir.txt)
.venv/bin/python experiments/reverse_engineer_top_submissions.py \
  --pr67-dir "$TMP/pr67" \
  --pr65-dir "$TMP/pr65" \
  --output-dir experiments/results/top_submission_reverse_roundtrip_20260501
```

The output is deterministic for pinned inputs: no timestamps, no host-local
paths, pinned commits, expected archive hashes enforced, and stable JSON key
ordering.

## Pinned Inputs

- PR #67 `qpose14_qzs3_filmq9g_slsb1_r55`
  - commit: `696d4a1e64a7f2d9aada3e3833be3c91ad394c21`
  - archive bytes: `276564`
  - archive SHA-256:
    `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`
  - container member: `p`
- PR #65 `henosis_qz_n3z_r25_clean`
  - commit: `a8b53b5280ee8f05db65740cd48cf7c321a55497`
  - archive bytes: `284425`
  - archive SHA-256:
    `b331cb4f6df9d8929db966b943b8c73624cdf3b6db71acbde361570852e59e68`
  - container member: `x`

## Canonical Output

`experiments/results/top_submission_reverse_roundtrip_20260501/archive_anatomy.json`

Key facts recorded there:

- Local `JointFrameGenerator` reference:
  - module: `src/tac/quantizr_faithful_renderer.py`
  - parameters: `87836`
  - state-dict keys: `111`
- PR #67:
  - `p` bytes: `276464`
  - mask segment: `219472` Brotli bytes -> `223385` raw bytes
  - QZS3 model segment: `56093` Brotli bytes -> `59288` raw bytes
  - QP1 pose segment: `899` Brotli bytes -> `1140` raw bytes
  - QZS3 decode validation: `111` keys, `87836` parameters, all finite
- PR #65:
  - `x` bytes: `284325`
  - first 24-bit segment lengths:
    `[219472, 57074, 1487, 1400, 226, 106, 149, 154, 223, 273, ...]`

## Scientific Boundaries

- These artifacts explain public byte allocation and decoder packing, but they
  are not our score evidence.
- Any rank, frontier, or promotion claim still requires a repo-built
  `archive.zip`, exact archive SHA/bytes, full `contest_auth_eval.py --device
  cuda`, `600` samples, T4/equivalent hardware, component recomputation, and
  adjudication.
- The QZS3 packer only applies to Quantizr-faithful `JointFrameGenerator`
  archives. It cannot shrink the active OWV3 C-051 frontier directly.
- The useful transfer is architectural: train or recover a strong JFG/SegMap/
  soft-LUT representation, then pack it with QZS3/QP1/single-blob atoms and
  exact-eval the complete archive.

## Verification

```bash
.venv/bin/python -m py_compile \
  experiments/reverse_engineer_top_submissions.py \
  src/tac/tests/test_reverse_engineer_top_submissions.py

.venv/bin/pytest -q \
  src/tac/tests/test_reverse_engineer_top_submissions.py \
  src/tac/tests/test_quantizr_faithful_renderer.py \
  src/tac/tests/test_canonical_local_e2e_smoke.py::test_smoke_renderer_magic_knows_qzs3
```

Observed focused result: `20 passed`.

## Live Transfer Into Q-FAITHFUL - 2026-05-01T19:55Z

The public packer anatomy has now been transferred into the local
Q-FAITHFUL/JointFrameGenerator lane as a build-only, score-claim-false
postprocess.

Implementation fixes:

- `scripts/remote_q_faithful_postprocess_fixed.sh` accepts crash-recovery
  training checkpoints with a `model` key.
- The same script writes JSON-safe archive validation provenance via
  `dataclasses.asdict`.
- The remote script hash deployed to Vast after the fixes was
  `baf60c14a1e2f552777547b23e7ee492d8c307e2ea6d69b00f0b98582e120b12`.

Early snapshot bytes:

```text
raw_qfai_half_posebin  563206  8ad1b494e03fba010a663d18f5d9b5b530e3fdcbd5fccf7633b0db295f4ad33e
qzs3_half_posebin      284486  751f4f9105a3479548685b2a70041b222ddae35ec25bae601b33284e76e48bf3
qzs3_rp2_qpose14       276651  1d5d5e86cb0902f306f1e0968385033172c97ab2b882866beffbf9b26a2acca4
qzs3_rp2_qp1           273048  e70bb3fbda789e4fb8880e80b672a608e318bc0084415958c52fc6631a6a20f4
```

Local mirror:

```text
experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T1947Z_fix3/
```

Diagnostic exact eval queued:

```text
job=exact_eval_qfaithful_snapshot_qzs3_rp2_qpose14_l40s_20260501T1952Z
state=.omx/state/qfaithful_snapshot_qzs3_rp2_qpose14_l40s_batch_jobs_20260501T1952Z.json
archive_sha256=1d5d5e86cb0902f306f1e0968385033172c97ab2b882866beffbf9b26a2acca4
archive_bytes=276651
machine=g6e.4xlarge / L40S
score_claim=false until CUDA artifact lands
```

Scientific boundary:

- These bytes match the PR #67/#63 scale and prove the local packer shape.
- They do not prove the early checkpoint is good. The L40S exact eval will
  answer whether the current phase-1 checkpoint has already entered the
  qpose14 scorer basin or remains only a byte-good representation.

## Current Public Floor Attachment Anatomy - 2026-05-01T20:06Z

Fresh public-board check: `qpose14` remains listed at `0.32`;
`unified_brotli` and `quantizr` remain listed at `0.33`.

Canonical command now includes current-floor attachment archives:

```bash
.venv/bin/python experiments/reverse_engineer_top_submissions.py \
  --output-dir experiments/results/top_submission_reverse_roundtrip_20260501 \
  --work-dir /tmp/pact_topsubs_current_20260501 \
  --current-floor-archive-dir experiments/results/top_submission_current_floor_20260501/external_archives
```

Output:

```text
experiments/results/top_submission_reverse_roundtrip_20260501/archive_anatomy.json
```

Current-floor external archive facts:

- PR #63 `qpose14`: `287573` bytes, SHA
  `e012ebeffcc1e1655f4d674d0a779c1bf4cd41cfa82746de2ff6f73692e82a66`.
  Single zip member `p`, `287473` bytes. Fixed slices:
  `mask_obu_br=219472`, `model_torch_quantized_br=66841`,
  `pose_qpose14_uint16_br=1160`.
- PR #64 `unified_brotli`: `287165` bytes, SHA
  `7e48da0be75f915d6a4cf76a4679f8c1fbe689f82d69c7559f6ba1c2cb1e981d`.
  Single zip member `p`, `287065` bytes. One Brotli stream of
  `mask_obu=223385`, `model_torch_quantized=91582`,
  `pose_velocity_delta_uint16_int16=1200`.

Critical transfer finding:

- PR #63/#64 are not the older PR #67 QZS3 payload. They use the same
  `JointFrameGenerator` parameter count (`87836`) but a Torch serialized
  quantized renderer payload: `40` FP4-packed modules, `3` dense FP16 weight
  modules, and `46` dense FP16 entries.
- PR #64's score-equivalent byte edge over PR #63 is pure packing/pose
  allocation: single-stream Brotli over raw mask/model/pose, plus
  velocity-only pose delta side-channel. It has the same raw mask SHA and same
  raw model SHA as PR #63.
- Our Q-FAITHFUL snapshot QZS3/RP2 archives are materially smaller than PR
  #63/#64. That is good only if exact CUDA says the QZS3 renderer quantization
  remains in the scorer basin. If QZS3 collapses while runtime passes, the
  next fallback is a PR63-style Torch quantized model payload before abandoning
  the current checkpoint.

Evidence boundary:

- This section is `external_plus_empirical_byte_anatomy`, not our score
  evidence.
- Any transfer claim requires a repo-built archive and exact CUDA auth eval on
  those exact bytes.

## PR63 Torch-FP4 Transfer Implementation - 2026-05-01T20:20Z

The PR63/PR64 renderer payload is now represented by repo code rather than
one-off reverse-engineering notes.

New deterministic build/runtime support:

- `src/tac/quantizr_torch_fp4_codec.py`: exports and loads the PR63-style
  Torch-FP4 `JointFrameGenerator` payload shape.
- `submissions/robust_current/inflate_renderer.py`: detects Torch-FP4 payloads
  before generic PyTorch checkpoint fallback and wraps the generator in the
  same Q-FAITHFUL pair API as QZS3.
- `experiments/repack_quantizr_faithful_qzs3_archive.py`: accepts
  `--renderer-codec torch_fp4`.
- `scripts/remote_q_faithful_postprocess_fixed.sh`: emits both QZS3 and
  Torch-FP4 postprocess archives from future checkpoints.

Important custody note:

- Public PR63 uses Torch's legacy pickle serialization. That embeds
  process-local storage IDs, so the repo exporter uses Torch's deterministic
  zip serializer for custody; Brotli-compressed size is only a small overhead
  versus legacy. The runtime loader accepts both the downloaded public legacy
  payload and the deterministic repo-built payload.

Verification:

- `py_compile` passed for touched Python.
- `bash -n scripts/remote_q_faithful_postprocess_fixed.sh` passed.
- Focused tests: `27 passed` across Torch-FP4 codec, QZS3 packer, renderer
  payload, and reverse-engineering tests. This includes decoding the downloaded
  public PR63 model payload when the archive fixture is present.

PR64 container transfer:

- Added repo support for `pr64_len_table`, the public `unified_brotli`-style
  decompressed payload shape: `<III> + renderer + masks + pose`, stored as one
  Brotli-compressed `p` member.
- The runtime unpacker recognizes this shape only when the three positive
  lengths exactly consume the payload, then decodes pose bytes by their own
  magic (`PCD1`, `QP14`, `QP1`, `PVL1`, `PVR1`) or leaves raw fp16 poses
  untouched.
- Focused renderer-payload coverage now includes PR64 length-table qpose14
  round-trip through archive extraction and runtime unpack.

## Public Floor Recheck and Transfer Rule - 2026-05-01T21:21Z

Fresh external check:

- Official comma leaderboard still lists:
  - `qpose14`: `0.32`;
  - `unified_brotli`: `0.33`;
  - `quantizr`: `0.33`.
- PR #63 remains the best public score visible in the leaderboard snapshot.

Local custody recheck:

```text
PR63 qpose14 archive.zip
bytes=287573
sha256=e012ebeffcc1e1655f4d674d0a779c1bf4cd41cfa82746de2ff6f73692e82a66
zip member: p, 287473 bytes

PR64 unified_brotli archive.zip
bytes=287165
sha256=7e48da0be75f915d6a4cf76a4679f8c1fbe689f82d69c7559f6ba1c2cb1e981d
zip member: p, 287065 bytes
```

Transfer rule:

- PR #64's byte improvement over PR #63 is still pure packer/pose allocation,
  not a different renderer basin:
  `single Brotli stream(mask, model, pose) + length table + velocity-only pose`.
- Therefore, future Q-FAITHFUL checkpoints should be exported in this order:
  1. QZS3/RP2/QP1 or qpose14 when the checkpoint is demonstrably in basin.
  2. PR64-style unified Brotli container if it reduces bytes without changing
     decoded tensors.
  3. PR63-style Torch-FP4 fallback only if QZS3 quantization is the collapse
     source and the checkpoint otherwise has scorer geometry.
- No new exact eval should be spent on early out-of-basin snapshots just
  because they are byte-small.

## Q-FAITHFUL 21:31Z Top-Submission Container Transfer - 2026-05-01T21:39Z

The public PR64 container was applied to a newer live Q-FAITHFUL checkpoint
snapshot without stopping the trainer.

Snapshot custody:

```text
remote_instance=Vast 35959478
snapshot=lane_q_faithful_results/postprocess_snapshots/training_state_snapshot_20260501T2131Z.pt
snapshot_sha256=2d98d481b21cdf4f188e27bf73c133e7b64274ce62f854e345c89b22dcd065b1
snapshot_size=1587860
postprocess_dir=experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/postprocess_fixed_snapshot_20260501T2131Z_fix1
```

Archive byte screen:

```text
qzs3_pr64_qp1:      272995 bytes, sha256=1a35d4d8899afa47602137efd36a427cdadc6e3a25615c4c54ec51c1ac73374f
qzs3_rp2_qp1:       273069 bytes, sha256=7dbdc8e9e7b2fe4c348e483bff4f319f58019ed3d4dc154af0f1118e28d87492
qzs3_pr64_qpose14:  276542 bytes, sha256=70ac01f7446db7766577829a7ec0fc7ab633676408ad8c3ffdd662f2e66a0f3d
qzs3_rp2_qpose14:   276629 bytes, sha256=fd5a462551e1a52b56690a7bc055a9a281c6c6f12dca2258aa18b71333c3a81f
torchfp4_pr64_qpose14: 288984 bytes, sha256=6b41aba2614dac935e01671ee4b8f4455ba0ad329fe545bb8673af8a1d5bb2ed
```

Dispatch decision:

- Exact-eval the qpose14 variant first despite the larger byte count, because
  the early snapshot failure was a renderer/PoseNet basin failure and QP1 adds
  large pose quantization error.
- If qpose14 returns in basin, QP1 becomes the immediate byte-reduction follow
  up.
- If qpose14 remains out of basin, do not spend on QP1; wait for the next
  training checkpoint or add learned pose/ego-motion conditioning.

Queued exact diagnostic:

```text
job=exact_eval_qfaithful_snapshot_2131_pr64_qpose14_l40s_20260501T2138Z
archive_bytes=276542
archive_sha256=70ac01f7446db7766577829a7ec0fc7ab633676408ad8c3ffdd662f2e66a0f3d
machine=g6e.4xlarge / L40S
component_trace=true
state=.omx/state/qfaithful_snapshot_2131_pr64_qpose14_l40s_batch_jobs_20260501T2138Z.json
```

## Public Floor Exact Trace Reproduction - 2026-05-01T22:09Z

Evidence grade:

- Public archive score claims remain `external`; the local reruns below are
  `A score-grade CUDA` diagnostics on L40S with exact archive custody and
  component trace cross-checks. They are not our submission frontier.

Fresh public sources checked:

- Official leaderboard: `https://comma.ai/leaderboard`
- PR #63: `https://github.com/commaai/comma_video_compression_challenge/pull/63`
- PR #64: `https://github.com/commaai/comma_video_compression_challenge/pull/64`

Current visible lowest scores:

```text
0.32 qpose14          PR #63
0.33 unified_brotli  PR #64
0.33 quantizr        PR #55
0.37 fp4_mask_gen    PR #62
0.38 selfcomp        PR #56
```

State-derived local CUDA harvests:

```text
PR63 qpose14
job=exact_eval_public_pr63_qpose14_trace_l40s_20260501T2149Z
artifact_dir=experiments/results/lightning_batch/exact_eval_public_pr63_qpose14_trace_l40s_20260501T2149Z
archive_bytes=287573
archive_sha256=e012ebeffcc1e1655f4d674d0a779c1bf4cd41cfa82746de2ff6f73692e82a66
gpu=NVIDIA L40S
n_samples=600
score=0.32518843312932477
pose=0.00052823
seg=0.00061026
component_trace_sha256=99c7cb20d20b18ec798dc59c9be075d4ba5ecfdc24ea8d95cbdafb32bfc0c53c
component_trace_cross_check_all_match=true

PR64 unified_brotli
job=exact_eval_public_pr64_unified_trace_l40s_20260501T2150Z
artifact_dir=experiments/results/lightning_batch/exact_eval_public_pr64_unified_trace_l40s_20260501T2150Z
archive_bytes=287165
archive_sha256=7e48da0be75f915d6a4cf76a4679f8c1fbe689f82d69c7559f6ba1c2cb1e981d
gpu=NVIDIA L40S
n_samples=600
score=0.33137914516864686
pose=0.00062634
seg=0.00061026
component_trace_sha256=81dc854f749f52bcea6ba2bf4426dbb8040405343adb7737beb5f0f905e5fd78
component_trace_cross_check_all_match=true
```

Reverse-engineering conclusion strengthened by exact traces:

- The public floor is not just a packer trick. It is the combination of an
  in-basin JointFrameGenerator-family renderer, public-mask-class geometry at
  roughly `0.00061` SegNet distance, a one-scalar pose manifold, and extreme
  payload packing around a single stored `p` zip member.
- PR64 proves the non-velocity pose channels can be dropped for this renderer
  basin: its pose worsens relative to PR63, but the total remains public-floor
  because the rate term improves and SegNet is unchanged.
- Future Q-FAITHFUL exports should first prove renderer-basin geometry against
  the public mask/one-scalar pose contract, then apply QZS3, QP1, and PR64
  packing. Byte-small out-of-basin checkpoints are invalid optimization
  targets.

C-051 versus public-floor trace comparison:

```text
comparison_json=experiments/results/component_trace_comparison_c051_vs_public_floor_20260501/trace_comparison.json
best_reference=pr63_qpose14_public_floor
c051_score=0.9867718508244918
pr63_score=0.32518863897296757
total_gap=0.6615832118515242
gap_pose=0.11604423850120972
gap_seg=0.3414705165511501
gap_rate=0.20406845679916435
archive_delta_bytes=306474
```

Top excess pair indices, C-051 over PR63:

```text
combined: 127, 75, 109, 133, 514, 125, 517, 522, 111, 45
pose:     127, 75, 109, 125, 514, 179, 378, 90, 111, 289
seg:      133, 517, 109, 522, 177, 514, 45, 127, 521, 516
```

Implication:

- Pair repair is a useful microscope, but it cannot be the only optimization
  axis. The public-floor gap is dominated by global SegNet geometry and rate.
- The highest-EV floor attack is now:
  `public-pose-basis renderer basin -> global mask/renderer byte reduction ->
  pair/component atom water-fill -> exact CUDA archive eval`.

## Q Snapshot Public-Pose-Basis Negative - 2026-05-01T23:16Z

Exact diagnostic:

```text
job=exact_eval_qfaithful_snapshot_2146_pr64_qp1_l40s_20260501T2201Z
archive=qzs3_pr64_qp1
archive_bytes=273103
archive_sha256=a34f493b77e3a2ccba7e059134127e9b3cb6e774a41862143d369fa3f5fc81af
score=22.065520725118258
pose=46.18100739
seg=0.00393906
gpu=NVIDIA L40S
component_trace_sha256=383a383aa75fce26a5c8962ed8adcfa6128b16fb8e6cf69fe21a9b4dbdbcd7e8
```

Interpretation:

- Public PR64-style velocity-only/QP1 pose packing is not a magic rescue for an
  out-of-basin renderer checkpoint.
- The 21:46Z Q snapshot was already catastrophic under qpose14; QP1 made the
  PoseNet basin mismatch worse.
- Public PR63/PR64 remain the architecture/packer target, but the next
  implementation must first train into their renderer geometry, not merely
  pack our current checkpoint using their container.

Action taken:

- Mirrored the live trainer checkpoint, metadata, and telemetry to
  `experiments/results/lane_q_faithful_retrain_20260501/remote_artifacts/stopped_out_of_basin_20260501T2316Z`.
- Stopped the specific Vast trainer process after evidence preservation.
- Freed the RTX 4090 for a different, evidence-driven dispatch.

Reference-relative Q trace comparison:

```text
comparison_json=experiments/results/component_trace_comparison_qfaithful_2146_qp1_vs_public_floor_20260501/trace_comparison.json
best_reference=pr63_qpose14_public_floor
score_delta_vs_pr63=21.74033131907131
pose_delta_vs_pr63=21.417086479367764
seg_delta_vs_pr63=0.3328798187552214
rate_delta_vs_pr63=-0.009634979051677844
archive_delta_bytes_vs_pr63=-14470
```

Top catastrophic QP1 excess pairs versus PR63:

```text
combined: 72, 75, 89, 71, 77, 67, 73, 443, 78, 76
```

The Q failure is therefore not a small set of repairable pair atoms. It is a
global PoseNet manifold failure with localized peaks around the same highway
motion segment plus later pose discontinuities. The next valid Q experiment
must change the training contract before the packer.

## Public-Floor PVL1 Repack Candidate - 2026-05-01T23:36Z

Current public board check:

```text
qpose14          0.32  PR #63
unified_brotli  0.33  PR #64
quantizr        0.33  PR #55
```

Candidate:

```text
label=pr63_pr64_pvl1
archive=experiments/results/public_floor_contract_variants_20260501/pr63_pr64_pvl1/archive.zip
archive_bytes=286960
archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
source_runtime_archive=/tmp/pact_pr63_contract/source_runtime/archive.zip
source_runtime_archive_sha256=9e2b49437e63ead93329157b64b5bfc3208c825ec95df1ec929c1ccad2c3fb1c
payload_member=p
payload_format=pr64_len_table
pose_codec=pose_fp16_velocity_only_v1
pose_payload_bytes=1208
decoded_pose_sha256=cc99e99c28b2ea686439b226ee504ba3a0d82fd8eb8550f4fed05d35ece5dc40
source_decoded_pose_sha256=cc99e99c28b2ea686439b226ee504ba3a0d82fd8eb8550f4fed05d35ece5dc40
pose_error_max_abs_by_dim=0,0,0,0,0,0
masks_sha256=a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb
renderer_sha256=d97849d15859ae013ec983de8c1e2f638e63f3876fef658a8b7781bcfaa16a5f
score_claim=false
evidence_grade=empirical_until_exact_cuda_harvest
```

Expected score under exact decoded-geometry parity:

```text
public_pr63_score=0.32518843312932477
public_pr63_bytes=287573
candidate_bytes=286960
delta_bytes=-613
formula_only_rate_delta=-25*613/37545489=-0.0004081811801578958
expected_score_if_components_identical=0.3247802519491669
```

Dispatch:

```text
lightning_manifest=.omx/state/public_floor_pvl1_20260501T2330Z_manifest.json
l40s_job=exact_eval_public_floor_pr63_pr64_pvl1_l40s_20260501T2332Z
l40s_state=.omx/state/public_floor_pvl1_l40s_batch_jobs_20260501T2332Z.json
t4_job=exact_eval_public_floor_pr63_pr64_pvl1_t4_20260501T2332Z
t4_state=.omx/state/public_floor_pvl1_t4_batch_jobs_20260501T2332Z.json
vast_h100_instance=35985850
vast_h100_label=public_floor_pvl1_h100diag_20260501
vast_h100_remote_log_dir=/workspace/pact/experiments/results/vast_h100_public_floor_pvl1_20260501
```

Adversarial interpretation:

- This is not a verbatim public archive submission. It is a public-floor-basin
  contract repack: PR63 decoded renderer/mask/pose geometry, PR64 one-member
  Brotli length-table container, and a PVL1 pose stream that decodes to the
  identical PR63 pose bytes.
- The only allowed claim before exact eval is byte/contract parity. Score
  depends on exact CUDA auth eval of this archive.
- If exact components match PR63, this is already in the public leaderboard
  band and should receive T4 promotion on identical bytes.

## PVL1 Exact H100 Diagnostic And Variant Sweep - 2026-05-01T23:58Z

PVL1 exact CUDA diagnostic landed on H100 and beat the public PR63 score in
the same public-floor basin, pending T4 promotion:

```text
archive=experiments/results/public_floor_contract_variants_20260501/pr63_pr64_pvl1/archive.zip
archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
archive_bytes=286960
hardware=NVIDIA H100 NVL
evidence_grade=A diagnostic, not promotion-grade
score_recomputed_from_components=0.3246902093443082
component_trace_score=0.3246906061663023
avg_posenet_dist=0.0005270522588307358
avg_segnet_dist=0.0006101735606353031
n_samples=600
harvest=experiments/results/vast_harvest/public_floor_pvl1_h100diag_fix2_20260501/
```

The same archive is now queued/running for fixed-manifest T4 promotion:

```text
job=exact_eval_public_floor_pr63_pr64_pvl1_t4_fix1_20260501T2353Z
state=.omx/state/public_floor_pvl1_t4_fix1_batch_jobs_20260501T2353Z.json
manifest=.omx/state/public_floor_pvl1_fix1_20260501T2352Z_manifest.json
manifest_file_sha256=57ba9e6d08cec022d050e99402c0039754e45d203117117b94c135349dd818ea
status_at_2026-05-01T23:59Z=Running
```

The first four follow-up public-floor packer/pose variants were screened on
the same H100. None beat PVL1:

```text
variant                              score                  pose_dist      seg_dist       bytes
pr63_pr64_pvl1                       0.3246902093443082     0.00052705     0.00061017     286960
pr63_pr64_pvr1_top256                0.3248499593443082     0.00052705     0.00061017     287200
pr64_pr64_pvr1_top64                 0.3310104760247508     0.00062228     0.00061017     287011
pr64_pr64_qpose14                    0.33108572602475084    0.00062228     0.00061017     287124
pr63_pr64_barevel_rendererfirst      19.534443931576337     37.18033981    0.00061017     287182
```

Interpretation:

- PVL1 remains the only current promotion candidate.
- `pvr1_top256` is scientifically valid but byte-regressive relative to PVL1.
- PR64-source `qpose14`/`pvr1_top64` trigger a repeatable PoseNet regression
  and should not be promoted without pose geometry reconciliation.
- Bare public velocity delta decoding is catastrophically wrong in the current
  local contract; likely axis/order/scale mismatch. This is a measured
  implementation failure, not a kill of the public codec family.

## PVL1 T4 Promotion Landed - 2026-05-02T00:14Z

Fixed-manifest Lightning T4 promotion landed and was re-adjudicated after
fixing the adjudicator's regression-threshold sign/semantics bug:

```text
job=exact_eval_public_floor_pr63_pr64_pvl1_t4_fix1_20260501T2353Z
artifact_dir=experiments/results/lightning_batch/exact_eval_public_floor_pr63_pr64_pvl1_t4_fix1_20260501T2353Z
archive=experiments/results/public_floor_contract_variants_20260501/pr63_pr64_pvl1/archive.zip
archive_sha256=4479badf2aeb489e182ad57a5bba7de10c475f2395d69cdf462462e7a7879610
archive_bytes=286960
hardware=Tesla T4
n_samples=600
score_recomputed_from_components=0.3247176275031171
score_reported_rounded=0.32
avg_posenet_dist=0.00052391
avg_segnet_dist=0.00061261
component_trace_score=0.3247177087654878
evidence_grade=A++ contest T4
promotion_eligible=true
```

Comparison to exact public PR63 qpose14 trace:

```text
score_delta_vs_public_pr63=-0.00047080562620765987
archive_delta_bytes=-613
posenet_relative=0.9918217443159231
segnet_relative=1.0038508176842658
component_gates=passed
```

The public leaderboard currently rounds this band as `0.32`; this archive is a
leaderboard-band deploy candidate with exact T4 custody. The next public-floor
work should target PR67/QZS3/QP1 byte parity or a real learned Q-FAITHFUL
artifact; additional PR64/PVR1 variants are dominated unless they beat
`286960` bytes or improve components.

## QZS3/QP1 Public-Floor Packer And Pose Line Search - 2026-05-02T01:05Z

PR67/QZS3/QP1 fixed-slice repack landed as the new A++ frontier:

```text
job=exact_eval_public_floor_qzs3_qp1_t4_20260502T0036Z
artifact_dir=experiments/results/lightning_batch/exact_eval_public_floor_qzs3_qp1_t4_20260502T0036Z
archive=experiments/results/public_floor_qzs3_qp1_packer_20260502/pr63_qzs3_qp1_fixedslice/archive.zip
archive_sha256=c5260473c26c4d4537d99d4a6a18b8ff0d9d1a901f6db17cd2208559e1010362
archive_bytes=276296
hardware=Tesla T4
n_samples=600
score_recomputed_from_components=0.3243472585872431
avg_posenet_dist=0.00062614
avg_segnet_dist=0.00061244
evidence_grade=A++ contest T4
promotion_eligible=true
```

Interpretation:

- The useful reverse-engineered #1/#2 trick in this step is archive-side
  QZS3/QP1 packing inside the already measured public-floor renderer basin.
- The T4 result proves the QZS3/QP1 byte atom composes with the public-floor
  geometry well enough to beat PVL1, despite controlled PoseNet drift.
- PR67 fixed-slice parsing must tolerate variable pose payload length after
  pose line-search. The runtime parser now validates plausible generated PR67
  model lengths by Brotli/QZS3/QP1 structure instead of assuming one brittle
  split point.

The first pose-coordinate line search on the same QZS3/QP1 archive produced a
strong H100 diagnostic candidate that is now T4-promoted:

```text
h100_archive=experiments/results/vast_harvest/archive_eval_line_search_qzs3_qp1_fixedslice_20260502T0057Z/archive.zip
h100_archive_sha256=8c9000f67eb21f366299fe033e3e6031ab63992e8067758600e43d0091c9a9fa
h100_archive_bytes=276427
h100_score=0.32114254758178584
h100_posenet=0.00057865
h100_segnet=0.00061012
t4_job=exact_eval_line_search_qzs3_qp1_t4_20260502T0100Z
t4_score=0.3218613619571356
t4_posenet=0.00058608
t4_segnet=0.00061244
t4_evidence=A++ contest T4
t4_promotion_eligible=true
t4_state=.omx/state/line_search_qzs3_qp1_t4_batch_jobs_20260502T0100Z.json
```

The active H100 continuation has already improved the line-search objective
beyond the first checkpoint:

```text
baseline_obj=0.259544306
radius3_pass2_obj=0.257372331
radius5_pass2_obj=0.255006542
radius8_pass2_obj=0.253897808
radius13_pass2_obj=0.253772163
```

Decision rule:

- The first line-search archive is the current A++ frontier because its T4
  adjudicated artifact lands inside gates and below C-053.
- The stronger r8 continuation checkpoint is still only H100 diagnostic until
  `exact_eval_line_search_qzs3_qp1_r8_t4_20260502T0110Z` lands.
- Continue H100 line-search while it improves the checkpoint objective, but
  every new submit-grade checkpoint needs exact T4/equivalent confirmation on
  identical bytes.

## Learned/Directional Pose Proposal Patch - 2026-05-02T01:20Z

The next pose-search extension preserves the same archive format and exact
acceptance rule, but replaces purely arbitrary symmetric candidate grids with
opt-in directional and differentiable proposal stages:

```text
script=experiments/line_search_pose_refinement.py
new_cli=--delta-sets
new_cli=--gradient-delta-sets
new_cli=--gradient-backtrack-deltas
contract=proposal-only; final acceptance still requires rounded archive objective improvement
verification=5 focused tests passed, py_compile passed, git diff --check passed
```

Mathematical interpretation:

- `--delta-sets` tests sparse/asymmetric integer neighborhoods, useful when
  one direction or scale works better than another.
- `--gradient-delta-sets` estimates `d PoseNetLoss / d col0` through the
  differentiable renderer/PoseNet path without rounding, then proposes charged
  integer pose deltas along the measured descent direction plus a small
  backtrack guard.
- The final byte stream remains QP1/Brotli inside `archive.zip`; no scorer or
  hidden dependency is loaded at inflate time.

## Public Leaderboard / PR Refresh - 2026-05-02T01:27Z

Live web refresh against the public repo confirms the active public-floor
targets and the exact competitor claims to beat or exploit:

```text
repo=https://github.com/commaai/comma_video_compression_challenge
official_readme_leaderboard_lines=0.32 qpose14 (#63), 0.33 unified_brotli (#64), 0.33 quantizr (#55)
open_pr67=qpose14_qzs3_filmq9g_slsb1_r55, rounded 0.31, bytes 276564
open_pr65=henosis_qz_n3z_r25_clean, rounded 0.32, exact local approx 0.31968005, bytes 284425
```

Source notes:

- PR #67 reports PoseNet `0.00048597`, SegNet `0.00061000`, `276564` bytes,
  and says the payload includes QZS3 grouped variable-bit-depth quantization,
  delta/VLQ pose encoding, and a single blob:
  <https://github.com/commaai/comma_video_compression_challenge/pull/67>
- PR #65 reports `284425` bytes and exact local score approximately
  `0.31968005`:
  <https://github.com/commaai/comma_video_compression_challenge/pull/65>
- The official README defines the score formula, 30-minute T4 GPU evaluation
  rule, and current merged leaderboard:
  <https://github.com/commaai/comma_video_compression_challenge>

The current exact C-057 score `0.3157562807844823` is below the merged README
top rounded score band but still above the open PR #67 rounded `0.31` claim.
The immediate target is therefore not "be around 0.32"; it is to beat PR #67's
exact geometry, especially PoseNet around `0.00048597`, without giving back
the QZS3/QP1 byte advantage.

## C-056 Frontier Update And Pose-Manifold Implication - 2026-05-02T01:35Z

The r8 scalar continuation is now exact T4-promoted:

```text
claim=C-056
archive_sha256=c68b7522d4d2c8a89771e491c5956b0fb3460e744b3adba9f410f053783044b1
archive_bytes=276426
score=0.3159064496962538
pose=0.00049846
seg=0.00061244
hardware=Tesla T4
samples=600
promotion_eligible=true
```

This puts our exact promoted score below the rounded public `0.32` band and
within striking distance of the open PR #67 geometry. The remaining gap to the
PR #67-reported PoseNet `0.00048597` is now small enough that pose manifold
search is the highest-EV path:

- Use scalar radius search only until local objective improvements flatten.
- Then move to anisotropic radii: independent coordinate steps, gradient-signed
  integer deltas, hard-pair windows, temporal DCT/spline/jerk bases, and
  qpose residual atoms.
- Treat every accepted pose edit as a charged atom in QP1/Brotli or a successor
  pose payload; no uncharged runtime side channels.

The active r13 H100 diagnostic already reaches score `0.31514356926681697`
with PoseNet `0.00049102`, archive bytes `276423`, and SHA
`d3f3300531886d9dcb3553baffdd201567e3adaf7b746a7f405b15ad6c23b148`. Its T4
confirmation is queued as
`exact_eval_line_search_qzs3_qp1_r13_t4_20260502T0128Z`; until that lands, it
is a search anchor, not a promotion claim.

## C-057/C-058 Frontier And PR69/PR70 Quarantine - 2026-05-02T02:30Z

The anisotropic basis continuation landed as C-057, then a one-byte
active-subspace micro-update landed as C-058:

```text
claim=C-057
archive_sha256=63e6213ae154b5b5ce164829c15e675ad6d7819a9bdb1e8c9b2f099374fa7009
archive_bytes=276423
score=0.3157562807844823
pose=0.00049637
seg=0.00061244
hardware=Tesla T4
samples=600
promotion_eligible=true
artifact_dir=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_basis_r13_t4_20260502T0200Z

claim=C-058
archive_sha256=5145fb57be574b85639856d239420ffa35e605e32664f93e06753b120b21633f
archive_bytes=276422
score=0.3157555307844823
pose=0.00049637
seg=0.00061244
hardware=Tesla T4
samples=600
promotion_eligible=true
artifact_dir=experiments/results/lightning_batch/exact_eval_line_search_qzs3_qp1_active_fix2_t4_20260502T0250Z
```

PR #67 remains the external contest-faithful target:

```text
pr67_public_reported_rounded=0.31
pr67_archive_bytes=276564
pr67_pose=0.00048597
pr67_seg=0.00061000
pr67_formula_from_public_fields_approx=0.31486416405239354
c057_minus_pr67_public_fields_approx=0.0008920961309787923
```

This comparison is an external target comparison only. It does not promote or
demote PR #67 until its exact archive is reproduced through our custody path.

Public rule-boundary submissions checked during the report update:

```text
pr68=loophole_v2, closed proof-of-concept, 22-byte empty archive, payload moved into inflate.py
pr69=houdini, open, no filled maintainer eval report at inspection time, data-flow boundary refactor
pr70=mask_decoder, open, reports 0.19 and 57329 bytes, author says bytes were moved from archive into inflate.py
```

Quarantine rule:

- PR #68/#70 are invalid under this repo's stricter payload-closure standard
  because score-affecting data is script-side rather than charged in
  `archive.zip`.
- PR #69 is external/unverified boundary evidence until exact charged-payload
  eval and payload closure exist.
- These PRs can justify compliance hardening and report sections, but they must
  not enter frontier, ranking, promotion, or scientific-score tables.
