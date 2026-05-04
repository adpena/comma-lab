# Mixed/Local QZS Block Allocation - 2026-05-02 Codex

## Scope

Prototype builder:
`experiments/build_mixed_qzs_block_candidate.py`.

This is byte-screening only. Current QZS3 carries one global block-size field,
so local per-tensor block allocation is not exact-evaluable until a charged
runtime decoder exists. All emitted mixed/local candidates must keep
`score_claim=false`, `promotion_eligible=false`, and
`exact_evaluable_archive=false`.

## Candidate Contract

- Source archive and optional source evidence path are SHA/byte recorded.
- Existing helpers are reused for source unpacking, QZS3 decode, PR64
  mask-first single-blob packing, and QP1 pose packing.
- Mixed/local renderer payloads use prototype magic `MQZ1` with a deterministic
  JSON header containing policy, per-FP4-tensor block sizes, segment bytes, and
  SHA-backed archive provenance.
- Global `global:<block>` policies are exact-evaluable through the existing
  QZS3 runtime contract, but still remain non-promotable unless exact CUDA auth
  eval is run on the emitted archive bytes.

## Evidence Boundary

No remote jobs were dispatched for this prototype. Any bytes produced by this
builder are empirical byte-screen evidence only unless a later exact CUDA
`archive.zip -> inflate.sh -> upstream/evaluate.py` run records the exact
archive SHA, bytes, device, sample count, and component gates.

## Local Byte Screen - C-059 Source

Source archive:
`experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/archive.zip`.

Source evidence path:
`experiments/results/lightning_batch/exact_eval_qzs3_b32_maskfirst_qp1_fix1_t4_20260502T0331Z/contest_auth_eval.json`.

Outputs root:
`experiments/results/mixed_local_qzs_block_allocation_20260502/`.

| policy | bytes | sha256 | exact-evaluable | promotion |
|---|---:|---|---|---|
| `global:32` | `276347` | `cf44aa7fdb13b9ab6236aefde1f5f58e915d7dfa99128246235443841396c6ab` | `true`, existing QZS3 runtime | `false`, no new exact CUDA eval |
| `mixed:48:frame1_head=32,pose_mlp=32` | `276162` | `526e8ccb3e8bac957810e3f2065c6ff25074f900d9c7ea43fb26a607d5be27d0` | `false`, MQZ1 decoder absent | `false` |
| `mixed:64:frame1_head=32,frame2_head=32,pose_mlp=32` | `276075` | `3f4ab7fe9d7196de828cb5f58992e5aae6c7bef85911e6c0311e16460c988830` | `false`, MQZ1 decoder absent | `false` |

The mixed/local byte deltas are `-185` and `-272` bytes versus C-059. These
are not score claims; they only show that local block metadata plus protected
block choices can be packed deterministically under the current single-blob
accounting model.

## H100 Exact-Eval Dispatch - 2026-05-02T04:42Z

The mixed MQZ1 candidate is now exact-evaluable and was dispatched for fast
CUDA screening.

Candidate:

- Path: `experiments/results/mixed_local_qzs_block_allocation_20260502/mixed_64_frame1_head_32_frame2_head_32_pose_mlp_32/archive.zip`
- Bytes: `276095`
- SHA-256: `f35fe2c49a891f9c5dbb816f22a82dc8ac4dc318c0f46d8bf070324dd47cb4e0`
- Byte delta vs C-059: `-252` bytes, formula-only rate delta about
  `-0.0001678` if component distances remain unchanged.

Dispatch notes:

- Initial fresh H100 instance `35995924` failed before scoring on the
  lightweight NVDEC preprobe; no scientific conclusion was drawn, and the
  instance was destroyed.
- The archive was redeployed on known-good H100 instance `35995649` as
  `pact_eval_mixed_qzs_c059_reuse_20260502T0443Z` using explicit anchor
  staging and the canonical archive-only eval wrapper.
- Evidence remains non-promotable until the H100 exact JSON lands; T4 promotion
  is allowed only if component distances hold or improve.

## H100 Exact-Eval Result - 2026-05-02T04:46Z

The mixed MQZ1 candidate is A-negative for this measured implementation.

Artifact:

- Local harvest: `experiments/results/vast_harvest/archive_eval_mixed_qzs_c059_20260502/contest_auth_eval.json`
- Archive bytes: `276095`
- Archive SHA-256: `f35fe2c49a891f9c5dbb816f22a82dc8ac4dc318c0f46d8bf070324dd47cb4e0`
- Recomputed score: `2.079372901891654`
- PoseNet: `0.32721686`
- SegNet: `0.00086619`
- Hardware: H100 diagnostic CUDA, not T4 promotion.

Interpretation:

- The MQZ1 runtime path is closed enough for exact CUDA eval, but the measured
  mixed block allocation destroys pose geometry.
- The 252-byte rate win is swamped by PoseNet collapse, so no T4 promotion is
  justified.
- This does not kill mixed/local QZS broadly. It retires the measured
  `mixed:64:frame1_head=32,frame2_head=32,pose_mlp=32` policy and turns the
  next search requirement into scorer-aware block allocation: tensor groups
  must be chosen by component response or differentiable pose/seg sensitivity,
  not by byte delta alone.

---

## Component-Aware MQZ1 Builder - 2026-05-02T05:00Z

A component-aware follow-up to the byte-only MQZ1 failure was implemented and
byte-screened. It defaults to the exact-evaluated C-059 QZS3 b32 block size and
protects `shared_trunk`, `frame1_head`, and `pose_mlp`; only frame2-only tensor
groups move to block64. This is empirical until exact CUDA evidence lands.

Local outputs: `experiments/results/mixed_local_qzs_component_aware_20260502/summary.json`.

Candidates:

| policy | bytes | delta vs C-059 | sha256 |
|---|---:|---:|---|
| `component-aware-v1:frame2_pre64` | `276513` | `+166` | `7f4623c5ac9efe741f724d609afa4882ec3788109ee333ebc4bffd5d618dc6d6` |
| `component-aware-v1:frame2_block2_pre64` | `276393` | `+46` | `e00a28103ac42740d476646cde8eca7f87fa1628217915895e0fc81e551bb95e` |
| `component-aware-v1:frame2_all64` | `276282` | `-65` | `86684e1eff2c111c2f266f43a762952ae64976c8d25816b8935fdfbe42d36d52` |

Verification:

```text
.venv/bin/python -m pytest src/tac/tests/test_build_mixed_qzs_block_candidate.py -q
14 passed
```

Dispatch:

- The byte-positive `component-aware-v1:frame2_all64` archive was dispatched on
  known-good H100 instance `35995649` as
  `pact_eval_mixed_qzs_component_frame2_all64_20260502T0505Z`.
- This is a diagnostic exact CUDA screen only. T4 promotion is justified only if
  component distances remain inside the C-059 trust region.

## Component-Aware MQZ1 H100 Result - 2026-05-02T05:12Z

The component-aware follow-up is a scoped A-negative result for the measured
`frame2_all64` implementation. The source-coherence failure from the first
attempt was fixed by syncing the compact `fp4_block_sizes` decoder to the
remote H100 before rerun; the corrected rerun reached exact CUDA scoring.

Corrected H100 artifact:

```text
artifact=experiments/results/vast_harvest/archive_eval_mixed_qzs_component_frame2_all64_fix1_20260502/contest_auth_eval.json
archive_bytes=276282
archive_sha256=86684e1eff2c111c2f266f43a762952ae64976c8d25816b8935fdfbe42d36d52
score=0.5508608388775322
posenet=0.00818155
segnet=0.00080862
hardware=NVIDIA H100 80GB HBM3
samples=600
promotion=false
```

Classification:

- The measured policy saved only `65` archive bytes versus C-059 and increased
  PoseNet by more than an order of magnitude.
- The first failure (`MQZ1 header missing fp4_tensors`) was a remote
  source/candidate coherence bug, not scorer evidence. It is preserved under
  `experiments/results/vast_harvest/archive_eval_mixed_qzs_component_frame2_all64_20260502_failed_stale_decoder/`.
- The corrected failure is real A-negative evidence for this measured
  component-aware block policy. It does not kill mixed/local QZS; it says the
  next allocator must use direct component-response or differentiable
  pose/seg sensitivity before changing any tensor block size.

No T4 promotion is justified for this archive.
