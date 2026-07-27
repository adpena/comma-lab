# G109 — fresh upstream-batch16 V9 training-target capsule

Status: BUILD + local fake-based verification only; full n600 materialization
has not been launched.

Authority: encoder-only target-input custody. This capsule is not a candidate,
score row, trainer configuration, or pointer mutation.

## Exact input coordinate

G109 consumes the compile-ready G46 receipt at:

`/Volumes/VertigoDataTier/pact/taskspace_fresh_teacher_batch16_20260726/12_encoder_only_receipt.json`

The typed config binds the receipt file SHA-256
`556c6b20f12ae7f5c6b8a1a2d08c6d2c2e32e7831ab552938c7866f3256dfad1`,
its sealed receipt/preflight/pair-checkpoint roots, the exact source video,
the G46 SegNet weights, the PoseNet weights, and the portable upstream source
closure. The production preflight recursively invokes G46's compile-ready
loader before it writes a G109 receipt.

Production geometry is fixed:

- 600 chronological non-overlapping source pairs;
- upstream evaluator default batch size 16, with one final 8-pair batch;
- source shape `(B,2,874,1164,3)` uint8 from `AVVideoDataset`;
- SegNet logits `(B,5,384,512)` float32;
- PoseNet target `pose[..., :6]`, shape `(B,6)` float32.

## Real materialization

One callback owns both scorer products for each source batch. It preprocesses
the same source tensor through `DistortionNet`, forwards SegNet and PoseNet,
then returns the SegNet logits and PoseNet first-six output together. Before
any batch checkpoint is committed, the core:

1. computes `argmax(SegNet logits)`;
2. requires exact equality with the corresponding G46 label slice;
3. computes `top1_logit - top2_logit`;
4. validates finite float32 margins and PoseNet-6 targets; and
5. atomically installs label, margin, and pose shards, then commits the
   self-hashed batch checkpoint last.

A mismatch is a refusal with the first global pair/pixel coordinate. No dense
batch bytes are committed on that path.

## Resume and storage

The 38 global batch checkpoints are immutable. On resume, chronological source
bytes are decoded and rehashed for every completed batch, while the expensive
scorer callback is skipped. If a crash happened after one or more batch arrays
but before the checkpoint, the missing batch is recomputed; existing arrays
must be byte-identical before the checkpoint can commit.

All production output is restricted to the Pact SSD waterfall. Preflight
requires 12 GiB free. Atomic scratch is success-cleaned. Crash-left scratch is
SHA-256 certified in a machine-readable cleanup receipt before deletion.
Durable batches and aggregates are never automatically deleted; cold-store
certification remains required.

## Aggregate and consumer ABI

Receipt schema:

`tac.taskspace_v9_training_target_capsule_aggregate.v1`

The aggregate owns:

- `seg_labels_n600.u8`, shape `(600,384,512)`, byte-equal to G46;
- `seg_top1_minus_top2_margin_n600.f32`, shape `(600,384,512)`;
- `source_pose6_n600.f32`, shape `(600,6)`; and
- deterministic `v9_training_target_capsule.npz` with members
  `seg_labels_u8`, `seg_top1_minus_top2_margin_f32`, and
  `source_pose6_f32`.

The drop-in strict consumer is:

```python
loader = V9TrainingTargetCapsuleLoaderV1.open(
    receipt_path,
    expected_sha256=receipt_file_sha256,
)
targets = loader.targets
labels = targets.seg_labels_u8
margins = targets.seg_top1_minus_top2_margin_f32
poses = targets.source_pose6_f32
```

The loader reopens the external receipt SHA, self-hash, preflight, recursive
G46 custody, source/model/upstream/runtime closure, every batch checkpoint and
raw array, the digest chain, and all deterministic NPZ members. Its memmaps are
the preferred trainer ABI; the NPZ is the compatibility projection.

The trainer checkpoint binding is the following exact projection. The external
receipt-file identity is deliberately separate from the receipt's canonical
self-hash:

```text
training_target_capsule.aggregate_receipt
  = {path, bytes, sha256} of 21_v9_training_target_capsule_receipt.json
training_target_capsule.aggregate_receipt_sha256
  = receipt["aggregate_receipt_sha256"]
training_target_capsule.preflight_sha256
  = receipt["preflight_sha256"]
training_target_capsule.batch_digest_chain_sha256
  = receipt["batch_digest_chain_sha256"]
training_target_capsule.g46_receipt_sha256
  = receipt["g46_custody"]["receipt_sha256"]
training_target_capsule.source_video_sha256
  = receipt["source_custody"]["source_video"]["sha256"]
training_target_capsule.segnet_weights_sha256
  = receipt["scorer_custody"]["segnet_weights"]["sha256"]
training_target_capsule.posenet_weights_sha256
  = receipt["scorer_custody"]["posenet_weights"]["sha256"]
training_target_capsule.arrays.seg_labels_u8
  = receipt["raw_arrays"]["labels"]
training_target_capsule.arrays.seg_top1_minus_top2_margin_f32
  = receipt["raw_arrays"]["margins"]
training_target_capsule.arrays.source_pose6_f32
  = receipt["raw_arrays"]["poses"]
```

Each array binding is exactly `{path, bytes, sha256, shape, dtype}`. A trainer
resume must compare the complete projection before loading optimizer, scheduler,
EMA, or stage state. Binding only the NPZ path, only the aggregate self-hash, or
only one of the three arrays is insufficient.

Known consumer bridge blocker: G105 currently requires
`margin_aggregate_schema =
"tac.taskspace_batch16_margin_base_scorer_aggregate.v1"` (the G78 schema).
G109 intentionally emits
`"tac.taskspace_v9_training_target_capsule_aggregate.v1"`. G109 does not alias
or impersonate the G78 schema, and it does not modify G105. A versioned G105
consumer bridge must explicitly admit and bind the G109 projection above before
the fresh V9 training route is launchable.

This closes target inputs only. A V9 launch remains forbidden until the typed
trainer DSL independently fixes the live batch/config coordinate and binds
this receipt SHA. Dense targets and scorer weights may never enter candidate
bytes.

Triality closure for this apparatus unit:

- DAG: exact G46 compile-ready source closure → G109 joint batch16 targets →
  G111 governed V9 training/checkpoints;
- DSL/control: the strict loader and checkpoint projection above are the only
  admitted target-input contract;
- equations: `label = argmax_k z_k`,
  `margin = max_k z_k - secondmax_k z_k`, and
  `pose6 = PoseNet(source_pair)["pose"][..., :6]`.

## Governed commands (not executed by G109)

```bash
uv run python tools/materialize_taskspace_v9_training_target_capsule_n600.py \
  .omx/research/configs/taskspace_v9_training_target_capsule_n600_20260727.json \
  --preflight-only

# Heavy n600 scorer replay: only through a governed admission environment.
uv run python tools/materialize_taskspace_v9_training_target_capsule_n600.py \
  .omx/research/configs/taskspace_v9_training_target_capsule_n600_20260727.json \
  --materialize
```

Pointer delta: zero. No n600 scorer forward or candidate/evaluator run occurred
in this landing.
