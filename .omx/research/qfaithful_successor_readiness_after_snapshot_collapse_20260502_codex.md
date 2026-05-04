# Q-FAITHFUL Successor Readiness After Snapshot Collapse - 2026-05-02

Date: 2026-05-02T10:46:51Z

Evidence grade: design/readiness review plus exact-artifact synthesis.
Score claim: false.
Remote jobs launched: false.
CMG3 files touched: false.

## Exact Evidence Boundary

The active frontier anchor remains C-067 exact Tesla T4 custody:

- artifact:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.json`
- archive bytes: `276214`
- archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`
- score recomputed from components: `0.31561703078448233`
- PoseNet: `0.00049637`
- SegNet: `0.00061244`
- samples: `600`

The old Q-FAITHFUL snapshot family is a measured implementation collapse, not a
family kill:

- `qfaithful_snapshot_qzs3_rp2_qpose14_l40s`: `276651` bytes,
  score `8.420915711675079`, PoseNet `4.47971487`, SegNet `0.01543638`.
- `qfaithful_snapshot_2131_pr64_qpose14_l40s`: `276542` bytes,
  score `20.910165502494504`, PoseNet `40.93836975`,
  SegNet `0.00492795`.
- `qfaithful_snapshot_2146_pr64_qp1_l40s`: `273103` bytes,
  score `22.065520725118258`, PoseNet `46.18100739`,
  SegNet `0.00393906`.
- `qfaithful_zoom_runtime_fix_h100sxm`: same geometry-closed archive SHA
  `f64dcb3d12db394efa9b0e0f924bb62b6b24f096d66baf9ed83447077d4f9b61`,
  `274257` bytes, patched runtime consumed charged `zoom_scalars.bin`, but
  score stayed `22.147631187370024` with PoseNet `46.54520035`.

The rate side is already close enough: a `274257` byte successor has rate
contribution `0.18261647890642735`, leaving only `0.13300055187805498` total
distortion budget to beat C-067. With C-067-like SegNet, PoseNet must remain
around `0.000515` or lower. The next run is therefore a geometry/distortion
repair problem, not a packer-only problem.

## Current Blockers

1. The current Q-FAITHFUL training path surfaces `pose_dim=6`, but
   `train_renderer.py` does not feed a loaded pose stream into
   `_QuantizrFaithfulShim.forward(...)`; if no pose is supplied, the shim builds
   zero poses. A successor dispatch is not contest-faithful until training uses
   the exact deployed nonzero pose stream or fails closed.
2. Post-hoc charged `zoom_scalars.bin` closure is insufficient for the old
   snapshot. The runtime now consumes it, yet exact H100 score remains `22.1476`.
   A successor must either train/export a geometry-aware half-frame contract
   from the start, or use a full-frame control as a diagnostic upper-bound.
3. The legacy `scripts/remote_lane_q_faithful_jointgen.sh` is not the next-run
   launch surface as-is. Its script/test expectations around half-frame mask
   construction and QFAI export paths disagree, and the script predates the
   stricter snapshot contract.
4. Full-frame masks are useful as a control, but likely not the primary beat
   path unless archive bytes stay near C-067. With components matching C-067,
   the archive must be below `276214` bytes; larger full-frame archives require
   unrealistically large component gains.

## Deterministic Next-Run Design

Primary run: `q_faithful_successor_geom_v1`.

- Architecture: `variant="quantizr_faithful"`, JointFrameGenerator,
  `pose_dim=6`, no zero-pose fallback in training or export.
- Training contract: five-stage QAT++ using the existing profile schedule
  `600/1500/400/400/100` epochs at `1e-3/5e-4/1e-4/5e-5/1e-5`, EMA decay
  `0.997`, `eval_roundtrip=True`, FP4 residual codebook, robust scale, and
  stochastic QAT.
- Data contract: train against the exact deployed mask and pose stream. For
  half-frame successors, the mask expansion or renderer geometry consumed at
  inflate must be the same geometry simulated during training. For full-frame
  controls, declare `mask_frame_contract=full` and do not require zoom geometry.
- Export contract: pack the EMA shadow, not live weights. The export metadata
  must record checkpoint SHA, EMA/live selection, pose source path/SHA/shape,
  mask source path/SHA/frame contract, QAT codebook/scale policy, and exact
  archive member sizes/SHA.
- Packer contract: deterministic archive bytes through
  `experiments/repack_quantizr_faithful_qzs3_archive.py` or a successor with the
  same safe-ZIP checks. All score-affecting geometry and pose bits must be in
  `archive.zip`; no scorer, sidecar, host path, or network input at inflate.

## Required Gates Before Any Dispatch

Run these locally after the narrow pose-stream/export patch lands:

```bash
.venv/bin/python -m py_compile src/tac/experiments/train_renderer.py scripts/q_faithful_snapshot_loop.py experiments/repack_quantizr_faithful_qzs3_archive.py src/tac/quantizr_faithful_export.py src/tac/quantizr_faithful_renderer.py
.venv/bin/python -m pytest src/tac/tests/test_q_faithful_snapshot_loop.py src/tac/tests/test_inflate_renderer_zoom_geometry.py src/tac/tests/test_quantizr_faithful_renderer.py src/tac/tests/test_quantizr_torch_fp4_codec.py -q
.venv/bin/python -m pytest src/tac/tests/test_remote_lane_q_faithful_jointgen_script.py -q
```

Add or repair focused tests so they prove:

- Q-FAITHFUL training passes nonzero pose tensors into the generator and rejects
  missing poses for `pose_dim>0`.
- QFAI/QZS3 export metadata records `packed_from_ema_shadow=true` plus pose,
  mask, QAT, and frame-contract SHA custody.
- Full-frame snapshots remain promotable without zoom geometry, while
  half-frame snapshots are promotable only when charged geometry is preserved
  and consumed by the runtime contract.

Before any future remote training or exact eval dispatch, claim the lane:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id q_faithful_successor_geom_v1 --platform <platform> --instance-job-id <instance-or-job-id> --agent codex:qfaithful_successor --predicted-eta-utc <YYYY-MM-DDTHH:MMZ> --status training --notes "five-stage QAT++ Q-FAITHFUL successor; no score claim until exact CUDA JSON"
```

Promotion gate remains exact CUDA:

```bash
WORKSPACE=/workspace/pact ARCHIVE_PATH=/workspace/pact/<candidate>/archive.zip ARCHIVE_LABEL=q_faithful_successor_geom_v1 LOG_DIR=/workspace/pact/<candidate>/exact_cuda PREDICTED_LOW=0.25 PREDICTED_HIGH=1.0 CONTROLLED_BASELINE=C-067_A++_T4 bash scripts/remote_archive_only_eval.sh
```

Do not spend T4 promotion time unless a fast CUDA diagnostic or local byte/math
screen shows archive bytes near C-067 and component distances plausibly near
PoseNet `0.0005`, SegNet `0.00061`.
