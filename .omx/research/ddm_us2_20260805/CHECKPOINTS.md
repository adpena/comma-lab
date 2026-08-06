# ddm_us2 Checkpoints

## 2026-08-05 checkpoint 001

Mode: $0, read-only upstream/runtime inspection. No scorer run, no dispatch, no
archive mutation.

Loaded surfaces:

- Charter: `.omx/tmp/codex_runs/us2_prompt.md`
- Common contract: `.omx/tmp/codex_runs/_common_contract.md`
- Governing files: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`
- Recall: `.omx/research/ddm_us1_upstream_reread_20260731.md`,
  `.omx/research/ddm_ua2_upstream_defenses_and_budget_surface_20260731.md`,
  `.omx/research/ddm_pz1_pose_axis_cx1_base_20260803.md`,
  `.omx/research/ddm_na2_negative_audit_20260803.md`

Predict-before-read artifact created before upstream source inspection:
`PREDICTIONS.md`.

Current source-diff observations:

- `upstream/evaluate.py` confirms dynamic rate denominator from
  `rglob("*")`, per-batch torch-scalar accumulation, and Python score
  composition after `.item()`.
- `upstream/modules.py` confirms SegNet and PoseNet both resize by
  `torch.nn.functional.interpolate(..., mode="bilinear")` with no explicit
  `align_corners` or `antialias` kwargs, then Seg uses last-frame logits and
  Pose converts the resized float RGB pair through `rgb_to_yuv6`.
- `upstream/frame_utils.py` confirms `seq_len = 2`, camera size 874x1164,
  scorer size 384x512, complete-pair grouping, no file-size multiple check in
  `TensorVideoDataset`, and `rgb_to_yuv6` full-range BT.601 arithmetic.
- Local live R surface
  `experiments/train_witness_realized_through_R_mlx.py` routes through
  `apply_contest_faithful_roundtrip_nhwc`: render grid to camera bicubic,
  uint8 STE at camera resolution, then bilinear downsample to scorer resolution
  with no trailing uint8. The stale scorer-resolution-uint8 twin is deprecated.

Next read-only inspections:

- Scorer checkpoints and object topology under the upstream venv, without
  running a scorer forward.
- Contest runtime dependency closure and workflow wall.
- `upstream/videos/0.mkv` metadata and lineage facts.
