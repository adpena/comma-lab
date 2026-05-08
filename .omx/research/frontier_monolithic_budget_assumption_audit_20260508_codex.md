# Frontier Monolithic Budget Assumption Audit - 2026-05-08

Scope: adversarial search for code/docs/tools that still imply
ZIP-member-level mask, pose, renderer, or component budgets for PR101/PR106-style
HNeRV frontier archives after the monolithic archive finding. Evidence grade:
`empirical_archive_layout_cpu_no_score`. Score claim: false.

## Verification

Command:

```bash
.venv/bin/python tools/pr106_archive_decomposition.py \
  --summary-text \
  --output-json /tmp/frontier_monolithic_archive_layout_check_20260508.json
.venv/bin/python -m pytest src/tac/tests/test_frontier_archive_layout.py -q
```

Result: PR101 and PR106 are both single-member ZIP packets, and the focused
layout tests passed (`3 passed`).

- PR101 local archive:
  `experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip`
  has one member `x`; parser-proven sections are `decoder_blob` at offset `0`
  len `162164`, `latent_blob` at offset `162164` len `15387`, and
  `sidecar_blob` at offset `177551` len `607`.
- PR106 local archive:
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/archive.zip`
  has one member `0.bin`; parser-proven sections are `ff_header` at offset `0`
  len `4`, `decoder_packed_brotli` at offset `4` len `170278`, and
  `latents_and_sidecar_brotli` at offset `170282` len `15849`.

## Guardrails Already Correct

- `src/tac/frontier_archive_layout.py:1` states that single-member HNeRV packets
  invalidate ZIP-member-level component budgets but do not prove logical streams
  absent.
- `tools/pr106_archive_decomposition.py:2` reports physical ZIP custody plus
  parser-proven logical sections instead of filename categories.
- `tools/all_lanes_preflight.py:618` validates that PR101/PR106 frontier runs
  reject member-level component, mask, and pose budgets.
- `src/tac/codec_stack_planner.py:855` sets
  `member_level_component_budgets_valid=false` and requires internal parser
  proof for logical-stream budgets.

## Stale-Risk Findings

1. `.omx/research/pr106_pose_axis_forensic_memo_20260507_claude.md:50`
   says the "pose stream IS targetable"; lines `52`, `124`, and `136` treat
   PR101 as if it ships an explicit pose stream or `poses.pt`. Supersession:
   PR101 has no parser-proven pose ZIP member or `poses.pt`; any PR101/PR106
   pose-axis intervention must target decoder/latent/sidecar sections or emit a
   new runtime packet with charged-byte proof.

2. `tools/codec_op_param_sweep_manifest.py:227`-`228` uses
   "PR101 pose blob ~3,600" as an example baseline substream. Supersession:
   this is not valid for the stock PR101 archive. The tool can still sweep
   CodecOps on synthetic or alternate substrates, but PR101-specific manifests
   must cite `decoder_blob`, `latent_blob`, or `sidecar_blob`, not a pose blob.

3. `src/tac/tests/test_codec_pipeline_composability.py:5` and
   `src/tac/tests/test_codec_pipeline_composability.py:63` combine
   `Op1_PR101SplitBrotli` with `Op_KLPoseStream` on a synthetic state dict. This
   is acceptable as a generic pipeline smoke test only if not interpreted as
   archive-substitution readiness for PR101. `tools/pr101_archive_substitution_surgery.py:14`
   carries the stricter PR101 archive-surgery rule.

4. `.omx/research/public_hnerv_frontier_deconstruction_20260504_codex.md:233`
   proposes a PR106 stacking lane with a RAFT-derived pose sidecar. Supersession:
   this must be treated as a new charged sidecar/runtime-packet design, not a
   replacement of an existing PR106 pose member.

## Docs Clarified

- `docs/quantizr_archive_layout_confirmation_20260504.md` now narrows the PR106
  mismatch to absence of separate mask/pose ZIP members and the parser-proven
  HNeRV decoder-plus-latent surface.
- `docs/qpose14_seg_tile_actions_paradigm_extension_20260504.md` now narrows
  non-portability to absence of separate tile/mask/pose ZIP members and PR106's
  parser-proven global latent surface.

## Operating Rule

For PR101/PR106-style HNeRV archives, budget claims must name parser-proven
internal sections with offset, length, SHA-256, and decode/runtime consumption.
Member-name categories such as `masks.mkv`, `optimized_poses.pt`, `pose.npy`,
or `renderer.bin` are invalid unless the target archive actually exposes those
members.
