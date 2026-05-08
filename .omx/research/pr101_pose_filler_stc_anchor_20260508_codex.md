# PR101 Pose Filler-STC Anchor - 2026-05-08

## Scope

This records the CPU byte-anchor run for
`tools/pr101_pose_filler_stc_anchor.py`, using
`tac.codec.pose_filler_stc_codec` as a Filler/STC-inspired pose-codec
alternative to `lane_pd_v2`.

PR101 is monolithic and exposes no separate pose payload, so this is a
representative pose-distribution anchor, not a deployable PR101 archive
rewrite.

Score claim: `false`.

Ready for exact eval dispatch: `false`.

Dispatch blockers:

- `awaiting_filler_vs_pd_v2_dispatch_comparison`
- `substrate_pr101_monolithic_no_separate_pose_payload`

## Command

```bash
.venv/bin/python tools/pr101_pose_filler_stc_anchor.py \
  --output-dir experiments/results/pr101_pose_filler_stc_codex_20260508Tlocal
```

## Result

The baseline lab pose fixture exceeded the shared int8 quantizer ceiling, so
the tool fell back to the synthetic representative smooth random-walk fixture
used for PD-V2 comparison.

| Codec | Bytes |
|---|---:|
| Filler-STC blob | 3,960 |
| PD-V2 blob | 4,360 |
| Delta | -400 (-9.17%) |

Verdict: `byte_anchor_landed`.

Generated manifest:
`experiments/results/pr101_pose_filler_stc_codex_20260508Tlocal/build_manifest.json`
(ignored rebuildable artifact).

## Interpretation

This is a valid codec-byte signal for the representative pose distribution, but
it is not a PR101 archive promotion path until the monolithic substrate exposes
a separate pose payload or a full archive rewrite path is built.
