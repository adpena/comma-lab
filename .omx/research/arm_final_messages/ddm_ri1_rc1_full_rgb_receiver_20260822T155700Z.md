RC1 is receiver-closed and folded at `INSTANCE` scope. The exact 113,006 B archive scored **17.306291367262396** on the completed n600 local advisory—catastrophically worse than DX2.

| Component | Result |
|---|---:|
| `d_seg` | 0.01605413 |
| `d_pose` | 24.41603851 |
| Seg contribution | 1.605413 |
| Pose contribution | 15.625632310405873 |
| Rate contribution | 0.0752460568565241 |
| Lane IoU | 0.1274806340345608 |

The actual copied DX2 shipping renderer produced and retained the 3.66 GB full-RGB output. Deterministic archive/token repeats matched, and mutations of all five paid sections were refused. The retained per-class post-pass reproduced both scorer components across all 600 pairs.

Artifacts:

- [Committed evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_ri1_rc1_full_rgb_receiver_20260822.md)
- [Sealed MAIN disposition](/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/SEALED_MAIN_DISPOSITION.json)
- [Advisory score receipt](/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/advisory_r1/contest_auth_eval.json)
- [Per-class retained diagnostic](/Volumes/APDataStore/pact/ddm_ri1_rc1_full_rgb_receiver/per_class_r1/RESULT.json)

Commits: `41a8583b16`, `96f6ee03be`, `5f9f3d9f47`.

The row is `[env-mismatch advisory]`, not contest authority. No Modal/GPU dispatch occurred; the sealed disposition is `FOLDED` with `dispatch_argv=null`. The pointer remains **DX2 S=0.14821987563243377 @ 180,368 B `[contest-CUDA T4, n600]`**.

## LIVE-HYPOTHESES

- A scorer-aware terminal program may preserve RC1’s rate advantage because RC1 optimized token agreement, not SegNet/PoseNet cells.
- Temporal/topological Lane and pose constraints may matter more than class allocation: CB2 found Lane already received 8.97× its area share, yet Lane and pose still collapsed.

## DEAD-ENDS

- This exact RC1 K=2,048/i3 plus shipping DX2 receiver is closed: `S=17.306291367262396`.
- Overall 98.796% token agreement and byte arithmetic cannot substitute for evaluator evidence.
- Simple class-area reweighting is closed by CB2’s measured allocation audit.
- Raising RC1 to K=4,096 remains byte-dead at 158,933 B.
- The existing in-evaluator retention helper is stale against the current `DistortionNet` API; RI1’s retained post-pass replaced it for this result.

