---
schema: ddm_ic2_optimal_incumbent_pose_typed.dag_feed.v1
date_utc: 2026-07-24
research_only: true
score_claim: false
main_review_required: true
---

# IC2 DAG + FEED

```text
W_seg exact WS1 archive (SHA 264a09ab...)
  |
  +-- nested_preuint8_archive ----> receive_preuint8_q8_archive
  +-- warm_start_payload ---------> temporal affine + frame1 reassert
  |
  v
PA1(frame0) moment-derived affine [0 counted bytes; frame0 Seg-free]
  |
  v
IC2 typed exporter -> Brotli-Q11 .ddj5 -> parse-back/byte-home proof
  |
  v
inflate.sh declared-dependency bootstrap -> receiver -> 38+38 preserved stages
  |
  v
frozen n600 scorer, batch32, threads4
  |
  +-- d_seg 0.024124510023328993
  +-- d_pose 65.03498712932134
  +-- bytes 131154
  +-- S_adv 28.00173925293584
  |
  v
compare to v0 S_adv 23.66179213623354
  |
  +-- nonpromotion / pointer unchanged
  +-- blocker: compact code->photometry pose inverse absent
```

E2 `nested_pose6` terminates before the packet boundary; it has no outgoing
receiver-owned edge and therefore cannot enter the composition DAG. #601/#605
terminate as scoped controls. No inferred edge is added.

## FEED

- Sensitivity-map contribution: measured PA1 interaction binds the W_seg parent
  to a Pose response (`delta d_pose=-81.3299453255564`) while preserving Seg;
  this is an interaction row, not an additive prior.
- Pareto constraint: reject IC2 because `delta S=+4.339947116702302` versus v0
  despite 428 fewer bytes and lower d_seg.
- Bit allocator: allocate zero additional bytes until a receiver-owned pose
  coordinate has measured negative joint reduced cost on the W_seg parent.
- Cathedral/autopilot: candidate status is
  `MEASURED_NON_INCUMBENT__COMPACT_POSE_CARRIER_ABSENT`; no exact dispatch.
- Continual-learning posterior: the PA1/W_seg interaction is a scoped negative
  for this composition only; compact xi/YUV6 carrier successors remain open.
- Probe disambiguator: future carrier must compare `hold`, `xi-only`,
  `YUV6-only`, and `xi+YUV6` under the same W_seg archive, coder, batch32
  scorer, and exact receiver. Component measurements may rank acquisition but
  cannot decide admission.

