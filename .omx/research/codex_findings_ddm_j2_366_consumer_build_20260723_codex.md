---
title: Codex findings - DDM J2 #366 executable consumer
utc: 2026-07-23T01:48:29Z
lane_id: ddm_j1_366_joint_descent_ticket
verdict: CONSUMER_READY_TICKET_REVALIDATION_OWED
verdict_scope: build and bounded real-cache consumer verification only
---

# Disposition

The J1 blocker `PREP_COMPLETE_EXECUTION_BLOCKED_BY_MISSING_CONSUMER_AND_REAL_PREFLIGHT` is closed at the consumer-build boundary. Commit `c5a418a286` supplies the hash-verified typed compiler, lossless V15-to-parameters adapter, low-dimensional MLX Seg/Pose objective, canonical resume-registry checkpoints, SSD/same-outdir/governor guards, actual n600-cache memory measurement, forced-kill resume proof, and governed dry-run.

# Measured findings

- Stage 00 re-emits exact archive bytes: 133,941 bytes, SHA-256 `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`; inherited d_seg is exactly `0.027470296224` by byte identity to the settled receipt.
- Typed lift: 163 island tracks, 2,197 knots, 2,047 shape templates, 6 Lane seeds, 6 shared row-band templates, 706 optimizer coordinates. The counted Lane-seed archive is 134,211 bytes (`+270`).
- Receiver ownership probes change both counted bytes and camera output for island, Lane, and shared-template groups. This closes the no-op/unowned-trainable-group instance gate.
- Fused R forward and both direct/custom VJPs are bit-identical to NumPy-fp32 on the per-chip gate; custom grouped backward reports active.
- Actual consumer peak RSS is 3.931182861 GiB. The preregistered conservative envelope projects 5.931182861 GiB, SAFE under 116 GiB; the system-aware governor admits it.
- Step 1 lowered pair-447 d_seg `0.0228118896484375 → 0.0227762870490551` and Pose MSE `38.90690994262695 → 38.906776428222656`; checkpoint then survived an intentional rc=23 process exit.
- A new process restored the canonical registry plus Adam/EMA/RNG/config state bit-exactly and lowered d_seg again to `0.022705078125` and Pose MSE to `38.906646728515625`.

# Checklist closure versus remaining authority

Closed by this branch: executable typed compile/hash; exact V15 warm-start lift and reemit; counted Lane seed; group receiver ownership; forbidden-payload boundary; SSD preflight; same-outdir single flight; atomic preserved stage checkpoints; canonical resume registry; forced-kill/new-process restore; sibling governed launcher; real actual-consumer memory receipt; system governor; fused-R/custom-backward gate; startup identity telemetry; stage-00 archive/camera identity; bounded real-cache descent; governed dry-run.

Still open by design: MAIN round-1 landing review; sealed-ticket revalidation after merge; separate operator GO; final long-campaign schedule/freshness review; chunked n600 exact stage-exit metrics and all per-stratum/rate telemetry; exact contest-CPU/CUDA box fork. These are launch/promotion gates, not consumer-build defects.

The keyed lane `ddm_j1_366_joint_descent_ticket` is now L2 with its five local build/empirical/preflight/review/memory gates bound to the receipt. The registry-wide validator still reports 110 missing legacy evidence paths that predate and do not involve this lane; the target-lane audit itself is internally consistent and its evidence path exists.

Evidence: `.omx/research/ddm_j2_366_consumer_build_receipt_20260723.json`. External immutable bytes live under `/Volumes/VertigoDataTier/pact/experiments/results/ddm_j2_366_consumer_build_c5a418a286_20260723T014300Z`. `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`; pointer `0.1910828242 [contest-CPU]` unchanged.
