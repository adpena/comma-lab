# DDM E2 export DAG / FEED — 2026-07-23

`research_only=true` · `score_claim=false` · pointer unchanged.

## Executable DAG

```text
seeded state archive (134,211 B; not a packet member)
  ├─ chart anchors / gradients / residuals
  ├─ semantic composed labels
  └─ pose6 target (3,600 B; pre-export apparatus only)
       ↓
E2 typed exporter
       ↓
counted packet
  ├─ manifest.json              9,592 B home
  ├─ base/chart.ddb            18,513 B home
  ├─ semantic/composed.dds    315,153 B home
  └─ ZIP structure                208 B
       ↓
strict E2 receiver
  ├─ complete audit-triple gate
  ├─ complete ordered-pair redundancy gate
  ├─ single-owner/correction gate
  └─ frame1-only semantic home
       ↓
3,662,409,600 raw RGB bytes
       ↓
SegNet(frame1) + PoseNet(frame0, frame1), batch 16
```

The pose branch stops before the counted packet: `BLOCKED_NOT_PRESENT(compact code→photometry inverse)`.

## Sensitivity-pricing FEED

```text
DDMRuntimePerturbationV1
  → strict counted packet reopen
  → edit one chart/semantic coordinate
  → serialize actual .ddb/.dds member
  → parse back with receiver blob contract
  → realize baseline and perturbed RGB
  → batch-16 SegNet/PoseNet
  → {delta d_seg, delta d_pose, delta bytes, delta S/byte}
```

Consumers: g2 compact-shearlet/rank-4 costate ranking; p581r sparsemax/Hopfield probes; E2 audit-clause sensitivity tolerances. The API refuses receiver-inert edits and records the unique stream owner of every output change.

## SN1S source-class (ii) stage FEED

```text
perfect target argmax
  → native-grid paint argmax
  → float bicubic-up / bilinear-down R argmax
  → camera round/clamp uint8 R argmax
  → official SegNet.preprocess_input + scorer argmax
       × exact packet fact-owner partition
         ├─ chart: semantic code == 0
         └─ semantic: semantic code > 0
       → {argmax Δ, introduced errors, corrected errors, errors after}
```

The 38 preserved batch rows aggregate to chart `852,384` and semantic `2,523,156` final errors. Their `3,375,540 / 117,964,800` closure exactly reproduces `d_seg=0.028614807129`. The official scorer-consumption transition has zero argmax differences because its input tensor is exactly equal to the manually factored uint8 R-down tensor. SN1S can consume these source-class (ii) rows from `ddm_e2_runtime_verification_receipt_v2.json` without replay.

## Triality

- DSL: E2 typed configs and `DDMRuntimePerturbationV1`.
- DAG: the three flows above, with the pose break explicit.
- Equations: `S=100 d_seg + sqrt(10 d_pose) + 25 B/37,545,489`; marginal admission at `-ΔS/ΔB > 25/37,545,489`; ordered redundancy `bytes(B)-[bytes(A||B)-bytes(A)]`; stage conservation `E_after=E_before+E_introduced-E_corrected`.

## FEED verdict

E2 is a verified receiver/export apparatus and a sensitivity instrument. It is not a promotable candidate: the per-coordinate/per-boundary tolerance fields and compact pose inverse remain owed. MAIN must review this branch before landing.
