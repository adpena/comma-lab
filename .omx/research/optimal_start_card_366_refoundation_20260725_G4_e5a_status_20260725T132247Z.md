---
schema: optimal_start_card_366_refoundation.g4_e5a_status.v1
date_utc: 2026-07-25T13:22:47Z
research_only: true
score_claim: false
pointer_moved: false
main_landing_review_required: true
---

# Optimal-start card G4 status delta — E5A

Disposition for the canonical card's G4 row:

> **G4 PASS, local research-only, pending MAIN landing review.** The copied
> step-50 checkpoint was materialized from its live-resume `theta` shadow into
> the exact receiver-closed WS1 state (138,813 bytes, SHA
> `2a2c0367150f8c8c0953dfb5c1485e238bbc9995c37385e149e52ae22f506241`).
> E5 compiled a deterministic 130,101-byte packet (SHA
> `fb69964da2649c310b7694416ff9863e13f54594af215cd771dcd50f5898a85d`);
> repository and embedded-runtime parse-back both reproduce the state
> byte-identically. LA1's 128,254-byte figure is **REBASED**, not confirmed as a
> complete packet: selected frames are 127,951 bytes, the canonical semantic
> bundle is 128,001 bytes, and complete receiver-closed packet cost is 130,101
> bytes. Exact batch-32 advisory n600 is `d_seg=0.06974277072482639`,
> `d_pose=35.499820809591` on `[macOS-CPU frozen-scorer advisory]`.

Rate comparisons are exact archive bytes:

- `130101 - 128254 = +1847` versus LA1's prospective non-container figure.
- `130101 - 130789 = -688` versus the coordinated post-CC3 byte reference.
- `130101 - 131294 = -1193` versus the prior E5 W_joint packet.

This row closes G4 only. It does not green G1-G3, G5-G6, authorize a campaign
fire, promote a score, or move the frontier pointer. The canonical card file is
absent from this older isolated branch and is hot on MAIN; MAIN must review and
apply this delta rather than accepting an add/add reconstruction.
