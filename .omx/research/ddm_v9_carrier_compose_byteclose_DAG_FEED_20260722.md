# FEED — DDM V9 receiver-closed carrier composition

**research_only=true · score_claim=false · MAIN review required**

```text
v6 fixed receiver archive (exact bytes/hash)
  ├─ chart.zip
  │    └─ sole Pose6 member ───────────────────────────────┐
  ├─ Road mask + boundary/event members                    │
  ├─ Lane LBND2 + event/component members                  │
  ├─ Undrivable boundary/event members                     │
  ├─ Movable event member                                  │
  └─ MyCar static-hood member                              │
                                                            │
optional G2CS1 Lane coefficient symbols                     │
  └─ strict parse + canonical re-encode                     │
       └─ chart coefficient update                          │
            └─ generic region-coherent Lane rerasterizer    │
                                                            │
canonical merge order                                       │
Undrivable → Road → Lane → MyCar → Movable                  │
  └─ deterministic uint8 pair receiver ◀────────────────────┘
       ├─ frozen SegNet last-frame argmax, chunked full-P
       ├─ official PoseNet YUV6, chunked full-P
       ├─ exact outer + nested byte-home ledger
       └─ n64/n256 advisory receipts + n600 wall-clock projection
```

## Triality legs

- **DSL/config:** `DirectDescriptionV9CarrierComposeConfigV1` is strict, local-only, hash-bound, window-bounded, and forbids score/d_seg/d_pose claims.
- **DAG/receiver:** the outer archive consumes every nested role through one deterministic receiver. Pose6 has one inherited counted home. G2CS1 symbols act before rasterization; pixel residuals cannot be expressed.
- **Equations:** `.omx/research/ddm_v9_carrier_compose_canonical_equations_20260722.md` defines exact score, merge, chart correction, byte attribution, and surgical admission.

## Solver-stack wire-in disposition

This landing is explicitly `research_only=true`; it does not claim the six production hooks. It emits the following reusable feed:

1. **Sensitivity-map contribution:** Lane/Movable conditional errors plus frozen target-margin strata identify the next Fisher-margin solve surface.
2. **Pareto constraint:** a nonempty chart/event update must improve hard semantic cells and stay inside the measured Pose tube at exact byte price.
3. **Bit allocator:** nested unique-home rows establish present class/Pose byte occupancy; admission stops at `25/37,545,489` marginal score units/byte.
4. **Autopilot dispatch:** none; local-only authority and no paid dispatch.
5. **Continual learning:** canonical #603 append-note plus this FEED and result receipts.
6. **Probe disambiguator:** the next probe compares joint G2CS1 coefficients and xi-transported birth/death events under one hard receiver gate; neither interpretation is silently selected.

## Exact downstream blocker

`JOINT_FISHER_MARGIN_G2CS1_PLUS_XI_EVENT_SOLVE_OWED`: rank Lane/Movable chart and event DOFs by flip-distance × margin band × curvature, predict with corrected inner-Jacobian/secant custody, then admit only through exact receiver bytes + Seg/Pose. Current zeros are an honest non-admission, not evidence against the family.

Pointer `0.1910828242 [contest-CPU]` unchanged.
