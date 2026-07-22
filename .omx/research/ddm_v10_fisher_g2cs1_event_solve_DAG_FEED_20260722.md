# FEED — DDM V10 Fisher G2CS1 + transported-event solve

**research_only=true · score_claim=false · MAIN review required**

```text
bound V6 fixed-AR1 receiver archive
  └─ V9 five-role semantic receiver
       ├─ sole counted Pose6 ordinal-code path ────────────────────────┐
       ├─ Lane LBND2 chart                                             │
       ├─ Road semantic mask                                           │
       └─ Lane/Movable semantic layers                                 │
                                                                        │
encoder-only frozen target cells + margins                              │
  └─ rank-4 pair norm + Fisher curvature + flip distance               │
       └─ mechanism-diverse candidate inventory                         │
            ├─ Lane G2CS1 c3 delta                                      │
            ├─ Road cubic boundary coefficients                         │
            └─ Lane/Movable birth/death bbox + Pose6 gains ◀────────────┘
                 └─ strict v10 packet parse + canonical re-encode
                      └─ semantic rerasterization, no pixel stream
                           └─ canonical-batch SegNet + official YUV6 PoseNet
                                └─ measured greedy rate/Pose admission
                                     └─ exact 0/5/15/40/100 KiB ladder
```

## Triality

- **DSL/config:** `DirectDescriptionV10FisherEventSearchConfigV1` binds local-only authority,
  source hashes/windows, canonical scorer batch, four-family inventory minima, exact added-byte
  budgets, pose containment, and immutable candidate/budget checkpoints.
- **DAG/receiver:** `direct_description_carrier_compose.py` extends the V9 receiver in place with
  strict `G2BC1` Road-boundary and `G2EV1` topology packets while retaining `G2CS1` Lane symbols and
  the sole nested Pose6 owner.
- **Equations:** `.omx/research/ddm_v10_fisher_g2cs1_event_solve_canonical_equations_20260722.md`
  defines acquisition, semantic transport, exact measured admission, rate, and scoped falsification.

## Solver-stack feed

1. **Sensitivity map:** candidate acquisition stores target/predicted class pair, exact head normal,
   target margin, Fisher curvature, and connected semantic support. Low flip distance receives high
   priority; no correct site receives value.
2. **Pareto constraint:** exact Seg error must fall, full objective distortion gain must exceed the
   exact rate term, and d_pose must not rise above the baseline tube.
3. **Bit allocator:** nested candidates are considered in measured admission order. Unspent requested
   budget is explicit vocabulary exhaustion; the allocator never fills with arbitrary symbols.
4. **Cathedral/autopilot:** no dispatch hook is enabled. Local `$0` measurements and `.not_a_candidate`
   archives only; MAIN must review before any downstream use.
5. **Continual learning:** the #603 append-note, ladder receipts, round-1 inventory-starvation finding,
   and this FEED preserve both positive and negative mechanism signal.
6. **Probe disambiguation:** all four candidate families are guaranteed representation in the bounded
   inventory. Exact scorer replay, not the proposal score, arbitrates them.

## Measured trajectory

- n64: `51,668 B / .045286496480 / 159.104827981350` →
  `53,021 B / .042511622111 / 159.093118922196`; 18/32 admitted, including two genuine
  Pose6-transported births.
- n256: base remains `72,397 B / .040169219176 / 157.798907948748`; 0/12 admitted because
  every positive-Seg Road candidate breached strict Pose containment.
- Requested 15/40/100 KiB rungs compile to the same exact selected bytes as the 5 KiB rung (n64) or
  base (n256), establishing an INSTANCE vocabulary plateau before the near-200 KB falsifier.

## Downstream blocker

`INSTANCE_VOCABULARY_EXPRESSIVENESS_BOUND`: expand structured shape/phase/transport DOFs without
introducing a pixel residual, then rerun the same exact admission law. If a residual carrier becomes
necessary, it must use a governed curvelet/shearlet representation and preserve strict receiver,
rate, Pose, and scorer custody. This result does not authorize a family closure or pointer movement.

Pointer `0.1910828242 [contest-CPU]` unchanged.
