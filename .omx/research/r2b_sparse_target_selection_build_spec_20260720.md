# R2b sparse target-selection build specification

Date: 2026-07-20 UTC  
Lane: `lane_r2b_sparse_target_selection_20260720`  
Authority: delegated local-only BUILD + n600 measurement; no paid dispatch, launch, score, promotion, or pointer mutation.

## Outcome contract

Measure the capstone-to-exact-target `0.0471083800 S` nonrate gap at the frozen hard CPU-Torch scorer, decomposed into SegNet flip cells and PoseNet pair/dimension squared-error mass. Build a sparse, receiver-replayable decision stream over the existing V10 rounded two-plane descriptor, measure several reverse-waterfill knees in exact charged bytes, and admit it only if a real n600 byte-closed candidate passes the hard oracle while the stream stays at or below `70,748 B`.

## Representation

The existing C1 descriptor supplies rounded RGB scorer planes. A decision names `(pair, frame, scorer-cell, sign)`; the scorer cell's disjoint factor-2 camera block is replaced by a deterministic bounded-uint8 solution whose exact numerator moves to the chosen side of the same rounding bin. The magnitude is derived from the resize-coefficient gcd and bin boundary, so the receiver needs no source numerator or payload magnitude. Coordinates are sorted gap-ULEB, signs are bit-packed, and the complete stream is Brotli-compressed and parsed back before use.

The encoder may consult the source block only to choose the sign. The receiver re-derives both the target numerator and bounded block from the rounded plane plus the charged sign. Any infeasible block, changed rounded byte, parse-back difference, or non-exact numerator is `HARD_REJECT`.

## Measurement and gates

1. Re-run target and capstone raw bytes through one frozen CPU-Torch model at official batch geometry and require aggregate `d_seg=0.00015196`, `d_pose=0.00010184` within exact receipt tolerance.
2. Emit histograms by source class, target top1-top2 margin band, tie-tight/interior stratum, pair, and Pose dimension. Seg cells carry the exact additive `100/(600*384*512)` score mass; Pose rows retain squared-error mass and exact nonlinear cumulative score recovery.
3. Rank Seg decisions by the target Fisher proxy `abs(top1-top2 margin)`, then report stream bytes and oracle-scheduled recoverable score at several knees. Stop where measured marginal scheduled recovery per charged byte falls below `25/37,545,489`.
4. Decode the KKT-selected stream into a fresh SSD candidate raw, prove deterministic parse-back and exact bounded numerators, and run the real n600 hard oracle. Candidate recovery is authoritative only for `[macOS-CPU advisory]`; the scheduled curve remains explicitly an oracle upper bound unless each knee is scored.
5. Preserve receipt, stream, and reproducibility/cleanup record; candidate raw is rebuildable bulk and may be removed only after its bytes/hash and rebuild command are recorded.

## Verdict scope

Failure of this one-bit signed rounding-bin formulation is a `FORMULATION` negative only. Sparse decisions, multi-level magnitude codes, corrected inner-Jacobian selection, curvelet/shearlet carriers, and the compact shared receiver remain open; no negative here closes the family or paradigm.

## Triality

- DSL/receiver: consumes the existing rounded two-plane descriptor and factor-2 bounded-uint8 solver; no new launch lever.
- Equations: consumes `bounded_uint8_resize_preimage_cell_feasibility_v1`, Fisher/margin ranking, corrected inner-Jacobian guidance, and reverse-waterfill/KKT byte price; registers no new law without measured support.
- DAG: `M2 direct numerator rate-dead -> R2b sparse signed rounding decisions -> byte gate + hard oracle -> compact receiver follow-on or scoped negative`.

MAIN must independently review and merge any adopted result.
