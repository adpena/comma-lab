**Charter incomplete: `nice 10` was denied by the sandbox.** No segment encode or draw ran.

Verified the **119,784 B envelope**—119,779 raw bytes plus 5 framing bytes—and all **213,733 retained locations**. **21,311 locations remain missing.**

The [memo](/Users/adpena/Projects/pact/.omx/research/ddm_bnd2_boundary_segment_code_real_encode_20260910.md) and [handoff](/Users/adpena/Projects/pact/.omx/research/ddm_bnd2_20260910/FINAL_HANDOFF.json) preserve the evidence. Two code reviews and Ruff passed. Git writes were denied; verified fallback commit **`6dbbcad245`** is bundled, **not landed on main**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — ddm_bnd2:** finish the encoding experiment; consumer `.omx/state/canonical_task_status.jsonl`, bnd1 `::ITEM_1`; trigger: verified nice 10 or explicit operator constraint change.
- **FOLDED — MAIN coordination, ddm_bnd2 execution:** perform the draw; same consumer, bnd1 `::ITEM_2`; trigger: twin-verified lossless total ≤114,784 B and scorer allocation.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** land the verified bundle; same consumer, bnd2 `::LANDING`; trigger: harvest in a Git-writable context.

## LIVE-HYPOTHESES

- Segment coding may save bytes because misses concentrate near edges; complete coherence and actual price remain untested.
- The existing gb2 generator successor may create longer coherent structures rather than address isolated misses.

## DEAD-ENDS

- Ancestor tc1 traces cannot supply current evidence: field hashes differ.
- Attributed coding costs are not detachable payloads; achieved lengths are not lower bounds.
- One losing grammar cannot establish that every richer grammar loses.

Own-vehicle frontier unchanged: **S 0.13791730003757818 @ 180,388 B [contest-CUDA T4 n600]**.

