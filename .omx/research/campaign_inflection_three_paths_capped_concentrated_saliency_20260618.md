# Campaign inflection — three paths capped; sub-0.15 needs a concentrated-saliency vehicle (2026-06-18)

**The decisive verdict of the QAT-pivot cycle.** All `[contest-CPU advisory]`; pointer UNMOVED at 0.19110.
Stated plainly per the GOAL firewall: this campaign has NOT moved the pointer, and the three obvious
near-term paths to sub-0.15 are now all rigorously, measured-capped. The value is the ruling-out + the
design principle it reveals.

## The three capped paths (measured, this campaign)
1. **bc20 d_seg-descent** (the 50k margin-hinge run, stopped): d_seg capacity-floored ~0.0022 (best S 0.401
   > basin 0.378); the small basis can't reach frontier-grade d_seg. Capacity-limited.
2. **Path A — train a fresh higher-capacity decoder** (desk-calc): DOMINATED. Best modelled S_QAT 0.241 at
   bc36 — beats bc20 but loses to the 0.191 frontier; native capacity scaling forfeits the rate headroom.
3. **Path B — QAT-shrink the EXISTING frontier** (int5 score-aware QAT-finetune, 600-pair CPU-authority):
   CAPS at S=0.483. d_pose recovers fully under int5 (precision margin) but **d_seg plateaus ~0.0035 (6× the
   floor)** — the frontier's d_seg-critical structure is DENSE low-frequency capacity a coarse bit-grid can't
   represent. Score-aware allocation is second-order; the finetune is the dominant lever; CE > margin-hinge
   at the coarse grid (margin-hinge over-sharpens).

## The unifying diagnosis (why all three cap)
The capacity↔rate tension is IRREDUCIBLE for a **dense** decoder: d_seg-criticality is SPREAD across the
decoder weights/stages (flat saliency: bc20 5.5× spread, frontier geometry-verdict FALSE — the big early
low-res stages carry the d_seg signal, not a cheap output band). So:
- low d_seg ⇒ needs capacity ⇒ needs bytes (can't shrink — d_seg-critical structure is everywhere);
- low bytes ⇒ shrink/coarsen ⇒ spills d_seg (you coarsen d_seg-critical weights).
QAT proved this: it recovers d_pose (sparse, precision-marginal) but NOT d_seg (dense, structural).

## The design principle the cap reveals (the sub-0.15 vehicle)
**A CONCENTRATED-SALIENCY decoder: d_seg-critical capacity in a SMALL, high-precision, byte-cheap CORE;
d_seg-blind detail/recon in a LARGE, low-precision/prunable/shed PERIPHERY.** Then rate-win (shed/coarsen the
periphery) and d_seg-floor (protect the small high-precision core) stop fighting. Candidate mechanisms (the
design step ranks them):
- **Factored decoder:** a tiny d_seg-core (low-freq, the SegNet-decision band 96×128–192×256, high precision)
  + a cheap detail/recon adder (high-freq + pose, low precision/int4/prunable).
- **Saliency-concentrating regularizer in training:** penalize SPREAD |∂margin/∂w| (the margin_saliency_map),
  reward concentration — train the decoder to push d_seg-criticality into a designated small core, so the rest
  is freely shrinkable (makes QAT/pruning a net win post-hoc).
- **Structured low-rank/sparsity on the periphery** (the d_seg-blind detail) + dense small core.
- **Geometry-anchored d_seg-core:** the road↔lane boundary is low-dimensional (openpilot lane polys) → a tiny
  parametric/structured d_seg-core (#138 lane prior) + a cheap learned periphery.
This IS the capstone (#78 "OUR OWN small learned basis, extremely scorer-optimized"), now with the precise
principle: concentrate saliency so capacity and rate decouple.

## Honest framing + the strategic fork
- This is a deeper R&D direction (multi-day), NOT a near-term pointer-mover. The easy routes are exhausted.
- Incremental levers remain (the #137 boundary sidecar / #138 lane prior could give small d_seg cuts ON the
  frontier → a sub-0.19 pointer nudge), but they do NOT address the rate and are NOT the sub-0.15 path.
- The sub-0.15 path = the concentrated-saliency own-vehicle. Next step (MVP-first): a $0 DESIGN of it
  (architecture/regularizer ranking + a $0 saliency-concentration feasibility probe) BEFORE the multi-day
  build. The build is the bigger commitment that the design gates.

## Cross-refs
`campaign_math_review_*` (the master-gradient + capacity↔rate stationarity), `frontier_int5_score_aware_qat_finetune_*`
(Path-B cap), `capacity_rd_score_aware_qat_pivot_*` (Path-A dominated), `frontier_margin_saliency_qat_bitalloc_prior_*`
(flat saliency), `label_noise_floor_RESOLUTION_*` (frontier d_seg ~0.0003 floor), the SoT
`SESSION_SYNTHESIS_SoT_20260617_20260618.md`. Capstone task #78.
