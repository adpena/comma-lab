# ROAD-ANOMALY PROBE — crucible_v6 run-1 Road-class d_seg stuck at ~0.398

**Date:** 2026-07-08 · **Axis:** [macOS-numpy advisory, n600 existing telemetry] NON-PROMOTABLE ·
**Pointer 0.19110 UNMOVED (means).** · **$0 static + existing-telemetry only; run-1 (pid 63069) UNTOUCHED, read-only.**

## Question
Why does crucible_v6 run-1 start much worse + converge slower than #205 (d_seg 0.130 @ep125 vs #205 CE-floor
0.005 @ep225)? Decomposed: Road0 d_seg 0.398 nearly flat (0.424→0.398 over ep50→125) while ALL siblings
converge (Lane1 0.039, Undriv2 0.074, Movable3 0.0069, MyCar4 0.0028).

## STORES CONSULTED
- ORCHESTRATION_LEDGER.md (T5 crucible, run-1 = the island-birth arm) + launch.sh (the config-as-run).
- run.log stages: `verdict` (d_seg_by_class, flip_share_by_class), `handoff_readiness` (per-class part_frac —
  the decisive signal), `annulus_convergence` (interior-vs-annulus split), `structured_init`, `logit_adjust`,
  `island_seed/island_amplify/persistence_loss/seg_chroma_boundary`, `lane_render_band`.
- Code: `experiments/train_levelset_witness_realized_through_R_mlx.py` (lane-band gating L3499-3607) +
  `src/tac/witness_dsl/curriculum_dsl.py` (FEED-07a `DirectionalBasisRebalance`, `regime="lane_offloaded"` L2523-2557).
- CLAUDE.md canonical class order (Road0/Lane1/Undriv2/Movable3/MyCar4) — MEMORY L80.

## VERDICT: H1-mechanism FALSIFIED · H1-SPIRIT CONFIRMED by a DIFFERENT actuator · H2/H3 secondary/no

The operator's INTUITION (lane over-paint eats GT-Road → Road flips → Road pinned) is **CORRECT**.
The named MECHANISM (H1: the analytic `lane_render_band` compositing PRE-R) is **the wrong actuator** —
it is gated OFF until ep350. The real over-painter is the **rare-class-birth loss stack**
(`--seed-islands --witness-alone-island-loss --island-amplify --persistence-loss/recall
--logit-adjust`, all targeting classes [1,3]=lane,movable), which over-grows Lane+Movable INTO Road.

### The decisive signal — witness partition area vs GT (handoff_readiness.part_frac, ep125, n600)
| class | witness paints | GT area | ratio | 
|---|---|---|---|
| Road0 | 0.1407 | 0.2323 | **0.61× (UNDER)** |
| Lane1 | 0.0805 | 0.00585 | **13.8× OVER** |
| Undriv2 | 0.4679 | 0.4952 | 0.94× |
| Movable3 | 0.0568 | 0.0124 | **4.6× OVER** |
| MyCar4 | 0.2541 | 0.2541 | 1.00× (structured-init solved; hood IoU 0.993) |

**Mass conservation is exact:** majority DEFICIT (Road 0.0916 + Undriv 0.0273 = **0.1189**) ≈ rare EXCESS
(Lane 0.0747 + Movable 0.0444 = **0.1191**). The over-painted lane+movable area is stolen almost entirely
from Road. Road's within_flip floor (~0.40) IS the over-paint: the witness assigns ~9% of GT-Road pixels to
lane/movable, and every one is a Road flip. Flip mass @ep125 = 71% Road + 28% Undriv = 99% (rare classes ~0).

### Kill of H1 (the analytic band) — three independent nails, all $0
1. **Temporal:** `lane_render_band` stage logs `"start_epoch": 350`; trainer L3503/3516 composites only at
   `epoch >= start`. Current epoch 125 → **band NOT active**. Cannot cause an ep0-125 anomaly.
2. **Init:** `structured_init` logs `lane_px: 0, lane_static_mask_px: 0, part_frac{"1":0.0}` — the lane prior
   painted ZERO lane at init (lane not static → empty mask). No road-as-lane over-paint from init either.
3. **Arithmetic (the prompt's 10-min test):** Road needs 0.398×27.4M/600 = **~18.2K flipped px/frame**.
   The band is ~GT-lane-sized: GT lane = 690639/600 = 1151 px/frame; band recall 0.5475 → band ≈ 1-4K
   px/frame even generously. 4K << 18.2K, AND it's 0 while gated. The band is an order of magnitude too small
   AND temporally inactive. H1-as-mechanism is impossible for this window.

Note the `regime="lane_offloaded"` design (curriculum_dsl L2534): lane is SUPPOSED to be offloaded to the
free band, so the witness basis is starved along-tangent (`--freq-along 4` vs `--freq-across 32`). Result: in
the ep0-350 window lane is neither carried by the basis NOR painted by the band — yet the witness over-paints
lane 13.8× anyway. That isolates the actuator to the **loss**, not the representation.

### The actuator (mechanism, positively identified)
The loss deliberately births + amplifies the rare classes with growth-only (recall) pressure and no matching
precision/area cap:
- `logit_adjust` offsets `[Road −1.46, Lane −5.14, Undriv −0.70, Movable −4.39, MyCar −1.37]` — Menon
  adjustment massively boosts Lane/Movable in the loss AND de-weights majority Road/Undriv recall. (Verdict
  reads RAW logits, so we SEE the majority damage the loss stops penalizing.)
- `persistence_loss` target_classes [1,3], recall_weight 1.0; `island_amplify` classes [1,3], hinge,
  margin_target 1.0, persist=inverse_thickness; `seed_islands` support_frac 0.021 (shielded). All push
  lane+movable to GROW. The CE precision penalty that would say "don't paint lane on road" is weakened by
  logit-adjust. Net imbalance → runaway lane/movable growth into Road.

### H2 (basis starved road/horizon boundary): SECONDARY at most
Undrivable is the #2 error, and Road↔Undrivable is the largest shared separatrix. But Undrivable is
converging fine (0.149→0.074) with near-correct area (0.468 vs 0.495), and `annulus_convergence` shows
**79% of flip mass is INTERIOR, not annulus** (annulus_flip_mass_share 0.21). The dominant residual is
interior Road-area theft, not a starved boundary. H2 is not the driver.

### H3 (verdict artifact / EMA lag): NO
`handoff_readiness.part_frac` is computed on the witness partition (loss-side), SEPARATE from the EMA-shadow
verdict — both show the identical over-paint. The seg loss term is still 4.15 @ep125 (genuinely high). Not
an EMA/grading artifact. (The known d_pose EMA-lag IS present — d_pose 9.6→3.6→2.1 — but that's orthogonal
to the Road d_seg finding.)

### Flat vs slow
Road within_flip: 0.436→0.424→0.417→0.400→0.398 (ep25→125) — declining but decelerating to a **loss-imposed
floor** (~0.35-0.40) set by the maintained lane/movable over-paint. Effectively pinned, not analytic-flat.
It will NOT reach #205's CE-floor while the growth machinery over-grows unchecked.

## IMPLICATIONS

**(a) Run-1's value as baseline.** It is a valid, honest measurement of the **island-birth arm at full
strength** — the high-recall (over-grown) end of the lane/movable precision-recall tradeoff. Its converged
d_seg will be dominated by the Road over-paint floor (~0.10-0.13), NOT sub-0.01. Do NOT compare its total
d_seg against #205's CE-floor as if it were the same objective — #205 had NONE of this rare-class-birth
stack, so its Road/Undriv converged freely. Keep run-1 running to characterize island birth + the ep300/350
handoffs (seg-chroma-boundary ep300, tau ep300, lane-band ep350, muon ep726), but read it as the birth arm,
not a CE-floor attempt.

**(b) v7 config — the fix is NOT band width/mode/regime.** The band is not the actuator; a regime flip to
`lane_carried` or a band-mask would not touch the ep0-350 over-paint. The fix is **balancing recall with
precision on the birthed classes**, options (council to arbitrate, ranked cheapest-decisive first):
  1. **Add a precision/area penalty** on lane+movable: penalize the witness painting class∈{1,3} where GT∉{1,3}
     (a false-positive-area term). Directly opposes the runaway. Cheapest, most targeted.
  2. **Cap the growth to GT area** — make island-amplify/persistence-recall gates RETRACT once a class's
     part_frac reaches its GT area (a completion/retraction law on the per-class-λ homotopy; check whether the
     LADDER homotopy has a "stop-growing-after-birth" event — it appears to be missing, which is the M2-class
     "no completion guarantee" defect at the per-class level).
  3. **Soften logit-adjust** on lane/movable (offsets −5.14/−4.39 are extreme) — they de-weight Road recall
     more than needed to birth the rare classes.
  4. **Delay/ramp** the birth stack so Road/Undriv establish a clean majority partition first, THEN birth
     rare islands under a precision gate (curriculum ordering).

**(c) Operator's expectation gap: FULLY EXPLAINED.** crucible_v6 starts worse + converges slower than #205
because it is a fundamentally different (multi-objective, rare-class-birth) arm, not a regression. The
"nearly flat Road" is the rare-class-birth machinery over-growing Lane (13.8×) + Movable (4.6×) into GT-Road
with recall pressure and no precision counter-force, imposing a Road floor. It is a **design imbalance
(recall-without-precision), not a bug and not the lane band.** No H0 residual — the part_frac + mass
conservation close the account.

## Candidate law (for council, NOT registered here)
"Rare-class-birth with recall/persistence growth pressure and no matching precision/area cap imposes a
majority-class d_seg FLOOR = the conserved over-painted area, independent of capacity or training time."
Sister of the dash-erasure homogenization law (a floor law) but on the OVER-paint side. Registration deferred
to the council per triality discipline (this is a diagnostic, not a byte-closed measurement).

## Follow-up NOT run (and why)
The dynamic confusion matrix (Road→{Lane,Movable,...} destination) was NOT run: `part_frac` already IS the
destination signal (13.8× lane + 4.6× movable over-paint, mass-conserved from Road), the direction is clear
and load-bearing at n600 scale from existing telemetry, and run-1 is live + memory-contended (~6.5 GiB free
pages; ~55 GiB reclaimable) — no need to contend with the pointer-relevant run for a signal already in hand.
