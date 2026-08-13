# ddm_vd1 run-c census verdict — gen-1 event alphabet REFUTED at n600 authority; falsifier fired → gen-2

**Date:** 2026-08-12 · **Owner:** MAIN (Fable) · **Axis:** [contest-CUDA T4 exact-upstream affected-pair n600 delta]
**Modal:** fc-01KZWM334XKMRQE4YFJ5PX3574 · run_id `ddm_vd1_20260812c` · Tesla T4 · 677.95 s · rc=0 · status COMPLETE
**Ledgers:** call-id `harvested` + lane claim `completed_harvested` (modal:ddm_vd1_20260812c) — both closed 2026-08-12.

## Custody

- Base: cp135 composed archive, sha `6eb1a3b79cb167e03372339e07e93cae13b6ba3114a9eb917288bb038622edb6`, 186,252 B (the effective frontier, S 0.16195513827824176).
- Results: `/Volumes/VertigoDataTier/pact/ddm_vd1_20260812/main_harvest/results/FINAL_RESULT.json`
  (sha `6c53628184f55722f87fcb7e3dadc8b6c9a70025a804e00cfcbecb6674004973` — matches the harvested `final_result_sha256`)
  + `EVENT_RESULTS.jsonl` (sha `a97400d32878318d8eb657a36e62f523e4db48e402b292c09e611d2104b500b3`, 646,298 B, 200 rows).
- Full payload retention on volume `comma-ddm-vd1-event-validator-retained/ddm_vd1_20260812c` (base token plane, all per-event payloads/tokens/frames/scorer tensors/deltas, all GT batches) — P0 KEEP-THE-PAYLOAD satisfied.
- Harvested return persisted: `main_harvest/MODAL_RETURN.{pkl,json}`.

## The measured verdict (per-event exact, all 200 gen-1 ec1 events)

| Quantity | Value |
|---|---|
| Events measured | 200 (K arithmetic held: 677.95 s actual vs 935.916 s predicted vs 1,800 s budget) |
| Affected pairs | 6 of 600: [7, 18, 53, 73, 76, 96] |
| Net-flip-positive events | 26 / 200 |
| Pose-budget-pass events (per-event ≤2.95e-9 global d_pose) | 29 / 200 |
| **Eligible (intersection)** | **5** — ec1_0164, ec1_0168, ec1_0004, ec1_0104, ec1_0003 |
| Optimistic additive seg gain | +6 flips = **5.086263020833333e-06 S** |
| Additive pose of the 5 | 6.539e-09 (≪ 1.3e-7 global budget) |
| **Pre-registered falsifier (bar 0.000216)** | **FIRED — 42× under the bar** |

**Mechanism:** the binding constraint is POSE, not seg reach. Flip-positive and pose-safe events are nearly
disjoint (26 ∩ 29 = 5). The js7 calibration (marginal 603 S/unit at base d_pose 6.88e-6 → stack budget
1.3e-7) is what the gen-1 alphabet cannot live inside. Verdict scope: INSTANCE (gen-1 ec1 alphabet on the
cp135 base) — the sparse-event FAMILY stays open pending gen-2; the CARRIER is proven free (jo1: +3 B / 200
events) and the VALIDATOR is now a proven reusable n600-authority instrument.

## Routing (both fired 2026-08-12)

1. **ddm_cp5v** (codex, rank 1): compose the 5 eligible events at +≤3 B per the census contract
   (`main_final_exact_row_required: true`) → MAIN buys ONE final T4 row (~$0.15, #381). Purpose: close the
   loop with authority AND calibrate additivity-through-exact-compose — the step that killed js7
   (projection −0.00058 vs realized +0.00147). Expected row if additivity holds: ~0.1619501 (−3e-6 net).
2. **ddm_gv2** (codex, rank 2): gen-2 alphabet — pose-null-BY-CONSTRUCTION (js4 projector / Q3 placement)
   events on the Road↔Lane hub edge (49.2% of flips; lc1's 5,557 Lane→Road over-claim receipt as the
   anti-pattern). Same store schema → the vd1 validator re-fires UNCHANGED. Pre-registered falsifier: best
   optimistic eligible gain < 0.000216 S ⇒ sparse-event family FORMULATION-closed on this base; seg leg
   routes to implicit edge conditioning (js1 stage-0).

## #381 spend (this vd1 chain, 2026-08-12/13)

Run-c T4 677.95 s ≈ $0.11 + 8 failed dispatch attempts (died pre/early-GPU, container-minutes) ≈ $0.05–0.10.
**Chain total ≈ $0.20.** Next planned: cp5v final row ~$0.15 + gv2 validator re-fire ~$0.15.

## Pointer

Effective frontier UNMOVED: cp135 S 0.16195513827824176 @ 186,252 B [contest-CUDA T4 n600].
Own-vehicle: lc2 0.16959899569230852 @ 187,226 B. This unit bought the INSTRUMENT row (per-event exact
census), not a pointer move — the composed row is the next exact candidate.
