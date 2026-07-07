# Islands treatment arm — the OWED composed-surface ceiling arithmetic ($0, offline)

**Date:** 2026-07-07 · **Axis:** `[macOS-CPU/MLX advisory] NON-PROMOTABLE` — pointer 0.19110 UNMOVED (means).
**Owed by:** FEED-07c CAVEAT ("the exact composed-surface ceiling is SYMPOSIUM arithmetic, owed before
launch") + the T3 symposium `.omx/research/council_t3_symposium_islands_treatment_arm_20260706.md`.
**Gate:** if the band's LOWER edge < 0.005 total-S improvement → STOP (launch not worth it).

## Verdict first

**GATE PASSES — PROCEED.** Predicted ΔS ceiling band for full island treatment on the COMPOSED
surface: **[~0.02 conservative lower edge, ~0.26 upper]** (seg-term S units, advisory surface).
The lower edge is ≥4× the 0.005 gate under every conservative derating tried below.

## Inputs (all labeled)

| Quantity | Value | Provenance |
|---|---|---|
| Witness-alone d_seg (ep225, n600) | 0.025734 | MEASURED — `lane_share_probe_ep225_n600.json` |
| wa island share of flips | 0.639 (lane 0.1909 + movable 0.4481 + boundary overlap) | MEASURED (same probe; self-flagged UPPER bound for composed) |
| wa within_flip lane / movable | 0.8391 / 0.9315 | MEASURED (same probe) |
| Live composed verdict d_seg | ep225 0.004869 · ep300 0.004571 · ep350 0.003953 · BEST ep425 0.003636 | MEASURED — trainer n600 verdict (`annulus_live.jsonl` mechanics + `costate_shadow.jsonl` + `levelset_best.json`) |
| **Composed per-class (ep300)** | table below | **MEASURED HERE** — `annulus_live_maps/maps_BEST_ep300.npz` argmax (16-pair strided subset, the SAME maps behind the `annulus_live.jsonl` row) vs `gt_n600.npz['lstars']` |

### The live composed per-class data (MEASURED in this memo, ep300, 16-pair advisory subset)

The annulus renderer (`witness_annulus_convergence.py` → `attr.process_ckpt`, the REAL
render-through-R + frozen CPU-torch SegNet path, self-orient fixed point mean_iters=4.0)
reproduces the live verdict surface: subset d_seg **0.004785** vs trainer n600 **0.004571**
(+4.7% subset error) — so, contra the probe's "composed surface unreconstructable" caveat, the
composed surface IS reconstructable from the ckpt via the annulus path, and its per-class
decomposition is computable directly (not merely boundable):

| class | GT area | within_flip | share_of_d_seg | part_frac (pred mass) | mass vs GT |
|---|---|---|---|---|---|
| 0 Road | 0.23350 | 0.00646 | 0.3153 | 0.23480 | +0.6% |
| **1 Lane** | 0.00577 | **0.36478** | **0.4396** | 0.00447 | **−22.5% (deficit)** |
| 2 Undrivable | 0.49466 | 0.00079 | 0.0822 | 0.49489 | +0.05% |
| **3 Movable** | 0.01109 | **0.05288** | **0.1226** | 0.01083 | −2.3% |
| 4 MyCar | 0.25498 | 0.00076 | 0.0404 | 0.25500 | +0.0% |

**Islands (lane+movable) = 0.5622 of composed d_seg.**

## What this RESOLVES (and corrects)

1. The probe's wa island share **0.639 is confirmed as an UPPER bound**: measured composed share
   = **0.562**. The transfer loss is small — the ceiling survives composition nearly intact.
2. **The "both rare classes fully unborn on the composed surface" premise is CONTRADICTED at
   ep300** (it was true at ep0 / on the witness-alone surface): composed part_frac is 77% of GT
   mass for lane, 98% for movable. Plain CE + the fixed schedule DID birth most of the rare-class
   mass on the composed surface. What remains is: **lane = THE composed-surface target** (44.0% of
   d_seg; 36.5% of GT lane px still flip; 22.5% predicted-mass deficit ≈ missing dashes = residual
   un-birth + boundary jitter), **movable largely solved** (5.3% within-flip). This does NOT
   deflate the arm — it sharpens its mechanism: the headroom is dominated by lane dash
   birth/placement, exactly what SeedIslandEased (VP-tangent along-lane widening) + the
   nucleus-guarded hand-off target. The prompt's lower-bound branch ("both rare classes fully
   unborn on the composed surface") is therefore NOT derivable — it is empirically false at ep300;
   the direct measurement replaces the bound.
3. Consistency check the wa shares could never pass: if composed part_frac were truly 0, composed
   d_seg would be ≥ area(lane)+area(movable) = 0.0182 > the measured 0.0046 — the premise was
   arithmetically impossible. The composed measurement closes that contradiction.

## The ceiling band (ΔS = 100·Δd_seg; treatment adds ~0 archive bytes — loss/schedule flags only;
w_pose=0 in both arms ⇒ pose term unchanged ⇒ ΔS_total ≈ ΔS_seg on this advisory surface)

**(a) Prompt's upper bound (wa shares × composed d_seg, ep225):**
100 × 0.639 × 0.004869 = **0.311** — DERIVED, loose (superseded by the measured composed share).

**(b) MEASURED-composed full-island-fix ceiling:**
- at ep300 (n600 d_seg 0.004571): 100 × 0.5622 × 0.004571 = **0.257**
- at BEST ep425 (0.003636, assuming shares stable — ASSUMED): 100 × 0.5622 × 0.003636 = **0.204**
- decomposition at BEST: lane-only **0.160**, movable-only **0.045**.

**(c) Conservative LOWER edges (each independently clears the gate):**
- Movable-only full fix (the probe-independent PROVEN-transfer SDF-dilation GO lever): **0.045**;
  derated 50% for placement inefficiency: **0.022**.
- Lane at the #300-MEASURED soft-gate effect (lane within-flip −45% while total d_seg descended):
  0.45 × 0.160 = **0.072**.
- Ultra-conservative "birth-only" floor: lane mass-deficit px (0.005766−0.004468 = 0.001298 of px)
  fixed at 50% efficiency → Δd_seg 6.5e-4 → **0.065**.

**Band: [≈0.02, ≈0.26] ΔS.** Lower edge 0.02 ≥ 4× the 0.005 gate → **PROCEED**.

## Honest risk (the band is a CEILING, not a prediction)

Measured DOWNSIDE exists: uniform amplification and the un-gated paint-seed were net-NEGATIVE
(paint-seed plateaued d_seg ~0.027 vs control 0.0078 at matched ep50 — seed-absorption starvation,
#300). The treatment arm exists precisely because the #300 witness-alone-island-loss +
margin-gated amplify + #315 nucleus-guarded hand-off are the measured/derived fixes for that
mechanism; the matched-epoch A/B vs the live control is the arbiter. From-scratch arm ⇒ the cost
of a failure is compute only.

## Assumptions (labeled)

1. 16-pair strided subset ≈ n600 for shares (subset d_seg +4.7% vs trainer n600 at the same
   epoch; the n600 wa probe + n600 trainer verdicts corroborate scale). ADVISORY NON-PROMOTABLE;
   per the allergic-to-non-n600 rule this feeds a launch decision band, not a score claim.
2. Shares stable ep300→ep425 (ASSUMED; annulus convergence rates are small at ep300).
3. "Full island fix" is unreachable (R-through argmax jitter floor remains) — (b) is a ceiling.
4. Every number here is `[macOS advisory]`; the pointer (0.19110) moves ONLY through a
   byte-closed `upstream/evaluate.py` exact row.

## Repro

```python
# composed per-class table:
z = np.load(".../annulus_live_maps/maps_BEST_ep300.npz"); pred = z["argmax"]
lst = np.load(".../gt_n600.npz", allow_pickle=True)["lstars"]
gt = np.stack([lst[p] for p in list(range(0,600,37))[:16]]).astype(np.int8)
# flips = pred != gt; per-class within_flip / share_of_d_seg / part_frac as in the table.
```
