# ddm_qs5 r2 verdict + the "No naive or toy ever" enforcement wave (2026-08-13)

## QS5 R2 VERDICT — REFUSED +2.519822e-6 [contest-CUDA T4 dual-axis worker, n600]

Call `fc-01KZYY8W31ZMABWFENFMQE6NJQ` (~$0.16; r1 refusal ~$0.02). Candidate 186,278 B
sha 0911cef…67a1. Matched worker-family instruments (base 34,970 flips / d_pose
6.885642960696714e-6 / 186,252 B):

| leg | realized | ΔS |
|---|---|---|
| seg | 34,953 flips (−17; 132 changed px) | **−1.441108e-5** |
| pose | d_pose 6.88500995238428e-6 (repeat identical) | **−3.814320e-7** |
| rate | +26 B | **+1.731233e-5** |
| **net** | | **+2.519822e-6 REFUSED** (thresholds: ≥21 flips negative, ≥33 super-band) |

**Product 1 — the in-compile compensation is PROVEN (MEASURED).** d_pose landed BELOW
base: the exact-object Schur solve not only cancelled frame-1 pose leakage, it slightly
improved it. The qs4 disaster (+2.396e-4 from a stale carried compensation) is fully
cured; the perturbation-specificity law now has its constructive converse. Any future
frame-1 seg candidate carries ~zero pose tax when compensation is solved in-compile.

**Product 2 — the 17-flip realization ceiling, reproduced ×2 (MEASURED).** qs4 realized
net 17 from 100 changed pixels; qs5 realized net 17 from 132 (17 connective sites
restored, zero recovery). Identical net from different supports reads as a structural
ceiling of THIS 3-pair object family (517/523+connective), not noise. The seg model
consistently over-predicts ~3.4× here.

**Consequences.** eu4's gated union (needed ≥30 flips from qs5's child) FAILS its own
gate → union candidate DEAD as gated. The micro-edit family is PARKED at this support:
with pose free and coding at ~4 B/pair, the family's binding question ("≥21 flips at
≤26 B") is answered NO twice on this object. Pose-first allocation (eu4: 69.38% of gap)
is unambiguous; the route is pk4 (optimal-form frame-0 pose representation).

Family ledger (all full-precision worker rows): JO1 +2.16e-4 · re1 indeterminate ·
qs1 REFUSED +2.43e-5 · **qs2 ADMITTED −4.374914e-6 (banked)** · qs4 REFUSED +2.44e-4 ·
qs5 REFUSED +2.52e-6 (near-miss). Portable assets: in-compile compensation (proven) ·
4.0 B/pair coder · collateral B/H model (benefit-exact) · breakeven 0.785 flips/B.

## THE ENFORCEMENT WAVE (operator ×2: "No naive or toy" → "No naive or toy ever")

pk3 exposed the leak: a clean charter did not stop a toy model at BUILD time, and the
FIRE path had no gate — the toy-derived sealed order (9-pair Jacobians, LOO −16.70%)
was one command from dispatch. Landed, three sites:

1. **SPAWN — strict charter lint by default** (`tools/codex_arm_queue.py`): refusal is
   the default; `TAC_CHARTER_LINT_STRICT=0` is the explicit tracked escape.
2. **FIRE — `tools/fire_order_toy_gate.py`** (fail-closed, self-test 3 branches +
   both-real-store controls executed): refuses TOY-BRACKET-bound stores, withhold
   receipts, and model-derived candidates lacking a passed generalization receipt.
   Precision = `ADJUDICATION_FIRE_OK.json` typed receipt (placeholder-rejected). MAIN
   runs it before every sealed `exact_command_argv`. Controls: pk3 store REFUSED;
   qs5 store PASS_WITH_ADJUDICATION (candidate mechanical; only projections were
   honestly toy-labeled — receipt written).
3. **BUILD — pk4's charter** makes the holdout gate fail-closed in code before compile
   (pk3: 23/23 in-sample winners were 0/23 LOO) + structural pose-placeholder emission
   (two arms tripped the worker law).

Memory: `no_naive_or_toy_ever_structural_enforcement_20260813` + MEMORY.md poison-row
extension. pk3 fire receipt: `FIRE_WITHHELD_TOY_DERIVED.json` in its store.

Lane: `ddm_qs5_resolve_compensation_n600_20260813` closed terminal. Modal ≈ $3.9/$20.
