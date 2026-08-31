# ddm_hzs1_horizon_shape — the 37.47× leverage nobody spent: horizon carries 67.43% of the gap on 9.5% of the packet, and only its SHIFT was ever swept (owning memo: ddm_hzs1_horizon_shape_20260831.md)

## MANDATE

Operator 2026-08-31: *"Continue with all believe in yourself"* + *"Gestalt not naive or toy or simply pursuing cheapest"* + *"Use a swarm of opus and codex subagents"*

`ddm_gf1_capacity_gap_decomposition_20260831.md` §8c priced every stream of the HG1 analytic
generator in its OWN CODED bytes for the first time. One row is unlike the others:

| stream | coded B | % of packet | % of capacity gap | fix-vs-code leverage |
|---|---:|---:|---:|---:|
| lane | 36,044 | 75.7% | 24.03% | 3.93× |
| movable | 6,624 | 13.9% | 0.08% | 1.94× |
| **horizon** | **4,536** | **9.5%** | **67.43%** | **37.47×** |
| mycar | 95 | 0.2% | 8.47% | 1.13× |

**Horizon holds two-thirds of the entire capacity gap while occupying under a tenth of the packet,
and fixing it is 37.47× more efficient than coding its output.** That is the largest leverage ratio
anywhere in the decomposition and it has never been spent — because the only horizon knob ever swept
was a **per-frame vertical SHIFT**, which bought 6.1% (gf1 §6). Shift is one parameter of a curve.
The curve's SHAPE — its per-column profile, its curvature, its scene-conditioned tilt — is unswept.

The bar this must clear is stated and cheap to evaluate:

```
REACTIVATION FRONTIER (gf1's own, unchanged):  packet_B + 0.2909 * mismatches < 85,020
current:  47,603 + 0.2909 * 1,325,033  =  433,055 B  =  5.094x
```

Horizon perfect ⇒ 173,155 B (2.037×). Horizon+lane perfect ⇒ 80,530 B (**0.947× — CLEARS**).
Horizon alone does not clear; but it is the cheapest 67.43% on the board and it decides whether the
two-stream corner (§1's k=2 row) is reachable at all.

## SCOPE

1. **Characterize the horizon residual before parameterizing anything.** Where do horizon's
   893,436 mismatches (67.43% of 1,325,033) actually sit — per column, per row-band, per frame,
   per scene regime? Use gf1's own retained payloads; reproduce 1,325,033 as a control FIRST
   (every gf1-lineage script in this window does, and it is why they cost seconds).
2. **Decide the parameterization FROM the residual**, not from convenience. If the residual is a
   smooth per-column offset, fit that. If it is scene-conditioned, condition it. If it is a
   curvature the current model cannot represent, say which term is missing. State the DOF count
   and the byte cost of each candidate BEFORE measuring its benefit — the sharp-optimum law
   (#1214, memo `ddm_oe1_*`) says added parameters usually lose, so price both sides.
3. **Measure through the SAME instrument** gf1 used: `count_nonzero` token-field mismatch against
   the lb1 field AND the DALI GT, real coders for any byte claim, n600. Do not switch quantity
   mid-arm — token-field mismatch is NOT d_seg, and the pincer memo §5 records why conflating
   them is forbidden.
4. **Report the frontier arithmetic on every rung:** new `packet_B + 0.2909 × mismatches` and its
   ratio to 85,020. A rung that improves horizon but grows the packet may lose; show it.
5. **Denominator + honest ceiling.** How many parameterizations enumerated, how many measured,
   what is the best achievable horizon residual at ≤ +2,000 B, and does the k=2 corner become
   reachable or not. ⚠ The horizon+lane corner also demands **lane at 0.0178% error** — if your
   horizon result makes k=2 arithmetically possible only under an unmeasured lane assumption,
   SAY SO and do not claim the corner.

## HARD CONSTRAINTS

- `upstream/` READ-ONLY. NO Modal fire from the arm (MAIN owns dispatch + single-flight).
- The local SCORER LANE belongs to MAIN, always. Emit typed fire orders; an honest partial plus a
  fire order is the CORRECT outcome, never a failure.
- Serializer commits w/ post-edit `--expected-content-sha256`; `.py` = 2 genuine review passes.
- ALWAYS KEEP THE PAYLOAD; bulky receipts to `/Volumes/APDataStore/pact/ddm_hzs1_horizon_shape/`.
  Vertigo is at capacity — APDataStore only.
- **Reproduce 1,325,033 as a control before trusting any derived number.** Two arithmetic errors in
  this lineage this window were caught exactly this way (raw-vs-coded denominator; a negative
  packet size). If your control does not reproduce, STOP and report that.
- Sole ownership of `.omx/research/ddm_hzs1_*`. Do not touch lb1's shipped bytes or runtime.

## PRIOR NEGATIVE SIGNAL (bearing dead-ends this charter consumes)

- `ddm_gf1_generator_form_capacity_verdict_20260830.md` — the HG1 family REFUSED at 5.09×,
  verdict_scope FORMULATION. This arm works INSIDE that refusal's own stated reactivation frontier;
  it does not reopen the verdict.
- `ddm_gf1_capacity_gap_decomposition_20260831.md` §6 — the composite paint ORDER is EXHAUSTED
  (all 24 permutations swept, current order optimal; it is a receiver-code constant so sweeping it
  costs 0 archive bytes and it still gave nothing). Per-frame horizon SHIFT buys 6.1%. Movable's
  oracle-precision ceiling — the single largest lever by that framing — still lands at 3.523×.
  Do not re-sweep order; do not re-sweep shift alone.
- §8 (same memo) — the PACKET axis is SPENT: best available packet reduction is **14 B** (0.03% of
  headroom), and all 10.30× of the remaining gap is DISTORTION. A rung that proposes to win by
  shrinking the packet is dead before it starts.
- `ddm_gestalt_generate_vs_serialize_pincer_20260831.md` §6 — cross-family convergence: two
  structurally unrelated generator families land 0.0005 percentage points apart (1.1232% vs
  1.1237%), spread 1.16× across seven fields, best object 63.1× from the demand. If horizon-shape
  lands in that same band, that is CONFIRMING evidence for a class bar, not a failure of your fit —
  report it as such.
- `ddm_oe1_*` / #1214 — the HPAC optimum is measured SHARP in every direction across five arms.
  Expect losses; the exception class (fcd1's B/H/W win-win) came from DECOMPOSING outcomes, not
  from a better fit.

## OPTIMAL FORM

- Family exemplar: `ddm_rd2_hg1_rate_distortion_curve` — the **reference** form: full-set, real
  coders, four controls green, a CURVE not a point. Receipt:
  `/Volumes/APDataStore/pact/ddm_rd2_hg1_rate_distortion_curve/retained/rd2_phaseA_byte_curve.json`.
  Provenance pin: charter custody commit `885fe82498`.
- SCOPE reductions declared per row (frame subsets for exploration are legal — but any VERDICT
  number is n600). MECHANISM reductions FORBIDDEN: a horizon model fit with a toy objective or a
  toy coder produces no verdict.
- **PRIOR-LAW PREDICTION (falsifiable):** the class-bar reading (pincer §6) predicts horizon-shape
  will improve the residual but land the composed field back in the 1.12–1.30% band, i.e. the
  frontier ratio stays ≫1 and the k=2 corner stays unreachable. FALSIFIER: a measured
  parameterization that drives horizon's contribution below ~10% of the gap at ≤ +2,000 B — that
  would put the k=2 corner in play and REFUTE the class-bar reading. Count it plainly either way.

## DELIVERABLE

`.omx/research/ddm_hzs1_horizon_shape_20260831.md` — the residual characterization first; the
parameterization table (DOF · byte cost · measured residual · frontier ratio) with its denominator;
the control line reproducing 1,325,033; the honest ceiling at ≤ +2,000 B; the k=2 reachability
verdict with the lane assumption stated explicitly. Commit via the serializer. End with the
own-vehicle frontier line.
