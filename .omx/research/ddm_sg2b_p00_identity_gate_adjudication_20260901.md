# ddm_sg2b p00 identity-gate adjudication — GATE-DEFECT (cross-lineage), instrument SOUND, null REBASED (2026-09-01, MAIN)

## Verdict

**The identity gate AS WRITTEN fired (exact-equality violated). The mechanism is a CROSS-LINEAGE
GATE DEFECT, not an instrument defect. The distortion-leg fire chain PROCEEDS with the null
rebased to p00's own measured row.**

## The measurement (p00 = byte-identical afr1 null through fire_local_advisory)

- Archive sha `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` (== afr1
  canonical, verified in run.log) · 180,002 B · n600 · rc=0 · 1,364 s total
  (inflate 888.7 s, evaluate 468.1 s).
- Measured: d_seg **0.0003474** · d_pose **0.00014701** · S **0.19293782638440826**
  (recomputed from components, exact match) · axis `cpu_env_mismatch_advisory`,
  gt lineage PYAV_YUV420_TO_RGB.
- Receipt: `/Volumes/VertigoDataTier/pact/ddm_sg2b_scmdl_distortion_leg_build/fire_main/p00/contest_auth_eval.json`.

## Why the gate fired: the gate compared across instruments

The fire order (MAIN_FIRE_ORDER.json, authored by the sg2b arm, consumed by MAIN un-caught)
required p00 to reproduce d_seg 0.00020139 / d_pose 6.37e-06 EXACTLY. Those are afr1's
**contest-CUDA T4 / DALI_NVDEC** values. `fire_local_advisory` measures on **CPU / PYAV** —
a different GT-decode lineage by upstream's own device fork (`upstream/evaluate.py:31-42`).
This is the m143 cross-regime constant-transfer genus INSIDE a pre-registered gate, and the
GT-lineage fork is a named law (na10 #1140; #1340 instrument-join; ddm_pi2). Neither the arm
nor MAIN caught it at authoring/consumption time — recorded as a recall failure of that class.

## Why the instrument is adjudicated SOUND (three independent legs)

1. **Pose reconciles to the NAMED cross-lineage delta**: cuda 6.37e-06 + pi2 additive
   +1.4061e-04 = 0.00014698 vs measured 0.00014701 — diff 3.0e-08 ≈ one 8dp report ulp.
2. **Seg fork same sign/order**: measured ratio 1.7250× vs pi2's 1.4425× on the cp135 body
   (the factor is body-dependent; pi2's was a different body).
3. **Byte identity**: the archive IS afr1's bytes by sha — no defect in the compose harness is
   possible on the null; the delta is entirely the measuring lineage.
Blockers: only `auth_eval_environment_mismatch` (uv group undeclared — the standard advisory
env caveat carried by every row on this axis).

## The rebased null (BINDING for the gate-2 distortion leg)

p01/p02/p03 deltas are computed **against p00's own row, same instrument, same lineage**:
Δd_seg = d_seg(p_k) − 0.0003474 · Δd_pose = d_pose(p_k) − 0.00014701. CPU-torch advisory
determinism precedent: pk4 r3 repeat-noise 0.0. These deltas are ADVISORY
(`score_claim=false`, promotion none) and join rxc1's exact rate leg at the gate-2 boundary
(#1374). Cross-lineage caveat carried forward: a CPU Δ is not a CUDA Δ; any ADMIT at gate 2
that depends on distortion sign near zero must re-buy the winning candidate on the T4 axis
before promotion.

## Fire state

p00 ADJUDICATED (this memo) → p01 FIRED next (order argv verbatim) → p02 → p03 serial.
verdict_scope: INSTANCE (afr1 body, CPU/PYAV advisory instrument).

Own-vehicle frontier: S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha
cbb8d928…d405bf25 — UNMOVED by this adjudication.
