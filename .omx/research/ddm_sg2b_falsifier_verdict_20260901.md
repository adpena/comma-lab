# SG2B pre-registered falsifier — VERDICT: X-alone axis CLOSED (3/3 legs REFUSED, dose-response measured) — 2026-09-01, MAIN

STORES CONSULTED: sg2b p00 identity-gate adjudication (rebased null) · fire_main/p01,p02,p03
contest_auth_eval.json receipts · ddm_gd2 frozen/live law · ddm_x012 crossing ledger ·
MAIN_FIRE_ORDER.json (the pre-registration).

## The pre-registration (quoted)

The sg2b distortion leg fired p01→p02→p03 as serial advisories against the p00 rebased null
(d_seg 0.0003474 · d_pose 0.00014701 · S 0.19293783, cpu_env_mismatch_advisory axis). The
pre-registered rule: **all three Δd_seg ≥ 0 ⇒ the SCMDL X-alone axis formally closes**, and
the #1374 distortion leg purifies to the G/M-coupled cell.

## The three legs (measured, same instrument, same null)

| leg | edit sites | d_seg | d_pose | Δd_seg | Δd_pose | dist-only ΔS |
|---|---:|---:|---:|---:|---:|---:|
| p01 | 1,084 | 0.00035325 | 0.00036401 | **+5.85e-6** | +2.170e-4 | +0.0221 |
| p02 | 2,831 | 0.00036271 | 0.00095376 | **+1.531e-5** | +8.068e-4 | +0.0608 |
| p03 | 9,723 | 0.00040953 | 0.00353203 | **+6.213e-5** | +3.385e-3 | +0.1558 |

Axis: `[macOS-CPU cpu_env_mismatch_advisory]`, PYAV GT, n600, score_claim=false throughout.
p03 receipt: fire_main/p03/contest_auth_eval.json (rc=0, 1,134 s, launch_counter 713).

## Dose-response (the mechanism read)

Site dose 1× / 2.61× / 8.97× produced Δd_seg 1× / 2.62× / 10.62× (≈linear, slightly super)
and Δd_pose 1× / 3.72× / 15.60× (super-linear, ≈dose^1.25). Two consequences:

1. **Seg never improves at any dose** — the X-alone edits are not merely pose-taxed wins;
   they are seg-NEGATIVE from the first rung. There is no small-dose regime to retreat to.
2. **Pose damage accelerates with dose** — the sa1 linear-in-mass law (0.91× of linear) is
   the FLOOR here, not the ceiling; this family is worse than the uncompensated-FiLM family
   it echoes.

## VERDICT

**REFUSED 3/3 — the SCMDL X-alone axis is formally CLOSED** (verdict_scope: FAMILY on this
body/instrument — realized-cell X-only edits on the afr1/sg2b lineage without G/M coupling).
The #1374 distortion leg purifies to the G/M-coupled cell: any X change must be priced
JOINTLY with the model/coder refit through rxc1's exact instrument (gate 1), never as a
standalone field edit. This is also the freshest, best-dosed leg of gd2's frozen-space law:
three more field-touching refusals with a measured dose curve.

Consumers: #1374 gate-2 routing · ddm_x012 Door-D closure (fourth independent leg) ·
gd2 law ledger. The sg2b build stores remain retained (runtimes p01–p03 + receipts, AP +
Vertigo custody, nothing discarded).

Own-vehicle frontier: S 0.14797617125559104 @ 180,002 B [contest-CUDA T4 n600], afr1 sha
cbb8d928…d405bf25 — UNMOVED by this verdict.
