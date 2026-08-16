# ps1u r2 — REFUSED +1.686e-02 S; the uncapped pose solve made pose 8.93× WORSE

`verdict: REFUSED` · `verdict_scope: INSTANCE` (candidate archive
`97048f9fe1845a2b0b602dbdaf5f85e87fb19dee0e6cc57503fe5fd60096bef8`, 183,347 B,
on the hv1 ep0634 base) · axis `[contest-CUDA T4 n600, COMPONENT-ONLY]`
· call `fc-01M05JNY5VWA152YF1MBKS37HE`, 735.1 s, rc=0, ~$0.16.

## The arithmetic, same-instrument where the instrument allows

Base = **hv1 ep0634**, sha `80d9c8c6…`, 182,759 B, S 0.15959729295498598.
(The result schema labels its field `candidate_changed_pixels_vs_cp135` — an
inherited `re1t` field name. The REQUEST's `base_archive_sha256` is `80d9c8c6…`,
so the base is hv1. Adjudicating on the schema label would have been wrong-object.)

| axis | measurement | ΔS |
|---|---|---|
| seg | −37 flips (34,970 → 34,933 of 117,964,800) | **−3.136529e-05** |
| rate | 183,347 − 182,759 = **+588 B** | **+3.915251e-04** |
| pose | d_pose 6.145931e-05 vs 6.88e-06 = **8.93×** | **+1.649641e-02** |
| **net** | | **+1.685657e-02 → REFUSE** |

Repeat pass identical (`repeat_noise_mse: 0.0`), so the pose figure is signal,
not instrument noise. The seg leg is same-instrument and valid. The pose base is
the evaluate.py T4 row rather than this run's own batch=16 forward — but 8.93× is
three orders past any plausible instrument delta, so the sign is robust.

## The mechanism: an ASSUMED zero in the one axis the build existed to move

`POSE_SCREEN_RESULT.json` carries, verbatim:

    "local_pose_delta": 0.0,
    "pose_unmeasured": true,
    "role": "... pose is UNMEASURED locally by the worker placeholder law"

The build never measured pose. It carried 0.0 as a **placeholder**, and the
placeholder was wrong by 8.93×. A pose-targeted candidate shipped with its pose
axis unmeasured and assumed neutral — the same genus as qs4's stale Schur
compensation (`#1039`), one level earlier: there a constant was transferred
across regimes; here a constant was never derived at all.

Note the shape: seg moved the RIGHT way (−37 real flips) and the archive GREW
588 B. So the +588 B did not buy pose — it cost rate AND pose together. Whatever
the solve wrote into the pose section is actively worse than what it replaced.

## Standing

- Lane `ddm_ps1u_dual_axis_pose_n600_20260816` CLOSED TERMINAL at harvest.
- Frontier UNMOVED: hv1 ep0634 S 0.15959729295498598 @ 182,759 B stands.
- The −37 seg flips are real and base-attached; they do not survive alone
  (they ride the same +588 B).
- Modal ≈ $6.9 / $20.

## Owed next, if this family is re-opened

A pose candidate may not dispatch with `pose_unmeasured: true`. The local screen
must measure d_pose on the candidate before the paid row, or the row is a
coin-flip on the axis it targets. That is the concrete fire-order for any ps1u
successor, and it is cheap: the pk4 chain already proved the torch-CPU pose path
runs locally at $0.
