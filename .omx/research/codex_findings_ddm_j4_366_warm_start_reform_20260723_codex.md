# Codex findings — DDM J4 366 warm-start reform

Date: 2026-07-23
Lane: `ddm_j4_366_warm_start_reform`

## Finding

The J3 regression is localized to the opening optimizer/admission policy, not
EMA and not the whole 368-coordinate family. With no warm-start moments,
implicit Adam \(\beta_2=0.999\) normalized each nonzero first-step gradient to a
quarter-quantum update. At step 4, 14 coordinates realized. An exact n600
group split showed:

- five island quanta: \(\Delta d_{\rm seg}=+0.0001086934407552104\),
  \(\Delta d_{\rm pose}=+0.00009996662967370917\);
- nine template quanta: \(\Delta d_{\rm seg}=+0.00003053453233506913\),
  \(\Delta d_{\rm pose}=-0.00010379410576844017\).

The pair447 four-pair surrogate admitted updates that the exact n600 receiver
surface rejected. EMA max magnitude was only `0.007489411`, below receiver
realization, so it was not the trigger.

## Reform landed

- explicit beta2 and a canonical-law-derived 2000-step LR rewarmup;
- hard per-coordinate quarter-quantum cap;
- shared-template freeze before first strict island admission;
- Pose objective delayed until first strict Seg admission;
- immediate exact n600 verdict on the first integer receiver boundary;
- rollback/block on either component regression or no component descent;
- additive historical J3 typed-config compatibility.

## Bounded remeasurement

J4 remained byte-identical for all four opening steps:

- \(\Delta d_{\rm seg}=0\)
- \(\Delta d_{\rm pose}=0\)
- realized coordinates: `0`
- verdict: `BLOCKED_REALIZED_NO_COMPONENT_DESCENT`

The reform removed the observed regression but did not establish descent. No
campaign was launched. Verdict scope is the J4 four-step opening instance; the
formulation and representation family remain open.

## Review requirement

This branch is research-only and `[macOS-CPU frozen-scorer advisory]`.
`score_claim=false`; pointer `0.1910828242 [contest-CPU]` unmoved. MAIN must
independently review the diff, receipts, and deferred C1 role-bucket mapping
before landing. The bounded result is not `READY_TO_FIRE`.
