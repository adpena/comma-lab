---
utc: 2026-07-23T03:50:00Z
lane_id: ddm_j3_366_fullrun_mode_and_ticket_reseal
verdict: BLOCKED_REALIZED_DSEG_REGRESSION
verdict_scope: INSTANCE
research_only: true
score_claim: false
---

# Findings

1. J2's 706 names overcounted the current receiver wire by 338: 326 aspect/rotation lift-metadata coordinates and 12 Lane BEV/range seeds were not encoded. The executable surface is 368 coordinates. The 2,197 individual-knot surface remains unadmitted absent counted grammar and measured marginal value/byte.
2. Actual batch-4 full-run geometry measured `104.09510249993764 s/step`, `10.47418212890625 GiB` peak, and `13.5690185546875 GiB` projected peak below the `116 GiB` ceiling.
3. The first full-path result string was falsely green because it classified memory admission only. Exact four-step n600 replay instead regressed `d_seg` by `0.00013291252983941085` and `d_pose` by `0.0000035612565909559635`.
4. The fixed launcher now computes an exact baseline for every full run, requires strict `d_seg` descent without Pose regression, checkpoints and returns nonzero on component regression, and cannot advance a blocked stage.
5. Resume review found timing-only telemetry could collide after a crash before checkpoint. Immutable step/verdict rows now reuse a prior row only when every non-timing field is identical.
6. Admission input and runtime measurement output are separate paths, so a campaign cannot overwrite its tracked preflight receipt.
7. Scheduled decisions use EMA, while an explicitly bounded pre-schedule smoke is labeled live-theta. Third-exposure exact verdicts permit a measured plateau stop at step 100 before the step-150 maximum; no predicted plateau has authority.

The negative is limited to the fixed pair-447 four-step warm-start instance on the 368-coordinate formulation. The formulation and family remain open. The highest-EV reformulation is a realized n600-admitted warm-start selector; the available v16 coupling operator is optional and requires its own exact-archive A/B.

Pointer `0.1910828242 [contest-CPU]` unchanged. MAIN must review the branch and must not fire the campaign while this blocker stands.
