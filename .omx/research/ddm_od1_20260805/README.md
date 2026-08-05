# OD1 SOL ULTRA package - 2026-08-05

Status: `SEALED_FOR_MAIN_REVIEW / SCORER-FREE / READY_TO_FIRE_WITH_MAIN_ONLY`.

Axis: `[macOS-CPU advisory / scorer-free planning and custody]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

## Contents

- `CAMPAIGN_SPEC.md`: ordered OD1 campaign: seg base first, joint pose recovery after, carrier/rate composition, and final gate chain.
- `RECALL_EVIDENCE.md`: governing reads, searches, recall found beyond the charter seeds, and plan changes.
- `LAUNCH_TICKETS.md`: sealed tickets and exact fire order for MAIN.
- `BLOCKERS.md`: typed blockers for rungs that are not sealed.
- `od1_seal.json`: machine-readable summary of the package.

## Verdict

OD1 does not move the pointer and does not claim a score. The sealed campaign route is:

1. Use the solved-field seg base as the first-class base, not as a direct ship row.
2. Recover pose after the seg base through joint/in-loop or frame_0 pose carriage, with R8 judging only the final composition.
3. Fold PE receiver-consumed carrier/rate sections after runtime consumption proof, not before survival measurement.
4. Queue scorer work behind MAIN's single-slot discipline.

`READY_TO_FIRE` remains owned by MAIN per the charter. This package gives MAIN a fire order and the blockers that must be cleared before any pointer claim.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
