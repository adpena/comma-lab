---
task: 603
feeds_task: 613
review_round: 1
research_only: true
main_landing_review_required: true
---

# Codex adversarial round 1: DDM v4 structured members

## Disposition

`PASS_WITH_SCOPED_BLOCKERS` for the n64 representation-membership measurement.  `REFUSE` for any
score, d_seg, promotion, or full-tolerance claim.

## Self-attacks and fixes

1. **Permissive ZIP parsing could hide appended bytes.**  The first implementation verified known
   members but a ZIP reader would still tolerate trailing data.  The receiver verifier now
   recompiles the parsed program and requires canonical byte equality; a regression test mutates
   the archive with an appended byte and requires refusal.
2. **Cross-class membership could be mislabeled as target efficacy.**  Every candidate now reports
   the full five-class matrix.  Road-role target Road is zero while MyCar is
   `0.997696884575`; the memo preserves this as palette/routing debt.
3. **Uncommitted source could create false custody.**  The first real execution reached final
   publication and refused because its producer module was not committed at the claimed SHA.  The
   producer was serializer-committed before the authoritative rerun.
4. **Pose availability could be confused with Pose quality.**  The receipt names the metric
   `pose_complete` and makes no d_pose claim.
5. **Empty event support could be overgeneralized.**  Movable has zero records only within the
   measured first-64-frame prefix; verdict scope prevents a family-level negative.
6. **Approximate byte-box language could mask exact overshoot.**  Each archive stores exact bytes,
   SHA-256, unique byte homes, and both approximate and strict cap checks.

## Remaining MAIN review debt

- Independently re-derive LBND2 receiver parity and all receipt hashes.
- Inspect Road palette semantics versus source-mask geometry using a paired disambiguator.
- Verify the 31 focused tests on MAIN after merge.
- Preserve pointer `0.1910828242 [contest-CPU]`; no score authority entered this arm.
