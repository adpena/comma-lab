# Codex findings — DDM WS1 Seg-lexicographic warm starts — 2026-07-24

## Disposition

Two distinct, exact n600 advisory warm starts now exist and are wired into the train-decision and J5/#366 surfaces. No training or exact contest evaluation was launched.

| candidate | d_seg | d_pose | bytes | MyCar errors | role |
|---|---:|---:|---:|---:|---|
| W_seg: `temporal_affine_16knot_frame1_seglex96_hood_masked` | 0.024124510023328993 | 146.3649324958955 | 138,031 | 37,619 | best exact strict-Seg/hood-safe start |
| W_joint: `statistics_hard_analytic_composed_frame1` | 0.07051923116048177 | 36.6181847780574 | 138,801 | 4,072,489 | current best joint-S MENU1 start |

All figures are `[macOS-CPU frozen-scorer advisory]`, realized through the exact uint8 receiver and frozen scorer path. `score_claim=false`; the pointer remains `0.1910828242 [contest-CPU]`.

## Composition and per-class truth

The settled V19B prefix contributes 10 strict sequential Seg moves. Re-ranking the 104 joint-accepted V19C decisions by `incremental Seg term < 0`, ignoring Pose, retained 96 and rejected eight. Full source replay produced a 137,827-byte receiver with SHA-256 `4fbba057b10c64d85f73ea2da3287f5fbd3f794c71ef0762fe0e0e50a224ea2d`.

W_seg per-class errors are Lane 299,944; Movable 352,093; MyCar 37,619; Road 1,870,275; Undrivable 285,912. The filtered base itself has exactly 37,237 MyCar errors, so the solved hood was preserved before amplitude composition.

The accepted temporal move was measured both ways. Hood masking costs 676 total Seg errors versus its unmasked form (`delta d_seg = +5.730523003472793e-06`) but improves Pose by `-4.377564051146408` and keeps MyCar at 37,619 rather than allowing a non-local composition claim. The support is decoder-derived and counts zero additional bytes; the 139-byte stored MC1 support remains a proof-equivalent reference only.

## Preregistered falsifier

Measured opening gaps are 4.639472113715278 Seg score units in W_seg’s favor and 19.12179159806879 Pose score units in W_joint’s favor, giving `R* = 4.1215446777965665`.

The typed four-step J5 smoke spec starts from both candidates, checkpoints every step, and adopts W_seg only when the Pose-progress/Seg-advantage-erosion ratio clears R*, Pose progresses, and Seg does not regress. Pose stall, any Seg regression, or a subcritical ratio keeps W_joint. This landing intentionally does not execute that smoke.

The 2026-07-24 P0 metric directives are consumed: margin-Fisher/rank-4 scorer hyperplanes and the exact low-rank Pose quadratic are authoritative; measured quadratic blocks use Hessian/SPD normal coordinates from step one; Euclidean identity-L2 is a labeled control with cosine and relative-norm readback, never the adjudicator.

## Adversarial self-review round 1

1. **Found:** the first WS1 receipt audited the 96/8 split but measured amplitude on the settled V19C endpoint, so it could not honestly claim an exact filtered receiver.
   **Fix:** performed the full source replay, emitted the SHA-bound Seglex96 archive, and reran base/unmasked/masked n600 scorer streams. The final artifacts and J5 ticket now point only to the filtered receipt.
2. **Found:** MENU1’s 4.07M MyCar result could be mistaken for a base-hood failure.
   **Fix:** explicit per-class guards prove filtered base MyCar = 37,237 and final W_seg MyCar = 37,619; the 4.07M value remains correctly scoped to W_joint composition damage.
3. **Found:** the initial J5 readback text did not encode the later P0 metric-first/Hessian directive.
   **Fix:** the spec now requires scorer-coordinate Fisher/Pose-quadratic authority, second-order geometry where measured, and Euclidean control columns.
4. **Remaining debt:** contest-CPU/CUDA parity, an actual J5 slope result, and any promotion authority remain absent by design. MAIN must full-diff review before landing.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; v7.5/v8 vehicle specs; DDM doctrine through 9b+correction; MENU1, MC1, V19B/V19C, and J5 receipts/configs; canonical equation registry; per-arm and broadcast inboxes.
