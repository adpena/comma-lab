# OD2 next if resumed - 2026-08-05

Status: `QUEUED_WITH_FIRE_ORDER / DO_NOT_PROMOTE_CURRENT_ROW`.

Axis: `[macOS-CPU frozen-scorer advisory]`.

## Current State

OD2 produced a real n32 staged-composition measurement:

- Stage 1 regional phase / SQ1-class solve: eta `0.554438560272866`, but cap-bound on 29/32 rows.
- Stage 2 frame_0 k=4 DCT carriage: seg preserved 32/32, d_pose mean `0.0007588698333620414` versus same-row baseline `0.0008014285623403339`.
- Projected n600 carriage bytes: 57,600 B, rate cost `0.03835347569983707` S.
- Subset advisory projection versus current own vehicle: `S = 0.6917440272267846`.

This is not a pointer row. The next run must not present it as one.

## Fire Order

1. Close Stage-1 terminality before any promotion wording.
   - Use the same pair set unless a new selection manifest is deliberately generated.
   - Consume st2 only as a solve-budget ranking prior, not as a paint mask.
   - Required output: stop census with no cap-default ambiguity and explicit `{converged, cap-best, pre-plateau, failed}` denominators.

2. Build a receiver-closed archive only after Stage 1 is no longer cap-bound or after MAIN accepts a cap-bound floor as an explicit bounded prototype.
   - The k=4 coefficients are counted payload, not free code.
   - The generic DCT basis may be free under rule 118.
   - Receiver parse-back and runtime changed-pixel proof are required before any scorer survival claim.

3. Run n>=32 final-composition scorer gate on the exact receiver output.
   - Use the same selection receipt or create a new stratified-random receipt with seed, strata, and governing ratios.
   - Recompute S from components. Do not use rounded evaluator output.

4. Only after the n32 receiver-closed gate improves the current own-vehicle line, queue full n600.
   - Claim the lane first.
   - Respect one full-n600 scorer job at a time and chunk <=120.
   - Use durable SSD output roots.

## Measured Compute For Scale-Up

Observed OD2 CPU timing:

- Smoke pair, reduced 5/5 settings: 37.6s.
- First two n32 rows with optional `poseonly` control: 182.8s and 164.2s pair logs.
- Resume run without optional `poseonly`: `DONE 32 t=3738.4s`, covering 30 new rows plus setup.

Projection for this exact Mac CPU advisory runner:

- Cheapdct-only row time: about `124.6s/pair` from the 30-row resume.
- n600 projected wall time: about `20.8h` at the same settings.
- chunk=120 projected wall time: about `4.15h/chunk`.

Do not launch the n600 projection as-is. The current row is not receiver-closed and Stage 1 remains cap-bound.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
