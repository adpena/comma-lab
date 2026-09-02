Implemented and landed the QBR1 fair-form burn preparation.

- Durable handoff: [ddm_qbr1_born_fairform_burn_prep_20260902.md](/Users/adpena/Projects/pact/.omx/research/ddm_qbr1_born_fairform_burn_prep_20260902.md)
- Runner: [ddm_qbr1_born_fairform_burn_prep.py](/Users/adpena/Projects/pact/experiments/ddm_qbr1_born_fairform_burn_prep.py)
- Tests: [ddm_qbr1_born_fairform_burn_prep_test.py](/Users/adpena/Projects/pact/experiments/ddm_qbr1_born_fairform_burn_prep_test.py)
- MAIN fire order: [SEALED_MAIN_FIRE_ORDER.json](/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/SEALED_MAIN_FIRE_ORDER.json)
- Final handoff commit: `1099c365f9`

Status: `BUILD-COMPLETE / SEAL-CONDITIONAL-ON-MAIN-RESUME-SMOKE / BURN-NOT-FIRED`.

Measured and verified:

- Six same-start configs: three seeds × `100/100` control and `100/0` zero-native treatment.
- Exact 5,000-update schedules, EMA LawRef, 16-step checkpoint cadence, retained milestones at 0/1000/2000/3000/4000/5000.
- Re-derived runtime: 2.136656 s/update; projected finish plus six realizations is 18.613416 h before overhead.
- Real-B=16 parent peak: 2,572,632,064 B.
- Six-cell storage projection: 22,541,950,716 B; live APDataStore preflight passed with the required 8 GiB reserve.
- Five tests passed, Ruff passed, both Python files received two review passes, and all six retained configs revalidated.

Not measured: no QBR scorer outcome, treatment sign, Metal burn, Modal invocation, contest evaluation, or pointer movement. The arm correctly left the scorer and Metal lanes to MAIN.

Own-vehicle frontier: **NOT MOVED**. AFR1 remains `S=0.14797617125559104`, 180,002 B, SHA-256 `cbb8d928a8ccdd3f5103da1d4a8d38d0662a5e5615266b923b5f8350d405bf25` `[contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **Disposition `SEALED-BLOCKED-ON-MAIN-SCORER-LANE`; owner MAIN; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/resume_smoke/`; fire trigger:** MAIN holds a unique live scorer claim; run the sealed real-B=16 resume smoke and require cursor, live-state, EMA-state, and archive equality.
- **Disposition `SEALED-AWAITING-MAIN-LIVE-CLAIMS`; owner MAIN; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/runs/`; fire trigger:** resume smoke passes, qxr1/QXO1 has been consumed and QBR1 remains non-dominated, and fresh scorer plus Metal claims are bound; fire the six cells sequentially.
- **Disposition `AWAITING-SIX-CELL-RESULTS`; owner MAIN or named harvester; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/ADJUDICATION_RESULT.json`; fire trigger:** all six step-5000 results and retained payloads are complete; run mechanical adjudication.
- **Disposition `CONDITIONAL-N600-BUY`; owner MAIN; consumer store `/Volumes/APDataStore/pact/ddm_qbr1_born_fairform_burn_prep/n600/`; fire trigger:** adjudication returns `OPTIMIZATION_LIVE_DISTORTION_ROUTE`; build a same-object retained n600 ticket.

## LIVE-HYPOTHESES

- Zero-native may win at least two seeds because removing the internal interface objective eliminates competition with the score-facing realized path while holding every other variable fixed.
- Joint pose supervision may bring the changed object inside its byte-conditioned pose corner because it acts through rendered RGB and the real scorer from update zero.
- A zero-of-three treatment result would type the wall as capacity/object rather than optimizer coupling, because this experiment removes the single-seed and proxy-loss confounds.
- qxr1/QXO1 may render the burn dominated before firing; that is why its realized result is a MAIN decision input.

## DEAD-ENDS

- The exact 106,832-byte BR2 archive is closed at instance scope: its realized distortion is about 1,045.997× its lawful sub-0.12 allowance.
- BR2 distortion cannot be transferred to any QBR cell.
- Raw r5 balanced CE is closed as a start law because it caused Road-to-Lane over-paint; QBR uses the reviewed existence-majority lineage.
- One- or two-seed conclusions are inadmissible for this discriminator.
- The rounded 2.135 s/update estimate is superseded by the retained 2.136656 s/update measurement.
- Process survival alone is not an acceptable resume proof; the sealed smoke requires exact cursor, live-state, EMA-state, and archive equality.