# ERRATA — the 8dp-band + instrument-mixing error class in today's adjudications (2026-08-13, MAIN)

Operator prompt: "You have made other errors." Self-audit found them. Source receipts re-read.

## The root facts (receipt-verified at source)
1. **Every evaluate.py canonical S in this campaign is an 8dp-REPORT reconstruction, not an
   exact number.** `contest_auth_eval.json` declares it:
   `canonical_score_source = report_8dp_components_plus_exact_archive_bytes` with
   `report_8dp_score_worst_case_abs_error_bound = 3.51e-6` — dominated by pose, where ONE
   8dp ULP of d_pose (1e-8) is worth ≈6.02e-6 S at the current operating point.
   The floor "0.16195513827824176" is therefore a BAND: 0.1619551 ± 3.5e-6.
2. **The 16-digit d_pose 6.885642960696714e-6 quoted all day as "the floor's pose" appears
   in NO receipt** — the cp135 receipt knows only 6.88e-06. The 16-digit figure traces to
   arm memos (po1/pz4r lineage): a WORKER-instrument component welded onto an evaluate.py
   headline by memo citation, then propagated by MAIN (me) into m04, JO1, re1, qs1.

## Per-verdict re-grades
- **re1 Round-1 full-auth "+4.03e-6 WORSE" → RETRACTED as a signed verdict; INDETERMINATE.**
  Decomposition of the reported delta: pose leg = one 8dp ULP (+6.02e-6) + seg leg (−2e-6)
  = +4.02e-6 ≈ the whole number. True d_pose delta could be ~0 (both sides straddling a
  rounding boundary). The seg leg (−2 net flips) is real (worker sign gate). Round-1 may be
  a genuine micro-win — resolvable only at full precision (below).
- **THE FAMILY LAW re-scoped**: the JO1 row (+2.05e-4 pose, ~35 ULPs) STANDS; the re1 row
  (+5.7e-6) is RETRACTED (quantization-scale). The "7–40× pose-dominated" range collapses
  to "JO1-measured e-4-scale for six events; SINGLE-edit leakage unresolved at instrument
  precision." Q3/Schur compensation remains cheap insurance, no longer proven-necessary
  for single edits. js6b's census partially calibrated on the re1 row — its 0/200 verdict
  stands on the realization-efficiency evidence (qs1's full-precision worker row), not on
  the retracted calibration point.
- **The sign-gate memo's "new floor 0.16195344" projection → RETRACTED**: a −1.7e-6 move
  is below the canonical instrument's resolution; it was never claimable.
- **qs1 "REFUSED +2.43e-5" → SIGN ROBUST** (delta ≫ the ±7e-6 two-sided band); but the
  "pose compensation PROVEN, 50–1800× cure" precision claim is WITHDRAWN — the qs1 pose
  delta (+1.87e-10 d_pose) compared a worker measurement against a memo-propagated
  pseudo-receipt value. Bounded honest claim: qs1 pose leakage ≤ instrument-band scale;
  the exact cure factor awaits the same-instrument base measurement.

## The derived operating laws (new; consumed by qs2's directive)
1. **Sub-1e-5 admission requires SAME-INSTRUMENT FULL-PRECISION component rows** (the
   dual-axis worker measures float-precision d_pose + exact flip counts). evaluate.py 8dp
   canonicals adjudicate only deltas ≫ ~7e-6 (two-sided worst case).
2. **The minimum worthwhile candidate is band-sized**: stop buying candidates whose
   expected |ΔS| < ~1e-5 — the canonical instrument cannot certify them. Micro-edits must
   COMPOSE into a super-band candidate before an evaluate.py dispatch.
3. **The resolving measurement (one dispatch, ~$0.16)**: the dual-axis worker with a BASE
   leg + the re1-candidate leg — full-precision seg+pose for base, re1, and any qs2
   candidate on ONE instrument. Resolves re1's sign AND gives qs2 its matched base.
   Folded into qs2's sealed fire-order requirements (directive 2).

## Corrections applied in this landing
- This errata memo (canonical). Headline pointers added to: re1 full-auth memo · sign-gate
  memo · qs1 verdict memo (per the stale-headline law — corrections at the HEADLINE).
- Memory: new durable law file + MEMORY.md m04 row annotated with the band.
- qs2 directive 2: base+re1 legs mandatory in the fire-order; band-sized composite target.

## CORRECTION TO THIS ERRATA (2026-08-13, later same day — honesty cuts both ways)
Root fact 2 over-claimed. The 16-digit d_pose 6.885642960696714e-6 IS receipt-backed:
po1's T4 dispatch measured the cp135 BASE decode directly with a deterministic repeat
(`d_pose_decoded_first == d_pose_decoded_repeat`, po1 memo :237), and pz4r explicitly
paired it with 34,970 flips as "the contest-CUDA T4 CP135 reference" while itself warning
against instrument mixing. Corrected status: the value exists in NO evaluate.py CANONICAL
receipt (that part stands — the canonical knows only 6.88e-06) but is a real WORKER-FAMILY
measurement. Consequences: qs1's pose-cure claim upgrades WITHDRAWN → WORKER-FAMILY-
MATCHED (batch-shape pin between po1's worker and the js6b worker still owed); the re1
CANONICAL retraction and the 8dp-band law are UNCHANGED. The qs2 R2 dispatch (fired
2026-08-13, call fc-01KZYQJE57P4FV9PS0SXMA2BXM) measures the candidate on the same worker
family — its verdict uses the po1 base pair as the matched base, caveated by the
batch-shape pin.
