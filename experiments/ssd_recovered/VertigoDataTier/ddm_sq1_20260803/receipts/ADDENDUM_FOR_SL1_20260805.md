# ADDENDUM (MAIN, 2026-08-05, operator-directed) — READ BEFORE USING THE SQ2/CW1 ROWS

CUSTODY FACT: the sq2 uncap100 solve (η 0.9113) banked METRICS ONLY (the
sq1_stage_n32_uncap100_sq2.json rows). The solved FRAMES were never persisted — the
solution was discarded along with the (wrong-stage) R8 kill.

INSTRUCTION for sl1 leg 1 (supersedes "load the corrected base"):
1. RE-SOLVE deterministically from the banked base
   (/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2/archive.zip, same
   solver, same sq1_selected_pairs.npy) — but run UNCAPPED TO CONVERGENCE (plateau
   criterion, per-pair convergence trace; NO arbitrary step cap — #935/#850, operator
   directive). #850 measured 13–23%/iter descent at the old cap, so the converged η
   likely exceeds 0.9113: report the TRUE endpoint.
2. PERSIST the solved frames this time (npz per pair, SHA'd, under
   ddm_sl1_20260805/ on this SSD — certify-or-block applies).
3. THEN run the terminal-pose composition + composed-stage R8 per your charter.
Same for the et1 bases if only metrics exist there.

S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.
