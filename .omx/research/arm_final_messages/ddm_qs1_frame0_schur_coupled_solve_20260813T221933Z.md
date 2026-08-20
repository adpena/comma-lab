Implemented and committed QS1 at `1c5557096d`.

Measured locally on all 14 retained Q3 proposals:

- 11/14 passed the conservative screen; 3 were held.
- Maximum exact integer/receiver-realized Pose leakage cancellation: `99.995054%`.
- Six independent pair edits compiled into one repeat-identical archive.
- Candidate: `186,329 B`, `+77 B` versus CP135, SHA-256 `e474d452…db9d`.
- Conservative compiled screen margin: `+0.00010815412615687239 S`.

Receiver parse-back exactly preserved the intended int12 lattice, token stream, residual, and archive member. No Modal job, scorer lane, or exact evaluation ran. Realized Seg, official T4 Pose, complete S, and promotion remain unmeasured.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_qs1_frame0_schur_coupled_solve_20260813.md)
- [Compile result](/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/COMPILE_RESULT.json)
- [Corrected sealed r2 fire order](/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/SEALED_FIRE_ORDER.json)
- [Candidate archive](/Volumes/VertigoDataTier/pact/ddm_qs1_20260813/compile_workspace/retained/candidates/qs1_combined_unique_pairs/primary/qs1_objects/archive.zip)

The original r1 transport receipt was folded after an import defect was caught locally; it was preserved byte-identically, and r2 successfully verified every sealed input. Verification passed: Ruff, bytecode compilation, 10 tests, payload-retention gate, and two review passes.

Effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — owner: `MAIN` sole scorer-lane router; consumer store: `/Volumes/VertigoDataTier/pact/ddm_qs1_20260813`; fire trigger: no active n600 exact-eval/Modal lane, lane `ddm_qs1_dual_axis_n600_20260813` is claimed, and every sealed r2 hash verifies. Execute the exact argv in `SEALED_FIRE_ORDER.json`, then recompute complete S locally and admit only realized `delta S < 0`.

## LIVE-HYPOTHESES

- The six-pair candidate merits its sealed T4 verdict because the exact `+77 B` debit and conservative Pose bound still leave `0.00010815412615687239 S`; realized Seg value is the remaining uncertainty.
- Additional CP135-coupled proposals may exist beyond the 14-pair diagnostic set, given 11 survivors and several cancellations above 99.8%, but expansion should wait for this candidate’s verdict.
- Same-pair joint proposal solves may outperform the strongest-single-proposal fold if T4 reveals a narrow interaction deficit.

## DEAD-ENDS

- LC2/PZ4R sensitivity or fit-number transfer into CP135 is closed by the lineage mismatch.
- Five overlapping positive proposals were folded; compiling all eleven would double-count shared pairs.
- Three unchanged negative-margin instances are held and should not be recompiled.
- The invalid r1 fire order is folded and superseded by r2.
- Re-running the 14-pair local census before the sealed T4 verdict is closed; its retained archive repeat is byte-identical.