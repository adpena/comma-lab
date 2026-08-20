RE1T is built, sealed, tested, and committed. MAIN has not fired Modal.

- Receipt: [ddm_re1t_t4_gate_prep_20260813.md](/Users/adpena/Projects/pact/.omx/research/ddm_re1t_t4_gate_prep_20260813.md)
- Request SHA: `6d343139…efb79`
- Fire-order SHA: `b0ae14d1…900f2`
- Implementation commit: `27c828d1d1`
- Receipt commit: `46746a2349`
- Validation: 17 focused/RE1X tests passed; Ruff, compilation, payload-retention check, sealed-loader replay, and two review passes passed.
- Measured: scorer-free custody, hashes, runtime pin, 28-record blocker rehash, deterministic bundle, and storage headroom.
- Not measured: CUDA decode, candidate flips, Pose, exact score, or spend.
- Effective frontier remains CP135 `S=0.16195513827824176 @ 186,252 B [contest-CUDA T4,n600]`.
- Own-vehicle frontier remains LC2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4,n600]`.

## NEXT_IF_RESUMED

- **Disposition:** `QUEUED-WITH-A-FIRE-ORDER`. **Owner:** MAIN sole scorer-lane router. **Consumer store:** `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/probability_object_race/ddm_re1_20260813/full_n600_exact/round_01_singleton_best`. **Fire trigger:** no competing full-n600 scorer or Modal single-flight; reverify request `6d343139…efb79`, archive `7be3eb94…7dfa`, and runtime tree `63b93187…dc75`, then execute the exact command in `RE1T_T4_FIRE_ORDER.json`.

## LIVE-HYPOTHESES

- RE1 is receiver-null and produces CP135’s exact T4 field. This is plausible because all decoded downstream components previously closed identically.
- The one categorical cell survives rendering and removes at least one T4 error at equal bytes. This remains plausible because it is a full lattice change, though prior favorable evidence came from another surface.

## DEAD-ENDS

- Local CPU/Metal decode is closed: the exact F26 runtime requires CUDA.
- Runtime patching, CP135-raw substitution, and token equality as a score proxy test the wrong object.
- Reusing the failed RE1X run ID breaks byte-identical resume custody.
- The zero Pose placeholder is not evidence; Seg-only admission cannot become a composition or score claim.
- If the T4 field is identical to CP135, this instance is dead and no Pose job may fire.

