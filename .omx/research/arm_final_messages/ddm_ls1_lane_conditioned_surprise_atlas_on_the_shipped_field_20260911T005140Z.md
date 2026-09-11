**LS1 is complete. No constructed mechanism demonstrated the required 25,899 B saving.** Independent n600 verification and byte-identical restart passed. Checkpoint marked COMPLETE.

The shipped decoder’s surprise totals **119,748.300 B**, versus **119,749 B** of actual RC64 bytes—a **0.000584%** discrepancy. Measurements are **[macOS-CPU advisory, scorer-free n600]**; no training, Modal, scorer, or candidate archive was run.

Atlas headline, rounded **bits**. Each geometry cell shows **boundary / interior**; “correct” retains correctly predicted symbols.

| Class | Isolated | Short | Long | Correct |
|---|---:|---:|---:|---:|
| Road | 177,809 / 573 | 47,838 / 469 | 14 / 0 | 144,464 |
| Lane | 202,560 / 38 | 48,661 / 11 | 0 / 0 | 69,450 |
| Undrivable | 57,251 / 109 | 15,761 / 7 | 58 / 0 | 41,981 |
| Movable | 57,597 / 42 | 16,379 / 15 | 0 / 0 | 25,375 |
| MyCar | 37,580 / 25 | 4,634 / 0 | 0 / 0 | 9,286 |

Lane occupies **0.586% of symbols** and consumes **33.479% of surprise**.

The oracle ladder grants free probability parameters. Plug-in bounds apply only to the specified constant-cell families; Miller–Madow (MM) values are bias-corrected estimates, **not universal ceilings**.

| Oracle | Plug-in upper saving | MM saving | MM shortfall against 25,899 B |
|---|---:|---:|---:|
| TC1 joint contexts | 11,568.821 B | 8,218.071 B | 17,680.929 B |
| Receiver-visible Lane geometry | 22,222.524 B | 17,534.216 B | 8,364.784 B |
| Granted complete previous-row geometry | 29,809.287 B | 24,863.638 B | 1,035.362 B |

The strongest plug-in fit crosses demand, but its bias correction reverses that answer before counting model costs.

**Localization:** Lane gain occupies rows **128–319**, but is spread across the video: covering 90% of positive Lane gain requires **503/600 pairs** and **82/384 rows** under the strongest grant. Its total signed Lane saving is only **11,303.496 B**. No physical dash-phase mechanism was established.

| Mechanism | Verdict |
|---|---|
| Bigger mixer | Specified constant-cell replacement falls short even with free parameters; full-resolution mixers remain unmeasured. |
| Receiver-visible context model | Specified replacement falls short by **3,676.476 B**, even at its plug-in upper bound. |
| Counted Lane carrier | Actual override map costs **594,003 B**; this map instance cannot pay. Compact carriers remain unpriced. |
| Born-vehicle generator | Gain and cost on this shipped field remain unknown. |

Source corrections matter: TC2’s approximately 5.5 KB was **oracle gain**, not map cost; GDC4 reported computed lengths, not serialized prices; TC1 did not establish a universal mixer ceiling.

[Full atlas and evidence](/Users/adpena/Projects/pact/.omx/research/ddm_ls1_lane_conditioned_surprise_atlas_on_the_shipped_field_20260911.md) · [Cumulative curves](/Volumes/VertigoDataTier/pact/ddm_ls1/atlas_curves.png) · [Next charter: LS2](/Users/adpena/Projects/pact/.omx/research/ddm_ls2_full_resolution_lane_probability_bound_charter_20260911.md)

Retained measurement custody: **6,278 files, 5.36 GB**. Both serializers returned **rc17** after sandbox Git-write denial; both fallback bundles and all eight owned-file hashes verified. **Shared HEAD is not landed.** [Landing receipt and bundle locations](/Users/adpena/Projects/pact/.omx/research/ddm_ls1_20260911/LANDING_RECEIPT.json)

Frontier unchanged: **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600] (move 44)**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — MAIN:** land verified LS1 bundles and harvest corrections into `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md`. **Fire trigger:** verified bundles available—satisfied.
- **QUEUED-WITH-A-FIRE-ORDER — MAIN / execution owner ddm_ls2:** execute the attached full-resolution probability-bound charter; **consumer store:** `/Volumes/VertigoDataTier/pact/ddm_ls2_full_resolution_lane_probability_bound/RESULT.json`. **Fire trigger:** MAIN harvests LS1 and revalidates archive/field identity.

**LIVE-HYPOTHESES**

- Compact full-resolution probability corrections: receiver-visible geometry adds information that coarse cells may underuse.
- Compact positioned Lane carriers: unavailable geometry adds oracle gain, although no affordable representation is demonstrated.

**DEAD-ENDS**

- These fixed-cell replacements as complete demand solutions: their free-table bounds fall short.
- The explicit override-map instance: its 594,003 B cost overwhelms savings.
- Transferring unshipped-field results or treating oracle estimates as universal ceilings: provenance and mathematical scope do not support those claims.
<!-- # FORMALIZATION_PENDING: measurement memo; the atlas lands in the equations leg when it becomes a law -->
