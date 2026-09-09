**A is sealed at 180,388 B, saving 384 B.** Its 38 public-decoded tensors are identical to the base. [Seal]( /Volumes/VertigoDataTier/pact/ddm_cmp2_compose/SEAL_ddm_cmp2_sm1_fe1_composed.json)

**The charter remains partial:** B is retained at 180,372 B, but fresh seg/pose verification requires MAIN’s scorer lane. No scorer or Modal run occurred. [Memo](/Users/adpena/Projects/pact/.omx/research/ddm_cmp2_compose_sm1_fe1_20260909.md)

Git writes were sandbox-blocked. The verified [fallback bundle](/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/serializer_fallback/20260909T201127.179507Z-33962/intended-commit.bundle) and [receipt](/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/serializer_fallback/20260909T201127.179507Z-33962/receipts.jsonl) are retained; nothing landed on main.

Frontier unchanged: **S 0.13817298987557713 @ 180,772 B [contest-CUDA T4 n600]**.

## NEXT_IF_RESUMED

All actions are **QUEUED-WITH-A-FIRE-ORDER**, owner **MAIN**; consumer paths are under `/Volumes/VertigoDataTier/pact/ddm_cmp2_compose/`.

- Verify and re-solve B → `B_ADMISSION.json`; trigger: scorer lane available.
- Evaluate the admitted selection once on T4 → `MAIN_EXACT_RESULT.json`; trigger: B disposition resolved and current seal valid.
- Rebase the composition → `REBASE_HANDOFF.json`; trigger: pointer changes before firing.
- Land the fallback and reconcile recorded events → `LANDING_RECEIPT.json`; trigger: Git-writable context.

## LIVE-HYPOTHESES

- A should realize its rate saving on T4 because its decoded state is unchanged.
- B may preserve another 16 B after re-solving pose; its actual code edit is verified.

## DEAD-ENDS

- Carrying fe1’s −61 B: current encoding prices it differently.
- Replacing container pricing with symbol cost: the measured symbol saving produces zero whole range-byte savings, while Brotli still changes archive size.
- Admitting B from ancestor neutrality: current shipped-byte scorer proof remains missing.

