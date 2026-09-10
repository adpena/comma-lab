Frontier unchanged: **S 0.13763861019288715 @ 180,233 B [contest-CUDA T4 n600]**.

Computed and retained these **conditional population lower bounds**:

| Error budget D | Conservative | Natural profile |
|---:|---:|---:|
| 6,270 | 2,467.625 B | 5,006 B |
| 12,540 | 1,404.5 B | 2,100.25 B |
| 25,080 | 255.25 B | 255.25 B |

All 600-frame invariant checks, numerical checks, Python reviews, and Ruff passed. **These bounds do not establish the incumbent’s entropy or achievable savings.** The required natural-video population and conditional-tail assumptions remain unproved.

[Full memo](/Volumes/APDataStore/pact/ddm_eb1_entropy_bound/checkout/.omx/research/ddm_eb1_entropy_lower_bound_boundary_description_20260910.md) · [Verified patch](/Volumes/APDataStore/pact/ddm_eb1_entropy_bound/landing.patch) · [Handoff](/Volumes/APDataStore/pact/ddm_eb1_entropy_bound/FINAL_HANDOFF.json)

Isolated commit: `d99092785`. Equation registered in isolation; live tree untouched.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER — owner MAIN:** import the verified landing into the canonical equation/task ledgers and GS3 memo. Consumer paths are recorded in the [harvest order](/Volumes/APDataStore/pact/ddm_eb1_entropy_bound/checkout/.omx/research/ddm_eb1_20260910/HARVEST_ORDER.json); fire on harvest of this landing.

## LIVE-HYPOTHESES

- Stronger conditional populations may tighten the bound: this construction freezes most geometry.
- Better coders or generators remain plausible: no applicable optimum was established.

## DEAD-ENDS

- Population bounds as individual-video or tail floors: wrong quantifiers and missing side-information conditioning.
- Weak lower bounds as proof of available savings: the optimum may be much higher.
- Treating 12,540 as verified T4 error or token disagreement: its arithmetic differs from the printed score; move-40 tokens disagree at **17,631 cells**.