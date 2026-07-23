# Codex session summary — 2026-07-23T10:12:37Z

Lane: `ddm_v18_column_generation_vocabulary`

## Landed

- Common exact-R post-solve master and resumable Probe B executor:
  `b4344414bd`, `274eb2a210`, `e7213d5644`, `c1bf7e5d2d`,
  `9e2f20be3f`.
- Post-campaign automatic receiver no-op recovery and permutation-gauge coder
  disposition: `817feaf972`.
- Immutable definitive run:
  `.omx/research/ddm_v18b_common_master_pricing_20260723T050800Z`.
- Final receipt SHA-256:
  `0d7e3535905cd48d42d7caeb6cfa8f56486a781bf16bbcb58cbe34afab014f55`.

## Result

`FALSIFIED_FORMULATION_THREE_CLEAN_PRICING_ROUNDS`.

The current exact-R v12 control is 0.034075563219 d_seg at all four caps,
43 bundles / 4,091 added bytes / 107,720 total bytes, which is
+0.000071894328 versus the legacy 0.034003668891 scorer-grid line. All three
pricing rounds generated 64 distinct columns, found zero negative reduced
costs, and admitted none. The generated base objective 43.918736599760 loses
to the rebased v12 objective 43.856535229213 at every byte cap. The
permutation-gauge race is 103,629 canonical bytes versus 103,629 as-is bytes
because the selected generated payload is empty.

Axis is `[macOS-CPU frozen-scorer advisory]`; `score_claim=false`;
pointer `0.1910828242 [contest-CPU]` is unchanged.

## Review boundary

MAIN must review and merge this branch. In particular, review the
post-solve/PREDICT separation, all SHA-bound receipt references, the exact
receiver no-op token catcher, same-receiver under-cap comparison semantics,
and the formulation-scoped falsifier. No paid dispatch or main-tree mutation
occurred.
