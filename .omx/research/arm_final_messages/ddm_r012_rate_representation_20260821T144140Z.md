Completed the charter and wrote the routing authority: [ddm_r012 memo](/Users/adpena/Projects/pact/.omx/research/ddm_r012_rate_representation_20260821.md).

Key conclusions:

- Exact unchanged-distortion requirement is **−42,470 B**, correcting the charter’s −42,463 B.
- Honest measured composable ceiling is **88 B**: fx5 −70 B plus dx1 −18 B.
- Even zeroing all distortion requires −238 B, so the measured stack remains 150 B short.
- fx5’s archive is retained and verified at 180,386 B, but its decode receipt and seal were absent; disposition is `BLOCKED(fx5 sibling decode/seal in progress)`.
- No scorer, governed launch, Modal call, or new payload was produced.

The primary `.git` was sandbox-read-only. Serializer commit `be01069bc8` was therefore created in a clean fallback clone and exported as a verified [9,932-byte bundle](/Volumes/APDataStore/pact/ddm_r012_rate_representation_commit_be01069bc8.bundle), SHA-256 `3b0af5e2c8df10b1f2d945c8986a189ed1e5d8f64dc4aa2dff6c7861c2b12b22`. The memo remains uncommitted in the primary checkout; its SHA-256 is `f98cd7df414908cc0bbe051a56ff700e966b78e4554022d0c41914480b810eca`.

## NEXT_IF_RESUMED

- `QUEUED-WITH-FIRE-ORDER` — owner: fx5 owner, then MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_fx5/t4_row_r1/`; fire trigger: decoded-token identity receipt and valid candidate seal exist, then MAIN claims and fires the T4 lane.
- `QUEUED-WITH-FIRE-ORDER` — owner: rate-representation successor; consumer store: `/Volumes/APDataStore/pact/ddm_r012_rate_representation/dx1_rc2/`; fire trigger: fx5 resolves, then dx1 yields byte-identical decode, at least 18 B net credit, and collision-free rr5/CAP1 signalling.
- `FOLDED` — owner: jo1/MAIN; consumer store: the joint frontier table in the memo; fire trigger: jo1 returns a measured distortion delta, which selects the residual rate requirement.

**Own-vehicle frontier: S 0.14827847122030852 @ 180,456 B `[contest-CUDA T4, n600]` — UNMOVED by ddm_r012.**

## LIVE-HYPOTHESES

- dx1 may retain its full 18 B because its coefficient object is byte-identical on rc2.
- CAP1 metadata packing may outperform dx1, but it requires current-body transfer and shares signalling.
- Warm-lineage width reduction may escape the fresh D56/F64 failure through inherited conditioning.
- A genuinely new representation remains necessary because the current measured stack cannot meet even the zero-distortion rate floor.

## DEAD-ENDS

- Fresh D56/F64 and unconditional W96 escalation: n120 score losses and no demonstrated capacity pressure.
- Current carrier rank/atom truncation: inadequate functional survival across six treatments.
- q3/q4 and deeper FiLM sparsity: score damage overwhelms byte credit.
- Current token-drop/carrier-repair formulation: measured pose loss dominates its real rate saving.
- Task #869’s −113,555 B projection: ancestor-body, task-lossy, and contradicted at the later operating point.
- Further memoryless section coding: measured remaining room is only a few bytes.